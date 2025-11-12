# quantbayes/stochax/robust_inference/eval.py
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.robust_inference.attacks import (
    runnerup_attack,
    cwtm_aware_attack,
    sia_blackbox,
    sia_whitebox,
    loss_max_attack,
)
from quantbayes.stochax.robust_inference.masks import choose_m_probs


# ---------------------------- Clean accuracy ---------------------------- #


def aggregator_clean_acc(agg, Ps: jnp.ndarray, y: jnp.ndarray) -> float:
    """Mean top-1 accuracy of the aggregator on probits Ps."""
    logits = jax.vmap(lambda P: agg(P, None, None)[0])(Ps)
    return float(jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32)))


# ---------------------------- Fast batched PGD-cw (single source of truth) ---------------------------- #


def _build_T_and_masks(
    ranks: jnp.ndarray, ms: jnp.ndarray, f: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    ranks: (B,n) inverse permutations (rank of each row)
    ms:    (B,)   number of corrupted rows per example (in {1..f})
    returns:
      T           (B,n,f): maps f-slot vectors to the last-f positions of perm
      mask_use_f  (B,f):   which of the f slots are actually used (keep last m)
      mask_use_n  (B,n):   which rows are actually corrupted
    """
    B, n = ranks.shape
    ridx = ranks - (n - f)  # (B,n)
    ridx_clipped = jnp.clip(ridx, 0, f - 1)  # (B,n)
    onehot = jax.vmap(lambda rc: jax.nn.one_hot(rc, f, dtype=jnp.float32))(
        ridx_clipped
    )  # (B,n,f)
    in_lastf = (ranks >= (n - f)).astype(jnp.float32)[..., None]  # (B,n,1)
    T = onehot * in_lastf  # (B,n,f)

    ar_f = jnp.arange(f, dtype=jnp.int32)[None, :]
    mask_use_f = (ar_f >= (f - ms[:, None])).astype(jnp.float32)  # (B,f)
    mask_use_n = (ranks >= (n - ms[:, None])).astype(jnp.float32)  # (B,n)
    return T, mask_use_f, mask_use_n


@eqx.filter_jit
def _pgd_cw_one_try_batched(
    agg,
    P_b: jnp.ndarray,
    y_b: jnp.ndarray,
    f: int,
    steps: int,
    step_size: float,
    mask_mode: str,
    key: jr.PRNGKey,
) -> jnp.ndarray:
    """
    One adversarial try for a whole batch (JIT’d).
    Returns per-sample correctness (1.0 if correct after attack, else 0.0), shape (B,).
    """
    B, n, K = P_b.shape
    p_m = choose_m_probs(n, f)

    k_m, k_perm, k_v = jr.split(key, 3)
    if mask_mode == "paper":
        ms = jr.categorical(k_m, jnp.log(p_m), shape=(B,)) + jnp.array(
            1, dtype=jnp.int32
        )  # (B,)
    else:
        ms = jnp.ones((B,), dtype=jnp.int32) * f

    perms = jax.vmap(lambda kk: jr.permutation(kk, n))(jr.split(k_perm, B))  # (B,n)
    ranks = jax.vmap(jnp.argsort)(perms)  # (B,n)
    T, mask_use_f, mask_use_n = _build_T_and_masks(
        ranks, ms, f
    )  # (B,n,f), (B,f), (B,n)

    V = jr.normal(k_v, shape=(B, f, K))  # (B,f,K)

    def loss_on_P(P_batch):
        logits = jax.vmap(lambda Pi: agg(Pi, None, None)[0])(P_batch)
        Bp, Kp = logits.shape
        zy = logits[jnp.arange(Bp), y_b]
        mask = jnp.eye(Kp, dtype=bool)[y_b]
        z_other = jnp.where(mask, -jnp.inf, logits).max(axis=1)
        return jnp.maximum(z_other - zy, 0.0).mean()

    def step(_, V_cur):
        Vhat = jax.nn.softmax(V_cur, axis=-1)  # (B,f,K)
        Vhat_sel = Vhat * mask_use_f[..., None]  # (B,f,K)
        rows_from_V = jnp.einsum("bnf,bfk->bnk", T, Vhat_sel)  # (B,n,K)
        P_tmp = P_b * (1.0 - mask_use_n[..., None]) + rows_from_V  # (B,n,K)

        gP = jax.grad(loss_on_P)(P_tmp)  # (B,n,K)
        gV = jnp.einsum("bnf,bnk->bfk", T, gP) * mask_use_f[..., None]  # (B,f,K)
        return V_cur + step_size * jnp.sign(gV)

    V_fin = jax.lax.fori_loop(0, steps, step, V)
    Vhat = jax.nn.softmax(V_fin, axis=-1)
    Vhat_sel = Vhat * mask_use_f[..., None]
    rows_from_V = jnp.einsum("bnf,bfk->bnk", T, Vhat_sel)
    P_adv = P_b * (1.0 - mask_use_n[..., None]) + rows_from_V

    logits_adv = jax.vmap(lambda Pi: agg(Pi, None, None)[0])(P_adv)
    yhat = jnp.argmax(logits_adv, axis=1)
    return (yhat == y_b).astype(jnp.float32)  # (B,)


def aggregator_pgd_cw_acc(
    agg,
    Ps: jnp.ndarray,
    y: jnp.ndarray,
    *,
    f: int,
    steps: int = 50,
    step_size: float = 5e-2,
    tries: int = 1,
    mask_mode: str = "paper",
    batch_size: int = 256,
    seed: int = 0,
) -> float:
    """
    Batched, JIT’d PGD-cw accuracy with **worst-of-tries** semantics.
    This is the single source of truth for adversarial evaluation.
    """
    N = Ps.shape[0]
    key = jr.PRNGKey(seed)
    acc_sum = 0.0

    num_batches = (N + batch_size - 1) // batch_size
    for b in range(num_batches):
        start = b * batch_size
        end = min(N, start + batch_size)
        key, k = jr.split(key)
        P_b = Ps[start:end]
        y_b = y[start:end].astype(jnp.int32)

        correct = jnp.ones((P_b.shape[0],), dtype=jnp.float32)

        def body(carry, kt):
            cur = _pgd_cw_one_try_batched(
                agg, P_b, y_b, f, steps, step_size, mask_mode, kt
            )
            return jnp.minimum(carry, cur), None

        keys = jr.split(k, tries)
        correct, _ = jax.lax.scan(body, correct, keys)
        acc_sum += float(correct.mean()) * (end - start)

        if b % max(1, num_batches // 10) == 0 or b + 1 == num_batches:
            done = end
            print(f"[PGD-cw] {done}/{N} ({100.0*done/N:.1f}%)")

    return acc_sum / N


# ---------------------------- Mixed attack bench (uses fast PGD-cw) ---------------------------- #


def quick_attack_bench(
    agg,
    Ps: jnp.ndarray,
    y: jnp.ndarray,
    f: int,
    *,
    seed: int = 0,
    pgd_steps: int = 50,
    pgd_step_size: float = 5e-2,
    pgd_tries: int = 1,
    pgd_batch_size: int = 256,
    attacks: dict | None = None,  # NEW: toggles
) -> Dict[str, float]:
    """
    Run a selectable subset of attacks. Example:
      attacks = {
        'pgd_cw': True,    # fast, robust, single source of truth
        'runnerup': False,
        'cwtm_aware': False,
        'sia_bb': False,
        'sia_wb': False,
        'lma': False,
      }
    If `attacks` is None → defaults to {'pgd_cw': True}.
    """
    if attacks is None:
        attacks = {"pgd_cw": True}

    key = jr.PRNGKey(seed)
    N = Ps.shape[0]
    res: Dict[str, float] = {}

    # cheap attacks (per-sample loops)
    cheap = ["runnerup", "cwtm_aware", "sia_bb", "sia_wb", "lma"]
    if any(attacks.get(k, False) for k in cheap):
        ok = {k: 0 for k in cheap}
        for i in range(N):
            P = Ps[i]
            yi = int(y[i])
            key, k1, k2 = jr.split(key, 3)

            if attacks.get("runnerup", False):
                Z = runnerup_attack(P, yi, f, k1)
                ok["runnerup"] += int(jnp.argmax(agg(Z, None, None)[0]) == yi)

            if attacks.get("cwtm_aware", False):
                Z = cwtm_aware_attack(P, f)
                ok["cwtm_aware"] += int(jnp.argmax(agg(Z, None, None)[0]) == yi)

            if attacks.get("sia_bb", False):
                Z = sia_blackbox(P, yi, f, k2)
                ok["sia_bb"] += int(jnp.argmax(agg(Z, None, None)[0]) == yi)

            if attacks.get("sia_wb", False):
                Z = sia_whitebox(P, yi, f, agg, k2)
                ok["sia_wb"] += int(jnp.argmax(agg(Z, None, None)[0]) == yi)

            if attacks.get("lma", False):
                Z = loss_max_attack(P, f, agg)
                ok["lma"] += int(jnp.argmax(agg(Z, None, None)[0]) == yi)

        for k in cheap:
            if attacks.get(k, False):
                res[k] = ok[k] / N

    # robust PGD-cw (fast, batched)
    if attacks.get("pgd_cw", True):
        res["pgd_cw"] = aggregator_pgd_cw_acc(
            agg,
            Ps,
            y,
            f=f,
            steps=pgd_steps,
            step_size=pgd_step_size,
            tries=pgd_tries,
            batch_size=pgd_batch_size,
            seed=seed,
        )

    return res


# ---------------------------- Curves: PGD-cw vs f (fast) ---------------------------- #


def pgd_cw_vs_f(
    agg,
    Ps: jnp.ndarray,
    y: jnp.ndarray,
    f_values: Iterable[int],
    *,
    steps: int = 50,
    step_size: float = 5e-2,
    tries: int = 1,
    batch_size: int = 256,
    seed: int = 0,
) -> Tuple[List[int], List[float]]:
    """Return (f_list, accuracies) for PGD-cw at different f."""
    f_list, accs = [], []
    for f in f_values:
        acc = aggregator_pgd_cw_acc(
            agg,
            Ps,
            y,
            f=f,
            steps=steps,
            step_size=step_size,
            tries=tries,
            batch_size=batch_size,
            seed=seed,
        )
        f_list.append(int(f))
        accs.append(float(acc))
    return f_list, accs
