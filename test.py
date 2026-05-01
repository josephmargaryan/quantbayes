import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, ConnectionPatch
from pathlib import Path


def make_finite_support_audit_schematic(
    out_path: str | Path = "fig_finite_support_audit_schematic.pdf",
    *,
    seed: int = 7,
    m: int = 8,
):
    """
    Schematic figure for the finite-support reconstruction audit.

    Left panel:
        Embedding/data space.
        Anchor u, policy ball B(u,r), feasible bank C(u,r), selected support S.

    Right panel:
        Release/model space.
        Candidate means f(D^- union {z_i}), Gaussian noise contours,
        observed release theta_tilde, and MAP prediction.

    This is a conceptual diagram. It does not project real embeddings.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)

    if m < 1:
        raise ValueError("m must be at least 1.")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    # -------------------------------------------------------------------------
    # Left panel: embedding space
    # -------------------------------------------------------------------------
    u = np.array([0.0, 0.0])
    r = 1.0

    n_bank = 55
    if m > n_bank:
        raise ValueError(f"m={m} is larger than the feasible bank size {n_bank}.")

    angles = rng.uniform(0, 2 * np.pi, size=n_bank)
    radii = r * np.sqrt(rng.uniform(0.02, 0.95, size=n_bank))
    bank = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    n_out = 32
    out_angles = rng.uniform(0, 2 * np.pi, size=n_out)
    out_radii = rng.uniform(1.15, 1.75, size=n_out)
    outside = np.column_stack(
        [out_radii * np.cos(out_angles), out_radii * np.sin(out_angles)]
    )

    # Greedy farthest-point support selection.
    d_to_anchor = np.linalg.norm(bank - u[None, :], axis=1)
    tail = np.argsort(-d_to_anchor)[: min(len(bank), 4 * m)]
    first = int(rng.choice(tail))

    selected = [first]
    min_d = np.linalg.norm(bank - bank[first][None, :], axis=1)
    min_d[first] = -np.inf

    while len(selected) < m:
        nxt = int(np.argmax(min_d))
        selected.append(nxt)

        d_new = np.linalg.norm(bank - bank[nxt][None, :], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[selected] = -np.inf

    support = bank[selected]
    true_idx = 2 if m > 2 else 0
    target = support[true_idx]

    ball = Circle(u, r, fill=False, linewidth=2.0, linestyle="--")
    ax_l.add_patch(ball)

    ax_l.scatter(
        outside[:, 0],
        outside[:, 1],
        s=22,
        alpha=0.22,
        label="outside policy ball",
    )

    ax_l.scatter(
        bank[:, 0],
        bank[:, 1],
        s=24,
        alpha=0.40,
        label=r"feasible bank $\mathcal{C}(u,r)$",
    )

    ax_l.scatter(
        support[:, 0],
        support[:, 1],
        s=72,
        marker="o",
        edgecolor="black",
        linewidth=0.8,
        label=r"finite support $S$",
    )

    ax_l.scatter(
        [u[0]],
        [u[1]],
        s=180,
        marker="*",
        edgecolor="black",
        linewidth=0.8,
        label=r"anchor $u$",
    )

    ax_l.scatter(
        [target[0]],
        [target[1]],
        s=115,
        marker="X",
        edgecolor="black",
        linewidth=0.9,
        label=r"hidden target $Z \in S$",
    )

    ax_l.annotate(
        r"$\mathcal{B}(u,r)$",
        xy=(0.65, 0.72),
        xytext=(0.95, 1.18),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=12,
    )

    ax_l.annotate(
        r"$S=\{z_1,\ldots,z_m\}$",
        xy=support[0],
        xytext=(-1.62, 1.34),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=12,
    )

    ax_l.set_title("1. Build bank and choose finite support")
    ax_l.set_xlabel("embedding coordinate 1")
    ax_l.set_ylabel("embedding coordinate 2")
    ax_l.set_aspect("equal", adjustable="box")
    ax_l.set_xlim(-1.85, 1.85)
    ax_l.set_ylim(-1.55, 1.55)
    ax_l.legend(loc="lower left", fontsize=8, frameon=True)

    # -------------------------------------------------------------------------
    # Right panel: release/model space
    # -------------------------------------------------------------------------
    A = np.array([[1.15, 0.35], [-0.25, 0.95]])

    means = support @ A.T + np.column_stack(
        [0.12 * support[:, 0] ** 2, -0.10 * support[:, 1] ** 2]
    )

    mu_target = means[true_idx]
    sigma_vis = 0.22
    theta_tilde = mu_target + rng.normal(scale=sigma_vis, size=2)

    sq_dists = np.sum((means - theta_tilde[None, :]) ** 2, axis=1)
    pred_idx = int(np.argmin(sq_dists))

    for mu in means:
        ell = Ellipse(
            mu,
            width=2.2 * sigma_vis,
            height=2.2 * sigma_vis,
            angle=0,
            fill=False,
            linewidth=1.1,
            alpha=0.40,
        )
        ax_r.add_patch(ell)

    ax_r.scatter(
        means[:, 0],
        means[:, 1],
        s=70,
        marker="o",
        edgecolor="black",
        linewidth=0.8,
        label=r"candidate means $f(D^- \cup \{z_i\})$",
    )

    ax_r.scatter(
        [mu_target[0]],
        [mu_target[1]],
        s=120,
        marker="X",
        edgecolor="black",
        linewidth=0.9,
        label="true mean",
    )

    ax_r.scatter(
        [theta_tilde[0]],
        [theta_tilde[1]],
        s=150,
        marker="P",
        edgecolor="black",
        linewidth=0.9,
        label=r"observed release $\widetilde{\theta}$",
    )

    ax_r.scatter(
        [means[pred_idx, 0]],
        [means[pred_idx, 1]],
        s=165,
        facecolors="none",
        edgecolors="black",
        linewidth=2.0,
        label=r"MAP prediction $\widehat{Z}$",
    )

    for i, mu in enumerate(means):
        lw = 2.0 if i == pred_idx else 0.8
        alpha = 0.85 if i == pred_idx else 0.25

        ax_r.plot(
            [theta_tilde[0], mu[0]],
            [theta_tilde[1], mu[1]],
            linewidth=lw,
            alpha=alpha,
        )

    ax_r.text(
        0.02,
        0.98,
        (
            r"$\widehat{z}"
            r"=\arg\max_i\left\{"
            r"\log \pi_i"
            r"-"
            r"\frac{"
            r"\left\Vert \widetilde{\theta}-f(D^- \cup \{z_i\}) \right\Vert^2"
            r"}{2\sigma^2}"
            r"\right\}$"
        ),
        transform=ax_r.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
    )

    ax_r.set_title("2. Bayesian decoding in release space")
    ax_r.set_xlabel("model coordinate 1")
    ax_r.set_ylabel("model coordinate 2")
    ax_r.set_aspect("equal", adjustable="box")
    ax_r.legend(loc="lower left", fontsize=8, frameon=True)

    # Connection arrow between panels.
    con = ConnectionPatch(
        xyA=(1.78, 0.0),
        coordsA=ax_l.transData,
        xyB=(ax_r.get_xlim()[0], np.mean(ax_r.get_ylim())),
        coordsB=ax_r.transData,
        arrowstyle="->",
        linewidth=1.8,
        shrinkA=8,
        shrinkB=8,
    )
    fig.add_artist(con)

    fig.suptitle(
        "Finite-support reconstruction audit: bank construction, support selection, and MAP decoding",
        fontsize=13,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=240)

    return fig, (ax_l, ax_r)


if __name__ == "__main__":
    make_finite_support_audit_schematic(
        "fig_finite_support_audit_schematic.pdf",
        seed=7,
        m=8,
    )
    plt.show()
