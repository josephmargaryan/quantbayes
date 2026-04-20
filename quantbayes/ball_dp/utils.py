import numpy as np


def choose_target_index(y_train, y_pool, *, preferred=17, min_pool_per_label=8):
    preferred = int(preferred)
    if 0 <= preferred < len(y_train):
        if np.sum(y_pool == y_train[preferred]) >= int(min_pool_per_label):
            return preferred
    for idx in range(len(y_train)):
        if np.sum(y_pool == y_train[idx]) >= int(min_pool_per_label):
            return idx
    raise ValueError(
        "Could not find a training target whose label has enough held-out same-label candidates."
    )


def choose_target_index(y_train, y_pool, *, preferred=17, min_pool_per_label=8):
    preferred = int(preferred)
    if 0 <= preferred < len(y_train):
        if np.sum(y_pool == y_train[preferred]) >= int(min_pool_per_label):
            return preferred
    for idx in range(len(y_train)):
        if np.sum(y_pool == y_train[idx]) >= int(min_pool_per_label):
            return idx
    raise ValueError(
        "Could not find a training target whose label has enough held-out same-label candidates."
    )


def build_same_label_finite_support(
    x_target,
    target_label,
    X_pool,
    y_pool,
    *,
    max_candidates=16,
):
    """Build a same-label finite support from held-out data and append the true target.

    This is the primary exact-identification prior used in the notebook.
    It guarantees that the true target is in the candidate support.
    """
    x_target = np.asarray(x_target, dtype=np.float32).reshape(-1)
    mask = np.asarray(y_pool) == int(target_label)
    pool = np.asarray(X_pool[mask], dtype=np.float32).reshape(np.sum(mask), -1)

    if len(pool) == 0:
        raise ValueError("No same-label held-out candidates available.")

    dists = np.linalg.norm(pool - x_target[None, :], axis=1)
    order = np.argsort(dists)

    chosen = []
    for j in order:
        cand = np.asarray(pool[j], dtype=np.float32)
        if np.allclose(cand, x_target, atol=1e-8, rtol=0.0):
            continue
        chosen.append(cand)
        if len(chosen) >= int(max_candidates) - 1:
            break

    chosen.append(x_target)

    uniq = []
    for z in chosen:
        if not any(np.allclose(z, u, atol=1e-8, rtol=0.0) for u in uniq):
            uniq.append(np.asarray(z, dtype=np.float32))

    X_candidates = np.stack(uniq, axis=0)
    y_candidates = np.full((len(X_candidates),), int(target_label), dtype=np.int32)
    return X_candidates, y_candidates


def build_same_label_support_from_center(
    center,
    target_label,
    X_pool,
    y_pool,
    *,
    radius,
    max_candidates=32,
):
    """Side-information-driven helper for future realistic runs.

    This does NOT guarantee the true target is in support. Use it when the center
    comes from attacker side information. Then either:
      - only evaluate targets satisfying ||z - center|| <= radius, or
      - treat prior misspecification as a separate experiment.
    """
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    mask = np.asarray(y_pool) == int(target_label)
    pool = np.asarray(X_pool[mask], dtype=np.float32).reshape(np.sum(mask), -1)
    if len(pool) == 0:
        raise ValueError("No same-label pool points available.")
    dists = np.linalg.norm(pool - center[None, :], axis=1)
    keep = np.where(dists <= float(radius))[0]
    if len(keep) == 0:
        raise ValueError("No same-label candidates lie inside the requested Ball.")
    order = keep[np.argsort(dists[keep])]
    order = order[: int(max_candidates)]
    X_candidates = np.asarray(pool[order], dtype=np.float32)
    y_candidates = np.full((len(X_candidates),), int(target_label), dtype=np.int32)
    return X_candidates, y_candidates


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from quantbayes.ball_dp import (
        ArrayDataset,
        make_finite_identification_prior,
        make_uniform_ball_prior,
    )
    from quantbayes.ball_dp.api import make_uniform_ball_attack_prior
    from quantbayes.fake_data import generate_binary_classification_data

    preferred_target_index = 0
    radius = 0.1
    df = generate_binary_classification_data()
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_ds = ArrayDataset(X_train, y_train, name="train")
    test_ds = ArrayDataset(X_test, y_test, name="test")

    num_classes = int(len(np.unique(y_train)))
    feature_dim = int(X_train.shape[1])

    train_norms = np.linalg.norm(X_train, axis=1)
    test_norms = np.linalg.norm(X_test, axis=1)
    B_public = float(np.max(train_norms))
    target_index = choose_target_index(
        y_train,
        y_test,
        preferred=preferred_target_index,
        min_pool_per_label=8,
    )
    target_label = int(y_train[target_index])
    x_target = np.asarray(X_train[target_index], dtype=np.float32).reshape(-1)
    true_record = train_ds.record(target_index)

    X_candidates, y_candidates = build_same_label_finite_support(
        x_target,
        target_label,
        X_test,
        y_test,
        max_candidates=16,
    )
    finite_prior = make_finite_identification_prior(X_candidates, weights=None)
    m_candidates = int(len(X_candidates))

    u_classmean = np.asarray(
        X_train[y_train == target_label].mean(axis=0),
        dtype=np.float32,
    ).reshape(-1)
    oracle_continuous_prior = make_uniform_ball_prior(center=x_target, radius=radius)
    oracle_attack_prior = make_uniform_ball_attack_prior(center=x_target, radius=radius)

    attack_eta_grid = tuple(float(radius * q) for q in (0.25, 0.50, 0.75, 1.00))
    finite_eta_grid = (0.5,)  # any eta < 1 is equivalent in the exact-ID theorem
    continuous_ratio_grid = (0.90, 0.95, 0.97, 0.98, 0.99, 0.995)
    continuous_eta_grid = tuple(float(radius * q) for q in continuous_ratio_grid)

    print("target_index:", target_index)
    print("target_label:", target_label)
    print("finite prior size m:", m_candidates)
    print("exact-ID oblivious baseline 1/m:", 1.0 / m_candidates)
    print(
        "distance(target, class mean center):",
        float(np.linalg.norm(x_target - u_classmean)),
    )
    print(
        "target inside class-mean Ball?",
        bool(np.linalg.norm(x_target - u_classmean) <= radius),
    )
    print(
        "target inside oracle-centered Ball?",
        bool(np.linalg.norm(x_target - x_target) <= radius),
    )
