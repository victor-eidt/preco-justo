"""Group-aware splits: rows from the same purchase never straddle a split.

Items bought in the same tender (or the same Compras.gov ``idCompra``) have
correlated prices, so a row-level split would leak. Both functions walk a
seeded permutation of ROWS and move whole groups, which has a useful
property: when every group has exactly one row, they reproduce
``numpy.random.default_rng(seed).permutation`` and scikit-learn's shuffled
``KFold`` exactly. The pilot numbers therefore stay reproducible.
"""

import numpy as np


def _group_rows(groups):
    groups = np.asarray(groups)
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    return members


def holdout_split(groups, frac=0.30, seed=12345):
    """Return ``(dev_idx, holdout_idx)``, both sorted, with whole groups."""
    groups = np.asarray(groups)
    n = len(groups)
    members = _group_rows(groups)
    perm = np.random.default_rng(seed).permutation(n)
    target = int(n * frac)
    hold, taken = [], set()
    for i in perm:
        if len(hold) >= target:
            break
        g = groups[i]
        if g in taken:
            continue
        taken.add(g)
        hold.extend(members[g])
    hold = np.array(sorted(hold), dtype=int)
    dev = np.setdiff1d(np.arange(n), hold)
    return dev, hold


def group_folds(groups, n_folds=5, seed=0):
    """Yield ``(train_idx, test_idx)`` pairs, whole groups, balanced sizes."""
    groups = np.asarray(groups)
    n = len(groups)
    members = _group_rows(groups)
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)  # what sklearn's KFold does

    sizes = np.full(n_folds, n // n_folds)
    sizes[: n % n_folds] += 1
    fold_of = np.full(n, -1)
    fold, filled, taken = 0, 0, set()
    for i in order:
        g = groups[i]
        if g in taken:
            continue
        taken.add(g)
        for j in members[g]:
            fold_of[j] = fold
        filled += len(members[g])
        if filled >= sizes[fold] and fold < n_folds - 1:
            fold, filled = fold + 1, 0
    for f in range(n_folds):
        test = np.where(fold_of == f)[0]
        train = np.where(fold_of != f)[0]
        yield train, test
