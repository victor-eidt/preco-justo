import numpy as np
from sklearn.model_selection import KFold

from precojusto.splits import group_folds, holdout_split


def test_holdout_keeps_groups_together():
    groups = np.repeat(np.arange(40), 3)  # 40 purchases, 3 items each
    dev, hold = holdout_split(groups, 0.30, seed=1)
    assert set(groups[dev]).isdisjoint(set(groups[hold]))
    assert len(dev) + len(hold) == len(groups)
    assert abs(len(hold) - 36) <= 3


def test_folds_keep_groups_together():
    groups = np.repeat(np.arange(30), 2)
    for tr, te in group_folds(groups, 5, seed=0):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        assert len(tr) + len(te) == len(groups)


def test_unique_groups_reproduce_row_level_split():
    """With one row per group the pilot's original row-level split must hold."""
    n = 101
    groups = np.arange(n)
    _, hold = holdout_split(groups, 0.30, seed=12345)
    perm = np.random.default_rng(12345).permutation(n)
    assert set(hold) == set(perm[: int(n * 0.30)])

    ours = [set(te) for _, te in group_folds(groups, 5, seed=0)]
    theirs = [set(te) for _, te in KFold(5, shuffle=True, random_state=0).split(groups)]
    assert ours == theirs
