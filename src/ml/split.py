# -*- coding: utf-8 -*-
"""Deterministic by-shot splits for cross-shot evaluation (ported verbatim from
WestBdryRec so splits match prior DCS work)."""
import numpy as np


def split_shots_3(shots, val_frac=0.1, test_frac=0.1, seed=0):
    """Deterministic 3-way split BY SHOT (no timeslice leakage).

    Seeded shuffle; val = first round(val_frac*N), test = next round(test_frac*N),
    train = the rest. Returns three sorted int lists.
    """
    shots = sorted(int(s) for s in shots)
    n = len(shots)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    n_test = max(1, int(round(test_frac * n))) if test_frac > 0 else 0
    val = sorted(shots[i] for i in perm[:n_val])
    test = sorted(shots[i] for i in perm[n_val:n_val + n_test])
    train = sorted(shots[i] for i in perm[n_val + n_test:])
    return train, val, test