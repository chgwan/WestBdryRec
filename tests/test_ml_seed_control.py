# -*- coding: utf-8 -*-
"""The seeding contract for the multi-seed variance study.

The whole experiment rests on one asymmetry: the training seed must vary weight
init / shuffle / dropout, and must NEVER vary the train/val/test split. A varying
split would change the 76 test shots and silently void comparability with every
published number -- the failure would look like "interesting variance", not a bug.

Spec: docs/superpowers/specs/2026-08-07-capacity-matched-multiseed-design.md 4, 7
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.models import ActSeqAttn, ActSeqGRU  # noqa: E402
from src.ml.split import split_shots_3  # noqa: E402


def _init(cls, seed, **kw):
    """Initial weights of a freshly seeded model, as a flat vector."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    m = cls(**kw)
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


ATTN = dict(n_act=22, n_out=34, d=64, heads=4, depth=2, ffn=256, pe="rope_idx")
GRU = dict(n_act=22, n_out=34, hidden=64, layers=2)


# ------------------------------------------------------------------- T1 ---
def test_same_seed_gives_identical_init():
    assert torch.equal(_init(ActSeqAttn, 0, **ATTN), _init(ActSeqAttn, 0, **ATTN))
    assert torch.equal(_init(ActSeqGRU, 0, **GRU), _init(ActSeqGRU, 0, **GRU))


def test_different_seeds_give_different_init():
    assert not torch.equal(_init(ActSeqAttn, 0, **ATTN), _init(ActSeqAttn, 1, **ATTN))
    assert not torch.equal(_init(ActSeqGRU, 0, **GRU), _init(ActSeqGRU, 1, **GRU))


def test_three_study_seeds_are_all_distinct():
    """Seeds {0,1,2} must give three genuinely different starting points."""
    vs = [_init(ActSeqAttn, s, **ATTN) for s in (0, 1, 2)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(vs[i], vs[j]), f"seeds {i} and {j} collided"


# ------------------------------------------------------------------- T2 ---
# THE most important test in this file.
def test_split_is_identical_regardless_of_training_seed():
    """The data split must NOT move when the training seed moves.

    split_shots_3's own `seed` argument is the SPLIT seed and is pinned at 0 by
    bench.load_filtered_split. If a future edit ever wires --seed into it, the test
    set changes and every cross-run comparison silently becomes invalid.
    """
    shots = list(range(50000, 50200))
    base = split_shots_3(shots)                      # default seed=0
    for training_seed in (0, 1, 2):
        torch.manual_seed(training_seed)
        np.random.seed(training_seed)
        again = split_shots_3(shots)                 # must be unaffected
        assert again == base, (
            f"the split moved when the training seed was {training_seed} -- "
            "the 76 test shots must be identical across all runs")


def test_split_helper_still_honours_its_own_seed_argument():
    """Guard the other direction: split_shots_3 is genuinely seeded, just not by us."""
    shots = list(range(50000, 50200))
    assert split_shots_3(shots, seed=0) != split_shots_3(shots, seed=7)


# ------------------------------------------------------------------- T3 ---
def test_seed_zero_is_the_default():
    """Existing call sites (which pass no seed) must take the seed=0 path."""
    import inspect
    from src.ml.train import train_m2_dcs, train_m3_dcs
    for fn in (train_m2_dcs, train_m3_dcs):
        p = inspect.signature(fn).parameters
        assert "seed" in p, f"{fn.__name__} must accept seed"
        assert p["seed"].default == 0, f"{fn.__name__} seed default must be 0"


# ------------------------------------------------------------------- T4 ---
def test_dataloader_shuffle_order_is_seeded_and_reproducible():
    """Shuffle order must depend on the seed and repeat for a fixed seed."""
    from torch.utils.data import DataLoader

    ds = list(range(64))

    def order(seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return [int(x) for x in DataLoader(ds, batch_size=1, shuffle=True, generator=g)]

    assert order(0) == order(0), "a fixed seed must reproduce the shuffle order"
    assert order(0) != order(1), "different seeds must give different orders"


# ------------------------------------------------------------------- T5 ---
def test_artifact_keys_are_declared_for_both_trainers():
    """The saved artifact must carry seed and best_val_mse so a bench row is traceable.

    Checked by source inspection: a real train call needs a GPU and a dataset, which
    this unit test deliberately avoids. Task 6's smoke run verifies the round-trip.
    """
    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "ml" / "train.py"
    text = src.read_text()
    m2 = text[text.index("def train_m2_dcs"):text.index("def _device")]
    m3 = text[text.index("def train_m3_dcs"):]
    for name, body in (("train_m2_dcs", m2), ("train_m3_dcs", m3)):
        assert '"seed": int(seed)' in body, f"{name} must save the seed"
        assert '"best_val_mse"' in body, f"{name} must save best_val_mse"
