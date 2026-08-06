# newTrain centre prediction + unattended retrain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three DCS models predict `r(θ)@32` **and** the absolute polar centre `(Rgeom, Zgeom)` on the filtered `NpzGeom` dataset, then run all six trainings unattended and report against the `NpzOrigin` baseline.

**Architecture:** One new focused module (`src/ml/target.py`) owns target assembly, standardization and column splitting; every training and predict path calls it instead of reading `Y` directly. The models already accept `n_out`, so widening 32 → 34 is a constructor argument. A runner script executes the six trainings sequentially, resumable, and emits one comparison report.

**Tech Stack:** Python 3 (conda env `torch`), numpy, scipy, torch 2.11 (A100), scikit-learn HistGBT, pytest.

**Spec:** `docs/superpowers/specs/2026-08-05-newtrain-rebuild-design.md` (approved). Read §4.1–4.2, §7 and §9 before starting.

## Global Constraints

- Python runs in the conda `torch` env: prefix every command with `conda run -n torch`.
- Never hardcode a core count. Use `len(os.sched_getaffinity(0))` or `nproc`.
- `src/data/filter.py` is the single source of truth for the criteria. Do not redefine a threshold anywhere else.
- There is **no fixed polar origin**. Never reintroduce `(2.5, 0)` into any target, model or metric. `θ` and `r` are measured about `(Rgeom, Zgeom)`.
- The target is **34 columns**: `[0:32]` = `r(θ)`, `[32]` = `Rgeom`, `[33]` = `Zgeom`, all absolute metres.
- Targets are standardized by **train-shot** mean/std only. Never compute statistics over val or test shots.
- Predictions saved to disk and passed to scoring are in **metres** (de-standardized).
- Do not touch `ProjDB/datasets/NpzOrigin/`, `configs/dcs_model.yml`'s channel list, or `ProjDB/trains/dcs_actuator/`. They are the baseline.
- Baseline to beat/report against, 76 test shots: M0 CCC 0.9579 / R² 0.8452, M1 0.9530 / 0.8284, M2 0.9668 / 0.8763.
- Do not `git commit` unless the plan step says to; never `git add -A`.
- 9 tests fail before this work starts (4 `KeyError: 'X_cnt'` in `test_build_npz*`, 5 collection errors from `src.ml.dataset.load_meta`). They are pre-existing. Do not fix them; do not let them mask new failures.

---

### Task 1: `src/ml/target.py` — target assembly and standardization

**Files:**
- Create: `src/ml/target.py`
- Test: `tests/test_ml_target.py`

**Interfaces:**
- Consumes: `NpzGeom/<shot>.npz` arrays `Y (nt,32)`, `center (nt,2)`, `valid (nt,)`.
- Produces, relied on by Tasks 2–5:
  - `N_RHO = 32`, `N_CENTER = 2`, `N_OUT = 34`
  - `load_target(npz_path) -> (T: np.ndarray (nt,34) float64, finite: np.ndarray (nt,) bool)`
  - `target_mean_std(npz_dir, shots, eps=1e-6) -> (mean: (34,) float64, std: (34,) float64)`
  - `standardize(T, mean, std) -> np.ndarray` and `destandardize(P, mean, std) -> np.ndarray`
  - `split_outputs(P) -> (rho: (...,32), centre: (...,2))`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ml_target.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for the 34-column DCS target (r(theta)@32 + absolute centre)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.target import (  # noqa: E402
    N_CENTER, N_OUT, N_RHO, destandardize, load_target, split_outputs,
    standardize, target_mean_std,
)


def _write(path, nt=6, nan_rows=(), valid=None):
    """A minimal NpzGeom-shaped file: Y, center, valid, time."""
    Y = np.tile(np.linspace(0.4, 0.6, N_RHO), (nt, 1))
    C = np.column_stack([np.full(nt, 2.44), np.full(nt, -0.02)])
    for j in nan_rows:
        Y[j] = np.nan
    v = np.ones(nt, bool) if valid is None else np.asarray(valid, bool)
    np.savez(path, Y=Y.astype(np.float32), center=C.astype(np.float32),
             valid=v, time=np.arange(nt, dtype=np.float32) * 0.01)
    return path


def test_load_target_concatenates_rho_then_centre(tmp_path):
    p = _write(tmp_path / "1.npz")
    T, finite = load_target(p)
    assert T.shape == (6, N_OUT) and N_OUT == N_RHO + N_CENTER
    assert np.allclose(T[:, :N_RHO], np.linspace(0.4, 0.6, N_RHO))
    assert np.allclose(T[:, N_RHO], 2.44)      # absolute Rgeom, not an offset
    assert np.allclose(T[:, N_RHO + 1], -0.02)
    assert finite.all()


def test_load_target_flags_nan_rows(tmp_path):
    p = _write(tmp_path / "1.npz", nan_rows=(2, 4))
    T, finite = load_target(p)
    assert not finite[2] and not finite[4] and finite.sum() == 4


def test_load_target_rejects_wrong_shapes(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, Y=np.zeros((5, 31), np.float32),
             center=np.zeros((5, 2), np.float32), valid=np.ones(5, bool))
    with pytest.raises(ValueError, match="rho columns"):
        load_target(p)
    np.savez(p, Y=np.zeros((5, N_RHO), np.float32),
             center=np.zeros((5, 3), np.float32), valid=np.ones(5, bool))
    with pytest.raises(ValueError, match=r"center must be"):
        load_target(p)


def test_target_mean_std_uses_only_finite_valid_rows(tmp_path):
    # row 0 invalid, row 1 NaN -> neither may influence the statistics
    _write(tmp_path / "1.npz", nt=4, nan_rows=(1,),
           valid=[False, True, True, True])
    mean, std = target_mean_std(tmp_path, [1])
    assert mean.shape == (N_OUT,) and std.shape == (N_OUT,)
    assert np.allclose(mean[N_RHO], 2.44)
    assert (std > 0).all(), "std must be floored so division is safe"


def test_standardize_roundtrip(tmp_path):
    p = _write(tmp_path / "1.npz")
    T, _ = load_target(p)
    mean = T.mean(0)
    std = np.maximum(T.std(0), 1e-6)
    Z = standardize(T, mean, std)
    assert np.allclose(destandardize(Z, mean, std), T, atol=1e-9)


def test_split_outputs():
    P = np.arange(2 * N_OUT, dtype=float).reshape(2, N_OUT)
    rho, centre = split_outputs(P)
    assert rho.shape == (2, N_RHO) and centre.shape == (2, N_CENTER)
    assert np.array_equal(centre[:, 0], P[:, N_RHO])
    with pytest.raises(ValueError, match="columns"):
        split_outputs(np.zeros((2, N_RHO)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n torch python -m pytest tests/test_ml_target.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.target'`

- [ ] **Step 3: Write the implementation**

Create `src/ml/target.py`:

```python
# -*- coding: utf-8 -*-
"""The DCS model target: r(theta)@32 plus the absolute polar centre.

34 columns: ``[0:32]`` is ``r(theta)`` about that slice's own centre, ``[32]`` is
``Rgeom`` and ``[33]`` is ``Zgeom``, all in absolute metres. There is deliberately no
fixed reference origin -- theta and r are already measured about (Rgeom, Zgeom), so an
offset against some constant point would leave two competing notions of origin in one
target vector.

Because the centre is absolute, ``Rgeom``'s ~2.44 m mean is out of scale with the ~0.53 m
radii even though the *spreads* match (0.060 / 0.028 / 0.056). Raw MSE would therefore let
one column of 34 dominate the early gradient into the shared trunk of the neural models,
so callers standardize with :func:`target_mean_std` (train shots only) and invert with
:func:`destandardize` before anything downstream sees the numbers.

Rationale and measurements: docs/superpowers/specs/2026-08-05-newtrain-rebuild-design.md §7.
"""
import pathlib

import numpy as np

N_RHO = 32                      # r(theta) columns
N_CENTER = 2                    # (Rgeom, Zgeom)
N_OUT = N_RHO + N_CENTER        # 34


def load_target(npz_path):
    """``(T (nt, 34) float64, finite (nt,) bool)`` for one NpzGeom shot.

    ``finite`` is True only where every column is finite. Slices rejected by the quality
    filters carry ``Y = NaN`` by design, so this is the caller's signal, not an error.
    """
    d = np.load(npz_path)
    Y = d["Y"].astype(float)
    C = d["center"].astype(float)
    if Y.ndim != 2 or Y.shape[1] != N_RHO:
        raise ValueError(f"{npz_path}: expected {N_RHO} rho columns, got {Y.shape}")
    if C.shape != (Y.shape[0], N_CENTER):
        raise ValueError(f"{npz_path}: center must be (nt, {N_CENTER}), got {C.shape}")
    T = np.concatenate([Y, C], axis=1)
    return T, np.isfinite(T).all(axis=1)


def target_mean_std(npz_dir, shots, eps=1e-6):
    """Per-column mean/std over ``shots``' finite **and** valid rows.

    Pass train shots only: statistics taken over val or test rows leak.
    """
    acc = []
    for s in shots:
        p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
        if not p.exists():
            continue
        T, finite = load_target(p)
        v = finite & np.load(p)["valid"].astype(bool)
        if v.any():
            acc.append(T[v])
    if not acc:
        raise ValueError(f"no finite valid target rows in {npz_dir} for {len(shots)} shots")
    A = np.concatenate(acc)
    return A.mean(axis=0), np.maximum(A.std(axis=0), eps)


def standardize(T, mean, std):
    """``(T - mean) / std`` with float64 arithmetic."""
    return (np.asarray(T, float) - np.asarray(mean, float)) / np.asarray(std, float)


def destandardize(P, mean, std):
    """Inverse of :func:`standardize`; returns metres."""
    return np.asarray(P, float) * np.asarray(std, float) + np.asarray(mean, float)


def split_outputs(P):
    """``(rho (..., 32), centre (..., 2))`` from a 34-column array in metres."""
    P = np.asarray(P, float)
    if P.shape[-1] != N_OUT:
        raise ValueError(f"expected {N_OUT} columns, got {P.shape[-1]}")
    return P[..., :N_RHO], P[..., N_RHO:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n torch python -m pytest tests/test_ml_target.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/ml/target.py tests/test_ml_target.py
git commit -m "feat(ml): 34-column DCS target with train-only standardization"
```

---

### Task 2: Fix the NaN-poisoned masked loss (spec §4.1–4.2)

The M2 objective masks by multiplication, and `NpzGeom` stores `Y = NaN` at rejected slices. `NaN × 0 = NaN`, so one rejected slice makes the batch loss **and every gradient** `NaN`. This task is independent of the 34-column change and must land first, because Task 3 cannot be validated while training silently produces `NaN`.

**Files:**
- Modify: `src/ml/train.py` — `DCSSeqDataset.__init__` (the `Y = np.load(p)["Y"]...` / `v = mask & ...` block, ~lines 213–216)
- Test: `tests/test_ml_seq_mask.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: the invariant that `DCSSeqDataset` rows contain **no non-finite target values**, while the mask still marks those steps invalid. Tasks 3–4 rely on it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ml_seq_mask.py`:

```python
# -*- coding: utf-8 -*-
"""The masked sequence loss must survive rejected slices.

NpzGeom stores Y = NaN where the quality filters rejected a slice. M2 masks by
multiplication, and NaN * 0 = NaN -- which poisons the loss and every gradient. The
dataset must therefore zero-fill non-finite targets while keeping them masked out.
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def test_nan_times_zero_poisons_a_multiplicative_mask():
    """Pin the failure mode itself, so nobody 'simplifies' the zero-fill away."""
    Y = torch.tensor([[1.0, float("nan"), 3.0]])
    pred = torch.zeros_like(Y, requires_grad=True)
    w = torch.tensor([[1.0, 0.0, 1.0]])
    loss = (((pred - Y) ** 2) * w).sum() / w.sum()
    assert torch.isnan(loss), "if this ever passes, the hazard is gone and so is the fix"
    loss.backward()
    assert torch.isnan(pred.grad).any(), "NaN target contaminates gradients"


def test_seq_dataset_zero_fills_rejected_targets(tmp_path, monkeypatch):
    from src.ml import train as T

    nt, n_rho, n_act = 8, 32, 3
    Y = np.tile(np.linspace(0.4, 0.6, n_rho), (nt, 1)).astype(np.float32)
    Y[3] = np.nan                                    # a filter-rejected slice
    A = np.ones((nt, n_act), np.float32)
    mask = np.ones(nt, bool)

    monkeypatch.setattr(T, "read_series", lambda p, cfg, ncm: (A.copy(), mask.copy()))
    p = tmp_path / "1.npz"
    np.savez(p, Y=Y, center=np.zeros((nt, 2), np.float32), valid=mask,
             time=np.arange(nt, dtype=np.float32))

    ds = T.DCSSeqDataset(tmp_path, [1], cfg={}, ncm={},
                         mean=np.zeros(n_act), std=np.ones(n_act))
    a, y, m = ds[0]
    assert torch.isfinite(y).all(), "targets handed to the loss must be finite"
    assert not bool(m[3]), "the rejected step must still be masked out"
    assert bool(m[0]) and bool(m[7])

    # the trainer's exact objective must now be finite, with zero gradient at the mask
    pred = torch.zeros_like(y, requires_grad=True)
    w = m.unsqueeze(-1).float()
    loss = (((pred - y) ** 2) * w).sum() / w.sum().clamp(min=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.allclose(pred.grad[3], torch.zeros(n_rho))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n torch python -m pytest tests/test_ml_seq_mask.py -q`
Expected: `test_nan_times_zero_poisons_a_multiplicative_mask` PASSES (it documents the hazard); `test_seq_dataset_zero_fills_rejected_targets` FAILS on `assert torch.isfinite(y).all()`.

- [ ] **Step 3: Write the implementation**

In `src/ml/train.py`, inside `DCSSeqDataset.__init__`, replace:

```python
            A, mask = read_series(p, cfg, ncm)
            Y = np.load(p)["Y"].astype(np.float32)
            v = mask & np.isfinite(Y).all(1) & np.isfinite(A).all(1)
            self.rows.append((A, Y, v))
```

with:

```python
            A, mask = read_series(p, cfg, ncm)
            Y = np.load(p)["Y"].astype(np.float32)
            v = mask & np.isfinite(Y).all(1) & np.isfinite(A).all(1)
            # The trainer masks by multiplication and NaN * 0 = NaN, so a single
            # filter-rejected slice would make the loss and every gradient NaN. Zero-fill
            # the target after the mask is computed -- the mask, not the value, is what
            # excludes the step. Same convention read_series already uses for the inputs.
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            self.rows.append((A, Y, v))
```

Then update the class docstring's last sentence to read:

```
    per-channel mean/std over valid steps; invalid steps are kept (inputs zeroed by
    ``read_series``, targets zeroed here) and masked out of the loss by the trainer.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n torch python -m pytest tests/test_ml_seq_mask.py -q`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/ml/train.py tests/test_ml_seq_mask.py
git commit -m "fix(ml): zero-fill rejected targets so the masked GRU loss stays finite"
```

---

### Task 3: Train all three models on the 34-column target

**Files:**
- Modify: `src/ml/train.py` — `train_m0_dcs`, `DCSSnapshotDataset.__init__`/`__getitem__`, `train_m1_dcs`, `DCSSeqDataset.__init__`, `train_m2_dcs`
- Test: `tests/test_ml_train34.py`

**Interfaces:**
- Consumes: Task 1's `N_OUT`, `load_target`, `target_mean_std`, `standardize`.
- Produces: artifacts that all carry `tgt_mean` and `tgt_std` (each `(34,)` float64) alongside their existing keys, and models whose output width is `N_OUT`. Task 4 reads those keys.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ml_train34.py`:

```python
# -*- coding: utf-8 -*-
"""The three DCS models must emit 34 outputs and ship target standardization stats."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.models import ActSeqGRU, ResMLP  # noqa: E402
from src.ml.target import N_OUT  # noqa: E402


def test_models_accept_34_outputs():
    m1 = ResMLP(n_in=7, n_out=N_OUT, hidden=16, depth=1, dropout=0.0)
    assert m1(torch.zeros(4, 7)).shape == (4, N_OUT)
    m2 = ActSeqGRU(n_act=5, n_out=N_OUT, hidden=8, layers=1, dropout=0.0)
    out = m2(torch.zeros(1, 6, 5), torch.ones(1, 6, dtype=torch.bool))
    assert out.shape == (1, 6, N_OUT)


def test_snapshot_dataset_yields_standardized_34_targets(tmp_path, monkeypatch):
    from src.ml import train as T

    nt, n_feat = 10, 6
    feats = np.random.default_rng(0).normal(size=(nt, n_feat)).astype(np.float32)
    mask = np.ones(nt, bool)
    monkeypatch.setattr(T, "read_snapshot", lambda p, cfg, ncm: (feats.copy(), mask.copy()))

    Y = np.tile(np.linspace(0.4, 0.6, 32), (nt, 1)).astype(np.float32)
    C = np.column_stack([np.full(nt, 2.44), np.full(nt, -0.02)]).astype(np.float32)
    np.savez(tmp_path / "1.npz", Y=Y, center=C, valid=mask,
             time=np.arange(nt, dtype=np.float32))

    from src.ml.target import target_mean_std
    tm, ts = target_mean_std(tmp_path, [1])
    ds = T.DCSSnapshotDataset(tmp_path, [1], cfg={}, ncm={},
                              mean=np.zeros(n_feat), std=np.ones(n_feat),
                              tgt_mean=tm, tgt_std=ts)
    _x, y = ds[0]
    assert y.shape == (N_OUT,)
    assert torch.isfinite(y).all()
    # constant columns standardize to ~0 given the floored std
    assert abs(float(y[32])) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n torch python -m pytest tests/test_ml_train34.py -q`
Expected: `test_models_accept_34_outputs` PASSES (models already take `n_out`); `test_snapshot_dataset_yields_standardized_34_targets` FAILS with `TypeError: __init__() got an unexpected keyword argument 'tgt_mean'`.

- [ ] **Step 3: Write the implementation**

**3a.** In `src/ml/train.py`, add to the imports near `from .metrics import ccc`:

```python
from .target import N_OUT, load_target, standardize, target_mean_std
```

**3b.** `DCSSnapshotDataset.__init__` — add the two keyword arguments and build the target from `load_target`. Replace the signature and the body's `Y`/`rows` lines:

```python
    def __init__(self, npz_dir, shots, cfg, ncm, mean, std, max_per_shot=None,
                 tgt_mean=None, tgt_std=None):
        self.keep = np.asarray(std, float) > 0
        self.mean = np.asarray(mean, float)[self.keep]
        self.std = np.asarray(std, float)[self.keep]
        self.tgt_mean = None if tgt_mean is None else np.asarray(tgt_mean, float)
        self.tgt_std = None if tgt_std is None else np.asarray(tgt_std, float)
        self.rows = []
        rng = np.random.default_rng(0)
        mps = max_per_shot if max_per_shot is not None else cfg.get("max_per_shot")
        for s in shots:
            p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
            if not p.exists():
                continue
            feats, mask = read_snapshot(p, cfg, ncm)
            T_all, finite = load_target(p)
            if self.tgt_mean is not None:
                T_all = standardize(T_all, self.tgt_mean, self.tgt_std)
            v = mask & finite & np.isfinite(feats).all(1)
            idx = np.where(v)[0]
            if mps and idx.size > mps:
                idx = np.sort(rng.choice(idx, mps, replace=False))
            Xk = feats[:, self.keep]
            for i in idx:
                self.rows.append((Xk[i], T_all[i].astype(np.float32)))
```

**3c.** `train_m0_dcs` — fit `N_OUT` regressors on the standardized target and persist the stats. Replace the loop body and dump:

```python
    tgt_mean, tgt_std = target_mean_std(npz_dir, train)
    Xs, Ys = [], []
    for s in train:
        p = npz_dir / f"{int(s)}.npz"
        if not p.exists():
            continue
        feats, mask = read_snapshot(p, cfg, ncm)
        T_all, finite = load_target(p)
        T_all = standardize(T_all, tgt_mean, tgt_std)
        v = mask & finite & np.isfinite(feats).all(1)
        idx = np.where(v)[0]
        if mps and idx.size > mps:
            idx = np.sort(rng.choice(idx, mps, replace=False))
        if idx.size:
            Xs.append(feats[idx]); Ys.append(T_all[idx])
    Xtr, Ytr = np.concatenate(Xs), np.concatenate(Ys)
    keep = keep_mask(Xtr.std(0))
    Xk = Xtr[:, keep]
    m0 = [HistGradientBoostingRegressor(**hp).fit(Xk, Ytr[:, a]) for a in range(Ytr.shape[1])]
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"m0": m0, "keep": keep, "cfg": cfg,
                 "tgt_mean": tgt_mean, "tgt_std": tgt_std}, out_path)
    pred = np.column_stack([m.predict(Xk) for m in m0])
    return {"n_train": int(len(Ytr)), "n_features": int(keep.sum()),
            "n_out": int(Ytr.shape[1]), "train_ccc": float(ccc(pred, Ytr))}
```

Assert the width right after the concat, so a dataset regression fails loudly:

```python
    assert Ytr.shape[1] == N_OUT, f"expected {N_OUT} target columns, got {Ytr.shape[1]}"
```

**3d.** `train_m1_dcs` — compute the stats, pass them to both datasets, widen the model, persist:

```python
    mean, std = _dcs_train_mean_std(npz_dir, train, cfg, ncm)
    tgt_mean, tgt_std = target_mean_std(npz_dir, train)
    ds_tr = DCSSnapshotDataset(npz_dir, train, cfg, ncm, mean, std,
                               tgt_mean=tgt_mean, tgt_std=tgt_std)
    ds_va = DCSSnapshotDataset(npz_dir, val, cfg, ncm, mean, std,
                               tgt_mean=tgt_mean, tgt_std=tgt_std)
    keep = ds_tr.keep
    model = ResMLP(n_in=int(keep.sum()), n_out=N_OUT,
                   hidden=cfg["hp"]["m1"]["hidden"],
                   depth=cfg["hp"]["m1"]["depth"], dropout=cfg["hp"]["m1"]["dropout"])
```

and add to the `torch.save` dict: `"tgt_mean": tgt_mean, "tgt_std": tgt_std, "n_out": N_OUT`.

**3e.** `DCSSeqDataset.__init__` — same two keyword arguments; build from `load_target`, standardize, then zero-fill (keep Task 2's fix, now applied to all 34 columns):

```python
    def __init__(self, npz_dir, shots, cfg, ncm, mean, std,
                 tgt_mean=None, tgt_std=None):
        self.mean = np.asarray(mean, float)
        self.std = np.maximum(np.asarray(std, float), 1e-6)
        self.tgt_mean = None if tgt_mean is None else np.asarray(tgt_mean, float)
        self.tgt_std = None if tgt_std is None else np.asarray(tgt_std, float)
        self.rows = []
        n_act = None
        for s in shots:
            p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
            if not p.exists():
                continue
            A, mask = read_series(p, cfg, ncm)
            T_all, finite = load_target(p)
            if self.tgt_mean is not None:
                T_all = standardize(T_all, self.tgt_mean, self.tgt_std)
            v = mask & finite & np.isfinite(A).all(1)
            # zero-fill AFTER masking: the trainer multiplies by the mask and
            # NaN * 0 = NaN, which would make the loss and every gradient NaN.
            T_all = np.nan_to_num(T_all, nan=0.0, posinf=0.0, neginf=0.0)
            self.rows.append((A, T_all.astype(np.float32), v))
            n_act = A.shape[1]
        self.n_act = int(n_act) if n_act is not None else 0
```

**3f.** `train_m2_dcs` — stats, dataset kwargs, widen, persist:

```python
    tgt_mean, tgt_std = target_mean_std(npz_dir, train)
    ds_tr = DCSSeqDataset(npz_dir, train, cfg, ncm, mean, std,
                          tgt_mean=tgt_mean, tgt_std=tgt_std)
    ds_va = DCSSeqDataset(npz_dir, val, cfg, ncm, mean, std,
                          tgt_mean=tgt_mean, tgt_std=tgt_std)
    hpm = cfg["hp"]["m2"]
    model = ActSeqGRU(n_act=ds_tr.n_act, n_out=N_OUT, hidden=hpm["hidden"],
                      layers=hpm["layers"], dropout=hpm["dropout"]).to(_device())
```

and add `"tgt_mean": tgt_mean, "tgt_std": tgt_std, "n_out": N_OUT` to its `torch.save` dict.

- [ ] **Step 4: Run the tests**

Run: `conda run -n torch python -m pytest tests/test_ml_target.py tests/test_ml_seq_mask.py tests/test_ml_train34.py -q`
Expected: PASS, 10 passed

- [ ] **Step 5: Smoke-train on 3 real shots**

Run:

```bash
conda run -n torch python -c "
import numpy as np, torch
from src.data.filter import CODES
from src.ml.train import train_m0_dcs, train_m1_dcs, train_m2_dcs
from src.ml.dcs_features import load_dcs_config
from src.proj_config import get_proj_config
cfg = get_proj_config(); mc = load_dcs_config('configs/dcs_model_geom.yml')
mc['hp']['m1']['epochs'] = mc['hp']['m2']['epochs'] = 2
shots = sorted(int(p.stem) for p in cfg.npzgeom_dir.glob('*.npz'))[:3]
out = cfg.trains_dir / '_smoke34'
print('m0', train_m0_dcs(cfg.npzgeom_dir, out/'m0.joblib', cfg=mc, shots=shots))
print('m1', train_m1_dcs(cfg.npzgeom_dir, out/'m1.pt', cfg=mc, shots=shots))
print('m2', train_m2_dcs(cfg.npzgeom_dir, out/'m2.pt', cfg=mc, shots=shots))
a = torch.load(out/'m2.pt', map_location='cpu', weights_only=False)
assert a['n_out'] == 34 and np.isfinite(a['tgt_mean']).all()
print('best_val_mse finite:', np.isfinite(a.get('best_val_mse', 0.0)))
"
```

Expected: `m0` reports `n_out: 34`; `m1`/`m2` report a **finite** `best_val_mse` (a `nan` here means Task 2's fix regressed). Then `rm -rf ProjDB/trains/_smoke34`.

- [ ] **Step 6: Commit**

```bash
git add src/ml/train.py tests/test_ml_train34.py
git commit -m "feat(ml): train M0/M1/M2 on 34 standardized outputs (rho + centre)"
```

---

### Task 4: Predict paths emit 34 de-standardized columns

**Files:**
- Modify: `scripts/train_dcs.py` — `_pred_m0`, `_pred_m1`, `_pred_m2`
- Test: `tests/test_pred_destandardize.py`

**Interfaces:**
- Consumes: Task 1's `destandardize`, `N_OUT`; Task 3's artifact keys `tgt_mean` / `tgt_std`.
- Produces: `<run>/m{0,1,2}_pred.npz` whose per-shot arrays are `(n_valid, 34)` **in metres**. Task 5 scores them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pred_destandardize.py`:

```python
# -*- coding: utf-8 -*-
"""Saved predictions must be 34 wide and in metres, not standardized units."""
import importlib.util
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.target import N_OUT, destandardize  # noqa: E402


def _load_train_dcs():
    spec = importlib.util.spec_from_file_location(
        "train_dcs", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train_dcs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_destandardize_is_the_inverse_used_by_predict():
    rng = np.random.default_rng(0)
    mean = np.r_[np.full(32, 0.5), 2.44, -0.02]
    std = np.r_[np.full(32, 0.05), 0.06, 0.03]
    metres = rng.normal(mean, std, size=(7, N_OUT))
    z = (metres - mean) / std
    assert np.allclose(destandardize(z, mean, std), metres)


def test_predict_helpers_reference_target_stats():
    """Every predict path must de-standardize; a missing call is a silent unit error."""
    mod = _load_train_dcs()
    src = pathlib.Path(mod.__file__).read_text()
    for fn in ("_pred_m0", "_pred_m1", "_pred_m2"):
        body = src.split(f"def {fn}(")[1].split("\ndef ")[0]
        assert "destandardize" in body, f"{fn} does not de-standardize its output"
        assert "tgt_mean" in body and "tgt_std" in body, f"{fn} ignores the target stats"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n torch python -m pytest tests/test_pred_destandardize.py -q`
Expected: FAIL — `_pred_m0 does not de-standardize its output`

- [ ] **Step 3: Write the implementation**

In `scripts/train_dcs.py`, add to the imports:

```python
from src.ml.target import N_OUT, destandardize  # noqa: E402
```

`_pred_m0` — read the stats from the artifact and invert:

```python
def _pred_m0(art, npz_dir, shots, cfg, ncm, out):
    a = joblib.load(art); m0, keep = a["m0"], a["keep"]
    tgt_mean, tgt_std = a["tgt_mean"], a["tgt_std"]
    preds = {}
    for s in shots:
        p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
        if not p.exists():
            continue
        feats, mask = read_snapshot(p, cfg, ncm)
        T_all, finite = load_target(p)
        v = mask & finite & np.isfinite(feats).all(1)
        if v.any():
            Xk = feats[v][:, keep]
            z = np.column_stack([m.predict(Xk) for m in m0])
            preds[int(s)] = destandardize(z, tgt_mean, tgt_std).astype(np.float32)
    save_predictions(out, preds)
```

`_pred_m1` — same treatment; also widen the model construction:

```python
    model = ResMLP(a["n_in"], n_out=a.get("n_out", N_OUT), hidden=hp["hidden"],
                   depth=hp["depth"], dropout=hp["dropout"]).to(_device()).eval()
```

and replace the prediction line with:

```python
                z = model(torch.from_numpy(Xk).to(_device())).cpu().numpy()
                preds[int(s)] = destandardize(z, a["tgt_mean"], a["tgt_std"]).astype(np.float32)
```

`_pred_m2` — widen and invert:

```python
    model = ActSeqGRU(n_act=a["n_act"], n_out=a.get("n_out", N_OUT), hidden=hp["hidden"],
                      layers=hp["layers"], dropout=hp["dropout"]).to(_device()).eval()
```

```python
                z = yhat[0].cpu().numpy()[v]
                preds[int(s)] = destandardize(z, a["tgt_mean"], a["tgt_std"]).astype(np.float32)
```

In all three, replace the `Y = np.load(p)["Y"]...` / `v = mask & np.isfinite(Y)...` lines with the `load_target` form shown for `_pred_m0`, and add the import:

```python
from src.ml.target import load_target  # noqa: E402
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n torch python -m pytest tests/test_pred_destandardize.py -q`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/train_dcs.py tests/test_pred_destandardize.py
git commit -m "feat(ml): predict paths emit 34 columns in metres"
```

---

### Task 5: Scoring — split columns, add centre and absolute-boundary metrics

`bench.score_predictions` compares predictions to `_shot_y` and **silently skips** any shot whose shape mismatches, returning `n_shots = 0` while still writing a bench row. A 34-wide prediction against a 32-wide truth would do exactly that.

**Files:**
- Create: `src/ml/score34.py`
- Modify: `scripts/train_dcs.py` — `_score`
- Test: `tests/test_score34.py`

**Interfaces:**
- Consumes: Task 1's `load_target`, `split_outputs`, `N_OUT`; `src.ml.metrics.boundary_metrics`; `src.ml.axis_frame.reconstruct_absolute`.
- Produces: `score_dcs34(pred_path, npz_dir, train_shots, test_shots, theta) -> dict` with keys `ccc`, `r2`, `similarity`, `rmse_cm`, `ccc_p90`, `n_shots`, `rgeom_mae_mm`, `zgeom_mae_mm`, `centre_rmse_mm`, `abs_bnd_rmse_mm`. Task 6 writes these to CSV.

- [ ] **Step 1: Write the failing test**

Create `tests/test_score34.py`:

```python
# -*- coding: utf-8 -*-
"""34-column scoring: rho metrics stay baseline-comparable; centre reported separately."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.predictions import save_predictions  # noqa: E402
from src.ml.target import N_OUT, N_RHO  # noqa: E402


def _fixture(tmp_path, n_shots=3, nt=20, err_m=0.0):
    """Write NpzGeom-shaped shots plus predictions offset by ``err_m`` metres."""
    theta = np.arange(N_RHO) / N_RHO * 2 * np.pi
    rng = np.random.default_rng(0)
    shots, preds = [], {}
    for k in range(n_shots):
        s = 100 + k
        rho = 0.5 + 0.02 * np.sin(theta)[None, :] + rng.normal(0, 1e-3, (nt, N_RHO))
        C = np.column_stack([np.full(nt, 2.44), np.full(nt, -0.02)])
        R = C[:, :1] + rho * np.cos(theta)[None, :]
        Z = C[:, 1:2] + rho * np.sin(theta)[None, :]
        bnd = np.stack([R, Z], axis=-1).astype(np.float32)
        np.savez(tmp_path / f"{s}.npz", Y=rho.astype(np.float32),
                 center=C.astype(np.float32), bnd_RZ=bnd,
                 valid=np.ones(nt, bool), time=np.arange(nt, dtype=np.float32))
        preds[s] = np.concatenate([rho + err_m, C + err_m], axis=1).astype(np.float32)
        shots.append(s)
    (tmp_path / "meta.json").write_text(
        '{"shots": [' + ",".join(f'{{"shot": {s}}}' for s in shots) + '],'
        ' "theta_deg": ' + str(list(np.degrees(theta))) + '}')
    save_predictions(tmp_path / "p.npz", preds)
    return shots, theta


def test_perfect_prediction_scores_perfectly(tmp_path):
    from src.ml.score34 import score_dcs34
    shots, theta = _fixture(tmp_path, err_m=0.0)
    m = score_dcs34(tmp_path / "p.npz", tmp_path, shots, shots, theta)
    assert m["n_shots"] == len(shots)
    assert m["ccc"] > 0.999 and m["r2"] > 0.999
    assert m["rgeom_mae_mm"] < 1e-6 and m["zgeom_mae_mm"] < 1e-6
    assert m["abs_bnd_rmse_mm"] < 1e-6


def test_known_offset_lands_in_the_right_units(tmp_path):
    from src.ml.score34 import score_dcs34
    shots, theta = _fixture(tmp_path, err_m=0.001)      # 1 mm on every column
    m = score_dcs34(tmp_path / "p.npz", tmp_path, shots, shots, theta)
    assert m["rgeom_mae_mm"] == pytest.approx(1.0, abs=1e-6)
    assert m["zgeom_mae_mm"] == pytest.approx(1.0, abs=1e-6)
    assert m["centre_rmse_mm"] == pytest.approx(np.sqrt(2.0), abs=1e-6)
    assert 0.5 < m["abs_bnd_rmse_mm"] < 5.0


def test_wrong_width_raises_instead_of_scoring_zero_shots(tmp_path):
    from src.ml.score34 import score_dcs34
    shots, theta = _fixture(tmp_path)
    bad = {s: np.zeros((20, N_RHO), np.float32) for s in shots}   # 32, not 34
    save_predictions(tmp_path / "bad.npz", bad)
    with pytest.raises(ValueError, match=str(N_OUT)):
        score_dcs34(tmp_path / "bad.npz", tmp_path, shots, shots, theta)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n torch python -m pytest tests/test_score34.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.score34'`

- [ ] **Step 3: Write the implementation**

Create `src/ml/score34.py`:

```python
# -*- coding: utf-8 -*-
"""Score 34-column DCS predictions: r(theta) plus the absolute polar centre.

The rho block is scored with exactly the metric the 32-column baseline used, so those
numbers stay comparable in kind. The centre is reported on its own scale in millimetres,
and the absolute-boundary RMSE combines both -- it is the only number that reflects the
error a downstream consumer of the reconstructed (R,Z) actually sees.

Width is validated up front and a raise is preferred to a skip: ``bench.score_predictions``
silently drops shots whose shapes disagree and returns ``n_shots = 0`` while still writing
a row, which looks like a scored run and is not one.
"""
import pathlib

import numpy as np

from .axis_frame import reconstruct_absolute
from .metrics import boundary_metrics, ccc
from .predictions import load_predictions
from .target import N_OUT, N_RHO, load_target, split_outputs


def _truth(npz_dir, shot, t_min=0.0):
    """``(T (n,34), bnd_RZ (n,32,2))`` over the rows the predict paths kept."""
    p = pathlib.Path(npz_dir) / f"{int(shot)}.npz"
    d = np.load(p)
    T, finite = load_target(p)
    v = finite & d["valid"].astype(bool) & (d["time"].astype(float) >= t_min)
    return T[v], d["bnd_RZ"][v].astype(float)


def score_dcs34(pred_path, npz_dir, train_shots, test_shots, theta):
    """Pooled metrics over ``test_shots``. ``theta`` is the (32,) angle grid in radians."""
    preds = load_predictions(pred_path)
    for s, yp in preds.items():
        if np.asarray(yp).shape[-1] != N_OUT:
            raise ValueError(f"shot {s}: predictions are {np.asarray(yp).shape[-1]} wide, "
                             f"expected {N_OUT}")
    rho_tr = np.concatenate([_truth(npz_dir, s)[0][:, :N_RHO] for s in train_shots])
    y_train_mean = rho_tr.mean(axis=0)

    rp, rt, cp, ct, ap, at = [], [], [], [], [], []
    per_shot_ccc = []
    for s in test_shots:
        s = int(s)
        if s not in preds:
            continue
        T, bnd = _truth(npz_dir, s)
        P = np.asarray(preds[s], float)
        if P.shape[0] != T.shape[0]:
            raise ValueError(f"shot {s}: {P.shape[0]} predicted rows vs {T.shape[0]} truth "
                             "rows -- the predict mask and the scoring mask disagree")
        pr, pc = split_outputs(P)
        tr_, tc = split_outputs(T)
        rp.append(pr); rt.append(tr_)
        cp.append(pc); ct.append(tc)
        per_shot_ccc.append(ccc(pr, tr_))
        Rp, Zp = reconstruct_absolute(pc[:, 0], pc[:, 1], pr, theta)
        ap.append(np.stack([Rp, Zp], axis=-1)); at.append(bnd)
    if not rp:
        raise ValueError("no test shot had predictions -- nothing was scored")

    m = boundary_metrics(np.concatenate(rp), np.concatenate(rt),
                         y_train_mean=y_train_mean)
    C_p, C_t = np.concatenate(cp), np.concatenate(ct)
    err = (C_p - C_t) * 1000.0                                  # mm
    m["rgeom_mae_mm"] = float(np.abs(err[:, 0]).mean())
    m["zgeom_mae_mm"] = float(np.abs(err[:, 1]).mean())
    m["centre_rmse_mm"] = float(np.sqrt((err ** 2).sum(axis=1).mean()))
    A_p, A_t = np.concatenate(ap), np.concatenate(at)
    m["abs_bnd_rmse_mm"] = float(np.sqrt((((A_p - A_t) * 1000.0) ** 2).sum(axis=-1).mean()))
    m["per_shot_ccc"] = per_shot_ccc
    m["ccc_p90"] = float(np.percentile(per_shot_ccc, 90)) if per_shot_ccc else float("nan")
    m["n_shots"] = len(per_shot_ccc)
    return m
```

Then in `scripts/train_dcs.py` replace `_score` with:

```python
def _score(npz_dir, pred_path, train_shots, test_shots):
    meta = load_meta(npz_dir)
    theta = np.deg2rad(np.asarray(meta["theta_deg"], float))
    m = score_dcs34(pred_path, str(npz_dir), train_shots, test_shots, theta)
    if m["n_shots"] == 0:
        raise SystemExit(f"{pred_path}: scored 0 shots -- refusing to write a bench row")
    return {"ccc": m["ccc"], "r2": m["r2"], "similarity": m["similarity"],
            "rmse_cm": m["rmse_cm"], "ccc_p90": m["ccc_p90"], "n_shots": m["n_shots"],
            "rgeom_mae_mm": m["rgeom_mae_mm"], "zgeom_mae_mm": m["zgeom_mae_mm"],
            "centre_rmse_mm": m["centre_rmse_mm"],
            "abs_bnd_rmse_mm": m["abs_bnd_rmse_mm"]}
```

Add `from src.ml.score34 import score_dcs34  # noqa: E402` to the imports, and extend the CSV fieldnames list in `main` to:

```python
        w = csv.DictWriter(f, fieldnames=["model", "ccc", "r2", "similarity", "rmse_cm",
                                          "ccc_p90", "n_shots", "rgeom_mae_mm",
                                          "zgeom_mae_mm", "centre_rmse_mm",
                                          "abs_bnd_rmse_mm"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n torch python -m pytest tests/test_score34.py -q`
Expected: PASS, 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/ml/score34.py scripts/train_dcs.py tests/test_score34.py
git commit -m "feat(ml): score rho, centre and absolute boundary from 34-column preds"
```

---

### Task 6: Unattended runner for the six trainings

Six trainings (runs A and B × M0/M1/M2) must survive being launched and left. The runner is sequential (each training already uses the GPU), logs per run, **skips work already done** so a crash resumes rather than restarts, and writes one report at the end.

**Files:**
- Create: `scripts/run_newtrain.py`
- Test: `tests/test_run_newtrain.py`

**Interfaces:**
- Consumes: Task 5's bench CSV columns; `scripts/train_dcs.py`'s CLI (`--npz-dir`, `--config`, `--run-name`, `--bench-out`).
- Produces: `ProjDB/Stats/dcs_predictor/bench_table_geom.csv` (one row per model per run, with a `run` column) and `docs/newtrain_results.md`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_newtrain.py`:

```python
# -*- coding: utf-8 -*-
"""The unattended runner must be resumable and must not fabricate results."""
import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_newtrain",
        pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_newtrain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_matrix_is_the_two_specified_runs():
    mod = _load()
    names = [r["run_name"] for r in mod.RUNS]
    assert names == ["dcs_actuator_geom", "dcs_actuator_geom_nope"]
    a, b = mod.RUNS
    assert a["config"].endswith("dcs_model_geom.yml")      # with time PE
    assert b["config"].endswith("dcs_model.yml")           # PE ablation
    assert all(r["npz_dir"].endswith("NpzGeom") for r in mod.RUNS)


def test_done_detects_a_finished_unit(tmp_path):
    mod = _load()
    run_dir = tmp_path / "dcs_actuator_geom"
    run_dir.mkdir()
    assert not mod.is_done(run_dir, "m0")
    (run_dir / "m0.joblib").write_text("x")
    assert not mod.is_done(run_dir, "m0"), "an artifact alone is not a finished unit"
    (run_dir / "m0_pred.npz").write_text("x")
    assert mod.is_done(run_dir, "m0")


def test_report_marks_missing_rows_rather_than_inventing_them(tmp_path):
    mod = _load()
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text(
        "run,model,ccc,r2,similarity,rmse_cm,ccc_p90,n_shots,rgeom_mae_mm,"
        "zgeom_mae_mm,centre_rmse_mm,abs_bnd_rmse_mm\n"
        "dcs_actuator_geom,m0,0.96,0.85,0.999,2.0,0.98,76,1.5,0.9,1.8,3.1\n")
    out = tmp_path / "report.md"
    mod.write_report(csv_path, out)
    text = out.read_text()
    assert "dcs_actuator_geom" in text and "0.96" in text
    assert "not run" in text, "absent runs must be shown as absent"
    assert "0.9579" in text, "the baseline must appear for comparison"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n torch python -m pytest tests/test_run_newtrain.py -q`
Expected: FAIL — `FileNotFoundError` / no `scripts/run_newtrain.py`

- [ ] **Step 3: Write the implementation**

Create `scripts/run_newtrain.py`:

```python
# -*- coding: utf-8 -*-
"""Run the newTrain retrain matrix unattended, then report against the baseline.

Six units: two runs (with and without the time positional encoding) x three models.
Sequential by design -- each training already saturates one GPU. Every unit is skipped if
its artifact *and* its predictions exist, so a crash or a kill resumes instead of
restarting. Nothing here invents a number: a unit that did not produce a bench row shows
up as "not run" in the report.

Usage:
  python scripts/run_newtrain.py                 # run everything outstanding
  python scripts/run_newtrain.py --report-only   # just regenerate the report
"""
import argparse
import csv
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.proj_config import get_proj_config  # noqa: E402

CFG = get_proj_config()
MODELS = ("m0", "m1", "m2")
RUNS = [
    {"run_name": "dcs_actuator_geom",
     "config": "configs/dcs_model_geom.yml",
     "npz_dir": "ProjDB/datasets/NpzGeom",
     "label": "with time PE (headline)"},
    {"run_name": "dcs_actuator_geom_nope",
     "config": "configs/dcs_model.yml",
     "npz_dir": "ProjDB/datasets/NpzGeom",
     "label": "no time PE (ablation)"},
]
BENCH = CFG.stats_dir / "dcs_predictor" / "bench_table_geom.csv"
REPORT = CFG.base_dir / "docs" / "newtrain_results.md"
BASELINE = {"m0": (0.9579, 0.8452), "m1": (0.9530, 0.8284), "m2": (0.9668, 0.8763)}


def is_done(run_dir, model):
    """A unit counts as finished only when both its artifact and predictions exist."""
    art = run_dir / (f"{model}.joblib" if model == "m0" else f"{model}.pt")
    return art.exists() and (run_dir / f"{model}_pred.npz").exists()


def train_one(run, model, log_dir):
    """Invoke train_dcs.py for one (run, model). Returns (ok, seconds)."""
    log = log_dir / f"{run['run_name']}_{model}.log"
    cmd = [sys.executable, "scripts/train_dcs.py", model,
           "--npz-dir", run["npz_dir"], "--config", run["config"],
           "--run-name", run["run_name"], "--bench-out", str(BENCH)]
    t0 = time.perf_counter()
    with log.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=str(CFG.base_dir), stdout=fh,
                             stderr=subprocess.STDOUT)
    return rc == 0, time.perf_counter() - t0


def _rows(csv_path):
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            out[(r.get("run", ""), r["model"])] = r          # last row wins
    return out


def write_report(csv_path, out_path):
    """Markdown comparison of every unit against the NpzOrigin baseline."""
    rows = _rows(csv_path)
    L = ["# newTrain retrain results", "",
         "Dataset `NpzGeom` (GMAG-native time base, S0-S5 filters, per-slice centre,",
         "34 outputs = r(theta)@32 + absolute (Rgeom, Zgeom)).",
         "Baseline is `NpzOrigin` / `dcs_actuator`, 76 test shots.", "",
         "Five things differ from the baseline at once (time base, slice population,",
         "target origin, input columns, output width), so a difference in the r(theta)",
         "numbers is **not attributable to any single one**. Run B isolates only the PE.",
         "A drop is not necessarily a regression: the baseline was partly scored on",
         "interpolated boundaries and on slices these filters reject.", ""]
    for run in RUNS:
        L += [f"## {run['run_name']} — {run['label']}", "",
              "| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | "
              "centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            r = rows.get((run["run_name"], m))
            b = BASELINE[m]
            if r is None:
                L.append(f"| {m} | not run | | | | | | | | {b[0]:.4f} / {b[1]:.4f} |")
                continue
            def g(k):
                v = r.get(k, "")
                try:
                    return f"{float(v):.4f}"
                except (TypeError, ValueError):
                    return "—"
            L.append(f"| {m} | {g('ccc')} | {g('r2')} | {g('rmse_cm')} | "
                     f"{g('rgeom_mae_mm')} | {g('zgeom_mae_mm')} | "
                     f"{g('centre_rmse_mm')} | {g('abs_bnd_rmse_mm')} | "
                     f"{r.get('n_shots','—')} | {b[0]:.4f} / {b[1]:.4f} |")
        L.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        log_dir = CFG.stats_dir / "dcs_predictor" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        BENCH.parent.mkdir(parents=True, exist_ok=True)
        for run in RUNS:
            run_dir = CFG.trains_dir / run["run_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            for m in MODELS:
                if is_done(run_dir, m):
                    print(f"[skip] {run['run_name']}/{m} already complete", flush=True)
                    continue
                print(f"[run ] {run['run_name']}/{m} ...", flush=True)
                ok, secs = train_one(run, m, log_dir)
                print(f"[{'ok  ' if ok else 'FAIL'}] {run['run_name']}/{m} "
                      f"in {secs / 60:.1f} min (log in {log_dir})", flush=True)
    p = write_report(BENCH, REPORT)
    print(f"report -> {p}")


if __name__ == "__main__":
    main()
```

The bench CSV needs a `run` column. In `scripts/train_dcs.py`'s `main`, add `"run"` as the first fieldname and set it on each row:

```python
        rows.append({"run": args.run_name or cfg["run"], "model": mdl, **sc})
```

and put `"run"` first in the `DictWriter` fieldnames list.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n torch python -m pytest tests/test_run_newtrain.py -q`
Expected: PASS, 3 passed

- [ ] **Step 5: Verify the report renders with no results yet**

Run: `conda run -n torch python scripts/run_newtrain.py --report-only && sed -n '1,24p' docs/newtrain_results.md`
Expected: a report where every model row reads `not run`, with the baseline column populated.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_newtrain.py scripts/train_dcs.py tests/test_run_newtrain.py
git commit -m "feat(scripts): resumable unattended runner for the newTrain matrix"
```

---

### Task 7: Pre-flight checks, then launch the six trainings

**Files:**
- Create: `tests/test_newtrain_preflight.py`
- Modify: `docs/newtrain_results.md` (generated), `docs/data_lineage.md` (results pointer)

**Interfaces:**
- Consumes: everything above.
- Produces: the populated bench CSV and results report.

- [ ] **Step 1: Write the pre-flight test**

Create `tests/test_newtrain_preflight.py`:

```python
# -*- coding: utf-8 -*-
"""Checks that must hold on the real dataset before six trainings are worth starting."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.axis_frame import reconstruct_absolute  # noqa: E402
from src.ml.dcs_features import load_dcs_config, load_meta, node_col_map  # noqa: E402
from src.ml.target import N_OUT, load_target, target_mean_std  # noqa: E402
from src.proj_config import get_proj_config  # noqa: E402

CFG = get_proj_config()
pytestmark = pytest.mark.skipif(not CFG.npzgeom_dir.exists(), reason="NpzGeom not built")


def _some_shots(n=6):
    return sorted(int(p.stem) for p in CFG.npzgeom_dir.glob("*.npz"))[:n]


def test_target_is_34_wide_and_centre_is_absolute():
    for s in _some_shots(3):
        T, finite = load_target(CFG.npzgeom_dir / f"{s}.npz")
        assert T.shape[1] == N_OUT
        assert finite.any()
        assert 2.0 < np.nanmedian(T[finite, 32]) < 3.0, "Rgeom must be absolute metres"


def test_truth_roundtrip_sets_the_absolute_metric_floor():
    """reconstruct_absolute on the TRUTH columns vs bnd_RZ -- the floor a model cannot beat."""
    theta = np.deg2rad(np.asarray(load_meta(CFG.npzgeom_dir)["theta_deg"], float))
    worst = 0.0
    for s in _some_shots(4):
        d = np.load(CFG.npzgeom_dir / f"{s}.npz")
        T, finite = load_target(CFG.npzgeom_dir / f"{s}.npz")
        v = finite & d["valid"].astype(bool)
        R, Z = reconstruct_absolute(T[v, 32], T[v, 33], T[v, :32], theta)
        err = np.hypot(R - d["bnd_RZ"][v, :, 0], Z - d["bnd_RZ"][v, :, 1])
        worst = max(worst, float(np.percentile(err, 99)) * 1000.0)
    print(f"\nabsolute-boundary floor (truth round-trip, p99): {worst:.2f} mm")
    assert worst < 20.0, "a >2 cm floor would mean the representation, not the model, is wrong"


def test_pe_channels_resolve_for_the_headline_config():
    meta = load_meta(CFG.npzgeom_dir)
    ncm = node_col_map(meta)
    mc = load_dcs_config("configs/dcs_model_geom.yml")
    missing = [c["node"] for c in mc["channels"] if c["node"] not in ncm]
    assert not missing, f"config lists channels absent from the dataset: {missing}"
    assert sum(1 for c in mc["channels"] if c["kind"] == "pe") == 10


def test_target_stats_come_from_train_shots_only():
    from src.ml import bench
    train, _val, test = bench.load_filtered_split(CFG.npzgeom_dir)
    mean, std = target_mean_std(CFG.npzgeom_dir, train[:20])
    assert mean.shape == (N_OUT,) and (std > 0).all()
    assert set(train).isdisjoint(test), "split must not overlap"
```

- [ ] **Step 2: Run the pre-flight and the whole new-code suite**

Run:

```bash
conda run -n torch python -m pytest tests/test_ml_target.py tests/test_ml_seq_mask.py \
  tests/test_ml_train34.py tests/test_pred_destandardize.py tests/test_score34.py \
  tests/test_run_newtrain.py tests/test_newtrain_preflight.py -q
```

Expected: all PASS. Note the printed absolute-boundary floor — model numbers are meaningless below it.

- [ ] **Step 3: Commit the pre-flight**

```bash
git add tests/test_newtrain_preflight.py
git commit -m "test: pre-flight checks for the newTrain retrain"
```

- [ ] **Step 4: Launch the six trainings unattended**

Run in the background and leave it:

```bash
conda run -n torch python scripts/run_newtrain.py > /tmp/newtrain_runner.log 2>&1 &
```

Poll with `tail -20 /tmp/newtrain_runner.log` and the per-unit logs in
`ProjDB/Stats/dcs_predictor/logs/`. If it dies, re-run the same command — finished units
are skipped.

Do **not** report any number until the corresponding unit prints `[ok  ]`.

- [ ] **Step 5: Regenerate the report and sanity-check it**

Run: `conda run -n torch python scripts/run_newtrain.py --report-only && cat docs/newtrain_results.md`

Check, and state plainly in the summary if any of these is violated:
- every unit has `n_shots` = 76 (a 0 means scoring silently matched nothing — Task 5 raises instead, so this should be impossible)
- `abs_bnd_rmse_mm` exceeds the truth-round-trip floor printed in Step 2
- `centre_rmse_mm` is finite and not absurd (`Rgeom` spans 2.065–2.701 m, so an MAE above ~60 mm means the centre head learned nothing)

- [ ] **Step 6: Add a results pointer to the lineage doc**

In `docs/data_lineage.md` §4b, after the "Consumers must opt in" paragraph, add:

```markdown
Retrain results (runs A and B, three models each): [`newtrain_results.md`](newtrain_results.md).
```

- [ ] **Step 7: Commit the results**

```bash
git add docs/newtrain_results.md docs/data_lineage.md ProjDB/Stats/dcs_predictor/bench_table_geom.csv
git commit -m "docs: newTrain retrain results vs the NpzOrigin baseline"
```

---

## Self-Review

**Spec coverage.** §4.1–4.2 (loss masking + the NaN×0 fix) → Task 2. §7.1 (34-column absolute target) → Task 1 + Task 3. §7.1a (train-only standardization) → Task 1 `target_mean_std`, applied in Task 3, inverted in Task 4. §7.2 (model changes) → Task 3. §7.3 (flat loss weighting) → no code: plain MSE over 34 standardized columns *is* the flat weighting; no task needed. §7.4 (split scoring, silent-skip hazard) → Task 5. §8 (run matrix, plumbing) → Task 6. §9 (runs A and B only) → Task 6 `RUNS`. §11 planned checks (34-wide assertion, `n_shots > 0`, truth round-trip floor) → Task 5 raises on width and empty scoring; Task 7 tests the floor.

**Placeholder scan.** No TBD/TODO; every code step carries the actual code; no "similar to Task N".

**Type consistency.** `load_target` returns `(T, finite)` and is called that way in Tasks 3, 4, 5, 7. `target_mean_std(npz_dir, shots)` → `(mean, std)`, stored as artifact keys `tgt_mean`/`tgt_std` (Task 3) and read under those exact names (Task 4). `split_outputs` → `(rho, centre)` used only in Task 5. `N_OUT`/`N_RHO` imported from `src.ml.target` everywhere. `is_done(run_dir, model)` and `write_report(csv_path, out_path)` match their tests.

**Known gap, deliberate.** Task 4's `_pred_*` paths and Task 5's `_truth` must select the same rows; the plan makes `score_dcs34` **raise** on a row-count mismatch rather than skip, so if the two masks ever diverge the run stops instead of silently scoring a subset.
