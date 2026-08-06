# src/ml/train.py
# -*- coding: utf-8 -*-
"""Training utilities: M0 HistGBT (``train_save``) + M1/M2 neural (``train_neural``)
+ predict-and-dump helpers.

``train_save`` fits the M0 shape model (32 per-angle HistGBTs) + axis model and
pickles them. ``train_neural`` trains a snapshot regression with AdamW + MSE and
early-stops on val MSE. ``predict_dump_snapshot`` / ``predict_dump_seq`` run a
trained neural model over the test shots and dump per-slice predictions.
"""
import copy
import math
import pathlib

import h5py
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from torch.utils.data import DataLoader

from . import bench
from .dataset import engineer, FEATURE_ORDER, keep_mask
from .dataset import SnapshotDataset, SeqDataset
from .dcs_features import load_dcs_config, load_meta, node_col_map, read_snapshot, read_series
from .metrics import ccc
from .target import N_OUT, load_target, standardize, target_mean_std
from .models import ResMLP, ActSeqGRU
from .predictions import save_predictions
from .split import split_shots_3


def _shot_xy(h5_dir, npz_dir, shot):
    """(X[valid], Y[valid], axis_r[valid], axis_z[valid]) for one shot, or None."""
    f = pathlib.Path(h5_dir) / f"{int(shot)}.h5"
    if not f.exists():
        return None
    X, vf = engineer(f)
    d = np.load(pathlib.Path(npz_dir) / f"{int(shot)}.npz")
    Y = d["Y"].astype(float)
    with h5py.File(f, "r") as h:
        ar = np.asarray(h["magnetic_axis_r"], float)
        az = np.asarray(h["magnetic_axis_z"], float)
    v = (vf & d["valid"].astype(bool) & np.isfinite(X).all(1) & np.isfinite(Y).all(1))
    return X[v], Y[v], ar[v], az[v]


def train_save(h5_dir, npz_dir, out_path, hp, shots=None, max_per_shot=None):
    """Fit M0 (32 per-angle HistGBTs) + axis model (2 HistGBTs) on the train shots;
    save joblib {m0, axis_r, axis_z, keep}; return a meta dict.

    ``shots=None`` uses the split's train shots; pass a list to override (smoke
    tests). ``max_per_shot`` caps per-shot slices (seeded) when given."""
    train = shots if shots is not None else bench.load_filtered_split(npz_dir)[0]
    rng = np.random.default_rng(0)
    Xs, Ys, Ar, Az = [], [], [], []
    for s in train:
        g = _shot_xy(h5_dir, npz_dir, s)
        if g is None:
            continue
        X, Y, a, c = g
        if max_per_shot and len(X) > max_per_shot:
            idx = np.sort(rng.choice(len(X), max_per_shot, replace=False))
            X, Y, a, c = X[idx], Y[idx], a[idx], c[idx]
        Xs.append(X); Ys.append(Y); Ar.append(a); Az.append(c)
    Xtr, Ytr = np.concatenate(Xs), np.concatenate(Ys)
    ar, az = np.concatenate(Ar), np.concatenate(Az)
    keep = Xtr.std(0) > 0
    Xk = Xtr[:, keep]
    m0 = [HistGradientBoostingRegressor(**hp).fit(Xk, Ytr[:, a]) for a in range(Ytr.shape[1])]
    axis_r = HistGradientBoostingRegressor(**hp).fit(Xk, ar)
    axis_z = HistGradientBoostingRegressor(**hp).fit(Xk, az)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"m0": m0, "axis_r": axis_r, "axis_z": axis_z, "keep": keep}, out_path)
    pred = np.column_stack([m.predict(Xk) for m in m0])
    ss_res = float(((Ytr - pred) ** 2).sum())
    ss_tot = float(((Ytr - Ytr.mean(0)) ** 2).sum())
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"hp": hp, "train_r2": train_r2, "n_train": int(len(Ytr)),
            "n_angles": int(Ytr.shape[1]), "n_features": int(keep.sum()),
            "kept_features": [n for n, k in zip(FEATURE_ORDER, keep) if k]}


def train_m0_dcs(npz_dir, out_path, cfg=None, shots=None, max_per_shot=None):
    """Fit 32 per-angle HistGBT on the DCS strict-actuator snapshot. No axis model.

    ``shots=None`` uses the split's train shots; pass a list to override (tests).
    """
    cfg = cfg or load_dcs_config()
    hp = cfg["hp"]["m0"]
    npz_dir = pathlib.Path(npz_dir)
    train = shots if shots is not None else bench.load_filtered_split(npz_dir)[0]
    ncm = node_col_map(load_meta(npz_dir))
    mps = max_per_shot if max_per_shot is not None else cfg.get("max_per_shot")
    rng = np.random.default_rng(0)
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
    assert Ytr.shape[1] == N_OUT, f"expected {N_OUT} target columns, got {Ytr.shape[1]}"
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


class DCSSnapshotDataset(torch.utils.data.Dataset):
    """Per-slice (engineered strict-actuator snapshot -> rho) over a shot set."""

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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = self.rows[i]
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


def _dcs_train_mean_std(npz_dir, shots, cfg, ncm):
    feats = []
    for s in shots:
        p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
        if not p.exists():
            continue
        f, m = read_snapshot(p, cfg, ncm)
        if m.any():
            feats.append(f[m])
    X = np.concatenate(feats)
    return X.mean(0), X.std(0)


def train_m1_dcs(npz_dir, out_path, cfg=None, shots=None):
    """Train ResMLP on the DCS snapshot; val early-stop; save torch state."""
    cfg = cfg or load_dcs_config()
    npz_dir = pathlib.Path(npz_dir)
    if shots is None:
        train, val, _ = bench.load_filtered_split(npz_dir)
    else:
        train, val = split_shots_3(list(shots))[:2]
    ncm = node_col_map(load_meta(npz_dir))
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
    hpm = cfg["hp"]["m1"]
    train_neural(model, DataLoader(ds_tr, batch_size=2048, shuffle=True),
                 DataLoader(ds_va, batch_size=2048), epochs=hpm["epochs"],
                 lr=hpm["lr"], patience=hpm["patience"])
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "keep": keep, "hp": hpm,
                "n_in": int(keep.sum()), "mean": mean, "std": std,
                "tgt_mean": tgt_mean, "tgt_std": tgt_std, "n_out": N_OUT}, out_path)
    return {"best_val_mse": float(model.best_val_mse),
            "stop_epoch": int(model.stop_epoch), "n_in": int(keep.sum())}


class DCSSeqDataset(torch.utils.data.Dataset):
    """Per-shot full actuator series (n_act -> 34 outputs (rho + centre)) over a shot set.

    Each item is one whole shot: ``(A(L, n_act), Y(L, 34), mask(L,))``. The
    series is not subsampled (the GRU is linear-time). Normalization uses the
    per-channel mean/std over valid steps; invalid steps are kept (inputs zeroed by
    ``read_series``, targets zeroed here) and masked out of the loss by the trainer.
    """

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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        A, Y, m = self.rows[i]
        A = ((A - self.mean) / self.std).astype(np.float32)
        return (torch.from_numpy(A), torch.from_numpy(Y),
                torch.from_numpy(m))


def _dcs_series_mean_std(npz_dir, shots, cfg, ncm):
    feats = []
    for s in shots:
        p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
        if not p.exists():
            continue
        a, m = read_series(p, cfg, ncm)
        if m.any():
            feats.append(a[m])
    X = np.concatenate(feats)
    return X.mean(0), X.std(0)


def _gru_forward(model, A, mk):
    """ActSeqGRU forward robust to cuDNN NOT_SUPPORTED on the long DCS series.

    Ensures a contiguous input and, on a cuDNN RuntimeError (no kernel for the
    ~40-58k-step sequence in this mode), retries with cuDNN disabled (native GRU)."""
    A = A.contiguous()
    try:
        return model(A, mk)
    except RuntimeError:
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        out = model(A, mk)
        torch.backends.cudnn.enabled = prev
        return out


def train_m2_dcs(npz_dir, out_path, cfg=None, shots=None):
    """Train ActSeqGRU on the DCS actuator series with masked MSE; val early-stop."""
    cfg = cfg or load_dcs_config()
    npz_dir = pathlib.Path(npz_dir)
    if shots is None:
        train, val, _ = bench.load_filtered_split(npz_dir)
    else:
        train, val = split_shots_3(list(shots))[:2]
    ncm = node_col_map(load_meta(npz_dir))
    mean, std = _dcs_series_mean_std(npz_dir, train, cfg, ncm)
    tgt_mean, tgt_std = target_mean_std(npz_dir, train)
    ds_tr = DCSSeqDataset(npz_dir, train, cfg, ncm, mean, std,
                          tgt_mean=tgt_mean, tgt_std=tgt_std)
    ds_va = DCSSeqDataset(npz_dir, val, cfg, ncm, mean, std,
                          tgt_mean=tgt_mean, tgt_std=tgt_std)
    hpm = cfg["hp"]["m2"]
    model = ActSeqGRU(n_act=ds_tr.n_act, n_out=N_OUT, hidden=hpm["hidden"],
                      layers=hpm["layers"], dropout=hpm["dropout"]).to(_device())
    opt = torch.optim.AdamW(model.parameters(), lr=hpm["lr"], weight_decay=1e-5)
    best, best_state, bad = 1e9, None, 0
    for ep in range(hpm["epochs"]):
        model.train()
        for A, Y, m in DataLoader(ds_tr, batch_size=1, shuffle=True):
            A = A.to(_device()).float(); Y = Y.to(_device()).float()
            mk = m.to(_device()).bool()
            pred = _gru_forward(model, A, mk)
            w = mk.unsqueeze(-1).float()
            loss = (((pred - Y) ** 2) * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for A, Y, m in DataLoader(ds_va, batch_size=1):
                A = A.to(_device()).float(); Y = Y.to(_device()).float()
                mk = m.to(_device()).bool()
                pred = _gru_forward(model, A, mk); w = mk.unsqueeze(-1).float()
                vl += float((((pred - Y) ** 2) * w).sum()); n += int(w.sum().item())
        vl = vl / max(n, 1)
        if vl < best - 1e-7:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hpm["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    out_path = pathlib.Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "n_act": ds_tr.n_act, "hp": hpm,
                "mean": mean, "std": std,
                "tgt_mean": tgt_mean, "tgt_std": tgt_std, "n_out": N_OUT}, out_path)
    model.best_val_mse = float(best)
    return {"best_val_mse": float(best), "n_act": ds_tr.n_act}


def _device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_neural(model, train_loader, val_loader, epochs=80, lr=1e-3,
                 patience=12, warmup=0, sched="none", weight_decay=1e-5,
                 verbose=False):
    """Train a snapshot regression ``model`` (MSE) with AdamW + val early stop.

    The loader is expected to yield ``(xb, yb)`` where ``model(xb) -> yb``.
    ``sched`` selects a learning-rate schedule applied per epoch:
    ``"none"`` (constant lr), or ``"cosine"`` (linear ``warmup`` epochs then
    half-cosine decay to 0). Returns the model loaded with the best-val-loss
    weights; also sets ``model.best_val_mse`` and ``model.stop_epoch``.
    If ``verbose``, prints per-epoch train/val MSE.
    """
    dev = _device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad, ep = 1e9, None, 0, 0
    decay_from = max(warmup, 1)
    for ep in range(epochs):
        if sched == "cosine":
            if warmup and ep < warmup:
                cur = lr * (ep + 1) / warmup
            else:
                cur = lr * 0.5 * (1 + math.cos(math.pi * (ep - decay_from) / max(epochs - decay_from, 1)))
            for g in opt.param_groups:
                g["lr"] = cur
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = ((model(xb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += float(loss.detach() * xb.size(0)); tr_n += xb.size(0)
        model.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += float(((model(xb) - yb) ** 2).sum()); n += yb.numel()
        vl = vl / max(n, 1)
        if verbose:
            print(f"  ep {ep:03d}  lr={opt.param_groups[0]['lr']:.5f}  "
                  f"train_mse={tr_loss/max(tr_n,1):.6f}  val_mse={vl:.6f}")
        if vl < best - 1e-7:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.best_val_mse = float(best)
    model.stop_epoch = ep
    return model


@torch.no_grad()
def predict_dump_snapshot(model, h5_dir, npz_dir, shots, mean, std, out_path):
    """Run a snapshot model over ``shots`` and dump per-slice predictions.

    Applies the same ``keep = std > 0`` column mask used at training time, so
    the model's ``n_in`` matches.
    """
    dev = _device(); model.eval(); model.to(dev)
    keep = keep_mask(std)
    mean_k = np.asarray(mean, float)[keep]
    std_k = np.asarray(std, float)[keep]
    preds = {}
    for s in shots:
        f = pathlib.Path(h5_dir) / f"{int(s)}.h5"
        if not f.exists():
            continue
        X, vfeat = engineer(f)
        d = np.load(pathlib.Path(npz_dir) / f"{int(s)}.npz")
        Y = d["Y"].astype(float)
        v = (vfeat & d["valid"].astype(bool)
             & np.isfinite(Y).all(1) & np.isfinite(X).all(1))
        if not v.any():
            continue
        Xt = torch.from_numpy(((X[v][:, keep] - mean_k) / std_k).astype(np.float32)).to(dev)
        preds[int(s)] = model(Xt).cpu().numpy().astype(np.float32)
    save_predictions(out_path, preds)


@torch.no_grad()
def predict_dump_seq(model, h5_dir, npz_dir, shots, raw_names, mean, std, out_path):
    """Run a sequence model over whole-shot actuator series and dump predictions."""
    import h5py
    dev = _device(); model.eval(); model.to(dev)
    mean = np.asarray(mean, float); std = np.maximum(np.asarray(std, float), 1e-6)
    preds = {}
    for s in shots:
        f = pathlib.Path(h5_dir) / f"{int(s)}.h5"
        if not f.exists():
            continue
        with h5py.File(f, "r") as h:
            t = np.asarray(h["time"], float)
            cols = []
            for nm in raw_names:
                if nm in h and getattr(h[nm], "shape", None) is not None:
                    cols.append(np.asarray(h[nm], float).reshape(-1))
                else:
                    cols.append(np.zeros(t.size))
            A = np.column_stack(cols).astype(np.float32)
        d = np.load(pathlib.Path(npz_dir) / f"{int(s)}.npz")
        Y = d["Y"].astype(float); valid = d["valid"].astype(bool)
        m = np.isfinite(A).all(1) & np.isfinite(Y).all(1) & valid
        if not m.any():
            continue
        At = torch.from_numpy(((A - mean) / std)[None].astype(np.float32)).to(dev)
        mt = torch.from_numpy(m[None]).to(dev)
        pred = model(At, mt)[0].cpu().numpy()
        preds[int(s)] = pred[m].astype(np.float32)
    save_predictions(out_path, preds)


# ── IMAS T0 strict-actuator M1/M2 (reuse SnapshotDataset / SeqDataset) ──

def _imas_snapshot_mean_std(h5_dir, npz_dir, shots):
    """Per-feature ``(mean, std)`` over valid train slices via :func:`engineer`."""
    feats = []
    for s in shots:
        f = pathlib.Path(h5_dir) / f"{int(s)}.h5"
        if not f.exists():
            continue
        X, vf = engineer(f)
        d = np.load(pathlib.Path(npz_dir) / f"{int(s)}.npz")
        Y = d["Y"].astype(float)
        v = (vf & d["valid"].astype(bool)
             & np.isfinite(Y).all(1) & np.isfinite(X).all(1))
        if v.any():
            feats.append(X[v])
    Xall = np.concatenate(feats)
    return Xall.mean(0), Xall.std(0)


def train_m1_imas(h5_dir, npz_dir, out_path, hp, shots=None, max_per_shot=None):
    """Train ResMLP on the IMAS T0 snapshot; val early-stop; save torch state.

    ``shots=None`` uses the split's train/val; pass a list to override (smoke)."""
    if shots is None:
        train, val, _ = bench.load_filtered_split(npz_dir)
    else:
        train, val = split_shots_3(list(shots))[:2]
    mean, std = _imas_snapshot_mean_std(h5_dir, npz_dir, train)
    ds_tr = SnapshotDataset(h5_dir, npz_dir, train, mean, std, max_per_shot)
    ds_va = SnapshotDataset(h5_dir, npz_dir, val, mean, std, max_per_shot)
    keep = ds_tr.keep
    model = ResMLP(n_in=int(keep.sum()), hidden=hp["hidden"],
                   depth=hp["depth"], dropout=hp["dropout"])
    train_neural(model, DataLoader(ds_tr, batch_size=2048, shuffle=True),
                 DataLoader(ds_va, batch_size=2048), epochs=hp["epochs"],
                 lr=hp["lr"], patience=hp["patience"])
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "keep": keep, "hp": hp,
                "n_in": int(keep.sum()), "mean": np.asarray(mean, float),
                "std": np.asarray(std, float)}, out_path)
    return {"best_val_mse": float(model.best_val_mse),
            "stop_epoch": int(model.stop_epoch), "n_in": int(keep.sum())}


def _imas_series_mean_std(h5_dir, npz_dir, shots, raw_names):
    """Per-channel ``(mean, std)`` over valid steps of the raw actuator series."""
    feats = []
    for s in shots:
        f = pathlib.Path(h5_dir) / f"{int(s)}.h5"
        if not f.exists():
            continue
        with h5py.File(f, "r") as h:
            t = np.asarray(h["time"], float)
            cols = [np.asarray(h[nm], float).reshape(-1)
                    if (nm in h and getattr(h[nm], "shape", None) is not None)
                    else np.zeros(t.size) for nm in raw_names]
            A = np.column_stack(cols)
        d = np.load(pathlib.Path(npz_dir) / f"{int(s)}.npz")
        Y = d["Y"].astype(float)
        valid = d["valid"].astype(bool)
        m = np.isfinite(A).all(1) & np.isfinite(Y).all(1) & valid
        if m.any():
            feats.append(A[m])
    Xall = np.concatenate(feats)
    return Xall.mean(0), Xall.std(0)


def train_m2_imas(h5_dir, npz_dir, out_path, hp, raw_names, shots=None):
    """Train ActSeqGRU on the IMAS T0 actuator series with masked MSE; val early-stop.

    Each item is one whole shot ``(A(L,n_act), Y(L,32), mask(L,))``; invalid steps
    are masked out of the loss. ``shots=None`` uses the split's train/val."""
    if shots is None:
        train, val, _ = bench.load_filtered_split(npz_dir)
    else:
        train, val = split_shots_3(list(shots))[:2]
    mean, std = _imas_series_mean_std(h5_dir, npz_dir, train, raw_names)
    ds_tr = SeqDataset(h5_dir, npz_dir, train, raw_names, mean, std)
    ds_va = SeqDataset(h5_dir, npz_dir, val, raw_names, mean, std)
    n_act = int(ds_tr.shots[0][0].shape[1]) if ds_tr.shots else len(raw_names)
    model = ActSeqGRU(n_act=n_act, hidden=hp["hidden"], layers=hp["layers"],
                      dropout=hp["dropout"]).to(_device())
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-5)
    best, best_state, bad = 1e9, None, 0
    for ep in range(hp["epochs"]):
        model.train()
        for A, Y, m in DataLoader(ds_tr, batch_size=1, shuffle=True):
            A = A.to(_device()).float(); Y = Y.to(_device()).float()
            mk = m.to(_device()).bool()
            pred = _gru_forward(model, A, mk); w = mk.unsqueeze(-1).float()
            loss = (((pred - Y) ** 2) * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for A, Y, m in DataLoader(ds_va, batch_size=1):
                A = A.to(_device()).float(); Y = Y.to(_device()).float()
                mk = m.to(_device()).bool()
                pred = _gru_forward(model, A, mk); w = mk.unsqueeze(-1).float()
                vl += float((((pred - Y) ** 2) * w).sum()); n += int(w.sum().item())
        vl = vl / max(n, 1)
        if vl < best - 1e-7:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    out_path = pathlib.Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "n_act": n_act, "hp": hp,
                "mean": np.asarray(mean, float), "std": np.asarray(std, float)}, out_path)
    model.best_val_mse = float(best)
    return {"best_val_mse": float(best), "n_act": n_act}
