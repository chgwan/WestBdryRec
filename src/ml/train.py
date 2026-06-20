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

from . import bench, features as F
from .dataset import keep_mask
from .predictions import save_predictions

BEST_HP = dict(max_iter=200, max_depth=8, learning_rate=0.1, l2_regularization=1.0)


def _shot_xy(h5_dir, npz_dir, shot):
    """(X[valid], Y[valid], axis_r[valid], axis_z[valid]) for one shot, or None."""
    f = pathlib.Path(h5_dir) / f"{int(shot)}.h5"
    if not f.exists():
        return None
    X, vf = F.engineer(f)
    d = np.load(pathlib.Path(npz_dir) / f"{int(shot)}.npz")
    Y = d["Y"].astype(float)
    with h5py.File(f, "r") as h:
        ar = np.asarray(h["magnetic_axis_r"], float)
        az = np.asarray(h["magnetic_axis_z"], float)
    v = (vf & d["valid"].astype(bool) & np.isfinite(X).all(1) & np.isfinite(Y).all(1))
    return X[v], Y[v], ar[v], az[v]


def train_save(h5_dir, npz_dir, out_path, hp=BEST_HP):
    """Fit M0 (32 per-angle HistGBTs) + axis model (2 HistGBTs) on the train shots;
    save joblib {m0, axis_r, axis_z, keep}; return a meta dict."""
    train, _val, _test = bench.load_filtered_split(npz_dir)
    Xs, Ys, Ar, Az = [], [], [], []
    for s in train:
        g = _shot_xy(h5_dir, npz_dir, s)
        if g is None:
            continue
        Xs.append(g[0]); Ys.append(g[1]); Ar.append(g[2]); Az.append(g[3])
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
            "kept_features": [n for n, k in zip(F.FEATURE_ORDER, keep) if k]}


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
def predict_dump_snapshot(model, h5_dir, npz_dir, shots, feature_fn, mean, std, out_path):
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
        X, vfeat = feature_fn(f)
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
