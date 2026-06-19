# src/ml/bench.py
# -*- coding: utf-8 -*-
"""Unified benchmarking: score a model's saved per-slice test predictions, pooled on test."""
import json
import pathlib
import numpy as np

from .split import split_shots_3
from .metrics import boundary_metrics
from .predictions import load_predictions


def load_filtered_split(npz_dir, val_frac=0.1, test_frac=0.1, seed=0):
    meta = json.loads(pathlib.Path(npz_dir).joinpath("meta.json").read_text())
    shots = [s["shot"] for s in meta["shots"]]
    return split_shots_3(shots, val_frac, test_frac, seed)


def _shot_y(npz_dir, shot):
    d = np.load(pathlib.Path(npz_dir) / f"{int(shot)}.npz")
    v = d["valid"].astype(bool)
    return d["Y"][v].astype(float)


def _per_shot_r2(pred, true, y_train_mean):
    ss_tot = float(((true - y_train_mean) ** 2).sum())
    ss_res = float(((true - pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def score_predictions(pred_path, npz_dir, train_shots, test_shots):
    preds = load_predictions(pred_path)
    ytr = np.concatenate([_shot_y(npz_dir, s) for s in train_shots])
    y_train_mean = ytr.mean(axis=0)
    yp_all, yt_all, per_shot = [], [], []
    for s in test_shots:
        s = int(s)
        if s not in preds:
            continue
        yt = _shot_y(npz_dir, s)
        yp = preds[s]
        if yp.shape != yt.shape:
            continue                                 # mis-aligned; skip defensively
        yp_all.append(yp); yt_all.append(yt)
        per_shot.append(_per_shot_r2(yp, yt, y_train_mean))
    m = boundary_metrics(np.concatenate(yp_all), np.concatenate(yt_all),
                         y_train_mean=y_train_mean)
    m["per_shot_r2"] = per_shot
    m["n_shots"] = len(per_shot)
    return m
