# -*- coding: utf-8 -*-
"""Arm B native-time context series and fixed scored-block batches.

The PF-context sweep joins the Task-1 sidecar (NpzGeomPFObs: ref/actual PF
plus Ip on each shot's own native axis) to the 34-column native target
row-for-row.  The join is an equality check, never a resample: a sidecar row
may pair only with the target row that shares its exact time value, so a
one-bit axis mismatch is a loud failure.  Arm B history validity depends only
on the measured signals (actual + Ip); reference dropouts may remove a row
from scoring (common_valid) but never from Arm B history.  Missingness is
recorded before any non-finite value is zero-filled, so the masks always
describe the source data.
"""
from __future__ import annotations

import dataclasses
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset

from .pf_context import SCORE_BLOCK, scored_windows
from .pf_observability import assemble_arm
from .pos_encoding import modal_cadence
from .target import load_target


@dataclasses.dataclass(frozen=True)
class ContextSeries:
    features: np.ndarray
    target: np.ndarray
    score_valid: np.ndarray
    history_valid: np.ndarray
    time: np.ndarray


def load_context_series(target_path, sidecar_dir) -> ContextSeries:
    target_path = pathlib.Path(target_path)
    sidecar_path = pathlib.Path(sidecar_dir) / target_path.name
    with np.load(sidecar_path) as d:
        ref = np.asarray(d["pf_ref"], np.float32)
        actual = np.asarray(d["pf_actual"], np.float32)
        ip_ref = np.asarray(d["ip_ref"], np.float32).reshape(-1, 1)
        common_valid = np.asarray(d["common_valid"], bool)
        side_time = np.asarray(d["time"], np.float64)
    with np.load(target_path) as d:
        target_time = np.asarray(d["time"], np.float64)
    if not np.array_equal(side_time, target_time):
        raise ValueError(
            f"{target_path.stem}: sidecar/target native time mismatch")
    target, target_finite = load_target(target_path)
    history_valid = np.isfinite(actual).all(1) & np.isfinite(ip_ref).all(1)
    score_valid = common_valid & target_finite
    features = assemble_arm(ref, actual, ip_ref, "B")
    features = np.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    target = np.nan_to_num(
        target, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n = target_time.size
    if any(x.shape[0] != n for x in (
            features, target, score_valid, history_valid)):
        raise ValueError(f"{target_path.stem}: native row-count mismatch")
    return ContextSeries(
        features, target, score_valid, history_valid, target_time)


@dataclasses.dataclass(frozen=True)
class ContextItem:
    features: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    position: torch.Tensor
    time: torch.Tensor
    history_valid: torch.Tensor


@dataclasses.dataclass(frozen=True)
class ContextBatch:
    features: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    position: torch.Tensor
    time: torch.Tensor
    history_valid: torch.Tensor
    real: torch.Tensor


class PFContextDataset(Dataset):
    def __init__(self, target_dir, sidecar_dir, shots, context,
                 feature_mean, feature_std, target_mean, target_std,
                 score_block=SCORE_BLOCK):
        self.context = context
        self.feature_mean = np.asarray(feature_mean, np.float32)
        self.feature_std = np.maximum(
            np.asarray(feature_std, np.float32), 1e-6)
        self.target_mean = np.asarray(target_mean, np.float32)
        self.target_std = np.maximum(
            np.asarray(target_std, np.float32), 1e-6)
        self.series = []
        self.cadences = []
        self.index = []
        for shot in map(int, shots):
            path = pathlib.Path(target_dir) / f"{shot}.npz"
            s = load_context_series(path, sidecar_dir)
            series_index = len(self.series)
            self.series.append((shot, s))
            # the modal cadence is a pure function of s.time: compute it once
            # per series here instead of the np.unique over the whole shot
            # axis that every __getitem__ window item would otherwise repeat
            self.cadences.append(modal_cadence(s.time))
            self.index.extend(
                (series_index, w) for w in
                scored_windows(s.time, context, score_block))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item_index):
        series_index, window = self.index[item_index]
        _shot, s = self.series[series_index]
        ws, we = window.window_start, window.window_end
        a = ((s.features[ws:we] - self.feature_mean) /
             self.feature_std).astype(np.float32)
        y = ((s.target[ws:we] - self.target_mean) /
             self.target_std).astype(np.float32)
        loss = s.score_valid[ws:we].copy()
        loss[:window.block_start - ws] = False
        cadence = self.cadences[series_index]
        position = ((s.time[ws:we] - s.time[ws]) /
                    cadence).astype(np.float32)
        local_time = (s.time[ws:we] - s.time[ws]).astype(np.float64)
        return ContextItem(
            torch.from_numpy(a), torch.from_numpy(y),
            torch.from_numpy(loss), torch.from_numpy(position),
            torch.from_numpy(local_time),
            torch.from_numpy(s.history_valid[ws:we]))


def pad_context_collate(items):
    batch, width = len(items), max(x.features.shape[0] for x in items)
    n_act, n_out = items[0].features.shape[1], items[0].target.shape[1]
    features = torch.zeros(batch, width, n_act, dtype=torch.float32)
    target = torch.zeros(batch, width, n_out, dtype=torch.float32)
    loss = torch.zeros(batch, width, dtype=torch.bool)
    position = torch.zeros(batch, width, dtype=torch.float32)
    time = torch.zeros(batch, width, dtype=torch.float64)
    history = torch.zeros(batch, width, dtype=torch.bool)
    real = torch.zeros(batch, width, dtype=torch.bool)
    for i, x in enumerate(items):
        n = x.features.shape[0]
        features[i, :n] = x.features
        target[i, :n] = x.target
        loss[i, :n] = x.loss_mask
        position[i, :n] = x.position
        time[i, :n] = x.time
        history[i, :n] = x.history_valid
        real[i, :n] = True
    return ContextBatch(
        features, target, loss, position, time, history, real)
