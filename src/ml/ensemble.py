# src/ml/ensemble.py
# -*- coding: utf-8 -*-
"""Weighted blend of model predictions (non-negative LS weights fit on val)."""
import numpy as np
from scipy.optimize import nnls


def fit_weights(stacked_preds, true):
    """Fit non-negative least-squares blend weights on a validation pool.

    stacked_preds : (n_models, N, D) array of per-model predictions on the same
        N slices, each with D outputs.
    true          : (N, D) array of validation targets.

    Returns weights : (n_models,) non-negative weights summing to 1. If NNLS
        degenerates (all-zero), falls back to a uniform average.
    """
    stacked_preds = np.asarray(stacked_preds, float)
    true = np.asarray(true, float)
    n_models = stacked_preds.shape[0]
    P = stacked_preds.reshape(n_models, -1).T          # (N*D, n_models)
    y = true.reshape(-1)
    w, _ = nnls(P, y)
    if w.sum() <= 0:
        w = np.ones(n_models) / n_models
    return w / w.sum()


def blend(stacked_preds, weights):
    """Weighted sum of model predictions.

    stacked_preds : (n_models, N, D); weights : (n_models,) -> (N, D).
    """
    stacked_preds = np.asarray(stacked_preds, float)
    weights = np.asarray(weights, float)
    return np.einsum("m,mnd->nd", weights, stacked_preds)
