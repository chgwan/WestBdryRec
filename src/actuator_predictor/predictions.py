# src/actuator_predictor/predictions.py
# -*- coding: utf-8 -*-
"""Save/load per-shot per-slice predictions {shot:int -> (n_valid,32) float}."""
import numpy as np


def save_predictions(path, preds):
    np.savez(path, **{str(int(s)): np.asarray(a, np.float32) for s, a in preds.items()})


def load_predictions(path):
    d = np.load(path)
    return {int(k): np.asarray(d[k], float) for k in d.files}
