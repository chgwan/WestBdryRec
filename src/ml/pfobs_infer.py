# -*- coding: utf-8 -*-
"""Final-test inference and native per-shot scoring (PF observability, Task 7).

Loads a Task-5 artifact ``<out-root>/<run_name>/m3.pt``, predicts every frozen
test shot on exactly the common-valid rows, and scores the predictions with
the Task-4 native closed-contour metrics against the raw truth polyline
``bnd_RZ``. Two contracts are pinned here and must never drift:

- the inference reader wraps ``PFObsSeriesReader`` in ``_finite_input_reader``
  imported from :mod:`src.ml.pfobs_train` -- the identical wrap the training
  path applies (Task-5 ledgered ruling), so train-time and inference-time arm
  inputs are byte-identical on real data;
- the scored-row gather is ``scripts/train_dcs.py:_pred_m3``'s window
  semantics: disjoint scored blocks at stride ``w - ctx``, each later block
  preceded by an attended-but-unscored context prefix, the first block scored
  from the first native row. The gathered row count is asserted against the
  predict mask ``v = common_valid & target-finite``.

Per-shot aggregation (fixed by the plan): slice-level mean symmetric,
Chamfer, area, centroid, elongation and triangularity values are averaged
over time; ``ccc`` is computed once over all radii in the shot; ``p95_mm``
comes from ALL bidirectional vertex-to-polyline distances pooled over the
shot; shot Hausdorff is the maximum slice Hausdorff. Slice percentiles are
never averaged into a shot percentile.
"""
import pathlib

import numpy as np
import torch

from .axis_frame import reconstruct_absolute
from .dataset import DCSWindowDataset
from .metrics import ccc
from .models import ActSeqAttn
from .native_metrics import native_contour_metrics, point_to_closed_polyline
from .pf_observability import PFObsSeriesReader
from .pfobs_train import _finite_input_reader
from .target import N_OUT, N_RHO, destandardize, load_target, split_outputs, standardize

REQUIRED_PE = "rope_time"
REQUIRED_N_ACT = 21

# Task-4 slice metrics averaged over time into the per-shot row. p95_mm and
# hausdorff_mm are deliberately absent: they pool/max over the whole shot.
SLICE_MEAN_KEYS = ("mean_symmetric_mm", "chamfer_rms_mm", "area_abs_m2",
                   "centroid_mm", "elongation_abs",
                   "triangularity_upper_abs", "triangularity_lower_abs")


def load_artifact(artifact_path):
    """Load one Task-5 artifact on the CPU and enforce the fixed contract.

    ``state`` is saved device-resident, so the load is always
    ``map_location="cpu"``; predicting requires exactly the pinned base
    contract values ``pe == rope_time`` and the 21-column arm input.
    """
    art = torch.load(pathlib.Path(artifact_path), map_location="cpu",
                     weights_only=False)
    missing = [k for k in ("state", "n_act", "hp", "pe", "mean", "std",
                           "tgt_mean", "tgt_std", "n_out", "arm", "seed",
                           "run_fingerprint") if k not in art]
    if missing:
        raise ValueError(f"{artifact_path}: artifact is missing {missing} "
                         "-- not a PF-observability m3 artifact")
    if art["pe"] != REQUIRED_PE:
        raise ValueError(f"{artifact_path}: pe must be {REQUIRED_PE!r}, "
                         f"got {art['pe']!r} -- the four arms are only "
                         "comparable under the fixed base model")
    if int(art["n_act"]) != REQUIRED_N_ACT:
        raise ValueError(f"{artifact_path}: the fixed arm input is "
                         f"{REQUIRED_N_ACT} columns, got {art['n_act']}")
    if int(art["n_out"]) != N_OUT:
        raise ValueError(f"{artifact_path}: n_out must be {N_OUT}, "
                         f"got {art['n_out']}")
    return art


def build_model(art):
    """Reconstruct the fixed ``ActSeqAttn`` from a validated artifact."""
    hp = art["hp"]
    model = ActSeqAttn(n_act=int(art["n_act"]), n_out=int(art["n_out"]),
                       d=hp["d_model"], heads=hp["heads"], depth=hp["depth"],
                       ffn=hp["ffn"], dropout=hp["dropout"], pe=art["pe"])
    model.load_state_dict(art["state"])
    return model.eval()


def artifact_reader(sidecar_dir, art):
    """The arm reader exactly as training used it (finite-wrapped)."""
    return _finite_input_reader(PFObsSeriesReader(sidecar_dir, art["arm"]))


def scored_row_indices(art, sidecar_dir, npz_path):
    """Native-time row indices of exactly the rows :func:`predict_shot` scores.

    ``v = common_valid & target-finite`` -- the predict-mask expression
    ``_pred_m2``/``_pred_m3`` use verbatim, recomputed with the same wrapped
    reader so caller and gatherer can never disagree on the row set.
    """
    _A, mask = artifact_reader(sidecar_dir, art)(npz_path, {}, {})
    _T, finite = load_target(npz_path)
    return np.flatnonzero(mask & finite)


def predict_shot(art, model, npz_dir, sidecar_dir, shot,
                 device=torch.device("cpu")):
    """``(P (n, 34) float32 in metres, n_scored)`` for one shot.

    One :class:`DCSWindowDataset` per shot over the artifact arm reader; every
    window's scored rows are gathered in block order (which is time order) and
    the row count is asserted against the predict mask before anything
    downstream sees the numbers.
    """
    hp = art["hp"]
    ds = DCSWindowDataset(
        pathlib.Path(npz_dir), [int(shot)], cfg={}, ncm={},
        mean=art["mean"], std=art["std"], pe=art["pe"],
        d_model=hp["d_model"], w=hp["window"], ctx=hp["ctx"],
        series_reader=artifact_reader(sidecar_dir, art))
    if not len(ds):
        raise RuntimeError(f"shot {shot}: no window was built -- the shot is "
                           "absent from the target dataset")
    rows = []
    with torch.no_grad():
        for j in range(len(ds)):
            aw, _y, lm, pos = ds[j]
            z = model(aw[None].to(device).float(),
                      pos[None].to(device).float())[0].cpu().numpy()
            rows.append(z[lm.numpy()])
    z = np.concatenate(rows)
    n_scored = int(ds.shots[0][2].sum())
    if z.shape[0] != n_scored:
        raise RuntimeError(
            f"shot {shot}: gathered {z.shape[0]} rows but the predict mask "
            f"has {n_scored} -- the window tiling and the mask disagree")
    P = destandardize(z, art["tgt_mean"], art["tgt_std"]).astype(np.float32)
    return P, n_scored


# ── native per-shot scoring (Task 4 metrics) ─────────────────────────
def uniform_contour(radii, centre, theta):
    """``(n, 32, 2)`` absolute closed contour from uniform-angle radii (n, 32)
    and the absolute polar centre (n, 2), all in metres."""
    centre = np.asarray(centre, float)
    R, Z = reconstruct_absolute(centre[:, 0], centre[:, 1], radii, theta)
    return np.stack([R, Z], axis=-1)


def _pooled_distances(pred_contours, true_contours):
    """``(n, 2*N)`` -- every slice's bidirectional vertex-to-polyline distances
    (pred->truth then truth->pred), the pool the shot ``p95_mm`` comes from.

    The distances are in the input units = METRES (what
    ``point_to_closed_polyline`` returns); the mm conversion happens once, in
    :func:`_aggregate_shot`, exactly as ``native_contour_metrics`` does for
    its slice-level values.
    """
    pred = np.asarray(pred_contours, float)
    true = np.asarray(true_contours, float)
    out = np.empty((len(pred), pred.shape[1] + true.shape[1]))
    for i, (pc, tc) in enumerate(zip(pred, true)):
        out[i, :pred.shape[1]] = point_to_closed_polyline(pc, tc)
        out[i, pred.shape[1]:] = point_to_closed_polyline(tc, pc)
    return out


def _aggregate_shot(metrics, pooled):
    """Shot row from per-slice metrics + the pooled (metre-unit) distance
    matrix: slice means for the averaged keys, the pooled 95th percentile
    converted to mm, and the maximum slice Hausdorff."""
    row = {k: float(np.asarray(metrics[k]).mean()) for k in SLICE_MEAN_KEYS}
    row["p95_mm"] = 1000.0 * float(np.percentile(pooled, 95))
    row["hausdorff_mm"] = float(np.asarray(metrics["hausdorff_mm"]).max())
    return row


def score_shot(P, T, bnd, theta, tgt_mean, tgt_std):
    """The model's per-shot metrics row + the per-slice mean-symmetric array.

    ``P`` / ``T`` are (n, 34) destandardized metres on the scored rows, ``bnd``
    is the native truth polyline (n, N, 2). The predicted absolute contour is
    the 32 uniform-angle radii about the predicted (Rgeom, Zgeom) centre; the
    truth it is compared against is ``bnd`` directly. ``radii_mse`` /
    ``centre_mse`` are target-standardized (the artifact's train statistics).
    """
    P = np.asarray(P, float)
    T = np.asarray(T, float)
    pr, pc = split_outputs(P)
    tr, tc = split_outputs(T)
    pred_c = uniform_contour(pr, pc, theta)
    metrics = native_contour_metrics(pred_c, bnd)
    row = _aggregate_shot(metrics, _pooled_distances(pred_c, bnd))
    row["n_slices"] = int(len(P))
    row["ccc"] = float(ccc(pr, tr))            # once, over all radii in the shot
    zs_p, zs_t = standardize(P, tgt_mean, tgt_std), standardize(T, tgt_mean, tgt_std)
    row["radii_mse"] = float(
        ((zs_p[:, :N_RHO] - zs_t[:, :N_RHO]) ** 2).mean(axis=1).mean())
    row["centre_mse"] = float(
        ((zs_p[:, N_RHO:] - zs_t[:, N_RHO:]) ** 2).mean(axis=1).mean())
    return row, np.asarray(metrics["mean_symmetric_mm"])


def floor_shot(T, bnd, theta):
    """The representation-only floor row for one shot -- no model involved.

    Reconstructs the truth contour from ITS 32 uniform-angle radii and true
    centre and compares that reconstruction against the native ``bnd``: the
    error the 32-angle representation itself costs, before any model error.
    """
    T = np.asarray(T, float)
    tr, tc = split_outputs(T)
    rec = uniform_contour(tr, tc, theta)
    metrics = native_contour_metrics(rec, bnd)
    row = _aggregate_shot(metrics, _pooled_distances(rec, bnd))
    row["n_slices"] = int(len(T))
    return row
