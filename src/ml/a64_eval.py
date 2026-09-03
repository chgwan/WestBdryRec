# -*- coding: utf-8 -*-
"""Dense-angle (a64) exploratory evaluation: floors, per-shot metrics, and
the paired bootstrap against the frozen 32-angle seed-0 baseline.

Task 5 of the dense-angle a64 plan.  This module is the importable, tested
core; ``exploration/work3_a64_eval.py`` (untracked by design) is a thin CLI
wrapper around :func:`main`.

Two frozen neighbours are read but never imported or modified:
``src/ml/pfobs_infer.py`` (the native scorer; pins ``n_out=34``) supplies the
metric construction, reimplemented here as :func:`reconstruct_and_floor`
(verified bit-exact upstream), and ``src/ml/pfctx_infer.py`` supplies the
scored-row gather semantics, copied in simplified form
(:func:`initialize_shot_gathers` / :func:`append_scored_block` /
:func:`finalize_shot_gathers`) so predictions land on exactly the dataset's
stored :class:`~src.ml.pf_context.ScoredWindow` scored rows.

Frozen references (read-only):
- baseline ``ProjDB/Stats/pf_context/final_test/h0512/s0/per_shot_metrics.csv``
- 32-theta floor table
  ``ProjDB/Stats/pf_observability/final_test/
  representation_floor_per_shot.csv``
  (rows ``common_valid & target-finite``, per-shot value = mean over slices).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import pathlib
import time

import numpy as np
import torch

from .models import ActSeqAttn
from .native_metrics import point_to_closed_polyline
from .pf_context import context_level, layer_attention_mask
from .pfctx_data import (
    PFContextDataset, load_context_series, pad_context_collate,
)
from .target import n_out_for, uniform_theta

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

DEFAULT_SPLIT = str(REPO_ROOT / "configs/splits/"
                               "communications_physics_campaign_v1.json")
DEFAULT_TARGET_DIR = str(REPO_ROOT / "ProjDB/datasets/NpzGeom")
DEFAULT_SIDECAR_DIR = str(REPO_ROOT / "ProjDB/datasets/NpzGeomPFObs")
DEFAULT_BASELINE = str(REPO_ROOT / "ProjDB/Stats/pf_context/final_test/"
                                   "h0512/s0/per_shot_metrics.csv")
DEFAULT_FLOOR_CSV = str(REPO_ROOT / "ProjDB/Stats/pf_observability/"
                                    "final_test/"
                                    "representation_floor_per_shot.csv")
DEFAULT_OUT_MD = "exploration/_out/work3_a64_results.md"
DEFAULT_OUT_PNG = "figs/pf_context/work3_a64_vs_baseline.png"

# The brief's frozen reference value for the 32-theta floor median (mm); the
# recomputed median is asserted against the frozen table, not this literal.
FROZEN_32THETA_FLOOR_MEDIAN_MM = 2.156773
# Controller ruling 3: per-shot self-check tolerance against the frozen floor
# table (per-shot value convention = mean over slices, then the CSV column).
FLOOR_SELFCHECK_TOL_MM = 1e-3

# Seed-light claim limit (docs/works2_3_progress.md), quoted verbatim in every
# results document of this study.
CLAIM_LIMIT_QUOTE = (
    "Future v2 results quantify operating-shot variability for one fixed, "
    "hash-addressed training run per experimental arm. They do not estimate "
    "neural-training seed variance.")

REQUIRED_A64_KEYS = (
    "state", "n_act", "n_out", "n_rho", "depth", "hp", "feature_mean",
    "feature_std", "target_mean", "target_std", "context_label", "seed",
    "exploratory")


# ── metric construction (frozen scorer's, copied) ────────────────────
def _uniform_contour(radii, centre, theta):
    """``(n, K, 2)`` absolute closed contour from uniform-angle radii
    ``(n, K)`` and the absolute polar centre ``(n, 2)``, in metres.

    Generic copy of ``pfobs_infer.uniform_contour``'s construction (that
    module is frozen and pins ``n_out=34``); the axis_frame broadcast is
    inlined so nothing downstream depends on a 32-column width.
    """
    radii = np.asarray(radii, float)
    centre = np.asarray(centre, float)
    theta = np.asarray(theta, float).reshape(-1)
    R = centre[:, 0:1] + radii * np.cos(theta)
    Z = centre[:, 1:2] + radii * np.sin(theta)
    return np.stack([R, Z], axis=-1)


def reconstruct_and_floor(bnd, radii, center, theta):
    """Per-slice mean-symmetric mm between the theta-grid reconstruction
    (radii + center -> contour) and the native polyline ``bnd``.

    Identical to ``pfobs_infer.score_shot``'s per-slice construction
    (``500 * (d_pt.mean() + d_tp.mean())`` over
    :func:`~src.ml.native_metrics.point_to_closed_polyline`).
    """
    pred = _uniform_contour(radii, center, theta)
    bnd = np.asarray(bnd, float)
    out = np.empty(len(pred))
    for i, (p, t) in enumerate(zip(pred, bnd)):
        out[i] = 500.0 * (point_to_closed_polyline(p, t).mean()
                          + point_to_closed_polyline(t, p).mean())
    return out


def paired_bootstrap(diffs, resamples=10_000, seed=20_260_820):
    """Percentile bootstrap of the median -- the established convention.

    Returns ``(median, 2.5th, 97.5th)`` of the paired differences; shots are
    resampled with replacement (PCG64 with the frozen selection seed)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), (resamples, len(diffs)))
    draws = np.median(np.asarray(diffs)[idx], axis=1)
    return (float(np.median(diffs)),
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)))


# ── artifact handling ────────────────────────────────────────────────
def load_artifact(artifact_path):
    """Load one a64 exploratory artifact on the CPU and check its contract."""
    art = torch.load(pathlib.Path(artifact_path), map_location="cpu",
                     weights_only=False)
    missing = [key for key in REQUIRED_A64_KEYS if key not in art]
    if missing:
        raise ValueError(f"{artifact_path}: artifact is missing {missing} "
                         "-- not an a64 exploratory m3 artifact")
    if not art["exploratory"]:
        raise ValueError(f"{artifact_path}: artifact is not exploratory "
                         "-- this evaluator never touches frozen runs")
    if int(art["n_out"]) != n_out_for(int(art["n_rho"])):
        raise ValueError(
            f"{artifact_path}: n_out {art['n_out']} disagrees with "
            f"n_rho {art['n_rho']} (expected {n_out_for(int(art['n_rho']))})")
    if int(art["depth"]) <= 0:
        raise ValueError(f"{artifact_path}: depth must be positive")
    context_level(art["context_label"])          # unknown label -> ValueError
    return art


def build_model(art):
    """Rebuild the exploratory ``ActSeqAttn`` from a validated artifact.

    ``pe`` is not recorded by the a64 driver, which pins ``rope_time`` (the
    frozen base contract); the strict state-dict load makes any shape drift
    loud.
    """
    hp = art["hp"]
    model = ActSeqAttn(n_act=int(art["n_act"]), n_out=int(art["n_out"]),
                       d=hp["d_model"], heads=hp["heads"],
                       depth=int(art["depth"]), ffn=hp["ffn"],
                       dropout=hp["dropout"], pe="rope_time")
    model.load_state_dict(art["state"])
    return model.eval()


# ── the per-shot gather: simplified copies of pfctx_infer's ──────────
def _new_gather(series):
    return {"expected": np.flatnonzero(series.score_valid),
            "time": np.asarray(series.time, np.float64),
            "row_index": [], "prediction": []}


def initialize_shot_gathers(dataset, shots):
    """One gather per requested shot, seeded with the shot's expected
    common-valid native rows (the contract finalize asserts against)."""
    present = {int(shot) for shot, _series in dataset.series}
    absent = sorted({int(s) for s in shots} - present)
    if absent:
        raise RuntimeError(
            f"shots {absent} produced no scored window -- absent from the "
            "target dataset or empty after the join")
    return {int(shot): _new_gather(series)
            for shot, series in dataset.series}


def append_scored_block(gathered, dataset, item_index, pred):
    """Append one window's scored block to its shot's gather.

    The scored rows come from the dataset's stored ``ScoredWindow`` and the
    item's exact loss mask (``score_valid`` restricted to the scored block),
    never from a reconstructed stride; ``pred`` is the model's standardized
    output for the whole window and is sliced by the same mask.
    """
    series_index, window = dataset.index[item_index]
    shot, series = dataset.series[series_index]
    ws, we = window.window_start, window.window_end
    pred = np.asarray(pred)
    if pred.shape[0] != we - ws:
        raise RuntimeError(
            f"shot {shot}: prediction block has {pred.shape[0]} rows but the "
            f"stored window is {we - ws} rows long")
    loss = series.score_valid[ws:we].copy()
    loss[:window.block_start - ws] = False
    local = np.flatnonzero(loss)
    rows = (ws + local).astype(np.int64)
    if local.size and (rows[0] < window.block_start
                       or rows[-1] >= window.block_end):
        raise RuntimeError(
            f"shot {shot}: scored rows fall outside the stored scored block "
            f"[{window.block_start}, {window.block_end})")
    gathered[int(shot)]["row_index"].append(rows)
    gathered[int(shot)]["prediction"].append(pred[local])


def finalize_shot_gathers(gathered, target_mean, target_std):
    """Destandardize and assert the per-shot row-index contract.

    Every shot's gathered rows must be strictly increasing, unique, and
    exactly the expected common-valid indices; any overlap, duplicate, or
    disagreement with the score mask is a loud failure.
    """
    out = {}
    for shot, gather in gathered.items():
        if not gather["row_index"]:
            raise RuntimeError(f"shot {shot}: no scored rows were gathered")
        rows = np.concatenate(gather["row_index"]).astype(np.int64)
        standardized = np.concatenate(gather["prediction"], axis=0)
        if rows.size != standardized.shape[0]:
            raise RuntimeError(
                f"shot {shot}: {rows.size} row indices vs "
                f"{standardized.shape[0]} predictions")
        if not np.all(np.diff(rows) > 0):
            raise RuntimeError(
                f"shot {shot}: gathered row indices must be strictly "
                "increasing and unique -- scored blocks overlapped or "
                "duplicated rows")
        if not np.array_equal(rows, gather["expected"]):
            missing = np.setdiff1d(gather["expected"], rows)
            raise RuntimeError(
                f"shot {shot}: gathered {rows.size} rows but the "
                f"common-valid rows are {gather['expected'].size} (missing "
                f"{missing.size}) -- the gather and the score mask disagree")
        prediction = (standardized.astype(np.float64)
                      * np.asarray(target_std, np.float64)
                      + np.asarray(target_mean, np.float64)
                      ).astype(np.float32)
        out[int(shot)] = {
            "row_index": rows,
            "timestamp": gather["time"][rows].astype(np.float64),
            "prediction": prediction}
    return out


@dataclasses.dataclass(frozen=True)
class ShotPrediction:
    """One shot's destandardized predictions and per-slice error on exactly
    the scored rows."""
    shot: int
    row_index: np.ndarray        # int64 native rows, strictly increasing
    timestamp: np.ndarray        # float64 native time at those rows
    prediction: np.ndarray       # float32 (n, n_out) absolute metres
    per_slice_mm: np.ndarray     # float64 (n,) mean-symmetric mm vs bnd_RZ


def _truth_polyline(target_dir, shot, row_index):
    """The native truth polyline ``bnd_RZ`` at exactly the predicted rows."""
    with np.load(pathlib.Path(target_dir) / f"{int(shot)}.npz") as data:
        return np.asarray(data["bnd_RZ"], np.float64)[row_index]


def predict_shot(artifact, dataset, model, theta, target_dir, device=None):
    """Predict every dataset shot in one window pass and score per slice.

    Mirrors ``pfctx_infer.predict_artifact``: each stored window is collated
    with :func:`~src.ml.pfctx_data.pad_context_collate`, attended through
    :func:`~src.ml.pf_context.layer_attention_mask` at the artifact's
    ``depth``, and its scored block is gathered by the stored
    ``ScoredWindow``; the result is destandardized with the artifact's
    target statistics.  ``per_slice_mm`` is
    :func:`reconstruct_and_floor` of the prediction against the native
    ``bnd_RZ`` -- the model's mean-symmetric metric, identical to
    ``pfobs_infer.score_shot``'s construction.  Returns
    ``{shot: ShotPrediction}``.
    """
    if device is None:
        device = torch.device("cpu")
    context = context_level(artifact["context_label"])
    shots = [shot for shot, _series in dataset.series]
    gathered = initialize_shot_gathers(dataset, shots)
    with torch.no_grad():
        for item_index in range(len(dataset)):
            batch = pad_context_collate([dataset[item_index]])
            attention = layer_attention_mask(
                batch.time.to(device), batch.history_valid.to(device),
                batch.real.to(device), context.seconds,
                int(artifact["depth"]))
            pred = model(batch.features.to(device), batch.position.to(device),
                         attn_mask=attention)[0].cpu().numpy()
            append_scored_block(gathered, dataset, item_index, pred)
    blocks = finalize_shot_gathers(
        gathered, artifact["target_mean"], artifact["target_std"])
    out = {}
    for shot, block in blocks.items():
        n_rho = block["prediction"].shape[1] - 2
        bnd = _truth_polyline(target_dir, shot, block["row_index"])
        per_slice = reconstruct_and_floor(
            bnd, block["prediction"][:, :n_rho],
            block["prediction"][:, n_rho:], theta)
        out[int(shot)] = ShotPrediction(
            int(shot), block["row_index"], block["timestamp"],
            block["prediction"], per_slice)
    return out


# ── representation floors ────────────────────────────────────────────
def floor_per_shot(target_path, sidecar_dir, n_rho):
    """``(per_shot_mean_mm, n_rows)``: the model-free representation floor.

    Reconstructs the truth contour from ITS OWN ``n_rho``-angle radii about
    the true centre (the stored ``Y`` when ``n_rho`` matches the stored
    width, ray-cast derived from ``bnd_RZ`` otherwise, exactly as the
    training data layer resolves it) and scores that reconstruction against
    the native ``bnd_RZ`` on exactly the score-valid rows for that
    ``n_rho``.  Per-shot value = mean over slices, the frozen floor table's
    own convention.
    """
    series = load_context_series(target_path, sidecar_dir, n_rho=int(n_rho))
    rows = np.flatnonzero(series.score_valid)
    if not rows.size:
        raise RuntimeError(f"{target_path.stem}: no score-valid rows")
    n = series.target.shape[1] - 2
    with np.load(pathlib.Path(target_path)) as data:
        bnd = np.asarray(data["bnd_RZ"], np.float64)[rows]
    target = series.target[rows].astype(np.float64)
    per_slice = reconstruct_and_floor(bnd, target[:, :n], target[:, n:],
                                      uniform_theta(n))
    return float(per_slice.mean()), int(rows.size)


def read_floor_csv(path):
    """``{shot: (mean_symmetric_mm, n_slices)}`` from a frozen floor table."""
    with open(path, newline="") as stream:
        return {int(row["shot"]): (float(row["mean_symmetric_mm"]),
                                   int(row["n_slices"]))
                for row in csv.DictReader(stream)}


def read_baseline_csv(path):
    """``{shot: mean_symmetric_mm}`` from a frozen per-shot metrics table."""
    with open(path, newline="") as stream:
        return {int(row["shot"]): float(row["mean_symmetric_mm"])
                for row in csv.DictReader(stream)}


def floor_self_check(shots, target_dir, sidecar_dir, floor_csv):
    """Recompute the 32-theta floor per shot and compare to the frozen table.

    Returns ``(floor32 {shot: mean_mm}, n_rows {shot: int}, max_dev_mm,
    frozen_median_mm)``; raises when a shot is missing from the frozen
    table, when the scored-row counts disagree, or when the max per-shot
    deviation reaches :data:`FLOOR_SELFCHECK_TOL_MM` (convention under
    suspicion: per-shot value = mean over slices, then the CSV's own
    per-shot column, over ``common_valid & target-finite`` rows).
    """
    frozen = read_floor_csv(floor_csv)
    floor32, n_rows = {}, {}
    max_dev = 0.0
    for shot in shots:
        if shot not in frozen:
            raise RuntimeError(
                f"shot {shot} is absent from the frozen floor table "
                f"{floor_csv} -- the self-check cannot be paired")
        value, rows = floor_per_shot(
            pathlib.Path(target_dir) / f"{shot}.npz", sidecar_dir, 32)
        floor32[shot], n_rows[shot] = value, rows
        frozen_value, frozen_rows = frozen[shot]
        if rows != frozen_rows:
            raise RuntimeError(
                f"shot {shot}: recomputed floor used {rows} scored rows but "
                f"the frozen table has {frozen_rows} -- the row convention "
                "disagrees before any value comparison")
        max_dev = max(max_dev, abs(value - frozen_value))
    if max_dev >= FLOOR_SELFCHECK_TOL_MM:
        raise AssertionError(
            f"32-theta floor self-check: max per-shot deviation {max_dev:.3e}"
            f" mm >= {FLOOR_SELFCHECK_TOL_MM:.0e} mm against {floor_csv} -- "
            "investigate the aggregation convention (per-shot value = mean "
            "over slices, then the CSV's own per-shot column) before "
            "loosening anything")
    return floor32, n_rows, max_dev, float(
        np.median([frozen[shot][0] for shot in shots]))


# ── outputs (main() only; tests never write here) ────────────────────
def write_results_md(path, *, artifact_path, art, shots, n_pred_rows,
                     a64_mm, baseline_mm, floor32, floor32_rows, max_dev,
                     frozen_floor_median, floor64, floor64_rows, paired):
    """Write the a64 results markdown (tables + the binding claim limits)."""
    lines = []
    add = lines.append
    add("# a64 exploratory evaluation -- dense-angle (64 theta) vs frozen "
        "32-theta baseline\n")
    add(f"run: `{artifact_path}`\n")
    add(f"- study `{art['study']}` (exploratory), context "
        f"`{art['context_label']}`, seed {int(art['seed'])}, "
        f"n_rho {int(art['n_rho'])} (n_out {int(art['n_out'])}), "
        f"depth {int(art['depth'])}")
    if "best_val_mse" in art:
        add(f"- epochs_completed {int(art.get('epochs_completed', -1))}, "
            f"stop_epoch {int(art.get('stop_epoch', -1))}, best_val_mse "
            f"{float(art['best_val_mse']):.6f}")
    add(f"- config_sha256 `{art.get('config_sha256', 'n/a')}`, "
        f"split_sha256 `{art.get('split_sha256', 'n/a')}`")
    add(f"- scored {len(shots)} frozen test shots, {n_pred_rows} predicted "
        f"rows, at {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
        " (device: cpu)\n")

    add("## Representation floors (model-free; reported, never subtracted)\n")
    add("| representation | median per-shot floor (mm) | mean per-shot floor"
        " (mm) | median scored rows |")
    add("|---|---:|---:|---:|")
    add(f"| 32 theta (frozen table) | {frozen_floor_median:.6f} | "
        f"{np.mean([floor32[s] for s in shots]):.6f} | "
        f"{int(np.median([floor32_rows[s] for s in shots]))} |")
    add(f"| 32 theta (recomputed self-check) | "
        f"{np.median([floor32[s] for s in shots]):.6f} | "
        f"{np.mean([floor32[s] for s in shots]):.6f} | "
        f"{int(np.median([floor32_rows[s] for s in shots]))} |")
    add(f"| 64 theta (derived, this run) | "
        f"{np.median([floor64[s] for s in shots]):.6f} | "
        f"{np.mean([floor64[s] for s in shots]):.6f} | "
        f"{int(np.median([floor64_rows[s] for s in shots]))} |")
    add("")
    add(f"32-theta self-check vs `{DEFAULT_FLOOR_CSV}`: max per-shot "
        f"deviation {max_dev:.3e} mm over {len(shots)} shots "
        f"(tolerance {FLOOR_SELFCHECK_TOL_MM:.0e} mm, row counts identical); "
        f"brief reference median {FROZEN_32THETA_FLOOR_MEDIAN_MM:.6f} mm.\n")

    add("## Model per-shot error (mean symmetric, mm)\n")
    add("| run | median | mean | min | max |")
    add("|---|---:|---:|---:|---:|")
    for label, table in (("a64 exploratory (64 theta)", a64_mm),
                         ("frozen h0512 s0 (32 theta)", baseline_mm)):
        values = np.array([table[s] for s in shots])
        add(f"| {label} | {np.median(values):.4f} | {values.mean():.4f} | "
            f"{values.min():.4f} | {values.max():.4f} |")
    add("")

    med, lo, hi = paired
    diffs = np.array([a64_mm[s] - baseline_mm[s] for s in shots])
    add("## Paired comparison (a64 64 theta minus frozen 32 theta, per shot)"
        "\n")
    add(f"- median difference **{med:+.4f} mm**, 95% percentile-bootstrap CI "
        f"[{lo:+.4f}, {hi:+.4f}] ({10_000:,} resamples, seed 20_260_820)")
    add(f"- {len(shots)} paired shots: {int((diffs < 0).sum())} improved, "
        f"{int((diffs > 0).sum())} worsened, "
        f"{int((diffs == 0).sum())} unchanged")
    add(f"- floor change 32 -> 64 theta: "
        f"{np.median([floor64[s] for s in shots]) - frozen_floor_median:+.4f}"
        " mm (median) -- mechanical, never subtracted from the effect\n")

    add("## Claim limits (binding)\n")
    add("> " + CLAIM_LIMIT_QUOTE + "\n")
    add(f"This a64 run is exactly one such single fixed, hash-addressed "
        "training run (config/split sha256 above; context "
        f"`{art['context_label']}`, seed {int(art['seed'])}, "
        f"n_rho {int(art['n_rho'])}, exploratory): it quantifies "
        "operating-shot variability only, does not estimate neural-training "
        "seed variance, and supports no confirmatory claim. Denser targets "
        "are a linear-on-polyline resampling of the same ~32 native samples; "
        "part of any improvement is representational and mechanical, the "
        "dense floor is reported beside model error and never subtracted, "
        "and the frozen v1 verdicts, selections, markers, and artifacts "
        "remain the only publication evidence.\n")

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def write_figure(path, *, shots, baseline_mm, a64_mm, floor32_median,
                 floor64_median):
    """Scatter: x = frozen seed-0 32-theta per-shot error, y = a64 64-theta
    per-shot error, with the diagonal and both representation-floor lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink_2, grid, series = "#0b0b0b", "#52514e", "#e7e6e2", "#2a78d6"
    x = np.array([baseline_mm[s] for s in shots])
    y = np.array([a64_mm[s] for s in shots])
    lo = float(min(x.min(), y.min(),
                   floor32_median, floor64_median))
    hi = float(max(x.max(), y.max(),
                   floor32_median, floor64_median))
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    ax.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot([lo, hi], [lo, hi], linestyle=(0, (4, 3)), linewidth=1.5,
            color=ink_2, label="y = x", zorder=2)
    ax.axvline(floor32_median, color=ink_2, linewidth=1.25,
               linestyle=(0, (1, 2)), zorder=1,
               label=f"32-theta floor (median {floor32_median:.3f} mm)")
    ax.axhline(floor64_median, color="#8a887f", linewidth=1.25,
               linestyle=(0, (5, 2, 1, 2)), zorder=1,
               label=f"64-theta floor (median {floor64_median:.3f} mm)")
    ax.scatter(x, y, s=28, color=series, alpha=0.75, linewidths=0.8,
               edgecolors="white", zorder=3,
               label=f"per-shot error ({len(shots)} shots)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("frozen h0512 seed 0, 32 theta (mm)", color=ink)
    ax.set_ylabel("a64 exploratory h0512 seed 0, 64 theta (mm)", color=ink)
    ax.set_title("Per-shot mean symmetric error: a64 64-theta "
                 "vs frozen 32-theta", color=ink)
    for spine in ax.spines.values():
        spine.set_color(grid)
    ax.tick_params(colors=ink_2)
    ax.legend(loc="upper left", frameon=False, labelcolor=ink_2)
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None):
    """CLI entry: evaluate one a64 artifact end to end and write md + png."""
    ap = argparse.ArgumentParser(
        description="a64 exploratory evaluation: 64-theta floor, per-shot "
                    "metrics, paired bootstrap vs the frozen seed-0 baseline")
    ap.add_argument("--artifact", required=True,
                    help="path to the a64 exploratory m3.pt artifact")
    ap.add_argument("--out-md", default=DEFAULT_OUT_MD)
    ap.add_argument("--out-png", default=DEFAULT_OUT_PNG)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--target-dir", default=DEFAULT_TARGET_DIR)
    ap.add_argument("--sidecar-dir", default=DEFAULT_SIDECAR_DIR)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help="frozen per-shot baseline CSV (read-only)")
    ap.add_argument("--floor-csv", default=DEFAULT_FLOOR_CSV,
                    help="frozen 32-theta floor table (read-only)")
    args = ap.parse_args(argv)

    def rooted(path):
        p = pathlib.Path(path)
        return p if p.is_absolute() else REPO_ROOT / p

    # Task 4's split reader (lazy: pulls the frozen training stack).
    from scripts.run_a64_exploratory import load_split

    art = load_artifact(args.artifact)
    n_rho = int(art["n_rho"])
    theta = uniform_theta(n_rho)
    target_dir = pathlib.Path(args.target_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    split = load_split(args.split, target_dir, sidecar_dir)
    shots = sorted(int(s) for s in split.test)
    if not shots:
        raise SystemExit("no test shots in the split -- nothing to evaluate")
    print(f"[a64] {len(shots)} test shots, context {art['context_label']}, "
          f"n_rho {n_rho}, seed {int(art['seed'])}")

    t0 = time.perf_counter()
    dataset = PFContextDataset(
        target_dir, sidecar_dir, shots, context_level(art["context_label"]),
        art["feature_mean"], art["feature_std"],
        art["target_mean"], art["target_std"], n_rho=n_rho)
    print(f"[a64] dataset construction {time.perf_counter() - t0:.1f}s "
          f"({len(dataset)} scored windows)")
    t1 = time.perf_counter()
    predictions = predict_shot(art, dataset, build_model(art), theta,
                               target_dir)
    a64_mm = {shot: float(p.per_slice_mm.mean())
              for shot, p in predictions.items()}
    print(f"[a64] prediction + per-slice scoring "
          f"{time.perf_counter() - t1:.1f}s "
          f"({sum(len(p.row_index) for p in predictions.values())} rows)")

    t2 = time.perf_counter()
    floor32, floor32_rows, max_dev, frozen_floor_median = floor_self_check(
        shots, target_dir, sidecar_dir, args.floor_csv)
    print(f"[a64] 32-theta floor self-check: max per-shot deviation "
          f"{max_dev:.3e} mm ({time.perf_counter() - t2:.1f}s)")
    floor64, floor64_rows = {}, {}
    for shot in shots:
        floor64[shot], floor64_rows[shot] = floor_per_shot(
            target_dir / f"{shot}.npz", sidecar_dir, n_rho)

    baseline = read_baseline_csv(args.baseline)
    missing = [s for s in shots if s not in baseline]
    if missing:
        raise RuntimeError(
            f"frozen baseline {args.baseline} is missing shots {missing} "
            "-- the paired comparison cannot be formed")
    diffs = np.array([a64_mm[s] - baseline[s] for s in shots])
    paired = paired_bootstrap(diffs)
    print(f"[a64] paired median {paired[0]:+.4f} mm "
          f"(95% CI [{paired[1]:+.4f}, {paired[2]:+.4f}])")

    md_path = write_results_md(
        rooted(args.out_md), artifact_path=args.artifact, art=art,
        shots=shots,
        n_pred_rows=sum(len(p.row_index) for p in predictions.values()),
        a64_mm=a64_mm, baseline_mm=baseline, floor32=floor32,
        floor32_rows=floor32_rows, max_dev=max_dev,
        frozen_floor_median=frozen_floor_median, floor64=floor64,
        floor64_rows=floor64_rows, paired=paired)
    png_path = write_figure(
        rooted(args.out_png), shots=shots, baseline_mm=baseline,
        a64_mm=a64_mm,
        floor32_median=float(np.median([floor32[s] for s in shots])),
        floor64_median=float(np.median([floor64[s] for s in shots])))
    print(f"[a64] wrote {md_path}")
    print(f"[a64] wrote {png_path}")
    return 0
