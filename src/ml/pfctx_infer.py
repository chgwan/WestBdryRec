# -*- coding: utf-8 -*-
"""Exact-context final-test inference and native per-shot scoring (Task 6).

Loads a pf-context artifact ``<out-root>/<run_name>/m3.pt`` and predicts every
requested shot on exactly the common-valid rows, reusing the training path
verbatim: :class:`PFContextDataset` + :func:`pad_context_collate` with the
artifact's own normalization arrays and ``score_block``, and
:func:`layer_attention_mask` built from the artifact's recorded ``depth``.
The scored-row gather uses the dataset's stored :class:`ScoredWindow` and loss
mask -- never a reconstructed stride -- and
:func:`finalize_shot_gathers` asserts, per shot, that the gathered native row
indices are strictly increasing, unique, and exactly the shot's common-valid
indices.

Native per-shot scoring reuses the upstream Essential-Work-2 final-scorer
helpers (:func:`src.ml.pfobs_infer.score_shot` over
:func:`src.ml.native_metrics.native_contour_metrics`) so both studies share
one metric implementation. The 32-angle representation floor is NEVER
recomputed or subtracted here: the scorer records the upstream floor table's
provenance and passes it through untouched.

Cell outputs (the final-scorer staging layout) per ``(context, seed)``:
``m3_pred.npz`` (compressed predictions, exact native row indices, float64
timestamps), ``per_shot_metrics.csv``, ``run_metadata.json``.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import io
import json
import os
import pathlib
import stat
import time

import numpy as np
import torch

from .models import ActSeqAttn
from .pf_context import context_level, layer_attention_mask
from .pfctx_data import PFContextDataset, pad_context_collate
from .pfctx_train import INPUT_WIDTH, N_OUT, REQUIRED_ARTIFACT_KEYS
from .pfobs_infer import score_shot
from .target import load_target

# The fixed base contract every scored artifact must carry. The frozen sweep
# config pins d_model/heads/ffn/dropout (validate_context_config); the
# artifact itself records depth/n_act/n_out/pe, so the values below only have
# to agree with the state dict -- and the strict load makes any drift loud.
CONTRACT_TOKENS = {
    "study": "pf_context", "arm": "B", "model": "ActSeqAttn",
    "pe": "rope_time", "time_axis": "native_gmag_bnd",
}
FROZEN_HP = {"d": 256, "heads": 8, "ffn": 1024, "dropout": 0.1}

PER_SHOT_HEADER = [
    "context", "seed", "shot", "n_slices", "mean_symmetric_mm", "p95_mm",
    "chamfer_rms_mm", "hausdorff_mm", "area_abs_m2", "centroid_mm",
    "elongation_abs", "triangularity_upper_abs", "triangularity_lower_abs",
    "ccc", "radii_mse", "centre_mse",
]
RUN_METADATA_FIELDS = {
    "study", "run_name", "context_label", "nominal_samples",
    "context_seconds", "per_layer_seconds", "score_block", "seed", "depth",
    "world_size", "effective_global_batch", "best_val_mse", "stop_epoch",
    "epochs_completed", "device", "scored_at", "n_shots", "shots",
    "n_pred_rows", "row_order_contract", "fingerprints", "floor",
    "transaction_provenance", "transaction_sha256", "artifact_sha256",
    "fingerprint_json_sha256", "validation_selection_sha256",
    "availability_audit_sha256", "work2_marker_sha256", "floor_sha256",
}


@dataclasses.dataclass(frozen=True)
class ScoredArtifactPayload:
    rows: tuple[dict[str, object], ...]
    prediction_bytes: bytes
    metrics_bytes: bytes
    metadata: dict[str, object]
    metadata_bytes: bytes


def validate_artifact_contract(artifact):
    """Every required key present and the fixed base contract satisfied.

    The claim scope is deliberately NOT checked here: smoke/dry rejection is
    the final scorer's readiness gate (Task 5's channels), while this
    primitive only pins what prediction itself requires.
    """
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ValueError(f"artifact is missing {missing} -- not a "
                         "pf-context m3 artifact")
    for key, expected in CONTRACT_TOKENS.items():
        if artifact[key] != expected:
            raise ValueError(
                f"artifact {key} must be {expected!r}, got "
                f"{artifact[key]!r} -- the context sweep is only comparable "
                "under the fixed base model")
    if int(artifact["n_act"]) != INPUT_WIDTH:
        raise ValueError(
            f"the fixed Arm B input is {INPUT_WIDTH} columns, got "
            f"{int(artifact['n_act'])}")
    if int(artifact["n_out"]) != N_OUT:
        raise ValueError(f"n_out must be {N_OUT}, got {artifact['n_out']}")
    context_level(artifact["context_label"])     # unknown label -> ValueError
    if int(artifact["score_block"]) <= 0:
        raise ValueError("score_block must be positive")
    if int(artifact["depth"]) <= 0:
        raise ValueError("depth must be positive")
    return artifact


def rebuild_model(artifact):
    """Reconstruct the production ``ActSeqAttn`` from a validated artifact."""
    model = ActSeqAttn(n_act=int(artifact["n_act"]), n_out=int(artifact["n_out"]),
                       depth=int(artifact["depth"]), pe=artifact["pe"],
                       **FROZEN_HP)
    model.load_state_dict(artifact["state"])
    return model


# ── the per-shot gather: exact scored order, asserted twice ──────────
@dataclasses.dataclass(frozen=True)
class ShotPrediction:
    """One shot's destandardized predictions on exactly the scored rows."""
    shot: int
    row_index: np.ndarray        # int64 native rows, strictly increasing
    timestamp: np.ndarray        # float64 native time at those rows
    prediction: np.ndarray       # float32 (n, 34) absolute metres


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
        out[int(shot)] = ShotPrediction(
            int(shot), rows, gather["time"][rows].astype(np.float64),
            prediction)
    return out


def predict_artifact(artifact, shots, target_dir, sidecar_dir, device):
    validate_artifact_contract(artifact)
    context = context_level(artifact["context_label"])
    dataset = PFContextDataset(
        target_dir, sidecar_dir, shots, context,
        artifact["feature_mean"], artifact["feature_std"],
        artifact["target_mean"], artifact["target_std"],
        score_block=artifact["score_block"])
    model = rebuild_model(artifact).to(device).eval()
    gathered = initialize_shot_gathers(dataset, shots)
    with torch.no_grad():
        for item_index in range(len(dataset)):
            batch = pad_context_collate([dataset[item_index]])
            attention = layer_attention_mask(
                batch.time.to(device), batch.history_valid.to(device),
                batch.real.to(device), context.seconds,
                artifact["depth"])
            pred = model(
                batch.features.to(device), batch.position.to(device),
                attn_mask=attention)[0].cpu().numpy()
            append_scored_block(gathered, dataset, item_index, pred)
    return finalize_shot_gathers(
        gathered, artifact["target_mean"], artifact["target_std"])


# ── native scoring and the staged cell outputs ───────────────────────
def prediction_key(shot, kind):
    """The m3_pred.npz key namespace: ``shot_<id>_<prediction|row_index|
    timestamp>``."""
    if kind not in ("prediction", "row_index", "timestamp"):
        raise ValueError(f"unknown prediction array kind {kind!r}")
    return f"shot_{int(shot)}_{kind}"


def final_predictions_bytes(predictions):
    """Return deterministic compressed prediction bytes in shot insertion order."""
    arrays = {}
    for shot, pred in predictions.items():
        arrays[prediction_key(shot, "prediction")] = np.asarray(
            pred.prediction, np.float32)
        arrays[prediction_key(shot, "row_index")] = np.asarray(
            pred.row_index, np.int64)
        arrays[prediction_key(shot, "timestamp")] = np.asarray(
            pred.timestamp, np.float64)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def save_final_predictions(path, predictions):
    """Compressed per-shot predictions, exact row indices, float64
    timestamps; row order is the exact scored order."""
    pathlib.Path(path).write_bytes(final_predictions_bytes(predictions))


def load_final_predictions(path):
    """Read a staged ``m3_pred.npz`` back, enforcing the row-index contract
    (strictly increasing, unique) on every shot."""
    out = {}
    with np.load(path) as data:
        shots = sorted({int(name.split("_")[1]) for name in data.files})
        for shot in shots:
            rows = data[prediction_key(shot, "row_index")].astype(np.int64)
            if not np.all(np.diff(rows) > 0):
                raise RuntimeError(
                    f"{path}: shot {shot} row indices must be strictly "
                    "increasing and unique -- duplicate predictions")
            out[shot] = ShotPrediction(
                shot, rows,
                data[prediction_key(shot, "timestamp")].astype(np.float64),
                data[prediction_key(shot, "prediction")].astype(np.float32))
    return out


def load_truth_rows(target_dir, shot, row_index):
    """``(T (n, 34), bnd (n, N, 2))`` -- the truth target block and native
    polyline at exactly the predicted rows."""
    path = pathlib.Path(target_dir) / f"{int(shot)}.npz"
    target_all, _finite = load_target(path)
    with np.load(path) as data:
        bnd = data["bnd_RZ"].astype(np.float64)[row_index]
    return target_all[row_index], bnd


def _atomic_json_dump(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(
        obj, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _csv_bytes(header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=header, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_csv_bytes(header, rows))


def _canonical_json_bytes(obj):
    return (json.dumps(
        obj, ensure_ascii=False, allow_nan=False,
        indent=2, sort_keys=True) + "\n").encode("utf-8")


def _open_nofollow_directory(path):
    path = pathlib.Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"cannot open final cell directory {path}: {exc}") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"{path}: final cell must be a real no-follow directory")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if ((opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISDIR(opened.st_mode)):
        os.close(descriptor)
        raise RuntimeError(f"{path}: final cell changed while opening")
    return descriptor


def _read_child_bytes(directory_fd, name, *, display):
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"{display}: final member must be regular")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _publish_cell_member(
        directory_fd, output_dir, name, payload, *, validator, state_label):
    payload = bytes(payload)
    try:
        existing = _read_child_bytes(
            directory_fd, name, display=pathlib.Path(output_dir) / name)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        validator(existing)
        if existing != payload:
            raise RuntimeError(
                f"{output_dir}/{name}: existing {state_label} differs")
        return
    temp_name = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor = None
    try:
        descriptor = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=directory_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while publishing {state_label}")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        temp_payload = _read_child_bytes(
            directory_fd, temp_name,
            display=pathlib.Path(output_dir) / temp_name)
        validator(temp_payload)
        try:
            os.link(
                temp_name, name,
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            existing = _read_child_bytes(
                directory_fd, name, display=pathlib.Path(output_dir) / name)
            validator(existing)
            if existing != payload:
                raise RuntimeError(
                    f"{output_dir}/{name}: concurrent {state_label} differs")
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temp_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _validate_prediction_bytes(payload, shots):
    expected = tuple(int(shot) for shot in shots)
    with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as data:
        keys = tuple(
            prediction_key(shot, kind)
            for shot in expected
            for kind in ("prediction", "row_index", "timestamp"))
        if tuple(data.files) != keys:
            raise RuntimeError("prediction NPZ keys/order changed")
        for shot in expected:
            prediction = data[prediction_key(shot, "prediction")]
            rows = data[prediction_key(shot, "row_index")]
            timestamps = data[prediction_key(shot, "timestamp")]
            if (prediction.ndim != 2 or prediction.shape[1] != 34
                    or rows.shape != (prediction.shape[0],)
                    or timestamps.shape != (prediction.shape[0],)):
                raise RuntimeError(f"shot {shot}: prediction member shapes changed")
            if (prediction.dtype != np.float32 or rows.dtype != np.int64
                    or timestamps.dtype != np.float64):
                raise RuntimeError(f"shot {shot}: prediction member dtypes changed")
            if (not np.isfinite(prediction).all()
                    or not np.isfinite(timestamps).all()
                    or not np.all(np.diff(rows) > 0)):
                raise RuntimeError(f"shot {shot}: invalid prediction values/rows")
    return True


def _validate_metrics_bytes(payload, shots):
    reader = csv.DictReader(io.StringIO(bytes(payload).decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != tuple(PER_SHOT_HEADER):
        raise RuntimeError("metrics header changed")
    rows = list(reader)
    if [int(row["shot"]) for row in rows] != [int(shot) for shot in shots]:
        raise RuntimeError("metrics shot membership/order changed")
    for row in rows:
        for field in PER_SHOT_HEADER[4:]:
            if not np.isfinite(float(row[field])):
                raise RuntimeError(f"metrics {field} is non-finite")
    return rows


def score_artifact_payload(
        artifact, shots, target_dir, sidecar_dir, theta,
        device="cpu", floor=None, *, transaction_provenance=None,
        transaction_sha256=None, artifact_sha256=None,
        fingerprint_json_sha256=None, validation_selection_sha256=None,
        availability_audit_sha256=None, work2_marker_sha256=None,
        floor_sha256=None):
    """Predict every shot of one artifact and write its staged cell.

    Predictions are gathered on exactly the common-valid rows, scored with
    the upstream native per-shot metrics against the raw truth polyline, and
    written as the cell's three files. Returns the per-shot metric rows.
    """
    validate_artifact_contract(artifact)
    predictions = predict_artifact(artifact, shots, target_dir, sidecar_dir,
                                   device)
    rows = []
    for shot in sorted(int(s) for s in shots):
        pred = predictions[shot]
        truth, bnd = load_truth_rows(target_dir, shot, pred.row_index)
        row, _mean_symmetric = score_shot(
            pred.prediction, truth, bnd, theta,
            artifact["target_mean"], artifact["target_std"])
        row.update(context=str(artifact["context_label"]),
                   seed=int(artifact["seed"]), shot=int(shot))
        rows.append(row)
    ordered_shots = [int(s) for s in sorted(int(s) for s in shots)]
    metadata = {
        "study": "pf_context",
        "run_name": (f"pfctx_{artifact['context_label']}"
                     f"_s{int(artifact['seed'])}"),
        "context_label": str(artifact["context_label"]),
        "nominal_samples": int(artifact["nominal_samples"]),
        "context_seconds": float(artifact["context_seconds"]),
        "per_layer_seconds": float(artifact["per_layer_seconds"]),
        "score_block": int(artifact["score_block"]),
        "seed": int(artifact["seed"]),
        "depth": int(artifact["depth"]),
        "world_size": int(artifact["world_size"]),
        "effective_global_batch": int(artifact["effective_global_batch"]),
        "best_val_mse": float(artifact["best_val_mse"]),
        "stop_epoch": int(artifact["stop_epoch"]),
        "epochs_completed": int(artifact["epochs_completed"]),
        "device": str(device),
        "scored_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "n_shots": len(rows),
        "shots": ordered_shots,
        "n_pred_rows": int(sum(r["n_slices"] for r in rows)),
        "row_order_contract": ("strictly increasing unique native rows "
                              "equal to each shot's common-valid indices"),
        "fingerprints": dict(artifact["fingerprints"]),
        "floor": floor,
        "transaction_provenance": transaction_provenance,
        "transaction_sha256": transaction_sha256,
        "artifact_sha256": artifact_sha256,
        "fingerprint_json_sha256": fingerprint_json_sha256,
        "validation_selection_sha256": validation_selection_sha256,
        "availability_audit_sha256": availability_audit_sha256,
        "work2_marker_sha256": work2_marker_sha256,
        "floor_sha256": floor_sha256,
    }
    prediction_bytes = final_predictions_bytes(predictions)
    metrics_bytes = _csv_bytes(PER_SHOT_HEADER, rows)
    metadata_bytes = _canonical_json_bytes(metadata)
    if transaction_provenance is not None:
        required = {
            "transaction_sha256": transaction_sha256,
            "artifact_sha256": artifact_sha256,
            "fingerprint_json_sha256": fingerprint_json_sha256,
            "validation_selection_sha256": validation_selection_sha256,
            "availability_audit_sha256": availability_audit_sha256,
            "work2_marker_sha256": work2_marker_sha256,
            "floor_sha256": floor_sha256,
        }
        missing = sorted(key for key, value in required.items() if value is None)
        if missing:
            raise RuntimeError(
                f"publication cell metadata provenance is incomplete: {missing}")
    _validate_prediction_bytes(prediction_bytes, ordered_shots)
    _validate_metrics_bytes(metrics_bytes, ordered_shots)
    if set(metadata) != RUN_METADATA_FIELDS:
        raise RuntimeError("run metadata schema construction changed")
    return ScoredArtifactPayload(
        rows=tuple(rows),
        prediction_bytes=prediction_bytes,
        metrics_bytes=metrics_bytes,
        metadata=metadata,
        metadata_bytes=metadata_bytes,
    )


def score_artifact(
        artifact, shots, target_dir, sidecar_dir, theta, output_dir,
        device="cpu", floor=None, **provenance):
    """Compatibility wrapper: build complete bytes, then publish one cell."""
    payload = score_artifact_payload(
        artifact, shots, target_dir, sidecar_dir, theta,
        device=device, floor=floor, **provenance)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m3_pred.npz").write_bytes(payload.prediction_bytes)
    (output_dir / "per_shot_metrics.csv").write_bytes(payload.metrics_bytes)
    (output_dir / "run_metadata.json").write_bytes(payload.metadata_bytes)
    return list(payload.rows)
