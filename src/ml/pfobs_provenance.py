# -*- coding: utf-8 -*-
"""Work 2 normalization, audit, validation, and final provenance.

This module owns the exact four-array normalization hash, the split-bound
train+validation source-audit identity, the immutable validation decision, the
persistent validation/final transaction flock, the representation-floor
contract, and the versioned publication final marker consumed by Work 3.
"""
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import socket
import time
from collections.abc import Mapping, Sequence

import numpy as np

from src.utils import PersistentFlock

SOURCE_AUDIT_SCHEMA = "pf_observability_source_audit_identity_v1"
AUDIT_GENERATION_VERSION = 1
PUBLICATION_AUDIT_SHOTS = 674
IDENTITY_FIELDS = {
    "schema",
    "audit_generation_version",
    "split_sha256",
    "expected_shot_count",
    "audited_shot_count",
    "unauditable_shot_count",
    "expected_shots_sha256",
    "audited_shots_sha256",
    "target_meta_sha256",
    "sidecar_meta_sha256",
    "source_audit_csv_sha256",
    "reconstruction_provenance_sha256",
    "source_sha256",
}


class SourceAuditIdentityError(ValueError):
    """Raised when a Work 2 source-audit identity is absent or inconsistent."""


VALIDATION_LOCK_NAME = ".validation_freeze.lock"
VALIDATION_LOCK_SCHEMA = "pf_observability_validation_freeze_lock_v1"


class ValidationFreezeLock:
    """One persistent state-family flock for training, validation, and final.

    Publication training uses ``shared=True`` for the complete artifact-writing
    invocation. Validation and final transitions retain the default exclusive
    lock. Shared holders never rewrite the ownership payload; an exclusive
    holder updates it only after acquiring the same inode.
    """

    def __init__(self, stats_root, *, shared=False):
        self.stats_root = pathlib.Path(stats_root)
        self.path = self.stats_root / VALIDATION_LOCK_NAME
        self.shared = bool(shared)
        self.fd = None
        self.identity = None
        self._lock = None

    def __enter__(self):
        self._lock = PersistentFlock(
            self.path,
            shared=self.shared,
            state_label="validation freeze",
            mode=0o600,
        )
        self._lock.__enter__()
        self.fd = self._lock.fd
        self.identity = self._lock.identity
        if not self.shared:
            payload = (json.dumps({
                "schema": VALIDATION_LOCK_SCHEMA,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_ns": time.time_ns(),
            }, indent=2, sort_keys=True) + "\n").encode("utf-8")
            try:
                os.ftruncate(self.fd, 0)
                os.lseek(self.fd, 0, os.SEEK_SET)
                view = memoryview(payload)
                while view:
                    written = os.write(self.fd, view)
                    if written <= 0:
                        raise OSError(
                            "short write while updating validation lock")
                    view = view[written:]
                os.fsync(self.fd)
            except Exception:
                self._lock.__exit__(None, None, None)
                self.fd = None
                self.identity = None
                self._lock = None
                raise
        return self

    def assert_held(self, *, path_only=False):
        del path_only  # retained for the historical call signature
        if self._lock is None:
            raise RuntimeError(
                f"{self.path}: validation freeze lock descriptor is closed")
        self._lock.assert_held()
        self.fd = self._lock.fd
        self.identity = self._lock.identity

    def __exit__(self, exc_type, exc, traceback):
        if self._lock is not None:
            self._lock.__exit__(exc_type, exc, traceback)
        self.fd = None
        self.identity = None
        self._lock = None
        return False


def write_exclusive_destination(path, payload, *, state_label="validation freeze"):
    """Create, write, fsync, and close one destination without replacement."""
    path = pathlib.Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            f"{path} appeared during {state_label}; refusing to overwrite "
            "concurrent state") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def normalization_stats_sha256(mean, std, tgt_mean, tgt_std):
    """Hash dtype, shape, and contiguous bytes of the four run statistics.

    The fixed order is input mean/std followed by target mean/std. No dtype or
    shape coercion is performed: changing either must move the digest just as
    changing an array value does.
    """
    digest = hashlib.sha256()
    for array in (mean, std, tgt_mean, tgt_std):
        data = np.ascontiguousarray(array)
        digest.update(f"{data.dtype.str}:{data.shape};".encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def _canonical_shot_sha256(shots: Sequence[int]) -> str:
    ordered = tuple(sorted(int(shot) for shot in shots))
    payload = "".join(f"{shot}\n" for shot in ordered).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _normalized_unique_shots(shots, *, label):
    normalized = tuple(int(shot) for shot in shots)
    if len(normalized) != len(set(normalized)):
        raise SourceAuditIdentityError(
            f"source audit identity {label} contains duplicate shots")
    return normalized


def _work2_hash_helpers():
    # Lazy to avoid coupling module import order to pf_observability's generic
    # split API. The source closure itself remains owned there.
    from .pf_observability import (  # pylint: disable=import-outside-toplevel
        existing_source_files,
        sha256_file,
        sha256_tree,
    )
    return existing_source_files, sha256_file, sha256_tree


def build_source_audit_identity(
    *,
    split_path,
    split,
    npz_dir,
    sidecar_dir,
    source_audit_csv,
    reconstruction_provenance,
    audited_shots,
    unauditable_shots,
    project_root,
    source_audit_csv_bytes=None,
    reconstruction_provenance_bytes=None,
):
    """Build the complete publication source-audit identity mapping.

    A valid identity covers exactly the 598 training plus 76 validation shots,
    with no duplicates, omissions, extras, or unauditable shots. Test and
    excluded membership therefore cannot enter the audited-shot hash.
    """
    existing_source_files, sha256_file, sha256_tree = _work2_hash_helpers()
    expected = _normalized_unique_shots(
        tuple(split.train) + tuple(split.validation), label="expected shots")
    audited = _normalized_unique_shots(audited_shots, label="audited shots")
    unauditable = _normalized_unique_shots(
        unauditable_shots, label="unauditable shots")
    if len(expected) != PUBLICATION_AUDIT_SHOTS:
        raise SourceAuditIdentityError(
            "source audit identity must cover exactly 674 train+validation "
            f"shots, got {len(expected)}")
    if unauditable:
        raise SourceAuditIdentityError(
            "source audit identity requires zero unauditable shots, got "
            f"{len(unauditable)}")
    if set(audited) != set(expected) or len(audited) != len(expected):
        missing = sorted(set(expected) - set(audited))
        extra = sorted(set(audited) - set(expected))
        raise SourceAuditIdentityError(
            "source audit identity audited shots do not equal the frozen "
            f"train+validation shots (missing={missing[:5]}, extra={extra[:5]})")

    npz_dir = pathlib.Path(npz_dir)
    sidecar_dir = pathlib.Path(sidecar_dir)
    project_root = pathlib.Path(project_root)
    source_audit_csv = pathlib.Path(source_audit_csv)
    reconstruction_provenance = pathlib.Path(reconstruction_provenance)
    return {
        "schema": SOURCE_AUDIT_SCHEMA,
        "audit_generation_version": AUDIT_GENERATION_VERSION,
        "split_sha256": sha256_file(split_path),
        "expected_shot_count": len(expected),
        "audited_shot_count": len(audited),
        "unauditable_shot_count": len(unauditable),
        "expected_shots_sha256": _canonical_shot_sha256(expected),
        "audited_shots_sha256": _canonical_shot_sha256(audited),
        "target_meta_sha256": sha256_file(npz_dir / "meta.json"),
        "sidecar_meta_sha256": sha256_file(sidecar_dir / "meta.json"),
        "source_audit_csv_sha256": (
            hashlib.sha256(bytes(source_audit_csv_bytes)).hexdigest()
            if source_audit_csv_bytes is not None
            else sha256_file(source_audit_csv)
        ),
        "reconstruction_provenance_sha256": (
            hashlib.sha256(bytes(reconstruction_provenance_bytes)).hexdigest()
            if reconstruction_provenance_bytes is not None
            else sha256_file(reconstruction_provenance)
        ),
        "source_sha256": sha256_tree(
            project_root, include=existing_source_files(project_root)),
    }


def _read_identity(path):
    path = pathlib.Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceAuditIdentityError(
            f"cannot read source audit identity at {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != IDENTITY_FIELDS:
        raise SourceAuditIdentityError(
            "source audit identity must contain exactly the versioned schema "
            "fields")
    return dict(payload)


def validate_source_audit_identity(
    identity_path,
    *,
    split_path,
    split,
    npz_dir,
    sidecar_dir,
    project_root,
):
    """Validate identity bytes against the current split, data, audit, and code.

    ``source_audit.csv`` and ``reconstruction_provenance.json`` are resolved
    beside ``identity_path``. Reusing an identity for a different split or
    changing either audited output, either dataset metadata file, or any pinned
    source file is a hard failure.
    """
    identity_path = pathlib.Path(identity_path)
    actual = _read_identity(identity_path)
    if actual.get("schema") != SOURCE_AUDIT_SCHEMA:
        raise SourceAuditIdentityError(
            "source audit identity schema is not the Work 2 publication schema")
    if actual.get("audit_generation_version") != AUDIT_GENERATION_VERSION:
        raise SourceAuditIdentityError(
            "source audit identity generation version changed")
    expected_shots = tuple(split.train) + tuple(split.validation)
    expected = build_source_audit_identity(
        split_path=split_path,
        split=split,
        npz_dir=npz_dir,
        sidecar_dir=sidecar_dir,
        source_audit_csv=identity_path.parent / "source_audit.csv",
        reconstruction_provenance=(
            identity_path.parent / "reconstruction_provenance.json"),
        audited_shots=expected_shots,
        unauditable_shots=(),
        project_root=project_root,
    )
    differing = [
        key for key in sorted(IDENTITY_FIELDS)
        if actual.get(key) != expected.get(key)
    ]
    if differing:
        raise SourceAuditIdentityError(
            "source audit identity disagrees with current publication inputs "
            f"({', '.join(differing)})")
    return actual


VALIDATION_DECISION_SCHEMA = "pf_observability_validation_decision_v1"
VALIDATION_DECISION_GENERATION_VERSION = 1
WORK2_ARMS = ("A", "B", "C", "D")
WORK2_SEEDS = (0, 1, 2, 3, 4)
PUBLICATION_VALIDATION_SHOTS = 76
VALIDATION_DECISION_FIELDS = {
    "schema",
    "decision_generation_version",
    "no_further_model_selection",
    "split",
    "matrix",
    "hashes",
    "runs",
}
VALIDATION_SPLIT_FIELDS = {
    "name",
    "version",
    "n_validation_shots",
    "validation_shots",
    "validation_shots_sha256",
}
VALIDATION_MATRIX_FIELDS = {
    "arms",
    "seeds",
    "run_prefix",
    "n_runs",
    "validation_matrix_path",
    "validation_matrix_header",
}
VALIDATION_HASH_FIELDS = {
    "validation_matrix_sha256",
    "validation_json_tree_sha256",
    "artifact_tree_sha256",
    "config_sha256",
    "split_sha256",
    "target_meta_sha256",
    "sidecar_meta_sha256",
    "shot_metadata_sha256",
    "slice_strata_sha256",
    "source_sha256",
    "source_audit_identity_sha256",
}
VALIDATION_RUN_FIELDS = {"run_name", "arm", "seed", "run_fingerprint"}
_COMMON_FINGERPRINT_HASH_FIELDS = (
    "config_sha256",
    "split_sha256",
    "target_meta_sha256",
    "sidecar_meta_sha256",
    "shot_metadata_sha256",
    "slice_strata_sha256",
    "source_sha256",
    "source_audit_identity_sha256",
)


class ValidationDecisionError(ValueError):
    """Raised when a Work 2 validation decision is absent or inconsistent."""


def canonical_work2_validation_decision_bytes(payload):
    """Serialize a Work 2 validation decision as stable UTF-8 JSON + LF."""
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationDecisionError(
            "validation decision is not deterministic JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _common_fingerprint_value(run_records, field):
    values = {record["run_fingerprint"].get(field) for record in run_records}
    if len(values) != 1:
        raise ValidationDecisionError(
            f"validation run fingerprints disagree on {field}")
    value = values.pop()
    if not isinstance(value, str) or not value:
        raise ValidationDecisionError(
            f"validation run fingerprint {field} is absent")
    return value


def _validate_validation_run_records(run_records, run_prefix):
    records = tuple(run_records)
    expected = tuple(
        (arm, seed, f"{run_prefix}_{arm.lower()}_s{seed}")
        for arm in WORK2_ARMS for seed in WORK2_SEEDS
    )
    if len(records) != len(expected):
        raise ValidationDecisionError(
            "validation decision requires exactly 20 run fingerprints")
    normalized = []
    for raw, (arm, seed, run_name) in zip(records, expected):
        if not isinstance(raw, Mapping) or set(raw) != VALIDATION_RUN_FIELDS:
            raise ValidationDecisionError(
                "validation run record must contain exactly run_name/arm/seed/"
                "run_fingerprint")
        fingerprint = raw["run_fingerprint"]
        if not isinstance(fingerprint, Mapping):
            raise ValidationDecisionError(
                f"validation run {run_name} fingerprint must be a mapping")
        if (raw["run_name"], raw["arm"], raw["seed"]) != (
                run_name, arm, seed):
            raise ValidationDecisionError(
                "validation run records are not in frozen arm-major order")
        if (fingerprint.get("arm"), fingerprint.get("seed")) != (arm, seed):
            raise ValidationDecisionError(
                f"validation run {run_name} fingerprint claims another cell")
        normalized.append({
            "run_name": run_name,
            "arm": arm,
            "seed": seed,
            "run_fingerprint": dict(fingerprint),
        })
    return tuple(normalized)


def build_work2_validation_decision(
    *,
    split,
    run_prefix,
    validation_matrix_header,
    validation_matrix_sha256,
    validation_json_tree_sha256,
    artifact_tree_sha256,
    run_records,
):
    """Build the complete canonical Work 2 validation-decision payload.

    The payload binds the exact 4x5 arm-major matrix, all full run
    fingerprints, exact 76 validation shots, the 20-file artifact tree,
    deterministic validation-table and validation-JSON hashes, and the common
    split/data/metadata/strata/source/audit hashes carried by every freshly
    validated fingerprint.
    """
    validation_shots = tuple(int(shot) for shot in split.validation)
    if (len(validation_shots) != PUBLICATION_VALIDATION_SHOTS
            or len(set(validation_shots)) != len(validation_shots)):
        raise ValidationDecisionError(
            "validation decision requires exactly 76 unique validation shots")
    records = _validate_validation_run_records(run_records, run_prefix)
    hashes = {
        "validation_matrix_sha256": str(validation_matrix_sha256),
        "validation_json_tree_sha256": str(validation_json_tree_sha256),
        "artifact_tree_sha256": str(artifact_tree_sha256),
        **{
            field: _common_fingerprint_value(records, field)
            for field in _COMMON_FINGERPRINT_HASH_FIELDS
        },
    }
    return {
        "schema": VALIDATION_DECISION_SCHEMA,
        "decision_generation_version": VALIDATION_DECISION_GENERATION_VERSION,
        "no_further_model_selection": True,
        "split": {
            "name": str(split.name),
            "version": int(split.version),
            "n_validation_shots": len(validation_shots),
            "validation_shots": list(validation_shots),
            "validation_shots_sha256": _canonical_shot_sha256(
                validation_shots),
        },
        "matrix": {
            "arms": list(WORK2_ARMS),
            "seeds": list(WORK2_SEEDS),
            "run_prefix": str(run_prefix),
            "n_runs": len(records),
            "validation_matrix_path": "generated/validation_matrix.csv",
            "validation_matrix_header": list(validation_matrix_header),
        },
        "hashes": hashes,
        "runs": list(records),
    }


def _validate_work2_validation_decision_schema(payload):
    if not isinstance(payload, Mapping) or set(payload) != \
            VALIDATION_DECISION_FIELDS:
        raise ValidationDecisionError(
            "validation decision must contain exactly the versioned fields")
    if payload.get("schema") != VALIDATION_DECISION_SCHEMA:
        raise ValidationDecisionError("validation decision schema changed")
    if payload.get("decision_generation_version") != \
            VALIDATION_DECISION_GENERATION_VERSION:
        raise ValidationDecisionError(
            "validation decision generation version changed")
    if payload.get("no_further_model_selection") is not True:
        raise ValidationDecisionError(
            "validation decision must prohibit further model selection")
    split = payload.get("split")
    matrix = payload.get("matrix")
    hashes = payload.get("hashes")
    runs = payload.get("runs")
    if not isinstance(split, Mapping) or set(split) != VALIDATION_SPLIT_FIELDS:
        raise ValidationDecisionError("validation decision split schema changed")
    if (split.get("n_validation_shots") != PUBLICATION_VALIDATION_SHOTS
            or not isinstance(split.get("validation_shots"), list)
            or len(split["validation_shots"]) != PUBLICATION_VALIDATION_SHOTS
            or len(set(split["validation_shots"]))
            != PUBLICATION_VALIDATION_SHOTS
            or split.get("validation_shots_sha256")
            != _canonical_shot_sha256(split["validation_shots"])):
        raise ValidationDecisionError(
            "validation decision does not bind exactly 76 validation shots")
    if not isinstance(matrix, Mapping) or set(matrix) != VALIDATION_MATRIX_FIELDS:
        raise ValidationDecisionError("validation decision matrix schema changed")
    if (matrix.get("arms") != list(WORK2_ARMS)
            or matrix.get("seeds") != list(WORK2_SEEDS)
            or matrix.get("n_runs") != 20
            or matrix.get("validation_matrix_path")
            != "generated/validation_matrix.csv"):
        raise ValidationDecisionError(
            "validation decision matrix is not the frozen 4x5 contract")
    if not isinstance(hashes, Mapping) or set(hashes) != VALIDATION_HASH_FIELDS:
        raise ValidationDecisionError("validation decision hash schema changed")
    if not isinstance(runs, list):
        raise ValidationDecisionError("validation decision runs must be a list")
    _validate_validation_run_records(runs, matrix.get("run_prefix"))
    return dict(payload)


def validate_work2_validation_decision(
    decision_path,
    *,
    expected_payload,
    validation_matrix_path,
):
    """Validate frozen bytes against a freshly rederived decision and table."""
    decision_path = pathlib.Path(decision_path)
    validation_matrix_path = pathlib.Path(validation_matrix_path)
    try:
        actual_bytes = decision_path.read_bytes()
        actual = json.loads(actual_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationDecisionError(
            f"cannot read validation decision at {decision_path}: {exc}") from exc
    actual = _validate_work2_validation_decision_schema(actual)
    expected = _validate_work2_validation_decision_schema(expected_payload)
    if actual_bytes != canonical_work2_validation_decision_bytes(actual):
        raise ValidationDecisionError(
            "validation decision bytes are not canonical")
    if actual_bytes != canonical_work2_validation_decision_bytes(expected):
        raise ValidationDecisionError(
            "validation decision disagrees with current validation artifacts")
    try:
        table_sha256 = hashlib.sha256(
            validation_matrix_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValidationDecisionError(
            f"cannot read validation matrix at {validation_matrix_path}: {exc}") \
            from exc
    if table_sha256 != actual["hashes"]["validation_matrix_sha256"]:
        raise ValidationDecisionError(
            "validation matrix hash disagrees with the frozen decision")
    return actual


PER_SHOT_METRICS_HEADER = (
    "run", "arm", "seed", "shot", "n_slices", "mean_symmetric_mm", "p95_mm",
    "chamfer_rms_mm", "hausdorff_mm", "area_abs_m2", "centroid_mm",
    "elongation_abs", "triangularity_upper_abs", "triangularity_lower_abs",
    "ccc", "radii_mse", "centre_mse", "cold_start_n",
    "cold_start_mean_symmetric_mm",
)
REPRESENTATION_FLOOR_HEADER = (
    "shot", "n_slices", "mean_symmetric_mm", "p95_mm", "chamfer_rms_mm",
    "hausdorff_mm", "area_abs_m2", "centroid_mm", "elongation_abs",
    "triangularity_upper_abs", "triangularity_lower_abs",
)
PUBLICATION_FINAL_TEST_SHOTS = 76
WORK2_FINAL_MARKER_SCHEMA = "pf_observability_final_test_v2"
WORK2_FINAL_MARKER_GENERATION_VERSION = 2
WORK2_FINAL_MARKER_FIELDS = {
    "schema",
    "marker_generation_version",
    "marker",
    "split",
    "dataset",
    "source_audit",
    "validation_decision",
    "representation_floor",
    "matrix",
    "runs",
}
WORK2_FINAL_SPLIT_FIELDS = {
    "name", "version", "claim_scope", "split_strategy", "counts", "ranges",
    "test", "excluded", "gap_policy", "exclusion_policy",
    "prior_test_access", "manifest_sha256", "test_shots_sha256",
    "excluded_shots_sha256",
}
WORK2_FINAL_DATASET_FIELDS = {
    "target_meta_sha256", "sidecar_meta_sha256", "shot_metadata_sha256",
    "slice_strata_sha256",
}
WORK2_FINAL_SOURCE_AUDIT_FIELDS = IDENTITY_FIELDS | {
    "source_audit_identity_sha256",
}
WORK2_FINAL_VALIDATION_FIELDS = VALIDATION_DECISION_FIELDS | {
    "validation_decision_sha256",
}
WORK2_FINAL_FLOOR_FIELDS = {
    "path", "sha256", "header", "n_shots", "shots", "shots_sha256",
    "n_slices",
}
WORK2_FINAL_MATRIX_FIELDS = {
    "config_sha256", "arms", "seeds", "run_prefix", "n_runs",
}
WORK2_FINAL_RUN_FIELDS = {
    "run_name", "arm", "seed", "n_shots", "n_pred_rows", "best_val_mse",
    "run_fingerprint", "per_shot_metrics_sha256", "m3_pred_sha256",
}


class RepresentationFloorError(ValueError):
    """Raised when the Work 2 representation floor is malformed or mismatched."""


class Work2FinalMarkerError(ValueError):
    """Raised when the versioned Work 2 final marker contract is invalid."""


def _strict_positive_int(value, *, label):
    if isinstance(value, bool):
        raise RepresentationFloorError(f"representation floor {label} must be positive")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RepresentationFloorError(
            f"representation floor {label} must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise RepresentationFloorError(
            f"representation floor {label} must be a positive integer")
    return parsed


def validate_representation_floor(
        path, *, expected_shots, expected_n_slices=None,
        require_publication_count=True):
    """Validate and hash the exact representation floor consumed by Work 3.

    The file must have the frozen header and exactly 76 unique rows, one for
    every expected test shot. ``n_slices`` is positive and every physical
    metric is finite. The returned digest is computed from the exact validated
    byte snapshot rather than reopening the path after parsing.
    """
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_file():
        raise RepresentationFloorError(
            f"{path}: representation floor must be a regular no-follow file")
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RepresentationFloorError(
            f"cannot read representation floor at {path}: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != REPRESENTATION_FLOOR_HEADER:
        raise RepresentationFloorError(
            "representation floor header is not the frozen contract")
    expected = tuple(int(shot) for shot in expected_shots)
    if len(set(expected)) != len(expected):
        raise RepresentationFloorError(
            "representation floor expects unique test shots")
    if require_publication_count and len(expected) != PUBLICATION_FINAL_TEST_SHOTS:
        raise RepresentationFloorError(
            "representation floor expects exactly 76 unique test shots")
    rows = list(reader)
    if len(rows) != len(expected):
        raise RepresentationFloorError(
            f"representation floor must contain exactly {len(expected)} rows")
    parsed_shots = []
    total_slices = 0
    for row_index, row in enumerate(rows, start=2):
        if None in row or set(row) != set(REPRESENTATION_FLOOR_HEADER):
            raise RepresentationFloorError(
                f"representation floor row {row_index} has the wrong schema")
        try:
            shot = int(row["shot"])
        except (TypeError, ValueError) as exc:
            raise RepresentationFloorError(
                f"representation floor row {row_index} shot is not an integer") \
                from exc
        if str(shot) != str(row["shot"]).strip():
            raise RepresentationFloorError(
                f"representation floor row {row_index} shot is not canonical")
        parsed_shots.append(shot)
        n_slices = _strict_positive_int(
            row["n_slices"], label=f"row {row_index} n_slices")
        if expected_n_slices is not None:
            expected_count = int(expected_n_slices.get(shot, -1))
            if n_slices != expected_count:
                raise RepresentationFloorError(
                    f"representation floor shot {shot} has {n_slices} slices, "
                    f"expected {expected_count}")
        total_slices += n_slices
        for field in REPRESENTATION_FLOOR_HEADER[2:]:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise RepresentationFloorError(
                    f"representation floor row {row_index} {field} is not numeric") \
                    from exc
            if not math.isfinite(value):
                raise RepresentationFloorError(
                    f"representation floor row {row_index} {field} is non-finite")
    if len(set(parsed_shots)) != len(parsed_shots):
        raise RepresentationFloorError(
            "representation floor contains duplicate shot rows")
    if tuple(parsed_shots) != expected:
        missing = sorted(set(expected) - set(parsed_shots))
        extra = sorted(set(parsed_shots) - set(expected))
        raise RepresentationFloorError(
            "representation floor shots differ from the frozen test shots "
            f"(missing={missing[:5]}, extra={extra[:5]})")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "header": list(REPRESENTATION_FLOOR_HEADER),
        "n_shots": len(rows),
        "shots": parsed_shots,
        "shots_sha256": _canonical_shot_sha256(parsed_shots),
        "n_slices": total_slices,
    }


def canonical_work2_final_marker_bytes(payload):
    """Serialize the Work 2 final marker as canonical UTF-8 JSON plus LF."""
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Work2FinalMarkerError(
            "Work 2 final marker is not deterministic JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _normalize_final_runs(runs, *, run_prefix, arms, seeds):
    records = tuple(runs)
    expected = tuple(
        (arm, seed, f"{run_prefix}_{arm.lower()}_s{seed}")
        for arm in arms for seed in seeds
    )
    if len(records) != len(expected):
        raise Work2FinalMarkerError(
            "Work 2 final marker requires all 20 run summaries")
    normalized = []
    for raw, (arm, seed, run_name) in zip(records, expected):
        if not isinstance(raw, Mapping) or set(raw) != WORK2_FINAL_RUN_FIELDS:
            raise Work2FinalMarkerError(
                "Work 2 final run summary schema changed")
        if (raw.get("run_name"), raw.get("arm"), raw.get("seed")) != (
                run_name, arm, seed):
            raise Work2FinalMarkerError(
                "Work 2 final runs are not in frozen arm-major order")
        fingerprint = raw.get("run_fingerprint")
        if (not isinstance(fingerprint, Mapping)
                or fingerprint.get("arm") != arm
                or fingerprint.get("seed") != seed):
            raise Work2FinalMarkerError(
                f"Work 2 final run {run_name} fingerprint claims another cell")
        if raw.get("n_shots") != PUBLICATION_FINAL_TEST_SHOTS:
            raise Work2FinalMarkerError(
                f"Work 2 final run {run_name} must score exactly 76 shots")
        if (type(raw.get("n_pred_rows")) is not int
                or raw["n_pred_rows"] <= 0):
            raise Work2FinalMarkerError(
                f"Work 2 final run {run_name} prediction row count is invalid")
        if not math.isfinite(float(raw.get("best_val_mse"))):
            raise Work2FinalMarkerError(
                f"Work 2 final run {run_name} best_val_mse is non-finite")
        normalized.append({
            **dict(raw),
            "run_fingerprint": dict(fingerprint),
        })
    return normalized


def build_work2_final_marker(
    *,
    split,
    dataset,
    source_audit,
    validation_decision,
    representation_floor,
    matrix,
    runs,
):
    """Build the complete Work 2 final marker written after final publication."""
    if not isinstance(split, Mapping):
        raise Work2FinalMarkerError("Work 2 final split disclosure must be a mapping")
    if (split.get("counts", {}).get("test") != PUBLICATION_FINAL_TEST_SHOTS
            or len(split.get("test", ())) != PUBLICATION_FINAL_TEST_SHOTS
            or len(set(split.get("test", ()))) != PUBLICATION_FINAL_TEST_SHOTS):
        raise Work2FinalMarkerError(
            "Work 2 final split must disclose exactly 76 unique test shots")
    if (split.get("counts", {}).get("excluded") != 9
            or len(split.get("excluded", ())) != 9
            or len(set(split.get("excluded", ()))) != 9):
        raise Work2FinalMarkerError(
            "Work 2 final split must disclose exactly nine excluded shots")
    access_counts = split.get("prior_test_access", {}).get("counts", {})
    if access_counts != {
            "pilot_test": 76, "train_overlap": 60,
            "validation_overlap": 7, "test_overlap": 0}:
        raise Work2FinalMarkerError(
            "Work 2 final split must disclose exact 60/7/0 prior-test overlap")
    if split.get("prior_test_access", {}).get("test_overlap") != []:
        raise Work2FinalMarkerError(
            "Work 2 final split current-test overlap must be empty")
    if not isinstance(matrix, Mapping) or set(matrix) != WORK2_FINAL_MATRIX_FIELDS:
        raise Work2FinalMarkerError("Work 2 final matrix schema changed")
    arms = tuple(matrix.get("arms", ()))
    seeds = tuple(matrix.get("seeds", ()))
    if (arms != WORK2_ARMS or seeds != WORK2_SEEDS
            or matrix.get("n_runs") != 20):
        raise Work2FinalMarkerError(
            "Work 2 final marker requires the frozen 4x5 matrix")
    floor = dict(representation_floor)
    if (floor.get("n_shots") != PUBLICATION_FINAL_TEST_SHOTS
            or floor.get("shots") != list(split["test"])
            or not isinstance(floor.get("sha256"), str)):
        raise Work2FinalMarkerError(
            "Work 2 final representation floor does not match the test split")
    marker = {
        "schema": WORK2_FINAL_MARKER_SCHEMA,
        "marker_generation_version": WORK2_FINAL_MARKER_GENERATION_VERSION,
        "marker": "FINAL_TEST_EVALUATED.json",
        "split": dict(split),
        "dataset": dict(dataset),
        "source_audit": dict(source_audit),
        "validation_decision": dict(validation_decision),
        "representation_floor": floor,
        "matrix": dict(matrix),
        "runs": _normalize_final_runs(
            runs,
            run_prefix=str(matrix["run_prefix"]),
            arms=arms,
            seeds=seeds,
        ),
    }
    _validate_work2_final_marker_schema(marker)
    return marker


def _require_sha256(value, *, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise Work2FinalMarkerError(f"Work 2 final {label} is not a SHA-256")
    return value


def _require_exact_mapping(value, fields, *, label):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise Work2FinalMarkerError(f"Work 2 final {label} schema changed")
    return dict(value)


def _validate_work2_final_marker_schema(payload):
    marker = _require_exact_mapping(
        payload, WORK2_FINAL_MARKER_FIELDS, label="marker")
    if (marker.get("schema") != WORK2_FINAL_MARKER_SCHEMA
            or marker.get("marker_generation_version")
            != WORK2_FINAL_MARKER_GENERATION_VERSION
            or marker.get("marker") != "FINAL_TEST_EVALUATED.json"):
        raise Work2FinalMarkerError("Work 2 final marker schema changed")

    split = _require_exact_mapping(
        marker["split"], WORK2_FINAL_SPLIT_FIELDS, label="split")
    if split.get("counts") != {
            "train": 598, "validation": 76, "test": 76, "excluded": 9}:
        raise Work2FinalMarkerError("Work 2 final split counts changed")
    test_shots = split.get("test")
    excluded = split.get("excluded")
    if (not isinstance(test_shots, list)
            or len(test_shots) != PUBLICATION_FINAL_TEST_SHOTS
            or len(set(test_shots)) != PUBLICATION_FINAL_TEST_SHOTS):
        raise Work2FinalMarkerError(
            "Work 2 final split must disclose exactly 76 unique test shots")
    if (not isinstance(excluded, list) or len(excluded) != 9
            or len(set(excluded)) != 9 or set(test_shots) & set(excluded)):
        raise Work2FinalMarkerError(
            "Work 2 final split must disclose nine disjoint excluded shots")
    if split.get("test_shots_sha256") != _canonical_shot_sha256(test_shots):
        raise Work2FinalMarkerError("Work 2 final test-shot hash changed")
    if split.get("excluded_shots_sha256") != _canonical_shot_sha256(excluded):
        raise Work2FinalMarkerError("Work 2 final excluded-shot hash changed")
    _require_sha256(split.get("manifest_sha256"), label="manifest hash")
    access = split.get("prior_test_access")
    if not isinstance(access, Mapping) or access.get("counts") != {
            "pilot_test": 76, "train_overlap": 60,
            "validation_overlap": 7, "test_overlap": 0}:
        raise Work2FinalMarkerError(
            "Work 2 final split must disclose exact 60/7/0 prior-test overlap")
    for field, count in (
            ("pilot_test", 76), ("train_overlap", 60),
            ("validation_overlap", 7), ("test_overlap", 0)):
        values = access.get(field)
        if (not isinstance(values, list) or len(values) != count
                or len(set(values)) != count):
            raise Work2FinalMarkerError(
                f"Work 2 final prior-test {field} disclosure changed")
    if access["test_overlap"]:
        raise Work2FinalMarkerError(
            "Work 2 final split current-test overlap must be empty")

    dataset = _require_exact_mapping(
        marker["dataset"], WORK2_FINAL_DATASET_FIELDS, label="dataset")
    for field, digest in dataset.items():
        _require_sha256(digest, label=f"dataset {field}")

    source_audit = _require_exact_mapping(
        marker["source_audit"], WORK2_FINAL_SOURCE_AUDIT_FIELDS,
        label="source audit")
    if (source_audit.get("schema") != SOURCE_AUDIT_SCHEMA
            or source_audit.get("audit_generation_version")
            != AUDIT_GENERATION_VERSION
            or source_audit.get("expected_shot_count") != PUBLICATION_AUDIT_SHOTS
            or source_audit.get("audited_shot_count") != PUBLICATION_AUDIT_SHOTS
            or source_audit.get("unauditable_shot_count") != 0):
        raise Work2FinalMarkerError("Work 2 final source-audit identity changed")
    for field in IDENTITY_FIELDS - {
            "schema", "audit_generation_version", "expected_shot_count",
            "audited_shot_count", "unauditable_shot_count"}:
        _require_sha256(source_audit.get(field), label=f"source audit {field}")
    _require_sha256(
        source_audit.get("source_audit_identity_sha256"),
        label="source-audit identity hash")
    if (source_audit["expected_shots_sha256"]
            != source_audit["audited_shots_sha256"]
            or source_audit["split_sha256"] != split["manifest_sha256"]
            or source_audit["target_meta_sha256"]
            != dataset["target_meta_sha256"]
            or source_audit["sidecar_meta_sha256"]
            != dataset["sidecar_meta_sha256"]):
        raise Work2FinalMarkerError(
            "Work 2 final source audit disagrees with split/dataset provenance")

    decision = _require_exact_mapping(
        marker["validation_decision"], WORK2_FINAL_VALIDATION_FIELDS,
        label="validation decision")
    decision_sha256 = _require_sha256(
        decision.pop("validation_decision_sha256"),
        label="validation-decision hash")
    decision = _validate_work2_validation_decision_schema(decision)
    for field, digest in decision["hashes"].items():
        _require_sha256(digest, label=f"validation decision {field}")
    if (decision["split"].get("name") != split.get("name")
            or decision["split"].get("version") != split.get("version")):
        raise Work2FinalMarkerError(
            "Work 2 final validation decision names another split")
    if hashlib.sha256(canonical_work2_validation_decision_bytes(
            decision)).hexdigest() != decision_sha256:
        raise Work2FinalMarkerError(
            "Work 2 final validation-decision hash disagrees with its payload")

    floor = _require_exact_mapping(
        marker["representation_floor"], WORK2_FINAL_FLOOR_FIELDS,
        label="representation floor")
    if (floor.get("path")
            != "final_test/representation_floor_per_shot.csv"
            or floor.get("header") != list(REPRESENTATION_FLOOR_HEADER)
            or floor.get("n_shots") != PUBLICATION_FINAL_TEST_SHOTS
            or floor.get("shots") != test_shots
            or floor.get("shots_sha256") != split["test_shots_sha256"]
            or type(floor.get("n_slices")) is not int
            or floor["n_slices"] <= 0):
        raise Work2FinalMarkerError(
            "Work 2 final representation-floor provenance changed")
    _require_sha256(floor.get("sha256"), label="representation-floor hash")

    matrix = _require_exact_mapping(
        marker["matrix"], WORK2_FINAL_MATRIX_FIELDS, label="matrix")
    arms = tuple(matrix.get("arms", ()))
    seeds = tuple(matrix.get("seeds", ()))
    if (arms != WORK2_ARMS or seeds != WORK2_SEEDS
            or matrix.get("n_runs") != 20
            or not isinstance(matrix.get("run_prefix"), str)
            or not matrix["run_prefix"]):
        raise Work2FinalMarkerError(
            "Work 2 final matrix is not the frozen 4x5 contract")
    _require_sha256(matrix.get("config_sha256"), label="config hash")
    runs = _normalize_final_runs(
        marker["runs"],
        run_prefix=matrix["run_prefix"],
        arms=arms,
        seeds=seeds,
    )
    for run in runs:
        _require_sha256(
            run["per_shot_metrics_sha256"],
            label=f"{run['run_name']} metrics hash")
        _require_sha256(
            run["m3_pred_sha256"], label=f"{run['run_name']} prediction hash")
        fingerprint = run["run_fingerprint"]
        expected_values = {
            "config_sha256": matrix["config_sha256"],
            "split_sha256": split["manifest_sha256"],
            "target_meta_sha256": dataset["target_meta_sha256"],
            "sidecar_meta_sha256": dataset["sidecar_meta_sha256"],
            "shot_metadata_sha256": dataset["shot_metadata_sha256"],
            "slice_strata_sha256": dataset["slice_strata_sha256"],
            "source_sha256": source_audit["source_sha256"],
            "source_audit_identity_sha256": source_audit[
                "source_audit_identity_sha256"],
        }
        for field, expected in expected_values.items():
            if fingerprint.get(field) != expected:
                raise Work2FinalMarkerError(
                    f"Work 2 final run {run['run_name']} {field} disagrees")
        _require_sha256(
            fingerprint.get("normalization_sha256"),
            label=f"{run['run_name']} normalization hash")
    decision_hashes = decision["hashes"]
    for field, expected in (
            ("config_sha256", matrix["config_sha256"]),
            ("split_sha256", split["manifest_sha256"]),
            ("target_meta_sha256", dataset["target_meta_sha256"]),
            ("sidecar_meta_sha256", dataset["sidecar_meta_sha256"]),
            ("shot_metadata_sha256", dataset["shot_metadata_sha256"]),
            ("slice_strata_sha256", dataset["slice_strata_sha256"]),
            ("source_sha256", source_audit["source_sha256"]),
            ("source_audit_identity_sha256",
             source_audit["source_audit_identity_sha256"])):
        if decision_hashes.get(field) != expected:
            raise Work2FinalMarkerError(
                f"Work 2 final validation decision {field} disagrees")
    decision_runs = {
        (record["arm"], record["seed"]): record["run_fingerprint"]
        for record in decision["runs"]
    }
    for run in runs:
        if decision_runs.get((run["arm"], run["seed"])) != \
                run["run_fingerprint"]:
            raise Work2FinalMarkerError(
                f"Work 2 final run {run['run_name']} differs from validation")
    return marker


def validate_work2_final_marker(path):
    """Read and validate canonical Work 2 final-marker bytes."""
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_file():
        raise Work2FinalMarkerError(
            f"{path}: Work 2 final marker must be a regular no-follow file")
    try:
        marker_bytes = path.read_bytes()
        payload = json.loads(marker_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Work2FinalMarkerError(
            f"cannot read Work 2 final marker at {path}: {exc}") from exc
    _validate_work2_final_marker_schema(payload)
    if marker_bytes != canonical_work2_final_marker_bytes(payload):
        raise Work2FinalMarkerError("Work 2 final marker bytes are not canonical")
    matrix = payload.get("matrix")
    if not isinstance(matrix, Mapping) or set(matrix) != WORK2_FINAL_MATRIX_FIELDS:
        raise Work2FinalMarkerError("Work 2 final matrix schema changed")
    arms = tuple(matrix.get("arms", ()))
    seeds = tuple(matrix.get("seeds", ()))
    if arms != WORK2_ARMS or seeds != WORK2_SEEDS or matrix.get("n_runs") != 20:
        raise Work2FinalMarkerError("Work 2 final matrix is not the frozen 4x5 contract")
    _normalize_final_runs(
        payload.get("runs", ()),
        run_prefix=str(matrix.get("run_prefix")),
        arms=arms,
        seeds=seeds,
    )
    return dict(payload)
