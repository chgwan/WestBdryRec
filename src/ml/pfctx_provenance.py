# -*- coding: utf-8 -*-
"""Work 3 availability-audit records, identity, and validation.

The publication identity covers exactly the 598 training plus 76 validation
shots at all seven frozen context levels. It binds the selected manifest, the
ordered 4,718-row CSV, both dataset metadata files, and the complete Work 3
source closure. Test and excluded shots cannot enter a valid identity.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import pathlib
import stat
from collections.abc import Mapping

import numpy as np

from .pf_context import (
    CONTEXT_LEVELS,
    existing_source_files,
    scored_windows,
)
from .pf_observability import sha256_file, sha256_tree
from .pfctx_data import load_context_series
from .pos_encoding import modal_cadence

AVAILABILITY_AUDIT_SCHEMA = "pf_context_availability_audit_identity_v1"
AVAILABILITY_AUDIT_GENERATION_VERSION = 1
PUBLICATION_AUDIT_SHOTS = 674
PUBLICATION_AUDIT_CONTEXTS = 7
PUBLICATION_AUDIT_ROWS = PUBLICATION_AUDIT_SHOTS * PUBLICATION_AUDIT_CONTEXTS
AVAILABILITY_COLUMNS = (
    "shot",
    "role",
    "context",
    "n_rows",
    "cadence_seconds",
    "max_gap_seconds",
    "gap_rows_over_1p5x_cadence",
    "visible_tokens_mean",
    "visible_tokens_min",
    "visible_tokens_max",
    "full_horizon_fraction",
    "score_rows",
    "history_valid_coverage",
)
AVAILABILITY_AUDIT_FIELDS = {
    "schema",
    "audit_generation_version",
    "split_sha256",
    "expected_shot_count",
    "audited_shot_count",
    "expected_shots_sha256",
    "audited_shots_sha256",
    "context_count",
    "contexts",
    "expected_row_count",
    "audited_row_count",
    "target_meta_sha256",
    "sidecar_meta_sha256",
    "context_availability_csv_sha256",
    "source_sha256",
}


class AvailabilityAuditIdentityError(ValueError):
    """Raised when the Work 3 availability audit is absent or inconsistent."""


@dataclasses.dataclass(frozen=True)
class AvailabilityCsvSnapshot:
    payload: bytes
    sha256: str
    rows: tuple[dict[str, str], ...]
    audited_shots: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ValidatedAvailabilityAudit:
    identity: dict[str, object]
    payload: bytes
    sha256: str
    csv: AvailabilityCsvSnapshot


def _canonical_shot_sha256(shots):
    payload = "".join(
        f"{int(shot)}\n" for shot in sorted(int(shot) for shot in shots)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_int(value, *, label):
    if type(value) is not str:
        raise AvailabilityAuditIdentityError(
            f"availability audit {label} must be a canonical integer")
    try:
        result = int(value)
    except ValueError as exc:
        raise AvailabilityAuditIdentityError(
            f"availability audit {label} must be a canonical integer") from exc
    if str(result) != value:
        raise AvailabilityAuditIdentityError(
            f"availability audit {label} must be a canonical integer")
    return result


def _finite_float(value, *, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AvailabilityAuditIdentityError(
            f"availability audit {label} must be finite") from exc
    if not np.isfinite(result):
        raise AvailabilityAuditIdentityError(
            f"availability audit {label} must be finite")
    return result


def _expected_shots(split):
    train = tuple(int(shot) for shot in split.train)
    validation = tuple(int(shot) for shot in split.validation)
    if len(train) != 598 or len(validation) != 76:
        raise AvailabilityAuditIdentityError(
            "availability audit requires exactly 598 train and 76 validation "
            f"shots, got {len(train)} and {len(validation)}")
    expected = train + validation
    if len(expected) != len(set(expected)):
        raise AvailabilityAuditIdentityError(
            "availability audit train+validation membership contains duplicates")
    if len(expected) != PUBLICATION_AUDIT_SHOTS:
        raise AvailabilityAuditIdentityError(
            "availability audit must cover exactly 674 train+validation shots, "
            f"got {len(expected)}")
    forbidden = set(int(shot) for shot in getattr(split, "test", ()))
    forbidden.update(int(shot) for shot in getattr(split, "excluded", ()))
    overlap = sorted(set(expected) & forbidden)
    if overlap:
        raise AvailabilityAuditIdentityError(
            "availability audit train+validation membership overlaps test or "
            f"excluded shots ({overlap[:5]})")
    return train, validation, expected


def _read_nofollow_regular_bytes(path, *, label):
    path = pathlib.Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise AvailabilityAuditIdentityError(
            f"cannot read {label} at {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise AvailabilityAuditIdentityError(
            f"{label} at {path} must be a regular no-follow file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AvailabilityAuditIdentityError(
            f"{label} at {path} must be a regular no-follow file") from exc
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)):
            raise AvailabilityAuditIdentityError(
                f"{label} at {path} must be a regular no-follow file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def availability_csv_snapshot_from_bytes(payload, split):
    """Parse, validate, and hash one immutable CSV byte snapshot."""
    payload = bytes(payload)
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise AvailabilityAuditIdentityError(
            "availability audit CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != AVAILABILITY_COLUMNS:
        raise AvailabilityAuditIdentityError(
            "availability audit CSV header must be exactly "
            + ",".join(AVAILABILITY_COLUMNS))
    rows = list(reader)
    train, validation, expected_shots = _expected_shots(split)
    contexts = tuple(level.label for level in CONTEXT_LEVELS)
    if len(contexts) != PUBLICATION_AUDIT_CONTEXTS:
        raise AvailabilityAuditIdentityError(
            "availability audit context grid must contain exactly seven levels")
    expected_keys = tuple(
        (shot, role, context)
        for role, shots in (("train", train), ("validation", validation))
        for shot in shots
        for context in contexts
    )
    actual_keys = []
    integer_fields = (
        "n_rows",
        "gap_rows_over_1p5x_cadence",
        "visible_tokens_min",
        "visible_tokens_max",
        "score_rows",
    )
    float_fields = (
        "cadence_seconds",
        "max_gap_seconds",
        "visible_tokens_mean",
        "full_horizon_fraction",
        "history_valid_coverage",
    )
    for index, row in enumerate(rows):
        if set(row) != set(AVAILABILITY_COLUMNS):
            raise AvailabilityAuditIdentityError(
                f"availability audit row {index} has an invalid schema")
        shot = _canonical_int(row["shot"], label=f"row {index} shot")
        role = str(row["role"])
        context = str(row["context"])
        actual_keys.append((shot, role, context))
        parsed_ints = {
            field: _canonical_int(
                row[field], label=f"row {index} {field}")
            for field in integer_fields
        }
        parsed_floats = {
            field: _finite_float(
                row[field], label=f"row {index} {field}")
            for field in float_fields
        }
        if (parsed_ints["n_rows"] < 2
                or parsed_ints["visible_tokens_min"] < 1
                or parsed_ints["visible_tokens_max"]
                < parsed_ints["visible_tokens_min"]
                or parsed_ints["score_rows"] < 0
                or parsed_ints["score_rows"] > parsed_ints["n_rows"]
                or parsed_ints["gap_rows_over_1p5x_cadence"] < 0
                or parsed_floats["cadence_seconds"] <= 0
                or parsed_floats["max_gap_seconds"] <= 0
                or parsed_floats["visible_tokens_mean"] <= 0
                or not 0.0 <= parsed_floats["full_horizon_fraction"] <= 1.0
                or not 0.0 <= parsed_floats["history_valid_coverage"] <= 1.0):
            raise AvailabilityAuditIdentityError(
                f"availability audit row {index} contains invalid values")
    if tuple(actual_keys) != expected_keys:
        raise AvailabilityAuditIdentityError(
            "availability audit CSV membership/order must be exactly frozen "
            "train then validation shots, each in seven-context order")
    if len(rows) != PUBLICATION_AUDIT_ROWS:
        raise AvailabilityAuditIdentityError(
            "availability audit CSV must contain exactly 4,718 rows, got "
            f"{len(rows)}")
    audited_shots = tuple(dict.fromkeys(shot for shot, _role, _ctx in actual_keys))
    if audited_shots != expected_shots:
        raise AvailabilityAuditIdentityError(
            "availability audit CSV shot membership differs from the frozen "
            "train+validation membership")
    return AvailabilityCsvSnapshot(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        rows=tuple(dict(row) for row in rows),
        audited_shots=audited_shots,
    )


def read_availability_csv_snapshot(path, split):
    """Read one no-follow regular CSV snapshot, then parse/hash those bytes."""
    payload = _read_nofollow_regular_bytes(
        path, label="availability audit CSV")
    return availability_csv_snapshot_from_bytes(payload, split)


def availability_rows_for_shot(job):
    """Return all requested context rows for one audit shot.

    ``job`` is a plain mapping so this function remains importable and picklable
    by :class:`concurrent.futures.ProcessPoolExecutor` workers.
    """
    shot = int(job["shot"])
    role = str(job["role"])
    target_dir = pathlib.Path(job["target_dir"])
    sidecar_dir = pathlib.Path(job["sidecar_dir"])
    labels = tuple(str(label) for label in job["contexts"])
    levels = {level.label: level for level in CONTEXT_LEVELS}
    try:
        selected = tuple(levels[label] for label in labels)
    except KeyError as exc:
        raise ValueError(f"unknown availability-audit context {exc.args[0]!r}") \
            from exc
    series = load_context_series(target_dir / f"{shot}.npz", sidecar_dir)
    native_time = series.time
    dt = np.diff(native_time)
    if native_time.ndim != 1 or native_time.size < 2:
        raise ValueError(f"shot {shot}: availability audit needs at least two rows")
    cadence = modal_cadence(native_time)
    rows = []
    for level in selected:
        windows = scored_windows(native_time, level)
        lengths = [window.window_end - window.window_start for window in windows]
        history = sum(
            int(series.history_valid[window.window_start:window.window_end].sum())
            for window in windows
        )
        rows.append({
            "shot": shot,
            "role": role,
            "context": level.label,
            "n_rows": int(native_time.size),
            "cadence_seconds": float(cadence),
            "max_gap_seconds": float(dt.max()),
            "gap_rows_over_1p5x_cadence": int((dt > 1.5 * cadence).sum()),
            "visible_tokens_mean": float(np.mean(lengths)),
            "visible_tokens_min": int(min(lengths)),
            "visible_tokens_max": int(max(lengths)),
            "full_horizon_fraction": float(
                np.mean((native_time - native_time[0]) >= level.seconds)),
            "score_rows": int(series.score_valid.sum()),
            "history_valid_coverage": history / float(sum(lengths)),
        })
    return rows


def build_availability_audit_identity(
    *,
    split_path,
    split,
    target_dir,
    sidecar_dir,
    availability_snapshot,
    project_root,
):
    """Build identity fields from one already validated CSV snapshot."""
    _train, _validation, expected_shots = _expected_shots(split)
    if not isinstance(availability_snapshot, AvailabilityCsvSnapshot):
        raise TypeError("availability_snapshot must be AvailabilityCsvSnapshot")
    audited_shots = availability_snapshot.audited_shots
    if audited_shots != expected_shots:
        raise AvailabilityAuditIdentityError(
            "availability audit snapshot membership differs from the frozen "
            "train+validation shots")
    project_root = pathlib.Path(project_root)
    target_dir = pathlib.Path(target_dir)
    sidecar_dir = pathlib.Path(sidecar_dir)
    contexts = [level.label for level in CONTEXT_LEVELS]
    return {
        "schema": AVAILABILITY_AUDIT_SCHEMA,
        "audit_generation_version": AVAILABILITY_AUDIT_GENERATION_VERSION,
        "split_sha256": sha256_file(split_path),
        "expected_shot_count": len(expected_shots),
        "audited_shot_count": len(audited_shots),
        "expected_shots_sha256": _canonical_shot_sha256(expected_shots),
        "audited_shots_sha256": _canonical_shot_sha256(audited_shots),
        "context_count": len(contexts),
        "contexts": contexts,
        "expected_row_count": PUBLICATION_AUDIT_ROWS,
        "audited_row_count": len(availability_snapshot.rows),
        "target_meta_sha256": sha256_file(target_dir / "meta.json"),
        "sidecar_meta_sha256": sha256_file(sidecar_dir / "meta.json"),
        "context_availability_csv_sha256": availability_snapshot.sha256,
        "source_sha256": sha256_tree(
            project_root, include=existing_source_files(project_root)),
    }


def canonical_availability_audit_bytes(identity):
    """Canonical UTF-8 JSON bytes for identity-last publication."""
    return (json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("utf-8")


def _identity_from_bytes(payload):
    try:
        identity = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AvailabilityAuditIdentityError(
            f"cannot parse availability audit identity: {exc}") from exc
    if (not isinstance(identity, Mapping)
            or set(identity) != AVAILABILITY_AUDIT_FIELDS):
        raise AvailabilityAuditIdentityError(
            "availability audit identity must contain exactly the versioned "
            "schema fields")
    return dict(identity)


def validate_availability_audit_identity(
    identity_path,
    *,
    split_path,
    split,
    target_dir,
    sidecar_dir,
    project_root,
):
    """Rebuild and compare the selected split/data/source audit identity."""
    identity_path = pathlib.Path(identity_path)
    identity_payload = _read_nofollow_regular_bytes(
        identity_path, label="availability audit identity")
    actual = _identity_from_bytes(identity_payload)
    if actual.get("schema") != AVAILABILITY_AUDIT_SCHEMA:
        raise AvailabilityAuditIdentityError(
            "availability audit identity schema is not the Work 3 publication "
            "schema")
    if (actual.get("audit_generation_version")
            != AVAILABILITY_AUDIT_GENERATION_VERSION):
        raise AvailabilityAuditIdentityError(
            "availability audit identity generation version changed")
    csv_snapshot = read_availability_csv_snapshot(
        identity_path.parent / "context_availability.csv", split)
    expected = build_availability_audit_identity(
        split_path=split_path,
        split=split,
        target_dir=target_dir,
        sidecar_dir=sidecar_dir,
        availability_snapshot=csv_snapshot,
        project_root=project_root,
    )
    differing = [
        field for field in sorted(AVAILABILITY_AUDIT_FIELDS)
        if actual.get(field) != expected.get(field)
    ]
    if differing:
        raise AvailabilityAuditIdentityError(
            "availability audit identity disagrees with current publication "
            f"inputs ({', '.join(differing)})")
    return ValidatedAvailabilityAudit(
        identity=actual,
        payload=identity_payload,
        sha256=hashlib.sha256(identity_payload).hexdigest(),
        csv=csv_snapshot,
    )


# ── Work 2 final dependency consumed by the Work 3 final transaction ──
WORK3_TRANSACTION_SCHEMA = "pf_context_final_transaction_v1"
WORK3_TRANSACTION_GENERATION_VERSION = 1
WORK3_TRANSACTION_FIELDS = {
    "schema",
    "transaction_generation_version",
    "split",
    "validation_selection",
    "availability_audit",
    "work2",
    "floor",
    "references",
    "matrix",
    "input_hashes",
    "cells",
}


class Work2DependencyError(ValueError):
    """Raised when Work 2's final marker/floor cannot authorize Work 3."""


class TransactionProvenanceError(ValueError):
    """Raised when final transaction provenance is malformed or noncanonical."""


@dataclasses.dataclass(frozen=True)
class Work2Dependency:
    """One validated canonical Work 2 marker and its exact floor snapshot."""

    marker: Mapping[str, object]
    marker_sha256: str
    floor: Mapping[str, object]
    floor_sha256: str


@dataclasses.dataclass(frozen=True)
class FinalReadiness:
    """The six immutable digests authorizing the Work 3 test transaction."""

    split_disclosure: Mapping[str, object]
    validation_selection_sha256: str
    availability_audit_sha256: str
    work2_marker_sha256: str
    floor_sha256: str
    transaction_sha256: str


def _work2_marker_from_bytes(payload):
    from .pfobs_provenance import (  # pylint: disable=import-outside-toplevel
        _validate_work2_final_marker_schema,
        canonical_work2_final_marker_bytes,
    )

    try:
        marker = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Work2DependencyError(
            f"cannot parse canonical Work 2 final marker: {exc}") from exc
    try:
        _validate_work2_final_marker_schema(marker)
    except ValueError as exc:
        raise Work2DependencyError(str(exc)) from exc
    if bytes(payload) != canonical_work2_final_marker_bytes(marker):
        raise Work2DependencyError(
            "Work 2 final marker bytes are not canonical schema-v2 bytes")
    return dict(marker)


def _strict_floor_positive_int(value, *, label):
    if isinstance(value, bool):
        raise Work2DependencyError(f"Work 2 floor {label} must be positive")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Work2DependencyError(
            f"Work 2 floor {label} must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise Work2DependencyError(
            f"Work 2 floor {label} must be a positive integer")
    return parsed


def _work2_floor_from_bytes(payload, *, expected_shots):
    from .pfobs_provenance import (  # pylint: disable=import-outside-toplevel
        PUBLICATION_FINAL_TEST_SHOTS,
        REPRESENTATION_FLOOR_HEADER,
    )

    payload = bytes(payload)
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise Work2DependencyError("Work 2 floor must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != REPRESENTATION_FLOOR_HEADER:
        raise Work2DependencyError(
            "Work 2 floor header is not the frozen representation-floor contract")
    expected = tuple(int(shot) for shot in expected_shots)
    if (len(expected) != PUBLICATION_FINAL_TEST_SHOTS
            or len(set(expected)) != PUBLICATION_FINAL_TEST_SHOTS):
        raise Work2DependencyError(
            "Work 3 publication split must contain exactly 76 unique test shots")
    rows = list(reader)
    if len(rows) != len(expected):
        raise Work2DependencyError(
            f"Work 2 floor must contain exactly {len(expected)} rows")
    parsed_shots = []
    total_slices = 0
    for row_index, row in enumerate(rows, start=2):
        if None in row or set(row) != set(REPRESENTATION_FLOOR_HEADER):
            raise Work2DependencyError(
                f"Work 2 floor row {row_index} has the wrong schema")
        try:
            shot = int(row["shot"])
        except (TypeError, ValueError) as exc:
            raise Work2DependencyError(
                f"Work 2 floor row {row_index} shot is not an integer") from exc
        if str(shot) != str(row["shot"]).strip():
            raise Work2DependencyError(
                f"Work 2 floor row {row_index} shot is not canonical")
        parsed_shots.append(shot)
        total_slices += _strict_floor_positive_int(
            row["n_slices"], label=f"row {row_index} n_slices")
        for field in REPRESENTATION_FLOOR_HEADER[2:]:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise Work2DependencyError(
                    f"Work 2 floor row {row_index} {field} is not numeric") \
                    from exc
            if not math.isfinite(value):
                raise Work2DependencyError(
                    f"Work 2 floor row {row_index} {field} is non-finite")
    if len(set(parsed_shots)) != len(parsed_shots):
        raise Work2DependencyError("Work 2 floor contains duplicate shot rows")
    if tuple(parsed_shots) != expected:
        missing = sorted(set(expected) - set(parsed_shots))
        extra = sorted(set(parsed_shots) - set(expected))
        raise Work2DependencyError(
            "Work 2 floor shots differ from the Work 3 frozen test shots "
            f"(missing={missing[:5]}, extra={extra[:5]})")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "header": list(REPRESENTATION_FLOOR_HEADER),
        "n_shots": len(rows),
        "shots": parsed_shots,
        "shots_sha256": _canonical_shot_sha256(parsed_shots),
        "n_slices": total_slices,
    }


def _work2_current_disclosure(split, split_path):
    from .publication_split import (  # pylint: disable=import-outside-toplevel
        canonical_shot_list_sha256,
        publication_disclosure,
    )

    disclosure = publication_disclosure(split)
    disclosure.update({
        "manifest_sha256": sha256_file(split_path),
        "test_shots_sha256": canonical_shot_list_sha256(split.test),
        "excluded_shots_sha256": canonical_shot_list_sha256(split.excluded),
    })
    return disclosure


def _canonical_identity_bytes(value):
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Work2DependencyError(
            "Work 2 dependency contains noncanonical JSON data") from exc


def validate_work2_dependency(
        marker_path, floor_path, *, split, split_path, target_dir, sidecar_dir):
    """Validate Work 2 marker v2 and its exact 76-row floor from byte snapshots.

    The canonical marker, current publication disclosure, dataset/metadata/strata
    hashes, selected floor location, floor bytes, and marker-recorded floor hash
    must all agree. No target NPZ is opened by this dependency gate.
    """
    marker_path = pathlib.Path(marker_path)
    floor_path = pathlib.Path(floor_path)
    try:
        marker_payload = _read_nofollow_regular_bytes(
            marker_path, label="Work 2 final marker")
    except AvailabilityAuditIdentityError as exc:
        raise Work2DependencyError(str(exc)) from exc
    marker = _work2_marker_from_bytes(marker_payload)
    expected_disclosure = _work2_current_disclosure(split, split_path)
    if _canonical_identity_bytes(marker.get("split")) != \
            _canonical_identity_bytes(expected_disclosure):
        raise Work2DependencyError(
            "Work 2 marker split/exclusions/disclosure differ from Work 3")

    target_dir = pathlib.Path(target_dir)
    sidecar_dir = pathlib.Path(sidecar_dir)
    expected_dataset = {
        "target_meta_sha256": sha256_file(target_dir / "meta.json"),
        "sidecar_meta_sha256": sha256_file(sidecar_dir / "meta.json"),
        "shot_metadata_sha256": sha256_file(split.shot_metadata),
        "slice_strata_sha256": sha256_tree(split.slice_strata_dir),
    }
    if _canonical_identity_bytes(marker.get("dataset")) != \
            _canonical_identity_bytes(expected_dataset):
        raise Work2DependencyError(
            "Work 2 marker data/metadata/strata hashes differ from Work 3")

    floor_record = marker.get("representation_floor")
    if not isinstance(floor_record, Mapping):
        raise Work2DependencyError("Work 2 marker floor disclosure is absent")
    expected_floor_path = marker_path.parent / str(floor_record.get("path", ""))
    if os.path.abspath(floor_path) != os.path.abspath(expected_floor_path):
        raise Work2DependencyError(
            "Work 2 floor path is not the marker-recorded final floor")
    try:
        floor_payload = _read_nofollow_regular_bytes(
            floor_path, label="Work 2 representation floor")
    except AvailabilityAuditIdentityError as exc:
        raise Work2DependencyError(str(exc)) from exc
    floor = _work2_floor_from_bytes(floor_payload, expected_shots=split.test)
    recorded_floor = {
        key: floor_record.get(key)
        for key in ("sha256", "header", "n_shots", "shots",
                    "shots_sha256", "n_slices")
    }
    if _canonical_identity_bytes(floor) != \
            _canonical_identity_bytes(recorded_floor):
        raise Work2DependencyError(
            "Work 2 representation floor bytes disagree with the final marker")
    marker_sha256 = hashlib.sha256(marker_payload).hexdigest()
    return Work2Dependency(
        marker=marker,
        marker_sha256=marker_sha256,
        floor=floor,
        floor_sha256=floor["sha256"],
    )


def validate_final_reference_transition(selection, references):
    """Require false-at-selection plus current true with unchanged records."""
    if not isinstance(selection, Mapping):
        raise RuntimeError("validation selection must be a mapping")
    selected_references = selection.get("references")
    if not isinstance(selected_references, Mapping):
        raise RuntimeError("validation selection references are absent")
    if selected_references.get("selection_lifecycle") != "false_at_selection":
        raise RuntimeError(
            "validation selection must carry false_at_selection lifecycle")
    if not isinstance(references, Mapping) or references.get("valid") is not True:
        raise RuntimeError("current external reference records are invalid")
    if references.get("frozen_before_final_test") is not True:
        raise RuntimeError(
            "final readiness requires frozen_before_final_test exactly true")
    selected_hash = selected_references.get("records_sha256")
    current_hash = references.get("records_sha256")
    if selected_hash != current_hash:
        raise RuntimeError(
            "external reference records changed after validation selection")
    return {
        "selection_lifecycle": "false_at_selection",
        "records_sha256": current_hash,
        "frozen_before_final_test": True,
    }


def build_transaction_provenance(
        *, split_disclosure, validation_selection, availability_audit, work2,
        floor, references, matrix, input_hashes, cells):
    """Build timestamp-free canonical provenance; its hash is stored outside it."""
    payload = {
        "schema": WORK3_TRANSACTION_SCHEMA,
        "transaction_generation_version": WORK3_TRANSACTION_GENERATION_VERSION,
        "split": dict(split_disclosure),
        "validation_selection": dict(validation_selection),
        "availability_audit": dict(availability_audit),
        "work2": dict(work2),
        "floor": dict(floor),
        "references": dict(references),
        "matrix": dict(matrix),
        "input_hashes": dict(input_hashes),
        "cells": [dict(cell) for cell in cells],
    }
    if set(payload) != WORK3_TRANSACTION_FIELDS:
        raise TransactionProvenanceError(
            "transaction provenance schema construction failed")
    canonical_transaction_provenance_bytes(payload)
    return payload


def _require_type_strict_json_equal(actual, expected, *, path):
    if type(actual) is not type(expected):
        raise TransactionProvenanceError(
            f"{path} type changed from {type(expected).__name__} to "
            f"{type(actual).__name__}")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise TransactionProvenanceError(f"{path} mapping keys changed")
        for key in sorted(expected):
            _require_type_strict_json_equal(
                actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise TransactionProvenanceError(f"{path} list length changed")
        for index, (actual_item, expected_item) in enumerate(
                zip(actual, expected)):
            _require_type_strict_json_equal(
                actual_item, expected_item, path=f"{path}[{index}]")
        return
    if actual != expected:
        raise TransactionProvenanceError(f"{path} value changed")


def canonical_transaction_provenance_bytes(payload):
    """Canonical finite UTF-8 transaction JSON with one final newline."""
    if (not isinstance(payload, Mapping)
            or set(payload) != WORK3_TRANSACTION_FIELDS):
        raise TransactionProvenanceError(
            "transaction provenance must contain exactly the versioned fields")
    if (type(payload.get("schema")) is not str
            or payload.get("schema") != WORK3_TRANSACTION_SCHEMA
            or type(payload.get("transaction_generation_version")) is not int
            or payload.get("transaction_generation_version")
            != WORK3_TRANSACTION_GENERATION_VERSION):
        raise TransactionProvenanceError(
            "transaction provenance schema/generation type changed")
    if "transaction_sha256" in payload:
        raise TransactionProvenanceError(
            "transaction provenance must not contain its own hash")
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TransactionProvenanceError(
            "transaction provenance is not deterministic finite JSON") from exc
    return (encoded + "\n").encode("utf-8")


def transaction_provenance_sha256(payload):
    return hashlib.sha256(
        canonical_transaction_provenance_bytes(payload)).hexdigest()


def validate_transaction_provenance_bytes(payload, *, expected=None):
    """Parse canonical transaction bytes and optionally require exact content."""
    try:
        parsed = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TransactionProvenanceError(
            f"cannot parse transaction provenance: {exc}") from exc
    canonical = canonical_transaction_provenance_bytes(parsed)
    if bytes(payload) != canonical:
        raise TransactionProvenanceError(
            "transaction provenance bytes are not canonical")
    if expected is not None:
        _require_type_strict_json_equal(
            parsed, expected, path="transaction_provenance")
    return parsed
