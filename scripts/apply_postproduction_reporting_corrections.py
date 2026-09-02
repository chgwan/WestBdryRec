#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay Task 16 generated-output corrections after Task 17 curation.

Version 3 chains the immutable version-2 correction manifest, authenticates the
current Task 17 Work 2 document as a non-owned successor, and corrects only
regenerated generated outputs. It never rewrites the curated document.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.pf_context import (  # noqa: E402
    existing_source_files as work3_source_files,
)
from src.ml.pf_observability import (  # noqa: E402
    existing_source_files as work2_source_files,
    sha256_tree,
)
from src.utils import (  # noqa: E402
    PublicationIOError,
    durable_publish_bytes,
    read_regular_nofollow,
)


SCRIPT_RELATIVE_PATH = "scripts/apply_postproduction_reporting_corrections.py"
PRIOR_MANIFEST_RELATIVE_PATH = (
    "ProjDB/Stats/postproduction_reporting_corrections/"
    "task16_reporting_corrections.json"
)
PRIOR_MANIFEST_SHA256 = (
    "37ef788b0856b800d697919c7cfa74d426e82480a886721e2e8f5599123e5e67"
)
PRIOR_MANIFEST_SCHEMA = "task16_postproduction_reporting_corrections_v2"
MANIFEST_RELATIVE_PATH = (
    "ProjDB/Stats/postproduction_reporting_corrections/"
    "task16_reporting_corrections_v3.json"
)
MANIFEST_SCHEMA = "task16_postproduction_reporting_corrections_v3"
MANIFEST_GENERATION_VERSION = 3

CONTEXT_CURVE_PATH = "ProjDB/Stats/pf_context/generated/context_test_curve.csv"
FIGURE_DATA_PATH = "ProjDB/Stats/pf_context/generated/figure_data.json"
WORK2_SUMMARY_PATH = "ProjDB/Stats/pf_observability/generated/analysis_summary.md"
WORK2_CURATED_SUCCESSOR_PATH = "docs/pf_observability_results.md"
WORK2_CURATED_SUCCESSOR_SHA256 = (
    "96e542a1761294173f191161435329d2d08c99963194071b7efdc792be60c62f"
)

EXPECTED_HASHES = {
    CONTEXT_CURVE_PATH: {
        "before": "08be25e252a76822a127517df600155d05330e1b592f7abef799336c3f50fbe4",
        "after": "2419e16b0725082b622ffa6bdbf6f4c5826e369149e090b5cafd10d475b1482c",
    },
    FIGURE_DATA_PATH: {
        "before": "d15cef57484e02e1eebedbe28057096a75c77b9c227f69bb31f063b9bf299e33",
        "after": "0b29d9465eb5552ee565e19ef4f7ecbcbe059d90521b734dc43db79c2946fab3",
    },
    WORK2_SUMMARY_PATH: {
        "before": "87ed35a9294e218a661b772536dcb5ccf5c42c8b95d71f054ab2af7029dec26c",
        "after": "ee688bcf29abe46836546505a4f68510b137e6acee88529544a6574913b32ef8",
    },
}

IMMUTABLE_INPUTS = {
    "work2_validation_decision": {
        "path": "ProjDB/Stats/pf_observability/validation_decision.json",
        "sha256": "e158b5df2c66884adf7345cd154a5a9a5789139d221d08986775149441ed9bea",
    },
    "work2_final_marker": {
        "path": "ProjDB/Stats/pf_observability/FINAL_TEST_EVALUATED.json",
        "sha256": "605001faa75be8dcc03d00430e68b15ecea542178c5677a860ce18228c7903c4",
    },
    "work2_representation_floor": {
        "path": (
            "ProjDB/Stats/pf_observability/final_test/"
            "representation_floor_per_shot.csv"
        ),
        "sha256": "e7a2bc48df12225d567fe0c28f091b0e4d0a5fcb239948f3f05764008988a13a",
    },
    "work3_validation_selection": {
        "path": "ProjDB/Stats/pf_context/validation_selection.json",
        "sha256": "798248947166a13883743f501056fe6b6fc9757bc80a654e6f28489bd0290a6f",
    },
    "work3_final_transaction": {
        "path": "ProjDB/Stats/pf_context/final_test/transaction_provenance.json",
        "sha256": "14c9d5b9be89e8f486bad1cc3211125f94822d290aa328d7ba91e92d7448361a",
    },
    "work3_final_marker": {
        "path": "ProjDB/Stats/pf_context/PFCTX_FINAL_TEST_EVALUATED.json",
        "sha256": "af24ba9672e4e84a6af92063c18c6539753da779ee452d565ead6ef25bc40c47",
    },
    "work2_generated_verdicts": {
        "path": "ProjDB/Stats/pf_observability/generated/verdicts.json",
        "sha256": "4aba9d86a686891a95886e725d0e1ed1126be3411f0d95cc52c0e1f34eae8317",
    },
    "work3_final_verdict": {
        "path": "ProjDB/Stats/pf_context/generated/final_verdict.json",
        "sha256": "443ac9d74169c4c9d86a06a89b28a15a3de0ee8e98fd12770a44c14cd0dda119",
    },
    "references_file": {
        "path": "configs/pf_timescale_references.yml",
        "sha256": "0498dd65fb15d4f71eda18e653a28e836c4841d89eff2477432a1183abe6bc11",
    },
}

# These hashes authenticate the separately versioned hardened future-run source,
# not the historical source hashes embedded in the immutable v1 final markers.
EXPECTED_V2_SOURCE_CLOSURES = {
    "work2": "1c7d9b0b4b035fb979bd329a74461216251ed7644e9257408fe56a55936b1caa",
    "work3": "e5fab7e173192d3ee2d9901079e48c7e8009c9169a3047e575c13b894a2ee54e",
}

CURVE_HEADER = (
    "context", "seconds", "n_shots", "median_mm", "ci95_lo_mm",
    "ci95_hi_mm", "role",
)
CURVE_ORDER = (
    "h2048", "h0001", "h0032", "h0128", "h0256", "h0512", "h1024",
)
OLD_WORK2_SCOPE = (
    "Results describe predictive observer-memory timescale evidence conditional "
    "on the `GMAG_BND` reconstruction product."
)
NEW_WORK2_SCOPE = (
    "Results describe controller-accessible magnetic-signal observability and "
    "current-input evidence conditional on the `GMAG_BND` reconstruction product."
)


class CorrectionError(RuntimeError):
    """The version-3 correction cannot authenticate current state."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def canonical_manifest_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, allow_nan=False,
            indent=2, sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CorrectionError("correction manifest is not finite JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _read(path: pathlib.Path, *, label: str) -> bytes:
    try:
        return read_regular_nofollow(path, label=label)
    except PublicationIOError as exc:
        raise CorrectionError(str(exc)) from exc


def _json_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise CorrectionError(f"{label} must be a JSON mapping")
    return parsed


def _open_real_directory(path: pathlib.Path, *, label: str) -> int:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CorrectionError(f"cannot open {label} at {path}: {exc}") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise CorrectionError(f"{label} at {path} must be a real no-follow directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CorrectionError(
            f"{label} at {path} must be a real no-follow directory") from exc
    opened = os.fstat(descriptor)
    if (not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)):
        os.close(descriptor)
        raise CorrectionError(f"{label} at {path} changed while opening")
    return descriptor


def _read_leaf_at(directory_fd: int, name: str, *, label: str):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise CorrectionError(f"{label} must be a regular no-follow file") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CorrectionError(f"{label} must be a regular no-follow file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), info
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(bytes(payload))
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while staging reporting correction")
        view = view[written:]


def _atomic_replace_regular(
        path: pathlib.Path, *, expected_current: bytes,
        replacement: bytes, original_mode: int) -> None:
    directory_fd = _open_real_directory(
        path.parent, label="reporting correction parent")
    temporary = (
        f".task16v3.{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    descriptor = None
    try:
        current, current_info = _read_leaf_at(
            directory_fd, path.name, label=f"reporting member {path.name}")
        if current != expected_current:
            raise CorrectionError(f"{path}: bytes changed after correction preflight")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            temporary, flags, stat.S_IMODE(original_mode), dir_fd=directory_fd)
        _write_all(descriptor, replacement)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        staged, _staged_info = _read_leaf_at(
            directory_fd, temporary, label="staged reporting correction")
        if staged != replacement:
            raise CorrectionError(f"{path}: staged correction bytes changed")
        late, late_info = _read_leaf_at(
            directory_fd, path.name, label=f"reporting member {path.name}")
        if (late != expected_current
                or (late_info.st_dev, late_info.st_ino)
                != (current_info.st_dev, current_info.st_ino)):
            raise CorrectionError(f"{path}: reporting member changed before commit")
        os.replace(
            temporary, path.name,
            src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except CorrectionError:
        raise
    except OSError as exc:
        raise CorrectionError(f"cannot publish reporting correction {path}: {exc}") \
            from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _parse_curve(payload: bytes):
    try:
        reader = csv.DictReader(io.StringIO(
            payload.decode("utf-8"), newline=""))
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise CorrectionError("context test curve is not valid UTF-8 CSV") from exc
    if tuple(reader.fieldnames or ()) != CURVE_HEADER:
        raise CorrectionError("context test curve header changed")
    if tuple(row.get("context") for row in rows) != CURVE_ORDER:
        raise CorrectionError("context test curve order changed")
    for row in rows:
        if set(row) != set(CURVE_HEADER) or row["n_shots"] != "76":
            raise CorrectionError("context test curve schema/count changed")
        for field in CURVE_HEADER[1:6]:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise CorrectionError("context test curve value is not numeric") \
                    from exc
            if not math.isfinite(value):
                raise CorrectionError("context test curve contains non-finite data")
    return rows


def _curve_nonrole_sha256(rows) -> str:
    stripped = []
    for row in rows:
        item = dict(row)
        item.pop("role")
        stripped.append(item)
    return sha256_bytes(json.dumps(
        stripped, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))


def _correct_context_curve(payload: bytes):
    before = _parse_curve(payload)
    roles = {row["context"]: row["role"] for row in before}
    if roles["h0512"] != "confirmatory" \
            or roles["h0256"] not in ("confirmatory", "descriptive"):
        raise CorrectionError("context roles are not the reviewed before/after state")
    output = []
    changed = 0
    for line in payload.splitlines(keepends=True):
        if line.startswith(b"h0256,") and line.endswith(b",confirmatory\n"):
            line = line[:-len(b"confirmatory\n")] + b"descriptive\n"
            changed += 1
        output.append(line)
    corrected = b"".join(output)
    if roles["h0256"] == "confirmatory" and changed != 1:
        raise CorrectionError("h0256 role could not be changed exactly once")
    after = _parse_curve(corrected)
    after_roles = {row["context"]: row["role"] for row in after}
    before_nonrole = _curve_nonrole_sha256(before)
    after_nonrole = _curve_nonrole_sha256(after)
    if (after_roles["h0512"], after_roles["h0256"]) != (
            "confirmatory", "descriptive") or before_nonrole != after_nonrole:
        raise CorrectionError("context correction changed hierarchy/numeric data")
    return corrected, {
        "nonrole_sha256_before": before_nonrole,
        "nonrole_sha256_after": after_nonrole,
    }


def _parse_figure(payload: bytes):
    data = _json_mapping(payload, label="figure data")
    if payload != canonical_manifest_bytes(data):
        raise CorrectionError("figure data bytes are not canonical JSON")
    curve = data.get("final_curve")
    if not isinstance(curve, list) or tuple(
            row.get("context") for row in curve) != CURVE_ORDER:
        raise CorrectionError("figure final curve schema/order changed")
    confirmation = data.get("confirmation")
    if (not isinstance(confirmation, dict)
            or confirmation.get("selected") != "h0512"
            or confirmation.get("predecessor") != "h0256"
            or confirmation.get("confirmed") is not False
            or confirmation.get("verdict") != "failed_to_confirm"):
        raise CorrectionError("figure confirmation no longer records failed h0512")
    return data


def _figure_nonrole_sha256(data) -> str:
    stripped = copy.deepcopy(data)
    for row in stripped["final_curve"]:
        row.pop("role", None)
    return sha256_bytes(json.dumps(
        stripped, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))


def _correct_figure_data(payload: bytes):
    before = _parse_figure(payload)
    roles = {row["context"]: row.get("role") for row in before["final_curve"]}
    if roles["h0512"] != "confirmatory" \
            or roles["h0256"] not in ("confirmatory", "descriptive"):
        raise CorrectionError("figure roles are not the reviewed before/after state")
    after = copy.deepcopy(before)
    rows = [row for row in after["final_curve"] if row["context"] == "h0256"]
    if len(rows) != 1:
        raise CorrectionError("figure data must contain one h0256 row")
    rows[0]["role"] = "descriptive"
    corrected = canonical_manifest_bytes(after)
    parsed = _parse_figure(corrected)
    before_nonrole = _figure_nonrole_sha256(before)
    after_nonrole = _figure_nonrole_sha256(parsed)
    if before_nonrole != after_nonrole:
        raise CorrectionError("figure correction changed numerical/order data")
    return corrected, {
        "nonrole_sha256_before": before_nonrole,
        "nonrole_sha256_after": after_nonrole,
    }


def _summary_non_scope_sha256(payload: bytes) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise CorrectionError("Work 2 analysis summary must be UTF-8") from exc
    normalized = text.replace(OLD_WORK2_SCOPE, "<WORK2_SCOPE>")
    normalized = normalized.replace(NEW_WORK2_SCOPE, "<WORK2_SCOPE>")
    return sha256_bytes(normalized.encode("utf-8"))


def _correct_work2_summary(payload: bytes):
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise CorrectionError("Work 2 analysis summary must be UTF-8") from exc
    old_count = text.count(OLD_WORK2_SCOPE)
    new_count = text.count(NEW_WORK2_SCOPE)
    if (old_count, new_count) == (1, 0):
        corrected = text.replace(OLD_WORK2_SCOPE, NEW_WORK2_SCOPE).encode("utf-8")
    elif (old_count, new_count) == (0, 1):
        corrected = payload
    else:
        raise CorrectionError(
            "Work 2 summary does not contain one exact before or after sentence")
    before_non_scope = _summary_non_scope_sha256(payload)
    after_non_scope = _summary_non_scope_sha256(corrected)
    if before_non_scope != after_non_scope:
        raise CorrectionError("Work 2 summary changed outside its scope sentence")
    return corrected, {
        "non_scope_sha256_before": before_non_scope,
        "non_scope_sha256_after": after_non_scope,
    }


TRANSFORMS = {
    CONTEXT_CURVE_PATH: _correct_context_curve,
    FIGURE_DATA_PATH: _correct_figure_data,
    WORK2_SUMMARY_PATH: _correct_work2_summary,
}


def _authenticate_exact(path: pathlib.Path, expected: str, *, label: str) -> bytes:
    payload = _read(path, label=label)
    actual = sha256_bytes(payload)
    if actual != expected:
        raise CorrectionError(
            f"{path}: unexpected SHA-256 {actual}; expected immutable {expected}")
    return payload


def _authenticate_predecessor(project_root: pathlib.Path) -> dict[str, object]:
    payload = _authenticate_exact(
        project_root / PRIOR_MANIFEST_RELATIVE_PATH,
        PRIOR_MANIFEST_SHA256,
        label="version-2 correction manifest",
    )
    parsed = _json_mapping(payload, label="version-2 correction manifest")
    if parsed.get("schema") != PRIOR_MANIFEST_SCHEMA:
        raise CorrectionError("version-2 correction manifest schema changed")
    return {
        "path": PRIOR_MANIFEST_RELATIVE_PATH,
        "sha256": PRIOR_MANIFEST_SHA256,
        "schema": PRIOR_MANIFEST_SCHEMA,
    }


def _authenticate_curated_successor(project_root: pathlib.Path) -> bytes:
    return _authenticate_exact(
        project_root / WORK2_CURATED_SUCCESSOR_PATH,
        WORK2_CURATED_SUCCESSOR_SHA256,
        label="Task 17 curated Work 2 non-owned successor",
    )


def _authenticate_immutables(project_root: pathlib.Path) -> dict[str, bytes]:
    payloads = {}
    for label, record in sorted(IMMUTABLE_INPUTS.items()):
        payloads[label] = _authenticate_exact(
            project_root / record["path"], record["sha256"],
            label=label.replace("_", " "),
        )
    return payloads


def _recompute_source_closures(project_root: pathlib.Path) -> dict[str, str]:
    try:
        work2 = sha256_tree(
            project_root, include=work2_source_files(project_root))
    except Exception as exc:
        raise CorrectionError(f"cannot recompute Work 2 source closure: {exc}") \
            from exc
    try:
        work3 = sha256_tree(
            project_root, include=work3_source_files(project_root))
    except Exception as exc:
        raise CorrectionError(f"cannot recompute Work 3 source closure: {exc}") \
            from exc
    current = {"work2": work2, "work3": work3}
    if current != EXPECTED_V2_SOURCE_CLOSURES:
        raise CorrectionError(
            "hardened v2 source closure drifted "
            f"({current} != {EXPECTED_V2_SOURCE_CLOSURES})")
    return current


def _manifest_state(path: pathlib.Path) -> bytes | None:
    if not os.path.lexists(path):
        return None
    return _read(path, label="version-3 correction manifest")


def _build_manifest(
        *, script_sha256: str, predecessor: Mapping[str, object],
        source_closures: Mapping[str, str],
        invariants: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    corrections = {}
    for relative in sorted(EXPECTED_HASHES):
        if relative == CONTEXT_CURVE_PATH:
            change = (
                "h0512 remains stage-1 confirmatory and failed; h0256 is "
                "descriptive because stage 2 was not reached"
            )
        elif relative == FIGURE_DATA_PATH:
            change = (
                "copy reviewed hierarchical roles into generated figure metadata "
                "without changing numerical/order content"
            )
        else:
            change = (
                "restore magnetic-signal observability/current-input scope wording "
                "after Work 2 generated-summary regeneration"
            )
        corrections[relative] = {
            "before_sha256": EXPECTED_HASHES[relative]["before"],
            "after_sha256": EXPECTED_HASHES[relative]["after"],
            "change": change,
        }
    curve = invariants[CONTEXT_CURVE_PATH]
    figure = invariants[FIGURE_DATA_PATH]
    summary = invariants[WORK2_SUMMARY_PATH]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "correction_generation_version": MANIFEST_GENERATION_VERSION,
        "script": {
            "path": SCRIPT_RELATIVE_PATH,
            "sha256": script_sha256,
        },
        "predecessor_manifest": dict(predecessor),
        "curated_successor": {
            "path": WORK2_CURATED_SUCCESSOR_PATH,
            "sha256": WORK2_CURATED_SUCCESSOR_SHA256,
            "ownership": "authenticated_non_owned_successor",
            "must_remain_byte_identical": True,
        },
        "corrections": corrections,
        "immutable_inputs": copy.deepcopy(IMMUTABLE_INPUTS),
        "source_closures": dict(source_closures),
        "numerical_invariance": {
            "context_test_curve_nonrole_sha256_before": curve[
                "nonrole_sha256_before"],
            "context_test_curve_nonrole_sha256_after": curve[
                "nonrole_sha256_after"],
            "figure_data_nonrole_sha256_before": figure[
                "nonrole_sha256_before"],
            "figure_data_nonrole_sha256_after": figure[
                "nonrole_sha256_after"],
            "work2_summary_non_scope_sha256_before": summary[
                "non_scope_sha256_before"],
            "work2_summary_non_scope_sha256_after": summary[
                "non_scope_sha256_after"],
            "all_context_order_shot_count_and_numeric_values_unchanged": True,
            "all_marker_decision_selection_transaction_floor_verdict_bytes_unchanged":
                True,
            "curated_successor_bytes_unchanged": True,
        },
        "hierarchy": {
            "selected": "h0512",
            "selected_stage": 1,
            "selected_role_after": "confirmatory",
            "selected_confirmed": False,
            "final_verdict": "failed_to_confirm",
            "predecessor": "h0256",
            "predecessor_stage_reached": False,
            "predecessor_role_after": "descriptive",
            "post_test_reselection": False,
        },
        "reason": [
            "Regenerated Work 3 outputs again need the reviewed hierarchy labels.",
            "Regenerated Work 2 summary again needs observability/current-input scope.",
            "Task 17 curated Work 2 prose is authenticated but never owned or rewritten.",
        ],
    }
    canonical_manifest_bytes(manifest)
    return manifest


def apply_postproduction_reporting_corrections(
        project_root: pathlib.Path | str = REPO_ROOT) -> dict[str, object]:
    project_root = pathlib.Path(project_root)
    script_payload = _read(
        project_root / SCRIPT_RELATIVE_PATH, label="version-3 correction script")
    script_sha256 = sha256_bytes(script_payload)
    manifest_path = project_root / MANIFEST_RELATIVE_PATH
    existing_manifest = _manifest_state(manifest_path)
    predecessor = _authenticate_predecessor(project_root)
    curated_before = _authenticate_curated_successor(project_root)
    _authenticate_immutables(project_root)
    source_closures = _recompute_source_closures(project_root)

    snapshots = {}
    corrected = {}
    invariants = {}
    states = {}
    for relative in sorted(EXPECTED_HASHES):
        path = project_root / relative
        payload = _read(path, label="generated reporting correction target")
        actual = sha256_bytes(payload)
        expected = EXPECTED_HASHES[relative]
        if actual == expected["before"]:
            states[relative] = "before"
        elif actual == expected["after"]:
            states[relative] = "after"
        else:
            raise CorrectionError(
                f"{relative}: unexpected SHA-256 {actual}; expected exact before "
                f"{expected['before']} or after {expected['after']}")
        output, invariant = TRANSFORMS[relative](payload)
        if sha256_bytes(output) != expected["after"]:
            raise CorrectionError(
                f"{relative}: transform does not produce reviewed after hash")
        snapshots[relative] = (payload, path.lstat())
        corrected[relative] = output
        invariants[relative] = invariant

    manifest = _build_manifest(
        script_sha256=script_sha256,
        predecessor=predecessor,
        source_closures=source_closures,
        invariants=invariants,
    )
    manifest_bytes = canonical_manifest_bytes(manifest)
    if existing_manifest is not None:
        if existing_manifest != manifest_bytes:
            raise CorrectionError("existing version-3 correction manifest bytes differ")
        if any(state != "after" for state in states.values()):
            raise CorrectionError(
                "version-3 manifest exists but a generated member is not after-state")
        return manifest

    for relative in sorted(EXPECTED_HASHES):
        if states[relative] == "after":
            continue
        payload, info = snapshots[relative]
        _atomic_replace_regular(
            project_root / relative,
            expected_current=payload,
            replacement=corrected[relative],
            original_mode=info.st_mode,
        )

    for relative, expected in sorted(EXPECTED_HASHES.items()):
        payload = _read(
            project_root / relative, label="corrected generated reporting member")
        if sha256_bytes(payload) != expected["after"]:
            raise CorrectionError(f"{relative}: corrected after hash changed")
    curated_after = _authenticate_curated_successor(project_root)
    if curated_after != curated_before:
        raise CorrectionError("Task 17 curated successor changed during correction")
    _authenticate_predecessor(project_root)
    _authenticate_immutables(project_root)
    if _recompute_source_closures(project_root) != source_closures:
        raise CorrectionError("hardened v2 source closures changed during correction")
    try:
        durable_publish_bytes(
            manifest_path,
            manifest_bytes,
            state_label="version-3 correction manifest",
        )
    except PublicationIOError as exc:
        raise CorrectionError(str(exc)) from exc
    published = _read(manifest_path, label="version-3 correction manifest")
    if published != manifest_bytes:
        raise CorrectionError("published version-3 correction manifest changed")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--project-root", type=pathlib.Path, default=REPO_ROOT,
        help="project root containing the authenticated completed scientific state",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = apply_postproduction_reporting_corrections(args.project_root)
    manifest_path = pathlib.Path(args.project_root) / MANIFEST_RELATIVE_PATH
    print(f"reporting correction v3 verified: {len(manifest['corrections'])} files")
    for relative, record in sorted(manifest["corrections"].items()):
        print(f"  {relative}: {record['after_sha256']}")
    print(
        f"curated successor preserved: {WORK2_CURATED_SUCCESSOR_PATH} "
        f"({WORK2_CURATED_SUCCESSOR_SHA256})")
    print(f"correction manifest: {manifest_path} ({sha256_bytes(manifest_path.read_bytes())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
