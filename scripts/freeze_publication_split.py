# -*- coding: utf-8 -*-
"""Freeze the deterministic forward-by-shot publication split bundle.

The command has no force or replacement mode. It validates every source and all
precomputed output bytes before writing anything, permits byte-identical partial
resume before the manifest exists, and writes the manifest last.
"""
import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import src.ml.publication_split as publication_split  # noqa: E402
from src.ml.publication_split import (  # noqa: E402
    PUBLICATION_CLAIM_SCOPE,
    PUBLICATION_EXCLUDED,
    PUBLICATION_NAME,
    PUBLICATION_POLICY_SCHEMA,
    PUBLICATION_SPLIT_STRATEGY,
    PUBLICATION_TEST_GROUP_COUNTS,
    PUBLICATION_THRESHOLD_VALUES,
    PUBLICATION_VERSION,
    MetadataThresholds,
    PublicationMembership,
    PublicationValidationError,
    ShotStatistics,
    assign_shot_metadata,
    build_forward_membership,
    canonical_json_bytes,
    canonical_shot_list_sha256,
    compute_metadata_thresholds,
    derive_pilot_test,
    derive_prior_test_access,
    shot_metadata_csv_bytes,
    validate_publication_payloads,
)
from src.ml.pf_observability import TIME_AXIS_TOKEN, sha256_tree  # noqa: E402
from src.utils import (  # noqa: E402
    PersistentFlock, PublicationIOError, durable_publish_bytes,
    read_regular_nofollow,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPECTED_COVERAGE_TOTALS = {
    "train": {"n_rows": 4322025, "n_valid": 4318828, "n_common": 4318828},
    "validation": {
        "n_rows": 1318663, "n_valid": 1318385, "n_common": 1318385,
    },
    "test": {"n_rows": 1391591, "n_valid": 1390655, "n_common": 1390655},
    "excluded": {"n_rows": 181127, "n_valid": 180948, "n_common": 180948},
}
EXPECTED_CORPUS_COVERAGE = {
    "n_rows": 7213406,
    "n_valid": 7208816,
    "n_common": 7208816,
}
MINIMUM_INFERENTIAL_TEST_GROUP_SIZE = 10


@dataclasses.dataclass(frozen=True)
class FreezePaths:
    """All exact source and output paths for one publication project root."""

    project_root: pathlib.Path
    corpus_meta: pathlib.Path
    sidecar_dir: pathlib.Path
    sidecar_meta: pathlib.Path
    coverage_csv: pathlib.Path
    flat_top_csv: pathlib.Path
    heating_csv: pathlib.Path
    pilot_manifest: pathlib.Path
    pilot_marker: pathlib.Path
    manifest_out: pathlib.Path
    metadata_out: pathlib.Path
    policy_out: pathlib.Path

    @classmethod
    def from_project_root(cls, project_root: str | pathlib.Path) -> "FreezePaths":
        root = pathlib.Path(project_root)
        sidecar_dir = root / "ProjDB/datasets/NpzGeomPFObs"
        stats_dir = root / "ProjDB/Stats"
        output_dir = stats_dir / "communications_physics_forward_v1"
        return cls(
            project_root=root,
            corpus_meta=root / "ProjDB/datasets/NpzGeom/meta.json",
            sidecar_dir=sidecar_dir,
            sidecar_meta=sidecar_dir / "meta.json",
            coverage_csv=stats_dir / "pf_observability/coverage.csv",
            flat_top_csv=stats_dir / "flat_top.csv",
            heating_csv=stats_dir / "heating_sustained_stats.csv",
            pilot_manifest=root / "configs/splits/pfobs_random_pilot.json",
            pilot_marker=(
                stats_dir / "pf_observability_pilot/FINAL_TEST_EVALUATED.json"),
            manifest_out=(
                root / "configs/splits/communications_physics_campaign_v1.json"),
            metadata_out=output_dir / "shot_metadata.csv",
            policy_out=output_dir / "slice_strata/strata_policy.json",
        )

    @property
    def input_paths(self) -> tuple[pathlib.Path, ...]:
        return (
            self.corpus_meta,
            self.sidecar_meta,
            self.coverage_csv,
            self.flat_top_csv,
            self.pilot_manifest,
            self.pilot_marker,
            self.heating_csv,
        )


@dataclasses.dataclass(frozen=True)
class FreezeBundle:
    """Validated payload objects and their exact precomputed publication bytes."""

    manifest: Mapping[str, object]
    metadata_rows: tuple[dict[str, object], ...]
    policy: Mapping[str, object]
    metadata_bytes: bytes
    policy_bytes: bytes
    manifest_bytes: bytes


def _read_json_mapping(path: pathlib.Path, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationValidationError(
            f"cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PublicationValidationError(f"{label} must be a JSON mapping")
    return payload


def _source_sha256(paths: FreezePaths) -> dict[str, str]:
    result = {}
    for path in paths.input_paths:
        try:
            relative = path.relative_to(paths.project_root).as_posix()
            payload = path.read_bytes()
        except (OSError, ValueError) as exc:
            raise PublicationValidationError(
                f"cannot hash publication source {path}: {exc}") from exc
        result[relative] = hashlib.sha256(payload).hexdigest()
    return result


def _source_identity(paths: FreezePaths) -> dict[str, object]:
    expected_tree = publication_split.PUBLICATION_FROZEN_IDENTITY[
        "sidecar_tree"]
    try:
        file_count = sum(
            1 for path in paths.sidecar_dir.rglob("*") if path.is_file())
        tree_hash = sha256_tree(paths.sidecar_dir)
    except OSError as exc:
        raise PublicationValidationError(
            f"cannot hash PF sidecar tree at {paths.sidecar_dir}: {exc}") from exc
    return {
        "source_sha256": _source_sha256(paths),
        "sidecar_tree": {
            "root": expected_tree["root"],
            "algorithm": expected_tree["algorithm"],
            "file_count": file_count,
            "sha256": tree_hash,
        },
    }


def _validate_frozen_source_identity(identity: Mapping[str, object]) -> None:
    expected = publication_split.PUBLICATION_FROZEN_IDENTITY
    if identity["source_sha256"] != expected["source_sha256"]:
        raise PublicationValidationError(
            "publication immutable source hashes do not match the frozen inputs")
    if identity["sidecar_tree"] != expected["sidecar_tree"]:
        raise PublicationValidationError(
            "publication sidecar tree identity does not match the frozen inputs")


def _source_identity_before_gathering(paths: FreezePaths) -> dict[str, object]:
    identity = _source_identity(paths)
    _validate_frozen_source_identity(identity)
    return identity


def _validate_source_identity_after_gathering(
    paths: FreezePaths,
    before: Mapping[str, object],
) -> None:
    after = _source_identity(paths)
    if after != before:
        raise PublicationValidationError(
            "publication sources changed during gathering")
    _validate_frozen_source_identity(after)


def _corpus_rows(
    paths: FreezePaths,
) -> tuple[tuple[int, ...], dict[int, float]]:
    payload = _read_json_mapping(paths.corpus_meta, label="NpzGeom metadata")
    raw_rows = payload.get("shots")
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
        raise PublicationValidationError("NpzGeom metadata shots must be a list")
    shots = []
    durations = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise PublicationValidationError("NpzGeom metadata rows must be mappings")
        try:
            shot = int(raw["shot"])
            duration = float(raw["t_end"]) - float(raw["t_start"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PublicationValidationError(
                "NpzGeom metadata rows require shot, t_start, and t_end") from exc
        if shot in durations:
            raise PublicationValidationError(
                f"NpzGeom metadata contains duplicate shot {shot}")
        if not np.isfinite(duration) or duration <= 0.0:
            raise PublicationValidationError(
                f"NpzGeom metadata shot {shot} has invalid record duration")
        shots.append(shot)
        durations[shot] = duration
    ordered = tuple(sorted(shots))
    if len(ordered) != 759 or (ordered[0], ordered[-1]) != (57281, 58511):
        raise PublicationValidationError(
            "NpzGeom metadata must contain exactly 759 unique shots from "
            "57281 through 58511")
    if payload.get("n_shots") is not None and payload.get("n_shots") != 759:
        raise PublicationValidationError("NpzGeom metadata n_shots must be 759")
    return ordered, durations


def _sidecar_shots(paths: FreezePaths) -> tuple[int, ...]:
    if not paths.sidecar_dir.is_dir():
        raise PublicationValidationError(
            f"PF sidecar directory does not exist: {paths.sidecar_dir}")
    shots = []
    for path in paths.sidecar_dir.glob("*.npz"):
        try:
            shot = int(path.stem)
        except ValueError as exc:
            raise PublicationValidationError(
                f"PF sidecar has non-shot filename {path.name}") from exc
        if str(shot) != path.stem:
            raise PublicationValidationError(
                f"PF sidecar has non-canonical shot filename {path.name}")
        shots.append(shot)
    ordered = tuple(sorted(shots))
    if len(ordered) != len(set(ordered)):
        raise PublicationValidationError("PF sidecar contains duplicate shot stems")
    return ordered


def _read_coverage(
    paths: FreezePaths,
    membership: PublicationMembership,
) -> dict[str, dict[str, int]]:
    try:
        with paths.coverage_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            required = {"shot", "n_rows", "n_valid", "n_common"}
            if not required.issubset(fields):
                raise PublicationValidationError(
                    "coverage CSV is missing required count columns")
            rows = list(reader)
    except OSError as exc:
        raise PublicationValidationError(
            f"cannot read coverage CSV at {paths.coverage_csv}: {exc}") from exc

    normalized = {}
    for raw in rows:
        try:
            shot = int(raw["shot"])
            counts = {
                key: int(raw[key]) for key in ("n_rows", "n_valid", "n_common")
            }
        except (TypeError, ValueError) as exc:
            raise PublicationValidationError(
                "coverage CSV contains a non-integer shot or count") from exc
        if shot in normalized:
            raise PublicationValidationError(
                f"coverage CSV contains duplicate shot {shot}")
        if not (0 <= counts["n_common"] <= counts["n_valid"]
                <= counts["n_rows"]):
            raise PublicationValidationError(
                f"coverage CSV shot {shot} has inconsistent counts")
        normalized[shot] = counts

    corpus = set(
        membership.train + membership.validation
        + membership.test + membership.excluded)
    if set(normalized) != corpus:
        raise PublicationValidationError(
            "coverage CSV must contain one unique row for the complete corpus")

    totals = {}
    for part in ("train", "validation", "test", "excluded"):
        shots = getattr(membership, part)
        actual = {
            key: sum(normalized[shot][key] for shot in shots)
            for key in ("n_rows", "n_valid", "n_common")
        }
        if actual != EXPECTED_COVERAGE_TOTALS[part]:
            raise PublicationValidationError(
                f"coverage totals for {part} changed: {actual}")
        totals[part] = actual
    corpus_totals = {
        key: sum(normalized[shot][key] for shot in corpus)
        for key in ("n_rows", "n_valid", "n_common")
    }
    if corpus_totals != EXPECTED_CORPUS_COVERAGE:
        raise PublicationValidationError(
            f"coverage totals for corpus changed: {corpus_totals}")
    totals["corpus"] = corpus_totals
    return totals


def _flat_top_rows(
    paths: FreezePaths,
    corpus: Sequence[int],
) -> dict[int, tuple[float, float]]:
    try:
        with paths.flat_top_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"shot", "ip_peak_ka", "total_flat_top_s"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise PublicationValidationError(
                    "flat_top CSV is missing required statistic columns")
            raw_rows = list(reader)
    except OSError as exc:
        raise PublicationValidationError(
            f"cannot read flat_top CSV at {paths.flat_top_csv}: {exc}") from exc

    rows = {}
    for raw in raw_rows:
        try:
            shot = int(raw["shot"])
        except (TypeError, ValueError) as exc:
            raise PublicationValidationError(
                "flat_top CSV contains a non-integer shot") from exc
        if shot in rows:
            raise PublicationValidationError(
                f"flat_top CSV contains duplicate shot {shot}")
        if shot not in set(corpus):
            rows[shot] = (np.nan, np.nan)
            continue
        try:
            ip_peak = float(raw["ip_peak_ka"])
            flat_top_duration = float(raw["total_flat_top_s"])
        except (TypeError, ValueError) as exc:
            raise PublicationValidationError(
                f"flat_top CSV shot {shot} has invalid statistics") from exc
        if not np.isfinite([ip_peak, flat_top_duration]).all():
            raise PublicationValidationError(
                f"flat_top CSV shot {shot} has non-finite statistics")
        rows[shot] = (ip_peak, flat_top_duration)
    missing = tuple(shot for shot in corpus if shot not in rows)
    if missing:
        raise PublicationValidationError(
            "flat_top CSV must provide complete corpus coverage; missing "
            f"{missing[:5]}")
    return {shot: rows[shot] for shot in corpus}


def _validate_sidecar_meta(paths: FreezePaths) -> None:
    payload = _read_json_mapping(paths.sidecar_meta, label="PF sidecar metadata")
    corpus_hash = hashlib.sha256(paths.corpus_meta.read_bytes()).hexdigest()
    expected = {
        "time_axis": TIME_AXIS_TOKEN,
        "n_shots": 759,
        "n_rows": EXPECTED_CORPUS_COVERAGE["n_rows"],
        "n_common": EXPECTED_CORPUS_COVERAGE["n_common"],
        "npzgeom_meta_sha256": corpus_hash,
    }
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            raise PublicationValidationError(
                f"PF sidecar metadata {key} changed: {payload.get(key)!r}")


def _pf_tracking_rms(path: pathlib.Path, shot: int) -> float:
    try:
        with np.load(path) as data:
            pf_ref = np.asarray(data["pf_ref"], np.float64)
            pf_actual = np.asarray(data["pf_actual"], np.float64)
            common_valid = np.asarray(data["common_valid"], bool).reshape(-1)
    except (OSError, KeyError, ValueError) as exc:
        raise PublicationValidationError(
            f"cannot read PF sidecar statistics for shot {shot}: {exc}") from exc
    if (pf_ref.shape != pf_actual.shape or pf_ref.ndim != 2
            or pf_ref.shape[1] != 10
            or common_valid.shape != (pf_ref.shape[0],)
            or not common_valid.any()):
        raise PublicationValidationError(
            f"PF sidecar shot {shot} has invalid tracking arrays")
    delta = np.asarray(pf_actual, np.float64)[common_valid] \
        - np.asarray(pf_ref, np.float64)[common_valid]
    if not np.isfinite(delta).all():
        raise PublicationValidationError(
            f"PF sidecar shot {shot} has non-finite common-valid tracking data")
    return float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))


def _gather_shot_statistics(
    paths: FreezePaths,
    membership: PublicationMembership,
) -> tuple[dict[int, ShotStatistics], dict[str, dict[str, int]]]:
    corpus, durations = _corpus_rows(paths)
    expected_corpus = tuple(sorted(
        membership.train + membership.validation
        + membership.test + membership.excluded))
    if corpus != expected_corpus:
        raise PublicationValidationError(
            "publication membership does not equal the NpzGeom corpus")
    sidecar_shots = _sidecar_shots(paths)
    if sidecar_shots != corpus:
        raise PublicationValidationError(
            "PF sidecar shot stems must equal the complete NpzGeom corpus")
    _validate_sidecar_meta(paths)
    coverage_totals = _read_coverage(paths, membership)
    flat_top = _flat_top_rows(paths, corpus)

    statistics = {}
    for shot in corpus:
        ip_peak, flat_top_duration = flat_top[shot]
        statistics[shot] = ShotStatistics(
            ip_peak_ka=ip_peak,
            record_duration_s=durations[shot],
            flat_top_duration_s=flat_top_duration,
            pf_tracking_rms_a=_pf_tracking_rms(
                paths.sidecar_dir / f"{shot}.npz", shot),
        )
    return statistics, coverage_totals


def gather_shot_statistics(
    paths: FreezePaths,
    membership: PublicationMembership,
) -> dict[int, ShotStatistics]:
    """Gather four statistics from one authenticated immutable source snapshot."""
    identity = _source_identity_before_gathering(paths)
    statistics, _coverage_totals = _gather_shot_statistics(paths, membership)
    _validate_source_identity_after_gathering(paths, identity)
    return statistics


def _heating_evidence(
    paths: FreezePaths,
    corpus: Sequence[int],
) -> dict[str, object]:
    expected = publication_split.PUBLICATION_UNSUPPORTED_DIMENSIONS[
        "auxiliary_heating"]
    try:
        with paths.heating_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"shot", "node"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise PublicationValidationError(
                    "heating statistics CSV is missing shot or node")
            raw_rows = list(reader)
    except OSError as exc:
        raise PublicationValidationError(
            f"cannot read heating statistics at {paths.heating_csv}: {exc}") from exc

    source_shots = set()
    nodes = set()
    seen_keys = set()
    for row in raw_rows:
        try:
            shot = int(row["shot"])
        except (TypeError, ValueError) as exc:
            raise PublicationValidationError(
                "heating statistics contain a non-integer shot") from exc
        node = row["node"]
        key = (shot, node)
        if key in seen_keys:
            raise PublicationValidationError(
                f"heating statistics contain duplicate heating row {key}")
        seen_keys.add(key)
        source_shots.add(shot)
        nodes.add(node)

    corpus_set = set(corpus)
    covered = tuple(sorted(corpus_set & source_shots))
    missing = tuple(sorted(corpus_set - source_shots))
    if tuple(missing) != tuple(expected["missing_shots"]):
        raise PublicationValidationError(
            "heating statistics do not reproduce the exact frozen missing shots")
    relative = paths.heating_csv.relative_to(paths.project_root).as_posix()
    actual = {
        "source": relative,
        "source_rows": len(raw_rows),
        "source_shots": len(source_shots),
        "nodes": sorted(nodes),
        "corpus_shots_covered": len(covered),
        "corpus_shots_total": len(corpus),
    }
    expected_evidence = {
        key: expected[key] for key in actual
    }
    if actual != expected_evidence:
        raise PublicationValidationError(
            f"heating statistics evidence changed: {actual}")
    return copy.deepcopy(expected)


def _prior_test_record(access) -> dict[str, object]:
    return {
        "pilot_test": list(access.pilot_test),
        "train_overlap": list(access.train_overlap),
        "validation_overlap": list(access.validation_overlap),
        "test_overlap": list(access.test_overlap),
        "counts": {
            "pilot_test": len(access.pilot_test),
            "train_overlap": len(access.train_overlap),
            "validation_overlap": len(access.validation_overlap),
            "test_overlap": len(access.test_overlap),
        },
    }


def _metadata_group_counts(rows, test_shots) -> dict[str, dict[str, int]]:
    test_set = set(test_shots)
    selected = [row for row in rows if row["shot"] in test_set]
    columns = (
        "ip_group",
        "record_duration_group",
        "flat_top_duration_group",
        "pf_tracking_group",
    )
    return {
        column: dict(Counter(row[column] for row in selected))
        for column in columns
    }


def _test_group_status(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        dimension: {
            label: {
                "n_shots": count,
                "status": (
                    "inferential" if count >= MINIMUM_INFERENTIAL_TEST_GROUP_SIZE
                    else "descriptive_only"
                ),
            }
            for label, count in labels.items()
        }
        for dimension, labels in counts.items()
    }


def build_freeze_bundle(paths: FreezePaths) -> FreezeBundle:
    """Build, serialize, and validate the complete publication freeze in memory."""
    source_identity = _source_identity_before_gathering(paths)
    corpus, _durations = _corpus_rows(paths)
    pilot_test = derive_pilot_test(paths.pilot_manifest, paths.pilot_marker)
    membership = build_forward_membership(corpus, pilot_test)
    statistics, coverage_totals = _gather_shot_statistics(paths, membership)
    heating_evidence = _heating_evidence(paths, corpus)
    _validate_source_identity_after_gathering(paths, source_identity)
    thresholds = compute_metadata_thresholds(statistics, membership.train)
    frozen_thresholds = MetadataThresholds(**PUBLICATION_THRESHOLD_VALUES)
    if thresholds != frozen_thresholds:
        raise PublicationValidationError(
            f"training-only thresholds changed: {thresholds}")
    metadata_rows = assign_shot_metadata(statistics, corpus, thresholds)
    metadata_bytes = shot_metadata_csv_bytes(metadata_rows)
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    if metadata_sha256 != publication_split.PUBLICATION_FROZEN_IDENTITY[
            "shot_metadata_sha256"]:
        raise PublicationValidationError(
            "candidate metadata hash does not match the frozen publication bytes")
    access = derive_prior_test_access(membership, pilot_test)

    source_hashes = source_identity["source_sha256"]
    pilot_manifest_rel = paths.pilot_manifest.relative_to(
        paths.project_root).as_posix()
    pilot_marker_rel = paths.pilot_marker.relative_to(
        paths.project_root).as_posix()
    gap_policy = {
        "embargo": "none",
        "chosen_before_training": True,
        "reason": "no_physically_justified_embargo_duration_available",
        "shot_number_interpretation": "ordering_only_not_campaign_or_calendar",
    }
    exclusion_policy = {
        "pilot_manifest": pilot_manifest_rel,
        "pilot_manifest_sha256": source_hashes[pilot_manifest_rel],
        "pilot_marker": pilot_marker_rel,
        "pilot_marker_sha256": source_hashes[pilot_marker_rel],
        "test_selection_rule": "latest_76_shots_absent_from_pilot_test",
        "excluded": list(PUBLICATION_EXCLUDED),
        "primary_test_population": "current_76_shot_test_only",
    }
    prior_test_access = _prior_test_record(access)
    metadata_rel = paths.metadata_out.relative_to(paths.project_root).as_posix()
    strata_dir_rel = paths.policy_out.parent.relative_to(
        paths.project_root).as_posix()
    manifest = {
        "name": PUBLICATION_NAME,
        "version": PUBLICATION_VERSION,
        "time_axis": TIME_AXIS_TOKEN,
        "claim_scope": PUBLICATION_CLAIM_SCOPE,
        "split_strategy": PUBLICATION_SPLIT_STRATEGY,
        "train": list(membership.train),
        "validation": list(membership.validation),
        "test": list(membership.test),
        "excluded": list(membership.excluded),
        "shot_metadata": metadata_rel,
        "slice_strata_dir": strata_dir_rel,
        "gap_policy": gap_policy,
        "exclusion_policy": exclusion_policy,
        "prior_test_access": prior_test_access,
    }

    test_group_counts = _metadata_group_counts(metadata_rows, membership.test)
    if test_group_counts != PUBLICATION_TEST_GROUP_COUNTS:
        raise PublicationValidationError(
            f"test metadata group counts changed: {test_group_counts}")
    test_group_status = _test_group_status(test_group_counts)
    if test_group_status != publication_split.PUBLICATION_TEST_GROUP_STATUS:
        raise PublicationValidationError(
            f"test metadata group status changed: {test_group_status}")
    source_coverage_checks = {
        "corpus_shots": 759,
        "sidecar_shots": 759,
        "coverage_rows": 759,
        "flat_top_corpus_rows": 759,
        "partition_totals": coverage_totals,
    }
    if source_coverage_checks != publication_split.PUBLICATION_SOURCE_COVERAGE_CHECKS:
        raise PublicationValidationError(
            f"source coverage checks changed: {source_coverage_checks}")
    unsupported_dimensions = copy.deepcopy(
        publication_split.PUBLICATION_UNSUPPORTED_DIMENSIONS)
    unsupported_dimensions["auxiliary_heating"] = heating_evidence
    policy = {
        "schema": PUBLICATION_POLICY_SCHEMA,
        "dimensions": copy.deepcopy(
            publication_split.PUBLICATION_POLICY_DIMENSIONS),
        "thresholds": dataclasses.asdict(thresholds),
        "train_shots_sha256": canonical_shot_list_sha256(membership.train),
        "shot_metadata_sha256": metadata_sha256,
        "source_sha256": source_hashes,
        "sidecar_tree": source_identity["sidecar_tree"],
        "source_coverage_checks": source_coverage_checks,
        "test_group_counts": test_group_counts,
        "test_group_status": test_group_status,
        "minimum_inferential_test_group_size": (
            MINIMUM_INFERENTIAL_TEST_GROUP_SIZE),
        "unsupported_dimensions": unsupported_dimensions,
        "gap_policy": gap_policy,
        "exclusion_policy": exclusion_policy,
        "prior_test_access": prior_test_access,
    }
    policy_bytes = canonical_json_bytes(policy)
    manifest_bytes = canonical_json_bytes(manifest)

    validate_publication_payloads(
        manifest,
        metadata_rows,
        policy,
        available_shots=set(corpus),
        project_root=paths.project_root,
        manifest_path=paths.manifest_out,
    )
    return FreezeBundle(
        manifest=manifest,
        metadata_rows=metadata_rows,
        policy=policy,
        metadata_bytes=metadata_bytes,
        policy_bytes=policy_bytes,
        manifest_bytes=manifest_bytes,
    )


def _validate_freeze_bundle(paths: FreezePaths, bundle: FreezeBundle) -> None:
    if not isinstance(bundle, FreezeBundle):
        raise TypeError("bundle must be a FreezeBundle")
    if shot_metadata_csv_bytes(bundle.metadata_rows) != bundle.metadata_bytes:
        raise PublicationValidationError(
            "freeze bundle metadata bytes are not the canonical payload bytes")
    if canonical_json_bytes(bundle.policy) != bundle.policy_bytes:
        raise PublicationValidationError(
            "freeze bundle policy bytes are not the canonical payload bytes")
    if canonical_json_bytes(bundle.manifest) != bundle.manifest_bytes:
        raise PublicationValidationError(
            "freeze bundle manifest bytes are not the canonical payload bytes")
    available_shots = {row["shot"] for row in bundle.metadata_rows}
    validate_publication_payloads(
        bundle.manifest,
        bundle.metadata_rows,
        bundle.policy,
        available_shots=available_shots,
        project_root=paths.project_root,
        manifest_path=paths.manifest_out,
    )


def _existing_bytes(path: pathlib.Path) -> bytes | None:
    if not os.path.lexists(path):
        return None
    try:
        return read_regular_nofollow(
            path, label="publication freeze member")
    except PublicationIOError as exc:
        raise RuntimeError(str(exc)) from exc


def publish_freeze_bundle(paths: FreezePaths, bundle: FreezeBundle) -> str:
    """Publish metadata/policy then manifest last under one persistent flock."""
    _validate_freeze_bundle(paths, bundle)
    expected = {
        paths.metadata_out: bundle.metadata_bytes,
        paths.policy_out: bundle.policy_bytes,
        paths.manifest_out: bundle.manifest_bytes,
    }
    # A completed historical bundle is a strict read-only verification path.
    # No writer can overwrite these destinations (all commits are no-replace),
    # so exact manifest-complete state needs no new operational lock inode.
    preexisting = {path: _existing_bytes(path) for path in expected}
    if preexisting[paths.manifest_out] is not None:
        if all(preexisting[path] == payload
               for path, payload in expected.items()):
            return "unchanged"
        raise RuntimeError(
            "publication manifest is already frozen and the bundle is missing "
            "or differs; it is not repairable in place")
    lock_path = paths.metadata_out.parent / ".publication_freeze.lock"
    with PersistentFlock(
            lock_path, shared=False,
            state_label="publication freeze") as publication_lock:
        publication_lock.assert_held()
        existing = {path: _existing_bytes(path) for path in expected}

        manifest_existing = existing[paths.manifest_out]
        if manifest_existing is not None:
            complete_and_identical = all(
                existing[path] == payload
                for path, payload in expected.items())
            if complete_and_identical:
                return "unchanged"
            raise RuntimeError(
                "publication manifest is already frozen and the bundle is "
                "missing or differs; it is not repairable in place")

        for path in (paths.metadata_out, paths.policy_out):
            if existing[path] is not None and existing[path] != expected[path]:
                raise RuntimeError(
                    f"publication freeze member {path} exists and differs; "
                    "create a new version")

        if existing[paths.metadata_out] is None:
            durable_publish_bytes(
                paths.metadata_out,
                bundle.metadata_bytes,
                state_label="publication metadata",
            )
        if existing[paths.policy_out] is None:
            durable_publish_bytes(
                paths.policy_out,
                bundle.policy_bytes,
                state_label="publication strata policy",
            )
        publication_lock.assert_held()
        durable_publish_bytes(
            paths.manifest_out,
            bundle.manifest_bytes,
            state_label="publication manifest",
        )
        return "created"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=ROOT,
        help="repository root containing the exact publication sources",
    )
    parser.add_argument(
        "--manifest-mode", choices=("publication", "generic"),
        default="publication",
        help="canonical publication root (default) or explicit generic fixture",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.manifest_mode == "publication"
            and publication_split.lexical_absolute(args.project_root)
            != publication_split.lexical_absolute(ROOT)):
        raise ValueError(
            "publication --project-root must be the canonical lexical "
            f"repository root {ROOT}; alternate roots require --manifest-mode "
            "generic")
    paths = FreezePaths.from_project_root(args.project_root)
    bundle = build_freeze_bundle(paths)
    status = publish_freeze_bundle(paths, bundle)
    print(f"publication freeze: {status}")
    for label, path, payload in (
        ("metadata", paths.metadata_out, bundle.metadata_bytes),
        ("policy", paths.policy_out, bundle.policy_bytes),
        ("manifest", paths.manifest_out, bundle.manifest_bytes),
    ):
        print(f"{label}: {path} sha256={hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
