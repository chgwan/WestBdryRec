"""Deterministic publication split, metadata, serialization, and validation.

The generic :class:`~src.ml.pf_observability.FrozenSplit` loader remains
unchanged.  This module adds the stricter forward-by-shot publication contract,
training-only metadata categorization, stable payload serialization, and an
explicit publication/generic loading seam.
"""
import csv
import dataclasses
import hashlib
import io
import json
import pathlib
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from decimal import Decimal
from types import MappingProxyType

import numpy as np

from .pf_observability import (
    PROJECT_ROOT,
    TIME_AXIS_TOKEN,
    FrozenSplit,
    load_split,
    sha256_tree,
)


class PublicationValidationError(ValueError):
    """Raised when a publication split or its frozen payloads are invalid."""


# Hardened future-run publication namespace. Publication entry points compare
# caller values lexically (absolute + normalized, never realpath-resolved) with
# these paths before opening split/data/target state. Explicit generic mode is
# the only compatibility path for alternate roots, prefixes, or matrix subsets.
PUBLICATION_MANIFEST_PATH = (
    PROJECT_ROOT / "configs/splits/communications_physics_campaign_v1.json"
)
PUBLICATION_TARGET_DIR = PROJECT_ROOT / "ProjDB/datasets/NpzGeom"
PUBLICATION_SIDECAR_DIR = PROJECT_ROOT / "ProjDB/datasets/NpzGeomPFObs"
PUBLICATION_OUT_ROOT = PROJECT_ROOT / "ProjDB/trains"
PUBLICATION_WORK2_CONFIG = PROJECT_ROOT / "configs/dcs_pf_observability.yml"
PUBLICATION_WORK2_STATS_ROOT = PROJECT_ROOT / "ProjDB/Stats/pf_observability"
PUBLICATION_WORK2_AUDIT_IDENTITY = (
    PUBLICATION_WORK2_STATS_ROOT / "source_audit_identity.json"
)
PUBLICATION_WORK2_DECISION = (
    PUBLICATION_WORK2_STATS_ROOT / "validation_decision.json"
)
PUBLICATION_WORK2_FINAL_DIR = PUBLICATION_WORK2_STATS_ROOT / "final_test"
PUBLICATION_WORK2_MARKER = (
    PUBLICATION_WORK2_STATS_ROOT / "FINAL_TEST_EVALUATED.json"
)
PUBLICATION_WORK2_FLOOR = (
    PUBLICATION_WORK2_FINAL_DIR / "representation_floor_per_shot.csv"
)
PUBLICATION_WORK2_PREFIX = "pfobs"
PUBLICATION_WORK2_ARMS = ("A", "B", "C", "D")
PUBLICATION_SEEDS = (0, 1, 2, 3, 4)
PUBLICATION_WORK3_CONFIG = PROJECT_ROOT / "configs/dcs_pf_context_sweep.yml"
PUBLICATION_WORK3_REFERENCES = PROJECT_ROOT / "configs/pf_timescale_references.yml"
PUBLICATION_WORK3_STATS_ROOT = PROJECT_ROOT / "ProjDB/Stats/pf_context"
PUBLICATION_WORK3_AUDIT_CSV = (
    PUBLICATION_WORK3_STATS_ROOT / "context_availability.csv"
)
PUBLICATION_WORK3_AUDIT_IDENTITY = (
    PUBLICATION_WORK3_STATS_ROOT / "context_availability.audit.json"
)
PUBLICATION_WORK3_SELECTION = (
    PUBLICATION_WORK3_STATS_ROOT / "validation_selection.json"
)
PUBLICATION_WORK3_FINAL_DIR = PUBLICATION_WORK3_STATS_ROOT / "final_test"
PUBLICATION_WORK3_TRANSACTION = (
    PUBLICATION_WORK3_FINAL_DIR / "transaction_provenance.json"
)
PUBLICATION_WORK3_MARKER = (
    PUBLICATION_WORK3_STATS_ROOT / "PFCTX_FINAL_TEST_EVALUATED.json"
)
PUBLICATION_WORK3_FINAL_BUILDING = (
    PUBLICATION_WORK3_STATS_ROOT.parent / "pf_context.final_building"
)
PUBLICATION_WORK3_PREFIX = "pfctx"
PUBLICATION_WORK3_CONTEXTS = (
    "h0001", "h0032", "h0128", "h0256", "h0512", "h1024", "h2048",
)


def lexical_absolute(path) -> pathlib.Path:
    """Absolute lexical path without resolving repository symlinks."""
    import os
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def require_publication_paths(
        manifest_mode: str,
        selected: Mapping[str, object],
        expected: Mapping[str, object]) -> None:
    """Reject every publication path outside the one canonical namespace."""
    if manifest_mode != "publication":
        return
    if set(selected) != set(expected):
        raise ValueError("publication path gate received an incomplete path set")
    for label in expected:
        actual_path = lexical_absolute(selected[label])
        expected_path = lexical_absolute(expected[label])
        if actual_path != expected_path:
            raise ValueError(
                f"publication --{label.replace('_', '-')} must be the canonical "
                f"lexical path {expected_path}, got {actual_path}; alternate "
                "roots are available only in explicit generic mode"
            )


PUBLICATION_NAME = "communications_physics_forward_by_shot_unseen_test_v1"
PUBLICATION_VERSION = 1
PUBLICATION_CLAIM_SCOPE = "publication"
PUBLICATION_SPLIT_STRATEGY = (
    "sorted_shot_number_forward_598_76_76_exclude_prior_test"
)
PUBLICATION_POLICY_SCHEMA = "communications_physics_forward_strata_policy_v1"
PUBLICATION_COUNTS = {
    "train": 598,
    "validation": 76,
    "test": 76,
    "excluded": 9,
}
PUBLICATION_RANGES = {
    "train": (57281, 58190),
    "validation": (58192, 58292),
    "test": (58293, 58511),
}
PUBLICATION_EXCLUDED = (
    58302, 58324, 58334, 58342, 58354,
    58358, 58366, 58404, 58417,
)
METADATA_FIELDS = (
    "shot",
    "ip_group",
    "record_duration_group",
    "flat_top_duration_group",
    "pf_tracking_group",
)
METADATA_LABELS = {
    "ip_group": ("low", "high"),
    "record_duration_group": ("short", "long"),
    "flat_top_duration_group": ("short", "long"),
    "pf_tracking_group": ("low", "high"),
}
PUBLICATION_THRESHOLD_VALUES = {
    "ip_peak_ka": 505.66,
    "record_duration_s": 11.914701,
    "flat_top_duration_s": 8.652,
    "pf_tracking_rms_a": 194.12958088660827,
}
PUBLICATION_TEST_GROUP_COUNTS = {
    "ip_group": {"low": 69, "high": 7},
    "record_duration_group": {"short": 24, "long": 52},
    "flat_top_duration_group": {"short": 23, "long": 53},
    "pf_tracking_group": {"low": 53, "high": 23},
}
PUBLICATION_MISSING_HEATING_SHOTS = (
    57302, 57350, 57362, 57367, 57377, 57397, 57398, 57406, 57452,
    57486, 57488, 57489, 57501, 57522, 57534, 57548, 57553, 57563,
    57586, 57588, 57591, 57600, 57601, 57618, 57624, 57628, 57656,
    57661, 57758, 57764, 57765, 57775, 57812, 57822, 57902, 57913,
    57965, 57992, 58014, 58043, 58109, 58203, 58210, 58221, 58255,
    58276, 58352, 58375, 58395, 58511,
)
PUBLICATION_POLICY_DIMENSIONS = {
    "ip_group": {
        "source": "ProjDB/Stats/flat_top.csv",
        "source_fields": ["ip_peak_ka"],
        "formula": "ip_peak_ka",
        "unit": "kA",
        "threshold_population": "publication_train_598_only",
        "threshold_statistic": "median",
        "labels": ["low", "high"],
        "lower_rule": "value < threshold",
        "upper_rule": "value >= threshold",
        "tie_rule": "equal_to_threshold_uses_upper",
    },
    "record_duration_group": {
        "source": "ProjDB/datasets/NpzGeom/meta.json",
        "source_fields": ["t_start", "t_end"],
        "formula": "t_end - t_start",
        "unit": "s",
        "threshold_population": "publication_train_598_only",
        "threshold_statistic": "median",
        "labels": ["short", "long"],
        "lower_rule": "value < threshold",
        "upper_rule": "value >= threshold",
        "tie_rule": "equal_to_threshold_uses_upper",
    },
    "flat_top_duration_group": {
        "source": "ProjDB/Stats/flat_top.csv",
        "source_fields": ["total_flat_top_s"],
        "formula": "total_flat_top_s",
        "unit": "s",
        "threshold_population": "publication_train_598_only",
        "threshold_statistic": "median",
        "labels": ["short", "long"],
        "lower_rule": "value < threshold",
        "upper_rule": "value >= threshold",
        "tie_rule": "equal_to_threshold_uses_upper",
    },
    "pf_tracking_group": {
        "source": "ProjDB/datasets/NpzGeomPFObs",
        "source_fields": ["pf_ref", "pf_actual", "common_valid"],
        "formula": (
            "sqrt(mean(square(float64(pf_actual)[common_valid] - "
            "float64(pf_ref)[common_valid]), dtype=float64))"),
        "unit": "A",
        "calculation": {
            "difference": "actual_minus_reference",
            "dtype": "float64_before_subtraction_and_squaring",
            "row_mask": "common_valid",
            "channels": 10,
            "aggregation": "all_selected_rows_and_channels",
        },
        "threshold_population": "publication_train_598_only",
        "threshold_statistic": "median",
        "labels": ["low", "high"],
        "lower_rule": "value < threshold",
        "upper_rule": "value >= threshold",
        "tie_rule": "equal_to_threshold_uses_upper",
    },
}
PUBLICATION_SOURCE_COVERAGE_CHECKS = {
    "corpus_shots": 759,
    "sidecar_shots": 759,
    "coverage_rows": 759,
    "flat_top_corpus_rows": 759,
    "partition_totals": {
        "train": {"n_rows": 4322025, "n_valid": 4318828,
                  "n_common": 4318828},
        "validation": {"n_rows": 1318663, "n_valid": 1318385,
                       "n_common": 1318385},
        "test": {"n_rows": 1391591, "n_valid": 1390655,
                 "n_common": 1390655},
        "excluded": {"n_rows": 181127, "n_valid": 180948,
                     "n_common": 180948},
        "corpus": {"n_rows": 7213406, "n_valid": 7208816,
                   "n_common": 7208816},
    },
}
PUBLICATION_TEST_GROUP_STATUS = {
    "ip_group": {
        "low": {"n_shots": 69, "status": "inferential"},
        "high": {"n_shots": 7, "status": "descriptive_only"},
    },
    "record_duration_group": {
        "short": {"n_shots": 24, "status": "inferential"},
        "long": {"n_shots": 52, "status": "inferential"},
    },
    "flat_top_duration_group": {
        "short": {"n_shots": 23, "status": "inferential"},
        "long": {"n_shots": 53, "status": "inferential"},
    },
    "pf_tracking_group": {
        "low": {"n_shots": 53, "status": "inferential"},
        "high": {"n_shots": 23, "status": "inferential"},
    },
}
PUBLICATION_UNSUPPORTED_DIMENSIONS = {
    "auxiliary_heating": {
        "status": "unsupported",
        "source": "ProjDB/Stats/heating_sustained_stats.csv",
        "source_rows": 4240,
        "source_shots": 848,
        "nodes": [
            "PowIC1_scope", "PowIC2_scope", "PowIC3_scope",
            "PowLH1_scope", "PowLH2_scope",
        ],
        "corpus_shots_covered": 709,
        "corpus_shots_total": 759,
        "missing_shots": list(PUBLICATION_MISSING_HEATING_SHOTS),
        "missing_rows_mean": "unknown_not_heating_off",
        "reason": (
            "incomplete_709_of_759_coverage_and_no_auditable_repository_"
            "generator"),
    },
    "configuration": {
        "status": "unsupported",
        "labels": ["limiter", "LSN", "USN", "DN"],
        "reason": "no_WEST_expert_approved_signal_mapping_or_labelled_test_shots",
    },
    "discharge_phase": {
        "status": "unsupported",
        "labels": ["ramp_up", "flat_top", "ramp_down"],
        "reason": "requires_per_row_phase_masks_joined_to_prediction_row_indices",
    },
}
PUBLICATION_FROZEN_IDENTITY = {
    "source_sha256": {
        "ProjDB/datasets/NpzGeom/meta.json":
            "72c0ec5b78bd897f08748b0444659da66b241e42607a6c12a690d6d2996b9c2f",
        "ProjDB/datasets/NpzGeomPFObs/meta.json":
            "f499c9eb10613dab6f98d57243a1fa288f57400e490d4cc2da537847a9b1df2b",
        "ProjDB/Stats/pf_observability/coverage.csv":
            "f6fbcb5f85c5afeae64ef0cf75cdda80f584b62ab696ce244b897181c10f804a",
        "ProjDB/Stats/flat_top.csv":
            "88b519e7c4ccbb8ac410e1cd0db5ea111460b4257104044620e83fb7b01a8dfb",
        "configs/splits/pfobs_random_pilot.json":
            "bf3bd97c3e17ff9a243430ecaf026cf1ef9bf2ae65d60836622fa934021a2c57",
        "ProjDB/Stats/pf_observability_pilot/FINAL_TEST_EVALUATED.json":
            "d865bcc56268fdb050eeb057bef9e79bde2fa9275fafc9b2a10fd1aa90b3ff3e",
        "ProjDB/Stats/heating_sustained_stats.csv":
            "2ec6bbd293b7af5a1fe72c469b8ae53ff337b9b35917f0da4db4f11ea3258e19",
    },
    "shot_metadata_sha256": (
        "550e7dc90a60e5eee9bd769ef95fae435a16666dc53f9b6362551ab9e3e3d23d"),
    "sidecar_tree": {
        "root": "ProjDB/datasets/NpzGeomPFObs",
        "algorithm": (
            "sha256_tree_v1_sorted_relative_path_colon_file_sha256_"
            "lf_join_no_final_lf"),
        "file_count": 760,
        "sha256": (
            "a43bab74cfa1fdf87fff034472a42f5b72f392a2a9ddcf24813b0e400449bfab"),
    },
}
_REQUIRED_POLICY_FIELDS = {
    "schema", "dimensions", "thresholds", "train_shots_sha256",
    "shot_metadata_sha256", "source_sha256", "sidecar_tree",
    "source_coverage_checks", "test_group_counts", "test_group_status",
    "minimum_inferential_test_group_size", "unsupported_dimensions",
    "gap_policy", "exclusion_policy", "prior_test_access",
}
_REQUIRED_GAP_POLICY = {
    "embargo": "none",
    "chosen_before_training": True,
    "reason": "no_physically_justified_embargo_duration_available",
    "shot_number_interpretation": "ordering_only_not_campaign_or_calendar",
}
_REQUIRED_TEST_SELECTION_RULE = "latest_76_shots_absent_from_pilot_test"
_REQUIRED_PRIMARY_TEST_POPULATION = "current_76_shot_test_only"


@dataclasses.dataclass(frozen=True)
class ShotStatistics:
    ip_peak_ka: float
    record_duration_s: float
    flat_top_duration_s: float
    pf_tracking_rms_a: float


@dataclasses.dataclass(frozen=True)
class MetadataThresholds:
    ip_peak_ka: float
    record_duration_s: float
    flat_top_duration_s: float
    pf_tracking_rms_a: float


@dataclasses.dataclass(frozen=True)
class PublicationMembership:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    excluded: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PriorTestAccess:
    pilot_test: tuple[int, ...]
    train_overlap: tuple[int, ...]
    validation_overlap: tuple[int, ...]
    test_overlap: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PublicationSplit(FrozenSplit):
    claim_scope: str
    split_strategy: str
    excluded: tuple[int, ...]
    gap_policy: Mapping[str, object]
    exclusion_policy: Mapping[str, object]
    prior_test_access: Mapping[str, object]
    manifest_path: pathlib.Path
    policy_path: pathlib.Path


def canonical_shot_list_sha256(shots: Iterable[int]) -> str:
    """Hash ascending decimal shot IDs, one per line with a final newline."""
    ordered = tuple(sorted(int(shot) for shot in shots))
    payload = "".join(f"{shot}\n" for shot in ordered).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize a JSON payload deterministically as UTF-8 with a final LF."""
    return (json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("utf-8")


def _normalized_statistics(
    statistics: Mapping[int, ShotStatistics],
) -> dict[int, ShotStatistics]:
    normalized = {}
    for raw_shot, row in statistics.items():
        shot = int(raw_shot)
        if shot in normalized:
            raise PublicationValidationError(
                "shot statistics contain duplicate shot IDs")
        if not isinstance(row, ShotStatistics):
            raise PublicationValidationError(
                f"shot {shot} statistics must be ShotStatistics")
        normalized[shot] = row
    return normalized


def _statistics_matrix(rows: Sequence[ShotStatistics]) -> np.ndarray:
    return np.asarray([
        [
            row.ip_peak_ka,
            row.record_duration_s,
            row.flat_top_duration_s,
            row.pf_tracking_rms_a,
        ]
        for row in rows
    ], dtype=np.float64)


def compute_metadata_thresholds(
    statistics: Mapping[int, ShotStatistics],
    train_shots: Sequence[int],
) -> MetadataThresholds:
    """Return four medians computed from exactly 598 unique training shots."""
    normalized = _normalized_statistics(statistics)
    train = tuple(int(shot) for shot in train_shots)
    if len(train) != len(set(train)):
        raise PublicationValidationError(
            "training metadata contains duplicate shots")
    missing = tuple(shot for shot in train if shot not in normalized)
    if missing:
        raise PublicationValidationError(
            f"training metadata missing statistics for shots {missing[:5]}")
    rows = [normalized[shot] for shot in train]
    matrix = _statistics_matrix(rows)
    if matrix.shape != (598, 4) or not np.isfinite(matrix).all():
        raise PublicationValidationError("training metadata must be finite 598x4")
    medians = np.median(matrix, axis=0)
    # With 598 rows, NumPy averages the two central binary64 values. Re-form
    # that same midpoint from their shortest decimal forms so source-decimal
    # medians such as 8.651/8.653 freeze as 8.652, not a one-ULP artifact.
    ordered = np.sort(matrix, axis=0)
    middle = ordered.shape[0] // 2
    decimal_midpoints = np.asarray([
        float((
            Decimal(str(float(ordered[middle - 1, column])))
            + Decimal(str(float(ordered[middle, column])))
        ) / 2)
        for column in range(ordered.shape[1])
    ], dtype=np.float64)
    if not np.allclose(
            decimal_midpoints, medians, rtol=0.0,
            atol=np.spacing(np.abs(medians))):
        raise PublicationValidationError(
            "training metadata median normalization changed its value")
    return MetadataThresholds(*map(float, decimal_midpoints))


def _pf_tracking_rms_a(pf_ref, pf_actual, common_valid) -> float:
    """RMS actual-minus-reference over all ten common-valid PF channels."""
    ref = np.asarray(pf_ref, dtype=np.float64)
    actual = np.asarray(pf_actual, dtype=np.float64)
    common = np.asarray(common_valid, dtype=bool).reshape(-1)
    if ref.shape != actual.shape or ref.ndim != 2 or ref.shape[1] != 10:
        raise PublicationValidationError(
            "PF tracking arrays must have matching (n_rows, 10) shapes")
    if common.shape != (ref.shape[0],) or not common.any():
        raise PublicationValidationError(
            "PF tracking requires common-valid rows matching the PF arrays")
    delta = actual[common] - ref[common]
    if not np.isfinite(delta).all():
        raise PublicationValidationError(
            "common-valid PF tracking values must be finite")
    return float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))


def assign_shot_metadata(
    statistics: Mapping[int, ShotStatistics],
    shots: Iterable[int],
    thresholds: MetadataThresholds,
) -> tuple[dict[str, object], ...]:
    """Assign training-threshold categories, sorting rows by shot number."""
    normalized = _normalized_statistics(statistics)
    selected = tuple(int(shot) for shot in shots)
    if len(selected) != len(set(selected)):
        raise PublicationValidationError("metadata shots contain duplicate shots")
    missing = tuple(shot for shot in selected if shot not in normalized)
    if missing:
        raise PublicationValidationError(
            f"metadata missing statistics for shots {missing[:5]}")
    threshold_values = np.asarray(dataclasses.astuple(thresholds), np.float64)
    if threshold_values.shape != (4,) or not np.isfinite(threshold_values).all():
        raise PublicationValidationError("metadata thresholds must be finite")
    selected_rows = [normalized[shot] for shot in selected]
    matrix = _statistics_matrix(selected_rows)
    if matrix.shape != (len(selected), 4) or not np.isfinite(matrix).all():
        raise PublicationValidationError("shot metadata statistics must be finite")

    result = []
    for shot in sorted(selected):
        row = normalized[shot]
        result.append({
            "shot": shot,
            "ip_group": (
                "low" if row.ip_peak_ka < thresholds.ip_peak_ka else "high"),
            "record_duration_group": (
                "short" if row.record_duration_s
                < thresholds.record_duration_s else "long"),
            "flat_top_duration_group": (
                "short" if row.flat_top_duration_s
                < thresholds.flat_top_duration_s else "long"),
            "pf_tracking_group": (
                "low" if row.pf_tracking_rms_a
                < thresholds.pf_tracking_rms_a else "high"),
        })
    return tuple(result)


def _validated_metadata_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_shots: Iterable[int] | None = None,
    csv_shots: bool = False,
) -> tuple[dict[str, object], ...]:
    normalized = []
    seen = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(METADATA_FIELDS):
            raise PublicationValidationError(
                "metadata rows must contain exactly the publication header fields")
        shot = (
            _canonical_csv_int(raw["shot"], label="metadata shot")
            if csv_shots else
            _strict_int(raw["shot"], label="metadata shot")
        )
        if shot in seen:
            raise PublicationValidationError(
                f"metadata contains duplicate shot {shot}")
        seen.add(shot)
        row = {"shot": shot}
        for column in METADATA_FIELDS[1:]:
            label = str(raw[column])
            if label not in METADATA_LABELS[column]:
                raise PublicationValidationError(
                    f"metadata {column} has invalid label {label!r}")
            row[column] = label
        normalized.append(row)
    normalized.sort(key=lambda row: row["shot"])
    if expected_shots is not None:
        expected = tuple(sorted(int(shot) for shot in expected_shots))
        actual = tuple(row["shot"] for row in normalized)
        if actual != expected:
            raise PublicationValidationError(
                "metadata rows must cover the complete sorted 759-shot corpus")
    return tuple(normalized)


def shot_metadata_csv_bytes(
    rows: Iterable[Mapping[str, object]],
) -> bytes:
    """Serialize categorical metadata with the exact stable publication CSV."""
    normalized = _validated_metadata_rows(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=METADATA_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(normalized)
    return buffer.getvalue().encode("utf-8")


def _as_sorted_shots(shots: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(shot) for shot in shots))


def validate_publication_membership(
    membership: PublicationMembership,
    corpus_shots: Iterable[int],
    pilot_test_shots: Iterable[int],
) -> None:
    """Validate all structural, chronological, and prior-test invariants."""
    corpus = _as_sorted_shots(corpus_shots)
    if len(corpus) != len(set(corpus)):
        raise PublicationValidationError(
            "publication corpus contains duplicate shots")

    pilot = _as_sorted_shots(pilot_test_shots)
    if len(pilot) != len(set(pilot)):
        raise PublicationValidationError(
            "pilot test contains duplicate shots")
    corpus_set = set(corpus)
    pilot_set = set(pilot)
    outside_pilot = pilot_set - corpus_set
    if outside_pilot:
        raise PublicationValidationError(
            "pilot test contains shots outside the publication corpus")

    parts = {
        "train": tuple(int(shot) for shot in membership.train),
        "validation": tuple(int(shot) for shot in membership.validation),
        "test": tuple(int(shot) for shot in membership.test),
        "excluded": tuple(int(shot) for shot in membership.excluded),
    }
    for name, shots in parts.items():
        if len(shots) != len(set(shots)):
            raise PublicationValidationError(
                f"publication {name} contains duplicate shots")
        if tuple(sorted(shots)) != shots:
            raise PublicationValidationError(
                f"publication {name} must be sorted")

    for name in ("train", "validation", "test"):
        if not parts[name]:
            raise PublicationValidationError(
                f"publication {name} must be non-empty")

    test_overlap = set(parts["test"]) & pilot_set
    if test_overlap:
        raise PublicationValidationError(
            "publication test overlaps the pilot test"
            f" ({sorted(test_overlap)[:5]})")

    if not (max(parts["train"]) < min(parts["validation"])
            and max(parts["validation"]) < min(parts["test"])):
        raise PublicationValidationError(
            "publication partitions violate chronology")

    test_start = min(parts["test"])
    expected_excluded = tuple(
        shot for shot in corpus if shot >= test_start and shot in pilot_set
    )
    if parts["excluded"] != expected_excluded:
        raise PublicationValidationError(
            "publication excluded set does not match prior-test shots"
            f" (expected={expected_excluded}, got={parts['excluded']})")

    names = tuple(parts)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            shared = set(parts[left_name]) & set(parts[right_name])
            if shared:
                raise PublicationValidationError(
                    f"publication {left_name} and {right_name} overlap")

    partition_union = set().union(*(set(shots) for shots in parts.values()))
    if partition_union != corpus_set:
        missing = sorted(corpus_set - partition_union)
        extra = sorted(partition_union - corpus_set)
        raise PublicationValidationError(
            "publication partition union does not equal corpus"
            f" (missing={missing[:5]}, extra={extra[:5]})")


def build_forward_membership(
    corpus_shots: Iterable[int],
    pilot_test_shots: Iterable[int],
    *,
    validation_size: int = 76,
    test_size: int = 76,
) -> PublicationMembership:
    """Build the no-embargo chronological publication partition."""
    shots = tuple(sorted(int(shot) for shot in corpus_shots))
    if len(shots) != len(set(shots)):
        raise PublicationValidationError(
            "publication corpus contains duplicate shots")
    if validation_size <= 0 or test_size <= 0:
        raise PublicationValidationError(
            "validation_size and test_size must be positive")
    pilot_test = frozenset(int(shot) for shot in pilot_test_shots)
    clean_candidates = tuple(shot for shot in shots if shot not in pilot_test)
    if len(clean_candidates) < test_size:
        raise PublicationValidationError(
            "publication corpus has fewer clean shots than test_size")
    test = clean_candidates[-test_size:]
    test_start = min(test)
    excluded = tuple(
        shot for shot in shots if shot >= test_start and shot in pilot_test
    )
    pretest = tuple(shot for shot in shots if shot < test_start)
    if len(pretest) < validation_size:
        raise PublicationValidationError(
            "publication corpus has fewer pretest shots than validation_size")
    validation = pretest[-validation_size:]
    train = pretest[:-validation_size]
    membership = PublicationMembership(train, validation, test, excluded)
    validate_publication_membership(membership, shots, pilot_test)
    return membership


def derive_prior_test_access(
    membership: PublicationMembership,
    pilot_test_shots: Iterable[int],
) -> PriorTestAccess:
    """Account for every pilot-test shot in the publication partition."""
    pilot = tuple(sorted(int(shot) for shot in pilot_test_shots))
    pilot_set = set(pilot)
    result = PriorTestAccess(
        pilot_test=pilot,
        train_overlap=tuple(shot for shot in membership.train if shot in pilot_set),
        validation_overlap=tuple(
            shot for shot in membership.validation if shot in pilot_set
        ),
        test_overlap=tuple(shot for shot in membership.test if shot in pilot_set),
    )
    accounted = (
        set(result.train_overlap)
        | set(result.validation_overlap)
        | set(result.test_overlap)
        | set(membership.excluded)
    )
    if accounted != pilot_set:
        raise PublicationValidationError("prior-test accounting does not cover 76 shots")
    return result


def _read_json(source, *, label: str):
    if isinstance(source, Mapping):
        return source
    path = pathlib.Path(source)
    return json.loads(path.read_text())


def _manifest_sha256(source) -> str:
    if isinstance(source, Mapping):
        raise TypeError("pilot manifest must be a path so its bytes can be verified")
    path = pathlib.Path(source)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_shots_from_manifest(manifest) -> tuple[int, ...]:
    if "test" in manifest:
        values = manifest["test"]
    elif isinstance(manifest.get("split"), Mapping):
        values = manifest["split"].get("test_shots")
    else:
        values = None
    if values is None:
        raise PublicationValidationError(
            "pilot manifest does not contain test shots")
    result = tuple(sorted(int(shot) for shot in values))
    if len(result) != len(set(result)):
        raise PublicationValidationError(
            "pilot manifest test contains duplicate shots")
    return result


def _test_shots_from_marker(marker) -> tuple[int, ...]:
    split = marker.get("split")
    if isinstance(split, Mapping) and split.get("test_shots") is not None:
        values = split["test_shots"]
    elif marker.get("test_shots") is not None:
        values = marker["test_shots"]
    elif marker.get("test") is not None:
        values = marker["test"]
    else:
        raise PublicationValidationError(
            "pilot marker does not contain test shots")
    result = tuple(sorted(int(shot) for shot in values))
    if len(result) != len(set(result)):
        raise PublicationValidationError(
            "pilot marker test contains duplicate shots")
    return result


def _split_hash_from_marker(marker) -> str:
    hashes = []

    def add(value):
        if value is not None:
            hashes.append(str(value).lower())

    add(marker.get("split_sha256"))
    add(marker.get("manifest_sha256"))
    split = marker.get("split")
    if isinstance(split, Mapping):
        add(split.get("split_sha256"))
        add(split.get("manifest_sha256"))
        add(split.get("sha256"))
    for run in marker.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        add(run.get("split_sha256"))
        fingerprint = run.get("run_fingerprint")
        if isinstance(fingerprint, Mapping):
            add(fingerprint.get("split_sha256"))

    if not hashes:
        raise PublicationValidationError(
            "pilot marker does not contain an embedded split hash")
    if len(set(hashes)) != 1:
        raise PublicationValidationError(
            "pilot marker contains inconsistent embedded split hashes")
    return hashes[0]


def derive_pilot_test(pilot_manifest, pilot_marker) -> tuple[int, ...]:
    """Verify and return the frozen pilot test membership.

    ``pilot_manifest`` must be a JSON path because the marker authenticates
    its exact bytes.  The marker may expose ``split_sha256`` directly or in
    the existing run fingerprint records, and its test list may be under
    ``split.test_shots`` or at the top level.
    """
    manifest = _read_json(pilot_manifest, label="pilot manifest")
    marker = _read_json(pilot_marker, label="pilot marker")
    actual_hash = _manifest_sha256(pilot_manifest)
    expected_hash = _split_hash_from_marker(marker)
    if actual_hash.lower() != expected_hash:
        raise PublicationValidationError(
            "pilot manifest bytes do not match marker split hash")

    manifest_test = _test_shots_from_manifest(manifest)
    marker_test = _test_shots_from_marker(marker)
    if manifest_test != marker_test:
        raise PublicationValidationError(
            "pilot manifest and marker test-shot membership differ")
    return manifest_test


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PublicationValidationError(f"{label} must be a mapping")
    return value


def _required(mapping: Mapping[str, object], key: str, *, label: str):
    if key not in mapping:
        raise PublicationValidationError(f"{label} missing required field {key}")
    return mapping[key]


def _strict_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise PublicationValidationError(
            f"{label} must be a canonical integer, not {value!r}")
    return value


def _strict_int_sequence(value: object, *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PublicationValidationError(f"{label} must be an integer list")
    return tuple(
        _strict_int(item, label=f"{label} item")
        for item in value
    )


def _strict_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicationValidationError(
            f"{label} must be a JSON number, not {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise PublicationValidationError(f"{label} must be finite")
    return result


def _canonical_csv_int(value: object, *, label: str) -> int:
    if type(value) is not str:
        raise PublicationValidationError(
            f"{label} must be a canonical decimal integer string")
    try:
        result = int(value)
    except ValueError as exc:
        raise PublicationValidationError(
            f"{label} must be a canonical decimal integer string") from exc
    if str(result) != value:
        raise PublicationValidationError(
            f"{label} must be a canonical decimal integer string")
    return result


def _json_safe_scalar(value: object, *, label: str) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and np.isfinite(value):
        return value
    raise PublicationValidationError(
        f"{label} contains a non-JSON-safe policy value {value!r}")


def _freeze_payload(value: object, *, label: str = "policy") -> object:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PublicationValidationError(
                    f"{label} contains a non-JSON-safe mapping key {key!r}")
            frozen[key] = _freeze_payload(item, label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_payload(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    return _json_safe_scalar(value, label=label)


def _thaw_payload(value: object, *, label: str = "policy") -> object:
    if isinstance(value, Mapping):
        thawed = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PublicationValidationError(
                    f"{label} contains a non-JSON-safe mapping key {key!r}")
            thawed[key] = _thaw_payload(item, label=f"{label}.{key}")
        return thawed
    if isinstance(value, (list, tuple)):
        return [
            _thaw_payload(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    return _json_safe_scalar(value, label=label)


def _resolve_project_path(
    value: object,
    *,
    project_root: pathlib.Path,
    label: str,
) -> pathlib.Path:
    if type(value) is not str or not value:
        raise PublicationValidationError(f"publication {label} path is required")
    path = pathlib.Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PublicationValidationError(
            f"publication {label} must be a lexical relative path without '..'")
    return pathlib.Path(project_root) / path


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise PublicationValidationError(f"{label} must be a SHA-256 hex digest")
    digest = value.lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise PublicationValidationError(f"{label} must be a SHA-256 hex digest")
    return digest


def _validate_exact_policy_section(
    policy: Mapping[str, object],
    key: str,
    expected: object,
) -> object:
    actual = _required(policy, key, label="strata policy")
    try:
        actual_bytes = canonical_json_bytes(actual)
        expected_bytes = canonical_json_bytes(expected)
    except (TypeError, ValueError) as exc:
        raise PublicationValidationError(
            f"strata policy {key} must be deterministic JSON data") from exc
    if actual_bytes != expected_bytes:
        raise PublicationValidationError(
            f"strata policy {key} does not match the frozen publication policy")
    return actual


def _validate_sidecar_tree(
    value: object,
    *,
    project_root: pathlib.Path,
) -> None:
    tree = _require_mapping(value, label="strata policy sidecar_tree")
    expected = PUBLICATION_FROZEN_IDENTITY["sidecar_tree"]
    try:
        actual_bytes = canonical_json_bytes(tree)
        expected_bytes = canonical_json_bytes(expected)
    except (TypeError, ValueError) as exc:
        raise PublicationValidationError(
            "strata policy sidecar_tree must be deterministic JSON data") from exc
    if actual_bytes != expected_bytes:
        raise PublicationValidationError(
            "strata policy sidecar_tree does not match the frozen identity")
    root = _resolve_project_path(
        tree["root"], project_root=project_root, label="sidecar tree root")
    file_count = _strict_int(
        tree["file_count"], label="strata policy sidecar_tree file_count")
    if not root.is_dir():
        raise PublicationValidationError(
            f"strata policy sidecar_tree root does not exist: {root}")
    actual_count = sum(1 for path in root.rglob("*") if path.is_file())
    if actual_count != file_count:
        raise PublicationValidationError(
            "strata policy sidecar_tree file count does not match source files")
    expected_hash = _require_sha256(
        tree["sha256"], label="strata policy sidecar_tree sha256")
    try:
        actual_hash = sha256_tree(root)
    except OSError as exc:
        raise PublicationValidationError(
            f"cannot hash strata policy sidecar_tree at {root}: {exc}") from exc
    if actual_hash != expected_hash:
        raise PublicationValidationError(
            "strata policy sidecar_tree hash does not match source files")


def _manifest_shots(
    manifest: Mapping[str, object],
    part: str,
) -> tuple[int, ...]:
    raw = _required(manifest, part, label="publication manifest")
    shots = _strict_int_sequence(
        raw, label=f"publication manifest {part}")
    if len(shots) != len(set(shots)):
        raise PublicationValidationError(
            f"publication manifest {part} contains duplicate shots")
    if tuple(sorted(shots)) != shots:
        raise PublicationValidationError(
            f"publication manifest {part} must be sorted")
    return shots


def _validate_gap_policy(value: object) -> dict[str, object]:
    policy = _require_mapping(value, label="gap_policy")
    for key, expected in _REQUIRED_GAP_POLICY.items():
        if policy.get(key) != expected:
            raise PublicationValidationError(
                f"gap_policy {key} must be {expected!r}")
    return dict(policy)


def _validate_exclusion_policy(
    value: object,
    *,
    project_root: pathlib.Path,
) -> tuple[dict[str, object], tuple[int, ...]]:
    policy = _require_mapping(value, label="exclusion_policy")
    if policy.get("test_selection_rule") != _REQUIRED_TEST_SELECTION_RULE:
        raise PublicationValidationError(
            "exclusion_policy test_selection_rule is invalid")
    if policy.get("primary_test_population") != _REQUIRED_PRIMARY_TEST_POPULATION:
        raise PublicationValidationError(
            "exclusion_policy primary_test_population is invalid")
    excluded = _strict_int_sequence(
        policy.get("excluded", ()), label="exclusion_policy excluded")
    if excluded != PUBLICATION_EXCLUDED:
        raise PublicationValidationError(
            "exclusion_policy must contain the exact nine excluded shots")

    pilot_manifest = _resolve_project_path(
        _required(policy, "pilot_manifest", label="exclusion_policy"),
        project_root=project_root,
        label="pilot manifest",
    )
    pilot_marker = _resolve_project_path(
        _required(policy, "pilot_marker", label="exclusion_policy"),
        project_root=project_root,
        label="pilot marker",
    )
    for path, key, label in (
        (pilot_manifest, "pilot_manifest_sha256", "pilot manifest"),
        (pilot_marker, "pilot_marker_sha256", "pilot marker"),
    ):
        if not path.is_file():
            raise PublicationValidationError(
                f"exclusion_policy {label} does not exist: {path}")
        expected = _require_sha256(
            _required(policy, key, label="exclusion_policy"),
            label=f"exclusion_policy {key}",
        )
        if _sha256_file(path) != expected:
            raise PublicationValidationError(
                f"exclusion_policy {label} hash does not match its bytes")
    return dict(policy), derive_pilot_test(pilot_manifest, pilot_marker)


def _validate_prior_test_access(
    value: object,
    expected: PriorTestAccess,
) -> dict[str, object]:
    access = _require_mapping(value, label="prior_test_access")
    expected_lists = {
        "pilot_test": expected.pilot_test,
        "train_overlap": expected.train_overlap,
        "validation_overlap": expected.validation_overlap,
        "test_overlap": expected.test_overlap,
    }
    for key, wanted in expected_lists.items():
        raw = _required(access, key, label="prior_test_access")
        actual = _strict_int_sequence(
            raw, label=f"prior_test_access {key}")
        if actual != wanted:
            raise PublicationValidationError(
                f"prior_test_access {key} does not match derived membership")
    counts = _require_mapping(
        _required(access, "counts", label="prior_test_access"),
        label="prior_test_access counts",
    )
    expected_counts = {
        "pilot_test": 76,
        "train_overlap": 60,
        "validation_overlap": 7,
        "test_overlap": 0,
    }
    try:
        actual_counts = {
            key: _strict_int(
                counts[key], label=f"prior_test_access counts {key}")
            for key in expected_counts
        }
    except KeyError as exc:
        raise PublicationValidationError(
            "prior_test_access counts are incomplete") from exc
    if actual_counts != expected_counts:
        raise PublicationValidationError(
            "prior_test_access counts must be exactly 76/60/7/0")
    if expected.test_overlap:
        raise PublicationValidationError(
            "publication current test overlap must be empty")
    return dict(access)


def _metadata_group_counts(
    rows: Sequence[Mapping[str, object]],
    test_shots: Iterable[int],
) -> dict[str, dict[str, int]]:
    test_set = set(int(shot) for shot in test_shots)
    selected = [row for row in rows if int(row["shot"]) in test_set]
    return {
        column: dict(Counter(str(row[column]) for row in selected))
        for column in METADATA_FIELDS[1:]
    }


def _validate_policy(
    value: object,
    *,
    metadata_rows: Sequence[Mapping[str, object]],
    metadata_bytes: bytes,
    membership: PublicationMembership,
    gap_policy: Mapping[str, object],
    exclusion_policy: Mapping[str, object],
    prior_test_access: Mapping[str, object],
    project_root: pathlib.Path,
) -> dict[str, object]:
    policy = _require_mapping(value, label="strata policy")
    if set(policy) != _REQUIRED_POLICY_FIELDS:
        raise PublicationValidationError(
            "strata policy must contain exactly the frozen required fields")
    if policy.get("schema") != PUBLICATION_POLICY_SCHEMA:
        raise PublicationValidationError(
            f"strata policy schema must be {PUBLICATION_POLICY_SCHEMA!r}")
    _validate_exact_policy_section(
        policy, "dimensions", PUBLICATION_POLICY_DIMENSIONS)

    thresholds = _require_mapping(
        _required(policy, "thresholds", label="strata policy"),
        label="strata policy thresholds",
    )
    threshold_fields = tuple(field.name for field in dataclasses.fields(
        MetadataThresholds))
    if set(thresholds) != set(threshold_fields):
        raise PublicationValidationError(
            "strata policy thresholds must contain exactly four metadata values")
    threshold_values = np.asarray([
        _strict_number(
            thresholds[key], label=f"strata policy threshold {key}")
        for key in threshold_fields
    ], dtype=np.float64)
    frozen_thresholds = np.asarray(
        [PUBLICATION_THRESHOLD_VALUES[key] for key in threshold_fields],
        dtype=np.float64,
    )
    if not np.array_equal(threshold_values, frozen_thresholds):
        raise PublicationValidationError(
            "strata policy must contain the exact frozen thresholds")

    expected_train_hash = canonical_shot_list_sha256(membership.train)
    if _require_sha256(
        _required(policy, "train_shots_sha256", label="strata policy"),
        label="strata policy train_shots_sha256",
    ) != expected_train_hash:
        raise PublicationValidationError(
            "strata policy training shot hash does not match publication train")
    expected_metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
    actual_metadata_hash = _require_sha256(
        _required(policy, "shot_metadata_sha256", label="strata policy"),
        label="strata policy shot_metadata_sha256",
    )
    if actual_metadata_hash != expected_metadata_hash:
        raise PublicationValidationError(
            "strata policy metadata hash does not match metadata bytes")
    if actual_metadata_hash != PUBLICATION_FROZEN_IDENTITY[
            "shot_metadata_sha256"]:
        raise PublicationValidationError(
            "strata policy metadata hash is not the frozen candidate metadata hash")

    sources = _require_mapping(
        _required(policy, "source_sha256", label="strata policy"),
        label="strata policy source_sha256",
    )
    if not sources:
        raise PublicationValidationError(
            "strata policy source_sha256 must not be empty")
    normalized_sources = {}
    for raw_path, raw_digest in sources.items():
        path = _resolve_project_path(
            raw_path,
            project_root=project_root,
            label="source",
        )
        expected = _require_sha256(
            raw_digest, label=f"strata policy source hash for {raw_path}")
        if not path.is_file() or _sha256_file(path) != expected:
            raise PublicationValidationError(
                f"strata policy source hash does not match {raw_path}")
        normalized_sources[raw_path] = expected
    if normalized_sources != PUBLICATION_FROZEN_IDENTITY["source_sha256"]:
        raise PublicationValidationError(
            "strata policy source_sha256 does not match the exact required "
            "immutable source hashes")
    for path_key, hash_key in (
        ("pilot_manifest", "pilot_manifest_sha256"),
        ("pilot_marker", "pilot_marker_sha256"),
    ):
        source = str(exclusion_policy[path_key])
        if source not in sources or str(sources[source]).lower() != str(
                exclusion_policy[hash_key]).lower():
            raise PublicationValidationError(
                f"strata policy source hashes must include {source}")
    _validate_sidecar_tree(
        _required(policy, "sidecar_tree", label="strata policy"),
        project_root=project_root,
    )
    _validate_exact_policy_section(
        policy, "source_coverage_checks", PUBLICATION_SOURCE_COVERAGE_CHECKS)

    actual_counts = _require_mapping(
        _required(policy, "test_group_counts", label="strata policy"),
        label="strata policy test_group_counts",
    )
    expected_counts = _metadata_group_counts(metadata_rows, membership.test)
    if expected_counts != PUBLICATION_TEST_GROUP_COUNTS:
        raise PublicationValidationError(
            "publication metadata must reproduce the exact frozen test-group counts")
    for column, wanted in expected_counts.items():
        actual = _require_mapping(
            actual_counts.get(column),
            label=f"strata policy test_group_counts {column}",
        )
        normalized = {
            str(key): _strict_int(
                count,
                label=f"strata policy test_group_counts {column} {key}",
            )
            for key, count in actual.items()
        }
        if normalized != wanted:
            raise PublicationValidationError(
                f"strata policy test_group_counts {column} does not match metadata")
    _validate_exact_policy_section(
        policy, "test_group_status", PUBLICATION_TEST_GROUP_STATUS)
    minimum_group_size = _strict_int(
        _required(
            policy, "minimum_inferential_test_group_size",
            label="strata policy",
        ),
        label="strata policy minimum_inferential_test_group_size",
    )
    if minimum_group_size != 10:
        raise PublicationValidationError(
            "strata policy minimum inferential test group size must be 10")
    _validate_exact_policy_section(
        policy, "unsupported_dimensions", PUBLICATION_UNSUPPORTED_DIMENSIONS)
    for key, wanted in (
        ("gap_policy", gap_policy),
        ("exclusion_policy", exclusion_policy),
        ("prior_test_access", prior_test_access),
    ):
        actual = _require_mapping(
            _required(policy, key, label="strata policy"),
            label=f"strata policy {key}",
        )
        if actual != wanted:
            raise PublicationValidationError(
                f"strata policy {key} does not match the manifest")
    return dict(policy)


def validate_publication_payloads(
    manifest: Mapping[str, object],
    metadata_rows: Iterable[Mapping[str, object]],
    policy: Mapping[str, object],
    *,
    available_shots: Collection[int] | None = None,
    project_root: pathlib.Path = PROJECT_ROOT,
    manifest_path: str | pathlib.Path | None = None,
) -> PublicationSplit:
    """Validate in-memory publication manifest, metadata, and policy payloads."""
    manifest = _require_mapping(manifest, label="publication manifest")
    version = _strict_int(
        _required(manifest, "version", label="publication manifest"),
        label="publication manifest version",
    )
    identity = {
        "name": PUBLICATION_NAME,
        "time_axis": TIME_AXIS_TOKEN,
        "claim_scope": PUBLICATION_CLAIM_SCOPE,
        "split_strategy": PUBLICATION_SPLIT_STRATEGY,
    }
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise PublicationValidationError(
                f"publication manifest {key} must be {expected!r}")
    if version != PUBLICATION_VERSION:
        raise PublicationValidationError(
            f"publication manifest version must be {PUBLICATION_VERSION!r}")

    parts = {
        part: _manifest_shots(manifest, part)
        for part in ("train", "validation", "test", "excluded")
    }
    for part, expected_count in PUBLICATION_COUNTS.items():
        if len(parts[part]) != expected_count:
            raise PublicationValidationError(
                f"publication manifest {part} must contain {expected_count} shots")
    for part, expected_range in PUBLICATION_RANGES.items():
        actual_range = (parts[part][0], parts[part][-1])
        if actual_range != expected_range:
            raise PublicationValidationError(
                f"publication manifest {part} range must be {expected_range}")
    if parts["excluded"] != PUBLICATION_EXCLUDED:
        raise PublicationValidationError(
            "publication manifest must contain the exact nine exclusions")

    corpus = tuple(sorted(
        parts["train"] + parts["validation"]
        + parts["test"] + parts["excluded"]
    ))
    if len(corpus) != 759 or len(set(corpus)) != 759:
        raise PublicationValidationError(
            "publication manifest union must contain 759 unique shots")
    if (corpus[0], corpus[-1]) != (57281, 58511):
        raise PublicationValidationError(
            "publication corpus range must be 57281 through 58511")
    if available_shots is not None:
        available = tuple(sorted(
            _strict_int(shot, label="available shot")
            for shot in available_shots
        ))
        if len(available) != len(set(available)) or available != corpus:
            raise PublicationValidationError(
                "publication corpus must equal the complete available-shot set")

    root = pathlib.Path(project_root)
    gap_policy = _validate_gap_policy(
        _required(manifest, "gap_policy", label="publication manifest"))
    exclusion_policy, pilot_test = _validate_exclusion_policy(
        _required(manifest, "exclusion_policy", label="publication manifest"),
        project_root=root,
    )
    membership = PublicationMembership(
        train=parts["train"],
        validation=parts["validation"],
        test=parts["test"],
        excluded=parts["excluded"],
    )
    validate_publication_membership(membership, corpus, pilot_test)
    derived_access = derive_prior_test_access(membership, pilot_test)
    prior_test_access = _validate_prior_test_access(
        _required(manifest, "prior_test_access", label="publication manifest"),
        derived_access,
    )

    rows = _validated_metadata_rows(metadata_rows, expected_shots=corpus)
    metadata_bytes = shot_metadata_csv_bytes(rows)
    _validate_policy(
        policy,
        metadata_rows=rows,
        metadata_bytes=metadata_bytes,
        membership=membership,
        gap_policy=gap_policy,
        exclusion_policy=exclusion_policy,
        prior_test_access=prior_test_access,
        project_root=root,
    )

    metadata_path = _resolve_project_path(
        _required(manifest, "shot_metadata", label="publication manifest"),
        project_root=root,
        label="shot_metadata",
    )
    strata_dir = _resolve_project_path(
        _required(manifest, "slice_strata_dir", label="publication manifest"),
        project_root=root,
        label="slice_strata_dir",
    )
    if manifest_path is None:
        resolved_manifest = (
            root / "configs/splits/communications_physics_campaign_v1.json")
    else:
        resolved_manifest = pathlib.Path(manifest_path)
        if not resolved_manifest.is_absolute():
            resolved_manifest = root / resolved_manifest
    return PublicationSplit(
        name=PUBLICATION_NAME,
        version=PUBLICATION_VERSION,
        time_axis=TIME_AXIS_TOKEN,
        train=membership.train,
        validation=membership.validation,
        test=membership.test,
        shot_metadata=metadata_path,
        slice_strata_dir=strata_dir,
        claim_scope=PUBLICATION_CLAIM_SCOPE,
        split_strategy=PUBLICATION_SPLIT_STRATEGY,
        excluded=membership.excluded,
        gap_policy=_freeze_payload(gap_policy),
        exclusion_policy=_freeze_payload(exclusion_policy),
        prior_test_access=_freeze_payload(prior_test_access),
        manifest_path=resolved_manifest,
        policy_path=strata_dir / "strata_policy.json",
    )


def _read_publication_json(path: pathlib.Path, *, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationValidationError(
            f"cannot read {label} JSON at {path}: {exc}") from exc


def validate_publication_manifest(
    path: str | pathlib.Path,
    *,
    available_shots: Collection[int] | None = None,
    project_root: pathlib.Path = PROJECT_ROOT,
) -> PublicationSplit:
    """Load and validate the complete three-file publication freeze bundle."""
    root = pathlib.Path(project_root)
    manifest_path = pathlib.Path(path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = _require_mapping(
        _read_publication_json(manifest_path, label="publication manifest"),
        label="publication manifest",
    )
    metadata_path = _resolve_project_path(
        _required(manifest, "shot_metadata", label="publication manifest"),
        project_root=root,
        label="shot_metadata",
    )
    strata_dir = _resolve_project_path(
        _required(manifest, "slice_strata_dir", label="publication manifest"),
        project_root=root,
        label="slice_strata_dir",
    )
    policy_path = strata_dir / "strata_policy.json"
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata_text = metadata_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise PublicationValidationError(
            f"cannot read publication metadata at {metadata_path}: {exc}") from exc
    reader = csv.DictReader(io.StringIO(metadata_text, newline=""))
    if tuple(reader.fieldnames or ()) != METADATA_FIELDS:
        raise PublicationValidationError(
            "publication metadata header must be exactly "
            + ",".join(METADATA_FIELDS))
    rows = _validated_metadata_rows(tuple(reader), csv_shots=True)
    canonical_metadata = shot_metadata_csv_bytes(rows)
    if metadata_bytes != canonical_metadata:
        raise PublicationValidationError(
            "publication metadata must use canonical sorted UTF-8 CSV bytes")
    policy = _require_mapping(
        _read_publication_json(policy_path, label="strata policy"),
        label="strata policy",
    )
    split = validate_publication_payloads(
        manifest,
        rows,
        policy,
        available_shots=available_shots,
        project_root=root,
        manifest_path=manifest_path,
    )
    if split.shot_metadata != metadata_path or split.policy_path != policy_path:
        raise PublicationValidationError(
            "publication metadata or policy path resolution is inconsistent")
    return split


def publication_disclosure(split: PublicationSplit) -> dict[str, object]:
    """Return the JSON-safe split disclosure copied into result artifacts."""
    if not isinstance(split, PublicationSplit):
        raise TypeError("publication_disclosure requires a PublicationSplit")

    return {
        "name": split.name,
        "version": split.version,
        "claim_scope": split.claim_scope,
        "split_strategy": split.split_strategy,
        "counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "excluded": len(split.excluded),
        },
        "ranges": {
            "train": [split.train[0], split.train[-1]],
            "validation": [split.validation[0], split.validation[-1]],
            "test": [split.test[0], split.test[-1]],
        },
        "test": list(split.test),
        "excluded": list(split.excluded),
        "gap_policy": _thaw_payload(split.gap_policy),
        "exclusion_policy": _thaw_payload(split.exclusion_policy),
        "prior_test_access": _thaw_payload(split.prior_test_access),
    }


def load_split_for_mode(
    path: str | pathlib.Path,
    *,
    manifest_mode: str,
    available_shots: Collection[int] | None,
    project_root: pathlib.Path = PROJECT_ROOT,
) -> FrozenSplit:
    """Load either the strict publication bundle or the unchanged generic split."""
    if manifest_mode == "publication":
        return validate_publication_manifest(
            path,
            available_shots=available_shots,
            project_root=project_root,
        )
    if manifest_mode == "generic":
        return load_split(
            path,
            available_shots=available_shots,
            project_root=project_root,
        )
    raise ValueError(f"unknown manifest mode: {manifest_mode}")
