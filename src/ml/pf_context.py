from __future__ import annotations

import dataclasses
import numpy as np

CADENCE_REF_SECONDS = 0.002048
SCORE_BLOCK = 512
ATTENTION_DEPTH = 6


@dataclasses.dataclass(frozen=True)
class ContextLevel:
    label: str
    nominal_samples: int
    seconds: float

    @property
    def per_layer_seconds(self) -> float:
        return self.seconds / ATTENTION_DEPTH


CONTEXT_LEVELS = tuple(
    ContextLevel(f"h{samples:04d}", samples,
                 round((samples - 1) * CADENCE_REF_SECONDS, 6))
    for samples in (1, 32, 128, 256, 512, 1024, 2048)
)
_CONTEXT_BY_LABEL = {x.label: x for x in CONTEXT_LEVELS}


def context_level(label: str) -> ContextLevel:
    try:
        return _CONTEXT_BY_LABEL[str(label)]
    except KeyError as exc:
        allowed = ", ".join(_CONTEXT_BY_LABEL)
        raise ValueError(
            f"context must be one of {allowed}, got {label!r}") from exc


@dataclasses.dataclass(frozen=True)
class ScoredWindow:
    window_start: int
    window_end: int
    block_start: int
    block_end: int


def scored_windows(time, level: ContextLevel, score_block: int = SCORE_BLOCK):
    t = np.asarray(time, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("native time must be a non-empty one-dimensional array")
    if not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("native time must be finite and strictly increasing")
    if score_block <= 0:
        raise ValueError("score_block must be positive")
    out = []
    for block_start in range(0, t.size, int(score_block)):
        block_end = min(block_start + int(score_block), t.size)
        cutoff = t[block_start] - level.seconds
        window_start = int(np.searchsorted(t, cutoff, side="left"))
        out.append(ScoredWindow(
            window_start, block_end, block_start, block_end))
    return out


import torch


def layer_attention_mask(time, history_valid, real,
                         context_seconds: float, depth: int = ATTENTION_DEPTH):
    if depth <= 0:
        raise ValueError("depth must be positive")
    if context_seconds < 0:
        raise ValueError("context_seconds must be non-negative")
    if time.ndim != 2 or history_valid.shape != time.shape or real.shape != time.shape:
        raise ValueError("time, history_valid, and real must have shape (B,L)")
    dt = time[:, :, None] - time[:, None, :]
    edge = float(context_seconds) / int(depth)
    allowed = ((dt >= 0.0) & (dt <= edge) &
               real[:, :, None] & real[:, None, :] &
               history_valid[:, None, :])
    diagonal = torch.eye(
        time.shape[1], dtype=torch.bool, device=time.device)[None]
    allowed = allowed | diagonal
    return allowed[:, None, :, :]


# ── Task 5: the frozen run matrix and its fingerprints ───────────────
import hashlib  # noqa: E402
import pathlib  # noqa: E402

from .pf_observability import sha256_file, sha256_tree  # noqa: E402

# Complete recursive Work 3 source closure. Every member is required so a
# removal cannot silently narrow a production fingerprint.
PFCTX_SOURCE_FILES = (
    "src/ml/models.py",
    "src/ml/pos_encoding.py",
    "src/ml/dataset.py",
    "src/data/imas_flat_top.py",
    "src/ml/dcs_features.py",
    "src/ml/target.py",
    "src/ml/axis_frame.py",
    "src/ml/metrics.py",
    "src/proj_config.py",
    "src/utils.py",
    "src/ml/pf_observability.py",
    "src/ml/publication_split.py",
    "src/ml/pfobs_provenance.py",
    "src/ml/pfobs_train.py",
    "src/ml/pfobs_infer.py",
    "src/ml/native_metrics.py",
    "src/ml/pf_context.py",
    "src/ml/pfctx_provenance.py",
    "src/ml/pfctx_data.py",
    "src/ml/pfctx_train.py",
    "src/ml/pfctx_infer.py",
    "scripts/run_pf_context_sweep.py",
    "scripts/score_pf_context_sweep.py",
    "exploration/pf_context_analysis.py",
    "scripts/nscc_python_entry.py",
)


def existing_source_files(source_root):
    """Return the complete strict Work 3 source closure, sorted."""
    root = pathlib.Path(source_root)
    missing = [
        str(name) for name in PFCTX_SOURCE_FILES
        if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Work 3 source closure is missing pinned files: {missing}")
    return tuple(sorted(str(name) for name in PFCTX_SOURCE_FILES))


@dataclasses.dataclass(frozen=True)
class MatrixEntry:
    context: ContextLevel
    seed: int
    run_name: str


def matrix_entries(contexts, seeds, prefix="pfctx"):
    entries = []
    for context in contexts:
        for seed in map(int, seeds):
            entries.append(MatrixEntry(
                context, seed, f"{prefix}_{context.label}_s{seed}"))
    if len({x.run_name for x in entries}) != len(entries):
        raise ValueError("duplicate context run name")
    return entries


def normalization_stats_sha256(feature_mean, feature_std, target_mean,
                               target_std):
    """sha256 over the four normalization arrays exactly as a run receives
    and records them (dtype + shape + bytes, feature pair first).

    Byte-identical to the trainer's ``normalization_sha256`` over the same
    four arrays, but defined here so ``--verify-only`` can recompute a run
    fingerprint from an artifact's stored arrays without importing the
    trainer module. Callers must pass float32 arrays -- the artifacts store
    float32, and a float64 hash would never match."""
    digest = hashlib.sha256()
    for array in (feature_mean, feature_std, target_mean, target_std):
        data = np.ascontiguousarray(array)
        digest.update(f"{data.dtype.str}:{data.shape};".encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


_COMMON_FINGERPRINT_FIELDS = {
    "study", "arm", "model", "pe", "time_axis", "config_sha256",
    "split_sha256", "target_meta_sha256", "sidecar_meta_sha256",
    "shot_metadata_sha256", "slice_strata_sha256",
    "availability_audit_sha256", "source_sha256",
}


def context_common_fingerprint(config_path, split_path, target_meta,
                               sidecar_meta, source_root, normalization_hash=None,
                               *, shot_metadata=None, slice_strata_dir=None,
                               availability_audit_sha256="absent",
                               precomputed_hashes=None):
    """Hash invocation-common Work 3 inputs exactly once.

    ``precomputed_hashes`` lets publication training reuse hashes freshly
    validated from the availability-audit snapshot. It may supply any hash
    field but never changes the emitted schema.
    """
    precomputed = dict(precomputed_hashes or {})

    def file_hash(field, path):
        return str(precomputed[field]) if field in precomputed else sha256_file(path)

    def tree_hash(field, root, include=None):
        if field in precomputed:
            return str(precomputed[field])
        return sha256_tree(root, include=include)

    common = {
        "study": "pf_context",
        "arm": "B",
        "model": "ActSeqAttn",
        "pe": "rope_time",
        "time_axis": "native_gmag_bnd",
        "config_sha256": file_hash("config_sha256", config_path),
        "split_sha256": file_hash("split_sha256", split_path),
        "target_meta_sha256": file_hash("target_meta_sha256", target_meta),
        "sidecar_meta_sha256": file_hash("sidecar_meta_sha256", sidecar_meta),
        "shot_metadata_sha256": (
            file_hash("shot_metadata_sha256", shot_metadata)
            if shot_metadata is not None else "absent"),
        "slice_strata_sha256": (
            tree_hash("slice_strata_sha256", slice_strata_dir)
            if slice_strata_dir is not None else "absent"),
        "availability_audit_sha256": str(availability_audit_sha256),
        "source_sha256": tree_hash(
            "source_sha256", source_root,
            include=existing_source_files(source_root)),
    }
    if normalization_hash is not None:
        common["normalization_sha256"] = str(normalization_hash)
    return common


def context_run_fingerprint_from_common(
        common_fingerprint, *, context_seconds, context_label, seed,
        normalization_hash=None):
    """Add only cell identity to one cached common fingerprint mapping."""
    common = dict(common_fingerprint)
    allowed = _COMMON_FINGERPRINT_FIELDS | {"normalization_sha256"}
    if not _COMMON_FINGERPRINT_FIELDS <= set(common) or not set(common) <= allowed:
        raise ValueError("incomplete or foreign Work 3 common fingerprint")
    normalized = (common.get("normalization_sha256")
                  if normalization_hash is None else str(normalization_hash))
    if normalized is None:
        raise ValueError("normalization_hash is required for a run fingerprint")
    return {
        "study": common["study"],
        "arm": common["arm"],
        "model": common["model"],
        "pe": common["pe"],
        "time_axis": common["time_axis"],
        "context_label": str(context_label),
        "context_seconds": float(context_seconds),
        "seed": int(seed),
        "config_sha256": common["config_sha256"],
        "split_sha256": common["split_sha256"],
        "target_meta_sha256": common["target_meta_sha256"],
        "sidecar_meta_sha256": common["sidecar_meta_sha256"],
        "shot_metadata_sha256": common["shot_metadata_sha256"],
        "slice_strata_sha256": common["slice_strata_sha256"],
        "normalization_sha256": normalized,
        "availability_audit_sha256": common["availability_audit_sha256"],
        "source_sha256": common["source_sha256"],
    }


def context_run_fingerprint(config_path, split_path, target_meta,
                            sidecar_meta, source_root, context_seconds,
                            context_label, seed, normalization_hash, *,
                            shot_metadata=None, slice_strata_dir=None,
                            availability_audit_sha256="absent"):
    """Build the complete fingerprint without a caller-owned common cache."""
    common = context_common_fingerprint(
        config_path, split_path, target_meta, sidecar_meta, source_root,
        normalization_hash,
        shot_metadata=shot_metadata,
        slice_strata_dir=slice_strata_dir,
        availability_audit_sha256=availability_audit_sha256,
    )
    return context_run_fingerprint_from_common(
        common,
        context_seconds=context_seconds,
        context_label=context_label,
        seed=seed,
    )


# ── Task 7: validation saturation selection and frozen final inference ─
import pandas as pd  # noqa: E402

# The frozen selection constants (configs/dcs_pf_context_sweep.yml:selection
# pins the same numbers; the analysis re-checks that file against them).
SELECTION_ANCHOR = "h2048"
SELECTION_MARGIN_MM = 1.0
SELECTION_ALPHA = 0.05
SELECTION_BOOTSTRAP_RESAMPLES = 10_000
SELECTION_BOOTSTRAP_SEED = 20_260_820
SELECTION_SEEDS = (0, 1, 2, 3, 4)


def confidence_table(contexts, medians, lowers, uppers):
    """The per-context interval table: index = context label, columns
    ``median`` / ``lower`` / ``upper`` (mm, differences against the anchor
    -- positive means the context is worse)."""
    return pd.DataFrame(
        {"median": np.asarray(medians, dtype=np.float64),
         "lower": np.asarray(lowers, dtype=np.float64),
         "upper": np.asarray(uppers, dtype=np.float64)},
        index=[str(context) for context in contexts])


def _confidence_table_to_json(intervals):
    """The JSON view of a :func:`confidence_table` (empty table -> {})."""
    if intervals is None or len(intervals) == 0:
        return {}
    return {str(label): {"median": float(row["median"]),
                         "lower": float(row["lower"]),
                         "upper": float(row["upper"])}
            for label, row in intervals.iterrows()}


@dataclasses.dataclass(frozen=True)
class Selection:
    """One frozen validation selection (or its unresolved outcome).

    ``verdict`` is ``saturated`` (a shortest equivalent tail exists),
    ``not_reached`` (only the anchor qualified), or
    ``unresolved_nonmonotone`` (a shorter context is materially BETTER than
    the anchor, so no monotone tail exists).  ``equivalent`` maps every
    shorter label to its validation-equivalence (the anchor is trivially
    True); ``intervals`` is the Bonferroni-simultaneous confidence table
    of the six shorter-vs-anchor contrasts."""

    verdict: str
    selected: str | None
    predecessor: str | None
    intervals: object
    equivalent: dict
    materially_better_contexts: tuple

    def to_json_dict(self):
        return {
            "verdict": str(self.verdict),
            "selected": (None if self.selected is None
                         else str(self.selected)),
            "predecessor": (None if self.predecessor is None
                            else str(self.predecessor)),
            "equivalent": {str(key): bool(value)
                           for key, value in self.equivalent.items()},
            "materially_better_contexts": [str(x)
                                           for x in
                                           self.materially_better_contexts],
            "intervals": _confidence_table_to_json(self.intervals),
        }


@dataclasses.dataclass(frozen=True)
class FinalConfirmation:
    """The hierarchical final test of one frozen selection (spec section 10).

    ``selected`` / ``predecessor`` are copied from the frozen validation
    selection and NEVER rechosen.  ``confirmed`` is the selected-vs-anchor
    ordinary-95%-interval equivalence test (with >= 4/5 seed-specific
    medians inside the margin); ``predecessor_inferior`` is attempted only
    after confirmation passes and is None otherwise (or when no
    predecessor exists)."""

    verdict: str
    selected: str | None
    predecessor: str | None
    intervals: object
    seed_effects: object
    confirmed: bool
    predecessor_inferior: bool | None
    n_shots: int

    def to_json_dict(self):
        seed_effects = {}
        if self.seed_effects is not None and len(self.seed_effects):
            for label, row in self.seed_effects.iterrows():
                seed_effects[str(label)] = {
                    str(seed): float(value)
                    for seed, value in row.items()
                    if pd.notna(value)}
        return {
            "verdict": str(self.verdict),
            "selected": (None if self.selected is None
                         else str(self.selected)),
            "predecessor": (None if self.predecessor is None
                            else str(self.predecessor)),
            "confirmed": bool(self.confirmed),
            "predecessor_inferior": (None if self.predecessor_inferior is None
                                     else bool(self.predecessor_inferior)),
            "n_shots": int(self.n_shots),
            "intervals": _confidence_table_to_json(self.intervals),
            "seed_effects": seed_effects,
        }


def seed_specific_medians(per_seed_shot, anchor=SELECTION_ANCHOR,
                          seeds=SELECTION_SEEDS):
    """Per-(context, seed) median of the paired per-shot differences
    against ``anchor``: index = context label, one column per seed.

    Paired on the shots the seed shares with the anchor (the full matrix
    scores identical shots, so this is every shot); a seed missing a
    context leaves NaN there, which the >= 4/5 rules count as outside."""
    frame = pd.DataFrame(per_seed_shot)
    for column in ("context", "seed", "shot", "mean_symmetric_mm"):
        if column not in frame.columns:
            raise ValueError(f"the per-seed shot table needs a {column!r} "
                             "column")
    medians = {}
    for seed in [int(s) for s in seeds]:
        seed_rows = frame[frame["seed"] == seed]
        anchor_curve = (seed_rows[seed_rows["context"] == anchor]
                        .set_index("shot")["mean_symmetric_mm"])
        if anchor_curve.empty:
            continue
        for context, group in seed_rows.groupby("context"):
            curve = group.set_index("shot")["mean_symmetric_mm"]
            common = curve.index.intersection(anchor_curve.index)
            if len(common) == 0:
                continue
            medians.setdefault(seed, {})[str(context)] = float(np.median(
                curve.loc[common].to_numpy()
                - anchor_curve.loc[common].to_numpy()))
    return pd.DataFrame(medians)


def bootstrap_context_differences(per_seed_shot, contexts, anchor="h2048",
                                  resamples=10000, alpha=0.05, seed=20260820):
    """Paired shot bootstrap of every context-vs-anchor median difference.

    The five seeds are averaged within (context, shot) FIRST, the paired
    differences are formed on identical shots, shots are resampled with
    replacement, and the centre is the median of the paired differences.
    The interval is Bonferroni-simultaneous: tail ``alpha / (2 * len(
    contexts))`` each side, so screening all six shorter contexts at
    ``alpha = 0.05`` gives the 99.1667% per-comparison coverage while a
    single-context call (the frozen final contrasts) gives the ordinary
    two-sided 95%.  A pure function of the inputs and the seed (PCG64)."""
    shot_means = (per_seed_shot.groupby(["context", "shot"], as_index=False)
                  ["mean_symmetric_mm"].mean())
    pivot = shot_means.pivot(
        index="shot", columns="context", values="mean_symmetric_mm")
    required = list(contexts) + [anchor]
    pivot = pivot.dropna(subset=required)
    if pivot.empty:
        raise ValueError(
            f"bootstrap_context_differences: no shot is common to {required}"
            " -- paired differences cannot be formed")
    differences = pivot[list(contexts)].to_numpy() - pivot[[anchor]].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(contexts)), dtype=np.float64)
    for i in range(resamples):
        index = rng.integers(0, differences.shape[0], differences.shape[0])
        draws[i] = np.median(differences[index], axis=0)
    tail = alpha / (2.0 * len(contexts))
    return confidence_table(
        contexts, np.median(differences, axis=0),
        np.quantile(draws, tail, axis=0),
        np.quantile(draws, 1.0 - tail, axis=0))


def select_validation_context(per_seed_shot, margin_mm=1.0, seed=20260820):
    """The frozen validation selection rule (spec section 8.1).

    A shorter context is validation-equivalent to the anchor only when its
    simultaneous interval lies entirely inside +/- ``margin_mm`` AND at
    least four of five seed-specific medians lie inside the same margin.
    ``H*`` is the shortest context whose whole tail is equivalent (the
    monotone-tail requirement); a materially BETTER shorter context makes
    the curve unresolved non-monotone and selects nothing."""
    shorter = [x.label for x in CONTEXT_LEVELS[:-1]]
    intervals = bootstrap_context_differences(
        per_seed_shot, shorter, resamples=10000, alpha=0.05, seed=seed)
    seed_effects = seed_specific_medians(per_seed_shot, anchor="h2048")
    equivalent = {}
    for label in shorter:
        row = intervals.loc[label]
        inside = row.lower > -margin_mm and row.upper < margin_mm
        stable = int((seed_effects.loc[label].abs() < margin_mm).sum()) >= 4
        equivalent[label] = bool(inside and stable)
    materially_better = [
        label for label in shorter
        if (intervals.loc[label].upper < -margin_mm and
            int((seed_effects.loc[label] < -margin_mm).sum()) >= 4)]
    if materially_better:
        return Selection(
            "unresolved_nonmonotone", None, None,
            intervals, equivalent, tuple(materially_better))
    equivalent["h2048"] = True
    labels = [x.label for x in CONTEXT_LEVELS]
    candidates = [
        label for i, label in enumerate(labels)
        if all(equivalent[x] for x in labels[i:])]
    selected = candidates[0]
    predecessor = labels[labels.index(selected) - 1] if selected != labels[0] else None
    verdict = "not_reached" if selected == "h2048" else "saturated"
    return Selection(verdict, selected, predecessor, intervals, equivalent, ())


def confirm_final_context(selection, per_seed_shot,
                          margin_mm=SELECTION_MARGIN_MM,
                          anchor=SELECTION_ANCHOR,
                          resamples=SELECTION_BOOTSTRAP_RESAMPLES,
                          alpha=SELECTION_ALPHA,
                          seed=SELECTION_BOOTSTRAP_SEED):
    """The hierarchical final confirmation of the FROZEN selection.

    Step 1 (saturation confirmation): the selected context's ordinary
    two-sided 95% percentile-bootstrap interval versus the anchor must lie
    entirely inside +/- ``margin_mm``, with >= 4/5 seed-specific medians
    inside the margin.  Step 2 (attempted only after step 1 passes, and
    only when a shorter predecessor exists): the predecessor is materially
    worse when the interval's LOWER bound exceeds +``margin_mm`` and >=
    4/5 seed-specific effects point that direction.  Every contrast is a
    single-context bootstrap call, so each interval is the ordinary 95%
    (spec section 10 -- no Bonferroni tail here) and the shared seed gives
    common random numbers.  The selection itself is never revisited: a
    failed confirmation reports failure, never a different level."""
    selected = selection.selected
    empty = confidence_table([], [], [], [])
    if selected is None:
        return FinalConfirmation(str(selection.verdict), None, None, empty,
                                 pd.DataFrame(), False, None, 0)
    n_shots = int(per_seed_shot[per_seed_shot["context"] == anchor]
                  .groupby("shot").ngroups)
    if selected == anchor:
        # only a context shorter than the anchor is tested against it
        return FinalConfirmation("confirmed", str(selected),
                                 (None if selection.predecessor is None
                                  else str(selection.predecessor)),
                                 empty, pd.DataFrame(), True, None, n_shots)
    contrasts = [str(selected)]
    if selection.predecessor is not None:
        contrasts.append(str(selection.predecessor))
    intervals = pd.concat([
        bootstrap_context_differences(
            per_seed_shot, [label], anchor=anchor, resamples=resamples,
            alpha=alpha, seed=seed)
        for label in contrasts])
    seed_effects = seed_specific_medians(per_seed_shot, anchor=anchor)
    row = intervals.loc[str(selected)]
    inside = bool(row["lower"] > -margin_mm and row["upper"] < margin_mm)
    stable = int((seed_effects.loc[str(selected)].abs() < margin_mm)
                 .sum()) >= 4
    confirmed = bool(inside and stable)
    predecessor_inferior = None
    if confirmed and selection.predecessor is not None:
        predecessor = str(selection.predecessor)
        prow = intervals.loc[predecessor]
        direction = int((seed_effects.loc[predecessor] > 0).sum()) >= 4
        predecessor_inferior = bool(prow["lower"] > margin_mm and direction)
    verdict = "confirmed" if confirmed else "failed_to_confirm"
    return FinalConfirmation(verdict, str(selected),
                             (None if selection.predecessor is None
                              else str(selection.predecessor)),
                             intervals, seed_effects, confirmed,
                             predecessor_inferior, n_shots)
