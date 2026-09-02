# -*- coding: utf-8 -*-
"""Run the PF-context (context, seed) validation matrix inside one torchrun job.

Every cell trains the frozen Arm B contract through
``src.ml.pfctx_train.train_one`` and writes ONLY a validation artifact:
``<out-root>/<run_name>/m3.pt`` plus this runner's ``fingerprint.json``. The
exact runner fingerprint is stored identically in both files; the trainer adds
only its scored-index hash separately.
Nothing here loads held-back shots, produces predictions or computes final
metrics -- those are Task 6's, explicitly gated stages, and this script never
references the final scorer.

The runner owns the process group's lifetime: ONE DistEnv is initialized
before the matrix loop and passed into every training call, a barrier
separates consecutive run decisions, and the group is destroyed once in a
top-level finally. ``train_one`` never initializes or destroys the group.

Resume: a cell is skipped only when its artifact AND its fingerprint.json
exist AND the stored payload matches the freshly computed one on EVERY
field. Any mismatch -- including a changed pinned source file or config --
is an ERROR: a stored run is never overwritten and there is no force flag;
move the directory aside or pass a different ``--run-prefix``. Two artifact
markers guard what fingerprints cannot see: ``claim_scope: 'smoke_only'``
is rejected under a production prefix, and ``claim_scope: 'dry_run'`` (the
test-path stub's marker; its fingerprint inputs are identical to a real
run's) is refused by every real run and by ``--verify-only``.

Smoke scope (the only sanctioned way to shrink a run): the manifest must
carry ``claim_scope: pilot_only``, the run prefix must contain ``smoke``,
and BOTH shot limits must be positive. The limited datasets are the first
sorted train/validation shots and every such artifact records
``claim_scope: smoke_only``, rejected under any production prefix.
``--epochs`` is the spec-6.4 engineering-pilot override, a separate gate:
accepted with a pilot_only manifest regardless of prefix or shot limits
(the resource-measurement pilot passes no limits and a plain prefix), and
rejected for any non-pilot manifest. In every epochs-override case the
frozen loop runs the requested epoch count through a wrapper around
``run_epochs`` -- the validated config itself is never mutated (the
validator pins hp.epochs=80) -- and the artifact keeps its natural
manifest scope.

``--verify-only`` loads metadata and artifacts only: it never imports the
trainer module and never opens a target NPZ file (the per-cell
normalization hash is recomputed from the artifact's own stored arrays).
``--audit-only`` reads the train and validation shots only and writes
``ProjDB/Stats/pf_context/context_availability.csv`` with the per
shot/context cadence, gap, visible-token, full-horizon, score-row and
history-valid coverage. Publication mode then writes the split/data/source-
bound ``context_availability.audit.json`` identity last.

Usage (inside a torchrun / qsub-trun job, from the repo root):
  torchrun --nproc_per_node 4 scripts/run_pf_context_sweep.py \
    --split configs/splits/pfobs_random_pilot.json --manifest-mode generic
  python scripts/run_pf_context_sweep.py --audit-only --manifest-mode generic \
    --split configs/splits/pfobs_random_pilot.json
  python scripts/run_pf_context_sweep.py --verify-only --manifest-mode generic \
    --split configs/splits/pfobs_random_pilot.json
"""
import argparse
import csv
import dataclasses
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.pf_context import (  # noqa: E402
    CONTEXT_LEVELS, ContextLevel, context_common_fingerprint, context_level,
    context_run_fingerprint, context_run_fingerprint_from_common,
    matrix_entries, normalization_stats_sha256,
)
from src.ml.pf_observability import FrozenSplit, sha256_file  # noqa: E402
from src.ml.publication_split import (  # noqa: E402
    PUBLICATION_MANIFEST_PATH,
    PUBLICATION_OUT_ROOT,
    PUBLICATION_SEEDS,
    PUBLICATION_SIDECAR_DIR,
    PUBLICATION_TARGET_DIR,
    PUBLICATION_WORK3_AUDIT_CSV,
    PUBLICATION_WORK3_AUDIT_IDENTITY,
    PUBLICATION_WORK3_CONFIG,
    PUBLICATION_WORK3_CONTEXTS,
    PUBLICATION_WORK3_FINAL_BUILDING,
    PUBLICATION_WORK3_MARKER,
    PUBLICATION_WORK3_PREFIX,
    PUBLICATION_WORK3_STATS_ROOT,
    load_split_for_mode,
    require_publication_paths,
)
from src.ml.pfctx_data import load_context_series  # noqa: E402
from src.ml.pfctx_provenance import (  # noqa: E402
    AVAILABILITY_COLUMNS,
    availability_csv_snapshot_from_bytes,
    availability_rows_for_shot,
    build_availability_audit_identity,
    canonical_availability_audit_bytes,
    validate_availability_audit_identity,
)
from src.ml.pfobs_provenance import ValidationFreezeLock  # noqa: E402
from src.ml.pfobs_train import (  # noqa: E402
    dist_barrier, dist_broadcast, init_dist, teardown_dist,
)
from src.utils import (  # noqa: E402
    PublicationIOError, durable_publish_bytes, read_regular_nofollow,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET_DIR = PUBLICATION_TARGET_DIR
SIDECAR_DIR = PUBLICATION_SIDECAR_DIR
CONFIG = PUBLICATION_WORK3_CONFIG
SPLIT = PUBLICATION_MANIFEST_PATH
OUT_ROOT = PUBLICATION_OUT_ROOT
STATS_ROOT = PUBLICATION_WORK3_STATS_ROOT
STATS_CSV = PUBLICATION_WORK3_AUDIT_CSV
AUDIT_IDENTITY = PUBLICATION_WORK3_AUDIT_IDENTITY
DEFAULT_PREFIX = PUBLICATION_WORK3_PREFIX
PILOT_MANIFEST = "pfobs_random_pilot.json"
CAMPAIGN_MANIFEST = "communications_physics_campaign_v1.json"

DECISION_RUN, DECISION_SKIP, DECISION_ABORT = 0, 1, 2

# One row per (shot, context): the memory-timescale feasibility readout.
AUDIT_COLUMNS = AVAILABILITY_COLUMNS


def _context_labels(contexts):
    return tuple(
        value.label if isinstance(value, ContextLevel) else str(value)
        for value in contexts
    )


def _require_publication_training_invocation(
        *, split, contexts, seeds, config_path, target_dir, sidecar_dir,
        out_root, prefix, source_root, audit_identity, stats_root, epochs,
        max_train_shots, max_validation_shots, dry_train):
    """Reject publication namespace, prefix, matrix, and override downgrades."""
    require_publication_paths(
        "publication",
        {
            "split": _split_path(split),
            "config": config_path,
            "target_dir": target_dir,
            "sidecar_dir": sidecar_dir,
            "out_root": out_root,
            "stats_root": stats_root,
            "audit_identity": audit_identity,
            "source_root": source_root,
        },
        {
            "split": SPLIT,
            "config": CONFIG,
            "target_dir": TARGET_DIR,
            "sidecar_dir": SIDECAR_DIR,
            "out_root": OUT_ROOT,
            "stats_root": STATS_ROOT,
            "audit_identity": AUDIT_IDENTITY,
            "source_root": pathlib.Path(CONFIG).parents[1],
        },
    )
    if _context_labels(contexts) != tuple(PUBLICATION_WORK3_CONTEXTS):
        raise ValueError(
            "publication --contexts must be the canonical seven-context matrix")
    if tuple(int(seed) for seed in seeds) != tuple(PUBLICATION_SEEDS):
        raise ValueError(
            "publication --seeds must be the canonical 0 1 2 3 4 matrix")
    if str(prefix) != PUBLICATION_WORK3_PREFIX:
        raise ValueError(
            "publication --run-prefix must be the frozen 'pfctx' prefix")
    if any(value is not None for value in (
            epochs, max_train_shots, max_validation_shots)) or dry_train:
        raise ValueError(
            "publication training rejects smoke/dry/epoch overrides; use "
            "explicit generic mode")


def _publication_training_state_paths(stats_root):
    stats_root = pathlib.Path(stats_root)
    return (
        stats_root / "validation_selection.json",
        stats_root / "validation_building",
        stats_root.parent / (stats_root.name + ".final_building"),
        stats_root / "final_test" / "transaction_provenance.json",
        stats_root / "final_test",
        stats_root / "PFCTX_FINAL_TEST_EVALUATED.json",
    )


def _reject_publication_training_state(stats_root):
    for path in _publication_training_state_paths(stats_root):
        if os.path.lexists(path):
            raise RuntimeError(
                f"{path} exists: publication training is locked after "
                "selection, staging, or final-state creation")


def _require_publication_cli(args, *, operation):
    if args.manifest_mode != "publication":
        return
    require_publication_paths(
        "publication",
        {
            "split": args.split,
            "config": args.config,
            "target_dir": args.target_dir,
            "sidecar_dir": args.sidecar_dir,
            "out_root": args.out_root,
            "stats_root": args.stats_root,
            "audit_identity": args.audit_identity,
        },
        {
            "split": SPLIT,
            "config": CONFIG,
            "target_dir": TARGET_DIR,
            "sidecar_dir": SIDECAR_DIR,
            "out_root": OUT_ROOT,
            "stats_root": STATS_ROOT,
            "audit_identity": AUDIT_IDENTITY,
        },
    )
    if _context_labels(args.contexts) != tuple(PUBLICATION_WORK3_CONTEXTS):
        raise ValueError(
            "publication --contexts must be the canonical seven-context matrix")
    if tuple(int(seed) for seed in args.seeds) != tuple(PUBLICATION_SEEDS):
        raise ValueError(
            "publication --seeds must be the canonical 0 1 2 3 4 matrix")
    if str(args.run_prefix) != PUBLICATION_WORK3_PREFIX:
        raise ValueError(
            "publication --run-prefix must be the frozen 'pfctx' prefix")
    if any(value is not None for value in (
            args.epochs, args.max_train_shots,
            args.max_validation_shots)):
        raise ValueError(
            "publication mode rejects smoke/epoch overrides; use generic mode")
    canonical_marker = pathlib.Path(STATS_ROOT) / \
        "PFCTX_FINAL_TEST_EVALUATED.json"
    if operation in {"training", "audit"} and os.path.lexists(canonical_marker):
        raise RuntimeError(
            f"{canonical_marker} exists: canonical publication state is "
            "globally final")


# ── the scoped split ─────────────────────────────────────────────────
def _load_manifest(path, manifest_mode, available_shots=None):
    """Load the requested split contract once before any matrix work."""
    return load_split_for_mode(
        path,
        manifest_mode=manifest_mode,
        available_shots=available_shots,
        project_root=REPO_ROOT,
    )


@dataclasses.dataclass(frozen=True)
class ScopedSplit(FrozenSplit):
    """A FrozenSplit carrying the manifest's ``claim_scope`` and the manifest
    path the run was fingerprinted against.

    ``load_split`` intentionally drops extra manifest keys, so the runner
    re-attaches them here: the trainer records the scope in every artifact
    through the ``getattr(split, "claim_scope", None)`` seam, and the
    fingerprint hashes the manifest file itself.
    """
    claim_scope: object = None
    split_path: pathlib.Path = None


def load_context_split(path, manifest_mode="publication", available_shots=None):
    """Load a publication or explicitly generic context-sweep manifest.

    A missing manifest is a hard error: this runner NEVER generates or
    substitutes the campaign manifest. The archived pilot manifest remains
    accepted only in generic mode while its ``claim_scope`` is ``pilot_only``.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"split manifest {path} does not exist -- this runner never "
            f"generates or substitutes {CAMPAIGN_MANIFEST}; pass an "
            f"existing manifest (e.g. configs/splits/{PILOT_MANIFEST})")
    data = json.loads(path.read_text())
    claim_scope = data.get("claim_scope")
    if path.name == PILOT_MANIFEST and claim_scope != "pilot_only":
        raise ValueError(
            f"{path.name} is the pilot manifest and is accepted only with "
            f"claim_scope 'pilot_only', got {claim_scope!r}")
    split = _load_manifest(path, manifest_mode, available_shots)
    if manifest_mode == "publication":
        return split
    # The generic loader intentionally drops extra manifest keys. Re-attach
    # the archived scope and exact manifest path used by the fingerprints.
    return ScopedSplit(**vars(split), claim_scope=claim_scope,
                       split_path=path)


def limited_split(split, max_train_shots, max_validation_shots):
    """The smoke-scoped split: the FIRST SORTED train/validation shots,
    deterministically, labelled ``claim_scope: smoke_only``."""
    return dataclasses.replace(
        split,
        train=tuple(sorted(int(s) for s in split.train))[:int(
            max_train_shots)],
        validation=tuple(sorted(int(s) for s in split.validation))[:int(
            max_validation_shots)],
        claim_scope="smoke_only")


def _validate_smoke_request(split, prefix, max_train_shots,
                            max_validation_shots):
    """The shot-limit gate; returns True when the run is smoke-scoped."""
    limits = (max_train_shots, max_validation_shots)
    if all(limit is None for limit in limits):
        return False
    if getattr(split, "claim_scope", None) != "pilot_only":
        raise ValueError(
            "--max-train-shots/--max-validation-shots are accepted only "
            "with a pilot manifest (claim_scope 'pilot_only')")
    if "smoke" not in str(prefix):
        raise ValueError(
            "--max-train-shots/--max-validation-shots are accepted only "
            "with a run prefix containing 'smoke'")
    if any(limit is None for limit in limits):
        raise ValueError(
            "--max-train-shots and --max-validation-shots must be given "
            "together")
    if int(max_train_shots) < 1 or int(max_validation_shots) < 1:
        raise ValueError("shot limits must be positive")
    return True


def _coerce_contexts(contexts):
    return [x if isinstance(x, ContextLevel) else context_level(x)
            for x in contexts]


def _check_shot_availability(split, target_dir, sidecar_dir):
    """Every train and validation shot must exist in both datasets. The
    held-back list is never consulted: this runner does not even ask."""
    def stems(directory):
        return {p.stem for p in pathlib.Path(directory).glob("*.npz")
                if p.stem.isdigit()}
    available = stems(target_dir) & stems(sidecar_dir)
    missing = sorted({str(s) for s in (*split.train, *split.validation)}
                     - available)
    if missing:
        raise ValueError(
            f"shots not available in both datasets: {', '.join(missing[:5])}"
            + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))


# ── normalization (train shots only; four separate arrays) ───────────
def context_normalization(target_dir, sidecar_dir, shots):
    """(feature_mean, feature_std, target_mean, target_std), float32, over
    the score-valid rows of the given TRAIN shots -- the rows the loss sees."""
    feature_rows, target_rows = [], []
    for shot in sorted(int(s) for s in shots):
        series = load_context_series(
            pathlib.Path(target_dir) / f"{shot}.npz", sidecar_dir)
        feature_rows.append(series.features[series.score_valid])
        target_rows.append(series.target[series.score_valid])
    if not any(len(rows) for rows in feature_rows):
        raise ValueError("no score-valid training rows for normalization")
    X = np.concatenate(feature_rows)
    Y = np.concatenate(target_rows)

    def pair(block):
        mean = block.mean(0).astype(np.float32)
        std = np.maximum(block.std(0), 1e-6).astype(np.float32)
        return mean, std

    feature_mean, feature_std = pair(X)
    target_mean, target_std = pair(Y)
    return feature_mean, feature_std, target_mean, target_std


def _broadcast_normalization(shots, target_dir, sidecar_dir, dist_env):
    """Rank 0 computes the train-shot normalization once; every rank
    receives the identical float32 arrays (they feed the fingerprint)."""
    if dist_env.is_main:
        stats = context_normalization(target_dir, sidecar_dir, shots)
    else:
        stats = tuple(np.zeros(n, np.float32) for n in (21, 21, 34, 34))
    return tuple(
        dist_broadcast(
            torch.as_tensor(np.asarray(array, np.float32),
                            device=dist_env.device), dist_env
        ).cpu().numpy().astype(np.float32)
        for array in stats)


# ── fingerprints and resume ──────────────────────────────────────────
def _split_path(split):
    path = (getattr(split, "split_path", None)
            or getattr(split, "manifest_path", None))
    if path is None:
        raise ValueError("split carries no manifest path for fingerprinting")
    return pathlib.Path(path)


def _validated_availability_audit_snapshot(
        split, manifest_mode, audit_identity, target_dir, sidecar_dir,
        source_root):
    """Validate one publication identity snapshot; generic pilots return None."""
    if manifest_mode != "publication":
        return None
    return validate_availability_audit_identity(
        pathlib.Path(audit_identity),
        split_path=_split_path(split),
        split=split,
        target_dir=target_dir,
        sidecar_dir=sidecar_dir,
        project_root=source_root,
    )


def entry_payload(entry, split, config_path, target_dir, sidecar_dir,
                  source_root, stats, availability_audit_sha256="absent"):
    """The complete runner-owned run fingerprint of one matrix cell."""
    return context_run_fingerprint(
        config_path=config_path, split_path=_split_path(split),
        target_meta=pathlib.Path(target_dir) / "meta.json",
        sidecar_meta=pathlib.Path(sidecar_dir) / "meta.json",
        source_root=source_root, context_seconds=entry.context.seconds,
        context_label=entry.context.label, seed=entry.seed,
        normalization_hash=normalization_stats_sha256(*stats),
        shot_metadata=getattr(split, "shot_metadata", None),
        slice_strata_dir=getattr(split, "slice_strata_dir", None),
        availability_audit_sha256=availability_audit_sha256)


def _common_fingerprint(
        split, config_path, target_dir, sidecar_dir, source_root,
        normalization_hash, audit_snapshot):
    precomputed = {}
    availability_sha256 = "absent"
    if audit_snapshot is not None:
        identity = audit_snapshot.identity
        availability_sha256 = audit_snapshot.sha256
        precomputed = {
            "split_sha256": identity["split_sha256"],
            "target_meta_sha256": identity["target_meta_sha256"],
            "sidecar_meta_sha256": identity["sidecar_meta_sha256"],
            "source_sha256": identity["source_sha256"],
        }
    return context_common_fingerprint(
        config_path=config_path,
        split_path=_split_path(split),
        target_meta=pathlib.Path(target_dir) / "meta.json",
        sidecar_meta=pathlib.Path(sidecar_dir) / "meta.json",
        source_root=source_root,
        normalization_hash=normalization_hash,
        shot_metadata=getattr(split, "shot_metadata", None),
        slice_strata_dir=getattr(split, "slice_strata_dir", None),
        availability_audit_sha256=availability_sha256,
        precomputed_hashes=precomputed,
    )


def _broadcast_common_fingerprint_envelope(envelope, dist_env):
    if dist_env.world_size == 1:
        return envelope
    objects = [envelope if dist_env.is_main else None]
    torch.distributed.broadcast_object_list(objects, src=0)
    return objects[0]


def _build_and_broadcast_common_fingerprint(
        split, config_path, target_dir, sidecar_dir, source_root,
        normalization_stats, audit_snapshot, dist_env):
    cause = None
    envelope = None
    if dist_env.is_main:
        try:
            common = _common_fingerprint(
                split, config_path, target_dir, sidecar_dir, source_root,
                normalization_stats_sha256(*normalization_stats),
                audit_snapshot)
            envelope = {"ok": True, "common": common}
        except Exception as exc:  # rank 0 must still join the one broadcast
            cause = exc
            envelope = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    envelope = _broadcast_common_fingerprint_envelope(envelope, dist_env)
    if not isinstance(envelope, dict) or type(envelope.get("ok")) is not bool:
        raise RuntimeError(
            "rank 0 broadcast an invalid common fingerprint envelope")
    if not envelope["ok"]:
        error_type = str(envelope.get("error_type", "Exception"))
        error_message = str(envelope.get("error_message", ""))
        failure = RuntimeError(
            "common fingerprint construction failed on rank 0 "
            f"({error_type}: {error_message})")
        if dist_env.is_main and cause is not None:
            raise failure from cause
        raise failure
    common = envelope.get("common")
    if not isinstance(common, dict):
        raise RuntimeError(
            "rank 0 broadcast an invalid common fingerprint envelope")
    return common


def entry_payload_from_common(entry, common_fingerprint, *,
                              normalization_hash=None):
    """Derive one cell without reopening any common fingerprint input."""
    return context_run_fingerprint_from_common(
        common_fingerprint,
        context_seconds=entry.context.seconds,
        context_label=entry.context.label,
        seed=entry.seed,
        normalization_hash=normalization_hash,
    )


def write_fingerprint(run_dir, fingerprint):
    """Durably publish the exact runner mapping after its matching ``m3.pt``."""
    run_dir = pathlib.Path(run_dir)
    artifact_path = run_dir / "m3.pt"
    artifact_bytes = read_regular_nofollow(
        artifact_path, label="Work 3 training artifact")
    try:
        artifact = torch.load(
            io.BytesIO(artifact_bytes), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PublicationIOError(
            f"cannot reload complete Work 3 m3.pt bytes: {exc}") from exc
    if (not isinstance(artifact, dict)
            or artifact.get("run_fingerprint") != fingerprint):
        raise PublicationIOError(
            "Work 3 fingerprint differs from its complete m3.pt artifact")
    payload = (json.dumps(
        fingerprint, ensure_ascii=False, allow_nan=False,
        indent=2, sort_keys=True) + "\n").encode("utf-8")

    def validate(candidate):
        try:
            parsed = json.loads(candidate.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PublicationIOError(
                f"cannot parse complete Work 3 fingerprint: {exc}") from exc
        if parsed != fingerprint or candidate != payload:
            raise PublicationIOError(
                "Work 3 fingerprint bytes are not the exact runner mapping")
        return parsed

    return durable_publish_bytes(
        run_dir / "fingerprint.json",
        payload,
        validator=validate,
        state_label="Work 3 training fingerprint",
    )


def _stored_payload(run_dir):
    path = pathlib.Path(run_dir) / "fingerprint.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _artifact_payload(run_dir):
    return torch.load(
        pathlib.Path(run_dir) / "m3.pt", map_location="cpu",
        weights_only=False)


def resume_decision(run_dir, fingerprint, prefix, dry_run=False):
    """RUN when no artifact is stored, SKIP on a full-fingerprint match.

    A stored artifact whose fingerprint.json disagrees -- or is missing, or
    whose recorded scope is smoke under a production prefix, or is a
    dry-run stub while the current run is real -- raises: the stored run is
    pinned provenance and is never silently overwritten. Every fingerprint
    field is compared, including exact parity with ``m3.pt.run_fingerprint``.
    """
    run_dir = pathlib.Path(run_dir)
    if not (run_dir / "m3.pt").is_file():
        return DECISION_RUN
    stored = _stored_payload(run_dir)
    if stored is None:
        raise RuntimeError(
            f"{run_dir.name}: m3.pt exists without fingerprint.json -- "
            "provenance cannot be verified; never overwrite (move the "
            "directory aside or pass a different --run-prefix)")
    compared = set(stored) | set(fingerprint)
    differing = [key for key in sorted(compared)
                 if stored.get(key) != fingerprint.get(key)]
    if differing:
        raise RuntimeError(
            f"{run_dir.name}: stored run disagrees with the requested run "
            f"({', '.join(differing)}) -- never overwrite; move the "
            "directory aside or pass a different --run-prefix")
    artifact = _artifact_payload(run_dir)
    artifact_fingerprint = artifact.get("run_fingerprint")
    if artifact_fingerprint != fingerprint:
        if isinstance(artifact_fingerprint, dict):
            differing = [
                key for key in sorted(set(artifact_fingerprint) | set(fingerprint))
                if artifact_fingerprint.get(key) != fingerprint.get(key)
            ]
            detail = ", ".join(differing) if differing else "schema"
        else:
            detail = "schema"
        raise RuntimeError(
            f"{run_dir.name}: artifact run_fingerprint disagrees with "
            f"fingerprint.json ({detail})")
    if artifact.get("fingerprints") != artifact_fingerprint:
        raise RuntimeError(
            f"{run_dir.name}: artifact fingerprint compatibility alias "
            "disagrees with artifact run_fingerprint")
    scope = artifact.get("claim_scope")
    if scope == "dry_run" and not dry_run:
        raise RuntimeError(
            f"{run_dir.name}: stored artifact is a dry-run stub "
            "(claim_scope 'dry_run') -- fingerprints cannot distinguish it "
            "from a trained run, so a real run refuses it; never overwrite "
            "(move the directory aside or pass a different --run-prefix)")
    if scope == "smoke_only" and "smoke" not in str(prefix):
        raise RuntimeError(
            f"{run_dir.name}: stored artifact carries claim_scope "
            f"'smoke_only' and is rejected under the production prefix "
            f"{prefix!r}")
    return DECISION_SKIP


def _broadcast_decision(dist_env, decision, error):
    """Rank 0's decision to every rank; ABORT raises on every rank."""
    flag = torch.tensor([int(decision)], dtype=torch.int64,
                        device=dist_env.device)
    dist_broadcast(flag, dist_env)
    got = int(flag.item())
    if got == DECISION_ABORT and error is None:
        error = RuntimeError(
            "rank 0 aborted the matrix -- see the rank 0 log for the cause")
    return got, error


def _log(dist_env, message):
    if dist_env.is_main:
        print(message, flush=True)


# ── the trainers: production, smoke wrapper, dry stub ────────────────
def _stats_pairs(stats):
    return (stats[0], stats[1]), (stats[2], stats[3])


def _smoke_train_one(pfctx_train, context, seed, split, config, target_dir,
                     sidecar_dir, run_dir, dist_env, stats, epochs,
                     run_fingerprint_payload):
    """The sanctioned epochs-override wrapper (spec 6.4 engineering pilot
    and smoke runs): override ONLY the epoch count the frozen loop runs.
    ``train_one`` still receives the validated config untouched (the
    validator pins hp.epochs=80); the wrapper swaps ``run_epochs`` for a
    shim that runs ``epochs`` epochs instead. The artifact keeps the split's
    natural claim_scope ('smoke_only' only when the shot limits are active)."""
    real_run_epochs = pfctx_train.run_epochs

    def smoke_run_epochs(ddp_model, optimizer, train_loader, validation_loader,
                         inner_context, inner_seed, hp, accumulation, env):
        smoke_hp = dict(hp)
        smoke_hp["epochs"] = int(epochs)
        return real_run_epochs(
            ddp_model, optimizer, train_loader, validation_loader,
            inner_context, inner_seed, smoke_hp, accumulation, env)

    pfctx_train.run_epochs = smoke_run_epochs
    try:
        return pfctx_train.train_one(
            context, seed, split, config, target_dir, sidecar_dir, run_dir,
            dist_env, *_stats_pairs(stats),
            run_fingerprint_payload=run_fingerprint_payload)
    finally:
        pfctx_train.run_epochs = real_run_epochs


def _dry_train_one(context, seed, split, config, target_dir, sidecar_dir,
                   run_dir, dist_env, stats, run_fingerprint_payload):
    """Test-path stub: no training, but a contract-complete artifact (an
    untrained production model plus the neutral engineering readouts).

    The stub writes the split re-labelled ``claim_scope: 'dry_run'``: every
    fingerprint input (config, split file, metas, normalization, source
    hash) is identical between dry and real runs, so the artifact itself
    must carry the marker that keeps a dry stub from ever skipping or
    passing as a trained run.
    """
    from src.ml.models import ActSeqAttn  # noqa: E402  (not the trainer)
    trainer = pfctx_train_module()
    hp = config["hp"]
    torch.manual_seed(int(seed))
    model = ActSeqAttn(n_act=21, n_out=34, d=hp["d_model"], heads=hp["heads"],
                       depth=hp["depth"], ffn=hp["ffn"],
                       dropout=hp["dropout"], pe="rope_time")
    accumulation, effective = trainer.microbatch_contract(
        hp["effective_global_batch"], dist_env.world_size,
        hp["microbatch_per_rank"])
    elapsed = time.perf_counter()
    result = {
        "best_val_mse": 0.0, "best_epoch": 0, "stop_epoch": 0,
        "epochs_completed": 0, "n_train_windows": 0,
        "n_validation_windows": 0,
        "elapsed_seconds": time.perf_counter() - elapsed,
        "seconds_per_epoch": 0.0, "mean_sequence_tokens": 0.0,
        "peak_memory_bytes": 0, "attention_backend_ops": [],
        "scored_index_sha256": "0" * 64,
        "sidecar_meta_sha256": sha256_file(
            pathlib.Path(sidecar_dir) / "meta.json"),
        "target_meta_sha256": sha256_file(
            pathlib.Path(target_dir) / "meta.json"),
    }
    if dist_env.is_main:
        trainer.write_context_artifact_atomic(
            run_dir, model, result, context, seed,
            dataclasses.replace(split, claim_scope="dry_run"), config,
            *_stats_pairs(stats), dist_env.world_size, effective,
            accumulation,
            run_fingerprint_payload=run_fingerprint_payload)
    return result


def pfctx_train_module():
    """The ONE lazy import of the trainer module: only the training path
    (production, smoke wrapper, dry stub) ever calls this, so --verify-only
    and --audit-only keep it out of the process."""
    from src.ml import pfctx_train
    return pfctx_train


def _train_entry(entry, split, config, target_dir, sidecar_dir, run_dir,
                 dist_env, stats, epochs, dry_train,
                 run_fingerprint_payload):
    if dry_train:
        return _dry_train_one(entry.context, entry.seed, split, config,
                              target_dir, sidecar_dir, run_dir, dist_env,
                              stats, run_fingerprint_payload)
    trainer = pfctx_train_module()
    if epochs is None:
        return trainer.train_one(
            entry.context, entry.seed, split, config, target_dir, sidecar_dir,
            run_dir, dist_env, *_stats_pairs(stats),
            run_fingerprint_payload=run_fingerprint_payload)
    return _smoke_train_one(
        trainer, entry.context, entry.seed, split, config, target_dir,
        sidecar_dir, run_dir, dist_env, stats, epochs,
        run_fingerprint_payload)


# ── the matrix loop ──────────────────────────────────────────────────
def _run_validation_matrix_locked(
        *, split, contexts=None, seeds=None, config_path=None,
        target_dir=None, sidecar_dir=None, out_root=None,
        prefix=DEFAULT_PREFIX, epochs=None,
        max_train_shots=None, max_validation_shots=None,
        dry_train=False, dist_env=None, source_root=None,
        manifest_mode="generic", audit_identity=None, stats_root=None):
    """Train every outstanding cell of the requested matrix; resumable.

    ``split`` is a :class:`ScopedSplit` from :func:`load_context_split`.
    The frozen config reaches ``train_one`` unmodified except in the
    smoke-scoped case, where the epochs wrapper runs instead. Rank 0 makes
    every skip/abort decision and broadcasts it so all ranks stay in
    lockstep.
    """
    config_path = pathlib.Path(CONFIG if config_path is None else config_path)
    target_dir = pathlib.Path(TARGET_DIR if target_dir is None
                              else target_dir)
    sidecar_dir = pathlib.Path(SIDECAR_DIR if sidecar_dir is None
                               else sidecar_dir)
    out_root = pathlib.Path(OUT_ROOT if out_root is None else out_root)
    source_root = pathlib.Path(REPO_ROOT if source_root is None else source_root)
    stats_root = pathlib.Path(STATS_ROOT if stats_root is None else stats_root)
    audit_identity = pathlib.Path(
        AUDIT_IDENTITY if audit_identity is None else audit_identity)
    if manifest_mode == "publication":
        _reject_publication_training_state(stats_root)
    audit_snapshot = _validated_availability_audit_snapshot(
        split, manifest_mode, audit_identity, target_dir, sidecar_dir,
        source_root)
    config = yaml.safe_load(config_path.read_text())
    pfctx_train_module().validate_context_config(config)

    smoke_scoped = _validate_smoke_request(split, prefix, max_train_shots,
                                           max_validation_shots)
    if epochs is not None:
        if int(epochs) < 1:
            raise ValueError("epochs must be >= 1")
        if getattr(split, "claim_scope", None) != "pilot_only" \
                and not dry_train:
            raise ValueError(
                "--epochs is the spec-6.4 engineering-pilot override: it is "
                "accepted only with a pilot manifest (claim_scope "
                "'pilot_only') or with dry_train=True; the 80-epoch "
                "production fits of a non-pilot manifest never override it")
    effective_split = (limited_split(split, max_train_shots,
                                     max_validation_shots)
                       if smoke_scoped else split)
    levels = _coerce_contexts(
        [x.label for x in CONTEXT_LEVELS] if contexts is None else contexts)
    seed_list = [int(s) for s in config["seeds"]] if seeds is None else [
        int(s) for s in seeds]
    entries = matrix_entries(levels, seed_list, prefix=prefix)
    _check_shot_availability(effective_split, target_dir, sidecar_dir)

    owned = dist_env is None
    dist_env = init_dist() if owned else dist_env
    try:
        _log(dist_env, f"matrix: {len(entries)} runs, contexts "
              f"{' '.join(x.label for x in levels)}, seeds "
              f"{' '.join(str(s) for s in seed_list)}, prefix {prefix!r}, "
              f"scope {getattr(effective_split, 'claim_scope', None)!r}, "
              f"world {dist_env.world_size}")
        _log(dist_env, f"target {target_dir}\nsidecar {sidecar_dir}"
              f"\nconfig {config_path}\nsplit {_split_path(effective_split)}\n"
              f"out-root {out_root}")
        stats = _broadcast_normalization(effective_split.train, target_dir,
                                         sidecar_dir, dist_env)
        common_fingerprint = _build_and_broadcast_common_fingerprint(
            effective_split, config_path, target_dir, sidecar_dir,
            source_root, stats, audit_snapshot, dist_env)
        trained = skipped = 0
        for index, entry in enumerate(entries):
            tag = f"[{index + 1}/{len(entries)}]"
            run_dir = out_root / entry.run_name
            decision, error, payload = DECISION_RUN, None, None
            if dist_env.is_main:
                try:
                    payload = entry_payload_from_common(
                        entry, common_fingerprint)
                    decision = resume_decision(run_dir, payload, prefix,
                                               dry_run=dry_train)
                except Exception as exc:
                    decision, error = DECISION_ABORT, exc
            decision, error = _broadcast_decision(dist_env, decision, error)
            if decision == DECISION_ABORT:
                raise error
            if decision == DECISION_SKIP:
                skipped += 1
                _log(dist_env, f"{tag} skip {entry.run_name} (artifact + "
                      "matching fingerprint)")
                dist_barrier(dist_env)
                continue
            _log(dist_env, f"{tag} run {entry.run_name}")
            if payload is None:
                payload = entry_payload_from_common(
                    entry, common_fingerprint)
            result = _train_entry(
                entry, effective_split, config, target_dir, sidecar_dir,
                run_dir, dist_env, stats, epochs, dry_train, payload)
            if dist_env.is_main:
                write_fingerprint(run_dir, payload)
            trained += 1
            _log(dist_env, f"{tag} done {entry.run_name} best_val_mse="
                  f"{float(result['best_val_mse']):.6f} stop_epoch="
                  f"{int(result['stop_epoch'])}")
            dist_barrier(dist_env)
        _log(dist_env, f"matrix complete: {trained} trained, {skipped} "
              f"skipped, {trained + skipped} cells")
        return 0
    finally:
        if owned:
            teardown_dist(dist_env)


def run_validation_matrix(
        split, contexts=None, seeds=None, config_path=None,
        target_dir=None, sidecar_dir=None, out_root=None,
        prefix=DEFAULT_PREFIX, epochs=None,
        max_train_shots=None, max_validation_shots=None,
        dry_train=False, dist_env=None, source_root=None,
        manifest_mode="generic", audit_identity=None, stats_root=None):
    """Hold one shared lifecycle lock for all publication artifact writes."""
    kwargs = {
        "split": split,
        "contexts": contexts,
        "seeds": seeds,
        "config_path": config_path,
        "target_dir": target_dir,
        "sidecar_dir": sidecar_dir,
        "out_root": out_root,
        "prefix": prefix,
        "epochs": epochs,
        "max_train_shots": max_train_shots,
        "max_validation_shots": max_validation_shots,
        "dry_train": dry_train,
        "dist_env": dist_env,
        "source_root": source_root,
        "manifest_mode": manifest_mode,
        "audit_identity": audit_identity,
        "stats_root": stats_root,
    }
    if manifest_mode != "publication":
        return _run_validation_matrix_locked(**kwargs)

    canonical_contexts = (
        [level.label for level in CONTEXT_LEVELS]
        if contexts is None else contexts
    )
    canonical_seeds = PUBLICATION_SEEDS if seeds is None else seeds
    normalized = {
        **kwargs,
        "contexts": canonical_contexts,
        "seeds": canonical_seeds,
        "config_path": pathlib.Path(CONFIG if config_path is None else config_path),
        "target_dir": pathlib.Path(TARGET_DIR if target_dir is None else target_dir),
        "sidecar_dir": pathlib.Path(
            SIDECAR_DIR if sidecar_dir is None else sidecar_dir),
        "out_root": pathlib.Path(OUT_ROOT if out_root is None else out_root),
        "source_root": pathlib.Path(
            REPO_ROOT if source_root is None else source_root),
        "audit_identity": pathlib.Path(
            AUDIT_IDENTITY if audit_identity is None else audit_identity),
        "stats_root": pathlib.Path(
            STATS_ROOT if stats_root is None else stats_root),
    }
    _require_publication_training_invocation(
        split=split,
        contexts=normalized["contexts"],
        seeds=normalized["seeds"],
        config_path=normalized["config_path"],
        target_dir=normalized["target_dir"],
        sidecar_dir=normalized["sidecar_dir"],
        out_root=normalized["out_root"],
        prefix=prefix,
        source_root=normalized["source_root"],
        audit_identity=normalized["audit_identity"],
        stats_root=normalized["stats_root"],
        epochs=epochs,
        max_train_shots=max_train_shots,
        max_validation_shots=max_validation_shots,
        dry_train=dry_train,
    )
    with ValidationFreezeLock(normalized["stats_root"], shared=True) as lock:
        lock.assert_held()
        _reject_publication_training_state(normalized["stats_root"])
        return _run_validation_matrix_locked(**normalized)


# ── --verify-only: metadata and artifacts only ───────────────────────
def _check_config_tokens(config):
    """The trainer-free contract subset (the full validator lives in the
    trainer module, which this mode must not import)."""
    tokens = {"study": "pf_context", "arm": "B", "model": "ActSeqAttn",
              "pe": "rope_time", "time_axis": "native_gmag_bnd"}
    for key, expected in tokens.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must equal {expected!r}")


def verify_only(args, split):
    """Report stored cells from metadata and artifacts only.

    No trainer import, no target NPZ access, no writes. The fresh
    fingerprint's normalization hash is recomputed from the artifact's own
    stored arrays, so a tampered or foreign artifact cannot pass by file
    metadata alone. Dry-run stubs (claim_scope 'dry_run') are refused under
    every prefix: they are orchestration proofs, never trained runs.
    """
    _require_publication_cli(args, operation="verify")
    config_path = pathlib.Path(args.config)
    target_dir = pathlib.Path(args.target_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    out_root = pathlib.Path(args.out_root)
    _check_config_tokens(yaml.safe_load(config_path.read_text()))
    smoke_scoped = _validate_smoke_request(split, args.run_prefix,
                                           args.max_train_shots,
                                           args.max_validation_shots)
    effective_split = (limited_split(split, args.max_train_shots,
                                     args.max_validation_shots)
                       if smoke_scoped else split)
    audit_snapshot = _validated_availability_audit_snapshot(
        split, args.manifest_mode, args.audit_identity,
        target_dir, sidecar_dir, REPO_ROOT)
    if args.manifest_mode == "publication":
        levels = list(CONTEXT_LEVELS)
        seeds = [0, 1, 2, 3, 4]
    else:
        levels = _coerce_contexts(args.contexts)
        seeds = [int(seed) for seed in args.seeds]
    entries = matrix_entries(levels, seeds, prefix=args.run_prefix)
    common_fingerprint = _common_fingerprint(
        effective_split, config_path, target_dir, sidecar_dir, REPO_ROOT,
        None, audit_snapshot)
    print(f"config {config_path}: contract tokens ok")
    print(f"split {split.name} v{split.version}: {len(split.train)} train / "
          f"{len(split.validation)} validation shots, claim_scope "
          f"{getattr(effective_split, 'claim_scope', None)!r}")
    ready = outstanding = rejected = 0
    for entry in entries:
        run_dir = out_root / entry.run_name
        if not (run_dir / "m3.pt").is_file():
            outstanding += 1
            print(f"outstanding {entry.run_name}")
            continue
        art = torch.load(run_dir / "m3.pt", map_location="cpu",
                         weights_only=False)
        stats = (art["feature_mean"], art["feature_std"],
                 art["target_mean"], art["target_std"])
        payload = entry_payload_from_common(
            entry, common_fingerprint,
            normalization_hash=normalization_stats_sha256(*stats))
        try:
            resume_decision(run_dir, payload, args.run_prefix,
                            dry_run=False)
        except RuntimeError as exc:
            rejected += 1
            print(f"REJECTED {entry.run_name}: {exc}")
            continue
        ready += 1
        print(f"ready {entry.run_name}")
    print(f"{len(entries)} runs: {ready} ready, {outstanding} outstanding, "
          f"{rejected} rejected")
    if args.manifest_mode == "publication":
        return 1 if (outstanding or rejected) else 0
    return 1 if rejected else 0


# ── --audit-only: the per-shot/context availability table ────────────
def _require_single_rank_audit_environment():
    names = ("WORLD_SIZE", "RANK", "LOCAL_RANK")
    values = {name: os.environ.get(name) for name in names}
    present = {name for name, value in values.items() if value is not None}
    if not present:
        return
    if present != set(names):
        raise RuntimeError(
            "--audit-only requires a consistent single-rank environment "
            "(WORLD_SIZE=1, RANK=0, LOCAL_RANK=0)")
    try:
        world, rank, local_rank = (
            int(values[name]) for name in names)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "--audit-only requires a consistent single-rank environment") from exc
    if (world, rank, local_rank) != (1, 0, 0):
        raise RuntimeError(
            "--audit-only is single-rank only; WORLD_SIZE=1, RANK=0, "
            "LOCAL_RANK=0 are required")


def _run_availability_jobs(jobs, workers):
    jobs = list(jobs)
    workers = int(workers)
    if workers <= 1:
        return [availability_rows_for_shot(job) for job in jobs]
    spawn = multiprocessing.get_context("spawn")
    root = str(REPO_ROOT)
    original_path = list(sys.path)
    sys.path[:] = [root] + [entry for entry in sys.path if entry != root]
    try:
        with ProcessPoolExecutor(
                max_workers=workers, mp_context=spawn) as executor:
            return list(executor.map(availability_rows_for_shot, jobs))
    finally:
        sys.path[:] = original_path


def _availability_csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=AUDIT_COLUMNS, lineterminator="\n",
        extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _audit_context_availability_locked(
        split, contexts=None, target_dir=None, sidecar_dir=None, csv_path=None,
        audit_identity=None, audit_workers=1, manifest_mode="generic",
        split_path=None, source_root=None):
    """Audit train+validation only and publish the identity last.

    Publication mode requires the exact 674-shot, seven-context contract and
    writes ``context_availability.csv`` before
    ``context_availability.audit.json``. Generic pilot mode remains an explicit
    CSV-only diagnostic path.
    """
    _require_single_rank_audit_environment()
    target_dir = pathlib.Path(TARGET_DIR if target_dir is None
                              else target_dir)
    sidecar_dir = pathlib.Path(SIDECAR_DIR if sidecar_dir is None
                               else sidecar_dir)
    csv_path = pathlib.Path(STATS_CSV if csv_path is None else csv_path)
    identity_path = pathlib.Path(
        AUDIT_IDENTITY if audit_identity is None else audit_identity)
    source_root = pathlib.Path(REPO_ROOT if source_root is None else source_root)
    split_path = pathlib.Path(_split_path(split) if split_path is None
                              else split_path)
    levels = _coerce_contexts(
        [x.label for x in CONTEXT_LEVELS] if contexts is None else contexts)
    workers = int(audit_workers)
    if workers < 1:
        raise ValueError("audit_workers must be positive")
    if manifest_mode == "publication":
        if (identity_path.parent != csv_path.parent
                or identity_path.name != "context_availability.audit.json"):
            raise ValueError(
                "--audit-identity must be context_availability.audit.json "
                "beside context_availability.csv")
        expected_labels = [level.label for level in CONTEXT_LEVELS]
        if [level.label for level in levels] != expected_labels:
            raise ValueError(
                "publication availability audit requires the exact frozen "
                "seven-context grid")
        if len(split.train) != 598 or len(split.validation) != 76:
            raise ValueError(
                "publication availability audit requires exactly 598 train "
                f"and 76 validation shots, got {len(split.train)} and "
                f"{len(split.validation)}")
    jobs = []
    shot_order = {}
    for role, shots in (("train", split.train),
                        ("validation", split.validation)):
        for shot in sorted(int(value) for value in shots):
            if shot in shot_order:
                raise ValueError(
                    "availability audit train+validation shots must be unique")
            shot_order[shot] = len(shot_order)
            jobs.append({
                "shot": shot,
                "role": role,
                "target_dir": str(target_dir),
                "sidecar_dir": str(sidecar_dir),
                "contexts": [level.label for level in levels],
            })
    results = _run_availability_jobs(jobs, workers)
    rows = [row for shot_rows in results for row in shot_rows]
    context_order = {level.label: index for index, level in enumerate(levels)}
    rows.sort(key=lambda row: (
        shot_order[int(row["shot"])], context_order[str(row["context"])]))
    expected_rows = len(jobs) * len(levels)
    keys = [(int(row["shot"]), str(row["context"])) for row in rows]
    expected_keys = [
        (int(job["shot"]), level.label)
        for job in jobs for level in levels
    ]
    if len(rows) != expected_rows or keys != expected_keys:
        raise RuntimeError(
            "availability audit worker results do not cover the exact ordered "
            "shot/context matrix")

    csv_payload = _availability_csv_bytes(rows)
    identity_payload = None
    if manifest_mode == "publication":
        availability_snapshot = availability_csv_snapshot_from_bytes(
            csv_payload, split)
        identity = build_availability_audit_identity(
            split_path=split_path,
            split=split,
            target_dir=target_dir,
            sidecar_dir=sidecar_dir,
            availability_snapshot=availability_snapshot,
            project_root=source_root,
        )
        identity_payload = canonical_availability_audit_bytes(identity)

    durable_publish_bytes(
        csv_path, csv_payload,
        state_label="Work 3 availability audit CSV")
    if manifest_mode == "publication":
        durable_publish_bytes(
            identity_path, identity_payload,
            state_label="Work 3 availability audit identity")
    print(f"audit: {len(rows)} rows -> {csv_path} "
          f"({len(split.train)} train + {len(split.validation)} validation "
          f"shots, {len(levels)} contexts)"
          + (f"; identity written last -> {identity_path}"
             if manifest_mode == "publication" else ""))
    return rows


def audit_context_availability(
        split, contexts=None, target_dir=None, sidecar_dir=None, csv_path=None,
        audit_identity=None, audit_workers=1, manifest_mode="generic",
        split_path=None, source_root=None):
    """Serialize publication audit writers under the lifecycle exclusive lock."""
    target_dir = pathlib.Path(TARGET_DIR if target_dir is None else target_dir)
    sidecar_dir = pathlib.Path(SIDECAR_DIR if sidecar_dir is None else sidecar_dir)
    csv_path = pathlib.Path(STATS_CSV if csv_path is None else csv_path)
    identity_path = pathlib.Path(
        AUDIT_IDENTITY if audit_identity is None else audit_identity)
    source_root = pathlib.Path(REPO_ROOT if source_root is None else source_root)
    split_path = pathlib.Path(
        _split_path(split) if split_path is None else split_path)
    kwargs = {
        "split": split,
        "contexts": contexts,
        "target_dir": target_dir,
        "sidecar_dir": sidecar_dir,
        "csv_path": csv_path,
        "audit_identity": identity_path,
        "audit_workers": audit_workers,
        "manifest_mode": manifest_mode,
        "split_path": split_path,
        "source_root": source_root,
    }
    if manifest_mode != "publication":
        return _audit_context_availability_locked(**kwargs)
    if (identity_path.parent != csv_path.parent
            or identity_path.name != "context_availability.audit.json"):
        raise ValueError(
            "--audit-identity must be context_availability.audit.json beside "
            "context_availability.csv")
    for path, label in (
            (csv_path, "Work 3 availability audit CSV"),
            (identity_path, "Work 3 availability audit identity")):
        if os.path.lexists(path):
            try:
                read_regular_nofollow(path, label=label)
            except PublicationIOError as exc:
                raise RuntimeError(str(exc)) from exc
    marker = csv_path.parent / "PFCTX_FINAL_TEST_EVALUATED.json"
    if os.path.lexists(marker):
        raise RuntimeError(
            f"{marker} exists: canonical publication state is globally final")
    with ValidationFreezeLock(csv_path.parent) as audit_lock:
        audit_lock.assert_held()
        if os.path.lexists(marker):
            raise RuntimeError(
                f"{marker} exists: canonical publication state is globally final")
        for path, label in (
                (csv_path, "Work 3 availability audit CSV"),
                (identity_path, "Work 3 availability audit identity")):
            if os.path.lexists(path):
                try:
                    read_regular_nofollow(path, label=label)
                except PublicationIOError as exc:
                    raise RuntimeError(str(exc)) from exc
        return _audit_context_availability_locked(**kwargs)


# ── the CLI ──────────────────────────────────────────────────────────
def _context_arg(value):
    return context_level(str(value))


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _epochs_arg(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("epochs must be >= 1")
    return number


def build_parser():
    ap = argparse.ArgumentParser(
        description="Validation-only PF-context (context, seed) matrix "
                    "runner")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true",
                       help="write ProjDB/Stats/pf_context/"
                            "context_availability.csv for the train and "
                            "validation shots, then exit")
    modes.add_argument("--verify-only", action="store_true",
                       help="report ready/outstanding/rejected cells from "
                            "metadata and artifacts only; writes nothing "
                            "and imports no trainer")
    ap.add_argument(
        "--audit-workers", type=_positive_int, default=4,
        help="parallel per-shot availability workers (default: 4)")
    ap.add_argument(
        "--audit-identity", default=str(AUDIT_IDENTITY),
        help="publication availability identity (default: ProjDB/Stats/"
             "pf_context/context_availability.audit.json)")
    ap.add_argument("--split", required=True,
                    help=f"frozen split manifest (e.g. configs/splits/"
                         f"{PILOT_MANIFEST}); never generated or "
                         f"substituted by this runner")
    ap.add_argument("--manifest-mode", choices=("publication", "generic"),
                    default="publication",
                    help="strict publication bundle validation (default) or "
                         "explicit archived pilot/smoke loading")
    ap.add_argument("--config", default=str(CONFIG),
                    help="frozen sweep configuration "
                         "(default: configs/dcs_pf_context_sweep.yml)")
    ap.add_argument("--target-dir", default=str(TARGET_DIR),
                    help="target dataset root (default: NpzGeom)")
    ap.add_argument("--sidecar-dir", default=str(SIDECAR_DIR),
                    help="PF sidecar dataset root (default: NpzGeomPFObs)")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="validation artifact root (default: ProjDB/trains)")
    ap.add_argument("--stats-root", default=str(STATS_ROOT),
                    help="publication lifecycle state root")
    ap.add_argument("--contexts", nargs="+", type=_context_arg,
                    default=list(CONTEXT_LEVELS),
                    help="context levels to run (default: the frozen "
                         "seven-level grid)")
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(PUBLICATION_SEEDS),
                    help="seeds to run (default: the frozen 0 1 2 3 4)")
    ap.add_argument("--epochs", type=_epochs_arg, default=None,
                    help="spec-6.4 engineering-pilot override of the frozen "
                         "epoch count: accepted only with a pilot manifest "
                         "(claim_scope pilot_only)")
    ap.add_argument("--max-train-shots", type=_positive_int, default=None,
                    help="pilot-only smoke limit: train on the first sorted "
                         "train shots only")
    ap.add_argument("--max-validation-shots", type=_positive_int,
                    default=None,
                    help="pilot-only smoke limit: validate on the first "
                         "sorted validation shots only")
    ap.add_argument("--run-prefix", default=DEFAULT_PREFIX,
                    help="run_name prefix (default: pfctx -> pfctx_h0512_s0)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    operation = "audit" if args.audit_only else (
        "verify" if args.verify_only else "training")
    if args.audit_only:
        _require_single_rank_audit_environment()
    _require_publication_cli(args, operation=operation)
    split = load_context_split(
        args.split, manifest_mode=args.manifest_mode, available_shots=None)
    if args.audit_only:
        audit_context_availability(
            split, contexts=args.contexts,
            target_dir=args.target_dir,
            sidecar_dir=args.sidecar_dir,
            csv_path=pathlib.Path(args.stats_root) / "context_availability.csv",
            audit_identity=args.audit_identity,
            audit_workers=args.audit_workers,
            manifest_mode=args.manifest_mode,
            split_path=args.split,
            source_root=REPO_ROOT)
        return 0
    if args.verify_only:
        return verify_only(args, split)
    return run_validation_matrix(
        split=split, contexts=args.contexts, seeds=args.seeds,
        config_path=args.config, target_dir=args.target_dir,
        sidecar_dir=args.sidecar_dir, out_root=args.out_root,
        prefix=args.run_prefix,
        epochs=args.epochs, max_train_shots=args.max_train_shots,
        max_validation_shots=args.max_validation_shots,
        source_root=REPO_ROOT, manifest_mode=args.manifest_mode,
        audit_identity=args.audit_identity, stats_root=args.stats_root)


if __name__ == "__main__":
    sys.exit(main())
