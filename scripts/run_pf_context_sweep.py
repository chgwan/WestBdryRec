# -*- coding: utf-8 -*-
"""Run the PF-context (context, seed) validation matrix inside one torchrun job.

Every cell trains the frozen Arm B contract through
``src.ml.pfctx_train.train_one`` and writes ONLY a validation artifact:
``<out-root>/<run_name>/m3.pt`` plus this runner's ``fingerprint.json`` (the
resume provenance; the .pt schema itself is Task 4's and stays untouched).
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
history-valid coverage.

Usage (inside a torchrun / qsub-trun job, from the repo root):
  torchrun --nproc_per_node 4 scripts/run_pf_context_sweep.py \
    --split configs/splits/pfobs_random_pilot.json
  python scripts/run_pf_context_sweep.py --audit-only --split configs/\
splits/pfobs_random_pilot.json
  python scripts/run_pf_context_sweep.py --verify-only --split configs/\
splits/pfobs_random_pilot.json
"""
import argparse
import csv
import dataclasses
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
    CONTEXT_LEVELS, ContextLevel, context_level, context_run_fingerprint,
    matrix_entries, normalization_stats_sha256, scored_windows,
)
from src.ml.pf_observability import (  # noqa: E402
    FrozenSplit, load_split, sha256_file,
)
from src.ml.pfctx_data import load_context_series  # noqa: E402
from src.ml.pfobs_train import (  # noqa: E402
    dist_barrier, dist_broadcast, init_dist, teardown_dist,
)
from src.ml.pos_encoding import modal_cadence  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "ProjDB/datasets/NpzGeom"
SIDECAR_DIR = REPO_ROOT / "ProjDB/datasets/NpzGeomPFObs"
CONFIG = REPO_ROOT / "configs/dcs_pf_context_sweep.yml"
OUT_ROOT = REPO_ROOT / "ProjDB/trains"
STATS_CSV = (REPO_ROOT / "ProjDB/Stats/pf_context"
             / "context_availability.csv")
DEFAULT_PREFIX = "pfctx"
PILOT_MANIFEST = "pfobs_random_pilot.json"
CAMPAIGN_MANIFEST = "communications_physics_campaign_v1.json"

DECISION_RUN, DECISION_SKIP, DECISION_ABORT = 0, 1, 2

# One row per (shot, context): the memory-timescale feasibility readout.
AUDIT_COLUMNS = (
    "shot", "role", "context", "n_rows", "cadence_seconds",
    "max_gap_seconds", "gap_rows_over_1p5x_cadence", "visible_tokens_mean",
    "visible_tokens_min", "visible_tokens_max", "full_horizon_fraction",
    "score_rows", "history_valid_coverage",
)


# ── the scoped split ─────────────────────────────────────────────────
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


def load_context_split(path):
    """Load a split manifest for the context sweep.

    A missing manifest is a hard error: this runner NEVER generates or
    substitutes the campaign manifest. The upstream pilot manifest is
    accepted only while its ``claim_scope`` is ``pilot_only``.
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
    split = load_split(path)
    # rebuild field-for-field (vars) rather than naming any single list:
    # this runner consumes train/validation only and stays agnostic to the
    # rest of the manifest's membership.
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
def entry_payload(entry, split, config_path, target_dir, sidecar_dir,
                  source_root, stats):
    """The run fingerprint of one cell: every field is compared on resume,
    so entry identity (context, seed) and every pinned input are inside it.
    The claim scope lives in the artifact itself and is checked by
    :func:`resume_decision`, not hashed here."""
    return context_run_fingerprint(
        config_path=config_path, split_path=split.split_path,
        target_meta=pathlib.Path(target_dir) / "meta.json",
        sidecar_meta=pathlib.Path(sidecar_dir) / "meta.json",
        source_root=source_root, context_seconds=entry.context.seconds,
        context_label=entry.context.label, seed=entry.seed,
        normalization_hash=normalization_stats_sha256(*stats))


def write_fingerprint(run_dir, run_name, fingerprint):
    """Atomically publish the cell's resume provenance next to the artifact;
    ``run_name`` is stored for diagnosis only, never compared."""
    path = pathlib.Path(run_dir) / "fingerprint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"run_name": run_name, **fingerprint},
                              indent=2, sort_keys=True))
    os.replace(tmp, path)


def _stored_payload(run_dir):
    path = pathlib.Path(run_dir) / "fingerprint.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _artifact_scope(run_dir):
    art = torch.load(pathlib.Path(run_dir) / "m3.pt", map_location="cpu",
                     weights_only=False)
    return art.get("claim_scope")


def resume_decision(run_dir, fingerprint, prefix, dry_run=False):
    """RUN when no artifact is stored, SKIP on a full-fingerprint match.

    A stored artifact whose fingerprint.json disagrees -- or is missing, or
    whose recorded scope is smoke under a production prefix, or is a
    dry-run stub while the current run is real -- raises: the stored run is
    pinned provenance and is never silently overwritten. Every fingerprint
    field is compared; ``run_name`` is diagnosis-only.
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
    compared = (set(stored) | set(fingerprint)) - {"run_name"}
    differing = [key for key in sorted(compared)
                 if stored.get(key) != fingerprint.get(key)]
    if differing:
        raise RuntimeError(
            f"{run_dir.name}: stored run disagrees with the requested run "
            f"({', '.join(differing)}) -- never overwrite; move the "
            "directory aside or pass a different --run-prefix")
    scope = _artifact_scope(run_dir)
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
                     sidecar_dir, run_dir, dist_env, stats, epochs):
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
            dist_env, *_stats_pairs(stats))
    finally:
        pfctx_train.run_epochs = real_run_epochs


def _dry_train_one(context, seed, split, config, target_dir, sidecar_dir,
                   run_dir, dist_env, stats):
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
            accumulation)
    return result


def pfctx_train_module():
    """The ONE lazy import of the trainer module: only the training path
    (production, smoke wrapper, dry stub) ever calls this, so --verify-only
    and --audit-only keep it out of the process."""
    from src.ml import pfctx_train
    return pfctx_train


def _train_entry(entry, split, config, target_dir, sidecar_dir, run_dir,
                 dist_env, stats, epochs, dry_train):
    if dry_train:
        return _dry_train_one(entry.context, entry.seed, split, config,
                              target_dir, sidecar_dir, run_dir, dist_env,
                              stats)
    trainer = pfctx_train_module()
    if epochs is None:
        return trainer.train_one(entry.context, entry.seed, split, config,
                                 target_dir, sidecar_dir, run_dir, dist_env,
                                 *_stats_pairs(stats))
    return _smoke_train_one(trainer, entry.context, entry.seed, split, config,
                            target_dir, sidecar_dir, run_dir, dist_env, stats,
                            epochs)


# ── the matrix loop ──────────────────────────────────────────────────
def run_validation_matrix(split, contexts=None, seeds=None, config_path=None,
                          target_dir=None, sidecar_dir=None, out_root=None,
                          prefix=DEFAULT_PREFIX, epochs=None,
                          max_train_shots=None, max_validation_shots=None,
                          dry_train=False, dist_env=None, source_root=None):
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
    source_root = REPO_ROOT if source_root is None else source_root
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
              f"\nconfig {config_path}\nsplit "
              f"{getattr(effective_split, 'split_path', None)}\n"
              f"out-root {out_root}")
        stats = _broadcast_normalization(effective_split.train, target_dir,
                                         sidecar_dir, dist_env)
        trained = skipped = 0
        for index, entry in enumerate(entries):
            tag = f"[{index + 1}/{len(entries)}]"
            run_dir = out_root / entry.run_name
            decision, error, payload = DECISION_RUN, None, None
            if dist_env.is_main:
                try:
                    payload = entry_payload(entry, effective_split,
                                            config_path, target_dir,
                                            sidecar_dir, source_root, stats)
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
            result = _train_entry(entry, effective_split, config, target_dir,
                                  sidecar_dir, run_dir, dist_env, stats,
                                  epochs, dry_train)
            if dist_env.is_main:
                write_fingerprint(run_dir, entry.run_name, payload)
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
    config_path = pathlib.Path(args.config)
    _check_config_tokens(yaml.safe_load(config_path.read_text()))
    smoke_scoped = _validate_smoke_request(split, args.run_prefix,
                                           args.max_train_shots,
                                           args.max_validation_shots)
    effective_split = (limited_split(split, args.max_train_shots,
                                     args.max_validation_shots)
                       if smoke_scoped else split)
    entries = matrix_entries(_coerce_contexts(args.contexts),
                             [int(s) for s in args.seeds],
                             prefix=args.run_prefix)
    print(f"config {config_path}: contract tokens ok")
    print(f"split {split.name} v{split.version}: {len(split.train)} train / "
          f"{len(split.validation)} validation shots, claim_scope "
          f"{getattr(effective_split, 'claim_scope', None)!r}")
    out_root = pathlib.Path(OUT_ROOT)
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
        payload = entry_payload(entry, effective_split, config_path,
                                TARGET_DIR, SIDECAR_DIR, REPO_ROOT, stats)
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
    return 1 if rejected else 0


# ── --audit-only: the per-shot/context availability table ────────────
def _availability_row(shot, role, level, series):
    native_time = series.time
    dt = np.diff(native_time)
    cadence = modal_cadence(native_time)
    windows = scored_windows(native_time, level)
    lengths = [w.window_end - w.window_start for w in windows]
    history = sum(int(series.history_valid[w.window_start:w.window_end]
                      .sum()) for w in windows)
    return {
        "shot": int(shot), "role": role, "context": level.label,
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
    }


def audit_context_availability(split, contexts=None, target_dir=None,
                               sidecar_dir=None, csv_path=None):
    """Write the availability CSV for the train and validation shots of
    every requested context; returns the rows written. Reads no other shot
    list and writes nothing else."""
    target_dir = pathlib.Path(TARGET_DIR if target_dir is None
                              else target_dir)
    sidecar_dir = pathlib.Path(SIDECAR_DIR if sidecar_dir is None
                               else sidecar_dir)
    csv_path = pathlib.Path(STATS_CSV if csv_path is None else csv_path)
    levels = _coerce_contexts(
        [x.label for x in CONTEXT_LEVELS] if contexts is None else contexts)
    rows = []
    for role, shots in (("train", split.train),
                        ("validation", split.validation)):
        for shot in sorted(int(s) for s in shots):
            series = load_context_series(
                target_dir / f"{shot}.npz", sidecar_dir)
            rows.extend(_availability_row(shot, role, level, series)
                        for level in levels)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)
    print(f"audit: {len(rows)} rows -> {csv_path} "
          f"({len(split.train)} train + {len(split.validation)} validation "
          f"shots, {len(levels)} contexts)")
    return rows


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
    with open(CONFIG) as fh:
        frozen = yaml.safe_load(fh)
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
    ap.add_argument("--split", required=True,
                    help=f"frozen split manifest (e.g. configs/splits/"
                         f"{PILOT_MANIFEST}); never generated or "
                         f"substituted by this runner")
    ap.add_argument("--config", default=str(CONFIG),
                    help="frozen sweep configuration "
                         "(default: configs/dcs_pf_context_sweep.yml)")
    ap.add_argument("--contexts", nargs="+", type=_context_arg,
                    default=[context_level(c["label"])
                             for c in frozen["contexts"]],
                    help="context levels to run (default: the frozen "
                         "seven-level grid)")
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[int(s) for s in frozen["seeds"]],
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
    split = load_context_split(args.split)
    if args.audit_only:
        audit_context_availability(split, contexts=args.contexts)
        return 0
    if args.verify_only:
        return verify_only(args, split)
    return run_validation_matrix(
        split=split, contexts=args.contexts, seeds=args.seeds,
        config_path=args.config, prefix=args.run_prefix,
        epochs=args.epochs, max_train_shots=args.max_train_shots,
        max_validation_shots=args.max_validation_shots)


if __name__ == "__main__":
    sys.exit(main())
