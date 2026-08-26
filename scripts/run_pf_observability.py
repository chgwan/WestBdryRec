# -*- coding: utf-8 -*-
"""Run the four-arm PF-observability validation matrix inside one torchrun job.

Every (arm, seed) cell trains the fixed base-model contract -- model
ActSeqAttn, ``pe=rope_time``, ``time_axis: native_gmag_bnd`` (asserted
against the shared config via ``validate_base_contract`` before any group
exists) -- through ``src.ml.pfobs_train.train_one`` and writes ONLY
validation artifacts: ``<out-root>/<run_name>/m3.pt`` plus
``validation.json``. Nothing here loads held-back shots, produces predictions
or computes final metrics -- those are later, explicitly gated stages.

The runner owns the process group's lifetime: ONE DistEnv is initialized
before the matrix loop and passed into every ``train_one`` call, a barrier
separates consecutive run decisions, and the group is destroyed once in a
top-level finally after the whole requested matrix. Repeated NCCL
initialization inside one torchrun process is unsupported, so ``train_one``
never initializes or destroys the group itself.

Resume: a cell is skipped when its artifact AND its validation JSON exist AND
the stored ``run_fingerprint`` matches the freshly computed one. A hash
mismatch is an ERROR -- a stored run is never overwritten; move the directory
aside or pass a different ``--run-prefix``. Rank 0 makes every skip/abort
decision and broadcasts it so all ranks stay in lockstep. A rank-local
failure inside ``train_one`` can still hang peers until the launcher kills
the job; the finally teardown at least keeps the failing rank's exit clean.

``--epochs`` replaces ``hp.m3.epochs`` for smoke-scale runs but is NOT part
of the fingerprint (which pins the config file's bytes); pair it with a
distinct ``--run-prefix``, as the NSCC smoke/pilot jobs do.

``--verify-only`` additionally enforces the spec-5.3 preflight coverage
gates: it joins ``--coverage-csv`` (the sidecar build's split-free audit,
carrying per-shot n_rows/n_valid/n_common) with the split manifest and
refuses an unscoreable held-out shot or a split under 95% target-valid
coverage -- reading no target npz at all.

Usage (inside a torchrun / qsub-trun job, from the repo root):
  torchrun --nproc_per_node 4 scripts/run_pf_observability.py \
    --split configs/splits/pfobs_random_pilot.json \
    --arms A B C D --seeds 0 --run-prefix pfobs_pilot
  python scripts/run_pf_observability.py --verify-only
"""
import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.build_pf_observability import (  # noqa: E402
    COVERAGE_GATE, check_coverage_gates,
)
from src.ml.dcs_features import load_dcs_config  # noqa: E402
from src.ml.pf_observability import (  # noqa: E402
    ARMS, load_split, run_fingerprint,
)
from src.ml.pfobs_train import (  # noqa: E402
    _available_shots, dist_barrier, dist_broadcast, init_dist, teardown_dist,
    train_one, validate_base_contract,
)
from src.proj_config import get_proj_config  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NPZ_DIR = REPO_ROOT / "ProjDB/datasets/NpzGeom"
SIDECAR_DIR = REPO_ROOT / "ProjDB/datasets/NpzGeomPFObs"
CONFIG = REPO_ROOT / "configs/dcs_pf_observability.yml"
# The FINAL campaign manifest (an Essential Work 1 output that does not exist
# yet); smoke and pilot runs pass --split explicitly, e.g.
# configs/splits/pfobs_random_pilot.json.
SPLIT = REPO_ROOT / "configs/splits/communications_physics_campaign_v1.json"
OUT_ROOT = REPO_ROOT / "ProjDB/trains"
DEFAULT_PREFIX = "pfobs"

DECISION_RUN, DECISION_SKIP, DECISION_ABORT = 0, 1, 2


def matrix_runs(arms=ARMS, seeds=(0, 1, 2, 3, 4), prefix=DEFAULT_PREFIX):
    """The requested matrix, seeds outer and arms inner.

    ``run_name = f"{prefix}_{arm.lower()}_s{seed}"``: the default prefix
    yields pfobs_a_s0 .. pfobs_d_s4, and the NSCC smoke's
    ``--run-prefix pfobs_smoke`` yields ProjDB/trains/pfobs_smoke_a_s0/.
    """
    return [
        {"arm": arm, "seed": int(seed),
         "run_name": f"{prefix}_{arm.lower()}_s{seed}"}
        for seed in seeds for arm in arms
    ]


def is_validation_done(run_dir):
    """A cell counts as stored only when BOTH files exist; a partial reruns."""
    run_dir = pathlib.Path(run_dir)
    return ((run_dir / "m3.pt").exists()
            and (run_dir / "validation.json").exists())


def _stored_fingerprint(run_dir):
    art = torch.load(pathlib.Path(run_dir) / "m3.pt", map_location="cpu",
                     weights_only=False)
    try:
        return art["run_fingerprint"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{run_dir}: m3.pt carries no run_fingerprint -- not a "
            "PF-observability artifact; never overwrite") from exc


def resume_decision(run_dir, fingerprint):
    """RUN when nothing (or only part) is stored, SKIP on a full match.

    Both files present but a differing fingerprint raises RuntimeError: the
    stored run is pinned provenance and is never silently overwritten.
    """
    run_dir = pathlib.Path(run_dir)
    if not is_validation_done(run_dir):
        return DECISION_RUN
    stored = _stored_fingerprint(run_dir)
    differing = [k for k in sorted(set(stored) | set(fingerprint))
                 if stored.get(k) != fingerprint.get(k)]
    if differing:
        raise RuntimeError(
            f"{run_dir}: stored run_fingerprint disagrees with the requested "
            f"run ({', '.join(differing)}) -- never overwrite; move the "
            "directory aside or pass a different --run-prefix")
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


def _log(dist_env, msg):
    if dist_env.is_main:
        print(msg, flush=True)


def run_matrix(args):
    """Train every outstanding cell of the requested matrix; resumable."""
    npz_dir = pathlib.Path(args.npz_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    config_path = pathlib.Path(args.config)
    split_path = pathlib.Path(args.split)
    out_root = pathlib.Path(args.out_root)
    runs = matrix_runs(arms=tuple(args.arms), seeds=tuple(args.seeds),
                       prefix=args.run_prefix)
    # symmetric preflight on every rank: a broken config, split or dataset
    # fails the job before any process group exists
    validate_base_contract(load_dcs_config(config_path))
    load_split(split_path, available_shots=_available_shots(
        npz_dir, sidecar_dir))

    dist_env = init_dist()
    try:
        if dist_env.is_main:
            epochs = ("config hp.m3.epochs"
                      if args.epochs is None else str(args.epochs))
            print(f"matrix: {len(runs)} runs, arms {' '.join(args.arms)}, "
                  f"seeds {' '.join(str(s) for s in args.seeds)}, "
                  f"epochs {epochs}, prefix {args.run_prefix!r}, "
                  f"world {dist_env.world_size}")
            print(f"target {npz_dir}\nsidecar {sidecar_dir}"
                  f"\nconfig {config_path}\nsplit {split_path}"
                  f"\nout-root {out_root}", flush=True)
        trained = skipped = 0
        for i, run in enumerate(runs):
            tag = f"[{i + 1}/{len(runs)}]"
            out_dir = out_root / run["run_name"]
            decision, error = DECISION_RUN, None
            if dist_env.is_main:
                try:
                    fingerprint = run_fingerprint(
                        run["arm"], run["seed"], config_path, split_path,
                        sidecar_dir, npz_dir)
                    decision = resume_decision(out_dir, fingerprint)
                except Exception as exc:
                    decision, error = DECISION_ABORT, exc
            decision, error = _broadcast_decision(dist_env, decision, error)
            if decision == DECISION_ABORT:
                raise error
            if decision == DECISION_SKIP:
                skipped += 1
                _log(dist_env, f"{tag} skip {run['run_name']} (artifact + "
                      "validation JSON + matching fingerprint)")
                dist_barrier(dist_env)
                continue
            _log(dist_env, f"{tag} run {run['run_name']}")
            meta = train_one(
                npz_dir, sidecar_dir, config_path, split_path, run["arm"],
                run["seed"], out_dir, dist_env,
                epochs_override=args.epochs)
            trained += 1
            _log(dist_env, f"{tag} done {run['run_name']} best_val_mse="
                  f"{meta['best_val_mse']:.6f} "
                  f"stop_epoch={meta['stop_epoch']}")
            dist_barrier(dist_env)
        _log(dist_env, f"matrix complete: {trained} trained, {skipped} "
              f"skipped, {trained + skipped} cells")
        return 0
    finally:
        teardown_dist(dist_env)


def verify_only(args):
    """Check config/split/sidecar, enforce the spec-5.3 coverage gates
    against the split-free audit, and report stored cells; write nothing."""
    npz_dir = pathlib.Path(args.npz_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    config_path = pathlib.Path(args.config)
    split_path = pathlib.Path(args.split)
    out_root = pathlib.Path(args.out_root)
    both = _available_shots(npz_dir, sidecar_dir)
    validate_base_contract(load_dcs_config(config_path))
    split = load_split(split_path, available_shots=both)
    print(f"config {config_path}: base contract ok")
    print(f"target {npz_dir} + sidecar {sidecar_dir}: "
          f"{len(both)} shots present in both")
    print(f"split {split.name} v{split.version}: {len(split.train)} train / "
          f"{len(split.validation)} validation shots, all present in both")
    gates = check_coverage_gates(args.coverage_csv, split)
    for part in ("train", "validation", "test", "overall"):
        g = gates[part]
        print(f"coverage[{part}]: {g['n_common']}/{g['n_valid']} "
              f"target-valid = {g['coverage_valid']:.4f} (gate "
              f"{COVERAGE_GATE:.0%}); {g['n_common']}/{g['n_rows']} "
              f"native rows = {g['coverage_rows']:.4f}")
    runs = matrix_runs(arms=tuple(args.arms), seeds=tuple(args.seeds),
                       prefix=args.run_prefix)
    ready = outstanding = mismatched = 0
    for run in runs:
        out_dir = out_root / run["run_name"]
        if not is_validation_done(out_dir):
            outstanding += 1
            print(f"outstanding {run['run_name']}")
            continue
        fingerprint = run_fingerprint(
            run["arm"], run["seed"], config_path, split_path, sidecar_dir,
            npz_dir)
        try:
            resume_decision(out_dir, fingerprint)
        except RuntimeError as exc:
            mismatched += 1
            print(f"HASH MISMATCH {run['run_name']}: {exc}")
            continue
        ready += 1
        print(f"ready {run['run_name']}")
    print(f"{len(runs)} runs: {ready} ready, {outstanding} outstanding, "
          f"{mismatched} hash-mismatched")
    return 1 if mismatched else 0


def _arm_arg(value):
    arm = value.upper()
    if arm not in ARMS:
        raise argparse.ArgumentTypeError(
            f"arm must be one of {ARMS}, got {value!r}")
    return arm


def _epochs_arg(value):
    epochs = int(value)
    if epochs < 1:
        raise argparse.ArgumentTypeError("epochs must be >= 1")
    return epochs


def build_parser():
    ap = argparse.ArgumentParser(
        description="Validation-only PF-observability (arm, seed) "
                    "matrix runner")
    ap.add_argument("--npz-dir", default=str(NPZ_DIR),
                    help="target dataset (default: ProjDB/datasets/NpzGeom)")
    ap.add_argument("--sidecar-dir", default=str(SIDECAR_DIR),
                    help="PF ref/actual sidecar "
                         "(default: ProjDB/datasets/NpzGeomPFObs)")
    ap.add_argument("--config", default=str(CONFIG),
                    help="shared arm config "
                         "(default: configs/dcs_pf_observability.yml)")
    ap.add_argument("--split", default=str(SPLIT),
                    help="frozen split manifest (default: the final "
                         "campaign manifest)")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="run directory root (default: ProjDB/trains)")
    ap.add_argument("--arms", nargs="+", type=_arm_arg, default=list(ARMS),
                    help="arms to run (default: A B C D)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="seeds to run (default: 0 1 2 3 4)")
    ap.add_argument("--epochs", type=_epochs_arg, default=None,
                    help="override hp.m3.epochs for smoke runs "
                         "(recorded in the artifact, not fingerprinted)")
    ap.add_argument("--run-prefix", default=DEFAULT_PREFIX,
                    help="run_name prefix (default: pfobs -> pfobs_a_s0)")
    ap.add_argument("--coverage-csv", default=str(
        get_proj_config().pfobs_stats_dir / "coverage.csv"),
        help="split-free per-shot coverage audit from the sidecar build, "
                         "joined with the split for the spec-5.3 preflight "
                         "gates (default: ProjDB/Stats/pf_observability/"
                         "coverage.csv)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check config/split/sidecar, enforce the spec-5.3 "
                         "coverage gates, and report stored cells; writes "
                         "nothing")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verify_only:
        return verify_only(args)
    return run_matrix(args)


if __name__ == "__main__":
    sys.exit(main())
