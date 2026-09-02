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
    --manifest-mode generic \
    --arms A B C D --seeds 0 --run-prefix pfobs_pilot
  python scripts/run_pf_observability.py --verify-only
"""
import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.build_pf_observability import (  # noqa: E402
    COVERAGE_GATE, check_coverage_gates,
)
from src.ml.dcs_features import load_dcs_config  # noqa: E402
from src.ml.pf_observability import (  # noqa: E402
    ARMS, run_fingerprint, sha256_file,
)
from src.ml.pfobs_provenance import (  # noqa: E402
    ValidationFreezeLock, normalization_stats_sha256,
    validate_source_audit_identity,
)
from src.ml.publication_split import (  # noqa: E402
    PUBLICATION_MANIFEST_PATH,
    PUBLICATION_OUT_ROOT,
    PUBLICATION_SEEDS,
    PUBLICATION_SIDECAR_DIR,
    PUBLICATION_TARGET_DIR,
    PUBLICATION_WORK2_ARMS,
    PUBLICATION_WORK2_AUDIT_IDENTITY,
    PUBLICATION_WORK2_CONFIG,
    PUBLICATION_WORK2_MARKER,
    PUBLICATION_WORK2_PREFIX,
    PUBLICATION_WORK2_STATS_ROOT,
    load_split_for_mode,
    require_publication_paths,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NPZ_DIR = PUBLICATION_TARGET_DIR
SIDECAR_DIR = PUBLICATION_SIDECAR_DIR
CONFIG = PUBLICATION_WORK2_CONFIG
SPLIT = PUBLICATION_MANIFEST_PATH
OUT_ROOT = PUBLICATION_OUT_ROOT
STATS_ROOT = PUBLICATION_WORK2_STATS_ROOT
AUDIT_IDENTITY = PUBLICATION_WORK2_AUDIT_IDENTITY
DEFAULT_PREFIX = PUBLICATION_WORK2_PREFIX

VALIDATION_DECISION_NAME = "validation_decision.json"
FINAL_MARKER_NAME = "FINAL_TEST_EVALUATED.json"
FINAL_DIR_NAME = "final_test"
STAGING_NAME = FINAL_DIR_NAME + ".staging"

DECISION_RUN, DECISION_SKIP, DECISION_ABORT = 0, 1, 2


def pfobs_train_module():
    """Import the trainer only after the publication shared-lock state recheck."""
    from src.ml import pfobs_train
    return pfobs_train


# Lazy compatibility seams retained for existing generic callers/tests. The
# publication core reaches them only after the shared-lock state recheck.
def validate_base_contract(config):
    return pfobs_train_module().validate_base_contract(config)


def training_normalization(*args, **kwargs):
    return pfobs_train_module().training_normalization(*args, **kwargs)


def init_dist():
    return pfobs_train_module().init_dist()


def dist_barrier(dist_env):
    return pfobs_train_module().dist_barrier(dist_env)


def dist_broadcast(tensor, dist_env):
    return pfobs_train_module().dist_broadcast(tensor, dist_env)


def teardown_dist(dist_env):
    return pfobs_train_module().teardown_dist(dist_env)


def train_one(*args, **kwargs):
    return pfobs_train_module().train_one(*args, **kwargs)


def _available_shots(npz_dir, sidecar_dir):
    """Shots present in both datasets without importing the trainer module."""
    def stems(directory):
        return {
            int(path.stem) for path in pathlib.Path(directory).glob("*.npz")
            if path.stem.isdigit()
        }
    return stems(npz_dir) & stems(sidecar_dir)


def _require_publication_invocation(args, *, operation):
    """Make publication roots, prefix, and 4x5 matrix non-downgradable."""
    del operation
    mode = getattr(args, "manifest_mode", "publication")
    if mode != "publication":
        return
    require_publication_paths(
        mode,
        {
            "npz_dir": args.npz_dir,
            "sidecar_dir": args.sidecar_dir,
            "config": args.config,
            "split": args.split,
            "out_root": args.out_root,
            "stats_root": args.stats_root,
            "audit_identity": args.audit_identity,
            "coverage_csv": args.coverage_csv,
        },
        {
            "npz_dir": NPZ_DIR,
            "sidecar_dir": SIDECAR_DIR,
            "config": CONFIG,
            "split": SPLIT,
            "out_root": OUT_ROOT,
            "stats_root": STATS_ROOT,
            "audit_identity": AUDIT_IDENTITY,
            "coverage_csv": STATS_ROOT / "coverage.csv",
        },
    )
    if tuple(args.arms) != tuple(PUBLICATION_WORK2_ARMS):
        raise ValueError(
            "publication --arms must be the canonical A B C D matrix")
    if tuple(int(seed) for seed in args.seeds) != tuple(PUBLICATION_SEEDS):
        raise ValueError(
            "publication --seeds must be the canonical 0 1 2 3 4 matrix")
    if str(args.run_prefix) != PUBLICATION_WORK2_PREFIX:
        raise ValueError(
            "publication --run-prefix must be the frozen 'pfobs' prefix")
    if args.epochs is not None:
        raise ValueError(
            "publication training cannot use --epochs; overrides are generic-only")


def _load_manifest(path, manifest_mode, available_shots=None):
    """Load the requested split contract once at the top-level preflight."""
    return load_split_for_mode(
        path,
        manifest_mode=manifest_mode,
        available_shots=available_shots,
        project_root=REPO_ROOT,
    )


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
        fingerprint = art["run_fingerprint"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{run_dir}: m3.pt carries no run_fingerprint -- not a "
            "PF-observability artifact; never overwrite") from exc
    try:
        normalization = normalization_stats_sha256(
            art["mean"], art["std"], art["tgt_mean"], art["tgt_std"])
        recorded = art["normalization_sha256"]
        fingerprinted = fingerprint["normalization_sha256"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{run_dir}: m3.pt carries no complete normalization_sha256 "
            "contract -- never overwrite") from exc
    if normalization != recorded or normalization != fingerprinted:
        raise RuntimeError(
            f"{run_dir}: normalization_sha256 does not match the stored "
            "normalization arrays -- artifact tampering or corruption")
    return fingerprint


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


def _lexical_absolute(path):
    """Absolute path without resolving repository symlinks such as ProjDB."""
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _publication_stats_paths(args):
    stats_root = _lexical_absolute(getattr(args, "stats_root", STATS_ROOT))
    identity_path = _lexical_absolute(
        getattr(args, "audit_identity", AUDIT_IDENTITY))
    expected = stats_root / "source_audit_identity.json"
    if identity_path != expected:
        raise ValueError(
            "--audit-identity must be source_audit_identity.json directly "
            "under --stats-root; copied or redirected audit bundles are "
            "not publication inputs")
    return stats_root, identity_path


def _validated_source_audit_sha256(args, split, npz_dir, sidecar_dir):
    """Validate the publication audit once; generic pilots use a sentinel."""
    if getattr(args, "manifest_mode", "publication") != "publication":
        return "absent"
    _stats_root, identity_path = _publication_stats_paths(args)
    validate_source_audit_identity(
        identity_path,
        split_path=args.split,
        split=split,
        npz_dir=npz_dir,
        sidecar_dir=sidecar_dir,
        project_root=REPO_ROOT,
    )
    return sha256_file(identity_path)


def _reject_publication_training_state(args):
    """Training locks only; verification intentionally does not call this."""
    if getattr(args, "manifest_mode", "publication") != "publication":
        return
    stats_root, _identity_path = _publication_stats_paths(args)
    for name in (
        VALIDATION_DECISION_NAME,
        FINAL_MARKER_NAME,
        FINAL_DIR_NAME,
        STAGING_NAME,
    ):
        path = stats_root / name
        if os.path.lexists(path):
            raise RuntimeError(
                f"{path} exists: publication training is locked after "
                "validation freeze or final-state creation")


def _run_matrix_locked(args):
    """Train every outstanding cell after the lifecycle lock/state recheck."""
    _reject_publication_training_state(args)
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
    split = _load_manifest(
        split_path, getattr(args, "manifest_mode", "publication"),
        available_shots=_available_shots(npz_dir, sidecar_dir),
    )
    audit_sha256 = _validated_source_audit_sha256(
        args, split, npz_dir, sidecar_dir)
    _reject_publication_training_state(args)

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
        normalization_by_arm = {}
        for i, run in enumerate(runs):
            tag = f"[{i + 1}/{len(runs)}]"
            out_dir = out_root / run["run_name"]
            decision, error = DECISION_RUN, None
            fingerprint = normalization = None
            if dist_env.is_main:
                try:
                    if run["arm"] not in normalization_by_arm:
                        normalization_by_arm[run["arm"]] = \
                            training_normalization(
                                npz_dir, sidecar_dir, split.train, run["arm"])
                    normalization = normalization_by_arm[run["arm"]]
                    normalization_sha256 = normalization_stats_sha256(
                        *normalization)
                    fingerprint = run_fingerprint(
                        run["arm"], run["seed"], config_path, split_path,
                        sidecar_dir, npz_dir,
                        normalization_sha256=normalization_sha256,
                        source_audit_identity_sha256=audit_sha256)
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
                epochs_override=args.epochs,
                normalization=normalization,
                run_fingerprint_payload=fingerprint,
                source_audit_identity_sha256=audit_sha256)
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


def run_matrix(args):
    """Hold the shared lifecycle lock for the complete publication training."""
    _require_publication_invocation(args, operation="training")
    if getattr(args, "manifest_mode", "publication") != "publication":
        return _run_matrix_locked(args)
    with ValidationFreezeLock(args.stats_root, shared=True) as training_lock:
        training_lock.assert_held()
        _reject_publication_training_state(args)
        return _run_matrix_locked(args)


def verify_only(args):
    """Check config/split/sidecar and report stored publication cells."""
    _require_publication_invocation(args, operation="verification")
    npz_dir = pathlib.Path(args.npz_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    config_path = pathlib.Path(args.config)
    split_path = pathlib.Path(args.split)
    out_root = pathlib.Path(args.out_root)
    both = _available_shots(npz_dir, sidecar_dir)
    validate_base_contract(load_dcs_config(config_path))
    split = _load_manifest(
        split_path, getattr(args, "manifest_mode", "publication"),
        available_shots=both,
    )
    audit_sha256 = _validated_source_audit_sha256(
        args, split, npz_dir, sidecar_dir)
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
        try:
            stored = _stored_fingerprint(out_dir)
            fingerprint = run_fingerprint(
                run["arm"], run["seed"], config_path, split_path,
                sidecar_dir, npz_dir,
                normalization_sha256=stored["normalization_sha256"],
                source_audit_identity_sha256=audit_sha256)
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
    ap.add_argument("--manifest-mode", choices=("publication", "generic"),
                    default="publication",
                    help="strict publication bundle validation (default) or "
                         "explicit archived generic-manifest loading")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="run directory root (default: ProjDB/trains)")
    ap.add_argument("--stats-root", default=str(STATS_ROOT),
                    help="canonical publication state root (default: "
                         "ProjDB/Stats/pf_observability)")
    ap.add_argument("--arms", nargs="+", type=_arm_arg, default=list(ARMS),
                    help="arms to run (default: A B C D)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="seeds to run (default: 0 1 2 3 4)")
    ap.add_argument("--epochs", type=_epochs_arg, default=None,
                    help="override hp.m3.epochs for smoke runs "
                         "(recorded in the artifact, not fingerprinted)")
    ap.add_argument("--run-prefix", default=DEFAULT_PREFIX,
                    help="run_name prefix (default: pfobs -> pfobs_a_s0)")
    ap.add_argument("--coverage-csv", default=str(STATS_ROOT / "coverage.csv"),
        help="split-free per-shot coverage audit from the sidecar build, "
                         "joined with the split for the spec-5.3 preflight "
                         "gates (default: ProjDB/Stats/pf_observability/"
                         "coverage.csv)")
    ap.add_argument("--audit-identity", default=str(AUDIT_IDENTITY),
                    help="split-bound source audit identity required in "
                         "publication mode (default: ProjDB/Stats/"
                         "pf_observability/source_audit_identity.json)")
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
