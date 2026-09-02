# -*- coding: utf-8 -*-
"""The one-shot final test transaction for the PF-context matrix.

Loads NO test target until every cell of the requested matrix (production:
all 35 = 7 contexts x 5 seeds) validates: artifact + fingerprint JSON present,
the stored ``run_fingerprint`` equal to the freshly computed one, the fixed
base contract intact (Arm B width 21, ``pe = rope_time``), and the artifact's
claim scope neither ``smoke_only`` nor ``dry_run`` (Task 5's channels). A
missing cell, any hash mismatch, or a smoke/dry artifact aborts before a
single test shot is opened.

The transaction runs under the stats root (default
``ProjDB/Stats/pf_context``): everything is written into the sibling staging
directory ``ProjDB/Stats/pf_context.final_building`` (a crashed run is
disposable and resumable -- cells whose three output files already exist are
not rescored), then verified complete, then the staging directory is
atomically renamed to ``ProjDB/Stats/pf_context/final_test`` and only THEN is
``PFCTX_FINAL_TEST_EVALUATED.json`` written, last. An existing marker (or an
existing ``final_test`` directory) refuses the run; no CLI flag removes or
bypasses either. The 32-angle representation floor is read from the upstream
final scorer's table when present and recorded as provenance -- it is never
recomputed and never subtracted from any metric.

Devices: ``--devices 0 1 2 3`` makes the non-DDP parent spawn one worker
process per device, the artifacts partitioned deterministically across them
(worker ``w`` of ``n`` scores matrix entry ``w::n``, a pure function of the
ordered matrix). The single ``--devices cpu`` case is the parent-process
convenience path used by tests and diagnostics. After the workers finish, the
parent validates every prediction file, every metric file, every frozen test
shot, and identical row indices across contexts per shot before publishing.

``--verify-only`` performs the readiness checks without importing
``src.ml.pfctx_infer``, creating workers, or opening any test target.

Usage (from the repo root, after the matrix runner completed):
  python scripts/score_pf_context_sweep.py --split <frozen-manifest>
  python scripts/score_pf_context_sweep.py --split <frozen-manifest> \
    --devices 0 1 2 3
  python scripts/score_pf_context_sweep.py --split <frozen-manifest> \
    --devices 0 1 2 3 --verify-only
"""
import argparse
import csv
import dataclasses
import datetime
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.pf_context import (  # noqa: E402
    CONTEXT_LEVELS, ContextLevel, context_level,
)
from src.ml.pf_observability import sha256_file  # noqa: E402
from src.ml.pfobs_provenance import ValidationFreezeLock  # noqa: E402
from src.ml.pfctx_provenance import (  # noqa: E402
    FinalReadiness,
    build_transaction_provenance,
    canonical_transaction_provenance_bytes,
    transaction_provenance_sha256,
    validate_final_reference_transition,
    validate_transaction_provenance_bytes,
    validate_work2_dependency,
)
from src.ml.pfctx_train import (  # noqa: E402
    INPUT_WIDTH, N_OUT, REQUIRED_ARTIFACT_KEYS,
)
from src.ml.publication_split import (  # noqa: E402
    PUBLICATION_MANIFEST_PATH,
    PUBLICATION_OUT_ROOT,
    PUBLICATION_SEEDS,
    PUBLICATION_SIDECAR_DIR,
    PUBLICATION_TARGET_DIR,
    PUBLICATION_WORK2_FLOOR,
    PUBLICATION_WORK2_MARKER,
    PUBLICATION_WORK3_AUDIT_IDENTITY,
    PUBLICATION_WORK3_CONFIG,
    PUBLICATION_WORK3_CONTEXTS,
    PUBLICATION_WORK3_MARKER,
    PUBLICATION_WORK3_PREFIX,
    PUBLICATION_WORK3_REFERENCES,
    PUBLICATION_WORK3_STATS_ROOT,
    require_publication_paths,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET_DIR = PUBLICATION_TARGET_DIR
SIDECAR_DIR = PUBLICATION_SIDECAR_DIR
CONFIG = PUBLICATION_WORK3_CONFIG
SPLIT = PUBLICATION_MANIFEST_PATH
OUT_ROOT = PUBLICATION_OUT_ROOT
STATS_ROOT = PUBLICATION_WORK3_STATS_ROOT

FINAL_DIR_NAME = "final_test"
BUILDING_SUFFIX = ".final_building"
MARKER_NAME = "PFCTX_FINAL_TEST_EVALUATED.json"
# The one-shot transaction scores exactly this matrix by default; anything a
# caller explicitly requests instead is scored as requested and recorded in
# the marker, but the production default is the full 35-cell matrix.
PRODUCTION_SEEDS = (0, 1, 2, 3, 4)
PRODUCTION_ARTIFACT_COUNT = len(CONTEXT_LEVELS) * len(PRODUCTION_SEEDS)
DEFAULT_DEVICES = ("0", "1", "2", "3")
# The upstream Essential-Work-2 final scorer's representation-floor table:
# read for provenance only, never recomputed or subtracted.
DEFAULT_WORK2_MARKER = PUBLICATION_WORK2_MARKER
DEFAULT_FLOOR_CSV = PUBLICATION_WORK2_FLOOR
DEFAULT_REFERENCES = PUBLICATION_WORK3_REFERENCES
DEFAULT_AUDIT_IDENTITY = PUBLICATION_WORK3_AUDIT_IDENTITY
TRANSACTION_NAME = "transaction_provenance.json"
WORK3_FINAL_MARKER_SCHEMA = "pf_context_final_test_v2"
WORK3_FINAL_MARKER_GENERATION_VERSION = 2
GENERIC_FINAL_MARKER_SCHEMA = "pf_context_generic_final_test_v1"
GENERIC_FINAL_MARKER_GENERATION_VERSION = 1
CELL_MEMBER_ORDER = (
    "per_shot_metrics.csv", "m3_pred.npz", "run_metadata.json")
CELL_MEMBER_NAMES = set(CELL_MEMBER_ORDER)

WORKER_FLAG = "--score-partition"


def _require_publication_invocation(args, *, operation):
    """Require the one canonical Work 3 state/dependency/artifact namespace."""
    if _manifest_mode(args) != "publication":
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
            "work2_marker": args.work2_marker,
            "floor_csv": args.floor_csv,
            "references": args.references,
        },
        {
            "split": SPLIT,
            "config": CONFIG,
            "target_dir": TARGET_DIR,
            "sidecar_dir": SIDECAR_DIR,
            "out_root": OUT_ROOT,
            "stats_root": STATS_ROOT,
            "audit_identity": DEFAULT_AUDIT_IDENTITY,
            "work2_marker": DEFAULT_WORK2_MARKER,
            "floor_csv": DEFAULT_FLOOR_CSV,
            "references": DEFAULT_REFERENCES,
        },
    )
    labels = tuple(
        value.label if isinstance(value, ContextLevel) else str(value)
        for value in args.contexts
    )
    if labels != tuple(PUBLICATION_WORK3_CONTEXTS):
        raise ValueError(
            "publication --contexts must be the canonical seven-context matrix")
    if tuple(int(seed) for seed in args.seeds) != tuple(PUBLICATION_SEEDS):
        raise ValueError(
            "publication --seeds must be the canonical 0 1 2 3 4 matrix")
    if str(args.run_prefix) != PUBLICATION_WORK3_PREFIX:
        raise ValueError(
            "publication --run-prefix must be the frozen 'pfctx' prefix")
    canonical_marker = pathlib.Path(STATS_ROOT) / MARKER_NAME
    if operation in {"score", "worker"} and os.path.lexists(canonical_marker):
        raise RuntimeError(
            f"{canonical_marker} exists: the canonical publication transaction "
            "is globally final")


def _load_runner():
    """scripts/ is not a package; load the validation runner by path so the
    matrix naming and fingerprint logic has exactly one implementation."""
    spec = importlib.util.spec_from_file_location(
        "run_pf_context_sweep",
        pathlib.Path(__file__).resolve().parent / "run_pf_context_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner()


def _load_analysis():
    spec = importlib.util.spec_from_file_location(
        "pf_context_analysis_for_final_scorer",
        REPO_ROOT / "exploration/pf_context_analysis.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Work 3 validation-selection analysis")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_analysis()


def _load_manifest(path, manifest_mode, available_shots=None):
    """Load the requested split contract once at a scorer preflight."""
    return RUNNER.load_context_split(
        path,
        manifest_mode=manifest_mode,
        available_shots=available_shots,
    )


def _manifest_mode(args):
    return getattr(args, "manifest_mode", "publication")


def pfctx_infer_module():
    """The ONE lazy import of the inference module: only the scoring path
    (workers and the single-CPU parent path) ever calls this, so
    ``--verify-only`` keeps it out of the process."""
    from src.ml import pfctx_infer
    return pfctx_infer


def building_root(args):
    """The staging directory: a SIBLING of the stats root, so the stats root
    itself only ever contains published content."""
    stats_root = pathlib.Path(args.stats_root)
    return stats_root.parent / (stats_root.name + BUILDING_SUFFIX)


def marker_path(args):
    return pathlib.Path(args.stats_root) / MARKER_NAME


def final_dir_path(args):
    return pathlib.Path(args.stats_root) / FINAL_DIR_NAME


def _atomic_json_dump(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(
        obj, ensure_ascii=False, allow_nan=False,
        indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _canonical_json_bytes(obj):
    return (json.dumps(
        obj, ensure_ascii=False, allow_nan=False,
        indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_regular_bytes(path, *, label):
    path = pathlib.Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"cannot read {label} at {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{label} at {path} must be a regular no-follow file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(
            f"{label} at {path} must be a regular no-follow file") from exc
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)):
            raise RuntimeError(f"{label} at {path} changed while opening")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_nofollow_directory(path, *, label):
    path = pathlib.Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"cannot open {label} at {path}: {exc}") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"{label} at {path} must be a real no-follow directory")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if (not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)):
        os.close(descriptor)
        raise RuntimeError(f"{label} at {path} changed while opening")
    return descriptor


def _open_child_directory(parent_fd, name, *, path, label):
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise RuntimeError(
            f"{label} at {path} must be a real no-follow directory") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError(
            f"{label} at {path} must be a real no-follow directory")
    return descriptor


def _read_regular_child_bytes(directory_fd, name, *, path, label):
    try:
        descriptor = os.open(
            name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeError(
            f"{label} at {path} must be a regular no-follow file") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(
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


def _leaf_stat_signature(info):
    return (
        int(info.st_dev), int(info.st_ino), stat.S_IFMT(info.st_mode),
        int(info.st_size), int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


@dataclasses.dataclass
class RetainedRegularFile:
    parent_fd: int
    name: str
    path: pathlib.Path
    label: str
    fd: int
    opened_signature: tuple

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _fstat_signature(self):
        if self.fd is None:
            raise RuntimeError(f"retained {self.label} descriptor is closed")
        try:
            info = os.fstat(self.fd)
        except OSError as exc:
            raise RuntimeError(
                f"retained {self.label} descriptor at {self.path} is invalid") \
                from exc
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                f"retained {self.label} at {self.path} is no longer regular")
        return info, _leaf_stat_signature(info)

    def read_snapshot(self):
        before, before_signature = self._fstat_signature()
        if before_signature != self.opened_signature:
            raise RuntimeError(
                f"retained {self.label} at {self.path} changed after secure open")
        chunks = []
        offset = 0
        while True:
            chunk = os.pread(self.fd, 1024 * 1024, offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        payload = b"".join(chunks)
        after, after_signature = self._fstat_signature()
        if (after_signature != before_signature
                or len(payload) != int(after.st_size)
                or int(before.st_size) != len(payload)):
            raise RuntimeError(
                f"retained {self.label} at {self.path} changed while reading")
        return payload, after_signature

    def assert_current(self, expected_signature):
        before, before_signature = self._fstat_signature()
        try:
            path_info = os.stat(
                self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                f"published {self.label} path at {self.path} is unavailable") \
                from exc
        after, after_signature = self._fstat_signature()
        path_signature = _leaf_stat_signature(path_info)
        if (not stat.S_ISREG(path_info.st_mode)
                or path_signature[:3] != before_signature[:3]):
            raise RuntimeError(
                f"published {self.label} path at {self.path} was replaced")
        if (before_signature != expected_signature
                or after_signature != expected_signature
                or path_signature != expected_signature
                or int(before.st_size) != int(after.st_size)):
            raise RuntimeError(
                f"published {self.label} at {self.path} changed after final read")


def _open_retained_regular_child(directory_fd, name, *, path, label):
    path = pathlib.Path(path)
    try:
        path_info = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(
            f"{label} at {path} must be a regular no-follow file") from exc
    if not stat.S_ISREG(path_info.st_mode):
        raise RuntimeError(
            f"{label} at {path} must be a regular no-follow file")
    try:
        descriptor = os.open(
            name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeError(
            f"{label} at {path} must be a regular no-follow file") from exc
    try:
        opened = os.fstat(descriptor)
        opened_signature = _leaf_stat_signature(opened)
        if (not stat.S_ISREG(opened.st_mode)
                or opened_signature[:3] != _leaf_stat_signature(path_info)[:3]):
            raise RuntimeError(f"{label} at {path} changed while opening")
        return RetainedRegularFile(
            directory_fd, str(name), path, label, descriptor,
            opened_signature)
    except BaseException:
        os.close(descriptor)
        raise


def _publish_prevalidated_bytes(
        path, payload, *, validator, state_label, directory_fd=None):
    """Publish and race-read only through one retained directory descriptor."""
    path = pathlib.Path(path)
    payload = bytes(payload)
    owns_fd = directory_fd is None
    if directory_fd is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        directory_fd = _open_nofollow_directory(
            path.parent, label=f"{state_label} parent")
    else:
        directory_fd = os.dup(directory_fd)
        owns_fd = True

    def existing_bytes():
        try:
            info = os.stat(
                path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                f"{state_label} at {path} must be a regular no-follow file")
        return _read_regular_child_bytes(
            directory_fd, path.name, path=path, label=state_label)

    temporary_name = f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary_fd = None
    try:
        existing = existing_bytes()
        if existing is not None:
            validator(existing)
            if existing != payload:
                raise RuntimeError(f"{path}: existing {state_label} differs")
            return path
        temporary_fd = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644, dir_fd=directory_fd)
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise OSError(f"short write while publishing {path}")
            view = view[written:]
        os.lseek(temporary_fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(temporary_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        complete = b"".join(chunks)
        if complete != payload:
            raise RuntimeError(f"{path}: temporary bytes are incomplete")
        validator(complete)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        try:
            os.link(
                temporary_name, path.name,
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                follow_symlinks=False)
        except FileExistsError:
            existing = existing_bytes()
            if existing is None:
                raise RuntimeError(f"{path}: destination race disappeared")
            validator(existing)
            if existing != payload:
                raise RuntimeError(f"{path}: concurrent {state_label} differs")
        os.fsync(directory_fd)
        return path
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        if owns_fd:
            os.close(directory_fd)


@dataclasses.dataclass
class FinalTreeDirectory:
    path: pathlib.Path
    root_fd: int

    def close(self):
        if self.root_fd is not None:
            os.close(self.root_fd)
            self.root_fd = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
        return False


@dataclasses.dataclass
class FinalCellDirectory:
    path: pathlib.Path
    root_fd: int
    context_fd: int
    cell_fd: int

    def close(self):
        for descriptor in (self.cell_fd, self.context_fd, self.root_fd):
            if descriptor is not None:
                os.close(descriptor)
        self.cell_fd = self.context_fd = self.root_fd = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
        return False


def open_final_tree_directory(root, *, create):
    root = pathlib.Path(root)
    if create:
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(root)
        except FileExistsError:
            pass
    return FinalTreeDirectory(
        root, _open_nofollow_directory(root, label="final tree root"))


@dataclasses.dataclass
class FinalMatrixDirectory:
    path: pathlib.Path
    tree: FinalTreeDirectory
    context_fds: dict[str, int]
    cell_fds: dict[str, int]

    @property
    def root_fd(self):
        return self.tree.root_fd

    def cell_fd(self, entry):
        return self.cell_fds[entry.run_name]

    def close(self):
        for descriptor in self.cell_fds.values():
            os.close(descriptor)
        for descriptor in self.context_fds.values():
            os.close(descriptor)
        self.cell_fds.clear()
        self.context_fds.clear()
        self.tree.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
        return False


@dataclasses.dataclass
class FinalLeafDescriptors:
    transaction: RetainedRegularFile
    cell_members: dict[str, dict[str, RetainedRegularFile]]

    def member(self, entry, name):
        return self.cell_members[entry.run_name][name]

    def all_files(self):
        if self.transaction is not None:
            yield self.transaction
        for members in self.cell_members.values():
            for name in CELL_MEMBER_ORDER:
                yield members[name]

    def close(self):
        for retained in reversed(list(self.all_files())):
            retained.close()
        self.cell_members.clear()
        self.transaction = None

    def assert_current(self, witnesses):
        retained_files = list(self.all_files())
        if set(witnesses) != {id(retained) for retained in retained_files}:
            raise RuntimeError("final retained-leaf witness set is incomplete")
        for retained in retained_files:
            retained.assert_current(witnesses[id(retained)])

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
        return False


def open_final_cell_directory(tree, entry, *, create):
    root_fd = os.dup(tree.root_fd)
    context_fd = cell_fd = None
    context_path = tree.path / entry.context.label
    seed_name = f"s{int(entry.seed)}"
    cell_path = context_path / seed_name
    try:
        if create:
            try:
                os.mkdir(entry.context.label, dir_fd=root_fd)
            except FileExistsError:
                pass
        context_fd = _open_child_directory(
            root_fd, entry.context.label, path=context_path,
            label="final context directory")
        if create:
            try:
                os.mkdir(seed_name, dir_fd=context_fd)
            except FileExistsError:
                pass
        cell_fd = _open_child_directory(
            context_fd, seed_name, path=cell_path,
            label="final cell directory")
        return FinalCellDirectory(
            cell_path, root_fd, context_fd, cell_fd)
    except BaseException:
        if cell_fd is not None:
            os.close(cell_fd)
        if context_fd is not None:
            os.close(context_fd)
        os.close(root_fd)
        raise


def open_final_matrix_directory(root, entries, *, require_complete):
    tree = open_final_tree_directory(root, create=False)
    context_fds = {}
    cell_fds = {}
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry.context.label, []).append(entry)
    try:
        root_names = set(os.listdir(tree.root_fd))
        allowed_root = set(grouped) | {TRANSACTION_NAME}
        if (not root_names <= allowed_root
                or TRANSACTION_NAME not in root_names
                or (require_complete and root_names != allowed_root)):
            raise RuntimeError(f"{root}: final root membership changed")
        _require_regular_child(
            tree.root_fd, TRANSACTION_NAME,
            path=pathlib.Path(root) / TRANSACTION_NAME)
        for context, context_entries in grouped.items():
            if context not in root_names:
                if require_complete:
                    raise RuntimeError(
                        f"{pathlib.Path(root) / context}: context is absent")
                continue
            context_path = pathlib.Path(root) / context
            context_fd = _open_child_directory(
                tree.root_fd, context, path=context_path,
                label="final context directory")
            context_fds[context] = context_fd
            context_info = os.stat(
                context, dir_fd=tree.root_fd, follow_symlinks=False)
            opened_context = os.fstat(context_fd)
            if (context_info.st_dev, context_info.st_ino) != (
                    opened_context.st_dev, opened_context.st_ino):
                raise RuntimeError(f"{context_path}: context changed after secure open")
            expected_seeds = {f"s{int(entry.seed)}" for entry in context_entries}
            seed_names = set(os.listdir(context_fd))
            if (not seed_names <= expected_seeds
                    or (require_complete and seed_names != expected_seeds)):
                raise RuntimeError(f"{context_path}: seed membership changed")
            for entry in context_entries:
                seed_name = f"s{int(entry.seed)}"
                if seed_name not in seed_names:
                    if require_complete:
                        raise RuntimeError(
                            f"{context_path / seed_name}: cell is absent")
                    continue
                cell_path = context_path / seed_name
                cell_fd = _open_child_directory(
                    context_fd, seed_name, path=cell_path,
                    label="final cell directory")
                cell_fds[entry.run_name] = cell_fd
                cell_info = os.stat(
                    seed_name, dir_fd=context_fd, follow_symlinks=False)
                opened_cell = os.fstat(cell_fd)
                if (cell_info.st_dev, cell_info.st_ino) != (
                        opened_cell.st_dev, opened_cell.st_ino):
                    raise RuntimeError(
                        f"{cell_path}: cell changed after secure open")
                members = set(os.listdir(cell_fd))
                if not members <= CELL_MEMBER_NAMES:
                    raise RuntimeError(
                        f"{cell_path}: unexpected members "
                        f"{sorted(members - CELL_MEMBER_NAMES)}")
                if ("run_metadata.json" in members
                        and members != CELL_MEMBER_NAMES):
                    raise RuntimeError(
                        f"{cell_path}: metadata commit exists without complete cell")
                if require_complete and members != CELL_MEMBER_NAMES:
                    raise RuntimeError(f"{cell_path}: cell membership is incomplete")
                for member in members:
                    _require_regular_child(
                        cell_fd, member, path=cell_path / member)
        return FinalMatrixDirectory(
            pathlib.Path(root), tree, context_fds, cell_fds)
    except BaseException:
        for descriptor in cell_fds.values():
            os.close(descriptor)
        for descriptor in context_fds.values():
            os.close(descriptor)
        tree.close()
        raise


def open_final_leaf_descriptors(matrix, entries):
    transaction = None
    cell_members = {}
    try:
        transaction = _open_retained_regular_child(
            matrix.root_fd, TRANSACTION_NAME,
            path=matrix.path / TRANSACTION_NAME,
            label="transaction provenance")
        for entry in entries:
            cell_path = _cell_dir(matrix.path, entry)
            members = {}
            cell_members[entry.run_name] = members
            for name in CELL_MEMBER_ORDER:
                label = {
                    "per_shot_metrics.csv": "cell metrics",
                    "m3_pred.npz": "cell predictions",
                    "run_metadata.json": "cell metadata",
                }[name]
                members[name] = _open_retained_regular_child(
                    matrix.cell_fd(entry), name, path=cell_path / name,
                    label=label)
        return FinalLeafDescriptors(transaction, cell_members)
    except BaseException:
        for members in reversed(list(cell_members.values())):
            for retained in reversed(list(members.values())):
                retained.close()
        if transaction is not None:
            transaction.close()
        raise


def transaction_path(args, *, root=None):
    return pathlib.Path(
        building_root(args) if root is None else root) / TRANSACTION_NAME


# ── readiness: every fingerprint validated before any test target ─────
def _expected_entries(args):
    contexts = [x if isinstance(x, ContextLevel) else context_level(str(x))
                for x in args.contexts]
    return RUNNER.matrix_entries(contexts, [int(s) for s in args.seeds],
                                 prefix=str(args.run_prefix))


def _check_artifact_contract(run_name, artifact):
    """The pfctx_infer-free contract subset (--verify-only must never import
    the inference module); the inference primitive re-checks the same tokens
    through ``validate_artifact_contract``."""
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ValueError(f"{run_name}: artifact is missing {missing}")
    tokens = {"study": "pf_context", "arm": "B", "model": "ActSeqAttn",
              "time_axis": "native_gmag_bnd"}
    for key, expected in tokens.items():
        if artifact[key] != expected:
            raise RuntimeError(
                f"{run_name}: artifact {key} must be {expected!r}, got "
                f"{artifact[key]!r}")
    if artifact["pe"] != "rope_time":
        raise RuntimeError(
            f"{run_name}: pe must be 'rope_time', got {artifact['pe']!r} -- "
            "the context sweep is only comparable under the fixed base model")
    if int(artifact["n_act"]) != INPUT_WIDTH:
        raise RuntimeError(
            f"{run_name}: the fixed Arm B input is {INPUT_WIDTH} columns, "
            f"got {int(artifact['n_act'])}")
    if int(artifact["n_out"]) != N_OUT:
        raise RuntimeError(
            f"{run_name}: n_out must be {N_OUT}, got {artifact['n_out']}")


def _check_entry_consistency(run_name, entry, artifact):
    if str(artifact["context_label"]) != entry.context.label:
        raise RuntimeError(
            f"{run_name}: artifact context_label "
            f"{artifact['context_label']!r} does not name its matrix cell "
            f"{entry.context.label!r}")
    if (float(artifact["context_seconds"]) != entry.context.seconds
            or int(artifact["nominal_samples"])
            != entry.context.nominal_samples):
        raise RuntimeError(
            f"{run_name}: artifact context geometry disagrees with the "
            "frozen grid")
    if int(artifact["seed"]) != int(entry.seed):
        raise RuntimeError(
            f"{run_name}: artifact seed {artifact['seed']} does not name its "
            f"matrix cell seed {entry.seed}")


def verify_final_readiness(args, split=None):
    """Every cell of the requested matrix stored, hash-valid, contract-clean,
    and scope-clean -- all before the caller touches any test target.

    Returns the validated cells (``entry`` / ``dir`` / ``artifact``). A
    missing cell is a ValueError naming the production count; a fingerprint
    mismatch is a RuntimeError (a stored run is never overwritten); a
    smoke/dry artifact is a RuntimeError extending the runner's channels.
    """
    if split is None:
        split = _load_manifest(
            args.split, _manifest_mode(args), available_shots=None)
    entries = _expected_entries(args)
    out_root = pathlib.Path(args.out_root)
    cells, missing = [], []
    for entry in entries:
        run_dir = out_root / entry.run_name
        if not ((run_dir / "m3.pt").is_file()
                and (run_dir / "fingerprint.json").is_file()):
            missing.append(entry.run_name)
            continue
        artifact = torch.load(run_dir / "m3.pt", map_location="cpu",
                              weights_only=False)
        _check_artifact_contract(entry.run_name, artifact)
        _check_entry_consistency(entry.run_name, entry, artifact)
        scope = artifact.get("claim_scope")
        if scope in ("smoke_only", "dry_run"):
            raise RuntimeError(
                f"{entry.run_name}: stored artifact carries claim_scope "
                f"{scope!r} and can never be part of the final test -- "
                "smoke and dry-run channels are refused by the final scorer")
        stats = (artifact["feature_mean"], artifact["feature_std"],
                 artifact["target_mean"], artifact["target_std"])
        fresh = RUNNER.entry_payload(
            entry, split, pathlib.Path(args.config),
            pathlib.Path(args.target_dir), pathlib.Path(args.sidecar_dir),
            REPO_ROOT, stats)
        stored = json.loads(
            (run_dir / "fingerprint.json").read_text())
        compared = (set(stored) | set(fresh)) - {"run_name"}
        differing = [key for key in sorted(compared)
                     if stored.get(key) != fresh.get(key)]
        if differing:
            raise RuntimeError(
                f"{entry.run_name}: stored run disagrees with the final "
                f"scorer's freshly computed fingerprint ({', '.join(differing)}) "
                "-- never score a drifted matrix; move the directory aside "
                "or retrain it under the current pinned inputs")
        cells.append({"entry": entry, "dir": run_dir, "artifact": artifact})
    if missing:
        raise ValueError(
            f"final scorer requires all {len(entries)} artifacts of the "
            f"requested matrix (production: {PRODUCTION_ARTIFACT_COUNT} "
            f"artifacts) -- missing: {', '.join(missing)}")
    return cells


def _stat_witness(path):
    path = pathlib.Path(path)
    link = path.lstat()
    target = path.stat()
    return {
        "path": str(path),
        "lstat": [link.st_dev, link.st_ino, link.st_mode, link.st_size,
                  link.st_mtime_ns, link.st_ctime_ns],
        "stat": [target.st_dev, target.st_ino, target.st_mode, target.st_size,
                 target.st_mtime_ns, target.st_ctime_ns],
    }


def _publication_readiness_witness(args, prepared):
    paths = {
        pathlib.Path(args.config), pathlib.Path(args.split),
        pathlib.Path(args.target_dir) / "meta.json",
        pathlib.Path(args.sidecar_dir) / "meta.json",
        pathlib.Path(args.references), pathlib.Path(args.audit_identity),
        pathlib.Path(args.audit_identity).parent / "context_availability.csv",
        pathlib.Path(args.work2_marker), pathlib.Path(args.floor_csv),
        pathlib.Path(args.stats_root) / ANALYSIS.VALIDATION_SELECTION_NAME,
        pathlib.Path(args.stats_root) / ANALYSIS.GENERATED_NAME
        / ANALYSIS.VALIDATION_TABLE_NAME,
    }
    split = prepared["split"]
    if split.shot_metadata is not None:
        paths.add(pathlib.Path(split.shot_metadata))
    if split.slice_strata_dir is not None:
        paths.update(
            path for path in pathlib.Path(split.slice_strata_dir).rglob("*")
            if path.is_file() or path.is_symlink())
    paths.update(REPO_ROOT / relative for relative in ANALYSIS.production_source_files())
    for cell in prepared["cells"]:
        paths.add(pathlib.Path(cell["artifact_path"]))
        paths.add(pathlib.Path(cell["fingerprint_path"]))
    validation_root = pathlib.Path(args.stats_root) / ANALYSIS.VALIDATION_BUILDING_NAME
    if validation_root.exists():
        paths.update(
            path for path in validation_root.rglob("*")
            if path.is_file() or path.is_symlink())
    return [_stat_witness(path) for path in sorted(paths, key=lambda p: str(p))]


def _publication_final_readiness(args, split=None):
    """Rederive selection and all dependencies before any test NPZ access."""
    expected_audit = pathlib.Path(args.stats_root) / (
        "context_availability.audit.json")
    if os.path.abspath(args.audit_identity) != os.path.abspath(expected_audit):
        raise RuntimeError(
            "publication final readiness requires context_availability.audit.json "
            "directly under the Work 3 stats root")
    if os.path.abspath(args.references) != os.path.abspath(DEFAULT_REFERENCES):
        raise RuntimeError(
            "publication final readiness requires configs/"
            "pf_timescale_references.yml")
    for path, label in (
            (pathlib.Path(args.audit_identity), "availability audit identity"),
            (pathlib.Path(args.references), "external reference records")):
        if not os.path.lexists(path) or not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"{path}: {label} must be a no-follow regular file")
    if split is None:
        split = _load_manifest(
            args.split, "publication", available_shots=None)
    prepared = ANALYSIS._prepare_validation_inputs(args)
    if tuple(int(shot) for shot in prepared["split"].test) != tuple(
            int(shot) for shot in split.test):
        raise RuntimeError("selection and final scorer loaded different test splits")
    derived = ANALYSIS._derive_validation_freeze(args, prepared)
    selection_bytes = _read_regular_bytes(
        pathlib.Path(args.stats_root) / ANALYSIS.VALIDATION_SELECTION_NAME,
        label="immutable Work 3 validation selection")
    table_bytes = _read_regular_bytes(
        pathlib.Path(args.stats_root) / ANALYSIS.GENERATED_NAME
        / ANALYSIS.VALIDATION_TABLE_NAME,
        label="immutable Work 3 validation table")
    if table_bytes != derived["table_bytes"]:
        raise RuntimeError(
            "immutable Work 3 validation table differs from byte rederivation")
    if selection_bytes != derived["selection_bytes"]:
        raise RuntimeError(
            "immutable Work 3 validation selection differs from byte rederivation")
    try:
        selection = ANALYSIS.validate_selection_lifecycle(
            json.loads(selection_bytes.decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot parse immutable Work 3 validation selection: {exc}") from exc
    if selection_bytes != ANALYSIS.json_bytes(selection):
        raise RuntimeError("immutable Work 3 validation selection is noncanonical")
    reference_transition = validate_final_reference_transition(
        selection, prepared["references"])
    dependency = validate_work2_dependency(
        args.work2_marker,
        args.floor_csv,
        split=prepared["split"],
        split_path=args.split,
        target_dir=args.target_dir,
        sidecar_dir=args.sidecar_dir,
    )
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    audit = prepared.get("audit")
    if audit is None:
        raise RuntimeError("publication final readiness requires availability audit")
    audit_record = {
        "path": str(pathlib.Path(args.audit_identity)),
        "schema": audit.identity["schema"],
        "sha256": audit.sha256,
        "identity": audit.identity,
    }
    work2_record = {
        "path": str(pathlib.Path(args.work2_marker)),
        "schema": dependency.marker["schema"],
        "marker_sha256": dependency.marker_sha256,
        "validation_decision_sha256": dependency.marker[
            "validation_decision"]["validation_decision_sha256"],
    }
    floor_record = {
        "path": str(pathlib.Path(args.floor_csv)),
        **dict(dependency.floor),
    }
    selection_record = {
        "path": str(pathlib.Path(args.stats_root)
                    / ANALYSIS.VALIDATION_SELECTION_NAME),
        "schema": selection["schema"],
        "sha256": selection_sha256,
        "selected": selection["selected"],
        "predecessor": selection["predecessor"],
        "verdict": selection["verdict"],
        "selection_lifecycle": "false_at_selection",
    }
    cells = prepared["cells"]
    cell_records = [{
        "run_name": cell["run_name"],
        "context": cell["entry"].context.label,
        "seed": int(cell["entry"].seed),
        "run_fingerprint": cell["run_fingerprint"],
        "artifact_sha256": cell["artifact_sha256"],
        "fingerprint_json_sha256": cell["fingerprint_json_sha256"],
    } for cell in cells]
    disclosure = selection.get("publication")
    if not isinstance(disclosure, dict):
        raise RuntimeError("selection lacks complete publication disclosure")
    transaction = build_transaction_provenance(
        split_disclosure=disclosure,
        validation_selection=selection_record,
        availability_audit=audit_record,
        work2=work2_record,
        floor=floor_record,
        references=reference_transition,
        matrix={
            "contexts": [level.label for level in CONTEXT_LEVELS],
            "seeds": list(PRODUCTION_SEEDS),
            "run_prefix": str(args.run_prefix),
            "devices": [str(device) for device in args.devices],
            "n_cells": len(cells),
        },
        input_hashes=prepared["fingerprints"],
        cells=cell_records,
    )
    transaction_sha256 = transaction_provenance_sha256(transaction)
    final = FinalReadiness(
        split_disclosure=disclosure,
        validation_selection_sha256=selection_sha256,
        availability_audit_sha256=audit.sha256,
        work2_marker_sha256=dependency.marker_sha256,
        floor_sha256=dependency.floor_sha256,
        transaction_sha256=transaction_sha256,
    )
    return {
        "final": final,
        "split": prepared["split"],
        "cells": cells,
        "selection": selection,
        "selection_bytes": selection_bytes,
        "audit": audit,
        "work2": dependency,
        "references": reference_transition,
        "transaction": transaction,
        "transaction_bytes": canonical_transaction_provenance_bytes(transaction),
        "transaction_sha256": transaction_sha256,
        "witness": _publication_readiness_witness(args, prepared),
    }


def _assert_same_publication_readiness(before, after, *, stage):
    if before["transaction_bytes"] != after["transaction_bytes"]:
        raise RuntimeError(f"publication final readiness changed during {stage}")
    if before.get("witness") != after.get("witness"):
        raise RuntimeError(
            f"publication final readiness witness changed during {stage}")


# ── scoring: deterministic partition, workers or the parent on CPU ────
def partition_entries(entries, n_workers):
    """Worker ``w`` of ``n`` scores ``entries[w::n]``.

    Deterministic by construction: a pure function of the ordered matrix and
    the worker count (no timing, no environment, no hash-order inputs), so
    the parent and every worker derive the identical disjoint partition, and
    re-deriving it in a fresh process yields the same assignment.
    """
    workers = int(n_workers)
    if workers < 1:
        raise ValueError("n_workers must be positive")
    return [list(entries)[i::workers] for i in range(workers)]


def _cell_dir(root, entry):
    return pathlib.Path(root) / entry.context.label / f"s{int(entry.seed)}"


def _cell_complete(cell_dir):
    return all((cell_dir / name).is_file() for name in (
        "m3_pred.npz", "per_shot_metrics.csv", "run_metadata.json"))


def _load_theta(target_dir):
    meta = json.loads(
        (pathlib.Path(target_dir) / "meta.json").read_text())
    if "theta_deg" not in meta:
        raise ValueError(
            f"{target_dir}/meta.json: no 'theta_deg' -- the uniform angle "
            "grid is required to rebuild contours")
    return np.deg2rad(np.asarray(meta["theta_deg"], float))


def _floor_provenance(args):
    """The upstream representation-floor table: located, hashed, counted --
    never recomputed, never subtracted from any metric."""
    path = pathlib.Path(getattr(args, "floor_csv", None) or DEFAULT_FLOOR_CSV)
    provenance = {"path": str(path), "present": path.is_file()}
    if provenance["present"]:
        with path.open(newline="") as fh:
            provenance["n_rows"] = max(
                sum(1 for _ in csv.DictReader(fh)), 0)
        provenance["sha256"] = sha256_file(path)
    return provenance


def score_entries(args, entries, device, split=None, *, readiness=None):
    """Score one deterministic partition, resuming only fully validated cells."""
    infer = pfctx_infer_module()
    publication = _manifest_mode(args) == "publication"
    if publication:
        readiness = (_publication_final_readiness(args, split=split)
                     if readiness is None else readiness)
        split = readiness["split"]
        transaction = readiness["transaction"]
        transaction_sha256 = readiness["transaction_sha256"]
        transaction_bytes = _read_regular_bytes(
            transaction_path(args), label="transaction provenance")
        validate_transaction_provenance_bytes(
            transaction_bytes, expected=transaction)
        if hashlib.sha256(transaction_bytes).hexdigest() != transaction_sha256:
            raise RuntimeError("staging transaction provenance hash changed")
        cells = {
            cell["entry"].run_name: cell for cell in readiness["cells"]}
        floor = {"path": str(args.floor_csv), **dict(readiness["work2"].floor)}
        device_token = "cpu" if str(device) == "cpu" else f"cuda:{device}"
    else:
        if split is None:
            split = _load_manifest(
                args.split, _manifest_mode(args), available_shots=None)
        transaction = None
        transaction_sha256 = None
        cells = {}
        floor = _floor_provenance(args)
        device_token = str(device)
    theta = _load_theta(args.target_dir)
    staging = building_root(args)
    staged = resumed = 0
    if publication:
        with open_final_tree_directory(staging, create=False) as tree:
            for entry in entries:
                current = cells[entry.run_name]
                with open_final_cell_directory(
                        tree, entry, create=False) as opened:
                    names = set(os.listdir(opened.cell_fd))
                    if names == CELL_MEMBER_NAMES:
                        _validate_staged_cell(
                            opened.path, current, split,
                            transaction_provenance=transaction,
                            transaction_sha256=transaction_sha256,
                            expected_device=device_token,
                            require_publication_count=True,
                            cell_fd=opened.cell_fd)
                        resumed += 1
                        print(
                            f"resume {entry.run_name} "
                            "(validated staged outputs)", flush=True)
                        continue
                    if not names <= CELL_MEMBER_NAMES:
                        raise RuntimeError(
                            f"{opened.path}: unexpected cell members "
                            f"{sorted(names - CELL_MEMBER_NAMES)}")
                    if "run_metadata.json" in names:
                        raise RuntimeError(
                            f"{opened.path}: metadata commit exists without "
                            "a complete cell")
                    print(f"score {entry.run_name} on {device_token}", flush=True)
                    payload = infer.score_artifact_payload(
                        current["artifact"], list(split.test),
                        pathlib.Path(args.target_dir),
                        pathlib.Path(args.sidecar_dir), theta,
                        device=device_token, floor=floor,
                        transaction_provenance=transaction,
                        transaction_sha256=transaction_sha256,
                        artifact_sha256=current["artifact_sha256"],
                        fingerprint_json_sha256=current[
                            "fingerprint_json_sha256"],
                        validation_selection_sha256=readiness[
                            "final"].validation_selection_sha256,
                        availability_audit_sha256=readiness[
                            "final"].availability_audit_sha256,
                        work2_marker_sha256=readiness[
                            "final"].work2_marker_sha256,
                        floor_sha256=readiness["final"].floor_sha256)
                    _commit_publication_cell_payload(
                        args, split, current, payload, readiness, opened,
                        expected_device=device_token)
                    staged += 1
    else:
        for entry in entries:
            cell_dir = _cell_dir(staging, entry)
            if _cell_complete(cell_dir):
                resumed += 1
                print(f"resume {entry.run_name} (staged outputs present)",
                      flush=True)
                continue
            artifact = torch.load(
                pathlib.Path(args.out_root) / entry.run_name / "m3.pt",
                map_location="cpu", weights_only=False)
            print(f"score {entry.run_name} on {device_token}", flush=True)
            infer.score_artifact(
                artifact, list(split.test), pathlib.Path(args.target_dir),
                pathlib.Path(args.sidecar_dir), theta, cell_dir,
                device=device_token, floor=floor)
            staged += 1
    print(f"partition done: {staged} scored, {resumed} resumed", flush=True)
    return staged, resumed


def _validate_parent_transaction_from_root(cell, readiness):
    transaction_bytes = _read_regular_child_bytes(
        cell.root_fd, TRANSACTION_NAME,
        path=cell.path.parent.parent / TRANSACTION_NAME,
        label="parent-published transaction provenance")
    validate_transaction_provenance_bytes(
        transaction_bytes, expected=readiness["transaction"])
    if hashlib.sha256(transaction_bytes).hexdigest() != \
            readiness["transaction_sha256"]:
        raise RuntimeError("parent-published transaction hash changed")


def _commit_publication_cell_payload(
        args, split, current, payload, readiness, cell, *, expected_device):
    """Revalidate after scoring and before every immutable member commit."""
    infer = pfctx_infer_module()

    def fresh(stage):
        updated = _publication_final_readiness(args, split=split)
        _assert_same_publication_readiness(readiness, updated, stage=stage)
        _validate_parent_transaction_from_root(cell, readiness)
        return updated

    fresh("post-score worker revalidation")
    fresh("prediction commit")
    _publish_prevalidated_bytes(
        cell.path / "m3_pred.npz", payload.prediction_bytes,
        validator=lambda data: infer._validate_prediction_bytes(
            data, payload.metadata["shots"]),
        state_label="prediction NPZ", directory_fd=cell.cell_fd)
    fresh("metrics commit")
    _publish_prevalidated_bytes(
        cell.path / "per_shot_metrics.csv", payload.metrics_bytes,
        validator=lambda data: infer._validate_metrics_bytes(
            data, payload.metadata["shots"]),
        state_label="metrics CSV", directory_fd=cell.cell_fd)
    fresh("metadata commit")
    _publish_prevalidated_bytes(
        cell.path / "run_metadata.json", payload.metadata_bytes,
        validator=lambda data: (
            data == payload.metadata_bytes
            and json.loads(data.decode("utf-8")) == payload.metadata)
        or (_ for _ in ()).throw(
            RuntimeError("run metadata differs from scored private bytes")),
        state_label="run metadata", directory_fd=cell.cell_fd)
    _validate_staged_cell(
        cell.path, current, split,
        transaction_provenance=readiness["transaction"],
        transaction_sha256=readiness["transaction_sha256"],
        expected_device=expected_device,
        require_publication_count=(
            _manifest_mode(args) == "publication"),
        cell_fd=cell.cell_fd,
    )


def _worker_command(args, worker_index, n_workers, device, *, lock_fd=None):
    command = [
        sys.executable, str(pathlib.Path(__file__).resolve()), WORKER_FLAG,
        "--worker-index", str(int(worker_index)),
        "--n-workers", str(int(n_workers)), "--device", str(device),
        "--split", str(args.split),
        "--manifest-mode", _manifest_mode(args),
        "--config", str(args.config),
        "--target-dir", str(args.target_dir),
        "--sidecar-dir", str(args.sidecar_dir),
        "--out-root", str(args.out_root),
        "--stats-root", str(args.stats_root),
        "--contexts", *[x.label for x in args.contexts],
        "--seeds", *[str(int(s)) for s in args.seeds],
        "--run-prefix", str(args.run_prefix),
        "--work2-marker", str(getattr(args, "work2_marker", None)
                               or DEFAULT_WORK2_MARKER),
        "--floor-csv", str(getattr(args, "floor_csv", None)
                           or DEFAULT_FLOOR_CSV),
        "--references", str(getattr(args, "references", None)
                            or DEFAULT_REFERENCES),
        "--audit-identity", str(getattr(args, "audit_identity", None)
                                or DEFAULT_AUDIT_IDENTITY),
    ]
    if lock_fd is not None:
        command += ["--lock-fd", str(int(lock_fd))]
    return command


def _validate_worker_authorization(args):
    lock_fd = getattr(args, "lock_fd", None)
    if type(lock_fd) is not int or lock_fd < 0:
        raise RuntimeError(
            "publication hidden worker requires inherited parent lock authorization")
    try:
        opened = os.fstat(lock_fd)
    except OSError as exc:
        raise RuntimeError("inherited parent lock authorization is not open") from exc
    lock_path = pathlib.Path(args.stats_root) / ".validation_freeze.lock"
    try:
        current = lock_path.lstat()
    except OSError as exc:
        raise RuntimeError("canonical parent lock path is unavailable") from exc
    if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (current.st_dev, current.st_ino)):
        raise RuntimeError(
            "inherited parent lock authorization does not match canonical lock")
    try:
        lock_payload = json.loads(os.pread(lock_fd, opened.st_size, 0).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot validate inherited parent lock metadata") from exc
    if (lock_payload.get("schema")
            != "pf_observability_validation_freeze_lock_v1"
            or type(lock_payload.get("pid")) is not int
            or lock_payload.get("pid") != os.getppid()):
        raise RuntimeError("inherited lock is not owned by the active parent")
    probe_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        probe = os.fstat(probe_fd)
        if (not stat.S_ISREG(probe.st_mode)
                or (probe.st_dev, probe.st_ino)
                != (opened.st_dev, opened.st_ino)):
            raise RuntimeError("canonical lock probe opened another inode")
        try:
            fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe_fd, fcntl.LOCK_UN)
            raise RuntimeError(
                "canonical lock is not actively held by the parent transaction")
    finally:
        os.close(probe_fd)
    with open_final_tree_directory(
            building_root(args), create=False) as transaction_root:
        transaction_bytes = _read_regular_child_bytes(
            transaction_root.root_fd, TRANSACTION_NAME,
            path=transaction_path(args),
            label="parent-published transaction provenance")
    validate_transaction_provenance_bytes(transaction_bytes)
    return {
        "lock_identity": (opened.st_dev, opened.st_ino),
        "transaction_sha256": hashlib.sha256(transaction_bytes).hexdigest(),
    }


def _stop_and_reap_workers(records, *, timeout=5.0):
    deadline = time.monotonic() + max(float(timeout), 0.0)
    for _index, _device, _command, process in records:
        try:
            if process.poll() is None:
                process.terminate()
        except BaseException:
            pass
    for _index, _device, _command, process in records:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
            continue
        except BaseException:
            pass
        try:
            if process.poll() is None:
                process.kill()
        except BaseException:
            pass
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException:
            pass


def _spawn_device_workers(args, devices, *, lock_fd=None,
                          poll_interval=0.05, stop_timeout=5.0):
    """Spawn inherited-log workers and stop siblings at the first failure."""
    commands = [_worker_command(
        args, index, len(devices), device, lock_fd=lock_fd)
        for index, device in enumerate(devices)]
    records = []
    try:
        for index, (device, command) in enumerate(zip(devices, commands)):
            popen_kwargs = {
                "cwd": str(REPO_ROOT),
                "env": {
                    **os.environ,
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
                    "HDF5_USE_FILE_LOCKING": "FALSE",
                },
            }
            if lock_fd is not None:
                popen_kwargs["pass_fds"] = (int(lock_fd),)
            process = subprocess.Popen(command, **popen_kwargs)
            records.append((index, device, command, process))
    except BaseException as exc:
        _stop_and_reap_workers(records, timeout=stop_timeout)
        if isinstance(exc, Exception):
            raise RuntimeError("could not start final scoring workers") from exc
        raise

    pending = {id(record[3]): record for record in records}
    try:
        while pending:
            for key, record in list(pending.items()):
                index, device, command, process = record
                try:
                    returncode = process.poll()
                except BaseException:
                    _stop_and_reap_workers(records, timeout=stop_timeout)
                    raise
                if returncode is None:
                    continue
                try:
                    waited = process.wait(timeout=stop_timeout)
                except BaseException:
                    _stop_and_reap_workers(records, timeout=stop_timeout)
                    raise
                if waited != 0:
                    _stop_and_reap_workers(records, timeout=stop_timeout)
                    raise RuntimeError(
                        f"device workers failed: worker {index} on {device} "
                        f"returned code {waited}; command: {command!r}")
                print(
                    f"device worker ok: worker {index}, device {device}",
                    flush=True)
                pending.pop(key)
            if pending:
                time.sleep(max(float(poll_interval), 0.0))
    except BaseException:
        _stop_and_reap_workers(records, timeout=stop_timeout)
        raise


def _canonical_csv_int(value, *, path, field, minimum=0):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{path}: {field} is not an integer") from exc
    if parsed < minimum or str(parsed) != str(value).strip():
        raise RuntimeError(
            f"{path}: {field} must be a canonical integer >= {minimum}")
    return parsed


def _finite_csv_float(value, *, path, field):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{path}: {field} is not numeric") from exc
    if not np.isfinite(parsed):
        raise RuntimeError(f"{path}: {field} is non-finite")
    return parsed


def _validate_cell_tree(cell_dir):
    cell_dir = pathlib.Path(cell_dir)
    if not os.path.lexists(cell_dir) or not stat.S_ISDIR(cell_dir.lstat().st_mode):
        raise RuntimeError(f"{cell_dir}: final cell must be a real no-follow directory")
    names = {entry.name for entry in cell_dir.iterdir()}
    if names != CELL_MEMBER_NAMES:
        raise RuntimeError(
            f"{cell_dir}: cell tree membership is not exact "
            f"(missing={sorted(CELL_MEMBER_NAMES - names)}, "
            f"extra={sorted(names - CELL_MEMBER_NAMES)})")
    for name in CELL_MEMBER_NAMES:
        path = cell_dir / name
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"{path}: final cell member must be a no-follow file")


def _require_type_strict_equal(actual, expected, *, path):
    if type(actual) is not type(expected):
        raise RuntimeError(
            f"{path} type changed from {type(expected).__name__} to "
            f"{type(actual).__name__}")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise RuntimeError(f"{path} mapping keys changed")
        for key in sorted(expected):
            _require_type_strict_equal(
                actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeError(f"{path} list length changed")
        for index, (actual_item, expected_item) in enumerate(
                zip(actual, expected)):
            _require_type_strict_equal(
                actual_item, expected_item, path=f"{path}[{index}]")
        return
    if actual != expected:
        raise RuntimeError(f"{path} value changed")


def _metadata_hash_expectations(transaction_provenance):
    return {
        "validation_selection_sha256": transaction_provenance[
            "validation_selection"]["sha256"],
        "availability_audit_sha256": transaction_provenance[
            "availability_audit"]["sha256"],
        "work2_marker_sha256": transaction_provenance[
            "work2"]["marker_sha256"],
        "floor_sha256": transaction_provenance["floor"]["sha256"],
    }


def _validate_staged_cell(
        cell_dir, cell, split, *, transaction_provenance,
        transaction_sha256, expected_device, require_publication_count=True,
        cell_fd=None, member_bytes=None):
    """Validate all three exact cell members before resume or publication."""
    cell_dir = pathlib.Path(cell_dir)
    metrics_path = cell_dir / "per_shot_metrics.csv"
    predictions_path = cell_dir / "m3_pred.npz"
    metadata_path = cell_dir / "run_metadata.json"
    if member_bytes is not None:
        if set(member_bytes) != CELL_MEMBER_NAMES:
            raise RuntimeError(
                f"{cell_dir}: retained member byte set is not exact")
        metrics_bytes = bytes(member_bytes["per_shot_metrics.csv"])
        prediction_bytes = bytes(member_bytes["m3_pred.npz"])
        metadata_bytes = bytes(member_bytes["run_metadata.json"])
    elif cell_fd is None:
        _validate_cell_tree(cell_dir)
        metrics_bytes = _read_regular_bytes(metrics_path, label="cell metrics")
        prediction_bytes = _read_regular_bytes(
            predictions_path, label="cell predictions")
        metadata_bytes = _read_regular_bytes(metadata_path, label="cell metadata")
    else:
        names = set(os.listdir(cell_fd))
        if names != CELL_MEMBER_NAMES:
            raise RuntimeError(f"{cell_dir}: cell tree membership is not exact")
        metrics_bytes = _read_regular_child_bytes(
            cell_fd, "per_shot_metrics.csv", path=metrics_path,
            label="cell metrics")
        prediction_bytes = _read_regular_child_bytes(
            cell_fd, "m3_pred.npz", path=predictions_path,
            label="cell predictions")
        metadata_bytes = _read_regular_child_bytes(
            cell_fd, "run_metadata.json", path=metadata_path,
            label="cell metadata")

    try:
        reader = csv.DictReader(io.StringIO(
            metrics_bytes.decode("utf-8"), newline=""))
    except (UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"cannot parse cell metrics at {metrics_path}: {exc}") \
            from exc
    if tuple(reader.fieldnames or ()) != tuple(
            pfctx_infer_module().PER_SHOT_HEADER):
        raise RuntimeError(f"{metrics_path}: metrics header changed")
    rows = list(reader)
    expected_shots = tuple(int(shot) for shot in split.test)
    if require_publication_count and len(expected_shots) != 76:
        raise RuntimeError("publication final scoring requires exactly 76 test shots")
    if len(rows) != len(expected_shots):
        raise RuntimeError(
            f"{metrics_path}: {len(rows)} rows for {len(expected_shots)} shots")
    parsed_shots = []
    counts = {}
    for row_index, row in enumerate(rows, start=2):
        if None in row or set(row) != set(pfctx_infer_module().PER_SHOT_HEADER):
            raise RuntimeError(f"{metrics_path}: row {row_index} has wrong schema")
        context = str(row["context"])
        seed = _canonical_csv_int(
            row["seed"], path=metrics_path, field=f"row {row_index} seed")
        shot = _canonical_csv_int(
            row["shot"], path=metrics_path, field=f"row {row_index} shot")
        n_slices = _canonical_csv_int(
            row["n_slices"], path=metrics_path,
            field=f"row {row_index} n_slices", minimum=1)
        if (context != cell["entry"].context.label
                or seed != int(cell["entry"].seed)):
            raise RuntimeError(
                f"{metrics_path}: row {row_index} claims another cell")
        for field in pfctx_infer_module().PER_SHOT_HEADER[4:]:
            _finite_csv_float(
                row[field], path=metrics_path,
                field=f"row {row_index} {field}")
        parsed_shots.append(shot)
        counts[shot] = n_slices
    if tuple(parsed_shots) != expected_shots:
        raise RuntimeError(
            f"{metrics_path}: shot membership/order differs from the frozen test")

    try:
        with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as data:
            expected_keys = tuple(
                pfctx_infer_module().prediction_key(shot, kind)
                for shot in expected_shots
                for kind in ("prediction", "row_index", "timestamp")
            )
            if tuple(data.files) != expected_keys:
                raise RuntimeError(
                    f"{predictions_path}: prediction member set/order changed")
            reference_rows = {}
            for shot in expected_shots:
                prediction = data[
                    pfctx_infer_module().prediction_key(shot, "prediction")]
                row_index = data[
                    pfctx_infer_module().prediction_key(shot, "row_index")]
                timestamp = data[
                    pfctx_infer_module().prediction_key(shot, "timestamp")]
                expected_count = counts[shot]
                if prediction.shape != (expected_count, 34):
                    raise RuntimeError(
                        f"{predictions_path}: shot {shot} prediction shape changed")
                if (row_index.shape != (expected_count,)
                        or timestamp.shape != (expected_count,)):
                    raise RuntimeError(
                        f"{predictions_path}: shot {shot} row/timestamp count changed")
                if (prediction.dtype != np.float32
                        or row_index.dtype != np.int64
                        or timestamp.dtype != np.float64):
                    raise RuntimeError(
                        f"{predictions_path}: shot {shot} dtype contract changed")
                if (not np.isfinite(prediction).all()
                        or not np.isfinite(timestamp).all()):
                    raise RuntimeError(
                        f"{predictions_path}: shot {shot} contains non-finite values")
                if not np.all(np.diff(row_index) > 0):
                    raise RuntimeError(
                        f"{predictions_path}: shot {shot} rows are not unique/increasing")
                reference_rows[shot] = np.asarray(row_index, np.int64)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot parse cell predictions at {predictions_path}: {exc}") from exc

    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse cell metadata at {metadata_path}: {exc}") \
            from exc
    infer = pfctx_infer_module()
    if metadata_bytes != _canonical_json_bytes(metadata):
        raise RuntimeError(f"{metadata_path}: metadata bytes are not canonical")
    if not isinstance(metadata, dict) or set(metadata) != infer.RUN_METADATA_FIELDS:
        raise RuntimeError(f"{metadata_path}: metadata schema changed")
    artifact = cell["artifact"]
    expected_identity = {
        "study": "pf_context",
        "run_name": cell["entry"].run_name,
        "context_label": cell["entry"].context.label,
        "nominal_samples": int(artifact["nominal_samples"]),
        "context_seconds": float(artifact["context_seconds"]),
        "per_layer_seconds": float(artifact["per_layer_seconds"]),
        "score_block": int(artifact["score_block"]),
        "seed": int(cell["entry"].seed),
        "depth": int(artifact["depth"]),
        "world_size": int(artifact["world_size"]),
        "effective_global_batch": int(artifact["effective_global_batch"]),
        "best_val_mse": float(artifact["best_val_mse"]),
        "stop_epoch": int(artifact["stop_epoch"]),
        "epochs_completed": int(artifact["epochs_completed"]),
        "device": str(expected_device),
        "n_shots": len(expected_shots),
        "shots": list(expected_shots),
        "n_pred_rows": sum(counts.values()),
        "row_order_contract": (
            "strictly increasing unique native rows equal to each shot's "
            "common-valid indices"),
        "fingerprints": cell["run_fingerprint"],
        "floor": transaction_provenance["floor"],
        "transaction_provenance": transaction_provenance,
        "transaction_sha256": transaction_sha256,
        "artifact_sha256": cell["artifact_sha256"],
        "fingerprint_json_sha256": cell["fingerprint_json_sha256"],
        **_metadata_hash_expectations(transaction_provenance),
    }
    for field, expected in expected_identity.items():
        if field not in metadata:
            raise RuntimeError(f"{metadata_path}: metadata field {field} is absent")
        _require_type_strict_equal(
            metadata[field], expected,
            path=f"run_metadata.{field}")
    try:
        scored_at = datetime.datetime.fromisoformat(metadata["scored_at"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{metadata_path}: scored_at is not ISO-8601") from exc
    if scored_at.tzinfo is None:
        raise RuntimeError(f"{metadata_path}: scored_at must carry a timezone")
    if not np.isfinite(float(metadata["best_val_mse"])):
        raise RuntimeError(f"{metadata_path}: best_val_mse is non-finite")
    return {
        "run_name": cell["entry"].run_name,
        "context": cell["entry"].context.label,
        "seed": int(cell["entry"].seed),
        "device": str(expected_device),
        "n_shots": len(rows),
        "n_pred_rows": sum(counts.values()),
        "best_val_mse": float(metadata["best_val_mse"]),
        "run_fingerprint": cell["run_fingerprint"],
        "artifact_sha256": cell["artifact_sha256"],
        "fingerprint_json_sha256": cell["fingerprint_json_sha256"],
        "per_shot_metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "m3_pred_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "run_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "row_index": reference_rows,
    }


def _verify_staging(args, entries, split, *, cells=None,
                    transaction_provenance=None, transaction_sha256=None,
                    root=None, require_publication_count=False, tree=None,
                    matrix=None, leaf_descriptors=None, leaf_witnesses=None):
    """Every cell complete, exact, provenance-bound, and row-aligned."""
    staging = pathlib.Path(building_root(args) if root is None else root)
    expected_shots = tuple(int(s) for s in split.test)
    if leaf_descriptors is not None and matrix is None:
        raise RuntimeError("retained final leaves require their retained matrix")
    if leaf_witnesses is not None and leaf_descriptors is None:
        raise RuntimeError("retained final-leaf witnesses require leaf descriptors")
    if transaction_provenance is not None and matrix is None:
        with open_final_matrix_directory(
                staging, entries, require_complete=True) as opened_matrix:
            return _verify_staging(
                args, entries, split, cells=cells,
                transaction_provenance=transaction_provenance,
                transaction_sha256=transaction_sha256, root=staging,
                require_publication_count=require_publication_count,
                tree=opened_matrix.tree, matrix=opened_matrix)
    if transaction_provenance is None:
        infer = pfctx_infer_module()
        reference_indices = {}
        summaries = []
        for entry in entries:
            cell = _cell_dir(staging, entry)
            absent = [name for name in CELL_MEMBER_NAMES
                      if not (cell / name).is_file()]
            if absent:
                raise RuntimeError(
                    f"{cell}: incomplete after scoring "
                    f"(missing {', '.join(absent)})")
            with (cell / "per_shot_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            predictions = infer.load_final_predictions(cell / "m3_pred.npz")
            if sorted(predictions) != sorted(expected_shots):
                raise RuntimeError(f"{cell}: prediction shots differ from split")
            for shot in expected_shots:
                rows_here = predictions[shot].row_index
                if shot in reference_indices and not np.array_equal(
                        rows_here, reference_indices[shot]):
                    raise RuntimeError(
                        f"shot {shot}: row indices differ across contexts")
                reference_indices.setdefault(shot, rows_here)
            metadata = json.loads((cell / "run_metadata.json").read_text())
            summaries.append({
                "run_name": metadata["run_name"],
                "context": metadata["context_label"],
                "seed": int(metadata["seed"]),
                "n_shots": len(rows),
                "n_pred_rows": int(sum(int(row["n_slices"]) for row in rows)),
                "best_val_mse": float(metadata["best_val_mse"]),
                "fingerprints": metadata["fingerprints"],
                "per_shot_metrics_sha256": sha256_file(
                    cell / "per_shot_metrics.csv"),
                "m3_pred_sha256": sha256_file(cell / "m3_pred.npz"),
                "run_metadata_sha256": sha256_file(
                    cell / "run_metadata.json"),
            })
        return summaries

    if leaf_descriptors is None:
        transaction_bytes = _read_regular_child_bytes(
            tree.root_fd, TRANSACTION_NAME,
            path=transaction_path(args, root=staging),
            label="transaction provenance")
    else:
        transaction_bytes, transaction_witness = \
            leaf_descriptors.transaction.read_snapshot()
        if leaf_witnesses is not None:
            leaf_witnesses[id(leaf_descriptors.transaction)] = \
                transaction_witness
    validate_transaction_provenance_bytes(
        transaction_bytes, expected=transaction_provenance)
    if hashlib.sha256(transaction_bytes).hexdigest() != transaction_sha256:
        raise RuntimeError("transaction provenance hash changed in staging")
    by_name = {cell["entry"].run_name: cell for cell in cells or ()}
    reference_indices = {}
    summaries = []
    devices = [str(device) for device in args.devices]
    for index, entry in enumerate(entries):
        current = by_name.get(entry.run_name)
        if current is None:
            raise RuntimeError(f"{entry.run_name}: readiness cell is absent")
        expected_device = (
            "cpu" if len(devices) == 1 and devices[0] == "cpu"
            else f"cuda:{index % len(devices)}")
        retained_member_bytes = None
        if leaf_descriptors is not None:
            retained_member_bytes = {}
            for name in CELL_MEMBER_ORDER:
                retained = leaf_descriptors.member(entry, name)
                payload, witness = retained.read_snapshot()
                retained_member_bytes[name] = payload
                if leaf_witnesses is not None:
                    leaf_witnesses[id(retained)] = witness
        summary = _validate_staged_cell(
            _cell_dir(staging, entry), current, split,
            transaction_provenance=transaction_provenance,
            transaction_sha256=transaction_sha256,
            expected_device=expected_device,
            require_publication_count=require_publication_count,
            cell_fd=matrix.cell_fd(entry),
            member_bytes=retained_member_bytes)
        row_indices = summary.pop("row_index")
        for shot in expected_shots:
            rows_here = row_indices[shot]
            if shot in reference_indices and not np.array_equal(
                    rows_here, reference_indices[shot]):
                raise RuntimeError(
                    f"shot {shot}: row indices differ across final cells")
            reference_indices.setdefault(shot, rows_here)
        summaries.append(summary)
    return summaries


def _assert_retained_matrix_paths(matrix, entries, final_dir):
    """Rebind every published path entry to the still-open matrix descriptors."""
    final_dir = pathlib.Path(final_dir)
    published_root = final_dir.lstat()
    opened_root = os.fstat(matrix.root_fd)
    if (not stat.S_ISDIR(published_root.st_mode)
            or (published_root.st_dev, published_root.st_ino)
            != (opened_root.st_dev, opened_root.st_ino)):
        raise RuntimeError(
            "published final root path differs from retained descriptor")
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry.context.label, []).append(entry)
    root_names = set(os.listdir(matrix.root_fd))
    expected_root = set(grouped) | {TRANSACTION_NAME}
    if root_names != expected_root:
        raise RuntimeError("published final root topology changed")
    for context, context_entries in grouped.items():
        context_fd = matrix.context_fds[context]
        context_path = final_dir / context
        path_info = os.stat(
            context, dir_fd=matrix.root_fd, follow_symlinks=False)
        opened_info = os.fstat(context_fd)
        if (not stat.S_ISDIR(path_info.st_mode)
                or (path_info.st_dev, path_info.st_ino)
                != (opened_info.st_dev, opened_info.st_ino)):
            raise RuntimeError(
                f"{context_path}: published context path was replaced")
        expected_seeds = {f"s{int(entry.seed)}" for entry in context_entries}
        if set(os.listdir(context_fd)) != expected_seeds:
            raise RuntimeError(f"{context_path}: published seed topology changed")
        for entry in context_entries:
            seed_name = f"s{int(entry.seed)}"
            cell_fd = matrix.cell_fd(entry)
            cell_path = context_path / seed_name
            path_info = os.stat(
                seed_name, dir_fd=context_fd, follow_symlinks=False)
            opened_info = os.fstat(cell_fd)
            if (not stat.S_ISDIR(path_info.st_mode)
                    or (path_info.st_dev, path_info.st_ino)
                    != (opened_info.st_dev, opened_info.st_ino)):
                raise RuntimeError(
                    f"{cell_path}: published seed path was replaced")
            if set(os.listdir(cell_fd)) != CELL_MEMBER_NAMES:
                raise RuntimeError(
                    f"{cell_path}: published member topology changed")


def _verify_retained_final_pass(
        args, readiness, entries, expected_summaries, final_dir, matrix,
        leaf_descriptors):
    """Re-read, validate, and stabilize every retained final leaf."""
    witnesses = {}
    summaries = _verify_staging(
        args, entries, readiness["split"], cells=readiness["cells"],
        transaction_provenance=readiness["transaction"],
        transaction_sha256=readiness["transaction_sha256"],
        root=final_dir, require_publication_count=True,
        tree=matrix.tree, matrix=matrix,
        leaf_descriptors=leaf_descriptors, leaf_witnesses=witnesses)
    _require_type_strict_equal(
        summaries, expected_summaries,
        path="published_final.pre_marker_runs")
    _assert_retained_matrix_paths(matrix, entries, final_dir)
    leaf_descriptors.assert_current(witnesses)
    return summaries


def _rebind_published_final_tree(
        args, readiness, entries, expected_summaries, final_dir):
    """Freshly bind and validate only the topology published at final_dir."""
    with open_final_matrix_directory(
            final_dir, entries, require_complete=True) as published_matrix:
        summaries = _verify_staging(
            args, entries, readiness["split"], cells=readiness["cells"],
            transaction_provenance=readiness["transaction"],
            transaction_sha256=readiness["transaction_sha256"],
            root=final_dir, require_publication_count=True,
            tree=published_matrix.tree, matrix=published_matrix)
    _require_type_strict_equal(
        summaries, expected_summaries, path="published_final.runs")
    return summaries


def _publish_marker_from_retained_final(
        args, readiness, entries, expected_summaries, final_dir,
        transaction_lock):
    """Keep final directories and all 106 leaves open through marker commit."""
    marker = marker_path(args)
    with open_final_matrix_directory(
            final_dir, entries, require_complete=True) as published_matrix:
        with open_final_leaf_descriptors(
                published_matrix, entries) as published_leaves:
            initial_summaries = _verify_staging(
                args, entries, readiness["split"], cells=readiness["cells"],
                transaction_provenance=readiness["transaction"],
                transaction_sha256=readiness["transaction_sha256"],
                root=final_dir, require_publication_count=True,
                tree=published_matrix.tree, matrix=published_matrix,
                leaf_descriptors=published_leaves)
            _require_type_strict_equal(
                initial_summaries, expected_summaries,
                path="published_final.initial_runs")

            pre_marker = _publication_final_readiness(
                args, split=readiness["split"])
            _assert_same_publication_readiness(
                readiness, pre_marker, stage="final marker publication")
            _validate_final_readiness_consistency(pre_marker)
            transaction_lock.assert_held()
            _assert_retained_matrix_paths(
                published_matrix, entries, final_dir)
            final_summaries = _verify_retained_final_pass(
                args, pre_marker, entries, expected_summaries, final_dir,
                published_matrix, published_leaves)
            payload = _publication_marker_payload(
                args, pre_marker, final_summaries)
            marker_bytes = _canonical_json_bytes(payload)
            _validate_publication_marker_bytes(marker_bytes)
            transaction_lock.assert_held()
            _publish_prevalidated_bytes(
                marker, marker_bytes,
                validator=_validate_publication_marker_bytes,
                state_label="Work 3 final marker")
            return final_summaries


def _validate_incomplete_publication_staging(args, readiness):
    """Count valid complete staged cells through one retained subset matrix."""
    entries = [cell["entry"] for cell in readiness["cells"]]
    building = building_root(args)
    with open_final_matrix_directory(
            building, entries, require_complete=False) as matrix:
        transaction_bytes = _read_regular_child_bytes(
            matrix.root_fd, TRANSACTION_NAME,
            path=transaction_path(args), label="transaction provenance")
        validate_transaction_provenance_bytes(
            transaction_bytes, expected=readiness["transaction"])
        if hashlib.sha256(transaction_bytes).hexdigest() != \
                readiness["transaction_sha256"]:
            raise RuntimeError("staging transaction hash changed")
        by_name = {
            cell["entry"].run_name: cell for cell in readiness["cells"]}
        staged = 0
        for index, entry in enumerate(entries):
            cell_fd = matrix.cell_fds.get(entry.run_name)
            if cell_fd is None:
                continue
            names = set(os.listdir(cell_fd))
            if names == CELL_MEMBER_NAMES:
                _validate_staged_cell(
                    _cell_dir(building, entry), by_name[entry.run_name],
                    readiness["split"],
                    transaction_provenance=readiness["transaction"],
                    transaction_sha256=readiness["transaction_sha256"],
                    expected_device=f"cuda:{index % 4}",
                    require_publication_count=True, cell_fd=cell_fd)
                staged += 1
            elif "run_metadata.json" in names:
                raise RuntimeError(
                    f"{_cell_dir(building, entry)}: metadata commit is incomplete")
        return staged


def _require_regular_child(directory_fd, name, *, path):
    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{path}: final member must be no-follow regular")


def _validate_publication_tree_fd(root, root_fd, entries, *, require_complete):
    by_context = {}
    for entry in entries:
        by_context.setdefault(entry.context.label, set()).add(
            f"s{int(entry.seed)}")
    names = set(os.listdir(root_fd))
    allowed = set(by_context) | {TRANSACTION_NAME}
    if not names <= allowed:
        raise RuntimeError(f"{root}: unexpected final members {sorted(names - allowed)}")
    if TRANSACTION_NAME not in names:
        raise RuntimeError(f"{root}: transaction provenance is absent")
    _require_regular_child(
        root_fd, TRANSACTION_NAME, path=root / TRANSACTION_NAME)
    if require_complete and names != allowed:
        raise RuntimeError(f"{root}: final context membership is incomplete")
    for context, expected_seeds in by_context.items():
        if context not in names:
            if require_complete:
                raise RuntimeError(f"{root / context}: context directory is absent")
            continue
        context_path = root / context
        context_fd = _open_child_directory(
            root_fd, context, path=context_path,
            label="final context directory")
        try:
            seed_names = set(os.listdir(context_fd))
            if not seed_names <= expected_seeds:
                raise RuntimeError(
                    f"{context_path}: unexpected seed members "
                    f"{sorted(seed_names - expected_seeds)}")
            if require_complete and seed_names != expected_seeds:
                raise RuntimeError(f"{context_path}: seed membership is incomplete")
            for seed_name in sorted(seed_names):
                cell_path = context_path / seed_name
                cell_fd = _open_child_directory(
                    context_fd, seed_name, path=cell_path,
                    label="final cell directory")
                try:
                    member_names = set(os.listdir(cell_fd))
                    if not member_names <= CELL_MEMBER_NAMES:
                        raise RuntimeError(
                            f"{cell_path}: unexpected cell members "
                            f"{sorted(member_names - CELL_MEMBER_NAMES)}")
                    if ("run_metadata.json" in member_names
                            and member_names != CELL_MEMBER_NAMES):
                        raise RuntimeError(
                            f"{cell_path}: metadata commit exists without "
                            "a complete cell")
                    if require_complete and member_names != CELL_MEMBER_NAMES:
                        raise RuntimeError(
                            f"{cell_path}: cell membership is incomplete")
                    for member in member_names:
                        _require_regular_child(
                            cell_fd, member, path=cell_path / member)
                finally:
                    os.close(cell_fd)
        finally:
            os.close(context_fd)


def _validate_publication_tree(root, entries, *, require_complete, tree=None):
    root = pathlib.Path(root)
    if tree is None:
        with open_final_tree_directory(root, create=False) as opened:
            return _validate_publication_tree_fd(
                root, opened.root_fd, entries,
                require_complete=require_complete)
    return _validate_publication_tree_fd(
        root, tree.root_fd, entries, require_complete=require_complete)


def _prepare_publication_staging(args, entries, readiness):
    building = building_root(args)
    if os.path.lexists(building):
        if not stat.S_ISDIR(pathlib.Path(building).lstat().st_mode):
            raise RuntimeError(f"{building}: staging must be a real directory")
    else:
        pathlib.Path(building).mkdir(parents=True)
    transaction_bytes = readiness["transaction_bytes"]
    with open_final_tree_directory(building, create=False) as tree:
        _publish_prevalidated_bytes(
            transaction_path(args), transaction_bytes,
            validator=lambda payload: validate_transaction_provenance_bytes(
                payload, expected=readiness["transaction"]),
            state_label="transaction provenance", directory_fd=tree.root_fd)
        by_context = {}
        for entry in entries:
            by_context.setdefault(entry.context.label, set()).add(
                f"s{int(entry.seed)}")
        for context, seed_names in by_context.items():
            try:
                os.mkdir(context, mode=0o755, dir_fd=tree.root_fd)
            except FileExistsError:
                pass
            context_fd = _open_child_directory(
                tree.root_fd, context, path=pathlib.Path(building) / context,
                label="final context directory")
            try:
                for seed_name in sorted(seed_names):
                    try:
                        os.mkdir(seed_name, mode=0o755, dir_fd=context_fd)
                    except FileExistsError:
                        pass
                    seed_fd = _open_child_directory(
                        context_fd, seed_name,
                        path=pathlib.Path(building) / context / seed_name,
                        label="final cell directory")
                    os.close(seed_fd)
            finally:
                os.close(context_fd)
        _validate_publication_tree(
            building, entries, require_complete=False, tree=tree)
    return pathlib.Path(building)


def _require_production_matrix(args):
    """The one-shot transaction must never be consumed by a subset.

    Under the production-default stats root the marker's no-rerun guarantee
    would make a subset matrix permanently block the full 35-cell run, so
    anything but the frozen 7x5 matrix is refused before any staging. A
    ``--stats-root`` pointing elsewhere (the fixtures' temporary roots, a
    diagnostic run) is the sanctioned escape and keeps the parameterized
    shape ``verify_final_readiness`` exposes.
    """
    if (_manifest_mode(args) != "publication"
            and pathlib.Path(args.stats_root).resolve()
            != pathlib.Path(STATS_ROOT).resolve()):
        return
    contexts = tuple(x.label for x in args.contexts)
    seeds = tuple(int(s) for s in args.seeds)
    canonical_contexts = tuple(x.label for x in CONTEXT_LEVELS)
    if (contexts != canonical_contexts
            or seeds != PRODUCTION_SEEDS
            or (_manifest_mode(args) == "publication"
                and str(args.run_prefix) != "pfctx")):
        raise RuntimeError(
            f"refusing a noncanonical/subset matrix (contexts {contexts}, "
            f"seeds {seeds}) "
            f"under the production stats root {STATS_ROOT}: the one-shot "
            "final test scores the full "
            f"{len(CONTEXT_LEVELS)}x{len(PRODUCTION_SEEDS)} production "
            "matrix and its marker can never be rerun -- point --stats-root "
            "at a separate root for a diagnostic run")


def _publication_marker_payload(args, readiness, summaries):
    if len(summaries) != PRODUCTION_ARTIFACT_COUNT:
        raise RuntimeError("publication marker requires exact 35 cell summaries")
    return {
        "schema": WORK3_FINAL_MARKER_SCHEMA,
        "marker_generation_version": WORK3_FINAL_MARKER_GENERATION_VERSION,
        "marker": MARKER_NAME,
        "evaluated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "publication": dict(readiness["final"].split_disclosure),
        "validation_selection": {
            "sha256": readiness["final"].validation_selection_sha256,
            "payload": readiness["selection"],
        },
        "availability_audit": {
            "sha256": readiness["final"].availability_audit_sha256,
            "identity": readiness["audit"].identity,
        },
        "work2": {
            "marker_sha256": readiness["final"].work2_marker_sha256,
            "marker": dict(readiness["work2"].marker),
        },
        "representation_floor": {
            "path": str(pathlib.Path(args.floor_csv)),
            **dict(readiness["work2"].floor),
        },
        "references": dict(readiness["references"]),
        "transaction": {
            "sha256": readiness["transaction_sha256"],
            "provenance": readiness["transaction"],
        },
        "input_hashes": dict(readiness["transaction"]["input_hashes"]),
        "matrix": {
            "contexts": [level.label for level in CONTEXT_LEVELS],
            "seeds": list(PRODUCTION_SEEDS),
            "run_prefix": str(args.run_prefix),
            "devices": [str(device) for device in args.devices],
            "n_cells": len(summaries),
            "n_shots": len(readiness["split"].test),
        },
        "runs": list(summaries),
    }


def _validate_publication_marker_bytes(payload):
    try:
        marker = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse Work 3 final marker: {exc}") from exc
    expected_fields = {
        "schema", "marker_generation_version", "marker", "evaluated_at",
        "publication", "validation_selection", "availability_audit", "work2",
        "representation_floor", "references", "transaction", "input_hashes",
        "matrix", "runs",
    }
    if (not isinstance(marker, dict) or set(marker) != expected_fields
            or type(marker.get("schema")) is not str
            or marker.get("schema") != WORK3_FINAL_MARKER_SCHEMA
            or type(marker.get("marker_generation_version")) is not int
            or marker.get("marker_generation_version")
            != WORK3_FINAL_MARKER_GENERATION_VERSION
            or marker.get("marker") != MARKER_NAME):
        raise RuntimeError("Work 3 final marker schema changed")
    if bytes(payload) != _canonical_json_bytes(marker):
        raise RuntimeError("Work 3 final marker bytes are not canonical")
    try:
        evaluated_at = datetime.datetime.fromisoformat(marker["evaluated_at"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Work 3 final marker timestamp is invalid") from exc
    if evaluated_at.tzinfo is None:
        raise RuntimeError("Work 3 final marker timestamp lacks timezone")
    matrix = marker.get("matrix")
    if (not isinstance(matrix, dict)
            or type(matrix.get("n_cells")) is not int
            or matrix.get("n_cells") != PRODUCTION_ARTIFACT_COUNT):
        raise RuntimeError("Work 3 final marker is not the exact 35-cell matrix")
    if (not isinstance(marker.get("runs"), list)
            or len(marker["runs"]) != PRODUCTION_ARTIFACT_COUNT):
        raise RuntimeError("Work 3 final marker lacks all cell summaries")
    transaction = marker.get("transaction", {})
    provenance = transaction.get("provenance")
    validate_transaction_provenance_bytes(
        canonical_transaction_provenance_bytes(provenance), expected=provenance)
    if transaction.get("sha256") != transaction_provenance_sha256(provenance):
        raise RuntimeError("Work 3 final marker transaction hash changed")
    return marker


_GENERIC_CURRENT_FIELDS = {
    "schema", "marker_generation_version", "marker", "evaluated_at", "split",
    "config_sha256", "source_sha256", "contexts", "seeds", "run_prefix",
    "devices", "n_artifacts", "n_shots", "floor", "runs",
}
_GENERIC_CURRENT_RUN_FIELDS = {
    "run_name", "context", "seed", "n_shots", "n_pred_rows",
    "best_val_mse", "fingerprints", "per_shot_metrics_sha256",
    "m3_pred_sha256", "run_metadata_sha256",
}
_GENERIC_ARCHIVED_FIELDS = _GENERIC_CURRENT_FIELDS - {
    "schema", "marker_generation_version"}
_GENERIC_ARCHIVED_RUN_FIELDS = _GENERIC_CURRENT_RUN_FIELDS - {
    "per_shot_metrics_sha256", "m3_pred_sha256", "run_metadata_sha256"}


def _generic_marker_payload(args, split, summaries, *, floor):
    return {
        "schema": GENERIC_FINAL_MARKER_SCHEMA,
        "marker_generation_version": GENERIC_FINAL_MARKER_GENERATION_VERSION,
        "marker": MARKER_NAME,
        "evaluated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "split": {"name": split.name, "version": int(split.version),
                  "n_test_shots": len(split.test),
                  "test_shots": [int(s) for s in split.test]},
        "config_sha256": summaries[0]["fingerprints"]["config_sha256"],
        "source_sha256": summaries[0]["fingerprints"]["source_sha256"],
        "contexts": [x.label for x in args.contexts],
        "seeds": [int(s) for s in args.seeds],
        "run_prefix": str(args.run_prefix),
        "devices": [str(d) for d in args.devices],
        "n_artifacts": len(summaries),
        "n_shots": len(split.test),
        "floor": floor,
        "runs": list(summaries),
    }


def _classify_generic_marker(marker):
    if not isinstance(marker, dict):
        raise RuntimeError("generic final marker must be a mapping")
    if marker.get("schema") == GENERIC_FINAL_MARKER_SCHEMA:
        if (set(marker) != _GENERIC_CURRENT_FIELDS
                or type(marker.get("marker_generation_version")) is not int
                or marker["marker_generation_version"]
                != GENERIC_FINAL_MARKER_GENERATION_VERSION
                or marker.get("marker") != MARKER_NAME
                or not isinstance(marker.get("runs"), list)
                or any(not isinstance(run, dict)
                       or set(run) != _GENERIC_CURRENT_RUN_FIELDS
                       for run in marker["runs"])):
            raise RuntimeError(
                "current generic final marker schema/member hashes are incomplete")
        return "current"
    if (set(marker) == _GENERIC_ARCHIVED_FIELDS
            and marker.get("marker") == MARKER_NAME
            and isinstance(marker.get("runs"), list)
            and all(isinstance(run, dict)
                    and set(run) == _GENERIC_ARCHIVED_RUN_FIELDS
                    for run in marker["runs"])):
        return "archived_legacy"
    raise RuntimeError(
        "generic final marker is neither current nor an exact archived legacy shape")


def _require_publication_devices(args):
    if _manifest_mode(args) == "publication" and [
            str(device) for device in args.devices] != ["0", "1", "2", "3"]:
        raise RuntimeError(
            "publication final scoring requires exact devices 0 1 2 3")


def run_final_transaction(args, split=None):
    """Validate, score, publish the final directory, then create marker last."""
    _require_publication_invocation(args, operation="score")
    stats_root = pathlib.Path(args.stats_root)
    marker = marker_path(args)
    final_dir = final_dir_path(args)
    building = building_root(args)
    _require_production_matrix(args)
    _require_publication_devices(args)

    if _manifest_mode(args) == "publication":
        with ValidationFreezeLock(stats_root) as transaction_lock:
            transaction_lock.assert_held()
            if os.path.lexists(marker):
                raise RuntimeError(
                    f"{marker} exists: the final test was already evaluated -- "
                    "a one-shot transaction never reruns")
            if os.path.lexists(final_dir):
                raise RuntimeError(
                    f"{final_dir} exists but its marker is missing -- resolve "
                    "the half-finished transaction manually")
            readiness = _publication_final_readiness(args, split=None)
            split = readiness["split"]
            entries = [cell["entry"] for cell in readiness["cells"]]
            _prepare_publication_staging(args, entries, readiness)

            pre_workers = _publication_final_readiness(args, split=split)
            _assert_same_publication_readiness(
                readiness, pre_workers, stage="pre-worker revalidation")
            _spawn_device_workers(
                args, [str(device) for device in args.devices],
                lock_fd=transaction_lock.fd)

            after_workers = _publication_final_readiness(args, split=split)
            _assert_same_publication_readiness(
                readiness, after_workers, stage="post-worker revalidation")
            with open_final_matrix_directory(
                    building, entries,
                    require_complete=True) as final_matrix:
                summaries = _verify_staging(
                    args, entries, split, cells=after_workers["cells"],
                    transaction_provenance=readiness["transaction"],
                    transaction_sha256=readiness["transaction_sha256"],
                    require_publication_count=True,
                    tree=final_matrix.tree, matrix=final_matrix)

                pre_publish = _publication_final_readiness(args, split=split)
                _assert_same_publication_readiness(
                    readiness, pre_publish,
                    stage="final-directory publication")
                transaction_lock.assert_held()
                if os.path.lexists(final_dir) or os.path.lexists(marker):
                    raise RuntimeError(
                        "final state appeared during the transaction")
                os.rename(building, final_dir)

            summaries = _publish_marker_from_retained_final(
                args, readiness, entries, summaries, final_dir,
                transaction_lock)
            print(
                f"final test evaluated: {len(summaries)} artifacts x "
                f"{len(split.test)} shots -> {final_dir}\nmarker {marker} "
                "written last",
                flush=True,
            )
            return 0

    if split is None:
        split = _load_manifest(args.split, "generic", available_shots=None)
    if os.path.lexists(marker):
        raise RuntimeError(
            f"{marker} exists: the final test was already evaluated -- a "
            "one-shot transaction never reruns and no flag bypasses it")
    if os.path.lexists(final_dir):
        raise RuntimeError(
            f"{final_dir} exists but its marker is missing -- resolve the "
            "half-finished transaction manually; never overwrite it")
    cells = verify_final_readiness(args, split=split)
    entries = [cell["entry"] for cell in cells]
    building.mkdir(parents=True, exist_ok=True)
    devices = [str(device) for device in args.devices]
    if len(devices) == 1 and devices[0] == "cpu":
        score_entries(args, entries, "cpu", split=split)
    else:
        _spawn_device_workers(args, devices)
    summaries = _verify_staging(args, entries, split)
    stats_root.mkdir(parents=True, exist_ok=True)
    os.rename(building, final_dir)
    _atomic_json_dump(
        _generic_marker_payload(
            args, split, summaries, floor=_floor_provenance(args)),
        marker)
    print(f"final test evaluated: {len(summaries)} artifacts x "
          f"{len(split.test)} shots -> {final_dir}\nmarker {marker} "
          "written last", flush=True)
    return 0


def _verify_completed_generic(args, split, marker):
    cells = verify_final_readiness(args, split=split)
    entries = [cell["entry"] for cell in cells]
    summaries = _verify_staging(
        args, entries, split, root=final_dir_path(args))
    recorded = {run["run_name"]: run for run in marker.get("runs", ())}
    if set(recorded) != {summary["run_name"] for summary in summaries}:
        raise RuntimeError("completed generic final marker run set changed")
    for summary in summaries:
        stored = recorded[summary["run_name"]]
        for field in (
                "per_shot_metrics_sha256", "m3_pred_sha256",
                "run_metadata_sha256"):
            if stored.get(field) != summary[field]:
                raise RuntimeError(
                    f"{summary['run_name']}: completed final {field} hash changed")
    return summaries


def _validate_final_readiness_consistency(readiness):
    final = readiness["final"]
    expected = {
        "validation_selection_sha256": hashlib.sha256(
            readiness["selection_bytes"]).hexdigest()
            if "selection_bytes" in readiness
            else readiness["transaction"]["validation_selection"]["sha256"],
        "availability_audit_sha256": readiness["audit"].sha256
            if hasattr(readiness["audit"], "sha256")
            else readiness["transaction"]["availability_audit"]["sha256"],
        "work2_marker_sha256": readiness["work2"].marker_sha256
            if hasattr(readiness["work2"], "marker_sha256")
            else readiness["transaction"]["work2"]["marker_sha256"],
        "floor_sha256": readiness["work2"].floor_sha256
            if hasattr(readiness["work2"], "floor_sha256")
            else readiness["transaction"]["floor"]["sha256"],
        "transaction_sha256": transaction_provenance_sha256(
            readiness["transaction"]),
    }
    for field, value in expected.items():
        if getattr(final, field) != value:
            raise RuntimeError(f"FinalReadiness {field} is internally inconsistent")
    if readiness["transaction_sha256"] != expected["transaction_sha256"]:
        raise RuntimeError("readiness transaction_sha256 is internally inconsistent")


def _verify_completed_publication(args, split):
    marker_bytes = _read_regular_bytes(
        marker_path(args), label="Work 3 final marker")
    marker = _validate_publication_marker_bytes(marker_bytes)
    readiness = _publication_final_readiness(args, split=split)
    _validate_final_readiness_consistency(readiness)
    expected_sections = {
        "publication": dict(readiness["final"].split_disclosure),
        "validation_selection": {
            "sha256": readiness["final"].validation_selection_sha256,
            "payload": readiness["selection"],
        },
        "availability_audit": {
            "sha256": readiness["final"].availability_audit_sha256,
            "identity": readiness["audit"].identity,
        },
        "work2": {
            "marker_sha256": readiness["final"].work2_marker_sha256,
            "marker": dict(readiness["work2"].marker),
        },
        "representation_floor": {
            "path": str(pathlib.Path(args.floor_csv)),
            **dict(readiness["work2"].floor),
        },
        "references": dict(readiness["references"]),
        "transaction": {
            "sha256": readiness["transaction_sha256"],
            "provenance": readiness["transaction"],
        },
        "input_hashes": dict(readiness["transaction"]["input_hashes"]),
        "matrix": {
            "contexts": [level.label for level in CONTEXT_LEVELS],
            "seeds": list(PRODUCTION_SEEDS),
            "run_prefix": str(args.run_prefix),
            "devices": [str(device) for device in args.devices],
            "n_cells": PRODUCTION_ARTIFACT_COUNT,
            "n_shots": len(readiness["split"].test),
        },
    }
    for field, expected in expected_sections.items():
        if field not in marker:
            raise RuntimeError(
                f"completed Work 3 marker lacks readiness field {field}")
        _require_type_strict_equal(
            marker[field], expected, path=f"final_marker.{field}")
    entries = [cell["entry"] for cell in readiness["cells"]]
    final_dir = final_dir_path(args)
    with open_final_matrix_directory(
            final_dir, entries, require_complete=True) as final_matrix:
        summaries = _verify_staging(
            args, entries, readiness["split"], cells=readiness["cells"],
            transaction_provenance=readiness["transaction"],
            transaction_sha256=readiness["transaction_sha256"],
            root=final_dir, require_publication_count=True,
            tree=final_matrix.tree, matrix=final_matrix)
    _require_type_strict_equal(
        marker.get("runs"), summaries, path="final_marker.runs")
    return marker


def verify_only(args, split=None):
    """Validate readiness or revalidate every member of a completed final."""
    _require_publication_invocation(args, operation="verify")
    mode = _manifest_mode(args)
    marker = marker_path(args)
    if mode == "publication":
        _require_production_matrix(args)
        _require_publication_devices(args)
        with ValidationFreezeLock(args.stats_root) as transaction_lock:
            transaction_lock.assert_held()
            split = _load_manifest(
                args.split, "publication", available_shots=None)
            if os.path.lexists(marker):
                _verify_completed_publication(args, split)
                print(f"already evaluated and verified: {marker}")
                return 0
            if os.path.lexists(final_dir_path(args)):
                raise RuntimeError(
                    "final directory exists without its Work 3 marker")
            readiness = _publication_final_readiness(args, split=split)
            entries = [cell["entry"] for cell in readiness["cells"]]
            building = building_root(args)
            staged = 0
            if os.path.lexists(building):
                staged = _validate_incomplete_publication_staging(
                    args, readiness)
            print(
                f"{len(readiness['cells'])} artifacts validated against "
                "selection, audit, Work 2 marker/floor, and frozen references")
            if os.path.lexists(building):
                print(
                    f"staging present: {staged}/{len(entries)} valid cells "
                    f"under {building}")
            return 0

    if split is None:
        split = _load_manifest(args.split, "generic", available_shots=None)
    if os.path.lexists(marker):
        try:
            marker_payload = json.loads(_read_regular_bytes(
                marker, label="generic final marker").decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot parse generic final marker: {exc}") from exc
        classification = _classify_generic_marker(marker_payload)
        if classification == "current":
            _verify_completed_generic(args, split, marker_payload)
            print(f"already evaluated and verified: {marker}")
        else:
            print(
                f"archived generic transaction present: {marker}; exact "
                "allowlisted legacy marker has no Task 11 member hashes")
        return 0
    cells = verify_final_readiness(args, split=split)
    print(f"{len(cells)} artifacts validated (fingerprint + artifact contract)")
    building = building_root(args)
    if building.exists():
        staged = sum(_cell_complete(_cell_dir(building, cell["entry"]))
                     for cell in cells)
        print(f"staging present: {staged}/{len(cells)} cells complete under "
              f"{building}")
    return 0


def _worker_main(args, split=None):
    if _manifest_mode(args) == "publication":
        _validate_worker_authorization(args)
        _require_production_matrix(args)
        if (args.n_workers != 4 or args.worker_index not in range(4)
                or str(args.device) != str(args.worker_index)):
            raise RuntimeError(
                "publication final worker identity must be exact index/device 0..3")
    if split is None:
        split = _load_manifest(
            args.split, _manifest_mode(args), available_shots=None)
    entries = _expected_entries(args)
    partition = partition_entries(entries, args.n_workers)[args.worker_index]
    staged, resumed = score_entries(
        args, partition, args.device, split=split)
    print(f"worker {args.worker_index}/{args.n_workers} device "
          f"{args.device}: {staged} scored, {resumed} resumed", flush=True)
    return 0


def _context_arg(value):
    return context_level(str(value))


def build_parser():
    parser = argparse.ArgumentParser(
        description="One-shot final test transaction of the PF-context "
                    "matrix (hash-gated, atomic, multi-device)")
    parser.add_argument("--split", required=True,
                        help="frozen split manifest (the test-shot source)")
    parser.add_argument("--manifest-mode", choices=("publication", "generic"),
                        default="publication",
                        help="strict publication bundle validation (default) "
                             "or explicit archived generic-manifest loading")
    parser.add_argument("--config", default=str(CONFIG),
                        help="frozen sweep configuration "
                             f"(default: {CONFIG})")
    parser.add_argument("--target-dir", default=str(TARGET_DIR),
                        help="target dataset (default: ProjDB/datasets/"
                             "NpzGeom)")
    parser.add_argument("--sidecar-dir", default=str(SIDECAR_DIR),
                        help="PF ref/actual sidecar (default: ProjDB/"
                             "datasets/NpzGeomPFObs)")
    parser.add_argument("--out-root", default=str(OUT_ROOT),
                        help="validation-artifact root (default: "
                             "ProjDB/trains)")
    parser.add_argument("--stats-root", default=str(STATS_ROOT),
                        help="stats root holding final_test and the marker "
                             f"(default: {STATS_ROOT})")
    parser.add_argument(
        "--work2-marker", default=str(DEFAULT_WORK2_MARKER),
        help="canonical Work 2 FINAL_TEST_EVALUATED.json dependency")
    parser.add_argument("--floor-csv", default=str(DEFAULT_FLOOR_CSV),
                        help="Work 2 representation-floor table, validated "
                             "against its final marker and never subtracted")
    parser.add_argument(
        "--references", default=str(DEFAULT_REFERENCES),
        help="external timescale records; final readiness requires the "
             "post-selection frozen_before_final_test=true transition")
    parser.add_argument(
        "--audit-identity", default=str(DEFAULT_AUDIT_IDENTITY),
        help="canonical Work 3 context_availability.audit.json")
    parser.add_argument("--contexts", nargs="+", type=_context_arg,
                        default=list(CONTEXT_LEVELS),
                        help="context levels to score (default: the frozen "
                             "seven-level grid)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(PUBLICATION_SEEDS),
                        help="seeds to score (default: the frozen 0 1 2 3 4)")
    parser.add_argument("--run-prefix", default="pfctx",
                        help="run_name prefix (default: pfctx -> "
                             "pfctx_h0512_s0)")
    parser.add_argument("--devices", nargs="+", default=list(DEFAULT_DEVICES),
                        help="one scoring worker process per device "
                             "(default: 0 1 2 3; 'cpu' scores in the "
                             "parent)")
    parser.add_argument("--verify-only", action="store_true",
                        help="readiness checks only: no inference module, "
                             "no workers, no test-target access")
    parser.add_argument(WORKER_FLAG, action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--device", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n-workers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    operation = "worker" if args.score_partition else (
        "verify" if args.verify_only else "score")
    _require_publication_invocation(args, operation=operation)
    if args.score_partition:
        if args.worker_index is None or args.n_workers is None:
            raise SystemExit(f"{WORKER_FLAG} requires --worker-index and "
                             "--n-workers")
        return _worker_main(args, split=None)
    split = _load_manifest(
        args.split, _manifest_mode(args), available_shots=None)
    if args.verify_only:
        return verify_only(args, split=split)
    return run_final_transaction(args, split=split)


if __name__ == "__main__":
    sys.exit(main())
