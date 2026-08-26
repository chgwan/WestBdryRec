# -*- coding: utf-8 -*-
"""Four-arm PF reference/actual input assembly for the observability study.

Each arm replaces the 18 strict actuator columns with a fixed 21-column input
assembled from the Task-1 native-time sidecar (NpzGeomPFObs): two 10-wide PF
blocks -- one of them zero-padded per arm -- plus the Ip reference column::

    A = [ref,      0,      Ip]   programmed only (blind feedforward)
    B = [0,        actual, Ip]   measured only
    C = [ref,      actual, Ip]   both, concatenated
    D = [ref, actual - ref, Ip]  both, residual form (invertibly equivalent to C)

``PFObsSeriesReader`` is a drop-in ``series_reader`` for ``DCSWindowDataset``:
it hands ``(A, common_valid)`` to the windowing unchanged (``cfg``/``ncm`` are
ignored -- the sidecar is self-describing) and refuses any shot whose sidecar
and target time axes are not byte-for-byte equal, so an arm row can never pair
with the wrong target row.
"""
import dataclasses
import hashlib
import json
import pathlib

import numpy as np

ARMS = ("A", "B", "C", "D")

# Experiment-wide contract token: the Task-1 sidecar meta.json, every split
# manifest and the NSCC gates all assert exactly this string.
TIME_AXIS_TOKEN = "native_gmag_bnd"

# The worktree/project root that relative manifest paths (shot_metadata,
# slice_strata_dir) resolve against -- derived from this file, never from
# the caller's current directory.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Modules whose bytes pin a run: the fixed model, positional encoding,
# dataset reader, this arm-assembly module, the PF-observability
# training/inference modules, the native metrics module and the matrix
# runner (the last four are created by later tasks of this plan). Every
# entry must exist when a fingerprint is computed -- a missing entry is a
# loud failure, not a silently narrower hash.
SOURCE_FILES = (
    "src/ml/models.py",
    "src/ml/pos_encoding.py",
    "src/ml/dataset.py",
    "src/ml/pf_observability.py",
    "src/ml/pfobs_train.py",
    "src/ml/pfobs_infer.py",
    "src/ml/native_metrics.py",
    "scripts/run_pf_observability.py",
)


def assemble_arm(pf_ref, pf_actual, ip_ref, arm):
    ref = np.asarray(pf_ref, np.float32)
    actual = np.asarray(pf_actual, np.float32)
    ip = np.asarray(ip_ref, np.float32).reshape(-1, 1)
    if ref.shape != actual.shape or ref.ndim != 2 or ref.shape[1] != 10:
        raise ValueError(
            "PF reference and actual arrays must both have shape (nt, 10)")
    if ip.shape[0] != ref.shape[0]:
        raise ValueError("Ip_ref length does not match PF arrays")
    zero = np.zeros_like(ref)
    blocks = {
        "A": (ref, zero), "B": (zero, actual),
        "C": (ref, actual), "D": (ref, actual - ref),
    }
    try:
        left, right = blocks[arm.upper()]
    except KeyError as exc:
        raise ValueError(
            f"arm must be one of {ARMS}, got {arm!r}") from exc
    return np.column_stack([left, right, ip]).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class PFObsSeriesReader:
    sidecar_dir: pathlib.Path
    arm: str

    def __call__(self, target_npz, _cfg, _ncm):
        shot = pathlib.Path(target_npz).stem
        with np.load(self.sidecar_dir / f"{shot}.npz") as d:
            A = assemble_arm(
                d["pf_ref"], d["pf_actual"], d["ip_ref"], self.arm)
            mask = d["common_valid"].astype(bool)
            side_time = d["time"].copy()
        with np.load(target_npz) as target:
            target_time = target["time"]
        if not np.array_equal(side_time, target_time):
            raise ValueError(
                f"shot {shot}: sidecar/target native time mismatch")
        return A, mask


def series_mean_std(npz_dir, shots, reader):
    rows = []
    for shot in shots:
        p = pathlib.Path(npz_dir) / f"{int(shot)}.npz"
        A, mask = reader(p, {}, {})
        if mask.any():
            rows.append(A[mask])
    if not rows:
        raise ValueError("no common-valid training rows")
    X = np.concatenate(rows)
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


# ── frozen splits and run fingerprints (Task 3) ──────────────────────
def sha256_file(path):
    """Streaming sha256 hex digest of one file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root, include=None):
    """sha256 over the sorted mapping of normalized project-relative paths
    to per-file digests.

    ``include=None`` hashes every file under ``root`` recursively;
    otherwise exactly the listed project-relative entries (duplicates
    collapse, order is irrelevant). Every included entry must exist and be
    a file -- a missing entry raises instead of narrowing the hash. The
    digest therefore moves when any covered file is added, removed, renamed
    or changed.
    """
    root = pathlib.Path(root)
    if include is None:
        rels = sorted({p.relative_to(root).as_posix()
                       for p in root.rglob("*") if p.is_file()})
    else:
        rels = sorted({pathlib.Path(p).as_posix() for p in include})
    missing = [r for r in rels if not (root / r).is_file()]
    if missing:
        raise FileNotFoundError(
            f"sha256_tree: missing files under {root}: {missing}")
    lines = [f"{r}:{sha256_file(root / r)}" for r in rels]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class FrozenSplit:
    name: str
    version: int
    time_axis: str
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    shot_metadata: pathlib.Path | None
    slice_strata_dir: pathlib.Path | None


def load_split(path, available_shots=None, project_root=PROJECT_ROOT):
    """Load and validate a frozen split manifest.

    Requires the native-time contract token, duplicate-free and
    pairwise-disjoint train/validation/test lists, and -- when
    ``available_shots`` is given (the shots present in both the target and
    feature datasets) -- that every listed shot is available. Relative
    ``shot_metadata``/``slice_strata_dir`` paths resolve against
    ``project_root``, never against the caller's current directory. The
    manifest JSON may carry extra keys (e.g. ``claim_scope``) that gates
    read from the raw JSON; they are not part of ``FrozenSplit``.
    """
    path = pathlib.Path(path)
    data = json.loads(path.read_text())
    if data.get("time_axis") != TIME_AXIS_TOKEN:
        raise ValueError(
            f"split {path}: time_axis must be {TIME_AXIS_TOKEN!r}, "
            f"got {data.get('time_axis')!r}")
    lists = {}
    for part in ("train", "validation", "test"):
        shots = tuple(int(s) for s in data[part])
        if len(set(shots)) != len(shots):
            raise ValueError(
                f"split {path}: {part} contains duplicate shots")
        lists[part] = shots
    for a, b in (("train", "validation"), ("train", "test"),
                 ("validation", "test")):
        shared = set(lists[a]) & set(lists[b])
        if shared:
            raise ValueError(
                f"split {path}: {a} and {b} must be disjoint "
                f"(share {sorted(shared)[:5]})")
    if available_shots is not None:
        missing = sorted({s for shots in lists.values() for s in shots
                          if s not in available_shots})
        if missing:
            detail = ", ".join(str(s) for s in missing[:5])
            if len(missing) > 5:
                detail += f" (+{len(missing) - 5} more)"
            raise ValueError(
                f"split {path}: shots not available in both datasets: "
                f"{detail}")

    def _resolve(p):
        if p in (None, ""):
            return None
        p = pathlib.Path(p)
        return p if p.is_absolute() else pathlib.Path(project_root) / p

    return FrozenSplit(
        name=str(data["name"]),
        version=int(data["version"]),
        time_axis=data["time_axis"],
        shot_metadata=_resolve(data.get("shot_metadata")),
        slice_strata_dir=_resolve(data.get("slice_strata_dir")),
        **lists)


def run_fingerprint(arm, seed, config_path, split_path, sidecar_dir,
                    npz_dir, project_root=PROJECT_ROOT):
    """The pinned provenance mapping stored in every training artifact.

    Hashes everything a run depends on: the fixed base-model contract
    values, the arm/seed, the config and split files, the sidecar and
    target dataset metas, the optional shot-metadata file and slice-strata
    tree, and the bytes of every :data:`SOURCE_FILES` module. Any change
    to any pinned input moves at least one field.
    """
    split = load_split(split_path, project_root=project_root)
    return {
        "model": "ActSeqAttn",
        "pe": "rope_time",
        "time_axis": "native_gmag_bnd",
        "arm": arm,
        "seed": int(seed),
        "config_sha256": sha256_file(config_path),
        "split_sha256": sha256_file(split_path),
        "sidecar_meta_sha256": sha256_file(
            pathlib.Path(sidecar_dir) / "meta.json"),
        "target_meta_sha256": sha256_file(
            pathlib.Path(npz_dir) / "meta.json"),
        "shot_metadata_sha256": (
            sha256_file(split.shot_metadata)
            if split.shot_metadata is not None else "absent"),
        "slice_strata_sha256": (
            sha256_tree(split.slice_strata_dir)
            if split.slice_strata_dir is not None else "absent"),
        "source_sha256": sha256_tree(project_root, include=SOURCE_FILES),
    }
