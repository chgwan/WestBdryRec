# -*- coding: utf-8 -*-
"""Build the native-time PF reference/actual observability sidecar (NpzGeomPFObs).

For every NpzGeom shot this writes ``NpzGeomPFObs/<shot>.npz`` holding the ten
PF coil scopes' ``dcs/<scope>/ref`` and ``dcs/<scope>/actual`` traces plus
``dcs/Ip_scope/ref``, each on the shot's OWN native NpzGeom time axis -- the
same grid ``Y``/``center`` live on, so sidecar rows pair one-to-one with the
trainable target without any resampling.

Alignment is exact-span matching and nothing else: the NpzGeom ``time`` vector
must equal a contiguous slice of ``MergedH5Gmag/<shot>.h5``'s ``time`` to
within a representation-aware tolerance (base ``ALIGN_ATOL_S``, widened to
``ALIGN_ATOL_RULE`` at late times so float32 storage drift -- never grid
drift -- still matches), and the sidecar copies that span verbatim. No
interpolation, no nearest-neighbour repair, no time-base flag, no fallback
grid -- a shot whose native axis is not an exact span fails the build.

``common_valid`` marks the rows usable for ref-vs-actual observability: the
NpzGeom ``valid`` mask, a finite target block (``Y`` and ``center``), t >= 0,
and finite ref/actual/Ip on every channel. It is an intersection, never a
repair: rows missing any required signal stay False.

``run`` refuses source/output identity, builds into ``NpzGeomPFObs.building``
via :func:`src.utils.pmap`, fails the entire build on any failed shot (leaving
no output at all), and renames the staging dir into place only after complete
success. It audits the result with an exact-keys ``meta.json`` and a
per-shot/channel finite-count ``coverage.csv`` (with the per-shot target-valid
denominator ``n_valid``) under ``ProjDB/Stats/pf_observability``.
:func:`check_coverage_gates` turns that split-free audit into the spec-5.3
preflight gates against a frozen split -- it reads no npz. Distributions,
correlations and residual quantiles are deliberately NOT computed here; those
need a train/test split and belong to the observability analysis task.

Importable as a package module; CLI::

    python -m src.data.build_pf_observability --workers 64
"""
import csv
import hashlib
import json
import pathlib
import shutil

import h5py
import numpy as np

from ..proj_config import get_proj_config
from ..utils import pmap

# The ten WEST PF coil scopes, in the sidecar's fixed column order.
PF_SCOPES = (
    "IBb_scope", "IDb_scope", "IEb_scope", "IFb_scope", "IFh_scope",
    "IEh_scope", "IDh_scope", "IBh_scope", "IXb_scope", "IXh_scope",
)
# NpzGeom time is float32 on the same clock as MergedH5Gmag's float64. The
# base tolerance absorbs small-t rounding; at late times (t ~ 20-30 s)
# float32 STORAGE drift reaches a couple of ULPs (~4e-6 s) on step-for-step
# grids, so the enforced tolerance is representation-aware:
# max(ALIGN_ATOL_S, 4 float32 ULPs at the native axis's own magnitude).
# The native step is 2048 us -- the wrong-span discrimination margin stays
# ~137-500x. Ruled 2026-08-20 after the Task-3 audit measured exactly this
# drift on 36 real train+validation shots.
ALIGN_ATOL_S = 2e-6
ALIGN_ATOL_RULE = f"max({ALIGN_ATOL_S:g}, 4 float32 ULPs)"

# Spec 5.3 preflight coverage gate: the fraction of NpzGeom TARGET-VALID
# slices (the ``valid`` flag AND a finite 34-column target) that stay
# common-valid must reach this floor overall AND within every split.
COVERAGE_GATE = 0.95


def find_native_span(merged_time, npz_time, atol=ALIGN_ATOL_S):
    """Slice of ``merged_time`` that ``npz_time`` equals exactly, or raise.

    Matching is exact-contiguous only: anchored at the merged sample closest
    to the first native sample, then ``np.allclose`` over the whole span at
    ``rtol=0``. The absolute tolerance is representation-aware: the native
    axis is stored float32 against MergedH5Gmag's float64, so the enforced
    floor is ``max(atol, 4 * float32 ULP at |nt|.max())`` -- pure storage
    rounding at the axis's own magnitude is matched, while any grid error
    (a skipped or doubled sample, a 2048 us step) remains orders of
    magnitude beyond it. Deliberately no interpolation, nearest-neighbour
    repair, time-base flag or fallback grid.
    """
    mt = np.asarray(merged_time, float).reshape(-1)
    nt = np.asarray(npz_time, float).reshape(-1)
    if nt.size == 0 or mt.size < nt.size:
        raise ValueError("native time axis is empty or longer than MergedH5Gmag")
    i0 = int(np.argmin(np.abs(mt - nt[0])))
    i1 = i0 + nt.size
    atol_eff = max(atol, 4 * float(np.spacing(np.float32(np.abs(nt).max()))))
    if i1 > mt.size or not np.allclose(
            mt[i0:i1], nt, rtol=0.0, atol=atol_eff):
        raise ValueError(
            "NpzGeom native time axis has no exact MergedH5Gmag span")
    return slice(i0, i1)


def build_one(source_npz, merged_h5, output_npz):
    """Write one sidecar ``<shot>.npz`` and return its audit info.

    Reads the NpzGeom source (``time``, ``valid``, ``Y``, ``center``) and the
    MergedH5Gmag file (``time``, ``dcs/<scope>/ref|actual``, ``dcs/Ip_scope/
    ref``); every path and length is validated inside the open-file block, so
    a failed shot raises before any output is written. The sidecar arrays are
    ``time``, ``pf_ref`` (nt, 10), ``pf_actual`` (nt, 10), ``ip_ref`` (nt, 1)
    and ``common_valid`` (nt,), all on the shot's native axis.
    """
    with np.load(source_npz) as src, h5py.File(merged_h5, "r") as hf:
        time = np.asarray(src["time"], np.float32)
        span = find_native_span(hf["time"][:], time)
        pf_ref = np.column_stack([
            np.asarray(hf[f"dcs/{node}/ref"][span], float)
            for node in PF_SCOPES
        ]).astype(np.float32)
        pf_actual = np.column_stack([
            np.asarray(hf[f"dcs/{node}/actual"][span], float)
            for node in PF_SCOPES
        ]).astype(np.float32)
        ip_ref = np.asarray(
            hf["dcs/Ip_scope/ref"][span], float
        ).reshape(-1, 1).astype(np.float32)
        target_finite = (
            np.isfinite(src["Y"]).all(1)
            & np.isfinite(src["center"]).all(1)
        )
        target_valid = src["valid"].astype(bool) & target_finite
        common = (
            target_valid
            & (time >= 0)
            & np.isfinite(pf_ref).all(1)
            & np.isfinite(pf_actual).all(1)
            & np.isfinite(ip_ref).all(1)
        )
    np.savez(output_npz, time=time, pf_ref=pf_ref,
             pf_actual=pf_actual, ip_ref=ip_ref,
             common_valid=common)
    return {
        "shot": int(pathlib.Path(source_npz).stem),
        "n_rows": int(time.size),
        "n_valid": int(target_valid.sum()),
        "n_common": int(common.sum()),
        "common_valid": common,
    }


def _build_shot(job):
    """Worker for :func:`run`: sidecar one shot, never raise across pmap.

    Returns ``{shot, ok, n_rows, n_valid, n_common, per-channel finite
    counts}`` on success or ``{shot, ok=False, error}`` on failure (build_one
    leaves no partial output; the unlink is belt-and-braces for audit-time
    failures).
    """
    shot, source, merged, out = job
    out = pathlib.Path(out)
    try:
        info = build_one(source, merged, out)
        with np.load(out) as d:
            counts = {
                "pf_ref_finite": np.isfinite(d["pf_ref"]).sum(0).tolist(),
                "pf_actual_finite": np.isfinite(d["pf_actual"]).sum(0).tolist(),
                "ip_ref_finite": int(np.isfinite(d["ip_ref"]).sum()),
            }
        return {"shot": shot, "ok": True,
                "n_rows": info["n_rows"], "n_valid": info["n_valid"],
                "n_common": info["n_common"], **counts}
    except Exception as exc:  # noqa: BLE001
        if out.exists():
            out.unlink()
        return {"shot": shot, "ok": False, "error": str(exc)}


def _write_coverage_csv(path, rows):
    """Per-shot/channel finite counts plus the per-shot target-valid
    denominator -- no distributions, correlations, quantiles or ranges
    (those belong to the post-split analysis task)."""
    header = (["shot", "n_rows", "n_common", "n_valid"]
              + [f"ref_{s}" for s in PF_SCOPES]
              + [f"actual_{s}" for s in PF_SCOPES]
              + ["ip_ref"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in sorted(rows, key=lambda r: r["shot"]):
            w.writerow([r["shot"], r["n_rows"], r["n_common"], r["n_valid"]]
                       + r["pf_ref_finite"] + r["pf_actual_finite"]
                       + [r["ip_ref_finite"]])


def check_coverage_gates(coverage_csv, split, threshold=COVERAGE_GATE):
    """Assert the spec-5.3 preflight coverage gates; read ONLY the audit.

    ``coverage.csv`` is split-free (one row per NpzGeom shot, written by
    :func:`run`), so joining it with a frozen split opens no npz at all --
    no held-back target file is touched. Gates: every shot of the split's
    held-out list keeps at least one common-valid row (100% scoreable), and
    ``n_common / n_valid`` -- the spec's TARGET-VALID denominator -- reaches
    ``threshold`` within each split AND overall. ``n_common / n_rows`` (all
    native rows, the builder's global figure) is reported alongside but is
    not the gate. Returns ``{part: {n_shots, n_rows, n_valid, n_common,
    coverage_rows, coverage_valid}}`` over train/validation/test/overall.
    """
    path = pathlib.Path(coverage_csv)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = {int(r["shot"]): r for r in reader}
        stale = [c for c in ("n_rows", "n_common", "n_valid")
                 if reader.fieldnames is None or c not in reader.fieldnames]
    if stale:
        raise RuntimeError(
            f"{path}: the coverage audit lacks column(s) {stale} -- rebuild "
            "the sidecar with the current builder before gating on it")
    parts = {"train": split.train, "validation": split.validation,
             "test": split.test,
             "overall": split.train + split.validation + split.test}
    report = {}
    for part, shots in parts.items():
        absent = sorted(int(s) for s in shots if int(s) not in rows)
        if absent:
            detail = ", ".join(str(s) for s in absent[:5])
            if len(absent) > 5:
                detail += f" (+{len(absent) - 5} more)"
            raise RuntimeError(
                f"{path}: split {split.name} {part} shots absent from the "
                f"coverage audit: {detail}")
        n_rows = sum(int(rows[int(s)]["n_rows"]) for s in shots)
        n_valid = sum(int(rows[int(s)]["n_valid"]) for s in shots)
        n_common = sum(int(rows[int(s)]["n_common"]) for s in shots)
        report[part] = {
            "n_shots": len(shots), "n_rows": n_rows, "n_valid": n_valid,
            "n_common": n_common,
            "coverage_rows": (n_common / n_rows) if n_rows else 0.0,
            "coverage_valid": (n_common / n_valid) if n_valid else 0.0}
    unscoreable = [int(s) for s in split.test
                   if int(rows[int(s)]["n_common"]) < 1]
    if unscoreable:
        detail = ", ".join(str(s) for s in unscoreable[:5])
        if len(unscoreable) > 5:
            detail += f" (+{len(unscoreable) - 5} more)"
        raise RuntimeError(
            f"coverage gate: held-out shots with zero common-valid rows are "
            f"never scoreable: {detail}")
    below = [p for p in ("train", "validation", "test", "overall")
             if report[p]["coverage_valid"] < threshold]
    if below:
        detail = ", ".join(
            f"{p} {report[p]['coverage_valid']:.4f}" for p in below)
        raise RuntimeError(
            f"coverage gate: n_common/n_valid below {threshold:.0%} in "
            f"{detail} -- stop before training and diagnose source "
            "availability (spec 5.3)")
    return report


def run(npz_dir=None, merged_dir=None, out_dir=None, workers=1):
    """Build the whole NpzGeomPFObs sidecar dataset and audit it.

    Processes every shot listed in ``<npz_dir>/meta.json`` (default: the real
    NpzGeom dataset) against ``<merged_dir>/<shot>.h5`` (default: the real
    MergedH5Gmag). Refuses source/output identity; builds into
    ``<out_dir>.building``; on ANY failed shot the entire build fails (staging
    removed, nothing renamed); only after complete success is the staging dir
    renamed into place and the audit written (``meta.json`` inside the dataset
    plus ``coverage.csv`` under ``pfobs_stats_dir``). Returns the build
    summary dict.
    """
    cfg = get_proj_config()
    npz_dir = pathlib.Path(npz_dir) if npz_dir else cfg.npzgeom_dir
    merged_dir = pathlib.Path(merged_dir) if merged_dir else cfg.mergedh5_gmag_dir
    out_dir = pathlib.Path(out_dir) if out_dir else cfg.pfobs_dir
    stats_dir = cfg.pfobs_stats_dir

    for src in (npz_dir, merged_dir):
        if out_dir.resolve() == src.resolve():
            raise ValueError(
                f"output directory {out_dir} must differ from source {src}")

    src_meta = npz_dir / "meta.json"
    shots = sorted(int(s["shot"])
                   for s in json.loads(src_meta.read_text())["shots"])

    staging = out_dir.parent / (out_dir.name + ".building")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    jobs = [(shot, npz_dir / f"{shot}.npz", merged_dir / f"{shot}.h5",
             staging / f"{shot}.npz") for shot in shots]
    results = pmap(_build_shot, jobs, workers, "build_pf_observability")

    failed = [(r["shot"], r["error"]) for r in results if not r["ok"]]
    if failed:
        shutil.rmtree(staging)
        detail = "; ".join(f"{s}: {e}" for s, e in failed[:5])
        raise RuntimeError(
            f"{len(failed)}/{len(shots)} shots failed: {detail}"
            + (f" (+{len(failed) - 5} more)" if len(failed) > 5 else ""))

    n_rows = int(sum(r["n_rows"] for r in results))
    n_common = int(sum(r["n_common"] for r in results))
    coverage = n_common / n_rows if n_rows else 0.0
    meta = {
        # experiment-wide contract identifier: configs/dcs_pf_observability.yml,
        # the split manifests and the NSCC post-build check all assert this token
        "time_axis": "native_gmag_bnd",
        "n_shots": len(results),
        "n_rows": n_rows,
        "n_common": n_common,
        "coverage_fraction": coverage,
        "source_npz_dir": str(npz_dir),
        "source_merged_dir": str(merged_dir),
        "pf_scopes": list(PF_SCOPES),
        "align_atol_s": ALIGN_ATOL_S,
        "align_atol_rule": ALIGN_ATOL_RULE,
        "npzgeom_meta_sha256": hashlib.sha256(
            src_meta.read_bytes()).hexdigest(),
    }
    with open(staging / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)

    stats_dir.mkdir(parents=True, exist_ok=True)
    coverage_csv = stats_dir / "coverage.csv"
    _write_coverage_csv(coverage_csv, results)

    print(f"NpzGeomPFObs: {len(results)} shots, {n_common}/{n_rows} rows "
          f"common (coverage {coverage:.4f}) -> {out_dir}")
    return {"n_shots": len(results), "n_rows": n_rows, "n_common": n_common,
            "coverage_fraction": coverage, "out_dir": str(out_dir),
            "coverage_csv": str(coverage_csv)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(workers=args.workers)


if __name__ == "__main__":
    main()
