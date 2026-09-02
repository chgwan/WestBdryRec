# -*- coding: utf-8 -*-
"""Train+validation-only PF reference/actual signal audit (source level).

Loads exactly ``split.train + split.validation`` -- never ``split.test`` --
from the NpzGeom targets and their MergedH5Gmag sources and writes one CSV
row per PF scope to ``ProjDB/Stats/pf_observability/source_audit.csv``:

- finite coverage: ``ref_finite_frac``/``act_finite_frac`` over ALL native
  rows, before any masking;
- robust ranges/variance: min/p01/p50/p99/max, population std and the
  sigma-scaled MAD of reference and actual over common-valid rows (the rows
  the experiment trains on);
- residuals (actual - ref): RMS and p01/p50/p99 over common rows;
- correlation: pooled Pearson r at lag 0 plus ``lag_samples``, the lag in
  native samples maximizing the pair-count-weighted mean per-shot
  correlation corr(ref[t], actual[t+lag]) over +-``LAG_MAX`` samples
  (positive = actual trails ref; ties resolve to the smallest |lag|);
- identical fraction (exact equality), near-identical fraction
  (|actual-ref| <= ``NEAR_TOL``) and same-sign fraction;
- constant/saturation: ``constant_frac`` counts common rows inside a
  within-shot run of >= ``FLAT_RUN_MIN`` identical consecutive values in
  either signal; ``saturation_frac`` counts such rows pinned exactly at
  that signal's pooled min or max; ``ref_constant``/``act_constant`` flag a
  degenerate pooled robust range (p01 == p99).

Per-shot signals and the common mask come from the Task-1 builder itself
(:func:`src.data.build_pf_observability.build_one` into an in-memory
sidecar), so the audit can never drift from the build semantics. A shot
whose native axis has no exact MergedH5Gmag span fails alignment and is
REPORTED as unauditable (spec 5.2: a data error, never repaired by
nearest-neighbour matching or interpolation; spec 5.3: diagnose source
availability before training) -- the pooled rows cover the auditable
shots, the summary and stdout carry the full accounting, and the Task-1
sidecar build stays the enforcement point.

If ``reconstruction_provenance.json`` is absent, the conservative template
below is written verbatim (status unknown, no evidence, claim limited to
observability of the GMAG_BND reconstruction product -- never independent
observation of the physical boundary). An existing provenance file is
never overwritten.

Usage:
  python scripts/audit_pf_observability.py \
    --split configs/splits/pfobs_random_pilot.json --manifest-mode generic
"""
import argparse
import csv
import io
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import h5py  # noqa: E402  (opened only inside build_one)
import numpy as np  # noqa: E402

from src.data.build_pf_observability import PF_SCOPES, build_one  # noqa: E402
from src.ml.pfobs_provenance import (  # noqa: E402
    ValidationFreezeLock, build_source_audit_identity,
)
from src.ml.publication_split import (  # noqa: E402
    PUBLICATION_MANIFEST_PATH,
    PUBLICATION_SIDECAR_DIR,
    PUBLICATION_TARGET_DIR,
    PUBLICATION_WORK2_AUDIT_IDENTITY,
    PUBLICATION_WORK2_MARKER,
    PUBLICATION_WORK2_STATS_ROOT,
    load_split_for_mode,
    require_publication_paths,
)
from src.proj_config import get_proj_config  # noqa: E402
from src.utils import (  # noqa: E402
    PublicationIOError, durable_publish_bytes, pmap, read_regular_nofollow,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NPZ_DIR = PUBLICATION_TARGET_DIR
MERGED_DIR = REPO_ROOT / "ProjDB/datasets/MergedH5Gmag"
SIDECAR_DIR = PUBLICATION_SIDECAR_DIR
STATS_ROOT = PUBLICATION_WORK2_STATS_ROOT
AUDIT_IDENTITY = PUBLICATION_WORK2_AUDIT_IDENTITY
SPLIT = PUBLICATION_MANIFEST_PATH
FINAL_MARKER = PUBLICATION_WORK2_MARKER

LAG_MAX = 64        # cross-correlation search window, native samples (~131 ms)
FLAT_RUN_MIN = 8    # shortest run of identical consecutive values, samples
NEAR_TOL = 1e-6     # |actual - ref| counted as near-identical

PROVENANCE_TEMPLATE = {
    "status": "unknown",
    "evidence": [],
    "claim_limit": "observability_of_the_GMAG_BND_reconstruction_product",
}
IDENTITY_NAME = "source_audit_identity.json"


def _load_manifest(path, manifest_mode, available_shots=None):
    """Load the requested split contract once before the source audit."""
    return load_split_for_mode(
        path,
        manifest_mode=manifest_mode,
        available_shots=available_shots,
        project_root=REPO_ROOT,
    )


CSV_COLUMNS = (
    "scope", "n_shots", "n_rows", "n_common",
    "ref_finite_frac", "act_finite_frac",
    "ref_min", "ref_p01", "ref_p50", "ref_p99", "ref_max",
    "ref_std", "ref_mad",
    "act_min", "act_p01", "act_p50", "act_p99", "act_max",
    "act_std", "act_mad",
    "resid_rms", "resid_p01", "resid_p50", "resid_p99",
    "corr", "lag_samples", "same_sign_frac",
    "identical_frac", "near_identical_frac",
    "constant_frac", "saturation_frac", "ref_constant", "act_constant",
)


def _pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sx, sy = x.std(), y.std()
    if x.size < 2 or sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _flat_run_rows(vals, min_run=FLAT_RUN_MIN):
    """Mask of rows inside a maximal run of >= ``min_run`` equal consecutive
    values (NaN breaks a run: NaN != NaN)."""
    n = vals.size
    if n == 0:
        return np.zeros(0, bool)
    change = np.flatnonzero(vals[1:] != vals[:-1])
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change, [n - 1]))          # inclusive
    lengths = ends - starts + 1
    out = np.zeros(n, bool)
    for s0, e0 in zip(starts[lengths >= min_run], ends[lengths >= min_run]):
        out[s0:e0 + 1] = True
    return out


def _audit_shot(job):
    """Worker: one shot's native-axis PF signals + common mask, or the error.

    Delegates to the Task-1 builder (sidecar written into an in-memory
    buffer, never to disk) so alignment and the common mask are exactly the
    build's semantics. Never raises across :func:`pmap`.
    """
    shot, npz_path, h5_path = job
    try:
        buf = io.BytesIO()
        build_one(npz_path, h5_path, buf)
        buf.seek(0)
        with np.load(buf) as d:
            return {"shot": shot, "ok": True,
                    "n_rows": int(d["time"].size),
                    "mask": d["common_valid"].astype(bool),
                    "ref": d["pf_ref"].astype(np.float32),
                    "act": d["pf_actual"].astype(np.float32)}
    except Exception as exc:  # noqa: BLE001
        return {"shot": shot, "ok": False, "error": str(exc)}


def _lag_stats(ref_col, act_col, mask, max_lag=LAG_MAX):
    """(lag_samples, weighted-mean correlation profile) per channel.

    Per shot the correlation corr(ref[t], actual[t+lag]) is computed over
    native-row pairs where both rows are common-valid (the mask may have
    holes, so shifted pairs are matched on the native grid, never on the
    compressed common rows); shots are combined weighted by pair count.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    corr_sum = np.zeros(lags.size)
    weight = np.zeros(lags.size)
    n0 = 0
    for m in mask:
        n = m.size
        rv, av = ref_col[n0:n0 + n], act_col[n0:n0 + n]
        for i, lag in enumerate(lags):
            t = np.arange(0, n - lag) if lag >= 0 else np.arange(-lag, n)
            pair = m[t] & m[t + lag]
            if pair.sum() < 2:
                continue
            c = _pearson(rv[t[pair]], av[(t + lag)[pair]])
            if np.isfinite(c):
                corr_sum[i] += c * pair.sum()
                weight[i] += pair.sum()
        n0 += n
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = corr_sum / weight
    best = None
    for i in np.argsort(np.abs(lags), kind="stable"):   # prefer smallest |lag|
        if weight[i] > 0 and (best is None or profile[i] > profile[best]):
            best = i
    return (int(lags[best]) if best is not None else 0), profile


def _fmt(v):
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.10g}"


def compute_rows(results):
    """The per-scope CSV rows pooled over the audited shots."""
    n_shots = len(results)
    n_rows = int(sum(r["n_rows"] for r in results))
    ref = np.concatenate([r["ref"] for r in results])
    act = np.concatenate([r["act"] for r in results])
    mask = np.concatenate([r["mask"] for r in results])
    if not mask.any():
        raise ValueError("no common-valid rows in the audited shots")
    n_common = int(mask.sum())

    rows = []
    lengths = [r_["n_rows"] for r_ in results]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    shot_masks = [r_["mask"] for r_ in results]
    for k, scope in enumerate(PF_SCOPES):
        rn, an = ref[:, k], act[:, k]
        r = rn[mask].astype(float)
        a = an[mask].astype(float)
        resid = a - r
        rq = np.quantile(r, [0.01, 0.5, 0.99])
        aq = np.quantile(a, [0.01, 0.5, 0.99])
        resq = np.quantile(resid, [0.01, 0.5, 0.99])
        r_mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        a_mad = 1.4826 * np.median(np.abs(a - np.median(a)))

        # flat runs per shot on the native grid (runs never cross shots)
        flat_r = np.concatenate([
            _flat_run_rows(rn[s:s + n]) for s, n in zip(offsets, lengths)])
        flat_a = np.concatenate([
            _flat_run_rows(an[s:s + n]) for s, n in zip(offsets, lengths)])
        sat_r = flat_r & ((rn == r.min()) | (rn == r.max()))
        sat_a = flat_a & ((an == a.min()) | (an == a.max()))
        constant_rows = (flat_r | flat_a) & mask
        saturated_rows = (sat_r | sat_a) & mask

        lag, _profile = _lag_stats(rn, an, shot_masks)
        row = {
            "scope": scope, "n_shots": n_shots, "n_rows": n_rows,
            "n_common": n_common,
            "ref_finite_frac": float(np.isfinite(rn).mean()),
            "act_finite_frac": float(np.isfinite(an).mean()),
            "ref_min": r.min(), "ref_p01": rq[0], "ref_p50": rq[1],
            "ref_p99": rq[2], "ref_max": r.max(), "ref_std": r.std(),
            "ref_mad": float(r_mad),
            "act_min": a.min(), "act_p01": aq[0], "act_p50": aq[1],
            "act_p99": aq[2], "act_max": a.max(), "act_std": a.std(),
            "act_mad": float(a_mad),
            "resid_rms": float(np.sqrt((resid ** 2).mean())),
            "resid_p01": resq[0], "resid_p50": resq[1], "resid_p99": resq[2],
            "corr": _pearson(r, a), "lag_samples": lag,
            "same_sign_frac": float((np.sign(a) == np.sign(r)).mean()),
            "identical_frac": float((a == r).mean()),
            "near_identical_frac": float((np.abs(resid) <= NEAR_TOL).mean()),
            "constant_frac": constant_rows.sum() / n_common,
            "saturation_frac": saturated_rows.sum() / n_common,
            "ref_constant": bool(rq[0] == rq[2]),
            "act_constant": bool(aq[0] == aq[2]),
        }
        rows.append(row)
    return rows, n_rows, n_common


def _run_audit_locked(split_path, npz_dir=None, merged_dir=None, sidecar_dir=None,
                      stats_dir=None, audit_identity=None, workers=1,
                      manifest_mode="publication"):
    """Audit ``split.train + split.validation`` (never ``split.test``).

    Publication mode requires exactly 674 successful shots and atomically
    writes the split-bound identity last, after the CSV and reconstruction
    provenance. Generic mode preserves the archived warning-and-pool behavior.
    """
    cfg = get_proj_config()
    npz_dir = pathlib.Path(npz_dir) if npz_dir else cfg.npzgeom_dir
    merged_dir = (pathlib.Path(merged_dir) if merged_dir
                  else cfg.mergedh5_gmag_dir)
    sidecar_dir = (pathlib.Path(sidecar_dir) if sidecar_dir
                   else cfg.pfobs_dir)
    stats_dir = (pathlib.Path(stats_dir) if stats_dir
                 else cfg.pfobs_stats_dir)
    identity_path = (pathlib.Path(audit_identity) if audit_identity
                     else stats_dir / IDENTITY_NAME)
    if identity_path.parent != stats_dir:
        raise ValueError(
            "--audit-identity must live in --stats-dir beside source_audit.csv "
            "and reconstruction_provenance.json")

    meta_shots = sorted(int(s["shot"]) for s in json.loads(
        (npz_dir / "meta.json").read_text())["shots"])
    available = {s for s in meta_shots
                 if (npz_dir / f"{s}.npz").is_file()
                 and (merged_dir / f"{s}.h5").is_file()}
    split = _load_manifest(
        split_path, manifest_mode, available_shots=available)
    shots = [int(s) for s in split.train + split.validation]  # test NEVER
    if manifest_mode == "publication" and len(shots) != 674:
        raise RuntimeError(
            f"publication source audit requires 674 train+validation shots, "
            f"got {len(shots)}")
    if manifest_mode == "publication" and os.path.lexists(identity_path):
        try:
            read_regular_nofollow(
                identity_path, label="Work 2 source audit identity")
        except PublicationIOError as exc:
            raise RuntimeError(str(exc)) from exc

    jobs = [(shot, npz_dir / f"{shot}.npz", merged_dir / f"{shot}.h5")
            for shot in shots]
    results = pmap(_audit_shot, jobs, workers, "audit_pf_observability")
    unauditable = [(r["shot"], r["error"]) for r in results if not r["ok"]]
    if unauditable:
        detail = "; ".join(f"{s}: {e}" for s, e in unauditable[:5])
        message = (f"{len(unauditable)}/{len(shots)} train+validation shots "
                   f"are unauditable ({detail}"
                   + (f" (+{len(unauditable) - 5} more)"
                      if len(unauditable) > 5 else "") + ")")
        if manifest_mode == "publication":
            raise RuntimeError(
                "publication source audit rejects every unauditable shot: "
                + message)
        # Generic diagnostics retain the historical report-and-pool path.
        print("WARNING: " + message.replace("are unauditable", "are not auditable"))
    result_shots = [int(r["shot"]) for r in results]
    if (len(results) != len(shots) or len(set(result_shots)) != len(shots)
            or set(result_shots) != set(shots)):
        raise RuntimeError(
            "source audit worker results do not cover exactly the requested "
            "train+validation shots")
    results = [r for r in results if r["ok"]]
    if not results:
        raise RuntimeError(
            "no auditable shots in train+validation; diagnose source "
            "availability before any training")
    unauditable_shots = [s for s, _e in unauditable]

    rows, n_rows, n_common = compute_rows(results)
    csv_path = stats_dir / "source_audit.csv"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow([_fmt(row[c]) for c in CSV_COLUMNS])
    csv_payload = buffer.getvalue().encode("utf-8")

    prov_path = stats_dir / "reconstruction_provenance.json"
    if os.path.lexists(prov_path):
        try:
            provenance_payload = read_regular_nofollow(
                prov_path, label="Work 2 reconstruction provenance")
        except PublicationIOError as exc:
            raise RuntimeError(str(exc)) from exc
        provenance_written = False
    else:
        provenance_payload = json.dumps(
            PROVENANCE_TEMPLATE, indent=2).encode("utf-8")
        provenance_written = True

    identity = None
    identity_payload = None
    if manifest_mode == "publication":
        identity = build_source_audit_identity(
            split_path=split_path,
            split=split,
            npz_dir=npz_dir,
            sidecar_dir=sidecar_dir,
            source_audit_csv=csv_path,
            reconstruction_provenance=prov_path,
            source_audit_csv_bytes=csv_payload,
            reconstruction_provenance_bytes=provenance_payload,
            audited_shots=[r["shot"] for r in results],
            unauditable_shots=unauditable_shots,
            project_root=REPO_ROOT,
        )
        identity_payload = (
            json.dumps(identity, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    # Preflight every destination before the first commit. Broken symlinks,
    # live symlinks, directories, and other nonregular state fail closed.
    for path, label in (
            (csv_path, "Work 2 source audit CSV"),
            (prov_path, "Work 2 reconstruction provenance"),
            (identity_path, "Work 2 source audit identity")):
        if manifest_mode != "publication" and path == identity_path:
            continue
        if os.path.lexists(path):
            try:
                read_regular_nofollow(path, label=label)
            except PublicationIOError as exc:
                raise RuntimeError(str(exc)) from exc

    durable_publish_bytes(
        csv_path, csv_payload, state_label="Work 2 source audit CSV")
    provenance_status = durable_publish_bytes(
        prov_path, provenance_payload,
        state_label="Work 2 reconstruction provenance")
    provenance_written = provenance_status == "created"
    identity_written = False
    if manifest_mode == "publication":
        identity_status = durable_publish_bytes(
            identity_path, identity_payload,
            state_label="Work 2 source audit identity")
        identity_written = identity_status == "created"  # commit LAST

    return {"shots": shots, "n_shots": len(shots),
            "audited_shots": [r["shot"] for r in results],
            "n_audited": len(results), "n_unauditable": len(unauditable),
            "unauditable_shots": unauditable_shots, "n_rows": n_rows,
            "n_common": n_common, "csv_path": str(csv_path),
            "provenance_path": str(prov_path),
            "provenance_written": provenance_written,
            "audit_identity_path": str(identity_path),
            "audit_identity_written": identity_written}


def run(split_path, npz_dir=None, merged_dir=None, sidecar_dir=None,
        stats_dir=None, audit_identity=None, workers=1,
        manifest_mode="publication"):
    """Run the publication audit under the state-family exclusive lock."""
    if manifest_mode == "publication":
        npz_dir = pathlib.Path(npz_dir) if npz_dir else pathlib.Path(NPZ_DIR)
        merged_dir = pathlib.Path(
            merged_dir) if merged_dir else pathlib.Path(MERGED_DIR)
        sidecar_dir = pathlib.Path(
            sidecar_dir) if sidecar_dir else pathlib.Path(SIDECAR_DIR)
        stats_dir = pathlib.Path(
            stats_dir) if stats_dir else pathlib.Path(STATS_ROOT)
        identity_path = pathlib.Path(
            audit_identity) if audit_identity else pathlib.Path(AUDIT_IDENTITY)
    else:
        cfg = get_proj_config()
        npz_dir = pathlib.Path(npz_dir) if npz_dir else cfg.npzgeom_dir
        merged_dir = (
            pathlib.Path(merged_dir) if merged_dir else cfg.mergedh5_gmag_dir)
        sidecar_dir = pathlib.Path(
            sidecar_dir) if sidecar_dir else cfg.pfobs_dir
        stats_dir = pathlib.Path(stats_dir) if stats_dir else cfg.pfobs_stats_dir
        identity_path = pathlib.Path(
            audit_identity) if audit_identity else stats_dir / IDENTITY_NAME
        return _run_audit_locked(
            split_path, npz_dir=npz_dir, merged_dir=merged_dir,
            sidecar_dir=sidecar_dir, stats_dir=stats_dir,
            audit_identity=identity_path, workers=workers,
            manifest_mode=manifest_mode)

    require_publication_paths(
        "publication",
        {
            "split": split_path,
            "npz_dir": npz_dir,
            "merged_dir": merged_dir,
            "sidecar_dir": sidecar_dir,
            "stats_dir": stats_dir,
            "audit_identity": identity_path,
        },
        {
            "split": SPLIT,
            "npz_dir": NPZ_DIR,
            "merged_dir": MERGED_DIR,
            "sidecar_dir": SIDECAR_DIR,
            "stats_dir": STATS_ROOT,
            "audit_identity": AUDIT_IDENTITY,
        },
    )
    marker = pathlib.Path(STATS_ROOT) / "FINAL_TEST_EVALUATED.json"
    if os.path.lexists(marker):
        raise RuntimeError(
            f"{marker} exists: canonical publication state is globally final")
    if os.path.lexists(identity_path):
        try:
            read_regular_nofollow(
                identity_path, label="Work 2 source audit identity")
        except PublicationIOError as exc:
            raise RuntimeError(str(exc)) from exc
    with ValidationFreezeLock(stats_dir) as audit_lock:
        audit_lock.assert_held()
        if os.path.lexists(marker):
            raise RuntimeError(
                f"{marker} exists: canonical publication state is globally final")
        return _run_audit_locked(
            split_path, npz_dir=npz_dir, merged_dir=merged_dir,
            sidecar_dir=sidecar_dir, stats_dir=stats_dir,
            audit_identity=identity_path, workers=workers,
            manifest_mode=manifest_mode)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True,
                    help="frozen split manifest JSON (train+validation only)")
    ap.add_argument("--manifest-mode", choices=("publication", "generic"),
                    default="publication",
                    help="strict publication bundle validation (default) or "
                         "explicit archived generic-manifest loading")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel per-shot loaders (default 4)")
    ap.add_argument("--npz-dir", default=None,
                    help="override the NpzGeom target dir")
    ap.add_argument("--merged-dir", default=None,
                    help="override the MergedH5Gmag source dir")
    ap.add_argument("--sidecar-dir", default=None,
                    help="override the NpzGeomPFObs sidecar dir whose dataset "
                         "metadata is bound into the publication identity")
    ap.add_argument("--stats-dir", default=None,
                    help="override ProjDB/Stats/pf_observability")
    ap.add_argument("--audit-identity", default=None,
                    help="publication source-audit identity output (default: "
                         "<stats-dir>/source_audit_identity.json)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = run(split_path=args.split, npz_dir=args.npz_dir,
                  merged_dir=args.merged_dir, sidecar_dir=args.sidecar_dir,
                  stats_dir=args.stats_dir, audit_identity=args.audit_identity,
                  workers=args.workers, manifest_mode=args.manifest_mode)
    prov = ("written (conservative template)" if summary["provenance_written"]
            else "already present, left untouched")
    unaudited = ""
    if summary["n_unauditable"]:
        unaudited = (f"; WARNING {summary['n_unauditable']} unauditable "
                     f"(no exact native span / load failure -- reported, "
                     f"never repaired)")
    print(f"source_audit: {summary['n_shots']} shots processed, "
          f"{summary['n_audited']} audited{unaudited} "
          f"({summary['n_rows']} rows, {summary['n_common']} common) -> "
          f"{summary['csv_path']}")
    print(f"provenance: {summary['provenance_path']}: {prov}")
    if summary["audit_identity_written"]:
        print(f"audit identity: {summary['audit_identity_path']} written last")


if __name__ == "__main__":
    main()
