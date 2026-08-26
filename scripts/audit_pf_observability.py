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
  python scripts/audit_pf_observability.py --split configs/splits/pfobs_random_pilot.json
"""
import argparse
import csv
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import h5py  # noqa: E402  (opened only inside build_one)
import numpy as np  # noqa: E402

from src.data.build_pf_observability import PF_SCOPES, build_one  # noqa: E402
from src.ml.pf_observability import load_split  # noqa: E402
from src.proj_config import get_proj_config  # noqa: E402
from src.utils import pmap  # noqa: E402

LAG_MAX = 64        # cross-correlation search window, native samples (~131 ms)
FLAT_RUN_MIN = 8    # shortest run of identical consecutive values, samples
NEAR_TOL = 1e-6     # |actual - ref| counted as near-identical

PROVENANCE_TEMPLATE = {
    "status": "unknown",
    "evidence": [],
    "claim_limit": "observability_of_the_GMAG_BND_reconstruction_product",
}

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


def run(split_path, npz_dir=None, merged_dir=None, stats_dir=None,
        workers=1):
    """Audit ``split.train + split.validation`` (never ``split.test``).

    Returns ``{shots, n_shots, n_rows, n_common, csv_path,
    provenance_path, provenance_written}``.
    """
    cfg = get_proj_config()
    npz_dir = pathlib.Path(npz_dir) if npz_dir else cfg.npzgeom_dir
    merged_dir = (pathlib.Path(merged_dir) if merged_dir
                  else cfg.mergedh5_gmag_dir)
    stats_dir = (pathlib.Path(stats_dir) if stats_dir
                 else cfg.pfobs_stats_dir)

    meta_shots = sorted(int(s["shot"]) for s in json.loads(
        (npz_dir / "meta.json").read_text())["shots"])
    available = {s for s in meta_shots
                 if (npz_dir / f"{s}.npz").is_file()
                 and (merged_dir / f"{s}.h5").is_file()}
    split = load_split(split_path, available_shots=available)
    shots = [int(s) for s in split.train + split.validation]  # test NEVER

    jobs = [(shot, npz_dir / f"{shot}.npz", merged_dir / f"{shot}.h5")
            for shot in shots]
    results = pmap(_audit_shot, jobs, workers, "audit_pf_observability")
    unauditable = [(r["shot"], r["error"]) for r in results if not r["ok"]]
    if unauditable:
        # A shot without an exact native span is a DATA FINDING, not a
        # repair job (spec 5.2: never nearest-neighbour or interpolate;
        # 5.3: stop before training and diagnose source availability).
        # The audit reports it and pools only the auditable shots; the
        # Task-1 sidecar build remains the enforcement point.
        detail = "; ".join(f"{s}: {e}" for s, e in unauditable[:5])
        print(f"WARNING: {len(unauditable)}/{len(shots)} train+validation "
              f"shots are not auditable ({detail}"
              + (f" (+{len(unauditable) - 5} more)" if len(unauditable) > 5
                 else "") + ")")
    results = [r for r in results if r["ok"]]
    if not results:
        raise RuntimeError(
            "no auditable shots in train+validation; diagnose source "
            "availability before any training")
    unauditable_shots = [s for s, _e in unauditable]

    rows, n_rows, n_common = compute_rows(results)
    stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = stats_dir / "source_audit.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for row in rows:
            w.writerow([_fmt(row[c]) for c in CSV_COLUMNS])

    prov_path = stats_dir / "reconstruction_provenance.json"
    if prov_path.exists():
        provenance_written = False
    else:
        prov_path.write_text(json.dumps(PROVENANCE_TEMPLATE, indent=2))
        provenance_written = True
    return {"shots": shots, "n_shots": len(shots),
            "audited_shots": [r["shot"] for r in results],
            "n_audited": len(results), "n_unauditable": len(unauditable),
            "unauditable_shots": unauditable_shots, "n_rows": n_rows,
            "n_common": n_common, "csv_path": str(csv_path),
            "provenance_path": str(prov_path),
            "provenance_written": provenance_written}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True,
                    help="frozen split manifest JSON (train+validation only)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel per-shot loaders (default 4)")
    ap.add_argument("--npz-dir", default=None,
                    help="override the NpzGeom target dir")
    ap.add_argument("--merged-dir", default=None,
                    help="override the MergedH5Gmag source dir")
    ap.add_argument("--stats-dir", default=None,
                    help="override ProjDB/Stats/pf_observability")
    args = ap.parse_args()
    summary = run(split_path=args.split, npz_dir=args.npz_dir,
                  merged_dir=args.merged_dir, stats_dir=args.stats_dir,
                  workers=args.workers)
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


if __name__ == "__main__":
    main()
