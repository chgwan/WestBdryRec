# -*- coding: utf-8 -*-
"""Run the newTrain retrain matrix unattended, then report against the baseline.

Six units: two runs (with and without the time positional encoding) x three models.
Sequential by design -- each training already saturates one GPU. Every unit is skipped if
its artifact *and* its predictions exist, so a crash or a kill resumes instead of
restarting. Nothing here invents a number: a unit that did not produce a bench row shows
up as "not run" in the report.

Usage:
  python scripts/run_newtrain.py                 # run everything outstanding
  python scripts/run_newtrain.py --report-only   # just regenerate the report
"""
import argparse
import csv
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.proj_config import get_proj_config  # noqa: E402

CFG = get_proj_config()
MODELS = ("m0", "m1", "m2")
RUNS = [
    {"run_name": "dcs_actuator_geom",
     "config": "configs/dcs_model_geom.yml",
     "npz_dir": "ProjDB/datasets/NpzGeom",
     "dataset": "NpzGeom — native GMAG_BND time base (V1)",
     "label": "with time PE (headline)"},
    {"run_name": "dcs_actuator_geom_nope",
     "config": "configs/dcs_model.yml",
     "npz_dir": "ProjDB/datasets/NpzGeom",
     "dataset": "NpzGeom — native GMAG_BND time base (V1)",
     "label": "no time PE (ablation)"},
    {"run_name": "dcs_actuator_uni500",
     "config": "configs/dcs_model_geom.yml",
     "npz_dir": "ProjDB/datasets/NpzUni500",
     "dataset": "NpzUni500 — uniform 500 Hz lattice, target interpolated (V2)",
     "label": "with time PE"},
    {"run_name": "dcs_actuator_uni500_nope",
     "config": "configs/dcs_model.yml",
     "npz_dir": "ProjDB/datasets/NpzUni500",
     "dataset": "NpzUni500 — uniform 500 Hz lattice, target interpolated (V2)",
     "label": "no time PE"},
]
BENCH = CFG.stats_dir / "dcs_predictor" / "bench_table_geom.csv"
REPORT = CFG.base_dir / "docs" / "newtrain_results.md"
BASELINE = {"m0": (0.9579, 0.8452), "m1": (0.9530, 0.8284), "m2": (0.9668, 0.8763)}


def is_done(run_dir, model):
    """A unit counts as finished only when both its artifact and predictions exist."""
    art = run_dir / (f"{model}.joblib" if model == "m0" else f"{model}.pt")
    return art.exists() and (run_dir / f"{model}_pred.npz").exists()


def train_one(run, model, log_dir):
    """Invoke train_dcs.py for one (run, model). Returns (ok, seconds)."""
    log = log_dir / f"{run['run_name']}_{model}.log"
    cmd = [sys.executable, "scripts/train_dcs.py", model,
           "--npz-dir", run["npz_dir"], "--config", run["config"],
           "--run-name", run["run_name"], "--bench-out", str(BENCH)]
    t0 = time.perf_counter()
    with log.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=str(CFG.base_dir), stdout=fh,
                             stderr=subprocess.STDOUT)
    return rc == 0, time.perf_counter() - t0


def _rows(csv_path):
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            out[(r.get("run", ""), r["model"])] = r          # last row wins
    return out


def write_report(csv_path, out_path):
    """Markdown comparison of every unit against the NpzOrigin baseline."""
    rows = _rows(csv_path)
    L = ["# newTrain retrain results", "",
         "Baseline is `NpzOrigin` / `dcs_actuator`, 76 test shots.",
         "34 outputs = r(theta)@32 + absolute (Rgeom, Zgeom); S0-S5 slice filters and",
         "the per-slice GMAG_GEOM centre apply to every run below.", "",
         "Several things differ from the baseline at once (time base, slice population,",
         "target origin, input columns, output width), so a difference in the r(theta)",
         "numbers is **not attributable to any single one**. The `_nope` runs isolate",
         "the PE; the `uni500` pair isolates the uniform axis against `geom`.",
         "A drop is not necessarily a regression: the baseline was partly scored on",
         "interpolated boundaries and on slices these filters reject.",
         "",
         "On `NpzUni500` the target itself is interpolated onto the lattice, so the",
         "S0-S5 filters there judge interpolated geometry -- see `src_gap_ms` and",
         "`meta.json:grid.fabricated_valid_slices` for how much.", ""]
    for run in RUNS:
        L += [f"## {run['run_name']} — {run['label']}", "",
              f"Dataset: {run['dataset']}", "",
              "| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | "
              "centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            r = rows.get((run["run_name"], m))
            b = BASELINE[m]
            if r is None:
                L.append(f"| {m} | not run | | | | | | | | {b[0]:.4f} / {b[1]:.4f} |")
                continue
            def g(k):
                v = r.get(k, "")
                try:
                    return f"{float(v):.4f}"
                except (TypeError, ValueError):
                    return "—"
            L.append(f"| {m} | {g('ccc')} | {g('r2')} | {g('rmse_cm')} | "
                     f"{g('rgeom_mae_mm')} | {g('zgeom_mae_mm')} | "
                     f"{g('centre_rmse_mm')} | {g('abs_bnd_rmse_mm')} | "
                     f"{r.get('n_shots','—')} | {b[0]:.4f} / {b[1]:.4f} |")
        L.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        log_dir = CFG.stats_dir / "dcs_predictor" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        BENCH.parent.mkdir(parents=True, exist_ok=True)
        for run in RUNS:
            run_dir = CFG.trains_dir / run["run_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            for m in MODELS:
                if is_done(run_dir, m):
                    print(f"[skip] {run['run_name']}/{m} already complete", flush=True)
                    continue
                print(f"[run ] {run['run_name']}/{m} ...", flush=True)
                ok, secs = train_one(run, m, log_dir)
                print(f"[{'ok  ' if ok else 'FAIL'}] {run['run_name']}/{m} "
                      f"in {secs / 60:.1f} min (log in {log_dir})", flush=True)
    p = write_report(BENCH, REPORT)
    print(f"report -> {p}")


if __name__ == "__main__":
    main()
