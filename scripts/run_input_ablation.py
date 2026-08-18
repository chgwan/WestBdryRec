# -*- coding: utf-8 -*-
"""Run the input-ablation matrix across this job's GPUs, then report.

Ablation spec (ablation_inputs.md), all m3b ActSeqAttn with --pe rope_time on
NpzGeom, 3 seeds each:

  full   18 channels  -> NO new runs; the existing drop_f0.0_rope_time_s{0,1,2}
                         (dropout study, f=0) ARE this arm
  nopha  16 channels   = full - PhaLH1/2_scope        (dcs_model_attn_nopha.yml)
  nopow  11 channels   = nopha - PowLH1/2 + PowIC1/2/3 (dcs_model_attn_nopha_nopow.yml)

Every unit is skipped when its artifact AND its predictions exist, so a crash
or a kill resumes instead of restarting. Writes its OWN bench CSV and report;
never touches the other results docs.

Usage (inside a PBS job holding 4 GPUs):
  python scripts/run_input_ablation.py --gpus 0 1 2 3
  python scripts/run_input_ablation.py --report-only
"""
import argparse
import csv
import os
import pathlib
import queue
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.proj_config import get_proj_config  # noqa: E402

CFG = get_proj_config()
NPZ = "ProjDB/datasets/NpzGeom"
ARMS = [("nopha", "configs/dcs_model_attn_nopha.yml", 16),
        ("nopha_nopow", "configs/dcs_model_attn_nopha_nopow.yml", 11)]
SEEDS = (0, 1, 2)
RUNS = [{"run_name": f"ab_{arm}_s{seed}", "arm": arm, "config": config,
         "seed": seed}
        for arm, config, _n in ARMS for seed in SEEDS]           # 6 runs
BENCH = CFG.stats_dir / "dcs_predictor" / "bench_table_ablation_inputs.csv"
DROPOUT_BENCH = CFG.stats_dir / "dcs_predictor" / "bench_table_dropout.csv"
REPORT = CFG.base_dir / "docs" / "input_ablation_results.md"

_lock = threading.Lock()


def is_done(run_dir):
    return (run_dir / "m3.pt").exists() and (run_dir / "m3_pred.npz").exists()


def train_one(run, gpu, log_dir):
    log = log_dir / f"{run['run_name']}.log"
    cmd = [sys.executable, "scripts/train_dcs.py", "m3",
           "--npz-dir", NPZ, "--config", run["config"],
           "--pe", "rope_time", "--seed", str(run["seed"]),
           "--drop-frac", "0.0", "--drop-seed", "0",
           "--run-name", run["run_name"], "--bench-out", str(BENCH)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               HDF5_USE_FILE_LOCKING="FALSE")
    t0 = time.perf_counter()
    with log.open("w") as fh:
        fh.write(f"GPU {gpu}\n" + " ".join(cmd) + "\n\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=str(CFG.base_dir), stdout=fh,
                             stderr=subprocess.STDOUT, env=env)
    return rc == 0, time.perf_counter() - t0


def _worker(gpu, work, log_dir):
    while True:
        try:
            run = work.get_nowait()
        except queue.Empty:
            return
        with _lock:
            print(f"[gpu{gpu}] start {run['run_name']}", flush=True)
        ok, dt = train_one(run, gpu, log_dir)
        with _lock:
            print(f"[gpu{gpu}] {'done' if ok else 'FAILED'} {run['run_name']} "
                  f"({dt / 60:.1f} min)", flush=True)


def _rows(csv_path):
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            out[r.get("run", "")] = r          # last row wins
    return out


def _f(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def write_report():
    """Per-run table + per-arm summary; the full arm reads the dropout CSV."""
    rows = _rows(BENCH)
    full = {f"drop_f0.0_rope_time_s{s}": r
            for s, r in _rows(DROPOUT_BENCH).items()
            if s.startswith("drop_f0.0_rope_time_s")}
    L = ["# Input ablation: full vs -Pha* vs -Pha*-Pow* (rope_time M3)", "",
         "Spec: `ablation_inputs.md`. All arms m3b ActSeqAttn, `--pe rope_time`,",
         "NpzGeom, seeds {0,1,2}. The **full** arm reuses the dropout study's",
         "f=0 rope_time runs (identical config + inputs + seed contract).", "",
         "## Per-run results", "",
         "| arm | n_ch | seed | CCC | R2 | RMSE cm | abs bnd mm | n |",
         "|---|---|---|---|---|---|---|---|"]
    entries = ([("full", 18, s, full.get(f"drop_f0.0_rope_time_s{s}"))
                for s in SEEDS]
               + [(run["arm"], dict(ARMS)[run["arm"]] if False else
                  next(n for a, _c, n in ARMS if a == run["arm"]), run["seed"],
                  rows.get(run["run_name"])) for run in RUNS])
    for arm, nch, seed, r in entries:
        if r is None:
            L.append(f"| {arm} | {nch} | {seed} | not run | | | | |")
            continue
        L.append(f"| {arm} | {nch} | {seed} | {_f(r,'ccc'):.5f} | "
                 f"{_f(r,'r2'):.4f} | {_f(r,'rmse_cm'):.4f} | "
                 f"{_f(r,'abs_bnd_rmse_mm'):.2f} | {r.get('n_shots','—')} |")

    L += ["", "## Per-arm summary (pooled bench CCC, mean over seeds)", "",
          "| arm | n_ch | mean CCC (SD) | mean RMSE cm (SD) |", "|---|---|---|---|"]
    for arm, nch in [("full", 18)] + [(a, n) for a, _c, n in ARMS]:
        rs = (list(full.values()) if arm == "full"
              else [rows[f"ab_{arm}_s{s}"] for s in SEEDS
                    if f"ab_{arm}_s{s}" in rows])
        vals = [v for v in (_f(r, "ccc") for r in rs) if v is not None]
        rms = [v for v in (_f(r, "rmse_cm") for r in rs) if v is not None]
        if len(vals) >= 2:
            L.append(f"| {arm} | {nch} | {statistics.mean(vals):.5f} "
                     f"({statistics.stdev(vals):.5f}) | "
                     f"{statistics.mean(rms):.4f} "
                     f"({statistics.stdev(rms):.4f}) |")
        elif vals:
            L.append(f"| {arm} | {nch} | {vals[0]:.5f} (—) | {rms[0]:.4f} (—) |")
        else:
            L.append(f"| {arm} | {nch} | not run | |")
    L += ["", "Analysis (paired per-shot, seed floor, clean-75): "
          "`exploration/input_ablation_analysis.py`.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L))
    return REPORT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        gpus = args.gpus if args.gpus else [0, 1, 2, 3]
        log_dir = CFG.base_dir / "logs" / "input_ablation"
        log_dir.mkdir(parents=True, exist_ok=True)
        BENCH.parent.mkdir(parents=True, exist_ok=True)
        work = queue.Queue()
        for run in RUNS:
            if is_done(CFG.trains_dir / run["run_name"]):
                print(f"skip {run['run_name']} (artifact + predictions exist)")
                continue
            work.put(run)
        print(f"{work.qsize()} runs outstanding on GPUs {gpus}", flush=True)
        threads = [threading.Thread(target=_worker, args=(g, work, log_dir))
                   for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(f"report -> {write_report()}")


if __name__ == "__main__":
    main()
