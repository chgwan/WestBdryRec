# -*- coding: utf-8 -*-
"""Run the 10-arm M3 attention matrix across the available GPUs, then report.

{NpzGeom, NpzUni500} x {rope_idx, rope_time, upe_idx, upe_time, upe_both}.

Unlike scripts/run_newtrain.py -- which is sequential because each GRU training
saturates a GPU -- an M3 arm at batch 16 does not, and 4 A100s are available, so
arms run concurrently, one per GPU. Every arm is skipped when its artifact AND its
predictions exist, so a crash or a kill resumes instead of restarting.

This writes its OWN report file. It must never call anything in run_newtrain.py:
that module's write_report rebuilds docs/newtrain_results.md from a template with
write_text, which would erase the hand-written V2, PE and Delta-t analyses.

Usage:
  python scripts/run_newtrain_attn.py                # run everything outstanding
  python scripts/run_newtrain_attn.py --gpus 0 1     # restrict to two GPUs
  python scripts/run_newtrain_attn.py --report-only  # just regenerate the report
"""
import argparse
import csv
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.pos_encoding import VARIANTS  # noqa: E402
from src.proj_config import get_proj_config  # noqa: E402

CFG = get_proj_config()
CONFIG = "configs/dcs_model_attn.yml"
DATASETS = [
    ("geom", "ProjDB/datasets/NpzGeom",
     "NpzGeom — native GMAG_BND time base (V1)"),
    ("uni500", "ProjDB/datasets/NpzUni500",
     "NpzUni500 — uniform 500 Hz lattice, target interpolated (V2)"),
]
RUNS = [{"run_name": f"dcs_attn_{tag}_{pe}", "npz_dir": d, "dataset": label,
         "pe": pe, "ds": tag}
        for tag, d, label in DATASETS for pe in VARIANTS]
BENCH = CFG.stats_dir / "dcs_predictor" / "bench_table_attn.csv"
REPORT = CFG.base_dir / "docs" / "newtrain_attention_results.md"
# The incumbent to beat, for context only (capacity-confounded -- see spec 5).
M2_GRU = {"geom": 0.9856, "uni500": 0.9716}

_lock = threading.Lock()


def is_done(run_dir):
    """Finished only when both the artifact and the predictions exist."""
    return (run_dir / "m3.pt").exists() and (run_dir / "m3_pred.npz").exists()


def train_one(run, gpu, log_dir):
    """Invoke train_dcs.py m3 for one arm pinned to ``gpu``. Returns (ok, seconds)."""
    log = log_dir / f"{run['run_name']}.log"
    cmd = [sys.executable, "scripts/train_dcs.py", "m3",
           "--npz-dir", run["npz_dir"], "--config", CONFIG,
           "--pe", run["pe"], "--run-name", run["run_name"],
           "--bench-out", str(BENCH)]
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
        ok, secs = train_one(run, gpu, log_dir)
        with _lock:
            print(f"[gpu{gpu}] {'ok  ' if ok else 'FAIL'} {run['run_name']} "
                  f"in {secs / 60:.1f} min", flush=True)
        work.task_done()


def _rows(csv_path):
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            out[r.get("run", "")] = r          # last row wins
    return out


def write_report(csv_path, out_path):
    """The 2 x 5 matrix. Its own file -- newtrain_results.md is never generated."""
    rows = _rows(csv_path)
    L = ["# M3 windowed-attention PE matrix", "",
         "10 arms: {NpzGeom (V1), NpzUni500 (V2)} x 5 positional encodings.",
         "`m3` = ActSeqAttn, d256/8h/6L (~4.8M params), causal, 2048-step windows",
         "with a 512-step unscored context prefix. 18 strict actuators only --",
         "positional information enters through the architecture, never as a column.",
         "",
         "**Read the pooled CCC below as context only.** The decision rests on the",
         "pre-registered paired per-shot tests in",
         "`exploration/attn_paired_justify.py`; pooled CCC at this accuracy level is",
         "single-shot-dominated (see `docs/newtrain_results.md`, the PE ablation).",
         "",
         "**Two confounds, stated up front.** (1) 4.8M params vs m2-GRU's ~30k is",
         "160x, so an m3 > m2 result reads as \"bigger model with attention\", not",
         "\"attention beats recurrence\". (2) Windowing gives m3 strictly less context",
         "than m2, which sees whole shots. Both are identical across all 10 arms, so",
         "neither threatens the within-matrix comparisons.",
         "",
         "Spec: `docs/superpowers/specs/2026-08-07-windowed-attention-pe-matrix-design.md`",
         ""]
    for tag, _d, label in DATASETS:
        L += [f"## {label}", "",
              "| PE variant | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | "
              "centre RMSE mm | abs bnd RMSE mm | n | m2-GRU CCC |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for pe in VARIANTS:
            r = rows.get(f"dcs_attn_{tag}_{pe}")
            ref = f"{M2_GRU[tag]:.4f}"
            if r is None:
                L.append(f"| `{pe}` | not run | | | | | | | | {ref} |")
                continue

            def g(k):
                try:
                    return f"{float(r.get(k, '')):.4f}"
                except (TypeError, ValueError):
                    return "—"
            L.append(f"| `{pe}` | {g('ccc')} | {g('r2')} | {g('rmse_cm')} | "
                     f"{g('rgeom_mae_mm')} | {g('zgeom_mae_mm')} | "
                     f"{g('centre_rmse_mm')} | {g('abs_bnd_rmse_mm')} | "
                     f"{r.get('n_shots', '—')} | {ref} |")
        L.append("")
    L += ["## Noise floor (the calibration)", "",
          "On NpzUni500 the axis is exactly uniform (100 % of intervals within 10 us,",
          "0 skipped samples), so `rope_time` and `rope_idx` are *numerically*",
          "identical encodings, as are `upe_time` and `upe_idx`. Their measured",
          "difference therefore **is** the single-seed noise floor. No V1 effect is",
          "claimed unless it exceeds it. See spec section 6.", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--gpus", type=int, nargs="*", default=None,
                    help="GPU ids to use (default: every visible GPU)")
    args = ap.parse_args()

    if not args.report_only:
        import torch
        gpus = args.gpus if args.gpus else list(range(torch.cuda.device_count()))
        if not gpus:
            raise SystemExit("no GPU visible -- refusing to run 10 arms on CPU")
        log_dir = CFG.stats_dir / "dcs_predictor" / "logs_attn"
        log_dir.mkdir(parents=True, exist_ok=True)
        BENCH.parent.mkdir(parents=True, exist_ok=True)
        work = queue.Queue()
        for run in RUNS:
            run_dir = CFG.trains_dir / run["run_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            if is_done(run_dir):
                print(f"[skip] {run['run_name']} already complete", flush=True)
            else:
                work.put(run)
        print(f"{work.qsize()} arms outstanding on GPUs {gpus}", flush=True)
        threads = [threading.Thread(target=_worker, args=(g, work, log_dir),
                                    daemon=True) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print(f"report -> {write_report(BENCH, REPORT)}")


if __name__ == "__main__":
    main()
