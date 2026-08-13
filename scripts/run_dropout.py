# -*- coding: utf-8 -*-
"""Run the dropout dose-response matrix across this job's GPUs, then report.

f in {0.0, 0.3, 0.6} x {rope_idx, rope_time} x seeds {0,1,2} = 18 runs, all m3b on
NpzGeom. Each unit is one independent single-GPU training -- NOT DDP: the arms must
stay comparable to the published m3b numbers, which were single-GPU.

The drop mask is per shot and independent of the training seed, so the two encodings
drop identical slices and the three seeds drop identical slices (spec section 4.2).

Every unit is skipped when its artifact AND its predictions exist, so a walltime kill
resumes instead of restarting.

This writes its OWN report file. It must never call anything in run_newtrain.py: that
module's write_report rebuilds docs/newtrain_results.md from a template with
write_text, which would erase the hand-written analyses.

Usage (inside a PBS job holding 4 GPUs):
  python scripts/run_dropout.py --gpus 0 1 2 3                 # everything outstanding
  python scripts/run_dropout.py --gpus 0 1 2 3 --fracs 0.0     # the gate arm only
  python scripts/run_dropout.py --report-only                  # regenerate the report
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
CONFIG = "configs/dcs_model_attn.yml"
FRACS = (0.0, 0.3, 0.6)
PES = ("rope_idx", "rope_time")
SEEDS = (0, 1, 2)
DROP_SEED = 0
RUNS = [{"run_name": f"drop_f{f}_{pe}_s{seed}", "f": f, "pe": pe, "seed": seed}
        for f in FRACS for pe in PES for seed in SEEDS]          # 18 runs
BENCH = CFG.stats_dir / "dcs_predictor" / "bench_table_dropout.csv"
REPORT = CFG.base_dir / "docs" / "newtrain_dropout_results.md"
# The capacity/variance study's m3b_rope_idx seeds -- the f=0 gate (spec section 6.1).
PUBLISHED_F0_ROPE_IDX = {0: 0.98738, 1: 0.98946, 2: 0.98519}
GATE_TOL = 0.002
PARAMS = 4_753_698

_lock = threading.Lock()


def is_done(run_dir):
    """Finished only when both the artifact and the predictions exist."""
    return (run_dir / "m3.pt").exists() and (run_dir / "m3_pred.npz").exists()


def train_one(run, gpu, log_dir):
    """Invoke train_dcs.py for one (f, pe, seed) pinned to ``gpu``."""
    log = log_dir / f"{run['run_name']}.log"
    cmd = [sys.executable, "scripts/train_dcs.py", "m3",
           "--npz-dir", NPZ, "--config", CONFIG,
           "--pe", run["pe"], "--seed", str(run["seed"]),
           "--drop-frac", str(run["f"]), "--drop-seed", str(DROP_SEED),
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
    """{run_name: row}, last row wins -- an NSCC rerun supersedes a transferred row."""
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            out[r.get("run", "")] = r
    return out


def _f(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def write_report(csv_path, out_path):
    """Per-run table + the dose-response + the f=0 gate. Its own file."""
    rows = _rows(csv_path)
    L = ["# Dropout dose-response: rope_idx vs rope_time", "",
         "18 runs: f in {0.0, 0.3, 0.6} x {rope_idx, rope_time} x seeds {0,1,2},",
         "all m3b (4,753,698 params) on NpzGeom, 2048-step windows over the",
         "**retained** sequence. Random per-step dropout breaks the affine",
         "equivalence t ~ c*i that made the two encodings identical on the real",
         "axis, so this is the one regime where the question is answerable.", "",
         "**Interpretation ceiling (spec section 1.4).** A win for `rope_time` here",
         "means *real-time PE helps under sparse/irregular sampling* -- NOT that",
         "timing helps on the real dense axis, where the PE matrix already showed",
         "it does not. Dropout degrades the implicit timing the actuator values",
         "otherwise carry; the honest framing is regime-specific.", "",
         "The drop mask is deterministic per shot and independent of the training",
         "seed, so at a given f both encodings and all three seeds see identical",
         "retained slices. CCC is over the retained test slices: comparable",
         "between encodings at a given f, **not** across f.", "",
         "Spec: `docs/superpowers/specs/2026-08-13-dropout-rope-idx-vs-time-design.md`",
         "Plan: `docs/superpowers/plans/2026-08-13-dropout-rope-idx-vs-time-nscc.md`",
         "Analysis: `exploration/dropout_analysis.py`", "",
         "## Per-run results", "",
         "| f | pe | seed | CCC | R2 | RMSE cm | centre mm | abs bnd mm | n |",
         "|---|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        r = rows.get(run["run_name"])
        if r is None:
            L.append(f"| {run['f']} | `{run['pe']}` | {run['seed']} | not run | | | | | |")
            continue

        def g(k):
            v = _f(r, k)
            return "—" if v is None else f"{v:.4f}"
        L.append(f"| {run['f']} | `{run['pe']}` | {run['seed']} | {g('ccc')} | "
                 f"{g('r2')} | {g('rmse_cm')} | {g('centre_rmse_mm')} | "
                 f"{g('abs_bnd_rmse_mm')} | {r.get('n_shots', '—')} |")

    L += ["", "## Dose-response (pooled bench CCC, mean over seeds)", "",
          "| f | rope_idx mean (SD) | rope_time mean (SD) | delta (time - idx) |",
          "|---|---|---|---|"]
    for f in FRACS:
        cell, means = {}, {}
        for pe in PES:
            vals = [_f(rows[n], "ccc") for n in
                    (f"drop_f{f}_{pe}_s{s}" for s in SEEDS) if n in rows]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                means[pe] = statistics.mean(vals)
                cell[pe] = f"{means[pe]:.5f} ({statistics.stdev(vals):.5f}) n={len(vals)}"
            elif vals:
                means[pe] = vals[0]
                cell[pe] = f"{vals[0]:.5f} (—) n=1"
            else:
                cell[pe] = "not run"
        d = (f"{means['rope_time'] - means['rope_idx']:+.5f}"
             if len(means) == 2 else "—")
        L.append(f"| {f} | {cell['rope_idx']} | {cell['rope_time']} | {d} |")

    L += ["", "## Gate zero — f=0 must reproduce the published m3b_rope_idx", "",
          "The gate confirms the dropout code is a no-op at f=0. If it fails, the",
          "f=0 path was perturbed and every downstream number is incomparable.", "",
          "| seed | published | f=0 rope_idx | delta | within +-0.002 |",
          "|---|---|---|---|---|"]
    for s, pub in PUBLISHED_F0_ROPE_IDX.items():
        r = rows.get(f"drop_f0.0_rope_idx_s{s}")
        got = None if r is None else _f(r, "ccc")
        if got is None:
            L.append(f"| {s} | {pub:.5f} | not run | | |")
            continue
        d = got - pub
        L.append(f"| {s} | {pub:.5f} | {got:.5f} | {d:+.5f} | "
                 f"{'yes' if abs(d) <= GATE_TOL else '**NO**'} |")
    L += ["", "> The f=0 `rope_time` runs have no 3-seed reference; they should land",
          "> near the PE matrix's single `rope_time` (0.9828) as a loose sanity",
          "> check only.", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, nargs="*", default=None,
                    help="GPU ordinals to use (default: all 4 in a g3 node)")
    ap.add_argument("--fracs", type=float, nargs="*", default=None,
                    help="restrict to these drop fractions (e.g. --fracs 0.0)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        gpus = args.gpus if args.gpus else [0, 1, 2, 3]
        log_dir = CFG.base_dir / "logs" / "dropout"
        log_dir.mkdir(parents=True, exist_ok=True)
        BENCH.parent.mkdir(parents=True, exist_ok=True)
        work = queue.Queue()
        todo = [r for r in RUNS
                if args.fracs is None or any(abs(r["f"] - f) < 1e-9 for f in args.fracs)]
        for run in todo:
            if is_done(CFG.trains_dir / run["run_name"]):
                print(f"skip {run['run_name']} (artifact + predictions exist)")
                continue
            work.put(run)
        n = work.qsize()
        print(f"{n} runs outstanding on GPUs {gpus}", flush=True)
        threads = [threading.Thread(target=_worker, args=(g, work, log_dir))
                   for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(f"report -> {write_report(BENCH, REPORT)}")


if __name__ == "__main__":
    main()
