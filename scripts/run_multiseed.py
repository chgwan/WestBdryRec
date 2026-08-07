# -*- coding: utf-8 -*-
"""Run the 4-arm x 3-seed capacity/variance matrix across the GPUs, then report.

Arms: m3-small (capacity-matched) and m2-GRU answer "does attention beat recurrence
at matched capacity"; m3-big rope_idx vs upe_both answer "does the positional effect
survive a real seed-variance estimate". All on NpzGeom (V1).

Every unit is skipped when its artifact AND its predictions exist, so a crash or a
kill resumes instead of restarting.

This writes its OWN report file. It must never call anything in run_newtrain.py:
that module's write_report rebuilds docs/newtrain_results.md from a template with
write_text, which would erase the hand-written analyses.

Usage:
  python scripts/run_multiseed.py                 # run everything outstanding
  python scripts/run_multiseed.py --gpus 0 1 3    # restrict to three GPUs
  python scripts/run_multiseed.py --report-only   # just regenerate the report
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
SEEDS = (0, 1, 2)
# (arm, model, config, pe) -- pe is None for the GRU
ARMS = [
    ("m3s_rope_idx", "m3", "configs/dcs_model_attn_small.yml", "rope_idx"),
    ("m2_nope",      "m2", "configs/dcs_model.yml",            None),
    ("m3b_rope_idx", "m3", "configs/dcs_model_attn.yml",       "rope_idx"),
    ("m3b_upe_both", "m3", "configs/dcs_model_attn.yml",       "upe_both"),
]
RUNS = [{"run_name": f"ms_{arm}_s{seed}", "arm": arm, "model": model,
         "config": config, "pe": pe, "seed": seed}
        for arm, model, config, pe in ARMS for seed in SEEDS]
BENCH = CFG.stats_dir / "dcs_predictor" / "bench_table_multiseed.csv"
REPORT = CFG.base_dir / "docs" / "newtrain_multiseed_results.md"
# Published single-seed numbers, for the gate-zero sanity check (spec 6.1).
PUBLISHED = {"m3b_rope_idx": 0.98736, "m2_nope": 0.98555}
PARAMS = {"m3s_rope_idx": 103_778, "m2_nope": 53_730,
          "m3b_rope_idx": 4_753_698, "m3b_upe_both": 4_753_698}

_lock = threading.Lock()


def is_done(run_dir, model):
    """Finished only when both the artifact and the predictions exist."""
    return (run_dir / f"{model}.pt").exists() and (run_dir / f"{model}_pred.npz").exists()


def train_one(run, gpu, log_dir):
    """Invoke train_dcs.py for one (arm, seed) pinned to ``gpu``."""
    log = log_dir / f"{run['run_name']}.log"
    cmd = [sys.executable, "scripts/train_dcs.py", run["model"],
           "--npz-dir", NPZ, "--config", run["config"],
           "--run-name", run["run_name"], "--seed", str(run["seed"]),
           "--bench-out", str(BENCH)]
    if run["pe"]:
        cmd += ["--pe", run["pe"]]
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
    """Per-arm seed table + the measured seed SD. Its own file."""
    rows = _rows(csv_path)
    L = ["# Capacity-matched attention + multi-seed variance", "",
         "4 arms x 3 seeds on NpzGeom (V1). The training seed controls weight init,",
         "shuffle and dropout only -- the train/val/test split is pinned at split",
         "seed 0, so all 12 runs score the same 76 test shots.", "",
         "**Seed 0 is a fresh draw, not a reproduction.** The published single-seed",
         "runs had no manual_seed call, so their init came from an unknown",
         "entropy-seeded draw. Gate zero (spec 6.1) only checks that seed 0 lands",
         "within +-0.002 of them, confirming the seeding plumbing did not break",
         "training. All claim tests use the 12 new runs only.", "",
         "Spec: `docs/superpowers/specs/2026-08-07-capacity-matched-multiseed-design.md`",
         "Analysis: `exploration/multiseed_analysis.py`", "",
         "| arm | params | seed | CCC | R2 | RMSE cm | centre mm | abs bnd mm | n |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm, model, _cfgp, _pe in ARMS:
        for seed in SEEDS:
            r = rows.get(f"ms_{arm}_s{seed}")
            p = f"{PARAMS[arm]:,}"
            if r is None:
                L.append(f"| `{arm}` | {p} | {seed} | not run | | | | | |")
                continue

            def g(k):
                try:
                    return f"{float(r.get(k, '')):.4f}"
                except (TypeError, ValueError):
                    return "—"
            L.append(f"| `{arm}` | {p} | {seed} | {g('ccc')} | {g('r2')} | "
                     f"{g('rmse_cm')} | {g('centre_rmse_mm')} | "
                     f"{g('abs_bnd_rmse_mm')} | {r.get('n_shots', '—')} |")
    L += ["", "## Per-arm seed variance (the measured noise floor)", "",
          "This replaces the predecessor matrix's single-sample V2 proxy (0.000406).",
          "",
          "| arm | mean CCC | seed SD | min | max |", "|---|---|---|---|---|"]
    for arm, _m, _c, _p in ARMS:
        vals = []
        for seed in SEEDS:
            r = rows.get(f"ms_{arm}_s{seed}")
            if r:
                try:
                    vals.append(float(r["ccc"]))
                except (TypeError, ValueError, KeyError):
                    pass
        if len(vals) >= 2:
            L.append(f"| `{arm}` | {statistics.mean(vals):.5f} | "
                     f"{statistics.stdev(vals):.6f} | {min(vals):.5f} | "
                     f"{max(vals):.5f} |")
        else:
            L.append(f"| `{arm}` | — | — | — | — |")
    L += ["", "## Gate zero — seed-0 sanity check (spec 6.1)", "",
          "| arm | published | seed 0 | delta | within +-0.002 |",
          "|---|---|---|---|---|"]
    for arm, pub in PUBLISHED.items():
        r = rows.get(f"ms_{arm}_s0")
        if r:
            try:
                got = float(r["ccc"])
                d = got - pub
                L.append(f"| `{arm}` | {pub:.5f} | {got:.5f} | {d:+.5f} | "
                         f"{'yes' if abs(d) <= 0.002 else '**NO**'} |")
                continue
            except (TypeError, ValueError, KeyError):
                pass
        L.append(f"| `{arm}` | {pub:.5f} | not run | — | — |")
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
            raise SystemExit("no GPU visible -- refusing to run 12 arms on CPU")
        log_dir = CFG.stats_dir / "dcs_predictor" / "logs_multiseed"
        log_dir.mkdir(parents=True, exist_ok=True)
        BENCH.parent.mkdir(parents=True, exist_ok=True)
        work = queue.Queue()
        for run in RUNS:
            run_dir = CFG.trains_dir / run["run_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            if is_done(run_dir, run["model"]):
                print(f"[skip] {run['run_name']} already complete", flush=True)
            else:
                work.put(run)
        print(f"{work.qsize()} runs outstanding on GPUs {gpus}", flush=True)
        threads = [threading.Thread(target=_worker, args=(g, work, log_dir),
                                    daemon=True) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print(f"report -> {write_report(BENCH, REPORT)}")


if __name__ == "__main__":
    main()
