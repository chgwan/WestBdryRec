# -*- coding: utf-8 -*-
"""Train IMAS T0 strict-actuator LCFS models and benchmark pooled CCC on test.

Mirrors scripts/train_dcs.py but on the IMAS corpus (cfg.imas_h5_dir /
imas_npz_dir). Inputs = the T0 strict actuators only (17 PF + b0 + LH + IC),
read via dataset.engineer (b0-normalised pf_norm) -- T2 globals/axes are
excluded by the fairness guardrail. M0 reuses train_save (shape + magnetic-axis
model); M1/M2 reuse the shared backbone (ResMLP / ActSeqGRU, train_neural).
Success metric: pooled Lin's CCC > 0.9 on the held-out test set (r(theta));
axis / absolute-(R,Z) metrics are reported separately for M0.

    python scripts/train_imas.py m0
    python scripts/train_imas.py m1 --epochs 80
    python scripts/train_imas.py all
    python scripts/train_imas.py m0 --max-shots 40        # smoke test
"""
import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import h5py  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.proj_config import get_proj_config  # noqa: E402
from src.ml import bench  # noqa: E402
from src.ml.dataset import RAW, engineer  # noqa: E402
from src.ml.models import ResMLP, ActSeqGRU  # noqa: E402
from src.ml.axis_frame import (reconstruct_absolute, debiased_axis_metrics,  # noqa: E402
                               absolute_rz_metrics)
from src.ml.predictions import save_predictions  # noqa: E402
from src.ml.train import (train_save, train_m1_imas, train_m2_imas,  # noqa: E402
                          predict_dump_snapshot, predict_dump_seq, _device)

CFG = get_proj_config()

# M1/M2 hyperparameters (m0_model.yml carries M0 hp only); CLI-overridable.
M1_HP = dict(hidden=256, depth=4, dropout=0.1, lr=1e-3, epochs=80, patience=12)
M2_HP = dict(hidden=64, layers=2, dropout=0.1, lr=1e-3, epochs=80, patience=12)


def _load_m0_cfg():
    with open(CFG.base_dir / "configs" / "m0_model.yml") as f:
        return yaml.safe_load(f)


# ── M0 (+axis): predict + score ────────────────────────────────────────

def _pred_m0(art, h5_dir, npz_dir, shots, out):
    """Dump per-shot rho predictions for the M0 artifact and collect per-shot
    (rho, axis_r/z) + truth aligned to NPZ-valid & finite-feat slices, for the
    axis / absolute-(R,Z) metrics."""
    m0, keep = art["m0"], art["keep"]
    preds, extra = {}, {}
    for s in shots:
        f = pathlib.Path(h5_dir) / f"{int(s)}.h5"
        if not f.exists():
            continue
        X, vf = engineer(f)
        d = np.load(pathlib.Path(npz_dir) / f"{int(s)}.npz")
        Y = d["Y"].astype(float)
        v = (vf & d["valid"].astype(bool)
             & np.isfinite(Y).all(1) & np.isfinite(X).all(1))
        if not v.any():
            continue
        Xk = X[v][:, keep]
        rho = np.column_stack([m.predict(Xk) for m in m0]).astype(np.float32)
        preds[int(s)] = rho
        with h5py.File(f, "r") as h:
            th = np.asarray(h["lcfs_theta"], float)
            tar = np.asarray(h["magnetic_axis_r"], float)[v]
            taz = np.asarray(h["magnetic_axis_z"], float)[v]
        extra[int(s)] = {"rho": rho, "ar": art["axis_r"].predict(Xk),
                         "az": art["axis_z"].predict(Xk),
                         "yrho": Y[v], "tar": tar, "taz": taz, "theta": th}
    save_predictions(out, preds)
    return extra


def _axis_metrics(extra):
    """De-biased axis R^2/RMSE + absolute (R,Z) LCFS metrics, pooled over shots."""
    ar_p, ar_t, az_p, az_t, Rp, Zp, Rt, Zt = [], [], [], [], [], [], [], []
    for e in extra.values():
        th = e["theta"]
        R, Z = reconstruct_absolute(e["ar"], e["az"], e["rho"], th)
        Rtr, Ztr = reconstruct_absolute(e["tar"], e["taz"], e["yrho"], th)
        ar_p.append(e["ar"]); ar_t.append(e["tar"])
        az_p.append(e["az"]); az_t.append(e["taz"])
        Rp.append(R); Zp.append(Z); Rt.append(Rtr); Zt.append(Ztr)
    ar_p, ar_t = np.concatenate(ar_p), np.concatenate(ar_t)
    az_p, az_t = np.concatenate(az_p), np.concatenate(az_t)
    m_r = debiased_axis_metrics(ar_p, ar_t, ar_t.mean())
    m_z = debiased_axis_metrics(az_p, az_t, az_t.mean())
    Rp = np.concatenate(Rp); Zp = np.concatenate(Zp)
    Rt = np.concatenate(Rt); Zt = np.concatenate(Zt)
    absm = absolute_rz_metrics(Rp, Zp, Rt, Zt, Rt.mean(0), Zt.mean(0))
    return {"axis_r_r2": m_r["r2"], "axis_r_rmse_cm": m_r["rmse"] * 100,
            "axis_z_r2": m_z["r2"], "axis_z_rmse_cm": m_z["rmse"] * 100,
            "abs_r2_R": absm["r2_R"], "abs_r2_Z": absm["r2_Z"],
            "abs_boundary_rmse_cm": absm["boundary_rmse_cm"]}


# ── M1 / M2: load state + predict ──────────────────────────────────────

def _pred_m1(art_path, h5_dir, npz_dir, shots, out):
    a = torch.load(art_path, map_location="cpu", weights_only=False)
    hp = a["hp"]
    model = ResMLP(a["n_in"], hidden=hp["hidden"], depth=hp["depth"],
                   dropout=hp["dropout"]).to(_device()).eval()
    model.load_state_dict(a["state"])
    predict_dump_snapshot(model, h5_dir, npz_dir, shots,
                          np.asarray(a["mean"], float),
                          np.asarray(a["std"], float), out)


def _pred_m2(art_path, h5_dir, npz_dir, shots, out):
    a = torch.load(art_path, map_location="cpu", weights_only=False)
    hp = a["hp"]
    model = ActSeqGRU(n_act=a["n_act"], hidden=hp["hidden"], layers=hp["layers"],
                      dropout=hp["dropout"]).to(_device()).eval()
    model.load_state_dict(a["state"])
    predict_dump_seq(model, h5_dir, npz_dir, shots, RAW,
                     np.asarray(a["mean"], float),
                     np.maximum(np.asarray(a["std"], float), 1e-6), out)


def _score(npz_dir, pred_path, train_shots, test_shots):
    m = bench.score_predictions(pred_path, str(npz_dir), train_shots, test_shots)
    psc = m.get("per_shot_ccc") or [np.nan]
    return {"ccc": m.get("ccc", float("nan")), "r2": m.get("r2", float("nan")),
            "similarity": m.get("similarity", float("nan")),
            "rmse_cm": m.get("rmse_cm", float("nan")),
            "ccc_p90": float(np.nanpercentile(psc, 90)),
            "n_shots": m.get("n_shots", 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=["m0", "m1", "m2", "all"])
    ap.add_argument("--run", default="imas_actuator")
    ap.add_argument("--max-shots", type=int, default=None,
                    help="subset the train pool (smoke test)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bench-out", default=None)
    args = ap.parse_args()

    h5_dir, npz_dir = CFG.imas_h5_dir, CFG.imas_npz_dir
    m0cfg = _load_m0_cfg()
    m1_hp, m2_hp = dict(M1_HP), dict(M2_HP)
    if args.epochs is not None:
        m1_hp["epochs"] = m2_hp["epochs"] = args.epochs

    train_full, _val, test = bench.load_filtered_split(npz_dir)
    smoke = sorted(train_full[:args.max_shots]) if args.max_shots else None
    train_for_score = train_full  # CCC floor uses the full train mean

    models = ["m0", "m1", "m2"] if args.model == "all" else [args.model]
    run_dir = CFG.trains_dir / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mdl in models:
        art = run_dir / f"{mdl}.{'joblib' if mdl == 'm0' else 'pt'}"
        pred = run_dir / f"{mdl}_pred.npz"
        if mdl == "m0":
            train_save(h5_dir, npz_dir, art, hp=m0cfg["hp"], shots=smoke)
            extra = _pred_m0(joblib.load(art), h5_dir, npz_dir, test, pred)
            sc = _score(npz_dir, pred, train_for_score, test)
            sc.update(_axis_metrics(extra))
        elif mdl == "m1":
            train_m1_imas(h5_dir, npz_dir, art, m1_hp, shots=smoke)
            _pred_m1(art, h5_dir, npz_dir, test, pred)
            sc = _score(npz_dir, pred, train_for_score, test)
        else:
            train_m2_imas(h5_dir, npz_dir, art, m2_hp, RAW, shots=smoke)
            _pred_m2(art, h5_dir, npz_dir, test, pred)
            sc = _score(npz_dir, pred, train_for_score, test)
        rows.append({"model": mdl, **sc})
        print(f"{mdl}: CCC={sc['ccc']:.4f} R2={sc['r2']:.4f} "
              f"RMSE={sc['rmse_cm']:.2f}cm ccc_p90={sc['ccc_p90']:.4f} "
              f"n_shots={sc['n_shots']}"
              + (f" | axisR2(R/Z)={sc.get('axis_r_r2', float('nan')):.3f}/"
                 f"{sc.get('axis_z_r2', float('nan')):.3f} "
                 f"absBndy={sc.get('abs_boundary_rmse_cm', float('nan')):.2f}cm"
                 if mdl == "m0" else ""))

    out = (pathlib.Path(args.bench_out) if args.bench_out
           else CFG.stats_dir / "imas_predictor" / "bench_table.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "ccc", "r2", "similarity", "rmse_cm", "ccc_p90", "n_shots",
              "axis_r_r2", "axis_z_r2", "abs_boundary_rmse_cm"]
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if out.stat().st_size == 0:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
