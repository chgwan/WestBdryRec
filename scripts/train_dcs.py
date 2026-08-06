# -*- coding: utf-8 -*-
"""Train DCS strict-actuator LCFS models and benchmark pooled CCC on test.

    python scripts/train_dcs.py m0
    python scripts/train_dcs.py m1 --epochs 50
    python scripts/train_dcs.py all
"""
import argparse, csv, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import joblib  # noqa: E402
from src.proj_config import get_proj_config  # noqa: E402
from src.ml import bench  # noqa: E402
from src.ml.dcs_features import (load_dcs_config, load_meta, node_col_map,  # noqa: E402
                                 read_snapshot, read_series)
from src.ml.models import ResMLP, ActSeqGRU  # noqa: E402
from src.ml.predictions import save_predictions  # noqa: E402
from src.ml.score34 import score_dcs34  # noqa: E402
from src.ml.train import (train_m0_dcs, train_m1_dcs, train_m2_dcs, _device)  # noqa: E402
from src.ml.target import N_OUT, destandardize  # noqa: E402
from src.ml.target import load_target  # noqa: E402

CFG = get_proj_config()


def _meta_ncm(npz_dir):
    meta = load_meta(npz_dir)
    return meta, node_col_map(meta)


def _pred_m0(art, npz_dir, shots, cfg, ncm, out):
    a = joblib.load(art); m0, keep = a["m0"], a["keep"]
    tgt_mean, tgt_std = a["tgt_mean"], a["tgt_std"]
    preds = {}
    for s in shots:
        p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
        if not p.exists():
            continue
        feats, mask = read_snapshot(p, cfg, ncm)
        T_all, finite = load_target(p)
        v = mask & finite & np.isfinite(feats).all(1)
        if v.any():
            Xk = feats[v][:, keep]
            z = np.column_stack([m.predict(Xk) for m in m0])
            preds[int(s)] = destandardize(z, tgt_mean, tgt_std).astype(np.float32)
    save_predictions(out, preds)


def _pred_m1(art, npz_dir, shots, cfg, ncm, out):
    a = torch.load(art, map_location="cpu", weights_only=False)
    keep, hp = a["keep"], a["hp"]
    model = ResMLP(a["n_in"], n_out=a.get("n_out", N_OUT), hidden=hp["hidden"],
                   depth=hp["depth"], dropout=hp["dropout"]).to(_device()).eval()
    model.load_state_dict(a["state"])
    mean = np.asarray(a["mean"], float)[keep]; std = np.maximum(np.asarray(a["std"], float)[keep], 1e-6)
    preds = {}
    with torch.no_grad():
        for s in shots:
            p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
            if not p.exists():
                continue
            feats, mask = read_snapshot(p, cfg, ncm)
            T_all, finite = load_target(p)
            v = mask & finite & np.isfinite(feats).all(1)
            if v.any():
                Xk = ((feats[v][:, keep] - mean) / std).astype(np.float32)
                z = model(torch.from_numpy(Xk).to(_device())).cpu().numpy()
                preds[int(s)] = destandardize(z, a["tgt_mean"], a["tgt_std"]).astype(np.float32)
    save_predictions(out, preds)


def _pred_m2(art, npz_dir, shots, cfg, ncm, out):
    a = torch.load(art, map_location="cpu", weights_only=False)
    hp = a["hp"]; mean = np.asarray(a["mean"], float); std = np.maximum(np.asarray(a["std"], float), 1e-6)
    model = ActSeqGRU(n_act=a["n_act"], n_out=a.get("n_out", N_OUT), hidden=hp["hidden"],
                      layers=hp["layers"], dropout=hp["dropout"]).to(_device()).eval()
    model.load_state_dict(a["state"])
    preds = {}
    with torch.no_grad():
        for s in shots:
            p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
            if not p.exists():
                continue
            A, mask = read_series(p, cfg, ncm)
            T_all, finite = load_target(p)
            v = mask & finite
            if v.any():
                X = np.ascontiguousarray(((A - mean) / std)[None]).astype(np.float32)
                mk = v[None]
                Xt = torch.from_numpy(X).to(_device())
                mk_t = torch.from_numpy(mk).to(_device())
                try:
                    yhat = model(Xt, mk_t)
                except RuntimeError:  # cuDNN has no eval-mode kernel for this long sequence
                    torch.backends.cudnn.enabled = False
                    yhat = model(Xt, mk_t)
                    torch.backends.cudnn.enabled = True
                z = yhat[0].cpu().numpy()[v]
                preds[int(s)] = destandardize(z, a["tgt_mean"], a["tgt_std"]).astype(np.float32)
    save_predictions(out, preds)


def _score(npz_dir, pred_path, train_shots, test_shots):
    meta = load_meta(npz_dir)
    theta = np.deg2rad(np.asarray(meta["theta_deg"], float))
    m = score_dcs34(pred_path, str(npz_dir), train_shots, test_shots, theta)
    if m["n_shots"] == 0:
        raise SystemExit(f"{pred_path}: scored 0 shots -- refusing to write a bench row")
    return {"ccc": m["ccc"], "r2": m["r2"], "similarity": m["similarity"],
            "rmse_cm": m["rmse_cm"], "ccc_p90": m["ccc_p90"], "n_shots": m["n_shots"],
            "rgeom_mae_mm": m["rgeom_mae_mm"], "zgeom_mae_mm": m["zgeom_mae_mm"],
            "centre_rmse_mm": m["centre_rmse_mm"],
            "abs_bnd_rmse_mm": m["abs_bnd_rmse_mm"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["m0", "m1", "m2", "all"])
    ap.add_argument("--shots", type=int, nargs="*", default=None, help="override train pool (tests)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bench-out", default=None)
    ap.add_argument("--npz-dir", default=None,
                    help="dataset to train on (default: cfg.mergednpz_dir); "
                         "e.g. ProjDB/datasets/NpzGeom for the filtered rebuild")
    ap.add_argument("--run-name", default=None,
                    help="artifact subdir under trains/ (default: cfg['run'])")
    ap.add_argument("--config", default=None,
                    help="model config (default: configs/dcs_model.yml); "
                         "e.g. configs/dcs_model_geom.yml adds the time-PE channels")
    args = ap.parse_args()

    npz_dir = pathlib.Path(args.npz_dir) if args.npz_dir else CFG.mergednpz_dir
    cfg = load_dcs_config(args.config)
    if args.epochs is not None:
        for k in ("m1", "m2"):
            cfg["hp"][k]["epochs"] = args.epochs
    _, ncm = _meta_ncm(npz_dir)
    if args.shots is not None:
        train = test = sorted(args.shots)
    else:
        train, _val, test = bench.load_filtered_split(npz_dir)

    models = ["m0", "m1", "m2"] if args.model == "all" else [args.model]
    run_dir = CFG.trains_dir / (args.run_name or cfg["run"])
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mdl in models:
        art = run_dir / f"{mdl}.{'joblib' if mdl == 'm0' else 'pt'}"
        pred = run_dir / f"{mdl}_pred.npz"
        if mdl == "m0":
            train_m0_dcs(npz_dir, art, cfg=cfg, shots=args.shots); _pred_m0(art, npz_dir, test, cfg, ncm, pred)
        elif mdl == "m1":
            train_m1_dcs(npz_dir, art, cfg=cfg, shots=args.shots); _pred_m1(art, npz_dir, test, cfg, ncm, pred)
        else:
            train_m2_dcs(npz_dir, art, cfg=cfg, shots=args.shots); _pred_m2(art, npz_dir, test, cfg, ncm, pred)
        sc = _score(npz_dir, pred, train, test)
        rows.append({"model": mdl, **sc})
        print(f"{mdl}: CCC={sc['ccc']:.4f} R2={sc['r2']:.4f} RMSE={sc['rmse_cm']:.2f}cm n_shots={sc['n_shots']}")

    out = pathlib.Path(args.bench_out) if args.bench_out else (CFG.stats_dir / "dcs_predictor" / "bench_table.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "ccc", "r2", "similarity", "rmse_cm",
                                          "ccc_p90", "n_shots", "rgeom_mae_mm",
                                          "zgeom_mae_mm", "centre_rmse_mm",
                                          "abs_bnd_rmse_mm"])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
