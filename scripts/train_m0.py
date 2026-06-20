# scripts/train_m0.py
# -*- coding: utf-8 -*-
"""Train M0 (shape) + the axis model and save the inference artifact to
ProjDB/trains/<run>/. Run once (or to retrain under a new --run).

    conda run -n torch python scripts/train_m0.py [--run m0_actuator]
"""
import argparse
import json
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.proj_config import get_proj_config          # noqa: E402
from src.ml import train             # noqa: E402


def main():
    cfg = get_proj_config()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(cfg.base_dir / "configs" / "m0_model.yml"))
    ap.add_argument("--run", default=None,
                    help="override the run name from config")
    ap.add_argument("--h5-dir", default=None)
    ap.add_argument("--npz-dir", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    run = args.run or model_cfg["run"]
    hp = model_cfg["hp"]

    h5_dir = pathlib.Path(args.h5_dir) if args.h5_dir else cfg.imas_h5_dir
    npz_dir = pathlib.Path(args.npz_dir) if args.npz_dir else cfg.imas_npz_dir

    run_dir = cfg.trains_dir / run
    out = run_dir / "m0_inference.joblib"
    meta = train.train_save(h5_dir, npz_dir, out, hp=hp)
    meta = {"run": run, "config": args.config, **meta}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved artifact -> {out}")
    print(f"train pooled R2 (shape) = {meta['train_r2']:.4f}  "
          f"n_train={meta['n_train']}  features={meta['n_features']}/{meta['n_angles']} angles")


if __name__ == "__main__":
    main()
