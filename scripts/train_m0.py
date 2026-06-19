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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.proj_config import get_proj_config          # noqa: E402
from src.actuator_predictor import infer             # noqa: E402

IMAS = pathlib.Path("/zhisongqu_data/chgwan/DataBase/WEST/IMAS")


def main():
    cfg = get_proj_config()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="m0_actuator")
    ap.add_argument("--h5-dir", default=None)
    ap.add_argument("--npz-dir", default=None)
    args = ap.parse_args()
    h5_dir = pathlib.Path(args.h5_dir) if args.h5_dir else IMAS
    npz_dir = pathlib.Path(args.npz_dir) if args.npz_dir else cfg.imas_npz_dir

    run_dir = cfg.trains_dir / args.run
    out = run_dir / "m0_inference.joblib"
    meta = infer.train_save(h5_dir, npz_dir, out)
    meta = {"run": args.run, **meta}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved artifact -> {out}")
    print(f"train pooled R2 (shape) = {meta['train_r2']:.4f}  "
          f"n_train={meta['n_train']}  features={meta['n_features']}/{meta['n_angles']} angles")


if __name__ == "__main__":
    main()
