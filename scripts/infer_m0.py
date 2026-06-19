# scripts/infer_m0.py
# -*- coding: utf-8 -*-
"""Load the M0 inference artifact and predict a shot's LCFS (shape + absolute),
writing results to ProjDB/inferences/<run>/. Requires train_m0.py to have run.

    conda run -n torch python scripts/infer_m0.py --shot 57821 [--run m0_actuator]
"""
import argparse
import pathlib
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.proj_config import get_proj_config          # noqa: E402
from src.ml import infer             # noqa: E402

IMAS = pathlib.Path("/zhisongqu_data/chgwan/DataBase/WEST/IMAS")


def main():
    cfg = get_proj_config()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--run", default="m0_actuator")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--h5-dir", default=None)
    args = ap.parse_args()
    h5_dir = pathlib.Path(args.h5_dir) if args.h5_dir else IMAS
    artifact_path = (pathlib.Path(args.artifact) if args.artifact
                     else cfg.trains_dir / args.run / "m0_inference.joblib")
    art = infer.load(artifact_path)

    res = infer.predict_shot(args.shot, art, h5_dir)
    out_dir = cfg.inferences_dir / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    if res is None:
        print(f"shot {args.shot}: no valid slices (missing h5?)")
        return
    np.savez(out_dir / f"{args.shot}.npz", **res)

    # true LCFS for comparison (flat-top mean)
    with h5py.File(h5_dir / f"{args.shot}.h5", "r") as h:
        th = np.asarray(h["lcfs_theta"], float)
        true_rho = np.asarray(h["lcfs_rho"], float)
        tar = np.asarray(h["magnetic_axis_r"], float)
        taz = np.asarray(h["magnetic_axis_z"], float)
    vi = res["valid_idx"]
    deg = np.degrees(th)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(deg, true_rho[vi].mean(0), "k-", lw=2, label="true")
    ax[0].plot(deg, res["lcfs_rho"].mean(0), "r--", label="pred")
    ax[0].set_title("shape r(theta) [flat-top mean]"); ax[0].set_xlabel("theta (deg)"); ax[0].legend()

    def _close(r, z):
        return np.append(r, r[0]), np.append(z, z[0])
    Rtp = tar[vi].mean() + true_rho[vi].mean(0) * np.cos(th)
    Ztp = taz[vi].mean() + true_rho[vi].mean(0) * np.sin(th)
    Rt, Zt = _close(Rtp, Ztp)
    Rp, Zp = _close(res["R"].mean(0), res["Z"].mean(0))
    ax[1].plot(Rt, Zt, "k-", lw=2, label="true")
    ax[1].plot(Rp, Zp, "r--", label="pred")
    ax[1].set_title("absolute LCFS (R,Z) [flat-top mean]"); ax[1].set_aspect("equal"); ax[1].legend()
    fig.suptitle(f"shot {args.shot} (run {args.run})")
    fig.tight_layout(); fig.savefig(out_dir / f"{args.shot}.png", dpi=130); plt.close(fig)
    print(f"wrote {out_dir / f'{args.shot}.npz'} + {out_dir / f'{args.shot}.png'}")


if __name__ == "__main__":
    main()
