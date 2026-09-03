# -*- coding: utf-8 -*-
"""Exploratory dense-angle (n_rho) training entry — NOT the frozen driver.

Reuses the frozen pf-context training components verbatim
(build_context_loaders / run_epochs / wrap_ddp) but none of the
publication identity machinery: validate_context_config pins the
7-context x 5-seed publication matrix and is deliberately not called.
Writes one plain artifact per (context, seed) with exploratory=True.

The driver writes ONLY under --out-root. Normalization is an n_rho-aware
copy of the sweep script's context_normalization (the frozen script must
not change), so the per-shot radii derivation cost shows up in both the
normalization pass and dataset construction — both are timed and printed
(spec 2.4 measurement).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ml.models import ActSeqAttn  # noqa: E402
from src.ml.pf_context import context_level  # noqa: E402
from src.ml.pf_observability import sha256_file  # noqa: E402
from src.ml.pfctx_data import load_context_series  # noqa: E402
from src.ml.pfctx_train import (  # noqa: E402
    build_context_loaders, microbatch_contract, run_epochs, wrap_ddp,
)
from src.ml.pfobs_train import init_dist, teardown_dist  # noqa: E402
from src.ml.target import n_out_for  # noqa: E402


def normalization(target_dir, sidecar_dir, shots, n_rho):
    """n_rho-aware copy of the sweep's context_normalization (train rows)."""
    feature_rows, target_rows = [], []
    for shot in sorted(int(s) for s in shots):
        series = load_context_series(
            pathlib.Path(target_dir) / f"{shot}.npz", sidecar_dir, n_rho=n_rho)
        feature_rows.append(series.features[series.score_valid])
        target_rows.append(series.target[series.score_valid])
    if not any(len(rows) for rows in feature_rows):
        raise ValueError("no score-valid training rows for normalization")
    X, Y = np.concatenate(feature_rows), np.concatenate(target_rows)

    def pair(block):
        return (block.mean(0).astype(np.float32),
                np.maximum(block.std(0), 1e-6).astype(np.float32))

    return (*pair(X), *pair(Y))


def load_split(path, target_dir, sidecar_dir):
    """Minimal split reader for the exploratory run (roles + exclusions).

    Both real manifests (configs/splits/pfobs_random_pilot.json and
    configs/splits/communications_physics_campaign_v1.json) carry the role
    lists at the top level (train/validation/test); the "roles" nesting is
    accepted too for hand-written test manifests. The campaign manifest's
    ``excluded`` shots appear in no role list, so intersecting the roles
    with the shots present in BOTH datasets is the only filtering needed.
    """
    manifest = json.loads(pathlib.Path(path).read_text())
    roles = manifest["roles"] if "roles" in manifest else manifest
    available = ({p.stem for p in pathlib.Path(target_dir).glob("*.npz")}
                 & {p.stem for p in pathlib.Path(sidecar_dir).glob("*.npz")})

    def keep(role):
        wanted = [int(x) for x in roles[role]]
        kept = [x for x in wanted if str(x) in available]
        if len(kept) != len(wanted):
            print(f"split {pathlib.Path(path).name}: {role} drops "
                  f"{len(wanted) - len(kept)} of {len(wanted)} shots not "
                  "present in both datasets")
        return kept

    class _Split:
        pass

    s = _Split()
    s.train = keep("train")
    s.validation = keep("validation")
    s.test = [int(x) for x in roles.get("test", []) if str(x) in available]
    s.split_path = pathlib.Path(path)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--target-dir",
                    default=str(REPO_ROOT / "ProjDB/datasets/NpzGeom"))
    ap.add_argument("--sidecar-dir",
                    default=str(REPO_ROOT / "ProjDB/datasets/NpzGeomPFObs"))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--run-prefix", default="a64")
    ap.add_argument("--contexts", nargs="+", default=["h0512"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-rho", type=int, default=None,
                    help="default: the config's n_rho")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train-shots", type=int, default=None)
    ap.add_argument("--max-validation-shots", type=int, default=None)
    args = ap.parse_args(argv)

    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    if not config.get("exploratory"):
        raise SystemExit("refusing to run: config lacks exploratory: true")
    n_rho = int(args.n_rho or config["n_rho"])
    hp = dict(config["hp"])
    if args.epochs is not None:
        hp["epochs"] = int(args.epochs)
    split = load_split(args.split, args.target_dir, args.sidecar_dir)
    train = (split.train[:args.max_train_shots]
             if args.max_train_shots else split.train)
    validation = (split.validation[:args.max_validation_shots]
                  if args.max_validation_shots else split.validation)
    if not train or not validation:
        raise SystemExit("refusing to run: empty train or validation role "
                         "after availability filtering")

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dist_env = init_dist()
    try:
        t0 = time.perf_counter()
        stats = normalization(args.target_dir, args.sidecar_dir, train, n_rho)
        print(f"normalization over {len(train)} shots: "
              f"{time.perf_counter() - t0:.1f}s "
              f"(n_rho={n_rho} derivation included)")
        for label in args.contexts:
            context = context_level(label)
            for seed in args.seeds:
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
                accumulation, effective = microbatch_contract(
                    hp["effective_global_batch"], dist_env.world_size,
                    hp["microbatch_per_rank"])
                t1 = time.perf_counter()
                train_loader, validation_loader = build_context_loaders(
                    args.target_dir, args.sidecar_dir, train, validation,
                    context, stats[:2], stats[2:], hp["microbatch_per_rank"],
                    int(seed), dist_env, n_rho=n_rho)
                print(f"{label} s{seed}: dataset construction "
                      f"{time.perf_counter() - t1:.1f}s "
                      f"({len(train_loader.dataset)} train / "
                      f"{len(validation_loader.dataset)} validation "
                      "windows)")
                model = ActSeqAttn(
                    n_act=config["input_width"], n_out=n_out_for(n_rho),
                    d=hp["d_model"], heads=hp["heads"], depth=hp["depth"],
                    ffn=hp["ffn"], dropout=hp["dropout"],
                    pe="rope_time").to(dist_env.device)
                ddp_model = wrap_ddp(model, dist_env)
                optimizer = torch.optim.AdamW(
                    ddp_model.parameters(), lr=hp["lr"], weight_decay=1e-5)
                t2 = time.perf_counter()
                result = run_epochs(
                    ddp_model, optimizer, train_loader, validation_loader,
                    context, int(seed), hp, accumulation, dist_env)
                print(f"{label} s{seed}: {result['epochs_completed']} epochs "
                      f"in {time.perf_counter() - t2:.1f}s "
                      f"(microbatch {hp['microbatch_per_rank']}, "
                      f"accumulation {accumulation}, effective "
                      f"{effective})")
                if dist_env.is_main:
                    run_dir = out_root / f"{args.run_prefix}_{label}_s{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    artifact = {
                        "study": "a64_exploratory", "exploratory": True,
                        "n_act": config["input_width"],
                        "n_out": n_out_for(n_rho), "n_rho": n_rho,
                        "depth": int(hp["depth"]),
                        "hp": dict(hp),
                        "context_label": label, "seed": int(seed),
                        "state": model.state_dict(),
                        "feature_mean": stats[0], "feature_std": stats[1],
                        "target_mean": stats[2], "target_std": stats[3],
                        "best_val_mse": float(result["best_val_mse"]),
                        "best_epoch": int(result["best_epoch"]),
                        "stop_epoch": int(result["stop_epoch"]),
                        "epochs_completed": int(result["epochs_completed"]),
                        "config_sha256": sha256_file(args.config),
                        "split_sha256": sha256_file(args.split),
                    }
                    torch.save(artifact, run_dir / "m3.pt")
                    print(f"saved {run_dir / 'm3.pt'} "
                          f"best_val_mse={result['best_val_mse']:.6f}")
    finally:
        teardown_dist(dist_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
