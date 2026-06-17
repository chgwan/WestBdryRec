# -*- coding: utf-8 -*-
"""Build per-shot IMAS-native NPZ arrays for the input-observability sweep.

Reads each ``IMAS/<shot>.h5`` (all datasets on ``time`` == absolute IMAS seconds)
and writes ``ProjDB/Npz/imas/<shot>.npz`` with:
  X (nt, F) float32  -- every tier channel concatenated (layout in meta.json)
  Y (nt, 32) float32 -- lcfs_rho (the precomputed boundary radius profile)
  time (nt) float32
  valid (nt) bool

``valid`` requires finite Y, finite X, and a sustained flat-top (from ``ip``).
The target ``lcfs_rho`` is the already-precomputed r(theta)@32 profile -- it is
NOT reprojected here.

Real-schema robustness:
  * ``h5py.Empty`` datasets (e.g. ``power_additional``) and absent datasets
    (e.g. ``lh_power_launched_total`` on some shots) are dropped per-channel via
    ``_read_ds`` returning ``None``; a whole group empty -> that group is dropped.
  * Non-finite rows (e.g. ``tau_energy_98`` NaNs) are excluded from ``valid``.

Importable as a package module.
"""
import json
import pathlib
import shutil

import h5py
import numpy as np
import yaml

from ..proj_config import get_proj_config
from ..utils import pmap
from .imas_flat_top import flat_top_window


def load_imas_config(path=None):
    """Load the IMAS tier/group contract; default = ``configs/imas_inputs.yml``."""
    cfg = get_proj_config()
    path = pathlib.Path(path) if path else cfg.base_dir / "configs" / "imas_inputs.yml"
    return yaml.safe_load(path.read_text())


def _read_ds(hf, name, nt):
    """Dataset ``name`` as ``(nt, nch)`` float; 1-D -> (nt, 1); Empty/absent -> None.

    Also returns ``None`` when the leading axis does not match ``nt`` (the h5's
    ``time`` length), so a misaligned/stranger dataset is dropped rather than
    crashing the concat."""
    if name not in hf:
        return None
    ds = hf[name]
    if getattr(ds, "shape", None) is None:          # h5py.Empty
        return None
    a = np.asarray(ds, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[0] != nt:
        return None
    return a


def _group_block(hf, spec, nt):
    """``(nt, k)`` block for one group; channels with absent/Empty datasets dropped.

    Returns ``None`` only if every channel is absent (the whole group is dropped)."""
    names = spec["dataset"] if isinstance(spec["dataset"], list) else [spec["dataset"]]
    blocks = [_read_ds(hf, nm, nt) for nm in names]
    blocks = [b for b in blocks if b is not None]
    return np.concatenate(blocks, axis=1) if blocks else None


def build_one(shot, imas_dir, npz_dir, cfg):
    """Build one NPZ. Returns ``(shot, ok, payload | err_str)``.

    On failure any partial output file is removed so a downstream glob never
    picks up a half-written NPZ."""
    src = pathlib.Path(imas_dir) / f"{int(shot)}.h5"
    out = pathlib.Path(npz_dir) / f"{int(shot)}.npz"
    try:
        with h5py.File(src, "r") as hf:
            if "time" not in hf:
                return shot, False, "no time"
            t = np.asarray(hf["time"], float)
            nt = t.size

            # --- target: lcfs_rho (already r(theta)@32, do NOT reproject) ---
            rho = _read_ds(hf, cfg["target"]["rho"], nt)
            if rho is None or rho.shape[1] != 32:
                return shot, False, "no lcfs_rho"
            Y = rho.astype(np.float32)

            # --- features: every tier group, in tier declaration order ---
            cols, layout, start = [], [], 0
            for tier, groups in cfg["tiers"].items():
                if not groups:                       # empty tier (e.g. T1) -> skip
                    continue
                for gname, spec in groups.items():
                    block = _group_block(hf, spec, nt)
                    if block is None:
                        continue
                    k = block.shape[1]
                    cols.append(block)
                    layout.append({"tier": tier, "group": gname,
                                   "cols": [start, start + k], "n_chan": k})
                    start += k
            X = (np.concatenate(cols, axis=1).astype(np.float32)
                 if cols else np.empty((nt, 0), np.float32))

            # --- validity: finite Y + finite X + flat-top window from ip ---
            valid = np.isfinite(Y).all(axis=1) & np.isfinite(X).all(axis=1)
            ip = _read_ds(hf, "ip", nt)
            if ip is not None:
                win = flat_top_window(t, ip[:, 0])
                if win is not None:
                    m = np.zeros(nt, bool)
                    m[win[0]:win[1] + 1] = True
                    valid &= m

        if not valid.any():
            return shot, False, "no valid slices"
        np.savez(out, X=X, Y=Y, time=t.astype(np.float32), valid=valid)
        return shot, True, {"shot": int(shot), "layout": layout,
                            "n_slices": int(nt), "n_valid": int(valid.sum())}
    except Exception as exc:  # noqa: BLE001
        if out.exists():
            out.unlink()
        return shot, False, str(exc)


def _build_worker(args):
    shot, imas_dir, npz_dir, cfg = args
    return build_one(shot, imas_dir, npz_dir, cfg)


def run(imas_dir=None, npz_dir=None, config_path=None, workers=1):
    """Build ``ProjDB/Npz/imas/<shot>.npz`` for every IMAS shot + write meta.json.

    ``npz_dir`` is recreated from scratch (rm -rf then mkdir) so each run is a
    clean snapshot. Returns ``(n_ok, n_fail)``."""
    cfg = get_proj_config()
    imas_dir = pathlib.Path(imas_dir) if imas_dir else cfg.imas_h5_dir
    npz_dir = pathlib.Path(npz_dir) if npz_dir else cfg.imas_npz_dir
    yml = load_imas_config(config_path)

    if npz_dir.exists():
        shutil.rmtree(npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)

    shots = sorted(int(p.stem) for p in imas_dir.glob("*.h5"))
    items = [(s, imas_dir, npz_dir, yml) for s in shots]
    results = pmap(_build_worker, items, workers, "build_imas_npz")
    ok = [p for _, good, p in results if good]
    fail = [(s, p) for s, good, p in results if not good]

    layout = ok[0]["layout"] if ok else []
    theta_deg = []
    # recover theta (degrees) from any successful shot for the record
    if ok:
        with h5py.File(imas_dir / f"{ok[0]['shot']}.h5", "r") as hf:
            if "lcfs_theta" in hf:
                theta_deg = np.degrees(np.asarray(hf["lcfs_theta"], float)).tolist()
    meta = {
        "theta_deg": theta_deg,
        "n_angles": 32,
        "inputs": layout,
        "shots": sorted([{k: d[k] for k in ("shot", "n_slices", "n_valid")} for d in ok],
                        key=lambda d: d["shot"]),
        "n_shots": len(ok),
    }
    (npz_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"done: {len(ok)} npz, {len(fail)} failed")
    for s, err in fail[:20]:
        print(f"  {s}: {err}")
    return len(ok), len(fail)
