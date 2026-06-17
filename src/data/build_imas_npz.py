# -*- coding: utf-8 -*-
"""Build per-shot IMAS-native NPZ arrays for the input-observability sweep.

Reads each ``IMAS/<shot>.h5`` (all datasets on ``time`` == absolute IMAS seconds)
and writes ``ProjDB/Npz/imas/<shot>.npz`` with:
  X (nt, F) float32  -- every tier channel concatenated (layout in meta.json)
  Y (nt, 32) float32 -- lcfs_rho (the precomputed boundary radius profile)
  time (nt) float32
  valid (nt) bool

``valid`` requires finite Y and a sustained flat-top window (from ``ip``); if
flat-top detection fails the finite-Y rows are kept as a fallback. Finiteness
over X is deliberately NOT enforced here -- the sweep's ``pool()`` checks
finiteness over the *selected* columns per sweep point, which is the correct
granularity (a shot missing ``lh_power`` still contributes to a T2-only sweep).
The target ``lcfs_rho`` is the already-precomputed r(theta)@32 profile -- it is
NOT reprojected here.

Canonical column layout:
  Every shot's ``X`` has the SAME columns in the SAME order, driven by the
  config (``cfg["tiers"]`` dict order -> groups dict order -> each group's named
  datasets). Absent/Empty/misaligned channels become **NaN columns** (NOT
  dropped), so ``d["X"][:, cols]`` is valid for every shot regardless of which
  optional channels a shot happens to carry. The canonical layout is recorded
  once in ``meta.json["inputs"]``.

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


def _group_names(spec):
    """Canonical channel list for a group (``dataset`` is a name or list of names)."""
    d = spec["dataset"]
    return list(d) if isinstance(d, list) else [d]


def _group_block(hf, spec, nt):
    """``(nt, n_chan)`` block for one group, aligned to the canonical channel
    order. Absent/Empty/misaligned channels become NaN columns (NOT dropped) so
    every shot's ``X`` has the same columns in the same order."""
    names = _group_names(spec)
    cols = [_read_ds(hf, nm, nt) if nm in hf else None for nm in names]
    cols = [np.full((nt, 1), np.nan, np.float32) if c is None else c for c in cols]
    return np.concatenate(cols, axis=1)


def canonical_layout(cfg):
    """Canonical column layout driven by ``cfg["tiers"]`` (dict order: tiers,
    then groups, then each group's named datasets). Same for every shot.

    Returns a list of ``{tier, group, names, cols, n_chan}`` with contiguous
    ``cols`` ranges; absent/Empty channels become NaN columns at build time."""
    layout, start = [], 0
    for tier, groups in cfg["tiers"].items():
        if not groups:
            continue
        for gname, spec in groups.items():
            names = _group_names(spec)
            k = len(names)
            layout.append({"tier": tier, "group": gname, "names": list(names),
                           "cols": [start, start + k], "n_chan": k})
            start += k
    return layout, start


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

            # --- features: canonical NaN-padded column layout (same for every shot) ---
            layout, f_total = canonical_layout(cfg)
            X = np.full((nt, f_total), np.nan, np.float32)
            for g in layout:
                block = _group_block(hf, cfg["tiers"][g["tier"]][g["group"]], nt)
                X[:, g["cols"][0]:g["cols"][1]] = block.astype(np.float32)

            # --- validity: finite Y; AND flat-top window only if detection succeeds.
            # NOTE: finiteness over X is NOT enforced here -- the sweep's pool()
            # checks finiteness over the *selected* columns per sweep point, which
            # is the correct granularity (a shot missing lh_power should still
            # contribute to a T2-only sweep). If flat-top detection returns None,
            # keep finite-Y rows as a fallback so the shot isn't dropped outright. ---
            valid = np.isfinite(Y).all(axis=1)
            ip = _read_ds(hf, "ip", nt)
            if ip is not None:
                win = flat_top_window(t, ip[:, 0])
                if win is not None:
                    m = np.zeros(nt, bool)
                    m[win[0]:win[1] + 1] = True
                    valid &= m

        if not valid.any():
            return shot, False, "no valid slices"
        out.parent.mkdir(parents=True, exist_ok=True)
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

    # Canonical layout is identical for every shot (NaN-padded absent channels),
    # so derive it from the config directly rather than any single shot's payload.
    layout, _ = canonical_layout(yml)
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
