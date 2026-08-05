# Data Lineage

How raw WEST acquisition data flows through this repo into the trainable
artifacts consumed by the LCFS reconstruction models. There are **two parallel
pipelines** — a WEST/DCS track and an IMAS-native track — both ending in a polar
`r(θ)` boundary target on a fixed 32-angle grid. Selection criteria (which shots
enter the dataset) are documented in [`README.md`](../README.md); this page is
about *what each stage reads, transforms, and writes*.

> **Directory note.** The WEST trainable dir on disk is `ProjDB/datasets/NpzOrigin`
> (formerly `MergedNpz`; the rename records that it uses coordinate origin
> `(2.5, 0)`). `src/proj_config.py:mergednpz_dir` still returns the old literal
> `"MergedNpz"`, so it currently points at a missing dir — see
> [§9 Directory & config map](#9-directory--config-map).

---

## 1. Overview

Both pipelines reconstruct the WEST last-closed-flux-surface (LCFS) from
control-system and diagnostic inputs:

- **WEST / DCS** — DCS scope archives (`.mat`) + GMAG equilibrium HDF5 (`.h5`),
  merged onto one ignitron-time grid, then encoded as polar radii about the fixed
  origin `(2.5, 0)` m. Target: `targets/GMAG_BND` → `Y = r(θ)@32`.
- **IMAS** — IMAS-native HDF5 export whose `lcfs_rho` is *already* the polar
  `r(θ)@32` profile (precomputed upstream); this repo only selects, flattens, and
  filters it.

Vessel/PFC wall geometry used for plotting comes from the public **ToFu** repo
(external); it is a visualization asset, not part of the training data.

---

## 2. Raw sources (external — not built by this repo)

| Source | Path | Contents | Time base |
|---|---|---|---|
| DCS archive | `ProjDB/datasets/DCSHeating/DCS_archive_<shot>.mat` | 30 `*_scope` nodes sharing `Ip_scope.time`; columns `0`=ref, `3`=actual | ignitron (t=0 @ ignitron) |
| GMAG equilibrium | `ProjDB/datasets/GMagH5/<shot>.h5` | `targets/GMAG_BND` + `inputs/GMAG_*` / `SMAG_*`, each with its own `<name>_time` | ignitron (t=0 @ ignitron)|
| IMAS export | `ProjDB/datasets/IMASH5/<shot>.h5` | `lcfs_rho` (50,32), `lcfs_theta`, `pf_*`, `ip`, heating powers, scalars | absolute IMAS seconds |
| Wall polygons | ToFu repo (`WEST-V0…V4`) | vessel / PFC R–Z contours | n/a |

**GMAG sampling-rate tiers** (from the time vectors; relevant to the `fs > 400 Hz`
quality gate): the GMAG/SMAG equilibrium suite runs at **488 Hz** (with ~9
anomalous shots at **30 Hz**), poloidal `GPOLO` at **977 Hz**, divertor `GDIV` at
**977 / 1953 Hz** (bimodal across the campaign), DCS scopes at ~1 kHz.

---

## 3. Lineage diagram

```mermaid
flowchart TD
    subgraph WEST["WEST / DCS pipeline  (ignitron time base)"]
        MAT["DCSHeating/*.mat<br/>DCS scopes (cols 0=ref, 3=actual)"]
        GH5["GMagH5/*.h5<br/>GMAG_BND + GMAG/SMAG inputs"]
        SEL["selection<br/>shot_status · scan_frequencies · flat_top<br/>→ Stats/shot_status.csv, flat_top.csv"]
        MH5["MergedH5/*.h5<br/>one shared 'time' grid<br/>(DCS native + GMAG resampled)"]
        WNPZ["NpzOrigin/*.npz + meta.json<br/>X(22) · Y=r(θ)@32 · S · bnd_RZ · valid"]
        MAT --> SEL
        GH5 --> SEL
        MAT --> MH5
        GH5 --> MH5
        SEL -. selects .-> MH5
        MH5 --> WNPZ
    end
    subgraph IMAS["IMAS-native pipeline  (absolute IMAS time base)"]
        IH5["IMASH5/*.h5<br/>lcfs_rho precomputed r(θ)@32<br/>pf / tf / heating / scalars"]
        INPZ["IMASNpz/*.npz + meta.json<br/>X (tiers T0/T2) · Y=lcfs_rho · valid<br/>(flat-top filter inside)"]
        IH5 --> INPZ
    end
    WNPZ --> Tdcs["train_dcs · infer"]
    INPZ --> Timas["train_imas · train_m0 · infer_m0"]
```

---

## 4. WEST / DCS stages

Driven in order by `scripts/run_data_pre.py`:

| # | Stage | Script / module | Reads | Writes | Key operation |
|---|---|---|---|---|---|
| 1 | status + freq scan | `src/data/shot_status.py` | `GMagH5/*.h5`, DCS `.mat` | `Stats/shot_status.csv`, `Stats/shot_freq_stats.csv` | per-shot quality categories + per-signal `fs_hz` |
| 2 | freq summary | `src/data/scan_frequencies.py` | `shot_freq_stats.csv` | `Stats/signal_freq_summary.csv` | per-signal fs aggregate + DCS `.mat` time-axis check |
| 3 | flat-top | `src/data/flat_top.py` | DCS Ip ref/actual | `Stats/flat_top.csv` | sustained Ip flat-top detection (selection criterion) |
| 4 | merge | `src/data/merge_dcs_bdry.py` | `.mat` + `GMagH5/*.h5` + selection CSVs | `MergedH5/<shot>.h5` | resample onto shared grid |
| 5 | build NPZ | `src/data/build_npz.py` | `MergedH5/*.h5`, `configs/base.yml` | `NpzOrigin/<shot>.npz` + `meta.json` | Cartesian → polar target |

**Merge (stage 4).** For shots passing the quality masks (nonzero bnd, `fs > 400 Hz`,
DCS ok) **and** total Ip flat-top `> 3.0 s`, the shared grid is the DCS ignitron
time clipped to the `GMAG_BND` span. DCS scopes are kept native (sliced to the
window); every GMAG input/target is linearly resampled onto that grid (no
extrapolation → NaN outside). Output has **one shared `time`** axis — no
per-signal `_time` datasets. → **759 shots**.

**Build NPZ (stage 5) — the Cartesian → polar target transform.** This is the
representation change that gives `NpzOrigin` its name:

1. Read `targets/GMAG_BND`, shape `(64, N)` interleaved `[R0, Z0, R1, Z1, …, R31, Z31]`.
2. `discharge_window` — keep the inclusive span of physically valid slices
   (finite, `R ∈ (1.8, 3.3)`, `R-span > 0.3 m`, `|Z| < 1.2 m`).
3. For each slice, reshape to `(32, 2)`, compute
   `θ = atan2(Z − Z0, R − R0) mod 2π` and `r = hypot(R − R0, Z − Z0)` about the
   fixed origin **`(2.5, 0)` m**, then periodic-interpolate `r` onto a uniform
   **32-angle grid** (`θ = 0` outboard / +R, CCW). Result: `Y = r(θ)@32`.
4. `X` — 22 input channels from `configs/base.yml:data.input_list`, read from
   `dcs/<scope>/<actual|ref>` (`_real` → actual, `_ref` → ref), with an
   `inputs/<node>` fallback for MDS+ signals. Layout:
   `LHW_real`(4) · `ICRH_phase_real`(3) · `ICRH_real`(3) · `Ip_ref`(1) ·
   `Ne_real`(1) · `PF_real`(10).
5. `S` — equilibrium scalar **labels** (targets only, never inputs):
   `beli = GMAG_SHAF[1]·1e-3`, `li = GMAG_BELI[5]·1e-3`.
6. `meta.json` records `origin`, `theta_deg`, the X layout, and dataset-wide
   normalization stats (`X/Y/S` mean & std over valid rows).

→ **758 npz** (one `MergedH5` shot dropped: no discharge window / zero valid slices).

---

## 5. IMAS stages

| Stage | Module | Reads | Writes | Key operation |
|---|---|---|---|---|
| build NPZ | `src/data/build_imas_npz.py` | `IMASH5/*.h5`, `configs/imas_inputs.yml` | `IMASNpz/<shot>.npz` + `meta.json` | flatten tiers + flat-top filter |

- **Target is precomputed.** `Y = lcfs_rho` is the upstream `r(θ)@32` profile — it
  is **not** reprojected here (unlike the WEST track).
- **Canonical X layout** from `imas_inputs.yml` tiers → groups → named datasets:
  **T0** (`pf_currents`, `tf`, `lh_power`, `ic_power`) and **T2** (`globals`,
  `axes`); T1 is empty. Every shot gets the **same columns in the same order**;
  absent / `h5py.Empty` / misaligned channels become **NaN columns** (not dropped),
  so `X[:, cols]` is valid for every shot.
- **Validity** requires finite `Y` **and** a sustained flat-top window (from `ip`);
  a shot with no detectable flat-top is dropped (`ok=False`). X finiteness is
  intentionally not enforced at build time — the input sweep checks it per selected
  column.

→ **4560 npz** from 6187 IMASH5 shots (the 1627 drops are the flat-top gate).

---

## 6. Trainable NPZ schema

**`NpzOrigin/<shot>.npz`** (per `meta.json:arrays`):

| Array | Shape | dtype | Meaning |
|---|---|---|---|
| `X` | `(nt, 22)` | float32 | input feature channels |
| `Y` | `(nt, 32)` | float32 | polar target `r(θ)` about origin `(2.5, 0)` |
| `S` | `(nt, 2)` | float32 | scalar labels `[beli, li]` (targets only) |
| `bnd_RZ` | `(nt, 32, 2)` | float32 | raw Cartesian boundary (R, Z), for plotting |
| `time` | `(nt,)` | float32 | ignitron-time grid |
| `valid` | `(nt,)` | bool | per-slice validity mask |

**`IMASNpz/<shot>.npz`**:

| Array | Shape | dtype | Meaning |
|---|---|---|---|
| `X` | `(nt, F)` | float32 | concatenated tier channels (NaN where absent) |
| `Y` | `(nt, 32)` | float32 | `lcfs_rho` (precomputed `r(θ)@32`) |
| `time` | `(nt,)` | float32 | absolute IMAS seconds |
| `valid` | `(nt,)` | bool | finite-Y + flat-top mask |

Both carry a sibling `meta.json` (origin/theta, input layout, normalization).

---

## 7. Cross-cutting facts

- **IMAS ↔ DCS time offset:** constant **+2.50 s** campaign-wide (IMAS later).
  The two pipelines are **not** on the same time base — WEST is ignitron-relative,
  IMAS is absolute seconds.
- **Shared wall geometry:** ToFu polygons, used only for visualization
  (`scripts/plot_*`), not fed into training.
- **Selection yields:** WEST 759 `MergedH5` → 758 `NpzOrigin` (from 1425 raw);
  IMAS 4560 `IMASNpz` (from 6187 `IMASH5`).

---

## 8. Build & run commands

```bash
# WEST / DCS — full pipeline (status → freq → flat_top → merge → build_npz)
python scripts/run_data_pre.py --workers 16

# WEST — individual stage (e.g. rebuild only the NPZ from existing MergedH5)
python -c "from src.data.build_npz import run; run(workers=16)"

# IMAS — build the IMAS-native NPZ (no scripts/ runner; importable module)
python -c "from src.data.build_imas_npz import run; run()"

# Train / infer
python scripts/train_dcs.py      # WEST (reads cfg.mergednpz_dir — see §9)
python scripts/train_imas.py     # IMAS
python scripts/train_m0.py       # IMAS M0 baseline
```

---

## 9. Directory & config map

`src/proj_config.py` exposes each dir as a property; current disk state:

| Property | Returns | On disk | Status |
|---|---|---|---|
| `dcsheating_dir` | `datasets/DCSHeating` | ✓ 1425 `.mat` | ok |
| `gmagh5_dir` | `datasets/GMagH5` | ✓ 1425 `.h5` | ok |
| `mergedh5_dir` | `datasets/MergedH5` | ✓ 759 `.h5` | ok |
| `mergednpz_dir` | `datasets/MergedNpz` | ✗ (renamed to `NpzOrigin`) | **stale — see below** |
| `imas_h5_dir` | `datasets/IMASH5` | ✓ 6187 `.h5` | ok |
| `imas_npz_dir` | `datasets/IMASNpz` | ✓ 4560 `.npz` | ok |
| `npz_dir` | `datasets/Npz` | ✗ (unused) | n/a |

**Known drift.** `mergednpz_dir` still returns `"MergedNpz"`, but the dir was
renamed to `NpzOrigin`. Consumers that go through this property —
`scripts/train_dcs.py`, `scripts/demo/plot_lcfs_slice.py`, and
`src/data/build_npz.py` (default output) — therefore point at a missing dir.
Minimal fix: change the literal in `proj_config.py` from `"MergedNpz"` to
`"NpzOrigin"` (renaming the property to `npzorigin_dir` and updating the 3 call
sites is the cleaner option).
