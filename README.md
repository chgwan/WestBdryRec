# WestBdryRec

ML reconstruction of the WEST last-closed-flux-surface (`targets/GMAG_BND`, 32
(R, Z) boundary points) from DCS control-system inputs.

## Data filter criteria

A shot enters the training dataset (`Merged/<shot>.h5`) only if it passes **all**
of the criteria below. They are evaluated from two tables and intersected by
`src/data/merge_dcs_bdry.py:load_selected_shots`.

### Quality criteria — `ProjDB/Stats/shot_status.csv` (`src/data/shot_status.py`)

| # | Criterion | Field | Rule |
|---|-----------|-------|------|
| 1 | Boundary present and non-trivial | `bnd_category` | contains `nonzero` (GMAG_BND exists and is not all-zero) |
| 2 | Boundary sampling rate | `bnd_fs_hz` | `> 400` Hz (nominal ≈ 488 Hz; excludes downsampled/corrupt records) |
| 3 | DCS record usable | `dcs_category` | `ok` (DCS h5 present and not truncated, i.e. ≥ `dcs_trunc_min` = 1000 samples) |

### Physics criterion — `ProjDB/Stats/flat_top.csv` (`src/data/flat_top.py`)

| # | Criterion | Field | Rule |
|---|-----------|-------|------|
| 4 | Sustained plasma flat-top | `total_flat_top_s` | `flag == "ok"` **and** total Ip flat-top `> flat_top_min_s` = 3.0 s |

The flat-top is detected on the DCS Ip reference (`Ip_scope_0`), restricted to the
plasma window and normalized by the plateau current, then its edges are refined
with the actual Ip (`Ip_scope_3`) q20 threshold (ported from the EAST analyzer).

All thresholds live in `src/proj_config.py` (`dcs_trunc_min`, `downsample_tol`,
`flat_top_min_s`) and are overridable via `PROJ_*` environment variables.

**Effect:** of the 1026 shots passing the quality criteria (1–3), 759 also pass
the flat-top criterion (4) and are written to `Merged/`.

## Pipeline

`python scripts/run_data_pre.py --workers 16` runs, in order:

1. `shot_status`  — boundary + DCS status and per-signal frequency scan → `shot_status.csv`
2. `scan_frequencies` — per-signal frequency summary + DCS `.mat` time-axis check
3. `flat_top` — per-shot Ip flat-top detection → `flat_top.csv`
4. `merge` — resample all DataOrg inputs/targets onto the calibrated DCS-Ip time
   grid (`t = 0` at the Ip-ref onset) and write the single-`time` `Merged/<shot>.h5`
