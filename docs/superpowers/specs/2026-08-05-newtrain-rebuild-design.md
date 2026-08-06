# newTrain rebuild — design

**Date:** 2026-08-05 (rev 2 — incorporates the answers to §9 and the centre-prediction addition)
**Project:** WestBdryRec — ML prediction of the WEST LCFS (`targets/GMAG_BND`, 32 (R,Z) points).
**Scope:** `newTrain.md` items 1–6 end to end, plus predicting `(Rgeom, Zgeom)`: rebase the
WEST/DCS pipeline on the GMagH5 time base, apply the per-slice quality filters, move the
target to a per-slice polar origin, add a time positional encoding, build a new NPZ
dataset, and retrain M0/M1/M2 to predict boundary **and** centre.

**Status.** Items 1–5 are built and verified. Item 6 and the centre prediction are
**not built** — the retrain was started and deliberately stopped for this review, and its
partial artifacts were deleted. **No accuracy numbers exist yet.**

---

## 1. Goal and success criteria

Produce a WEST/DCS training dataset in which every target slice is a real, usable LCFS
reconstruction, and retrain the three DCS models to predict the full boundary in
absolute geometry — `r(θ)@32` **and** the polar centre it is measured about.

Success:

1. All three models train on `NpzGeom` and emit 34 outputs.
2. `r(θ)` CCC / R² reported against the `NpzOrigin` baseline (M0 0.9579 / 0.8452,
   M1 0.9530 / 0.8284, M2 0.9668 / 0.8763, 76 test shots).
3. Centre error reported separately in mm, plus absolute-boundary RMSE — the metric the
   centre prediction actually unlocks.
4. The PE ablation quantifies what the positional encoding contributes.

---

## 2. Items 1–2 were mostly already satisfied

Item 1's `fs >= 488` gate and item 2's empty/corrupt drop are **already enforced** by
`merge_dcs_bdry.load_selected_shots`: `bnd_fs_hz > 400` excludes the 9 shots recorded at
30.518 Hz, and `bnd_category == nonzero` excludes 78 `absent` + 24 `allzero` + 1
`corrupt`. That 79 = absent + corrupt exactly matches the 79 GMagH5 files the filter
scanner cannot read. With `dcs_category == ok` and Ip flat-top > 3.0 s this selects
**759 shots**, unchanged.

So the only real work in item 1 is the **time base**.

---

## 3. Item 1 — GMAG-native time base  ✅ built

`merge_dcs_bdry.run(time_base="gmag")` (new default) makes the shared grid
`targets/GMAG_BND_time` itself, clipped to the DCS span, and resamples the DCS scopes
onto *it*. Previously the grid was the ~1 kHz DCS axis with `GMAG_BND` interpolated onto
it.

**Why.** The LCFS is the quantity being predicted. Interpolating it fabricates boundaries
between reconstruction samples, and — the sharper problem — the quality filters were
judging interpolated geometry, with interpolation across a rejected slice smearing it
into its neighbours.

### 3.1 The target rate is variable, not 488 Hz

An earlier revision of this spec called this axis "the native 488 Hz grid" and claimed the
target's true rate "was always 488 Hz". **Both were wrong.** Measured over all 1346 shots
with a boundary time axis:

| | |
|---|---|
| modal step | 2.048 ms → 488.3 Hz, in 1337 shots (9 shots run at 30.5 Hz) |
| within-shot `dt` spread / median | p50 **4.18**, p90 5.06, p99 6.19 |
| longest gaps inside a shot | **32–33 ms** (≈30 Hz stretches) |
| perfectly uniform shots | **0 of 1346** |

488 Hz is the *modal* step, not the rate. Every shot contains stretches where the
equilibrium reconstruction dropped out, so the axis is piecewise-488 Hz with gaps up to
~16× the nominal step. Consequences that follow from this, and are now stated rather than
assumed:

- The shot-level `fs >= 488` gate (§2) screens the **median** rate. It does not, and
  cannot, promise uniform spacing.
- **M2 (GRU) is step-indexed**, so it implicitly treats consecutive samples as equally
  spaced when they are not. The time positional encoding (§6) partly compensates by
  giving every step its absolute time, but this is a real limitation of the sequence
  model on this data, not something the rebuild fixes.
- `cum_heat` integrates with `np.diff(t)`, so it handles variable spacing correctly.

### 3.2 Two monotonicity defects found and fixed

Chasing the rate claim surfaced a genuine bug. **27 of the 759 selected shots carry a
non-increasing sample** in the boundary time axis — usually sub-millisecond jitter
(e.g. 57438: 14.60539 s → 14.60489 s), and once catastrophically (57416: `dt` = −42 943 s).

1. **Non-monotonic grid.** Taking a contiguous slice of that axis produced a merged
   `time` array with a backwards step in 10 shots. Anything integrating `np.diff(t)`
   (i.e. `cum_heat`) then subtracts instead of adds. Fix: select a **strictly increasing
   subsequence** rather than a contiguous slice — costs one sample in the affected shots.
2. **Non-monotonic interpolation source, the more serious one.** `resample_to_grid`
   passed that same axis to `np.interp` as `xp`. `np.interp` requires monotonic `xp` and
   returns silently wrong values otherwise — no error. This corrupted the stored boundary
   for **5 shots**. Fix: `resample_to_grid` now sorts and de-duplicates `ts` (with its
   values) when the axis is not ascending.

The first guard alone did not catch the second: the grid was clean while the boundary was
still wrong. Only comparing against the merge's own index selection exposed it — a
reverse `searchsorted` lookup cannot, because on a non-monotonic axis the lookup is itself
ill-defined.

**Verified after the fix, across all 759 shots:** 0 non-increasing time steps,
`grid == gb_t[sel]` 759/759, and the stored boundary equals the raw reconstruction
**759/759**. Shot 57416, which previously failed the build outright, is now usable.
Shot 57281: 41 280 samples @ ~1 kHz → 11 339 samples. Output
`ProjDB/datasets/MergedH5Gmag/`, 759 shots, 0 failed. `time_base="dcs"` still reproduces
the old grid, so the baseline stays reproducible.

**Cost:** roughly half as many slices per shot. Not information loss — the extra 1 kHz
samples were interpolated, never measured.

---

## 4. Item 3 — per-slice quality filters  ✅ built

`src/data/filter.py` S0–S5, unchanged from the approved filter design. Thresholds and
evidence: [`docs/lcfs_filters.md`](../../lcfs_filters.md); rationale:
[`2026-08-05-lcfs-slice-shape-gate-design.md`](2026-08-05-lcfs-slice-shape-gate-design.md).

Wired into `build_npz._build_one`:

1. Evaluate `SliceQuality` over the whole record.
2. Trim to `keep_span(q.keep)`. `discharge_window` is retained unchanged because
   `src/data/sweep_inputs.py` still calls it.
3. Failing slices inside the span stay in the arrays, marked invalid, so the sequence
   model sees an unbroken index sequence. Note this preserves the *index* sequence, not a
   uniform time step — the axis was never uniform (§3.1).
   
4. `valid = keep & isfinite(Y).all(axis=1)`.

Rejected slices get `Y = NaN`, never a projected profile. Verdicts ship with the data:
`fail`, `only`, `flags`.

### 4.1 Rejected slices are masked in the loss, not deleted  ⬜ to build

Broken slices stay in place and are excluded by **masking the loss**, rather than by
removing them from the record. For the sequence model that matters: deleting a slice would
splice two non-adjacent instants into neighbours and silently corrupt the temporal
structure the GRU is reading. Masking keeps every step where it belongs and contributes
exactly zero to the objective.

Per model, as the code already stands:

| model | how the mask is applied | status |
|---|---|---|
| M0 (HistGBT) | selects valid rows by index (`Y[idx]`), invalid rows never enter the fit | fine |
| M1 (ResMLP) | same row selection in `DCSSnapshotDataset` | fine |
| M2 (GRU) | keeps the whole series, multiplies the squared error by the mask | **needs the fix below** |

### 4.2 The masking must not be multiplicative against NaN

M2's objective is

```python
loss = (((pred - Y) ** 2) * w).sum() / w.sum().clamp(min=1.0)     # w = mask
```

`NpzGeom` stores `Y = NaN` at rejected slices, and **`NaN × 0 = NaN`** — so a single
rejected slice makes the whole batch loss `NaN` and every gradient `NaN`, destroying the
model on the first step. Verified directly: with one masked NaN target the loss is `nan`
and the gradient at *every* position is `nan`; zero-filling the target gives a finite loss
and a zero gradient at the masked position.

This is a landmine specific to the new dataset: `NpzOrigin`'s `Y` was finite for every
slice, because the old builder projected a profile even for garbage geometry. Making the
target honest (NaN where there is no usable boundary) is what exposes it.

**Fix:** in `DCSSeqDataset`, zero-fill non-finite targets *after* computing the mask —
exactly what `read_series` already does for the inputs ("invalid steps are kept (zeroed by
`read_series`) and masked out of the loss by the trainer"). The mask, not the target value,
is what excludes the slice. Alternatively index the loss (`pred[mk]`), but zero-filling
matches the existing convention and keeps the tensor shapes rectangular for the GRU.

**Guard to add:** a test that a shot containing rejected slices yields a finite loss and
finite gradients. Without it this failure is invisible in review and only shows up as a
`NaN` training curve.

using ml loss function to masked the broken slice

---

## 5. Target origin — per slice  ✅ built

`Y = r(θ)` about each slice's own `(Rgeom, Zgeom)` (`inputs/GMAG_GEOM[0:2] × 1e-3`),
stored as `center (nt, 2)`. `meta.json:origin` is the string `"per_slice_gmag_geom"`.

**The fixed origin is removed from the pipeline, not merely bypassed.** `build_npz.ORIGIN`,
the `origin=` parameter threaded through `run()`/`_build_one()`, and the
`meta.json:origin_legacy_fixed` key are all deleted: after this change nothing in the
build reads a constant origin, so keeping one around would only invite a future caller to
use it. `radii_on_grid(R, Z, origin, theta)` keeps its `origin` argument — it is a general
projection helper and the caller now always passes the slice's own centre.

**Forced, not cosmetic.** The filters judge enclosure and star-shapedness about that
centre, which is exactly `radii_on_grid`'s precondition; a fixed origin cannot satisfy it
for boundaries that exclude the origin. That docstring previously *claimed* monotonicity
was "verified" — false for about a third of raw slices, and when it fails `argsort(θ)`
silently emits a boundary that never existed.

**Fidelity measured:** reconstruction error 2.81 mm median / 6.22 mm max versus
3.08 mm / 5.83 mm for the fixed origin (240 slices). The ~3 mm is inherent to resampling
32 non-uniform vertex angles onto a uniform 32-grid; the origin move is neutral.

---

## 6. Item 4 — time positional encoding  ✅ built

```
pairs = 5                      # sin/cos pairs requested
d     = 2 * pairs = 10         # PE block width -- plays the role of d_model
i     = 0 .. pairs-1
PE[:, 2i]   = sin(t / 5**(2i/d))
PE[:, 2i+1] = cos(t / 5**(2i/d))
```

`t` = ignitron **seconds** (not sample index). Base **5** rather than 10000 because
positions are physical seconds over a 10–70 s discharge.

**Where `d` comes from.** It is not a free parameter: `d = 2 × pairs`, i.e. the width of
the PE block itself, which is the standard reading of the transformer formula where the
encoding fills the whole `d_model`. That is a **convention worth stating**, because here
the PE is 10 input columns appended to a 22-column feature vector — there is no embedding
of width 10 to identify `d_model` with. The choice is what spreads the frequencies:

| `d` | exponent `2i/d` reaches | periods | |
|---|---|---|---|
| **10** = 2·pairs (implemented) | 0.8 | **6.3 / 8.7 / 12.0 / 16.5 / 22.8 s** | a ladder spanning ramp-up → flat-top |
| 32 = full feature width | 0.25 | 6.3 / 6.9 / 7.7 / 8.5 / 9.4 s | five near-duplicate frequencies |

So `d = 10` is doing real work rather than being incidental. Both sin and cos are emitted
at each frequency, so the phase is unambiguous — sin alone cannot separate a rising from a
falling instant.

Occupies `X` columns **22–31**. `meta.json:time_pe` records pairs, base, formula and
column names; `meta.json:inputs` gains a `time_pe` entry so `n_features` matches the
array width. The existing derived features already include a raw linear `time_rel`; the
PE adds multi-scale periodic time, it does not replace it.

---

## 7. Addition — predict `(Rgeom, Zgeom)` as well  ⬜ to build

Requested. Without it, `Y` cannot be turned back into an absolute boundary at inference,
which is the one thing the per-slice origin costs us (§5).

### 7.1 Target layout — 34 columns, absolute

```
target[:,  0:32] = r(θ)@32   about (Rgeom, Zgeom)   [m]
target[:, 32]    = Rgeom                            [m]
target[:, 33]    = Zgeom                            [m]
```

The centre is predicted **absolutely**, not as an offset from anything. `(2.5, 0)` is not
part of this representation: `θ` and `r` are already measured about `(Rgeom, Zgeom)` (§5),
so reintroducing the old fixed origin as a reference point for the centre would drag back
the constant the design just removed, and would leave two competing notions of "origin" in
one target vector. An earlier revision of this spec proposed exactly that; it was wrong.

`reconstruct_absolute(R̂geom, Ẑgeom, r̂, θ)` then closes the loop with no constants at all.

Measured over 347 953 valid slices:

| column | mean | std | range |
|---|---|---|---|
| `r(θ)` | 0.526 | 0.056 | 0.157 … 0.760 |
| `Rgeom` | 2.440 | 0.060 | 2.065 … 2.701 |
| `Zgeom` | −0.016 | 0.028 | −0.544 … +0.560 |

### 7.1a Consequence — targets must be standardized

The three **spreads** are already commensurate (0.056 / 0.060 / 0.028). Only `Rgeom`'s
**mean** is out of scale, and that is enough to matter for the two neural models: `Y` is
consumed raw today, so with a zero-initialized output bias the `Rgeom` column starts with
a residual of ~2.44 m against ~0.5 m for the radii — roughly 24× the squared error on 1 of
34 columns. It converges once the bias learns, but until then it dominates the gradient
into the **shared** trunk, which is precisely the part we do not want perturbed.

So all 34 target columns are **standardized by train-set mean/std** before the loss and
inverted at predict time:

- statistics computed over the train shots' valid rows only (never val/test), mirroring
  what `_dcs_train_mean_std` already does for the *inputs*;
- stored in the model artifact alongside the existing input `mean`/`std`, so a checkpoint
  remains self-contained;
- applied inversely in the predict paths, so everything downstream — bench scoring, the
  centre metrics, `reconstruct_absolute` — sees metres, exactly as now.

This replaces the arbitrary-constant approach with the standard one, costs no magic
numbers, and makes the radii and centre columns commensurate rather than merely
similar-by-luck. M0 is unaffected in substance (it fits each column independently, so
scale cannot couple across columns) but is standardized too, for one code path instead of
two.

### 7.2 Model changes — smaller than it looks

- **M0** (`train_m0_dcs`) fits `range(Ytr.shape[1])` independent HistGBTs, so it follows
  the target width automatically: 34 regressors, no code change beyond building the
  34-column matrix.
- **M1** `ResMLP(n_in, n_out=32, …)` and **M2** `ActSeqGRU(n_act, n_out=32, …)` already
  take `n_out` as a constructor parameter. Pass 34. No architecture surgery.
- The two datasets (`DCSSnapshotDataset`, `DCSSeqDataset`) and the three predict paths in
  `scripts/train_dcs.py` must build/consume the 34-column target.

### 7.3 Loss weighting — deliberately left flat

Unweighted MSE across all 34 columns, so the centre carries 2/34 ≈ 6 % of the loss. This
is a **starting point, not a tuned choice**: no measurement yet justifies a weight, and
adding a knob before there is evidence is speculative. If the centre error comes out poor
relative to the ~3 mm representation floor, revisit with an explicit
`centre_loss_weight`. Stated here so the flat weighting is a recorded decision rather
than an oversight.

### 7.4 Scoring — must be split explicitly

`bench.score_predictions` compares `preds[s]` against `_shot_y(...)` and, **on shape
mismatch, prints to stderr and skips the shot**. A 34-wide prediction against a 32-wide
truth would therefore skip every shot and return `n_shots = 0` — a near-silent failure
that still writes a bench row. So the split is explicit, not incidental:

| metric | columns | why |
|---|---|---|
| `r(θ)` CCC / R² / RMSE | `[:, :32]` | keeps the **same definition as the baseline table**, so the comparison is at least like-for-like in kind |
| `centre` MAE / RMSE [mm] | `[:, 32:]`, separately for dR and dZ | the new capability, on its own scale |
| absolute boundary RMSE [mm] | `reconstruct_absolute(R̂geom, Ẑgeom, r̂, θ)` vs `bnd_RZ` | what the centre prediction is *for*; the only metric that reflects both errors together |

The absolute metric needs the truth `bnd_RZ`, already in every NPZ.

---

## 8. Items 5–6 — dataset and retrain

**Dataset** `ProjDB/datasets/NpzGeom/` (`cfg.npzgeom_dir`), from `MergedH5Gmag`. Built
and verified: **759 npz, 0 failed**, 7 213 406 in-span slices, **7 208 816 valid
(99.94 %)**, 32 features. Every selected shot now survives — 57416 failed the build
before the §3.2 monotonicity fixes.

Nothing is overwritten: new merge dir, new npz dir, new train run dirs, new bench CSV.
`NpzOrigin` remains the baseline.

**Plumbing** — `scripts/train_dcs.py` gains `--npz-dir`, `--config`, `--run-name`
(plus existing `--bench-out`), all defaulting to the old behaviour.

**`configs/dcs_model_geom.yml`** (new) = the same 18 strict actuators as `dcs_model.yml`
— "the same input from the previous" — **plus** the 10 PE channels with `kind: pe`,
`nan: zero`. Hyperparameters copied unchanged so capacity is not a confounder. `kind: pe`
is deliberately none of `pf`/`lh`/`ic`, so PE cannot leak into the derived
`pf_norm` / `lh_plus_ic` / `cum_heat` features.

**This config is load-bearing and easy to miss:** `dcs_features` selects columns by node
name, so a dataset column absent from the config is silently ignored. Without listing the
PE channels, item 4 would have had no effect and the run would still have looked
successful. Smoke-tested: 28 channels resolve, snapshot `(nt, 32)`, series `(nt, 28)`,
PE finite in [−1, 1].

---

## 9. Run matrix (answers folded in)

You chose (a) **and** (b), PE ablation only — no full factorial. Both runs predict the
centre, since that is a requirement rather than an experiment.

| run | dataset | config | PE | outputs | purpose |
|---|---|---|---|---|---|
| **A** `dcs_actuator_geom` | `NpzGeom` | `dcs_model_geom.yml` | ✔ 10 cols | 34 | headline result |
| **B** `dcs_actuator_geom_nope` | `NpzGeom` | `dcs_model.yml` | ✗ none | 34 | isolates the PE contribution |

Six trainings (3 models × 2 runs). Bench rows go to
`Stats/dcs_predictor/bench_table_geom.csv` with a `run` column.

**Interpretability, stated plainly.** Five things differ between run A and the baseline:
time base, slice population, target origin, input columns, and now output width. A score
difference **cannot be attributed** to any single one. Run B isolates only the PE axis,
which is the right one to isolate: the filters, time base and origin are justified by
correctness (no fabricated targets), so they do not need a score to defend them, whereas
the PE's sole justification is accuracy.

A score *drop* versus baseline would not necessarily be a regression. The baseline was
partly scored on interpolated boundaries and on slices this design rejects, so some of
its apparent accuracy came from predicting fabricated targets. Any writeup must say so
rather than presenting the baseline as a clean upper bound.

---

## 10. Decisions taken without explicit confirmation

| Decision | Alternative | Why this way |
|---|---|---|
| New dirs `MergedH5Gmag` / `NpzGeom`, nothing overwritten | rebuild in place | keeps the baseline scoreable; `run()` does `rmtree` first, so in-place would destroy it |
| New `configs/dcs_model_geom.yml` rather than editing `dcs_model.yml` | edit in place | the baseline's config must keep describing the baseline |
| `time_base="gmag"` became the **default** | keep `dcs` default | interpolating the target is a defect, not a preference; `"dcs"` still available |
| `kind: pe` as a new channel kind | reuse an existing kind | any existing kind would pull PE into a derived feature |
| PE at columns 22–31, appended right | interleave | keeps existing column indices stable for position-keyed readers |
| Centre predicted **absolutely**; the fixed origin is removed from the pipeline entirely | keep `(2.5, 0)` as a reference offset | your call — one notion of origin, not two; no magic constant anywhere |
| Targets **standardized** by train-set mean/std | raw targets | the only remaining scale problem is `Rgeom`'s 2.44 m mean perturbing the shared trunk early; standardization is the textbook fix and needs no constant |
| Centre loss weight left flat (2/34) | weight the centre up | no evidence yet; §7.3 records it as revisitable |

---

## 11. Verification status

**Done:**
- Fixed origin removed from production code and the artifact rebuilt to match:
  `build_npz.ORIGIN`, `filter.ORIGIN`, the `origin=` parameter and
  `meta.json:origin_legacy_fixed` are all gone; `NpzGeom`'s rebuilt `meta.json` confirms
  `origin == "per_slice_gmag_geom"` and no legacy key. `scripts/demo/plot_lcfs_slice.py`
  now projects about `(Rgeom, Zgeom)` (it previously parsed `meta["origin"]` as a
  coordinate pair, which would crash on the string) and refuses a slice with no valid
  centre instead of silently falling back.
- Merge: boundary bit-identical to raw GMagH5; 759/759 merged, 0 failed.
- Filters: 36 tests green (`tests/test_filter.py` 26, `tests/test_build_npz_filter.py` 10),
  including a guard that a normal X-point cusp stays kept.
- Build: 4-shot real-data check — every `valid` row has finite `Y`, every invalid row
  all-NaN, `flags == 0b111111` exactly where `valid`, `fail`/`only` agree.
- PE: `sin(0)=0`, `cos(0)=1`, values in [−1, 1], periods as specified.
- Feature path: all 28 config channels resolve against `NpzGeom`'s `meta.json`.
- Docs: `docs/data_lineage.md` §4b, `NpzGeom` schema table, directory map.

**Not done:** centre prediction (§7), the six trainings, and therefore every accuracy
claim.

**Planned checks for §7 before trusting a run:** a shape assertion that predictions are
34-wide and `n_shots > 0` (guarding the silent-skip path in §7.4); a round-trip test that
`reconstruct_absolute` on the *truth* columns reproduces `bnd_RZ` to the ~3 mm
representation floor, so the absolute metric's floor is known before reading model
numbers.

---

## 12. Pre-existing issues deliberately not fixed

- `cfg.mergednpz_dir` returns `"MergedNpz"`, which does not exist on disk (renamed
  `NpzOrigin`). Left alone; `NpzGeom` has its own property. 
- 9 failing tests unrelated to this work: 4 `KeyError: 'X_cnt'` in `reduce_stats`, 5
  collection errors from `src.ml.dataset.load_meta` not existing.

## 13. Out of scope

The IMAS track is untouched (`lcfs_rho` is precomputed upstream and carries no `(R,Z)` to
filter). No hyperparameter tuning: hyperparameters are copied from the baseline so
capacity is not a confounder. No full factorial ablation (your call in §9).
