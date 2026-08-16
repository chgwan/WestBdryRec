# LCFS dataset filters — summary

Every filter that decides whether a WEST LCFS slice enters the trainable dataset, with the
measured cost of each. This is the summary of `newTrain.md` items 1–3 as designed on
2026-08-05.

- **Rationale and evidence:** `docs/superpowers/specs/2026-08-05-lcfs-slice-shape-gate-design.md`
- **Scanner:** `exploration/lcfs_gate_check.py` — 1425 shots in ~70 s on 80 workers
- **Visual check:** `figs/LCFS/gate_check/gate_<crit>_p*.png` (grid galleries) and
  `figs/LCFS/gate_check/s1…s5/` (inspectable two-panel examples per criterion)

## Where the markers live

Per-slice markers are stored **in the GMagH5 files themselves**, so filtered slices can be
selected on read without rescanning:

```python
with h5py.File(f"ProjDB/datasets/GMagH5/{shot}.h5") as hf:
    keep = np.asarray(hf["quality/keep"])          # bool, on targets/GMAG_BND_time
    fail = np.asarray(hf["quality/fail"])          # 0 kept, 1..6 = S0..S5 (first failure)
    only = np.asarray(hf["quality/only"])          # criterion that ALONE rejects it
    cx   = np.asarray(hf["quality/convexity"])
```

`quality/` also holds `winding`, `outside_cm`, `margin_cm`, `depth_cm`, `r_span`, `absz_max`,
`min_edge_cm`, `n_nonfinite`. Group attrs record the criteria it was produced with
(`criteria_version`, `convex_min`, `wall_file`, `wall_tol_cm`, `r_span_min`) so **stale markers
are detectable** — check `criteria_version` before trusting them; these criteria have been
revised repeatedly.

Written by `exploration/lcfs_write_markers.py` (dry-run by default; `--apply` to write,
`--remove --apply` to undo; idempotent). Present in **1346 of 1425** files — the 79 without it
are the unreadable shots. Adds 827 MB uncompressed, gzip-4 on disk.

A portable side-car copy of the same table is at `figs/LCFS/gate_check/gate_slices.parquet`
(22,982,105 rows, 174 MB, zstd) for analysis that would rather not open 1346 HDF5 files.

> **On `NpzUni500` (newTrain V2) these criteria judge *interpolated* geometry.** That
> branch resamples the boundary onto a uniform 500 Hz lattice before filtering, so a
> smooth blend spanning a reconstruction dropout can pass. `NpzGeom` filters the raw
> reconstruction and is unaffected. Use `src_gap_ms` in the V2 NPZ to tell the two
> populations apart.

---

## 1. Shot level — still open

| filter | rule | status |
|---|---|---|
| sampling rate | drop GMagH5 with `fs < 488 Hz` | **not yet specified**; ~9 shots run at 30 Hz (`docs/data_lineage.md`) |
| empty / corrupt | drop unreadable h5 | **measured: 79 of 1425 shots** — no `targets/GMAG_BND`, or wrong shape |
| keep-fraction | drop shots left with too few valid slices | **not yet decided** |

Readable shots: **1346 of 1425**.

---

## 2. Slice level — designed and measured

Applied in order to the 32 `targets/GMAG_BND` vertices, that slice's `GMAG_GEOM`
`(Rgeom, Zgeom)`, and its ignitron time. Each slice is attributed to its **first** failure, so
the counts partition rather than double-count. S1 onward is quoted against the in-discharge pool,
since negative time is not a candidate for anything.

| # | filter | rule | rejected | % of `t ≥ 0` |
|---|---|---|---|---|
| **S0** | in the discharge | `t ≥ 0` (ignitron clock) | 7,490,725 | *32.594% of raw* |
| — | *in-discharge pool* | | *15,491,380* | *denominator below* |
| **S1** | finite & non-degenerate | all 64 coords finite, not all-zero, shortest edge > 1e-6 m | 98,473 | 0.636% |
| **S2** | inside the vessel | every vertex within **0.5 cm** of the vessel contour, **and** R-span > 0.3 m | 691,350 | 4.463% |
| **S3** | center enclosed | winding number of the vertex loop about `(Rgeom, Zgeom)` = ±1 | 5,074,207 | 32.755% |
| **S4** | simply connected | θ about `(Rgeom, Zgeom)` strictly monotonic | 462 | 0.003% |
| **S5** | convexity | `\|A_polygon\| / \|A_hull\| ≥ 0.995` | 6,148 | 0.040% |
| | **KEPT** | | **9,620,740** | **62.104%** (41.862% of raw) |

Raw total: 22,982,105 slices over the 1346 readable shots.

### What each one is for

**S0–S3 answer "is there a real reconstructed plasma at this timestamp?"** They account for 96%
of everything cut, and almost none of it is data loss — raw `GMAG_BND_time` spans ≈ −30 → +38 s,
so S0 removes the pre-ignitron third outright and S3's share is dominated by the post-plasma
tail, where `GMAG_GEOM` reverts to its `(0,0)` sentinel under a stale circular boundary.

**S4–S5 answer "is this shape usable as a target?"** Together they cost only **6,610 slices =
0.069%** of S1–S3 survivors. They are the surgical part, and the two slices originally flagged as
unacceptable fail here:

| reference slice | convexity | fails |
|---|---|---|
| `57620 @ t=39.101 s` — re-entrant V-notch, 11.9 cm deep | 0.9283 | **S5** (S0–S4 pass) |
| `58303 @ t=67.443 s` — vertices zigzag across the center, curve self-crosses | n/a | **S4** (S0–S3 pass) |

S4 and S5 catch the same pathology at different severities: a wedge that reaches the center makes
`r(θ)` genuinely multi-valued (S4); one that stops short leaves it single-valued but dented (S5).

### Marginal necessity — how much each criterion uniquely contributes

The funnel above attributes each slice to its **first** failure, which depends on the order the
criteria are applied in. Two order-free views matter more.

*Rejected by this criterion **alone**, every other criterion passing* — a criterion whose
exclusive set is empty rejects nothing the others would not have caught:

| criterion | rejects alone |
|---|---|
| S1 | 98,473 |
| S2 | **151** |
| S3 | **1** |
| S4 | **1** |
| S5 | 6,148 |

*Independent failure counts over the S0&S1 pool (15,392,907 slices)* — each criterion evaluated
on its own, ignoring the others:

| criterion | fails independently |
|---|---|
| S2 | 691,350 (4.491%) |
| S3 | 5,765,295 (37.454%) |
| S4 | 5,765,817 (37.458%) |
| S5 | 2,089,351 (13.573%) |

**S3 and S4 are very nearly the same test.** They fail on the same ~5.765 M slices (differing by
~500), and each uniquely adds exactly **1**. This is not a coincidence: if θ about a point
advances strictly monotonically around a closed 32-vertex loop, the total winding must be ±1, so
**S4 ⟹ S3** except when a single angular step exceeds π — the double-wrap case, which is that one
slice. Consequently the funnel's dramatic 5,074,207-vs-462 split between S3 and S4 is an
**ordering artifact**: swap their order and S4 would take the 5 M. Keep both anyway — S3 is the
cheap vectorised test that yields the interpretable "no equilibrium reconstructed" label, and S4
catches the double-wrap that S3 alone would pass.

**S5's independent count (2.09 M) dwarfs its first-failure count (6,148)** because a boundary
already condemned by S3/S4 usually has poor convexity too. Its 6,148 exclusive rejects are the
ones that matter: shapes nothing else objects to.

### Vessel contour used by S2

`ProjDB/datasets/west_geom/TFG_Ves_ExpWEST_StandardV2.txt` — ToFu format (`# key = value`
comments, a `count 0` row, then `count` rows of `R Z` in metres), 59 vertices spanning
R 1.788–3.298 m, **Z −0.798 → +0.869 m** (note the asymmetry).

This is the **D-shaped envelope**, not the true vessel: it sits strictly inside
`TFG_Ves_ExpWEST_InnerV0.txt` (591 pts), which reaches 30.8 cm beyond it. S2 is therefore
deliberately conservative — a tighter bound than the bare vacuum-vessel shell.

The 0.5 cm tolerance exists because limiter plasmas genuinely touch the inner bumper and the
reconstruction carries its own error. Measured over 15,392,907 in-discharge finite slices, the
outward excursion is p99 = **0.036 cm** but p99.9 = **36.5 cm** — legitimate slices hug the wall
to sub-millimetre, violators miss by tens of centimetres. Reject counts are flat from 0.5 cm
(152,267) to 2 cm (148,630), so 0.5 cm is the tightest value inside that gap.

---

## 3. Revisions to the original plan

**`newTrain.md` 3.1 — "strict convex, convexity score 1.0 or larger than 0.99" → area ratio ≥ 0.995.**
Strict convexity is not viable. Real WEST equilibria carry ~2 concave vertices ~1 mm deep at the
X-point cusp, so on good slices `n_concave == 0` keeps only **13.6%** while `n_concave ≤ 2` keeps
**100.0%** — no vertex-count threshold discriminates at all. The score has to be the integral
area/hull ratio. 0.995 rather than 0.99 because at 0.99 a further ~900 slices with a visible
5–7 cm step survive, and 0.995 makes a separate notch-depth criterion exactly redundant.

**`newTrain.md` 3.2 — "has Rgeom, Zgeom (sentinel of equilibrium reconstruction)" → enclosure.**
Strengthened from a plausibility box (`2.0 < Rgeom < 3.0`, `|Zgeom| < 0.6`) to requiring the
center to lie *inside the boundary*. Equivalent in effect on the sentinel cases but principled —
no hand-tuned box — and it additionally catches slices whose vertex order is scrambled so the
curve loops its own center twice (`winding = ±2`).

## 4. Filters added beyond the original plan

- **S0 `t ≥ 0`** — 32 slices at negative time passed every other criterion, carrying a
  plausible-looking boundary written before the ignitron fired.
- **S1 finite / non-degenerate** — all-zero boundaries in the samples between ignitron and
  breakdown.
- **S2 vessel containment** — replaces a hand-picked box `R ∈ (1.8, 3.3)`, `|Z| < 1.2` that
  accepted **120,730** slices with vertices more than 1 cm outside the vessel while rejecting only
  31,449, and never rejected anything the wall accepts. Switching changes the fate of only 95
  kept slices, because S3 already caught the rest — the criterion is now correct rather than
  coincidentally adequate.
- **S4 simple-connectedness** — required for `r(θ)` to be single-valued.
  `build_npz.radii_on_grid` already *assumes* this ("Vertices are angularly monotonic about the
  chosen origin (verified)") and the assumption is false for a third of raw slices; when it
  fails, `argsort(θ)` silently emits a boundary that never existed.

## 5. Rejected criteria — do not reintroduce without new evidence

| candidate | why |
|---|---|
| convexity by vertex count | see 3.1 — keeps 13.6% or 100.0%, nothing in between |
| notch depth ≤ 5 cm | rejects 290 ordinary diverted plasmas (legitimate X-point dips reach 7 cm), and subsumed: at convexity ≥ 0.995 no survivor has hull depth > 5 cm |
| max concave turn > 20° | scale-free, so dominated by vertex clustering near the X-point — a 1 mm wiggle between two close vertices scores 20° |
| local sagitta > 2 cm | same failure mode |
| segment-intersection test | implied by S4 |
| temporal smoothness | a slice passing S0–S5 is a physically valid shape; dropping it for jumping needs separate justification |

**The general lesson for shape scoring: only integral measures work.** Every local metric fires
on the X-point cusp, because a 32-point discretization of a high-curvature cusp is genuinely
kinked at the centimetre scale.

---

## 6. Consequence for item 4 (retrain)

The gate judges enclosure about `(Rgeom, Zgeom)`, not the fixed `(2.5, 0)`. Slices that do not
enclose `(2.5, 0)` therefore now pass — and for exactly those, the current fixed-origin
projection at `build_npz.py:247` fabricates `Y`. So the target moves to

```
Y[t] = r(θ) about (Rgeom[t], Zgeom[t])       # per-slice polar origin
```

with `center (nt, 2)` stored in the NPZ and `meta.json` recording
`origin: "per_slice_gmag_geom"` instead of the `[2.5, 0]` literal.

Absolute `(R, Z)` reconstruction then needs the center as well as `r(θ)`, so the retrain must
decide: either the model also predicts `(Rgeom, Zgeom)`, or evaluation stays in `r(θ)` space.
This is already the IMAS track's structure — `src/ml/axis_frame.py` `reconstruct_absolute()`
rebuilds absolute vertices from a per-slice axis plus `rho(θ)`, and IMAS `lcfs_rho` is already
that form — so the change makes both pipelines share one representation.

## 7. Known escape — shot 57486 (2026-08-16)

Passes every S0–S5 criterion (5 297 / 5 298 slices valid, boundary variability
exactly median — radii σ rank 36/76), yet is unmodellable by the M3 predictor
at every input density: per-shot CCC 0.25–0.31 vs ~0.995 typical, standardized
MSE 10–34 — **worse than predicting the train mean** (MSE ≈ 1), so the failure
is systematic extrapolation, not noise. The discharge expands and drifts
upward through a regime the actuators apparently do not explain, while the
model outputs a static average boundary.

Evidence: `docs/newtrain_dropout_results.md` §"Post-hoc correction" (18 runs,
both PEs, 3 seeds); figures `figs/shot57486/shot57486_result.png`,
`figs/shot57486/shot57486_lcfs_evolution.png` / `.mp4`.

Candidate criterion for the next filter revision (not adopted yet): flag any
shot on which **every** trained arm fails to beat the constant-baseline MSE —
an out-of-distribution-regime test computable at scoring time without hand
labelling. One observed case among the 76 test shots.
