# LCFS per-slice shape gate — design

**Date:** 2026-08-05
**Project:** WestBdryRec — ML prediction of the WEST LCFS (`targets/GMAG_BND`, 32 (R,Z) points).
**Scope:** the per-slice LCFS quality gate (`newTrain.md` item 3) and the target-representation
change it forces. Shot-level rejections (item 2), the GMagH5 time rebase (item 1) and the
retrain (item 4) are separate specs.

## Goal

Reject LCFS slices whose reconstructed boundary is not a physically usable simple closed
curve, so no model is trained or scored against a fabricated target. Two slices were named
as unacceptable and both must be rejected with margin:

| reference slice | why it is wrong | criterion that rejects it |
|---|---|---|
| `57620 @ t=39.101 s` | re-entrant V-notch on the lower-inboard boundary; convexity 0.928, notch 11.9 cm deep | S5 (S0–S4 all pass) |
| `58303 @ t=67.443 s` | ~8 vertices zigzag across the plasma center, curve self-crosses, `r(θ)` multi-valued | S4 (S0–S3 pass) |

Note on the second: its center *is* enclosed — winding is 1.000 about its own
`(Rgeom, Zgeom) = (2.464, 0.027)`, so S3 passes. It is the **monotonicity** half of S4 that
rejects it. (Winding about the fixed `(2.5, 0)` is 0, which is what an origin-based test would
have keyed on — a reminder that the two tests fail on different slices.)

## Why this matters beyond data hygiene

`src/data/build_npz.py:86` `radii_on_grid()` states in its docstring that vertices are
"angularly monotonic about the chosen origin (verified)". **That is false.** When it does not
hold, `np.argsort(θ)` silently reorders the vertices and the emitted `Y = r(θ)` is not a wrong
measurement of the real boundary — it is a different boundary that never existed. The gate's
primary job is to guarantee the precondition that this function already assumes.

## Evidence base

Campaign scan via `exploration/lcfs_gate_check.py --workers 80` over all 1425 `GMagH5/*.h5`
(1346 readable, **79 unreadable** — those belong to item 2), 22,982,105 slices. S1 onward is
quoted against the in-discharge pool, since negative time is not a candidate for anything:

| criterion | rejected (first failure) | % of in-discharge |
|---|---|---|
| S0 `t < 0` | 7,490,725 | 32.594% *of raw* |
| — in-discharge (`t ≥ 0`) | 15,491,380 | denominator below |
| S1 non-finite / degenerate | 98,473 | 0.636% |
| S2 outside the vessel / too small | 691,350 | 4.463% |
| S3 center not enclosed | 5,074,207 | 32.755% |
| S4 not simply connected | 462 | 0.003% |
| S5 convexity < 0.995 | 6,148 | 0.040% |
| **kept** | **9,620,740** | **62.104%** (41.862% of raw) |

Shape rejects (S4 + S5) are **6,610 slices = 0.069%** of S1–S3 survivors. The dataset size is
set almost entirely by S0 and S3, and neither is data loss: raw `GMAG_BND_time` spans
≈ −30 → +38 s, so S0 removes the pre-ignitron third outright and S3's remaining 33.5% is
dominated by the post-plasma tail, where `GMAG_GEOM` reverts to the `(0,0)` sentinel under a
stale circular boundary. Verified in the galleries.

**S0 is not merely cosmetic:** 32 slices at negative time passed every other criterion
(kept 9,620,867 → 9,620,835 when S0 was added), so without it a handful of pre-ignitron
samples would have entered the dataset carrying a plausible-looking boundary.

Per-criterion galleries in `figs/LCFS/gate_check/` (`gate_s1_p*.png` … `gate_s5_p*.png`, plus
`gate_near_p*.png` for kept slices just above the S5 threshold and `convexity_ladder.png` for
the threshold choice). Each panel marks the tested center (purple triangle) and, for S5, the
convex hull it falls short of. S0 deliberately has no gallery — `t < 0` is a bookkeeping cut,
not a shape judgement, so there is nothing to eyeball.

## Criteria

All operate on one slice: the 32 `targets/GMAG_BND` vertices `(R, Z)` in metres, that slice's
`inputs/GMAG_GEOM[0:2] × 1e-3 = (Rgeom, Zgeom)` in metres, and its ignitron time. Evaluated in
order; the first failure is the recorded reason. S5 is only evaluated where S0–S4 hold, since it
needs a sane star-shaped polygon.

**S0 — inside the discharge.** `t ≥ 0` on the ignitron time base. Before the ignitron fires
there is no plasma, so the entire negative-time half of the raw GMagH5 grid is skipped outright
rather than being argued about geometrically.

**S1 — finite and non-degenerate.** All 64 coordinates finite, not all-zero, and the shortest
edge longer than 1e-6 m. Post-S0 this catches the samples between ignitron and breakdown, where
the boundary is not yet written (galleries show all-zero slices at t ≈ 0.001 s).

**S2 — inside the vessel, and big enough to be a plasma.** Every one of the 32 vertices lies
inside the WEST vacuum-vessel inner contour, allowing a **0.5 cm** tolerance, and
`max R − min R > 0.3 m`.

The contour is `ProjDB/datasets/west_geom/TFG_Ves_ExpWEST_StandardV2.txt` — ToFu format
(`# key = value` comments, a `count 0` row, then `count` rows of `R Z` in metres), 59 vertices
spanning R 1.788–3.298 m, **Z −0.798 → +0.869 m**. Inside/outside is the even-odd crossing rule;
the excursion is the point-to-segment distance to the nearest wall edge.

*Why a tolerance rather than strict containment:* limiter plasmas genuinely touch the inner
bumper, and the reconstruction carries its own error. Measured over 15,392,907 in-discharge
finite slices, the outward excursion is p99 = **0.036 cm** but p99.9 = **36.5 cm** — legitimate
slices hug the wall to sub-millimetre, violators miss by tens of centimetres. The reject count is
flat from 0.5 cm (152,267) to 2 cm (148,630), so 0.5 cm is the tightest value inside that gap.

*Why not the box I first proposed* (`R ∈ (1.8, 3.3)`, `|Z| < 1.2`): it is too loose on Z by a
factor of ~1.4 and asymmetric in the wrong way — the vessel runs −0.798 → +0.869, not ±1.2.
Measured, the box **accepted 120,730 slices with vertices more than 1 cm outside the vessel**
while rejecting only 31,449 in total, and it never rejected anything the wall accepts
(`box rejects but wall-contained = 0`). Containment strictly dominates it.

*Honest scale of the fix:* switching from box to vessel changes the fate of only **95 kept
slices** (9,620,835 → 9,620,740), because S3 was already rejecting almost all out-of-vessel
slices for having no enclosed center. The criterion is now correct rather than coincidentally
adequate; it is not recovering or removing much data.

**S3 — the plasma center is enclosed.** `(Rgeom, Zgeom)` finite, and the winding number of the
vertex loop about it is ±1:

```
θ_k = atan2(Z_k − Zgeom, R_k − Rgeom)
Δ_k = wrap_to_pi(θ_{k+1} − θ_k)              # cyclic over 32 vertices
W   = Σ Δ_k / 2π                             # 0 => center outside, ±1 => enclosed
S3  = isfinite(Rgeom, Zgeom) and ||W| − 1| < 1e-6
```

This replaces the plausibility-box sentinel test (`2.0 < Rgeom < 3.0`, `|Zgeom| < 0.6`). A
sentinel or absent center lands far outside the boundary and fails enclosure, so no hand-tuned
box is needed; on a 3-shot check the two agreed to within 2 slices out of 21,758.

**S4 — simply connected.** `θ` about `(Rgeom, Zgeom)` advances strictly one way:
`all(Δ_k > 0) or all(Δ_k < 0)`. A curve that is star-shaped about an interior point cannot
self-intersect, so this delivers simple-connectedness *and* the single-valued `r(θ)` the target
representation needs. Rejects 480 slices — a deep wedge cutting in past the center.

**S5 — convexity score.** `|A_polygon| / |A_convex_hull| ≥ 0.995`, shoelace area over
`scipy.spatial.ConvexHull(...).volume`.

### Why the center, not the fixed (2.5, 0)

Enclosure and star-shapedness are judged about the plasma's own geometric center. An earlier
revision judged them about the fixed polar origin and rejected 6,409 slices at S4, of which
~5,900 were *smooth, convex, valid* equilibria that merely sat off-centre (e.g.
`58303 t=67.398–67.404`, centred near Z ≈ −0.35). Those are real plasmas; switching the test to
the plasma's own center keeps them and drops S4's count to 480.

### Why 0.995

`convexity_ladder.png` walks representative shapes down the score. At 0.995 the boundary is
smooth; at 0.992 a step appears; by 0.98 there is a pronounced V-notch; the reference slice sits
at 0.928. The cost of the choice is negligible: swept on the 9.62 M-slice pool that survives
S1–S4, 0.99 → 4,931 rejects, 0.995 → 5,747, 0.997 → 6,645 (that sweep predates S0, hence the
small offset from the 6,148 in the funnel above; S0 only removes slices, so the ordering holds).
0.995 additionally makes the notch-depth criterion redundant (below).

## Retired criteria — do not reintroduce without new evidence

| candidate | why it was rejected |
|---|---|
| convexity by **vertex count** (`n_concave == 0`, or `≤ 2`) | real WEST equilibria carry ~2 concave vertices ~1 mm deep at the X-point cusp. `n_concave ≤ 2` keeps 100.0% of good slices; `== 0` keeps 13.6%. No threshold discriminates. This retires `newTrain.md`'s "convexity score is 1.0". |
| notch depth ≤ 5 cm (deepest vertex below the hull) | rejects 290 ordinary diverted plasmas; legitimate X-point dips reach 7 cm. And it is **subsumed**: at convexity ≥ 0.995, zero surviving slices have hull depth > 5 cm, so its whole reject set already fails S5. |
| max concave turn angle > 20° | scale-free, so it is dominated by vertex clustering near the X-point: a 1 mm wiggle between two close vertices scores 20°. Montage of the 20° band is ordinary diverted plasmas. |
| local sagitta > 2 cm | same failure mode as the turn angle. |
| explicit segment-intersection test | implied by S4. |
| temporal-smoothness rejection | a slice that passes S0–S5 is a physically valid shape; dropping it for jumping needs its own justification. |

The general lesson for *shape scoring*: **only integral measures work.** Any local metric fires
on the X-point cusp, because a 32-point discretization of a high-curvature cusp is genuinely
kinked at the cm scale.

An earlier revision of this table also listed "wall containment against the ToFu polygon —
S2 already bounds the boundary". **That was wrong** and has been withdrawn: measurement showed
the box S2 accepted 120,730 slices with vertices more than 1 cm outside the vessel. Containment
is now S2 itself. The lesson there is the opposite one: a hand-picked bounding box is not a
substitute for the real geometry when the real geometry is available in the repo
(`ProjDB/datasets/west_geom/`).

## Target representation change

The gate no longer guarantees `(2.5, 0)` is inside the boundary, so the fixed-origin projection
at `build_npz.py:247` would fabricate `Y` for exactly the slices the gate now admits. Resolution:
**the target moves to the same center the gate validates.**

```
Y[t] = r(θ) about (Rgeom[t], Zgeom[t])        # per-slice polar origin
```

- Uniform 32-angle grid, `θ = 0` outboard (+R), CCW — unchanged.
- The NPZ gains `center (nt, 2)` float32; `meta.json` records `origin: "per_slice_gmag_geom"`
  instead of the `[2.5, 0]` literal, so a stale reader cannot silently misinterpret `Y`.
- `bnd_RZ (nt, 32, 2)` is unchanged and remains the ground truth for absolute-geometry checks.

**Consequence to accept explicitly:** with a fixed origin, `r(θ)` alone determined the absolute
boundary. With a per-slice center it does not — absolute `(R,Z)` reconstruction needs the center
too. This is already the IMAS/M0 track's structure: `src/ml/axis_frame.py:7`
`reconstruct_absolute(axis_r, axis_z, rho, theta)` rebuilds absolute vertices from a per-slice
axis plus `rho(θ)`, and the IMAS `lcfs_rho` target is already of this form. So this change makes
the two pipelines share one representation. Either the model also predicts `(Rgeom, Zgeom)`, or
evaluation stays in `r(θ)` space; that choice belongs to the retrain spec (item 4), not here.

## Components

**`src/data/lcfs_quality.py`** (new) — the criteria, vectorized over time, no I/O:

- `slice_quality(bnd, geom, t, convex_min=0.995) -> dict` — `bnd` `(32, 2, nt)` or `(64, nt)`,
  `geom` `(2, nt)` in metres, `t` `(nt,)` ignitron seconds. Returns per-slice bool arrays
  `s0…s5`, a combined `keep`, the first-failure label array, and the metrics `convexity`,
  `winding`, `depth_cm`, `margin_cm`.
- `radii_about_center(R, Z, center, theta_grid)` — the per-slice-origin projection.
- `load_wall(name)` / `wall_outside_cm(R, Z, wall)` — vessel contour loader (process-cached)
  and the containment excursion. `wall_outside_cm` accumulates edge-by-edge rather than
  broadcasting to `(V, 32, nt)`, which would cost ~250 MB per intermediate on a long shot and
  will not survive one worker per core.
- Thresholds are module constants and keyword arguments, not literals at the call site:
  `CONVEX_MIN=0.995`, `WALL_TOL_CM=0.5`, `R_SPAN_MIN=0.3`.

**`src/data/build_npz.py`** — read `GMAG_GEOM`, call `slice_quality`, fold `keep` into `valid`,
project `Y` about the per-slice center, store `center`, and record the rejection histogram
per shot in `meta.json` so the funnel is reproducible from the artifact alone.

**`exploration/lcfs_gate_check.py`** (exists, untracked) — the QC scanner and gallery generator
that produced the numbers above. Stays exploration-only; `src/data/lcfs_quality.py` becomes the
single source of truth and the scanner imports it rather than duplicating the formulas.

## Error handling

- Absent / wrong-shaped `GMAG_GEOM` → all-NaN center → every slice fails S3. The shot is
  reported, not crashed.
- `ConvexHull` failure on a degenerate polygon → `convexity = NaN` → fails S5 (NaN comparisons
  are false).
- A shot with zero surviving slices returns `ok=False` and writes no NPZ, matching the existing
  `build_one` contract.

## Testing

1. **Reference slices** — `57620 @ 39.101` fails S5 with convexity ≈ 0.928; `58303 @ 67.443`
   fails S4 with winding ≈ 0. Both asserted directly.
2. **Synthetic geometry** — unit circle scores convexity 1.0 and passes all; a circle with one
   vertex pulled to the center fails S4; a circle with a 5% notch fails S5; an all-zero slice
   fails S1; a center placed outside a valid circle fails S3; a perfectly good circle at
   `t = −1.0 s` fails S0 (the 32 real slices that motivated it).
3. **Retired-criterion guard** — a synthetic X-point cusp (2 concave vertices, 1 mm deep) must
   pass, protecting against a future count- or depth-based regression.
4. **Vessel contour** — `load_wall()` returns 59 points with R 1.788–3.298 and Z −0.798–0.869
   (asserting the header count matches the row count, and the Z asymmetry that the old box got
   wrong); a vertex 1 cm outside the wall fails S2 while one 2 mm outside passes.
5. **Round-trip** — for kept slices, `reconstruct_absolute(center, Y, θ)` returns the original
   vertices to within interpolation tolerance; this is the check that the per-slice-origin
   target is lossless.
6. **Campaign invariant** — rerunning the scanner reproduces the funnel table above.

## Relationship to the existing spec

`docs/superpowers/specs/2026-08-05-dcs-filtered-slice-rescore-design.md` (F1 solidity ≥ 0.99,
F2 sentinel box) is superseded on both counts: the threshold moves to 0.995 and the sentinel box
becomes the enclosure test. That spec re-scores existing predictions and can be re-run under
these criteria; its conclusions about *which* slices are filtered will shift slightly.

## Out of scope

GMagH5 time rebase and the `fs ≥ 488 Hz` gate (item 1); dropping the 79 unreadable shots and any
shot-level keep-fraction threshold (item 2); retraining M0/M1/M2 (item 4); whether the model
predicts the center (retrain spec); the IMAS pipeline, whose `lcfs_rho` is precomputed upstream
and carries no `(R,Z)` to gate.
