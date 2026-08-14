# newTrain retrain results

Baseline is `NpzOrigin` / `dcs_actuator`, 76 test shots.
34 outputs = r(theta)@32 + absolute (Rgeom, Zgeom); S0-S5 slice filters and
the per-slice GMAG_GEOM centre apply to every run below.

Several things differ from the baseline at once (time base, slice population,
target origin, input columns, output width), so a difference in the r(theta)
numbers is **not attributable to any single one**. The `_nope` runs isolate
the PE; the `uni500` pair isolates the uniform axis against `geom`.
A drop is not necessarily a regression: the baseline was partly scored on
interpolated boundaries and on slices these filters reject.

On `NpzUni500` the target itself is interpolated onto the lattice, so the
S0-S5 filters there judge interpolated geometry -- see `src_gap_ms` and
`meta.json:grid.fabricated_valid_slices` for how much.

## Summary — all arms

One table per arm. CCC is the primary metric (pooled over 76 test shots). The
Δt-ablation rows are marked † — that sub-experiment reports **mean per-shot** CCC, a
different aggregation (the doc warns it is not directly comparable to the pooled
values; the same no-PE arm is 0.9718 mean-per-shot vs 0.9856 pooled).

| Dataset | Model | PE / variant | CCC | R² | RMSE cm | centre mm | abs-bnd mm |
|---|---|---|---|---|---|---|---|
| **Baseline** NpzOrigin | m0 | — | 0.9579 | 0.8452 | — | — | — |
| | m1 | — | 0.9530 | 0.8284 | — | — | — |
| | m2 | — | 0.9668 | 0.8763 | — | — | — |
| **V1** NpzGeom | m0 | abs-PE | 0.9519 | 0.5523 | 1.7174 | 25.19 | 30.80 |
| | m1 | abs-PE | 0.9454 | 0.4835 | 1.8446 | 26.59 | 32.46 |
| | m2 | abs-PE | 0.9761 | 0.7805 | 1.2025 | 17.33 | 20.45 |
| V1 NpzGeom | m0 | no-PE | 0.9508 | 0.5416 | 1.7378 | 25.31 | 30.90 |
| | m1 | no-PE | 0.9520 | 0.5528 | 1.7164 | 22.61 | 28.41 |
| | **m2** | **no-PE** | **0.9856** | **0.8686** | **0.9305** | **14.89** | **17.10** |
| **V2** NpzUni500 | m0 | abs-PE | 0.9397 | 0.4635 | 1.9316 | 21.57 | 29.02 |
| | m1 | abs-PE | 0.9472 | 0.5308 | 1.8063 | 26.52 | 32.21 |
| | m2 | abs-PE | 0.9667 | 0.7082 | 1.4245 | 16.82 | 21.79 |
| V2 NpzUni500 | m0 | no-PE | 0.9387 | 0.4534 | 1.9497 | 23.71 | 30.94 |
| | m1 | no-PE | 0.9397 | 0.4618 | 1.9347 | 20.83 | 28.42 |
| | m2 | no-PE | 0.9716 | 0.7524 | 1.3121 | 15.87 | 19.97 |
| **V1** NpzGeom | **m3** | **rope_idx** | **0.9874** | **0.8848** | **0.8711** | **14.75** | **16.94** |
| | m3 | rope_time | 0.9828 | 0.8421 | 1.0199 | 13.41 | 16.72 |
| | m3 | upe_idx | 0.9861 | 0.8733 | 0.9136 | 14.92 | 17.03 |
| | m3 | upe_time | 0.9812 | 0.8283 | 1.0635 | 16.70 | 19.34 |
| | m3 | upe_both | 0.9748 | 0.7680 | 1.2364 | 18.35 | 21.93 |
| **V2** NpzUni500 | m3 | rope_idx | 0.9806 | 0.8321 | 1.0806 | 15.72 | 18.82 |
| | m3 | rope_time | 0.9763 | 0.7939 | 1.1972 | 14.02 | 18.44 |
| | m3 | upe_idx | 0.9760 | 0.7919 | 1.2029 | 15.35 | 19.19 |
| | m3 | upe_time | 0.9755 | 0.7872 | 1.2164 | 16.50 | 20.03 |
| | m3 | upe_both | 0.9748 | 0.7813 | 1.2333 | 15.17 | 19.04 |
| V1 NpzGeomDt † | m2 | no-PE | 0.9718 | — | — | — | — |
| | m2 | abs-PE | 0.9692 | — | — | — | — |
| | m2 | Δt | 0.9691 | — | — | — | — |
| | m2 | abs-PE + Δt | 0.9730 | — | — | — | — |

**Best overall:** `m3 / V1 NpzGeom / rope_idx` — CCC 0.9874, the only arm under 0.9 cm
RMSE, lowest abs-boundary (16.9 mm); beats the best GRU (`m2 / V1 / no-PE`, 0.9856) on
every metric, but m3 is 160× the params (capacity, not architecture — see
[§ the windowed-attention PE matrix](#the-windowed-attention-pe-matrix-m3)). **Native >
uniform** at every model/PE (m2 0.9856 > 0.9716; m3 rope_idx 0.9874 > 0.9806), robustly
p≈1e-12 ([§ V2 summary](#v2-summary--uniform-500-hz-time-base-npzuni500)). **PE ≈ no-PE**
for the GRU and **index ≡ time** for m3 (both paired-null) — the trained-on axis is
uniform to 0.34 %, so there is no rate/phase signal to exploit ([§ PE ablation](#the-pe-ablation-parsimony-not-accuracy)).
Snapshot models m0/m1 lag at ~0.95 CCC regardless of dataset/PE (no temporal context).

## Models

All three share the same 18 strict-actuator inputs and predict the same 34 outputs;
they differ in architecture and how much of the discharge each sees.

| model | architecture | temporal context | notes |
|---|---|---|---|
| **m0** | HistGBT snapshot — 34 independent per-output `HistGradientBoostingRegressor`s fit on the single-time-step actuator snapshot | none | weakest; no early stopping (iterative boosting). `train_m0_dcs` |
| **m1** | ResMLP — residual MLP on the same single-time-step snapshot | none | val early-stop. `train_m1_dcs`, `ResMLP` |
| **m2** | ActSeqGRU — linear-time GRU over the **full per-shot actuator series** | **yes** (whole discharge) | strongest; the only model that sees time evolution. `train_m2_dcs`, `ActSeqGRU` |

m0/m1 are snapshot models (one prediction per time-step from that step's actuators
alone); m2 reads the whole actuator series at once. That is why m2 alone recovers a
high R² — about the plasma's own centre `r(θ)` is pure *shape* (position removed),
which needs temporal context to predict well, where the snapshot models cannot.

## dcs_actuator_geom — with time PE (headline)

Dataset: NpzGeom — native GMAG_BND time base (V1)

| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 0.9519 | 0.5523 | 1.7174 | 5.8035 | 4.4158 | 25.1923 | 30.7964 | 76 | 0.9579 / 0.8452 |
| m1 | 0.9454 | 0.4835 | 1.8446 | 6.8898 | 4.8853 | 26.5932 | 32.4587 | 76 | 0.9530 / 0.8284 |
| **m2** | **0.9761** | **0.7805** | **1.2025** | **6.8827** | **4.0815** | **17.3323** | **20.4465** | **76** | 0.9668 / 0.8763 |

## dcs_actuator_geom_nope — no time PE (ablation)

Dataset: NpzGeom — native GMAG_BND time base (V1)

| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 0.9508 | 0.5416 | 1.7378 | 5.7871 | 4.4174 | 25.3110 | 30.8972 | 76 | 0.9579 / 0.8452 |
| m1 | 0.9520 | 0.5528 | 1.7164 | 6.9171 | 4.4172 | 22.6088 | 28.4067 | 76 | 0.9530 / 0.8284 |
| **m2** | **0.9856** | **0.8686** | **0.9305** | **6.6229** | **3.8503** | **14.8873** | **17.1015** | **76** | 0.9668 / 0.8763 |

## dcs_actuator_uni500 — with time PE

Dataset: NpzUni500 — uniform 500 Hz lattice, target interpolated (V2)

| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 0.9397 | 0.4635 | 1.9316 | 5.7946 | 4.1010 | 21.5704 | 29.0215 | 76 | 0.9579 / 0.8452 |
| m1 | 0.9472 | 0.5308 | 1.8063 | 7.7272 | 5.4941 | 26.5238 | 32.2139 | 76 | 0.9530 / 0.8284 |
| **m2** | **0.9667** | **0.7082** | **1.4245** | **6.6576** | **3.7467** | **16.8218** | **21.7883** | **76** | 0.9668 / 0.8763 |

## dcs_actuator_uni500_nope — no time PE

Dataset: NpzUni500 — uniform 500 Hz lattice, target interpolated (V2)

| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 0.9387 | 0.4534 | 1.9497 | 5.8680 | 4.2673 | 23.7136 | 30.9354 | 76 | 0.9579 / 0.8452 |
| m1 | 0.9397 | 0.4618 | 1.9347 | 6.9040 | 4.3727 | 20.8347 | 28.4212 | 76 | 0.9530 / 0.8284 |
| **m2** | **0.9716** | **0.7524** | **1.3121** | **6.5285** | **3.4625** | **15.8720** | **19.9744** | **76** | 0.9668 / 0.8763 |

> **Best overall — `dcs_actuator_geom_nope/m2` (m2 is the best model in every table above; this row is the best across all four):** CCC 0.9856 beats the baseline
> (0.9668), R² 0.869 matches it (0.876), and it is the only unit under 1 cm RMSE with the
> lowest centre / absolute-boundary error. This is the recommended deployment config —
> **on parsimony, not on a demonstrated accuracy win over the PE arm.** Its CCC advantage
> over `dcs_actuator_geom/m2` (0.9856 vs 0.9761) does not survive a paired per-shot test:
> no-PE wins 36 of 76 shots, median Δ = −0.00005, p = 0.93, and the pooled gap traces to a
> single shot. See [§ PE ablation](#the-pe-ablation-parsimony-not-accuracy).

> **V2 (`NpzUni500`, uniform 500 Hz) — negative result.** Interpolating the target onto a
> uniform lattice was meant to fix M2-GRU's unequal-step problem. It cost `r(θ)` shape
> accuracy on every model, while leaving centre prediction unaffected. On the model that
> mattered: `uni500_nope/m2` CCC 0.9716 vs `geom_nope/m2` 0.9856 (−0.014), R² 0.752 vs
> 0.869, abs-boundary 20.0 vs 17.1 mm. Paired per-shot, the native axis wins 66–73 of 76
> shots on CCC at p ≈ 1e-11…1e-13 across all six model × PE comparisons. The cost of
> training on fabricated (interpolated) boundaries outweighed the uniform-step benefit —
> the unequal-step hypothesis is falsified. V1's native-axis `geom_nope/m2` remains best;
> `NpzGeom` stays the recommended dataset. The V2 branch is retained as the controlled
> comparison.

## V2 summary — uniform 500 Hz time base (`NpzUni500`)

**What changed.** V2 rebuilt the dataset on a generated 2.0 ms (500 Hz) lattice
phase-locked to ignitron `t=0`, and linearly interpolated BOTH the actuator inputs
AND the LCFS target onto it — deliberately unlike V1's `NpzGeom`, which keeps the
target on its native, non-uniform `GMAG_BND_time` axis (modal 488 Hz, gaps to ~33 ms).
Those figures describe the **raw** GMAG axis. Inside the S0–S5 kept span — the axis
actually trained on — 99.28 % of intervals are within ±10 µs of 2.048 ms and only
0.0301 % (1,718 of 5,701,947) exceed 3.072 ms; per-shot effective rate spans
486.65–488.29 Hz, a 0.34 % spread (measure with `exploration/attn_axis_uniformity.py`).
So V1's PE, V2's uniform rebuild and the Δt feature were each addressing a 0.03 %
irregularity — which is why V2's interpolation cost dominated.
Everything else is held fixed: the same 18 strict actuators, the S0–S5 slice filters,
the per-slice `(Rgeom, Zgeom)` centre, the 34-column target. So the V2-vs-V1 delta
isolates one variable — the time axis. Dataset: 759 shots, 7,387,826 in-span slices,
99.88 % valid, on the exact 2 ms lattice; it covers the same discharge window as V1
(`meta.json:grid.clip_dropped_n_total` = 2,626,909 is the post-DCS-end GMAG tail,
correctly excluded — no actuator inputs there; 4,688 valid slices were interpolated
across a > 3.072 ms gap, recorded in the `src_gap_ms` array).

**The hypothesis.** M2's GRU is step-indexed, so on V1's non-uniform axis it treats
unequal Δt as equal — a modelling error. A uniform grid was expected to fix that and
lift M2.

**The result — negative.** The uniform axis cost shape accuracy on every model, M2
included (the model it was meant to help):

| model | metric | V1 `geom_nope` (native) | V2 `uni500_nope` (uniform) | Δ |
|---|---|---|---|---|
| m2 | CCC | 0.9856 | 0.9716 | −0.014 |
| m2 | R² | 0.8686 | 0.7524 | −0.116 |
| m2 | abs-boundary RMSE | 17.1 mm | 20.0 mm | +2.9 mm |
| m2 | centre RMSE | 14.9 mm | 15.9 mm | +1.0 mm |

The pattern holds for the PE arms too (`uni500/m2` CCC 0.9667 < `geom/m2` 0.9761) and
for the snapshot models M0/M1.

**Verified paired, per shot.** The rows above are pooled scalars, one per arm, so they
cannot distinguish a consistent effect from a few dominant shots. Re-scoring every run
per test shot and running a Wilcoxon signed-rank test over the 76 shots
(`exploration/v2_paired_justify.py`) confirms the conclusion, and localises it:

| metric | native wins (of 76) | median Δ | p | verdict |
|---|---|---|---|---|
| CCC | **66–73**, all 6 model × PE comparisons | +0.007 … +0.008 | 1e-11 … 1e-13 | native better |
| abs-boundary RMSE | 54–68 | +1.1 … +3.2 mm | 4e-5 … 3e-11 | native better |
| centre RMSE | 32–60 | −0.45 … +1.55 mm | > 0.2 in 5 of 6 | **no difference** (except `pe`/m1) |

The one centre exception is `pe`/m1 (native wins 60/76, +1.55 mm, p = 2e-6); the other
five comparisons, m2 among them, show no significant centre difference.

The tail agrees: per-shot CCC minimum 0.303 (native) vs 0.012 (uniform), and 2 shots
below CCC 0.95 vs 5. An effect this uniform across shots is not seed noise.

**Why — and why only the shape.** Interpolating the LCFS onto the lattice fabricated
boundaries between real reconstructions: smooth, plausible curves that are not themselves
measurements. That damages `r(θ)`, which is exactly what the fabrication invents. The
centre `(Rgeom, Zgeom)` is smooth and slowly varying, so linear interpolation reproduces
it faithfully — and the paired test finds no significant centre penalty for m2 or m0 (for
`nope/m2` the uniform axis is marginally *better*, median Δ −0.45 mm, p = 0.25). The
harm lands mostly where the interpolation adds information that was never measured;
training on those fabricated targets cost more than the uniform-step benefit gained.

**Conclusion.** The unequal-step hypothesis is falsified. V1's native-axis
`dcs_actuator_geom_nope/m2` remains the best config and `NpzGeom` the recommended
dataset. `NpzUni500` is retained as the controlled comparison and a record of the
negative result; do not re-try interpolating the LCFS target onto a uniform grid.
If a uniform axis is ever wanted again, the paired result says where to spend the
effort: resample the *inputs* only, and leave the target on its native reconstruction
times — the centre survived interpolation, the shape did not.

## The PE ablation: parsimony, not accuracy

The four tables above invite reading `geom_nope/m2` (CCC 0.9856) as more accurate than
`geom/m2` (0.9761). Paired per shot, that reading does not hold
(`exploration/v2_paired_justify.py`, Wilcoxon signed-rank, 76 test shots, CCC):

| arm | no-PE wins (of 76) | median Δ | p | reading |
|---|---|---|---|---|
| `geom`/m2 | 36 | −0.00005 | **0.93** | indistinguishable |
| `geom`/m1 | 27 | −0.00088 | 0.007 | **PE is better** |
| `geom`/m0 | 50 | +0.00016 | 0.012 | no-PE marginally better |
| `uni500`/m2 | 45 | +0.00032 | 0.48 | indistinguishable |
| `uni500`/m1 | 48 | +0.00079 | 0.007 | no-PE marginally better |
| `uni500`/m0 | 53 | +0.00017 | 9e-5 | no-PE marginally better |

On the headline unit the two arms are a coin flip, and for `geom/m1` the PE is
*significantly better* shot by shot. The pooled 0.9856 vs 0.9761 gap traces to **one
shot**: only 5 of 76 shots exceed |Δ| > 0.01, and the two largest are +0.170 and +0.042.
Both arms have exactly 2 shots below CCC 0.90 — they differ in how badly one of them
fails (per-shot minimum 0.303 with no PE, 0.133 with it).

**So the case for `_nope` is parsimony and tail behaviour, not accuracy.** The PE buys
nothing measurable on the typical shot, costs 10 input columns, and made one shot
substantially worse. That is enough to prefer it as the deployment config, and not
enough to claim it predicts better. Unlike the time-axis result, this effect size sits
well inside single-seed training variance; 3–5 seeds per arm would be needed to call it
either way. Every run here is one seed.


## Δt ablation — does sample spacing help M2?

The V1 time-PE encoded *absolute* discharge time; it showed no significant accuracy
benefit (nope vs abs-PE, paired p≈0.93). This ablation swaps in **Δt (sample spacing)**
as the time feature on the native `NpzGeom` axis, holding model/hp/split fixed. Two new
M2 arms: `dt` (18 actuators + Δt) and `dtabs` (18 + abs-PE + Δt), trained on `NpzGeomDt`.
Judged by paired per-shot CCC on the 76 test shots (pooled CCC is single-shot-dominated
at this level, as the PE null showed). The mean per-shot CCC column below is NOT the same all-samples aggregation as the headline CCC in the tables above; per-arm ordering can differ, so the decision rests on the paired test, not this context column.

| arm | input | mean per-shot CCC | vs no-PE: wins/76, median ΔCCC, p |
|---|---|---|---|
| no-PE | 18 | 0.9718 | — (baseline) |
| abs-PE | 18 + abs-t | 0.9692 | 40/76, +0.000052, p=0.93 (null, reproduces V1) |
| dt | 18 + Δt | 0.9691 | 31/76, −0.000401, p=0.111 |
| dt+abs | 18 + abs-t + Δt | 0.9730 | vs abs-PE: 22/76, −0.001014, p=5.9e-04 |

**Decision:** `dt` loses — 31/76 wins (<44) with median ΔCCC −0.000401 (<0) versus no-PE,
so **spacing hurts as a feature; test redundancy directly before any attention model**
(per spec §5/§6). This is reinforced, not rescued, on the stacked arm: adding Δt to abs-PE
makes it significantly *worse* (`dtabs` vs `abs`: 22/76, p=5.9e-04 — the only comparison
that clears p<0.05 here). The variable-rate attention model is not warranted: V2 already
falsified the unequal-step hypothesis, and an explicit Δt input hurts M2 whether alone or
alongside abs-PE. The warranted next step is the redundancy test — does Δt merely duplicate
signal the GRU already reads from the actuator series?

## The windowed-attention PE matrix (m3)

A new `m3` tier — `ActSeqAttn`, a pre-norm causal transformer (d256 / 8 heads / 6
layers / FFN 1024, ~4.8 M params) — reads fixed 2048-step windows of the 18 strict
actuators with a 512-step unscored context prefix, and predicts the same 34 outputs as
m0/m1/m2. Positional information enters **only through the architecture** (never as an
input column), as one of five variants spanning {relative, absolute} × {step index,
real time}: `rope_idx`, `rope_time` (rotary on Q/K), `upe_idx`, `upe_time` (additive
absolute table), `upe_both` (both, half the channels each).

This is newTrain V3, with its premise corrected by measurement: the trained-on
`GMAG_BND_time` axis is uniform to 0.34 % inside the S0–S5 span (above), so there is no
variable sampling rate to accommodate — the literal V3 question is moot. The matrix
instead tests whether attention beats recurrence, re-asks the PE question on an
architecture where it could matter, and replicates the native-vs-uniform result.

**Two confounds, held identical across all 10 arms.** (1) 4.8 M params vs m2-GRU's
~30 k is 160×, so m3 > m2 reads as "bigger model with attention", not "attention beats
recurrence". (2) Windowing gives m3 strictly less context than m2, which sees whole
shots. Neither threatens the within-matrix comparisons. Spec:
`docs/superpowers/specs/2026-08-07-windowed-attention-pe-matrix-design.md`; paired
analysis: `exploration/attn_paired_justify.py`.

### m3 on NpzGeom — native GMAG_BND time base (V1)

| PE variant | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | m2-GRU CCC |
|---|---|---|---|---|---|---|---|---|---|
| `rope_idx` | **0.9874** | **0.8848** | **0.8711** | 4.2009 | 2.7145 | 14.7509 | **16.9420** | 76 | 0.9856 |
| `rope_time` | 0.9828 | 0.8421 | 1.0199 | 4.3567 | 2.3769 | 13.4081 | 16.7196 | 76 | 0.9856 |
| `upe_idx` | 0.9861 | 0.8733 | 0.9136 | 4.6030 | 2.9338 | 14.9150 | 17.0317 | 76 | 0.9856 |
| `upe_time` | 0.9812 | 0.8283 | 1.0635 | 5.0887 | 3.2892 | 16.7040 | 19.3376 | 76 | 0.9856 |
| `upe_both` | 0.9748 | 0.7680 | 1.2364 | 4.5449 | 3.2238 | 18.3514 | 21.9333 | 76 | 0.9856 |

### m3 on NpzUni500 — uniform 500 Hz lattice (V2)

| PE variant | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | m2-GRU CCC |
|---|---|---|---|---|---|---|---|---|---|
| `rope_idx` | 0.9806 | 0.8321 | 1.0806 | 4.6554 | 2.9039 | 15.7150 | 18.8161 | 76 | 0.9716 |
| `rope_time` | 0.9763 | 0.7939 | 1.1972 | 4.3341 | 2.5740 | 14.0207 | 18.4410 | 76 | 0.9716 |
| `upe_idx` | 0.9760 | 0.7919 | 1.2029 | 4.6764 | 3.0945 | 15.3488 | 19.1937 | 76 | 0.9716 |
| `upe_time` | 0.9755 | 0.7872 | 1.2164 | 4.8572 | 2.8832 | 16.5019 | 20.0320 | 76 | 0.9716 |
| `upe_both` | 0.9748 | 0.7813 | 1.2333 | 5.2665 | 2.8677 | 15.1733 | 19.0409 | 76 | 0.9716 |

> **m3 edges out m2-GRU on the best arm.** `geom/rope_idx` CCC 0.9874 / R² 0.8848 beats
> `geom_nope/m2` (0.9856 / 0.8686), with the lowest RMSE (0.87 cm) and abs-boundary error
> (16.9 mm) of any arm. But m3 carries 160× the parameters, so this is a capacity
> advantage, not an architecture win (spec §5).

### Verdicts — pre-registered, paired per-shot (76 shots, Wilcoxon + Holm)

Pooled CCC above is context only (single-shot-dominated at this level, as the PE
ablation showed). The decision rests on the paired tests, every V1 effect gated on the
V2 noise floor. On the uniform V2 axis `rope_time≡rope_idx` and `upe_time≡upe_idx`
*numerically*, so their measured ΔCCC **is** the single-seed noise floor:
**|median ΔCCC| = 0.000406** — and the V2 `rope` pair reads p = 0.001 despite identical
encodings, so a naive p<0.05 bar would false-positive here.

| question | test | wins/76 | median ΔCCC | p | Holm | floor gate | verdict |
|---|---|---|---|---|---|---|---|
| index vs time (relative) | P1 `rope_idx` vs `rope_time` | 39 | +0.000006 | 0.91 | n.s. | INSIDE | **null** |
| index vs time (absolute) | P2 `upe_idx` vs `upe_time` | 45 | +0.000087 | 0.49 | n.s. | INSIDE | **null** |
| relative vs absolute | P3 `rope_time` vs `upe_both` (best by val MSE) | 54 | +0.000488 | 0.0059 | sig (≤0.0167) | exceeds (0.000488 > 0.000406) | **weak** |

**Index ≡ time is null** (P1/P2 inside the floor): real time does not beat step count,
relative or absolute. This confirms the trained-on axis is uniform to 0.34 % — there is
no rate information for a time encoding to exploit, and the model is if anything better
off ignoring the 0.03 % drift.

**Relative marginally ≥ absolute** (P3): the one effect that both survives Holm and
clears the noise floor, but its margin over the floor is thin (~20 %), so it is a weak
signal — under windowing, absolute discharge phase gives no measurable benefit over
relative position within the window. P3's arms were selected by val MSE (pre-registered,
to avoid test-peeking): val picked `rope_time`/`upe_both` whereas test CCC favours
`rope_idx`/`upe_idx` — a val/test disagreement that is itself a single-seed artifact,
and a reason to read P3 as suggestive, not definitive. Multi-seed would settle it.

**Native > uniform replicates, robustly and architecture-independently** — the strongest
result here, and the one solid prior finding holding up under attention (the GRU showed
66–73/76, p≈1e-12 then; attention shows 67–73/76, p≈1e-12 now):

| variant | native wins/76 | median ΔCCC | p |
|---|---|---|---|
| `rope_idx` | 73 | +0.0075 | 4.1e-12 |
| `rope_time` | 70 | +0.0073 | 1.4e-12 |
| `upe_idx` | 71 | +0.0076 | 1.6e-12 |
| `upe_time` | 67 | +0.0074 | 6.4e-11 |
| `upe_both` | 69 | +0.0080 | 2.1e-12 |

The effect (~0.0075 median ΔCCC) is ~18× the noise floor. It is not GRU-specific: the
LCFS-interpolation tax of fabricating boundaries between real reconstructions dominates
regardless of architecture.

**Bottom line.** The native axis still wins under attention (overwhelmingly); positional
information barely matters once context is windowed (index ≡ time null, relative
marginally ≥ absolute); and the variable-rate question that motivated V3 is moot because
the trained-on axis is already uniform. `m3` is a stronger predictor than `m2` on the
best arm but only by virtue of capacity. `NpzGeom` remains the recommended dataset;
`rope_idx` is the simplest positional choice that costs nothing measurable.


## Dropout dose-response: rope_idx vs rope_time (2026-08-13)

The PE matrix found `rope_idx ≈ rope_time` on the real axis because that axis is
uniform to 0.34 %, so `t ≈ c·i` and the two encodings are the same rotation. The
dose-response injects random per-step dropout (f ∈ {0, 0.3, 0.6}) to break that
equivalence and tests the encodings where they actually differ — 18 runs, m3b, 3
seeds. Results and verdict: [`newtrain_dropout_results.md`](newtrain_dropout_results.md).
