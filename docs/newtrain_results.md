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
| m2 | 0.9761 | 0.7805 | 1.2025 | 6.8827 | 4.0815 | 17.3323 | 20.4465 | 76 | 0.9668 / 0.8763 |

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
| m2 | 0.9667 | 0.7082 | 1.4245 | 6.6576 | 3.7467 | 16.8218 | 21.7883 | 76 | 0.9668 / 0.8763 |

## dcs_actuator_uni500_nope — no time PE

Dataset: NpzUni500 — uniform 500 Hz lattice, target interpolated (V2)

| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 0.9387 | 0.4534 | 1.9497 | 5.8680 | 4.2673 | 23.7136 | 30.9354 | 76 | 0.9579 / 0.8452 |
| m1 | 0.9397 | 0.4618 | 1.9347 | 6.9040 | 4.3727 | 20.8347 | 28.4212 | 76 | 0.9530 / 0.8284 |
| m2 | 0.9716 | 0.7524 | 1.3121 | 6.5285 | 3.4625 | 15.8720 | 19.9744 | 76 | 0.9668 / 0.8763 |

> **Best overall — `dcs_actuator_geom_nope/m2` (bolded above):** CCC 0.9856 beats the baseline
> (0.9668), R² 0.869 matches it (0.876), and it is the only unit under 1 cm RMSE with the
> lowest centre / absolute-boundary error. This is the recommended deployment config.

> **V2 (`NpzUni500`, uniform 500 Hz) — negative result.** Interpolating the target onto a
> uniform lattice was meant to fix M2-GRU's unequal-step problem, but it hurt every model.
> On the model that mattered: `uni500_nope/m2` CCC 0.9716 vs `geom_nope/m2` 0.9856 (−0.014),
> R² 0.752 vs 0.869, abs-boundary 20.0 vs 17.1 mm. The cost of training on fabricated
> (interpolated) boundaries outweighed the uniform-step benefit — the unequal-step hypothesis
> is falsified. V1's native-axis `geom_nope/m2` remains best; `NpzGeom` stays the recommended
> dataset. The V2 branch is retained as the controlled comparison.

