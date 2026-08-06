# newTrain retrain results

Dataset `NpzGeom` (GMAG-native time base, S0-S5 filters, per-slice centre,
34 outputs = r(theta)@32 + absolute (Rgeom, Zgeom)).
Baseline is `NpzOrigin` / `dcs_actuator`, 76 test shots.

Five things differ from the baseline at once (time base, slice population,
target origin, input columns, output width), so a difference in the r(theta)
numbers is **not attributable to any single one**. Run B isolates only the PE.
A drop is not necessarily a regression: the baseline was partly scored on
interpolated boundaries and on slices these filters reject.

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

| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 0.9519 | 0.5523 | 1.7174 | 5.8035 | 4.4158 | 25.1923 | 30.7964 | 76 | 0.9579 / 0.8452 |
| m1 | 0.9454 | 0.4835 | 1.8446 | 6.8898 | 4.8853 | 26.5932 | 32.4587 | 76 | 0.9530 / 0.8284 |
| m2 | 0.9761 | 0.7805 | 1.2025 | 6.8827 | 4.0815 | 17.3323 | 20.4465 | 76 | 0.9668 / 0.8763 |

## dcs_actuator_geom_nope — no time PE (ablation)

| model | CCC | R² | RMSE cm | Rgeom MAE mm | Zgeom MAE mm | centre RMSE mm | abs bnd RMSE mm | n | baseline CCC / R² |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 0.9508 | 0.5416 | 1.7378 | 5.7871 | 4.4174 | 25.3110 | 30.8972 | 76 | 0.9579 / 0.8452 |
| m1 | 0.9520 | 0.5528 | 1.7164 | 6.9171 | 4.4172 | 22.6088 | 28.4067 | 76 | 0.9530 / 0.8284 |
| **m2** | **0.9856** | **0.8686** | **0.9305** | **6.6229** | **3.8503** | **14.8873** | **17.1015** | **76** | 0.9668 / 0.8763 |

> **Best overall — `dcs_actuator_geom_nope/m2` (bolded):** CCC 0.9856 beats the baseline
> (0.9668), R² 0.869 matches it (0.876), and it is the only unit under 1 cm RMSE with the
> lowest centre / absolute-boundary error. This is the recommended deployment config.

