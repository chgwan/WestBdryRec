# Input ablation: full vs -Pha* vs -Pha*-Pow* (rope_time M3)

Spec: `ablation_inputs.md`. All arms m3b ActSeqAttn, `--pe rope_time`,
NpzGeom, seeds {0,1,2}. The **full** arm reuses the dropout study's
f=0 rope_time runs (identical config + inputs + seed contract).

## Per-run results

| arm | n_ch | seed | CCC | R2 | RMSE cm | abs bnd mm | n |
|---|---|---|---|---|---|---|---|
| full | 18 | 0 | not run | | | | |
| full | 18 | 1 | not run | | | | |
| full | 18 | 2 | not run | | | | |
| nopha | 16 | 0 | 0.98886 | 0.8986 | 0.8173 | 14.21 | 76 |
| nopha | 16 | 1 | 0.98945 | 0.9039 | 0.7959 | 16.97 | 76 |
| nopha | 16 | 2 | 0.99054 | 0.9141 | 0.7525 | 14.64 | 76 |
| nopha_nopow | 11 | 0 | 0.98637 | 0.8756 | 0.9052 | 16.08 | 76 |
| nopha_nopow | 11 | 1 | 0.98219 | 0.8368 | 1.0368 | 17.21 | 76 |
| nopha_nopow | 11 | 2 | 0.98735 | 0.8847 | 0.8715 | 17.13 | 76 |

## Per-arm summary (pooled bench CCC, mean over seeds)

| arm | n_ch | mean CCC (SD) | mean RMSE cm (SD) |
|---|---|---|---|
| full | 18 | 0.98857 (0.00208) | 0.8253 (0.0786) |
| nopha | 16 | 0.98962 (0.00085) | 0.7885 (0.0330) |
| nopha_nopow | 11 | 0.98530 (0.00274) | 0.9378 (0.0873) |

Analysis (paired per-shot, seed floor, clean-75): `exploration/input_ablation_analysis.py`.

## Analysis (2026-08-19, clean-75 — 57486 flagged, per the dropout-study correction)

| arm | n_ch | pooled CCC (SD) | pooled MSE (SD) | radii MSE | centre MSE |
|---|---|---|---|---|---|
| full | 18 | 0.99520 (0.00006) | 0.05047 (0.00071) | 0.05053 | 0.04954 |
| nopha | 16 | 0.99544 (0.00010) | 0.04802 (0.00150) | 0.04823 | 0.04458 |
| nopow | 11 | 0.99552 (0.00009) | 0.04688 (0.00114) | 0.04716 | 0.04247 |

Matched-seed paired per-shot tests vs full (3 seeds x 75 shots, Wilcoxon):
every comparison is INSIDE the seed floor, with p = 0.07–0.95 and shot-level
wins at 31–43/75 — coin flips throughout.

### Verdict

**The heating channels carry no measurable information for this model.**
Dropping the two LH phases (nopha) or all seven heating channels — the entire
LHCD + ICRH input group (nopow: Ip reference + 10 PF currents only) — leaves
pooled CCC unchanged to the 4th decimal and, if anything, marginally LOWERS
the loss (centre MSE −15 %, radii −7 %), though every difference is inside
the seed floor: the honest claim is "not needed", not "removal improves".
The magnetic inputs alone (11 of 18 channels) are sufficient.

Reading, with the target's provenance in mind: `GMAG_BND` is itself an
equilibrium reconstruction driven by the magnetic circuit, so PF + Ip being
(near-)sufficient is physically consistent — heating enters only through the
pressure balance it induces. For deployment this is a parsimony win: the
strict-actuator input set can shrink from 18 to 11 channels at no measured
cost, matching the V1-era "parsimony, not accuracy" finding.

Analysis: `exploration/input_ablation_analysis.py`, log
`logs/input_ablation_analysis.txt`. Arms: `ab_nopha_s{0,1,2}`,
`ab_nopha_nopow_s{0,1,2}` (configs `dcs_model_attn_nopha{,_nopow}.yml`);
full arm = the dropout study's `drop_f0.0_rope_time_s{0,1,2}` (f=0).
