# Dropout dose-response: rope_idx vs rope_time

18 runs: f in {0.0, 0.3, 0.6} x {rope_idx, rope_time} x seeds {0,1,2},
all m3b (4,753,698 params) on NpzGeom, 2048-step windows over the
**retained** sequence. Random per-step dropout breaks the affine
equivalence t ~ c*i that made the two encodings identical on the real
axis, so this is the one regime where the question is answerable.

**Interpretation ceiling (spec section 1.4).** A win for `rope_time` here
means *real-time PE helps under sparse/irregular sampling* -- NOT that
timing helps on the real dense axis, where the PE matrix already showed
it does not. Dropout degrades the implicit timing the actuator values
otherwise carry; the honest framing is regime-specific.

The drop mask is deterministic per shot and independent of the training
seed, so at a given f both encodings and all three seeds see identical
retained slices. CCC is over the retained test slices: comparable
between encodings at a given f, **not** across f.

Spec: `docs/superpowers/specs/2026-08-13-dropout-rope-idx-vs-time-design.md`
Plan: `docs/superpowers/plans/2026-08-13-dropout-rope-idx-vs-time-nscc.md`
Analysis: `exploration/dropout_analysis.py`

## Per-run results

| f | pe | seed | CCC | R2 | RMSE cm | centre mm | abs bnd mm | n |
|---|---|---|---|---|---|---|---|---|
| 0.0 | `rope_idx` | 0 | 0.9868 | 0.8798 | 0.8899 | 14.5174 | 16.9613 | 76 |
| 0.0 | `rope_idx` | 1 | 0.9884 | 0.8945 | 0.8335 | 14.1537 | 16.3636 | 76 |
| 0.0 | `rope_idx` | 2 | 0.9882 | 0.8926 | 0.8411 | 14.0069 | 16.2156 | 76 |
| 0.0 | `rope_time` | 0 | 0.9910 | 0.9181 | 0.7346 | 14.2381 | 15.8384 | 76 |
| 0.0 | `rope_time` | 1 | 0.9874 | 0.8856 | 0.8682 | 13.3000 | 15.7332 | 76 |
| 0.0 | `rope_time` | 2 | 0.9873 | 0.8843 | 0.8730 | 14.9540 | 17.3126 | 76 |
| 0.3 | `rope_idx` | 0 | 0.9842 | 0.8557 | 0.9746 | 14.8379 | 17.7638 | 76 |
| 0.3 | `rope_idx` | 1 | 0.9819 | 0.8345 | 1.0440 | 16.0968 | 19.1856 | 76 |
| 0.3 | `rope_idx` | 2 | 0.9826 | 0.8401 | 1.0259 | 15.9018 | 18.8360 | 76 |
| 0.3 | `rope_time` | 0 | 0.9877 | 0.8880 | 0.8589 | 14.7043 | 16.9390 | 76 |
| 0.3 | `rope_time` | 1 | 0.9870 | 0.8816 | 0.8829 | 15.8302 | 18.1415 | 76 |
| 0.3 | `rope_time` | 2 | 0.9862 | 0.8739 | 0.9111 | 15.3189 | 17.7691 | 76 |
| 0.6 | `rope_idx` | 0 | 0.9824 | 0.8390 | 1.0281 | 16.0070 | 18.9701 | 76 |
| 0.6 | `rope_idx` | 1 | 0.9829 | 0.8436 | 1.0132 | 16.8341 | 19.6922 | 76 |
| 0.6 | `rope_idx` | 2 | 0.9759 | 0.7773 | 1.2090 | 17.1205 | 21.0328 | 76 |
| 0.6 | `rope_time` | 0 | 0.9812 | 0.8267 | 1.0665 | 14.6051 | 17.9348 | 76 |
| 0.6 | `rope_time` | 1 | 0.9802 | 0.8181 | 1.0926 | 17.8525 | 21.1691 | 76 |
| 0.6 | `rope_time` | 2 | 0.9721 | 0.7405 | 1.3053 | 15.6212 | 20.2533 | 76 |

## Dose-response (pooled bench CCC, mean over seeds)

| f | rope_idx mean (SD) | rope_time mean (SD) | delta (time - idx) |
|---|---|---|---|
| 0.0 | 0.98781 (0.00087) n=3 | 0.98857 (0.00208) n=3 | +0.00076 |
| 0.3 | 0.98290 (0.00118) n=3 | 0.98697 (0.00075) n=3 | +0.00406 |
| 0.6 | 0.98042 (0.00392) n=3 | 0.97781 (0.00500) n=3 | -0.00261 |

## Gate zero — f=0 must reproduce the published m3b_rope_idx

The gate confirms the dropout code is a no-op at f=0. If it fails, the
f=0 path was perturbed and every downstream number is incomparable.

| seed | published | f=0 rope_idx | delta | within +-0.002 |
|---|---|---|---|---|
| 0 | 0.98738 | 0.98681 | -0.00057 | yes |
| 1 | 0.98946 | 0.98841 | -0.00105 | yes |
| 2 | 0.98519 | 0.98820 | +0.00301 | **NO** |

> The f=0 `rope_time` runs have no 3-seed reference; they should land
> near the PE matrix's single `rope_time` (0.9828) as a loose sanity
> check only.

## Verdict (pre-registered, spec §6.2)

**STRONG NULL.** The median matched-seed ΔCCC (rope_time − rope_idx) stays inside
the per-arm seed floor at every f:

| f | rope_idx mean (SD) | rope_time mean (SD) | median ΔCCC | floor (max SD) | clears |
|---|---|---|---|---|---|
| 0.0 | 0.98781 (0.00087) | 0.98857 (0.00208) | +0.000007 | 0.002082 | no |
| 0.3 | 0.98290 (0.00118) | 0.98697 (0.00075) | +0.000166 | 0.001180 | no |
| 0.6 | 0.98042 (0.00392) | 0.97781 (0.00500) | +0.000165 | 0.004997 | no |

Matched-seed Wilcoxon p-values: f=0.0 median p=4.253e-01; f=0.3 p=6.487e-01;
f=0.6 p=1.902e-01 — none significant. The 9-pairing cross-seed medians agree
(+0.000000 / +0.000199 / +0.000165). Reading: **the model ignores explicit timing
even under genuine irregularity** — knowing the real sampling gaps does not help
once the actuators are given; positional encoding is decorative for this task and
the actuators are everything. This extends the PE-matrix null (39/76, p=0.91 on
the real uniform axis) into the regime where the two encodings actually differ
(~140–489 RoPE cycles of drift per 2048-step window at f=0.3–0.6).

Honest nuances, visible in the tables above: at f=0.3 rope_time leads on 3/3 seeds
and the pooled-mean gap is +0.00406 — larger than the seed SDs — yet the
pre-registered per-shot median ΔCCC (+0.000166) sits inside the floor (0.001180):
the pooled-mean gap is within what slice-composition and seed luck produce, and the
paired per-shot statistic — the one that controls for shot difficulty — does not
clear it. At f=0.6 the arms are statistically indistinguishable in the other
direction (−0.00261 pooled, seeds 0/3). The dose-response is not monotone.

**Interpretation ceiling (spec §1.4).** This is a statement about *sparse/irregular
sampling*, not about the real dense axis (where timing was already known not to
help). It does NOT license "rope_idx is universally better" — at f=0.3 the
pooled means mildly favour rope_time — only "no encoding choice measurably
matters here."

Two facts a later reader needs: **CCC is over the retained test slices**, so values
are comparable between encodings at a given f but **not across f** (different
slices retained); and all 18 runs were trained **on NSCC** (torch 2.12.0, A100-40GB)
under the same run names, with the other machine's earlier 8 rows preserved in
`ProjDB/Stats/dcs_predictor/bench_table_dropout_pre_nscc.csv` (CSV read last-wins,
NSCC rows authoritative).

**Gate-zero disclosure.** The f=0/rope_idx reproduction missed tolerance on seed 2
(+0.00301 vs ±0.002; seeds 0/1 passed). Diagnosis: machine/run drift, not a code
perturbation — the deltas are mixed-sign and seed-dependent; the failing seed
*improved* (worst published seed became best on NSCC); the other machine's own
re-run of seed 2 had already moved +0.00089 from its published value on identical
code and hardware; and the f=0 path is a no-op by construction (`drop_frac<=0`
returns all-keep, the slice never executes) with byte-identical windows pinned by
`test_dataset_f0_reproduces_no_dropout`. Cross-machine drift of this size
(torch 2.12 vs 2.11, A100-40GB vs 80GB) was pre-flagged as observation O2 in the
plan. The between-arm comparison — the study's actual question — is unaffected:
both encodings run on the same machine, torch and seeds, so drift cancels.

## Post-hoc: the loss dose-response (2026-08-15)

> **Post-hoc, not pre-registered.** The §6.2 analysis above was fixed before the
> runs and its verdict stands. This section was added afterwards, at the authors'
> request, because CCC is concordance-only and scale-invariant — a prediction can
> preserve ranking while being systematically off in magnitude, which is exactly
> the error mode dropout should worsen. The metric here is the training objective
> itself: masked MSE over the standardized 34-column target, on the retained test
> slices (per-run `tgt_mean`/`tgt_std` from the artifacts). The same
> seed-SD floor and paired-Wilcoxon methodology is applied. Negative delta =
> `rope_time` better.

### Standardized test MSE

| f | rope_idx MSE (SD) | rope_time MSE (SD) | pooled delta | median per-shot dMSE | floor | time-better seeds | median p |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.12311 (0.00986) | 0.11696 (0.02128) | -0.00616 | +0.000028 | 0.021278 | 1/3 | 0.63 |
| 0.3 | 0.17783 (0.01351) | 0.13316 (0.00884) | **-0.04467** | -0.001948 | 0.013511 | **3/3** | 0.72 |
| 0.6 | 0.19976 (0.04236) | 0.22849 (0.05104) | +0.02872 | -0.001885 | 0.051038 | 0/3 | 0.25 |

### Physical error metrics (from the bench table; lower is better)

| metric | f=0.0 delta | f=0.3 delta | f=0.6 delta |
|---|---|---|---|
| RMSE radii (cm), idx vs time | 0.855 vs 0.825 (**-0.030**) | 1.015 vs 0.884 (**-0.131**) | 1.083 vs 1.155 (+0.071) |
| centre RMSE (mm), idx vs time | 14.23 vs 14.16 (**-0.062**) | 15.61 vs 15.28 (**-0.328**) | 16.65 vs 16.03 (**-0.628**) |
| abs-boundary RMSE (mm), idx vs time | 16.51 vs 16.29 (**-0.219**) | 18.60 vs 17.62 (**-0.979**) | 19.90 vs 19.79 (**-0.113**) |

### Reading

- **The pre-registered verdict is unchanged** — the median matched-seed per-shot
  difference stays inside the seed floor at every f, on MSE as on CCC.
- **But the loss view sharpens the f=0.3 nuance considerably.** Pooled MSE is
  **25 % lower** for `rope_time` at f=0.3 (0.1332 vs 0.1778), better on **3/3
  seeds**, the median per-shot delta is negative in **all 9 cross-seed pairings**,
  and every physical metric agrees (radii -0.13 cm, centre -0.33 mm, abs boundary
  -0.98 mm). CCC's pooled +0.004 understated this: CCC is blind to magnitude, and
  magnitude is where the moderate-dropout advantage lives. Note the gate
  asymmetry: the *pooled* delta (-0.045) exceeds the pooled seed-SD floor (0.0135)
  while the *median per-shot* delta (-0.0019) does not — the typical shot gains
  little, but the arm-level loss is decisively better.
- **At f=0.6 the pooled loss flips** (+0.029, `rope_idx` better on 3/3 seeds, RMSE
  agrees) while the median per-shot delta stays slightly negative — a skew
  signature: at heavy dropout `rope_time` degrades through a heavy tail (some
  shots much worse) rather than uniformly. The f=0.6 regime is also simply
  noisier: seed 2 degraded in *both* arms (MSE 0.25-0.29 vs 0.17-0.20).
- **Net.** The median shot is timing-insensitive at every f (the null), but the
  loss shows `rope_time` buying a real magnitude-accuracy advantage at moderate
  dropout and paying for it at heavy dropout — the "helps up to a point" shape
  (§6.2's third reading), not a universal null. Honest framing per §1.4: this
  says real-time PE helps *under moderate sparse sampling*; it still says nothing
  about the real dense axis.

Analysis: `exploration/dropout_analysis.py` (post-hoc section), log
`logs/dropout_analysis.txt`.
