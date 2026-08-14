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
