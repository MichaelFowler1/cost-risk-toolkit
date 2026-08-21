# Lot cost model assumptions and provenance - example_lots

*Generated 2026-08-21T22:42:18+00:00 by `cost_core` on Python 3.14.5 (Windows).*

This document is emitted automatically with every run. It records what was measured, what was assumed, and every validation gate that was applied. Numbers in the accompanying charts and tables come from the same run.

## 1. Source data

- Source: example_lots.csv
- 6 analogy lots covering 161 units, first unit numbered 1
- Cost basis declared: **recurring**
- Quantity definition declared: **unspecified**
- Dollars: constant **FY2026**

## 1.1 Lots as supplied and derived

| lot | units | cost | first unit | last unit | cumulative units | lot average cost | cumulative average cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lot 1 | 22 | 9.68e+07 | 1 | 22 | 22 | 4.4e+06 | 4.4e+06 |
| Lot 2 | 18 | 7.02e+07 | 23 | 40 | 40 | 3.9e+06 | 4.175e+06 |
| Lot 3 | 25 | 9e+07 | 41 | 65 | 65 | 3.6e+06 | 3.954e+06 |
| Lot 4 | 30 | 1.005e+08 | 66 | 95 | 95 | 3.35e+06 | 3.763e+06 |
| Lot 5 | 30 | 9.6e+07 | 96 | 125 | 125 | 3.2e+06 | 3.628e+06 |
| Lot 6 | 36 | 1.116e+08 | 126 | 161 | 161 | 3.1e+06 | 3.51e+06 |

## 2. Dollar basis and escalation

Costs were supplied already normalised to constant FY2026 dollars, as declared by the analyst on ingest. No inflation index was applied by this tool and no escalation assumption is embedded in the fit. Any error in the upstream normalisation passes through unaltered; the fitted slope and first-unit cost are stated in FY2026 dollars and must be escalated before comparison with a budget in any other year.

Two checks were run for escalation left in the data. The first asks whether cumulative average cost ever rises, which it should not on a learning curve. The second tests the log-log residuals for a systematic bend, since escalation compounds with time while learning compounds with log quantity and the mismatch shows up as convexity.

**Findings:**
- Lot(s) Lot 1 exceed the conventional Cook's distance flag and are setting this fit. Confirm each belongs in the sample before relying on the slope.

**Limits of these checks.** The level check only fires once escalation is severe enough to overwhelm learning, which on a typical profile takes about 10% a year. The curvature test does not fill that gap under a midpoint fit: the fitted slope moves with the escalation and the midpoint moves with the slope, so the trend is absorbed rather than left in the residuals, and the quadratic term barely responds. It is a test that these lots are one clean curve, not a test for escalation. Below roughly 10% a year, moderate escalation and genuinely slower learning cannot be told apart without a fiscal year attached to each lot.

## 3. Model selected

**LC+Rate** — Rate coefficient significant (|t| >= 2.0). Note: AICc favors LC at this sample size - state both in the BOE.

**Unit Cost = 7,477,686.37 * midpoint^(-0.110642) * qty^(-0.093588)**

Unit cost in constant FY2026 dollars, with the midpoint the unit whose cost equals the lot average. Because that midpoint depends on the slope being fitted, the fit iterates to a fixed point rather than solving in one pass.

- Learning slope 92.62%, rate slope 93.72%
- T1 7,477,686.37, SEE 0.0081 on the log scale, CV 0.8%
- 6 lots, 3 parameters, 3 degrees of freedom

Three models were fitted and all three priced every lot, so the alternatives are on the record rather than discarded.

## 3.1 Models compared

| Item | LC | Rate | LC+Rate |
| --- | --- | --- | --- |
| Fitted | Yes | Yes | Yes |
| SELECTED |  |  | YES |
| T1 ($K) | 5,897,536.83 | 15,219,515.25 | 7,477,686.37 |
| Learning exponent (b) | -0.127995 | - | -0.110642 |
| Learning curve slope | 91.51% | - | 92.62% |
| Rate exponent (c) | - | -0.444621 | -0.093588 |
| Rate slope | - | 73.48% | 93.72% |
| R2 (log) | 0.9850 | 0.7032 | 0.9978 |
| Adj R2 | 0.9812 | 0.6289 | 0.9963 |
| SEE (log) | 0.0181 | 0.0806 | 0.0081 |
| CV | 1.81% | 8.07% | 0.81% |
| MAPE | 1.30% | 4.44% | 0.42% |
| Mean bias | +0.01% | +0.22% | +0.00% |
| AICc | -32.55 | -14.65 | -13.99 |
| dAICc | 0.00 | 17.89 | 18.55 |

## 3.2 Coefficients

| term | value |
| --- | --- |
| selected_model | LC+Rate |
| selection_note | Rate coefficient significant (\|t\| >= 2.0). Note: AICc favors LC at this sample size - state both in the BOE. |
| equation | Unit Cost = 7,477,686.37 * midpoint^(-0.110642) * qty^(-0.093588) |
| T1_first_unit_cost | 7.478e+06 |
| b_learning_exponent | -0.1106 |
| learning_slope | 0.9262 |
| c_rate_exponent | -0.09359 |
| rate_slope | 0.9372 |
| lots_fitted | 6 |
| parameters | 3 |
| degrees_of_freedom | 3 |
| SEE_log | 0.008069 |
| cv | 0.00807 |
| r_squared_read_last | 0.9978 |

## 3.3 Per-lot fit quality

| lot | units | lot midpoint | lot average cost | fitted unit cost | percent error |
| --- | --- | --- | --- | --- | --- |
| Lot 1 | 22 | 8.688 | 4.4e+06 | 4.408e+06 | -0.1828 |
| Lot 2 | 18 | 31.01 | 3.9e+06 | 3.902e+06 | -0.04469 |
| Lot 3 | 25 | 52.45 | 3.6e+06 | 3.57e+06 | 0.8414 |
| Lot 4 | 30 | 79.98 | 3.35e+06 | 3.349e+06 | 0.01539 |
| Lot 5 | 30 | 110.1 | 3.2e+06 | 3.233e+06 | -1.022 |
| Lot 6 | 36 | 143.1 | 3.1e+06 | 3.088e+06 | 0.4022 |

## 4. Retransformation bias

The engine fits ln(unit cost) by ordinary least squares and then exponentiates back to dollars. That step is biased: with log-space errors of variance s², the retransformed value estimates the *median* and understates the *mean* by exp(s²/2).

On this data that factor is **1.00003**, an understatement of **0.003%** before any risk analysis begins. Duan's nonparametric smearing estimate agrees at 1.00002, so the lognormal assumption is not doing the work.

MUPE and ZMPE refit the same regressors under a proportional-error loss and drive the mean percentage error to zero. MUPE places the curve +0.002% relative to OLS and ZMPE +0.002%.

## 4.1 Fitting methods compared

| Method | T1 ($K) | b (learning) | c (rate) | Mean % error | MAPE | SEE (log) |
| --- | --- | --- | --- | --- | --- | --- |
| OLS | 7.478e+06 | -0.1106 | -0.09359 | 1.627e-05 | 0.00418 | 0.008069 |
| MUPE | 7.478e+06 | -0.1106 | -0.0936 | -4.046e-16 | 0.004176 | 0.00807 |
| ZMPE | 7.477e+06 | -0.1106 | -0.09361 | 8.707e-16 | 0.004168 | 0.00807 |

## 5. Influence

With 6 analogy lots a single lot can set the slope while every summary statistic still looks healthy. Leverage says which lot is unusual in the predictors; Cook's distance says which is actually moving the fit. The conventional flags are 2p/n and 4/n, and they are flags rather than verdicts -- the largest or smallest lot in a sample has high leverage by construction.

## 5.1 Leverage and influence

| Lot | Qty | Actual ($K) | Fitted ($K) | % error | Leverage | Cook's D | DFFITS | High leverage | Influential |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lot 1 | 22 | 4.4e+06 | 4.408e+06 | -0.1828 | 0.9721 | 21.42 | -10.54 | no | yes |
| Lot 2 | 18 | 3.9e+06 | 3.902e+06 | -0.04469 | 0.8402 | 0.03367 | -0.2603 | no | no |
| Lot 3 | 25 | 3.6e+06 | 3.57e+06 | 0.8414 | 0.1871 | 0.1017 | 0.6039 | no | no |
| Lot 4 | 30 | 3.35e+06 | 3.349e+06 | 0.01539 | 0.2269 | 4.601e-05 | 0.009593 | no | no |
| Lot 5 | 30 | 3.2e+06 | 3.233e+06 | -1.022 | 0.2797 | 0.291 | -1.524 | no | no |
| Lot 6 | 36 | 3.1e+06 | 3.088e+06 | 0.4022 | 0.494 | 0.1591 | 0.6166 | no | no |

## 6. Prediction intervals

Each priced lot carries a 80% **prediction** interval: the range a single new lot is expected to fall in, not the range the fitted line lies in. The two differ by exactly the residual variance, and that term does not shrink with more analogy lots. The multiplier is a t on 3 degrees of freedom, because sigma is estimated rather than known.

## 6.1 Priced lots with intervals

| Lot | Fiscal Year | Lot Quantity | Unit Cost ($K) | Unit Cost Lower | Unit Cost Upper | Lot Cost ($) | Lot Cost Lower | Lot Cost Upper | SE (log) | Level | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | <NA> | 22 | 5.887e+06 | 5.779e+06 | 5.997e+06 | 1.295e+08 | 1.271e+08 | 1.319e+08 | 0.01133 | 0.8 | prediction |
| 2 | <NA> | 18 | 5.114e+06 | 5.023e+06 | 5.206e+06 | 9.205e+07 | 9.041e+07 | 9.371e+07 | 0.01095 | 0.8 | prediction |
| 3 | <NA> | 25 | 4.825e+06 | 4.756e+06 | 4.895e+06 | 1.206e+08 | 1.189e+08 | 1.224e+08 | 0.008792 | 0.8 | prediction |
| 4 | <NA> | 30 | 4.605e+06 | 4.538e+06 | 4.673e+06 | 1.381e+08 | 1.361e+08 | 1.402e+08 | 0.008938 | 0.8 | prediction |
| 5 | <NA> | 30 | 4.445e+06 | 4.379e+06 | 4.512e+06 | 1.333e+08 | 1.314e+08 | 1.354e+08 | 0.009128 | 0.8 | prediction |
| 6 | <NA> | 36 | 4.318e+06 | 4.249e+06 | 4.388e+06 | 1.554e+08 | 1.53e+08 | 1.58e+08 | 0.009863 | 0.8 | prediction |

## 8. On R squared

R squared is reported last and should not be used as a validity check on this data. Unit cost falls monotonically against the lot midpoint by construction, so almost any downward-sloping model returns a high R squared. It measures how tightly the points hug the fitted line, which a wrong model can do perfectly well. The standard error of the estimate, the per-lot percentage errors and the influence table are the numbers to argue with.

## Assumptions applied

Each of these was applied by judgement rather than derived from the data. They are the first things a reviewer should push on, and the first things to revisit if the answer looks wrong.

| Topic | Assumption | Basis |
| --- | --- | --- |
| Dollar basis | Costs are constant FY2026 dollars; no inflation index applied by this tool. | Declared by the analyst on ingest. Normalisation, if any was needed, happened upstream and is not verifiable from this input. |
| Quantity definition | A 'unit' means: unspecified. | Declared by the analyst. Delivered, completed and accepted counts differ, and the difference shifts every point on the curve. |
| Lot sequencing | Lots are contiguous and in build order, starting at unit 1. | Implied by supplying lots as an ordered list. A prior buy the model has already learned through would need a higher first unit. |

**3 assumption(s) recorded.**

## GAO Cost Estimating and Assessment Guide

How this run addresses the four characteristics of a reliable estimate.

### Comprehensive

- All 6 reported lots included in the fit; three candidate models fitted and all three priced every lot.

### Well-documented

- Dollar basis, quantity definition and lot sequencing each recorded as declared assumptions with their basis; source data reproduced in full in table 1.1; the selection rule and its outcome stated in section 3.

### Accurate

- Retransformation bias of the log-space fit measured at 0.003% on this dataset and reported against MUPE and ZMPE refits, rather than left in the estimate.

### Credible

- Prediction intervals on every priced lot at 80%, influence diagnostics naming any lot that sets the fit, and a selection note stating why this model was chosen over the other two.
