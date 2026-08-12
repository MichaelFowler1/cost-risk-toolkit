# cost-risk-toolkit

[![tests](https://github.com/MichaelFowler1/cost-risk-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/MichaelFowler1/cost-risk-toolkit/actions/workflows/tests.yml)

A Python library and CLI for defense cost estimating. Point it at your own
production history — units and cost for each lot — and it fits a learning
curve, tells you which lots it misses, forecasts the next buy with prediction
intervals, and writes down every assumption it made. It also carries a full
synthetic CSDR/SRDR pipeline, parametric CERs, and correlated Monte Carlo risk
analysis.

**Fit a curve to your own lot data in one command:**

```bash
ce-core fit-lots --csv mylots.csv --dollar-year 2026 --out results/
```

```csv
lot,units,cost
LRIP 1,22,96800000
LRIP 2,18,70200000
FRP 1,25,90000000
```

Two columns is the whole input. [Jump to the details](#fitting-a-curve-to-your-own-lot-data),
including the four things that quietly ruin a lot fit and how the tool checks
for each.

![Learning-curve forecast and Monte Carlo cost risk](docs/hero.png)

*Real output — `cost_core` fits an 85% Wright learning curve and forecasts future lots (left), then runs a 10,000-iteration Monte Carlo total-cost simulation with P50/P80/P90 thresholds (right). Regenerate with `python make_hero.py`, which reads a local `data.csv` (not committed — `.gitignore` excludes `*.csv`).*

> **No real or proprietary data is committed to this repository.** You supply
> your own for `fit-lots`; nothing you pass in is stored here. Everything the
> repo ships with — and everything the test suite runs on — comes from a
> seeded generator producing invented programs in the *shape* of CADE
> submissions.

## What it does

| Module | Purpose |
| --- | --- |
| `cost_core.lots` | **Your own data:** units and cost per lot, in CSV or Excel. Derives lot boundaries, fits, and guards the four ways this input silently goes wrong |
| `cost_core.synth` | Seeded synthetic CSDR/SRDR generator: DD 1921, DD 1921-1, DD 1921-2, Cost and Hour Report (FlexFile), Quantity Data Report, SRDR (DD 2630) — with realistic pathologies to clean |
| `cost_core.ingest` | ETL to one normalised long table: WBS crosswalk, base-year normalisation, resubmission dedup, loud validation gates, row-level provenance |
| `cost_core.fitting` | Shared estimator: OLS, MUPE and ZMPE, with delta-method prediction and confidence intervals |
| `cost_core.learning_curve` | Wright (cumulative average) and Crawford (unit) theories, rate breaks, prediction intervals |
| `cost_core.cer` | Parametric CERs: log-log and linear, leverage and influence diagnostics, extrapolation warnings, small-sample guardrails |
| `cost_core.monte_carlo` | Correlated WBS-level risk: Gaussian copula or Iman–Conover, PSD repair, discrete risks, tornado, convergence |
| `cost_core.reporting` | S-curve, tornado, cost improvement curve, CER diagnostics, and the assumptions log |

## Installation

Python 3.11 or higher.

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -e .
```

Excel input needs one extra: `pip install -e ".[excel]"`. CSV works without it.

## Quick start

### With your own data

```bash
ce-core fit-lots --csv mylots.csv --dollar-year 2026 --forecast "30,40" --out results/
```

Prints the fitted slope and first-unit cost, the standard error and CV, an
interval on the slope, and a per-lot percentage error showing which lots the
curve misses. With `--out` it also writes a chart and an `ASSUMPTIONS.md`.
See [Fitting a curve to your own lot data](#fitting-a-curve-to-your-own-lot-data).

### With generated data, to see the whole pipeline

Generate synthetic submissions, ingest and normalise them, fit a learning curve
and a CER, simulate with correlation, and write charts, tables and an
assumptions log:

```bash
ce-core full-run --out artifacts/ --seed 7
```

That writes:

```
artifacts/
  ASSUMPTIONS.md              the written assumptions and provenance log
  charts/                     s_curve, tornado, learning_curve,
                              cer_diagnostics, summary_table  (PNG, 200 dpi)
  tables/                     every table behind those charts, as CSV
  artifacts/source_reports/   the six synthetic submissions
  artifacts/wbs_crosswalk.csv the persisted name crosswalk
  artifacts/inflation_index.csv
```

The same seed reproduces the run exactly.

## Fitting a curve to your own lot data

The simplest way in. Two columns, one row per lot:

```csv
lot,units,cost
LRIP 1,22,96800000
LRIP 2,18,70200000
FRP 1,25,90000000
FRP 2,30,100500000
```

```bash
ce-core fit-lots --csv mylots.csv --dollar-year 2026 --forecast "30,40" --out results/
```

The `lot` column is optional. Common header spellings (`Qty`, `Quantity`,
`Total Cost`, `Amount`, …) are recognised automatically; anything unusual is
named with `--units-col` / `--cost-col`. Currency formatting such as
`$1,200,000` is parsed. `.xlsx` works with `pip install cost_core[excel]`.

Everything else is derived: lot 1 is units 1–22, lot 2 is units 23–40, and so
on by running total. That is what turns a flat list of lots into positions on a
curve. You get the fitted slope and first-unit cost, standard error and CV, an
interval on the slope, a per-lot percentage error showing which lots the curve
misses, prediction intervals on forecast lots, and an `ASSUMPTIONS.md`.

### Re-using the curve on another program

The fit is also an estimating relationship you can lift and apply elsewhere.
The equation is printed and written to `equation.csv`:

```
Unit Cost(x) = 5,899,940.09 * x^(-0.128073)
```

`--price-lots` applies it to any buy profile from unit 1, producing the
learning curve table an analyst would build by hand — lot midpoints included:

```bash
ce-core fit-lots --csv mylots.csv --dollar-year 2026 --price-lots "10,15,20,25,30" --out results/
```

The `lot_midpoint` column is the *algebraic* midpoint: the unit whose cost
equals the lot average. Most tools approximate it, because they only have an
approximate lot average to work from. Here the lot average is exact, so the
midpoint is solved for directly — and there's a test asserting the cost at the
midpoint equals the lot average, which is its definition.

This is the analogy use case: price a program that has no cost history of its
own using the slope from one that does. **Its validity is a judgement, not a
result** — the slope carries across only if the two programs are comparable in
product, process, rate and contractor. Nothing in the data can confirm that,
so the assumptions log records it as an untested assumption and notes that the
extra error it introduces is not in any interval reported.

### Forecasting the next buy, with risk

`--forecast` prices future lots continuing from the last unit built, with
prediction intervals. `--simulate` then Monte Carlos them:

```bash
ce-core fit-lots --csv mylots.csv --dollar-year 2026 --forecast "30,40" --simulate 50000 --out results/
```

Unlike the WBS-level simulator, this needs no elicited distributions — the
uncertainty is *measured from the program's own history*. Two sources are
propagated: parameter uncertainty in the fitted slope and T1, which dominates
on a short series, and lot-to-lot scatter, which is what makes the answer a
prediction about a real lot. Residuals across future lots are correlated at
0.30 by default for the same reason WBS elements are — consecutive lots share
a workforce and a schedule, and treating them as independent understates the
spread of the whole buy.

It does *not* include schedule risk, requirement changes, or rate changes the
history never saw. That's a narrower claim than a full risk model, and the log
says so.

### Four things that quietly ruin a lot fit

The tool checks all four, because each one leaves a fit that looks perfectly
healthy while the slope is several points wrong.

**Nonrecurring cost in the totals.** Nonrecurring is front-loaded, so including
it makes early lots look expensive and the curve reads steeper than the
production process really is — overstating future savings. `--cost-basis` must
be declared and `total` warns.

**Escalation left in "constant" dollars.** `--dollar-year` is **required**: no
index is applied, but constant dollars are constant relative to a year, and an
output nobody can place in a year cannot be escalated or compared. Two checks
run for escalation still in the data — whether cumulative average cost ever
rises, and whether the log-log residuals bend. The second matters more:
escalation must exceed roughly 10%/yr before it turns the cumulative average
upward, whereas the bend is detectable from about 2%. Neither can separate
moderate escalation from genuinely slower learning without a fiscal year per
lot, and the log says so.

**Lots that don't start at unit 1.** If the programme has a prior buy the curve
has already learned through, `--first-unit` shifts the series. Otherwise the
fitted first-unit cost describes a unit nobody built.

**Too few lots.** Degrees of freedom are `lots − 2`. Two lots interpolate
exactly and are refused; three gives one degree of freedom and an interval too
wide to support a decision; five is the practical floor. The tool fits below
that but says so loudly.

## Usage guide

### Full run

```bash
ce-core full-run --out artifacts/ --seed 7 --iters 50000 --theory crawford --method mupe --correlation 0.30
```

`--clean` generates data with no reporting pathologies, which is how the tests
demonstrate that the pipeline recovers the generating truth exactly.

### Fit a learning curve

Takes a CSV with `program`, `lot`, `unit_quantity` and `unit_cost` columns.

```bash
ce-core fit-curve --csv your_history.csv --out model_params.json
```

### Forecast future lots

```bash
ce-core forecast --model model_params.json --quantities "32,64,128" --out forecast.csv
```

### Run a Monte Carlo simulation

*In PowerShell, wrap the JSON arguments in single quotes.*

```bash
ce-core simulate --n-iter 10000 --unit-cost-dist '{"type": "lognormal", "mean": 5.0, "sigma": 0.2}' --quantity-dist '{"type": "triangular", "left": 40, "mode": 50, "right": 75}' --out sim_results.csv
```

### As a library

```python
from cost_core.synth import generate_program
from cost_core.ingest import normalize_program
from cost_core.learning_curve import Theory, fit_from_progress_report

program = generate_program(seed=7)
data = normalize_program(program)                    # raises if a gate fails
curve = fit_from_progress_report(
    data.learning_curve_input(), theory=Theory.CRAWFORD, method="mupe"
)
curve.forecast_lots([[109, 132]], level=0.80, kind="prediction")
```

## Methodological choices

### Why MUPE and ZMPE, not just OLS

The standard cost fit is ordinary least squares in log space, followed by
exponentiating back to dollars. That retransformation is biased. If the
log-space errors are normal with variance `s²`, then

```
E[y | x] = f(x) · exp(s² / 2)
```

so the retransformed value estimates the **median** and understates the
**mean** by a factor of `exp(s²/2)`. On a 30% CV relationship that is roughly
a 4–5% understatement baked into the estimate before any risk analysis starts,
and it runs in the direction that makes a programme look cheaper.

Two unbiased alternatives are provided, both of which drive the mean
percentage error to exactly zero:

- **MUPE** (minimum-unbiased-percentage-error) minimises
  `Σ (y − f)² / f_prev²` by iteratively reweighted least squares. At its fixed
  point, the normal equation for a multiplicative scale parameter collapses to
  `Σ (y − f)/f = 0`.
- **ZMPE** (zero-percentage-bias minimum-percentage-error) minimises
  `Σ ((y − f)/f)²` *subject to* `Σ (y − f)/f = 0` — the same zero-bias
  property imposed as a constraint rather than emerging from the algebra, and
  a different slope.

`retransformation_bias()` measures the bias three ways — the theoretical
`exp(s²/2)`, Duan's nonparametric smearing estimate, and the observed shift
against MUPE and ZMPE — so the correction is quantified rather than asserted.

### Prediction intervals, not confidence intervals

These are not interchangeable and the confusion always runs the same
direction:

- A **confidence interval** covers the *mean response* at a point — where the
  fitted line is. It shrinks toward zero as the sample grows.
- A **prediction interval** covers a *single new observation* — where the next
  actual programme will land. It carries the residual scatter as well as the
  parameter uncertainty.

The variance relationship is exact:

```
Var_prediction = Var_confidence + σ²
```

That extra `σ²` is the spread of programmes about the line, and no amount of
additional data removes it. A cost estimate forecasts one new programme, so
the prediction interval is the correct one; `CER.predict()` takes `kind`
explicitly and defaults to `"prediction"`.

### Why correlation matters

Sampling WBS elements independently is the spreadsheet default and close to
the worst assumption available. Elements on one programme share a workforce, a
management chain, a supply base and a schedule — when one runs late they
mostly all run late. The variance of a sum is

```
Var(Σ Xᵢ) = Σ Var(Xᵢ) + 2 · Σ_{i<j} ρᵢⱼ · sdᵢ · sdⱼ
```

so for *k* equally variable elements at a common ρ, ignoring correlation
understates the variance of the total by exactly `1 + ρ(k−1)`. Ten elements at
ρ = 0.3 is a factor of **3.7 in variance** — nearly a doubling of the standard
deviation — and it lands on the upper tail, which is where the P80 lives.

Because independence is so rarely right, `RiskModel` applies a non-zero
default correlation when none is supplied, and **warns that it did so**. A
default is an assumption; an unstated assumption is the failure the
documentation exists to prevent. `correlation_impact()` reports the measured
and the closed-form inflation side by side, so the claim does not rest on the
simulation alone.

### Standard error and CV, not R²

R² measures how tightly points hug the fitted line, which a *wrong* model can
do perfectly well. In `tests/test_learning_curve.py`, data generated under
Crawford unit theory and fitted as a Wright curve returns R² ≈ 0.996 with a
demonstrably wrong forecast. Standard error is in dollars and CV is a
proportion; both are arguable. R² is reported, but last.

### Wright and Crawford are different theories

Wright's cumulative-average form says the *average* cost of the first x units
follows `T1·x^b`; Crawford's unit form says the cost of *unit* x does. Which
applies is a property of the production process, not a modelling preference.
`fit_curve()` makes the caller choose and `compare_theories()` reports both,
because the same data under the two gives materially different forecasts.

## Mapping to the GAO Cost Estimating and Assessment Guide

Every run emits an `ASSUMPTIONS.md` organised around the four characteristics
of a reliable estimate.

| Characteristic | How this library addresses it |
| --- | --- |
| **Comprehensive** | All six report shapes ingested; recurring and nonrecurring cost, five functional categories and discrete risks all modelled; the WBS crosswalk surfaces unmatched elements rather than dropping them |
| **Well-documented** | Row-level provenance from every output number to its source submission; the crosswalk and inflation index are persisted artifacts, not inline logic; the assumptions log separates what was measured from what was assumed, and counts the assumptions |
| **Accurate** | Validation gates reconcile row counts and dollar totals within and across reports and fail the run when they disagree; retransformation bias is measured and corrected; estimating methods are compared rather than assumed |
| **Credible** | Prediction intervals on every forecast; leverage and influence diagnostics on the CER; extrapolation flagged including hidden extrapolation; the correlation assumption's effect quantified against independence; P80 convergence checked |

## Project structure

```
cost_core/
  fitting.py          shared OLS / MUPE / ZMPE estimator and intervals
  lots.py             your own lot data: units and cost per lot
  learning_curve.py   Wright and Crawford theories, rate breaks
  monte_carlo.py      correlated risk simulation
  data_io.py          CSV and SQLite loading
  synth/              synthetic CSDR/SRDR generator
  ingest/             crosswalk, inflation, normalisation pipeline
  cer/                parametric CERs and diagnostics
  reporting/          charts, assumptions log, end-to-end run
cli/                  the ce-core command line interface
tests/                property tests — see below
```

## Tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -q
```

474 tests, run on Python 3.11 and 3.12 on every push. They assert mathematics
against closed-form answers rather than against recorded output. The strongest
ones:

- **Our OLS *is* the textbook OLS.** The generic estimator reproduces
  `scipy.stats.linregress` and the normal equations to machine precision, and
  the delta-method prediction interval reduces algebraically to
  `s·√(1 + 1/n + (x₀−x̄)²/Sxx)`.
- **MUPE and ZMPE drive the mean percentage error to exactly zero.** That is
  what their names mean, and it is asserted to 1e-9. ZMPE's sum of squared
  percentage errors is also proven to be no larger than MUPE's — a theorem,
  not a tuning outcome.
- **Cook's distance is checked against an actual leave-one-out refit.** The
  closed form is exact, so the test drops each programme, refits, and confirms
  the formula reproduces the movement in the fitted surface.
- **Variance inflation is exactly `1 + ρ(k−1)`.** Asserted across element
  counts and correlations, and confirmed against simulation.
- **Tornado variance shares sum to exactly one**, because the covariance
  decomposition `Var(T) = Σ Cov(Xᵢ, T)` is an identity when T is the sum.
- **A messy program normalises back to the generating truth to the cent.**
  Name drift, mixed then-year and base-year dollars, resubmitted periods and a
  mid-programme quantity change are all reversible by construction, so the
  pipeline that reverses them has no excuse for landing anywhere else.
- **Learning-curve identities are definitional.** Doubling quantity multiplies
  the right quantity by the slope under each theory; Wright's unit costs
  telescope back to its cumulative total; Crawford's lot cost is the exact sum
  of its units.
- **Simulations are seed-deterministic.** A P80 that moves between runs is not
  a number you can put in front of anyone.
- **Escalation detection is tested for its limits, not just its successes.**
  There is a test asserting that the rising-cumulative-average check *misses*
  4%/yr escalation — because it does, returning an 88.9% slope against a true
  85% with an R² of 0.99. That is why the curvature test exists, and there are
  tests that it fires from 2%/yr and stays silent on ordinary scatter.
- **Bad input is refused, not absorbed.** Zero degrees of freedom, a missing
  base year, two lots, fractional units, unmatched WBS names, non-positive
  costs in a log fit, a correlation matrix that is not symmetric, a rate break
  beyond the data, an unknown interval kind, an index asked for a year it does
  not cover — each raises rather than producing a plausible-looking number.
