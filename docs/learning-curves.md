# Learning curves and the lot cost engine

*Part of the [cost-core README](https://github.com/MichaelFowler1/cost-risk-toolkit#readme).*

## Learning curves from your own lot data

**Fit a curve to your own lot data in one command:**

```bash
ce-core fit-lots --csv my_lots.csv --dollar-year 2026 --out results/
```

```csv
lot,units,cost
LRIP 1,22,96800000
LRIP 2,18,70200000
FRP 1,25,90000000
```

Two columns is the whole input. [Jump to the details](https://github.com/MichaelFowler1/cost-risk-toolkit/blob/main/docs/learning-curves.md#fitting-a-curve-to-your-own-lot-data),
including the four things that quietly ruin a lot fit and how the tool checks
for each one.

![Learning curve forecast and Monte Carlo cost risk](https://raw.githubusercontent.com/MichaelFowler1/cost-risk-toolkit/main/docs/hero.png)

*Real output. `cost_core` fits an 85% Wright learning curve and forecasts future
lots (left), then runs a 10,000 iteration Monte Carlo total cost simulation with
P50/P80/P90 thresholds (right). Regenerate with `python make_hero.py`, which
reads a local `data.csv` that isn't committed, since `.gitignore` excludes
`*.csv`.*

> **No proprietary data is committed to this repository.** You supply your
> own for `fit-lots`, and nothing you pass in gets stored here. The synthetic
> pipeline and most of the test suite run on a seeded generator producing
> invented programs in the *shape* of CADE submissions. The one exception is
> public: `cost_core.public` reads DoD's released Selected Acquisition Reports,
> and its tests carry the text of a few pages from five of them, which are
> U.S. government works in the public domain.

## Quick start

### With your own data

```bash
ce-core fit-lots --csv my_lots.csv --dollar-year 2026 --forecast "30,40" --out results/
```

Costs come back in the units they went in, because the fit is scale free and
this command converts nothing. Headings that read `($K)` are the engine's own
column names, carried over from the desktop tool whose input column is
`AUC ($K)`, and the S-curve formats the same numbers as dollars, so read both
as "cost" in your file's units unless that file really is in thousands.

Prints the fitted slope and first unit cost, the standard error and CV, an
interval on the slope, and a per lot percentage error showing which lots the
curve misses. With `--out` it also writes those tables as CSV and an
`ASSUMPTIONS.md`; the one chart this command draws is the buy S-curve, and that
needs `--simulate`. See
[Fitting a curve to your own lot data](https://github.com/MichaelFowler1/cost-risk-toolkit/blob/main/docs/learning-curves.md#fitting-a-curve-to-your-own-lot-data).

### With generated data, to see the whole pipeline

Generate synthetic submissions, ingest and normalize them, fit a learning curve
and a CER, simulate with correlation, then write charts, tables and an
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

## The lot cost engine

The lot cost engine is `cost_core.lotmodel`, the code the desktop tool runs
on. Historical **analogy
lots** (fiscal year, quantity, unit cost) are the history; **estimate lots**
(fiscal year, quantity, complexity factor) are the buy being priced.

```python
from cost_core.lotmodel import run_lot_cost_model, generate_analyst_summary, enrich_run

projections, ctx = run_lot_cost_model(analogy_df, estimate_df)
summary = generate_analyst_summary(ctx, {"Program": "TEST"})
extras = enrich_run(ctx, projections, summary)
```

Three models get fitted to the analogy lots, and every estimate lot is priced
under all three, so the projections carry the models the tool *didn't* pick
right alongside the one it did:

```
LC        ln(cost) = ln(T1) + b*ln(lot midpoint)
Rate      ln(cost) = ln(T1) + c*ln(lot quantity)
LC+Rate   both terms together
```

Selection goes to LC+Rate when its rate coefficient is significant, to Rate when
the rate slope is significant *and* beats LC by more than the AICc tie
threshold, and to LC otherwise. Where AICc disagrees with the significance gate,
the summary says so instead of hiding it. Because the lot midpoint depends on
the slope you're fitting, the fit iterates to a fixed point. That's the Goal Seek
the original workbook did by hand.

### What the statistics layer adds

The estimate itself is untouched by any of this. A golden master test fails if a
single coefficient moves. What `enrich_run` reports is how much confidence
those numbers can carry.

**Retransformation bias.** The fit is OLS on `ln(cost)`, then exponentiated back.
That estimates the *median* and understates the *mean* by `exp(s²/2)`. MUPE and
ZMPE refit the same regressors under a proportional error loss and drive the
mean percentage error to zero, so the bias gets measured on your data instead of
argued about.

**Influence.** Six analogy lots is a normal sample here, and at that size one lot
can set the slope while every summary statistic still looks healthy. Leverage and
Cook's distance name it. On the reference programme in the tests, analogy lot 1
carries leverage 0.77 and Cook's D 4.00.

**Prediction intervals** on every projected lot. For a *new* lot, carrying the
residual scatter, with a t multiplier because sigma is estimated rather than
known.

**Buy risk.** A distribution over the total of the estimate lots with P50/P80/P90
and where the point estimate falls on it. Residuals across lots are correlated at
0.30 by default, for the same reason WBS elements are.

`enrich_run` returns the four tables (`Fit_Methods`, `Influence`,
`Prediction_Intervals`, `Buy_Risk`) as frames. The desktop tool writes the risk
ones into its workbook after the original three sheets, so an analyst who wants
only the original three still gets exactly those.

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
ce-core fit-lots --csv my_lots.csv --dollar-year 2026 --forecast "30,40" --out results/
```

The `lot` column is optional. Common header spellings (`Qty`, `Quantity`,
`Total Cost`, `Amount`, and so on) are recognized automatically, and anything
unusual you name with `--units-col` / `--cost-col`. Currency formatting like
`$1,200,000` parses fine. `.xlsx` works out of the box.

Everything else is derived. Lot 1 is units 1 to 22, lot 2 is units 23 to 40, and
so on by running total. That's what turns a flat list of lots into positions on a
curve. You get the fitted slope and first unit cost, standard error and CV, an
interval on the slope, a per lot percentage error showing which lots the curve
misses, prediction intervals on forecast lots, and an `ASSUMPTIONS.md`.

### Re-using the curve on another program

The fit is also an estimating relationship you can lift and apply somewhere else.
The equation gets printed and written to `equation.csv`. The four lots above
select LC+Rate, so theirs carries both terms:

```
Unit Cost = 7,424,501.70 * midpoint^(-0.108355) * qty^(-0.093348)
```

Where only the learning term survives the significance gate the `qty` factor is
absent. Drop `FRP 2` from that sample and the remaining three lots select LC,
printing `Unit Cost = 5,586,219.86 * midpoint^(-0.108548)`. The priced lots carry
whichever terms the equation does. There's a test for exactly that, because for a
while they didn't.

`--price-lots` applies the selected model to any buy profile from unit 1,
producing the learning curve table an analyst would build by hand:

```bash
ce-core fit-lots --csv my_lots.csv --dollar-year 2026 --price-lots "10,15,20,25,30" --out results/
```

The `lot_midpoint` column is the *algebraic* midpoint, meaning the unit whose
cost equals the lot average. Most tools approximate it, because they only have an
approximate lot average to work from. Here the lot average is exact, so the
midpoint gets solved for directly. There's a test asserting the cost at the
midpoint equals the lot average, which is its definition.

This is the analogy use case: price a program with no cost history of its own
using the slope from one that does. **Whether that's valid is a judgement, not a
result.** The slope carries across only if the two programs are comparable in
product, process, rate and contractor. Nothing in the data can confirm that, so
the assumptions log records it as an untested assumption and notes that the extra
error it introduces isn't in any interval reported.

### Forecasting the next buy, with risk

`--forecast` prices future lots continuing from the last unit built, with
prediction intervals. `--simulate` then Monte Carlos them:

```bash
ce-core fit-lots --csv my_lots.csv --dollar-year 2026 --forecast "30,40" --simulate 50000 --out results/
```

Unlike the WBS level simulator, this needs no elicited distributions. The
uncertainty is *measured from the program's own history*. Two sources get
propagated: parameter uncertainty in the fitted slope and T1, which dominates on
a short series, and lot to lot scatter, which is what makes the answer a
prediction about a real lot. Residuals across future lots are correlated at 0.30
by default for the same reason WBS elements are. Consecutive lots share a
workforce and a schedule, and treating them as independent understates the spread
of the whole buy.

It does *not* include schedule risk, requirement changes, or rate changes the
history never saw. That's a narrower claim than a full risk model, and the log
says so.

### How it fits

Three candidate models against the lot midpoint, one selected:

```
LC        ln(unit cost) = ln(T1) + b*ln(lot midpoint)
Rate      ln(unit cost) = ln(T1) + c*ln(lot quantity)
LC+Rate   both terms together
```

This engine came across from the desktop tool in lot-cost-model, which since
its 3.0.0 imports it rather than keeping a copy, so the two are the same code by
import rather than by descent. All three models get fitted
and all three price every lot, so the alternatives stay on the record. Because
the midpoint depends on the slope you're fitting, the fit iterates to a fixed
point.

### Four things that quietly ruin a lot fit

The tool checks all four, because each one leaves a fit that looks perfectly
healthy while the slope is several points wrong.

**Nonrecurring cost in the totals.** Nonrecurring is front loaded, so including it
makes early lots look expensive and the curve reads steeper than the production
process really is. That overstates future savings. `--cost-basis` has to be
declared, and `total` warns.

**Escalation left in "constant" dollars.** `--dollar-year` is **required**. No
index gets applied, but constant dollars are constant relative to a year, and an
output nobody can place in a year can't be escalated or compared. The tool checks
whether cumulative average cost ever rises, which can't happen on a learning
curve, and it also tests the log log residuals for a bend.

Be careful about what that second check can actually do. Fitted against the lot
midpoint it's a poor escalation detector: the fitted slope moves with the
escalation, the midpoint moves with the slope, and the trend gets absorbed
instead of being left in the residuals. It catches a rate break, a design change
or a production gap, not moderate escalation. The level check needs roughly 10% a
year before it bites. Below that, moderate escalation and genuinely slower
learning can't be told apart without a fiscal year attached to each lot, and the
log says exactly that. There's a parametrized test at 2%, 4% and 6% asserting the
miss, so the limit is documented rather than discovered later.

**Lots that don't start at unit 1.** If the program has a prior buy the curve has
already learned through, `--first-unit` shifts the series. Otherwise the fitted
first unit cost describes a unit nobody built.

**Too few lots.** Degrees of freedom are lots minus 2. Two lots interpolate
exactly and are refused. Three gives one degree of freedom and an interval too
wide to support a decision. Five is the practical floor. The tool fits below that
but says so loudly.

## Usage guide

### Full run

```bash
ce-core full-run --out artifacts/ --seed 7 --iters 50000 --theory crawford --method mupe --correlation 0.30
```

`--clean` generates data with no reporting pathologies, which is how the tests
show that the pipeline recovers the generating truth exactly.

### Fit a learning curve and forecast lots

```bash
ce-core fit-lots --csv my_lots.csv --dollar-year 2026 --forecast "32,64,128" --out results/
```

`ce-core template lots` writes a file to start from. The older `fit-curve` and
`forecast` commands did a cut-down version of this in two steps; they still
run, but they're deprecated and will go in 3.0.

### Simulate cost risk

`ce-core cost-risk` takes an estimate in Excel, one WBS element per row with
its range, plus risks and correlations, and gives the S-curve, the confidence
levels and the drivers (`ce-core demo cost-risk` shows it). The older
`simulate` command, which multiplied one uncertain unit cost by one uncertain
quantity, is deprecated and will go in 3.0.

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
