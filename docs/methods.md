# Methodological choices

*Part of the [cost-core README](https://github.com/MichaelFowler1/cost-risk-toolkit#readme).*

### Why MUPE and ZMPE, not just OLS

The standard cost fit is ordinary least squares in log space, followed by
exponentiating back to dollars. That retransformation is biased. If the log space
errors are normal with variance `s²`, then

```
E[y | x] = f(x) · exp(s² / 2)
```

so the retransformed value estimates the **median** and understates the **mean**
by a factor of `exp(s²/2)`. On a 30% CV relationship that's roughly a 4 to 5%
understatement baked into the estimate before any risk analysis even starts, and
it runs in the direction that makes a program look cheaper.

Two unbiased alternatives are provided, and both drive the mean percentage error
to exactly zero:

**MUPE** (minimum unbiased percentage error) minimizes `Σ (y - f)² / f_prev²` by
iteratively reweighted least squares. At its fixed point, the normal equation for
a multiplicative scale parameter collapses to `Σ (y - f)/f = 0`.

**ZMPE** (zero percentage bias minimum percentage error) minimizes
`Σ ((y - f)/f)²` *subject to* `Σ (y - f)/f = 0`. Same zero bias property, but
imposed as a constraint rather than emerging from the algebra, and it gives a
different slope.

`retransformation_bias()` measures the bias three ways: the theoretical
`exp(s²/2)`, Duan's nonparametric smearing estimate, and the observed shift
against MUPE and ZMPE. So the correction gets quantified instead of asserted.

### Prediction intervals, not confidence intervals

These aren't interchangeable, and the confusion always runs the same direction.

A **confidence interval** covers the *mean response* at a point, meaning where
the fitted line is. It shrinks toward zero as the sample grows.

A **prediction interval** covers a *single new observation*, meaning where the
next actual program will land. It carries the residual scatter as well as the
parameter uncertainty.

The variance relationship is exact:

```
Var_prediction = Var_confidence + σ²
```

That extra `σ²` is the spread of programs about the line, and no amount of
additional data removes it. A cost estimate forecasts one new program, so the
prediction interval is the correct one. `CER.predict()` takes `kind` explicitly
and defaults to `"prediction"`.

### Why correlation matters

Sampling WBS elements independently is the spreadsheet default and close to the
worst assumption available. Elements on one program share a workforce, a
management chain, a supply base and a schedule. When one runs late they mostly
all run late. The variance of a sum is

```
Var(Σ Xᵢ) = Σ Var(Xᵢ) + 2 · Σ_{i<j} ρᵢⱼ · sdᵢ · sdⱼ
```

so for *k* equally variable elements at a common ρ, ignoring correlation
understates the variance of the total by exactly `1 + ρ(k-1)`. Ten elements at
ρ = 0.3 is a factor of **3.7 in variance**, close to a doubling of the standard
deviation, and it lands on the upper tail, which is where the P80 lives.

Because independence is so rarely right, `RiskModel` applies a non zero default
correlation when none is supplied, and **warns that it did so**. A default is an
assumption, and an unstated assumption is the failure this documentation exists
to prevent. `correlation_impact()` reports the measured and the closed form
inflation side by side, so the claim doesn't rest on the simulation alone.

### Standard error and CV, not R²

R² measures how tightly points hug the fitted line, which a *wrong* model can do
perfectly well. In `tests/test_learning_curve.py`, data generated under Crawford
unit theory and fitted as a Wright curve returns R² of about 0.996 with a
demonstrably wrong forecast. Standard error is in dollars and CV is a proportion.
Both are arguable. R² is reported, but last.

### Projections have to satisfy the equation

The original desktop tool priced its Rate lots on the lot midpoint, which isn't
the variable that model regresses on, and priced LC+Rate without the rate factor
at all. So it printed an equation and then printed lot costs that didn't satisfy
it. On `example_lots.csv` that overstates a back cast of the fitted lots by 36%
against a known total, while the residual columns from the same run showed the
model tracking those lots to about 1%. One run, two formulas.

Dropping the term isn't a modeling choice you could defend. It evaluates the fit
at a lot quantity of one unit while keeping the learning position of the real
lot, and because the rate exponent is negative it only ever biases upward.

The corrected behavior is the default. `LegacyRateOmission` reproduces the old
numbers for anyone who has to match a legacy workbook, and the command line has
`--legacy-rate-omission` for it, which prints a warning when you use it. Passing
the old setting name raises instead of being ignored, because a caller who asked
for legacy behavior and quietly got something else is worse off than one who
gets an error.

The guard is a test that retypes the printed equation and evaluates it against
the printed projections, for all three models. That test is why this is a
paragraph about a fix rather than a known issue.

### Wright and Crawford are different theories

Wright's cumulative average form says the *average* cost of the first x units
follows `T1·x^b`. Crawford's unit form says the cost of *unit* x does. Which one
applies is a property of the production process, not a modeling preference.
`fit_curve()` makes the caller choose and `compare_theories()` reports both,
because the same data under the two gives materially different forecasts.
