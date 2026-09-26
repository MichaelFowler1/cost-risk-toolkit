# Working on the code

*Part of the [cost-core README](https://github.com/MichaelFowler1/cost-risk-toolkit#readme).*

## Reproducing the numbers: the numpy and scipy bounds

numpy and scipy also carry an upper bound, `numpy<2.5` and `scipy<1.18`. That is
about reproducing numbers, not about running: above those versions the library
imports and works, but it stops reproducing the figures this release pins. Both
land the unbiased MUPE and ZMPE refits in a slightly different place, which moves
golden leaves by up to 8e-9 and 4.5e-8 relative against a 1e-9 tolerance, and
scipy 1.18 also changes the simulated draws, which the frozen-draw tests hash.
Both are seen on Windows, where the goldens were captured. Elsewhere those refit
leaves carry a measured allowance for the machine's own last bits and the hashes
are not checked, so a bound has to be measured on Windows. Raising a bound means
rebaselining the goldens and the frozen draws against the new stack and writing
down what moved, the same process every other rebaseline here went through.
`CHANGELOG.md` carries the measurements.

## Project structure

```
cost_core/
  fitting.py          shared OLS / MUPE / ZMPE estimator and intervals
  lots.py             your own lot data: units and cost per lot
  cli.py              the ce-core command line interface
  lotmodel/           the lot cost engine: LC / Rate / LC+Rate on lot midpoints, summary, enrichment
  program/            WBS roll-up of several elements into one programme
  learning_curve.py   Wright and Crawford theories, rate breaks
  monte_carlo.py      correlated risk simulation
  data_io.py          CSV and SQLite loading
  synth/              synthetic CSDR/SRDR generator
  ingest/             crosswalk, inflation, normalization pipeline
  cer/                parametric CERs and diagnostics
  reporting/          charts, assumptions log, the Excel workbooks, end to end run
  public/             public SARs: fetch with provenance, catalogue, unit cost parser, panel
  aoa/                life-cycle cost of alternatives, spec files, growth from SAR history
  portfolio/          which programs to fund: integer program, marginal value, budget risk
  schedule/           activity networks, schedule risk and joint cost and schedule confidence
tests/                property tests, see below
```

## Tests

```bash
pip install -e ".[plots]" pytest
pytest tests/ -q
```

`requirements.txt` also exists, but it holds the exact versions this repo
develops against, and four of the five will not install on 3.9, three of them
not on 3.10. It is the pinned development set, not the way to set up a checkout
on an older interpreter. Install the package and let the version range in
`pyproject.toml` resolve.

759 tests, run on Python 3.9, 3.10, 3.11, 3.12, 3.13 and 3.14 on every push,
which is the whole supported range; four of them, the frozen-draw hashes, run
on Windows only. They assert mathematics against closed form
answers rather than against recorded output, with one deliberate exception:
tests/goldens pins what the lot engine produced on 6 September 2026, so a
refactor that moves a number has to say so. The strongest ones:

**Our OLS *is* the textbook OLS.** The generic estimator reproduces
`scipy.stats.linregress` and the normal equations to machine precision, and the
delta method prediction interval reduces algebraically to
`s·√(1 + 1/n + (x₀-x̄)²/Sxx)`.

**MUPE and ZMPE drive the mean percentage error to exactly zero.** That's what
their names mean, and it's asserted to 1e-9. ZMPE's sum of squared percentage
errors is also proven to be no larger than MUPE's, which is a theorem, not a
tuning outcome.

**Cook's distance is checked against an actual leave one out refit.** The closed
form is exact, so the test drops each program, refits, and confirms the formula
reproduces the movement in the fitted surface.

**Variance inflation is exactly `1 + ρ(k-1)`.** Asserted across element counts
and correlations, and confirmed against simulation.

**Tornado variance shares sum to exactly one**, because the covariance
decomposition `Var(T) = Σ Cov(Xᵢ, T)` is an identity when T is the sum.

**A messy program normalizes back to the generating truth to the cent.** Name
drift, mixed then year and base year dollars, resubmitted periods and a mid
program quantity change are all reversible by construction, so the pipeline that
reverses them has no excuse for landing anywhere else.

**Learning curve identities are definitional.** Doubling quantity multiplies the
right quantity by the slope under each theory. Wright's unit costs telescope back
to its cumulative total. Crawford's lot cost is the exact sum of its units.

**Simulations are seed deterministic.** A P80 that moves between runs isn't a
number you can put in front of anyone.

**Limits get tested, not just capabilities.** There's a parametrized test
asserting that both escalation checks stay silent at 2%, 4% and 6% a year under a
midpoint fit, while the fitted slope drifts several points off the truth. A
second test confirms the level check does catch 15%. Documenting where a
diagnostic stops working matters more than showing where it works.

**The projections satisfy the equation the tool prints.** Retyped by hand for
all three models and evaluated against the projected costs, to the cent the
column is rounded to. Flipping the legacy switch back on fails five of these,
which is how I know they'd have caught the original defect.

**Bad input is refused, not absorbed.** Zero degrees of freedom, a missing base
year, two lots, fractional units, unmatched WBS names, non positive costs in a
log fit, a correlation matrix that isn't symmetric, a rate break beyond the data,
an unknown interval kind, an index asked for a year it doesn't cover. Each one
raises instead of producing a plausible looking number.
