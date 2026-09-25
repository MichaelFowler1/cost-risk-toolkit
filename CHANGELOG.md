# Changelog

All notable changes to `cost_core` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries say what moved and by how much. A number that changes is not a
housekeeping detail here, because someone may have put the old one in a budget.

## [Unreleased]

### Added

- **`cost_core.schedule`: schedule risk and joint cost and schedule
  confidence (JCL).** An activity network with most-likely durations and
  uncertainty on them, finish-to-start links with lags, time-independent and
  time-dependent costs, a project standing army and discrete risks, simulated
  together with correlated durations and costs. The result gives the joint
  confidence of any budget and date, the frontier of pairs that reach a
  chosen joint confidence, the joint confidence of the point estimate, and
  how often each activity is critical. `critical_path` gives the
  deterministic schedule and float. `ce-core jcl --spec file.json` writes the
  tables, every draw and the JCL chart; `docs/jcl_example.json` is a worked
  example. No new dependency.

### Fixed

- **Portfolio optimisation works with PuLP 4.** PuLP 4.0.0, released
  2026-09-25 for Python 3.12 and up, no longer bundles a solver and reports
  a solve's status differently, so on a fresh install the portfolio module
  in 2.1.0 fails with an AttributeError. The `optimize` extra now asks for
  `pulp[cbc]`, which installs CBC under PuLP 4, the status is read the way
  each PuLP version reports it, and a PuLP with no solver gets a message
  saying how to install one. Tested on PuLP 2.9.0, 3.3.2 and 4.0.0.

## [2.1.0] - 2026-09-24

Three new subpackages, and real programs to test on. Nothing the engine
already produced changes: the lot models, the roll-up, the CERs, the risk
simulation and the workbooks give the same numbers as 2.0.0, and the goldens
pin that. Each new part that needs another library asks for it through an
optional extra, so a plain `pip install cost-core` pulls in nothing new.

### Added

- **`cost_core.public`: real programs from DoD's public Selected Acquisition
  Reports.** It lists every SAR and MSAR released since December 2010 (about
  a thousand), fetches each through the Internet Archive with its capture and
  SHA-256 recorded, reads the unit cost section (PAUC and APUC against the
  current and original baselines) from all three report templates, and
  stacks them into a program by year panel. Every row is checked against its
  own arithmetic and the result travels with it. `ce-core sar-panel` does
  the same from the command line. Reading PDFs needs the new `public` extra
  (pdfplumber); nothing else changes, and no engine number moves.
- **`cost_core.aoa`: life-cycle cost for an analysis of alternatives.** Cost
  lines phased by fiscal year, stated in base-year, then-year and present
  value dollars (real discount rate, as OMB Circular A-94 asks, with no
  default so the current rate has to be given), simulated with correlated
  uncertainty per alternative. The comparison reports P50, P80, the
  probability each alternative is cheapest, cost per unit of effectiveness,
  and which alternatives are dominated. `historical_growth` turns the SAR
  panel into an uncertainty distribution, one observation per program.
  `ce-core aoa --spec file.json` runs one from a JSON file and writes the
  tables and an S-curve chart; `docs/aoa_example.json` is a worked example.
- **`cost_core.portfolio`: which programs to fund.** A mixed-integer program
  over candidates with several funding options each, a budget per fiscal
  year, mandatory programs, dependencies and exclusive alternatives, solved
  with CBC through PuLP (the new `optimize` extra). `marginal_value` says what
  extra money would buy in each year, `frontier` sweeps the budget level, and
  `budget_risk` gives each year's chance of breaking its budget under
  correlated cost growth. `candidates_from_aoa` turns an AoA into exclusive
  candidates. `ce-core portfolio --spec file.json` runs one;
  `docs/portfolio_example.json` is a worked example. The solver is checked
  against brute-force enumeration on random portfolios.

## [2.0.0] - 2026-09-23

A license release. The engine, and every number it produces, is the same as
1.0.1's. What changes is the terms the library is offered under from here on,
which is why this is 2.0.0 rather than 1.0.2: anyone pinning below 2 stays on
the Apache-2.0 terms 1.0.x shipped with until they choose to move. Two
presentation fixes ride along, listed under Fixed.

One thing about 1.0.1 worth knowing. The copy on PyPI was built from `main`
after the tag, so it already carries those two fixes and the `v1.0.1` tag
doesn't. No number differs between the two. 2.0.0 is built from its tag, and
the publish workflow now refuses to build anything else.

### Changed

- **The license changes for everything after 1.0.1**, from the Apache License
  2.0 to the PolyForm Noncommercial License 1.0.0. Noncommercial use stays
  free, and so does use by schools and universities, public research
  organizations, government institutions and charities, whatever their
  funding. Commercial use now needs a license from the author. Nothing is
  withdrawn: 1.0.0 and 1.0.1, as tagged here and published on PyPI, were
  released under Apache-2.0 and stay under it. A new NOTICE file carries the
  `Required Notice:` line the license obliges anyone passing the software on to
  include, and every source file now opens with a copyright line and an SPDX
  license identifier.

- The publish workflow refuses to upload anything but a release tag `vX.Y.Z`
  whose version matches `pyproject.toml`, whether a release started it or it
  was run by hand. A manual run on `main` is how PyPI's 1.0.1 came to hold code
  its tag doesn't.

### Fixed

- The buy S-curve is readable again when the simulation throws a long tail.
  The axis followed the single most extreme draw, and parameter uncertainty on
  a short series is a t with a few degrees of freedom, so one iteration in five
  thousand landing at 2.6x the median squeezed the whole curve into a vertical
  line against the left spine. It now shows the 0.5th through 99.5th
  percentile, widened to keep P50, P80, P90 and the point estimate inside it,
  and the axis label says so whenever anything is cropped. No number moves: the
  curve is still drawn in full and the percentiles are what they were.

- `ce-core fit-lots` says which units its costs are in. The engine is scale
  free and hands back whatever it is given, but its column headings read `($K)`,
  carried over from the desktop tool whose input column is `AUC ($K)`, while
  the charts format the same numbers as dollars. Feed it a file in dollars, as
  the bundled `example_lots.csv` is, and a first-unit cost of $7.5M prints as
  `T1 ($K) 7,477,686`, which is a 1,000x misread waiting to happen. The command
  now states the convention in its output and in `--csv --help`, and the README
  says it as well. The headings are unchanged, because they are the engine's
  column names: the goldens pin them, the workbook writer builds sheets from
  them and the desktop tool reads them.

## [1.0.1] - 2026-09-11

A tests-only release. The engine's code and every number it produces are the
same as 1.0.0's; the one thing that reads differently is the version in the
provenance row. What changed is the source distribution, which ships the test
suite, and 1.0.0's failed on Linux.

### Fixed

- The test suite passes off Windows. Every golden and every frozen draw in it
  was captured on Windows, and the first push of 1.0.0 failed the same six
  tests on all six Linux lanes. Nothing in `cost_core` changed and no golden
  was rewritten; the tests assumed more was portable than is.

  Under Linux the same numpy on the same machine draws about 0.5% of the
  analytic risk model's values one to nine ulps away from Windows, so the four
  frozen-draw hashes can't hold there. That model is now pinned at p50, p80
  and p90 on every platform, to 1e-12 relative, and at the hashes on Windows
  only, where they still catch a change like scipy 1.18.1's that leaves the
  percentiles alone.

  The same last-bit differences move the MUPE and ZMPE refits in the
  learning-curve block of `tests/goldens/lots_cost_core`, because
  `cost_core.learning_curve` fits with central differences: fifty leaves, up
  to 3.8e-8 relative on T1, and the `scatter_0.3` MUPE loop stops at 12 steps
  instead of 11. Which leaves move depends on the CPU as well, through the
  kernel OpenBLAS picks for it: three GitHub lanes, and OpenBLAS forced onto
  its kernels without FMA, also moved an iteration count or two and flipped
  the sign of the `Mean bias` the two exact-fit fixtures print as zero, both
  ways between `-0.00%` and `+0.00%`. Off Windows, every MUPE and ZMPE statistic
  in that block is now compared under a measured allowance,
  `COMPARE_POLICY.expected_to_move_across_platforms`, the
  three iteration counts aren't compared, and those `Mean bias` cells pass
  when both sides print a zero. On Windows all of it stays at 1e-9. Nothing
  else in any golden moves, anywhere. That makes Windows the place a numpy or
  scipy cap raise has to be measured: under Linux, numpy 2.5.3 and scipy
  1.18.1 now both pass the whole suite.

  And a masked path in an error text keeps the separator of the machine that
  wrote it, so off Windows the error-path comparison now reads
  `<error_inputs>\` and `<error_inputs>/` as the same.

## [1.0.0] - 2026-09-09

The first release. This is the point where the library stopped being the back
half of a desktop tool and became something installable on its own: the lot cost
engine, the WBS roll-up and the shared estimator all live here, and the window
lives in [lot-cost-model](https://github.com/MichaelFowler1/lot-cost-model),
which today carries its own copy of the deterministic fit and imports this
library for the risk half. The version says the interface is now something worth
keeping stable, and the goldens are what hold it to that.

### Added

- A golden-master corpus under `tests/goldens`, read by
  `tests/test_goldens.py`. It holds what the engine produced on 2026-09-06 for
  36 engine cases, eleven WBS programmes (two of them `FULL_WBS` at a
  different buy scale), the CSV and Excel front door, three command line runs
  and the error texts, at full float precision. Every case
  is rebuilt from inputs stored beside it and compared under
  `COMPARE_POLICY.json`, so the tolerances are written down instead of hidden
  in assertions. It exists because the steps that follow re-express the
  engine's arithmetic, and a claim that nothing moved beyond one part in a
  billion needs something to be measured against.

- `cost_core.program`, the WBS roll-up. Several elements, each fitted to its
  own analogy history or derived as a percentage or a flat amount, priced
  against one shared lot schedule and correlated into one programme estimate.
  It came from the desktop tool, where it was `wbs.py` and could not be used
  without importing a window.

- An `empirical` distribution type in `cost_core.monte_carlo`. Its spec is
  `{"type": "empirical", "draws": ...}` and it is the draws themselves rather
  than a family fitted to them. The quantile function is `numpy.interp` over
  the Hazen plotting positions of the sorted draws, which returns each knot
  bit for bit and clamps instead of extrapolating. Draws must be at least two,
  finite and non-negative. The last is required rather than merely expected,
  because the sampler clamps costs at zero after drawing them and one negative
  draw would silently stop the column being a permutation of the draws.

### Changed

- The distribution name on PyPI is `cost-core`. The import name does not move:
  `import cost_core` is unchanged, and setuptools normalises the two spellings
  to the same artefact names, so the hyphen is only what you type after
  `pip install`.

- The sdist carries the whole of `tests/`, plus `example_lots.csv`,
  `requirements.txt` and this file, through a new `MANIFEST.in`. setuptools'
  default sweep picks up `tests/test_*.py` and nothing those modules import, so
  the sdist used to unpack into a suite that could not run: `pytest tests/ -q`
  never reached a test, because collection was interrupted by
  `ModuleNotFoundError: No module named 'golden_support'`, and `--collect-only`
  reported 704 tests collected and one error, the 48 in `test_goldens.py` being
  the ones it lost. It now unpacks into the same suite the repository runs,
  goldens included, and `pytest tests/ -q` gives the same 752 passed from an
  unpacked release as from a checkout.

- `requires-python` drops from `>=3.11` to `>=3.9`. Nothing here needed 3.11.
  The PEP 604 unions that looked like they needed 3.10 all sit in annotations
  in files carrying `from __future__ import annotations`, so none of them is
  evaluated at runtime, and the suite passes on 3.9 unmodified. The last
  3.9-capable releases of numpy, pandas, scipy and matplotlib -- 2.0.2, 2.3.3,
  1.13.1 and 3.9.4 -- all satisfy the dependency floors, so the floors did not
  have to move either.

- CI runs 3.9, 3.10, 3.11, 3.12, 3.13 and 3.14 on every push, up from 3.11 and
  3.12. Every lane now installs the package from the version range it declares
  instead of from `requirements.txt`, because four of the five pins in that file
  will not install on 3.9 and three of them will not install on 3.10, so a
  pinned install could not have been what the lower lanes ran. One lane applies
  the pins afterwards, so the goldens are still checked against the stack they
  were recorded on. A lane on every version also uninstalls matplotlib and
  imports the engine without it, which is the only thing that keeps the claim
  below honest.

  The 3.9 lane names `ubuntu-24.04` instead of `ubuntu-latest`. setup-python
  takes its interpreters from the manifest in `actions/python-versions`, which
  has linux builds of 3.9 up to 24.04 and no further while every other version in
  the matrix has a 26.04 build. `ubuntu-latest` is 24.04 today, so the lane runs;
  the day it moves on, a lane left following it would stop finding an interpreter
  and 3.9 is too far past end of life for a new build to appear. Naming the image
  keeps the floor actually tested rather than quietly skipped.

- **numpy and scipy now carry an upper bound, `numpy<2.5` and `scipy<1.18`.**
  These are reproducibility bounds, not compatibility ones. Above them the
  library imports and runs; what it stops doing is reproducing the numbers this
  release pins, and both effects were measured on Python 3.12 one package at a
  time.

  numpy 2.5.3 moves the unbiased refits in `tests/goldens/lots_cost_core`. The
  MUPE iteration lands somewhere slightly different and sometimes takes a
  different number of steps, 13 instead of 11 on the `scatter_0.3` case, so T1
  goes 2,712.4962079 -> 2,712.4962290, about 8 parts in a billion, and the
  nominally zero mean percentage error goes -4.3e-13 -> -2.6e-11. scipy 1.18.1
  moves the same refits further, T1 to 2,712.4963311, about 4.5 parts in a
  hundred million, and it also changes the draws the risk model produces, so
  all four frozen-draw hashes in `tests/test_monte_carlo.py` fail as well.
  Forty-one golden leaves move under scipy and forty-two under numpy, against
  a tolerance of 1e-9 relative and 1e-12 absolute.

  None of that is a bug in either library, and neither version was excluded
  before. Both declare Requires-Python >=3.12, so the 3.12, 3.13 and 3.14 lanes
  were all new enough to be offered them, and with nothing stopping them a lane
  resolved both together and failed five tests on numbers: 71 golden leaves in
  one case and the four frozen-draw hashes. The lanes whose tests actually ran
  on a resolved stack were 3.12 and 3.13; 3.14 resolves the same pair and is
  protected only because it applies `requirements.txt` before it runs the suite.
  Raising a bound means rebaselining the goldens and the frozen draws under the
  new stack and recording the movement, the way `COMPARE_POLICY.json` records
  every other rebaseline here, rather than quietly widening a tolerance until
  the failure stops.

- **matplotlib is an optional extra, `cost-core[plots]`, rather than a
  dependency.** It draws the charts and does nothing else here, so a caller who
  wants numbers should not have to install a plotting stack to get them: the lot
  engine, the CERs, the risk simulation and the Excel workbooks all import and
  run without it. `cost_core.monte_carlo` no longer imports pyplot at module
  scope, `plot_distribution` imports it when called, and anything that draws --
  `cost_core.reporting.charts`, `ce-core full-run`, `ce-core fit-lots --simulate
  --out` -- fails with a message naming the extra rather than a bare
  `No module named 'matplotlib'`. `fit-lots --out` on its own draws nothing and
  needs nothing: the buy S-curve is the only chart that command writes and it
  only exists with `--simulate`, so the eight tables and the `ASSUMPTIONS.md`
  come out of a bare install unchanged. **This changes what a `pip install` pulls: an
  install that used to produce charts now needs `pip install 'cost-core[plots]'`.**

- The lot engine, the analyst summary and the Excel workbook writer are the
  desktop tool's current versions rather than the copy taken from it on 21
  August. The engine itself had not moved in between; the summary had gained
  three provenance rows, and the workbook writer three risk sheets, data
  labels, a prediction band and an S-curve. Every function was carried across
  from the tool's own source rather than retyped, so no number moved.

- `save_complete_excel_workbook` now lives in `cost_core.reporting`, with the
  other reporting code, and `cost_core.lotmodel` re-exports it. Importing
  `cost_core.reporting` no longer pulls matplotlib: its chart and pipeline
  names load on first use, so the workbook writers can be imported without it.

- `generate_analyst_summary` takes an optional `provenance` mapping, so a
  caller can stamp its own version rather than the library's.

- The three lot models are fitted through `cost_core.fitting` instead of
  through the private least squares, MUPE and ZMPE the lot engine carried.
  The coefficients move by about a part in a hundred billion, and that move is
  the old error leaving: against an exact solve in sixty digits the old normal
  equations sat 1.1e-11 out and the new path sits 1.5e-14. Model selection
  does not change on any of the 36 pinned cases and the analyst summary is
  identical character for character.

  Two things do move further than that. The unbiased refits on the
  `Fit_Methods` table shift by up to 3.3e-7 relative on their coefficients,
  because the shared estimator's iteration stops on its own least squares
  tolerance and its ZMPE starts from the MUPE answer rather than from ordinary
  least squares. One leaf goes further in relative terms and not in absolute
  ones: the ZMPE learning exponent of the flat reference series is 0.0036265630
  against 0.0036265683, which is 1.5e-6 relative on a move of 5.3e-9, and it is
  large only because the number it divides into is nearly zero. That is why the
  exponent carries an absolute allowance as well as a relative one;
  `COMPARE_POLICY._atol_on_the_exponents` has the reasoning. The ZMPE row was
  never reproducible better than about 1e-7 against itself, measured by
  perturbing its own seed, so this is inside the noise that row always had.
  And `DFFITS` now reports zero below two degrees
  of freedom, where the leave-one-out variance is zero divided by zero and the
  old formula returned whatever the rounding produced.

- The significance gate no longer forms a rate t-statistic when the fit has no
  residual spread to measure against. A series priced straight off the curve
  reproduces itself exactly, so the coefficient and its standard error are
  both at the floating-point floor and their ratio is decided by the last bit:
  two solvers agreeing to twelve digits, pricing every lot identically, could
  recommend different models. The t is reported as unavailable there instead,
  which is the test `cost_core.cer.diagnostics` already applies before
  dividing by the same quantity, and leaves the learning curve as the default.
  Across every lot fit the test suite performs the ratio is either below
  1.1e-14 or above 2.6e-4, so nothing real sits near the line.

- The programme roll-up hands each fitted element's simulated buy totals to
  the risk model directly, as an `empirical` marginal, instead of summarising
  them as a two-parameter lognormal first. An element total is a sum of
  correlated lognormal lot costs and is not itself lognormal, so no
  two-parameter fit reproduced it. Because the draw count equals the
  programme's iteration count, the copula returns a permutation of those very
  draws, so **each element's percentiles inside the programme simulation are
  now identical to its standalone percentiles**, at all 99 levels, under `==`.
  Under the old handoff none of the 99 levels agreed and the worst was 0.79%
  to 1.23% out.

  The programme percentiles move as a result. They fall through the working
  range and rise in the far tail, because the fitted lognormal was too fat in
  the shoulder and too thin at the extreme:

  | figure | TEST_PROGRAM | | FULL_WBS | |
  | --- | --- | --- | --- | --- |
  | | before -> after | change | before -> after | change |
  | P50 | 197,359,861.00 -> 197,248,653.81 | -0.056% | 235,806,082.37 -> 235,679,973.43 | -0.053% |
  | P80 | 199,334,505.70 -> 198,764,987.35 | -0.286% | 238,045,329.47 -> 237,399,495.65 | -0.271% |
  | P90 | 200,431,772.55 -> 199,833,437.80 | -0.299% | 239,289,630.07 -> 238,611,118.46 | -0.284% |
  | P99 | 202,989,641.04 -> 204,171,353.96 | +0.582% | 242,190,252.94 -> 243,530,315.40 | +0.553% |
  | standard deviation | 2,361,702.58 -> 2,337,709.99 | -1.016% | 2,678,170.73 -> 2,650,963.13 | -1.016% |
  | reserve to P80 | 2,168,821.67 -> 1,599,303.32 | -26.3% | 2,459,443.78 -> 1,813,609.96 | -26.3% |
  | point estimate sits at | P46.3875 -> P47.9625 | +1.575 points | P46.3875 -> P47.9625 | +1.575 points |

  All of these are at 8,000 iterations, seed 11, element correlation 0.25 and
  lot correlation 0.30. The reserve is the headline: the old P80 was carrying
  $569,518 on `TEST_PROGRAM` and $645,834 on `FULL_WBS` that came from the
  summary's own error rather than from anything measured.

  The standard deviation falls 1.02% although the element variances rise about
  4%. A Gaussian copula attenuates a heavier-tailed marginal more, so the
  achieved Pearson correlation between the three fitted elements drops from a
  mean of 0.262 to 0.216 while Spearman does not move: pair by pair, 0.279,
  0.241 and 0.267 become 0.225, 0.189 and 0.235. Variance rose, covariance
  fell further, and the total narrowed.

  Nothing else moved. Each element's own simulated buy, the deterministic
  roll-up and every engine number are byte-identical, and so is every
  non-empirical marginal in the risk model, which is now pinned by a frozen
  seeded simulation in `tests/test_monte_carlo.py`.

- `tests/goldens/wbs_TEST_PROGRAM.json.gz` and
  `tests/goldens/wbs_FULL_WBS.json.gz` are rebaselined in their
  `program_level_percentiles` block, and only there, to the numbers above.
  They are still compared leaf by leaf rather than excluded, so the new values
  are pinned as tightly as the old ones were.
  `COMPARE_POLICY.exclude_after_step3` records what was rebaselined, when, why
  and by how much, and `tests/goldens/README.md` carries the same account.

- The command line moved from a top-level `cli` package to `cost_core.cli`.
  A wheel that installs a package called `cli` into site-packages shares that
  name with every other distribution that does, and installing or removing
  either can break the other. The `ce-core` command is unchanged, and
  `python -m cost_core.cli` replaces `python -m cli.ce_core_cli`.

### Removed

- `cost_core.gui` and the `ce-core gui` subcommand. It was a copy of the
  desktop tool's window taken on 21 August, already older than the tool by the
  time it landed, and two windows over one engine is one too many. The window
  to keep is [lot-cost-model](https://github.com/MichaelFowler1/lot-cost-model).
  **Anyone importing `cost_core.gui` or
  running `ce-core gui` has to move to that repository.**

- `cost_core.lotmodel.mathx.ols_fit` and `.solve_model`, the lot engine's
  private least squares and its midpoint iteration, replaced by
  `cost_core.lotmodel.models` on top of `cost_core.fitting`. `mathx` keeps the
  lot midpoint, the unit tracking and the two column helpers, which are
  geometry and parsing. The private MUPE and ZMPE in
  `cost_core.lotmodel.enrich` are gone the same way.

- `cost_core.program.SPEC_TOLERANCE` and `cost_core.program._lognormal_spec`,
  along with the test that bounded the fit against the element it summarised.
  There is no longer an approximation there to bound.

[Unreleased]: https://github.com/MichaelFowler1/cost-risk-toolkit/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/MichaelFowler1/cost-risk-toolkit/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/MichaelFowler1/cost-risk-toolkit/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/MichaelFowler1/cost-risk-toolkit/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/MichaelFowler1/cost-risk-toolkit/releases/tag/v1.0.0
