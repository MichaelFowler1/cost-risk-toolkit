# Changelog

All notable changes to `cost_core` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Nothing has been released yet, so everything sits under Unreleased.

Entries say what moved and by how much. A number that changes is not a
housekeeping detail here, because someone may have put the old one in a budget.

## [Unreleased]

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
  least squares. The ZMPE row was never reproducible better than about 1e-7
  against itself, measured by perturbing its own seed, so this is inside the
  noise that row always had. And `DFFITS` now reports zero below two degrees
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
  | P50 | 197,359,861.00 -> 197,248,653.81 | -0.056% | 235,806,082.37 -> 235,679,973.43 | -0.054% |
  | P80 | 199,334,505.70 -> 198,764,987.35 | -0.286% | 238,045,329.47 -> 237,399,495.65 | -0.271% |
  | P90 | 200,431,772.55 -> 199,833,437.80 | -0.299% | 239,289,630.07 -> 238,611,118.46 | -0.284% |
  | P99 | 202,989,641.04 -> 204,171,353.96 | +0.582% | 242,190,252.94 -> 243,530,315.40 | +0.553% |
  | standard deviation | 2,361,702.58 -> 2,337,709.99 | -1.016% | 2,678,170.73 -> 2,650,963.13 | -1.016% |
  | reserve to P80 | 2,168,821.67 -> 1,599,303.32 | -26.3% | 2,459,443.78 -> 1,813,609.96 | -26.3% |
  | point estimate sits at | P46.39 -> P47.96 | +1.58 points | P46.39 -> P47.96 | +1.58 points |

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

### Removed

- `cost_core.gui` and the `ce-core gui` subcommand. It was a copy of the
  desktop tool's window taken on 21 August, already older than the tool by the
  time it landed, and two windows over one engine is one too many. The window
  is [lot-cost-model](https://github.com/MichaelFowler1/lot-cost-model); this
  library is the engine it drives. **Anyone importing `cost_core.gui` or
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
