# Goldens

What the estimating engine produced on **2026-09-06**, case by case, at full
float precision. `tests/test_goldens.py` recomputes every one of them with
`cost_core` alone and compares.

They exist because of what comes next. Step 2 of the refactor re-expresses the
three lot models (LC, Rate, LC+Rate) on `cost_core.fitting` and deletes the
engine's private OLS along with the private MUPE and ZMPE in
`cost_core/lotmodel/enrich.py`. The rule for that step is that **no number
moves by more than 1e-9 relative** except where a later step says so. These
files are what makes that rule checkable rather than hopeful.

## Where they came from

Captured by `capture_goldens.py`, which runs the desktop tool's engine
(`lot-cost-model` at git `72afb3c`) and this library side by side on the same
inputs and writes both. The library was at `d7cdea2` at capture time.

The engine goldens here are the **desktop tool's** values -- the reference the
library has to reproduce. The library-side twins (`engine_<case>.lib.json`) are
deliberately not copied: they would only ever agree with themselves.

Everything is stored as `<name>.json.gz`, the gzip of the exact LF bytes the
capture wrote. Floats are Python's `repr` (shortest round-trip), never rounded;
`NaN` is the string `"nan"`, `pd.NA` is `"NA"`, `NaT` is `"NaT"`, infinities are
`"inf"` / `"-inf"`, and JSON `null` means Python `None`. A leaf that turns from
NaN into None is therefore visible.

`COMPARE_POLICY.json` is stored plain, not gzipped, because it is
documentation: tolerances, exclusions and the reasoning behind both. The
repository's blanket `*.json` rule (which keeps real contractor cost data out of
git) is lifted for that one file, in `.gitignore`, and for nothing else. It
holds no data.

| File | What it pins |
|---|---|
| `engine_<case>.json.gz` (36) | one `run_lot_cost_model` + `generate_analyst_summary` + `generate_fit_chart_data` case, with `ctx`, `derived_stats`, unrounded midpoints, projections, the summary, the fit chart, and the whole `enrich` layer |
| `wbs_TEST_PROGRAM.json.gz`, `wbs_FULL_WBS.json.gz` | `roll_up` deterministic and simulated, element summaries, fiscal-year views, buy-profile sensitivity, per-element standalone risk |
| `wbs_extra_programs.json.gz` | the smaller roll-up fixtures: zero-quantity lots, gap years, twelve elements, scaled buys, phase-total profiles |
| `lots_cost_core.json.gz` | `cost_core.lots` end to end: series, fits, reports, intervals, priced plans, forecasts, simulations, assumption logs, `learning_curve.compare_methods` |
| `cli_fit_lots.json.gz` | three `ce-core fit-lots` invocations: every CSV written, its sha256 and its round-trip-parsed numbers |
| `error_paths.json.gz` | every refusal text the two suites assert a substring of |

`test_golden_census` asserts that exactly these 42 files are present, by name.
The engine test takes its 36 parameters from a directory glob, so a golden that
went missing -- deleted, or never committed -- would otherwise take its own test
away with it and the suite would go green on fewer cases than it claims.

## They are never regenerated on a whim

A disagreement means **the code moved**, not the golden. Updating a golden is a
deliberate decision with a reason written down, taken by the step that is
allowed to move that block -- never a way to make a red test go green.

The mutation check that came with these files is the evidence they bite. Of
the 48 tests: seeding the midpoint solver differently turns 40 red, nudging
`lmp_func` by 2e-9 turns 40 red, and moving the default lot correlation from
0.30 to 0.31 turns 4 red. Inside `summary.py`, multiplying SEE by 1 + 1e-6
turns 35 red, `2 * kp` -> `2.00001 * kp` in AICc turns 34 red, and the two
statistics that reach no golden at better than two decimal places -- MAPE and
mean bias -- turn the two `test_summary_precision` cases red at the same
1 + 1e-6. And even at `GOLDEN_STAGE = 3`, where the programme-level percentiles
would stop being golden, moving `monte_carlo.DEFAULT_CORRELATION` from 0.25 to
0.26 still turns 2 red.

## The stage switch

`test_goldens.py` carries `GOLDEN_STAGE`.

* **Stage 2** (today) applies `COMPARE_POLICY.exclude_from_step2` only. That
  list is provenance, loop counters and informational fields -- nothing the
  engine computes. `iteration_detail` (the `Iter` / `Delta` of the midpoint
  fixed-point loop) is excluded, *except* for `app_example_maxiter_1`,
  `app_example_maxiter_2` and `app_example_tol_1e-14`, which exist to pin the
  bookkeeping itself and keep it golden.
* **Stage 3** would additionally apply `exclude_after_step3`, and applies
  nothing, because that list is empty. It was going to hold the WBS
  `program_level_percentiles` block, which step 3 moved on purpose when the
  lognormal handoff became a raw-draw handoff. Step 3 rebaselined that block
  and left `GOLDEN_STAGE = 2` instead, so the block is still compared leaf by
  leaf and nothing in this directory is unpinned;
  `COMPARE_POLICY.exclude_after_step3` is now the record of what was
  rebaselined and by how much. The stage-3 branch stays in the walk so a later
  step has somewhere to put an exclusion it can justify. The three roll-up
  knobs inside the block -- `n_iter`, `seed` and `correlation` -- are asserted
  directly in `test_wbs_program`, outside the policy walk, so they would stay
  golden even at stage 3. They are inputs, not results: `correlation` is
  `monte_carlo.DEFAULT_CORRELATION`, and changing that default must not be
  able to move numbers with nothing going red.

## What each later step may change

**Step 2** (three lot models onto `cost_core.fitting`) landed on 2026-09-07.
What it moved is written out in
`COMPARE_POLICY.expected_to_move_beyond_rtol_in_step_2`, which carries the
measured size of each move and the reasoning, and
`golden_support.OVERRIDES` / `golden_support._EXCLUDE_STEP2` are built from
that entry so the policy and the test cannot disagree. In summary:

* the MUPE and ZMPE rows of every `compare_fitting_methods` frame, including
  the `Fit_Methods` sheet and the CLI `fit_methods.csv`, once the private
  `_fit_mupe` / `_fit_zmpe` gave way to `fitting.fit_all_methods`. Measured
  maxima 1.5e-6 relative on `b (learning)`, 3.3e-7 on `c (rate)`, 1.6e-7 on
  `T1`, 3.3e-7 on `MAPE`, 5.9e-9 on `SEE (log)` and 6.9e-10 absolute on
  `Mean % error`, taken on both supported lanes, which agree on every figure
  to four digits. Each allowance is the smallest
  power of ten that clears its own measurement by at least three times, which
  keeps every one of them well under the four significant figures the table
  prints; the policy's `_how_the_allowances_are_sized` works that rule through
  column by column, `_measured_maxima` carries the measurements and
  `_sizes_the_prototype_proposed` says which of the step-2 brief's provisional
  sizes measurement overturned and why. The OLS row,
  `log_residual_variance`, `theoretical_factor`, `smearing_factor`,
  `mupe_over_ols`, `zmpe_over_ols` and `percent_understated` stay inside 1e-9;
* `derived_stats` `Bias`, which is a near-zero quantity and needs an absolute
  allowance rather than the general 1e-12. Measured maximum 4.06e-12 across
  nine leaves, allowed 1e-10;
* `DFFITS` where the fit has fewer than two degrees of freedom. It is now
  reported as 0.0, because the leave-one-out residual scale it divides by does
  not exist there and the published values (4.4e5 to 2.1e6, against a
  conventional flag of about 1) were amplified rounding error. The change is in
  `cost_core/cer/diagnostics.py`, which is the CER package's shared
  `compute_diagnostics`, so it reaches every caller at fewer than two degrees
  of freedom and not only the lot engine; the 13 golden leaves it moves happen
  all to be lot fits;
* the two exact-fit lots fixtures, `clean_series` and `midpoint_curve_recovery`,
  whose costs are the model evaluated exactly. Their residual scale is at the
  floating-point floor, so `F`, the standard errors, `AICc`, the t-statistics
  and the residuals are all ratios of rounding errors and are carved out. What
  is **not** carved out is which model the tool selects: the t-statistic that
  decides it was crossing the 2.0 gate on both fixtures, and rather than hide
  that behind an exclusion, `summary.py` now declines to form a t out of a
  residual scale that is not there. Both fixtures select `LC` as they always
  did, and their selected model, printed equation, midpoints, unit costs, lot
  costs, intervals and risk draws are all still compared under the general
  rule. The one printed cell that changed is `t (rate coefficient)`, from
  `0.93` and `0.11` to `n/a`;
* the CLI `csv_text` and `sha256` byte-identity checks, which carry the last
  digit of every coefficient. The numbers behind them are still compared,
  through `csv_contents`, at rtol 1e-9 -- and a `csv_contents` cell that is
  text on both sides but spells a number is now compared as a number, which it
  was not before;
* the two `| MUPE |` and `| ZMPE |` rows of the markdown assumption logs, which
  print the same table. Every other line of those documents is still compared
  byte for byte.

Two of those are changes of behaviour rather than tolerance decisions, and both
are recorded in `COMPARE_POLICY.expected_to_move_beyond_rtol_in_step_2.
also_changed_in_step_2`: `DFFITS` below two degrees of freedom, and the rate
t-statistic on a fit with no residual scale. Both are narrow. `DFFITS` is
zeroed only below two degrees of freedom, which reaches 13 golden leaves, all
listed by case in the policy. The t-statistic is withheld only when the residual
scale is under 1e-10 of the fitted values: instrumenting every lot fit the suite
performs finds 17,953 of them, and that ratio is either at or below 1.1e-14 (14
fits, every one inside the two exact-fit fixtures) or at or above 2.6e-4. The
threshold sits in an empty band, four orders of magnitude above everything it
catches and six below everything it does not.

Nothing else moved. `enrich_run_*.warnings_raised` did not gain the
`fitting.fit` df<3 warning after all: `models.suppress_low_df` drops it, because
the engine has never warned at three or four lots and a refactor is not the
place to start.

**Step 3** (raw-draw handoff) landed on 2026-09-07. It rebaselined the WBS
`program_level_percentiles` block of the two programmes that have one, and
moved the risk sheets of the programme workbook with it. It moved nothing else:
`element_standalone_simulate_buy`, `simulated_run_point_total` and every
`simulate`/risk block in the engine goldens are byte-identical, which is the
point -- the element draws did not change, only what the programme roll-up does
with them. "Rebaselined goldens" below has the numbers.

## Tolerances, in one paragraph

Unrounded leaves (`ctx`, `derived_stats`, `midpoints.forecast_unrounded`,
`SE (log)`, influence columns, the lots fit numbers) are the authority and are
compared at rtol 1e-9, atol 1e-12. Leaves the engine rounds before writing
(2 dp on costs, 4 dp on midpoints and shares) get the rounding unit as an
absolute tolerance, because a value within 1e-9 of a half-unit boundary flips by
exactly one unit in the last place; a move of more than one unit is a real
regression. Sums of rounded leaves get `n_terms` units. Absolute means absolute:
the rtol term of the general rule is *not* added on top of either, or a $1e8 lot
cost would get ten 2-dp units of slack and the `LC F` column (1e26) would get
1e19 of them. The one thing that is added is four ulps of the value -- 1e-15
relative -- because above about 1e5 the float grid is coarser than the 1e-6 of a
unit the policy allows for a flip, and at 1e26 a 2-dp unit does not exist at all
(`golden_support._grid_slack`). `test_rounded_tolerance_is_absolute` pins this.
`ctx.cfg` is compared
for exact equality, so a changed engine default is seen rather than tolerated.
`golden_support.compare_with_policy` reads all of this out of
`COMPARE_POLICY.json` rather than hard-coding it. Inside a DataFrame record an
integer that has become a float is tolerated, because pandas 2.3.3 and pandas
3.0.3 disagree about when an integer column stays integer; everywhere else --
`ctx.n_keep`, `ctx.mdl_*.N` / `K` / `DF` -- that type change is reported.

## 1e-9 is tight enough to be stack-dependent

Which is why `pyproject.toml` caps numpy below 2.5 and scipy below 1.18. Both
versions land the unbiased MUPE and ZMPE refits in a slightly different place --
on `scatter_0.3` the MUPE iteration takes 13 steps instead of 11 under numpy
2.5.3, and `lots_cost_core` T1 moves 2,712.4962079 -> 2,712.4962290 there and ->
2,712.4963311 under scipy 1.18.1, 8e-9 and 4.5e-8 relative. Forty-two leaves
move under numpy and forty-one under scipy, all of them in the refit block, plus
nominally zero quantities like a mean percentage error at 1e-13 that moves to
1e-11 and so fails in relative terms while meaning nothing. scipy 1.18.1 also
changes the draws the risk model produces, which the frozen-draw hashes in
`tests/test_monte_carlo.py` catch on Windows, the one platform they are checked
on.

So a golden failure right after a numpy or scipy upgrade is the expected
behaviour of a 1e-9 pin, not a regression in this repo. Raising a cap belongs in
the rebaseline process below: measure what moved, write it into
`COMPARE_POLICY.json`, and say so here. Widening the tolerance instead would
throw away the thing these files exist to detect.

## And platform-dependent

The goldens were captured on Windows, and under Linux the same refit block
moves by the same kind of amount with no change of stack at all: fifty leaves,
every one of them in the MUPE and ZMPE results of `lots_cost_core`'s
`learning_curve_compare_methods`, up to 3.8e-8 relative on T1, and the
`scatter_0.3` MUPE loop stops at 12 steps instead of 11. It depends on the
machine as well as the platform. The same numpy on the same computer computes
different last bits under the two, OpenBLAS picks its kernels by CPU, and
`cost_core.learning_curve` fits with central differences, which magnify all of
it. Forcing OpenBLAS onto its kernels without FMA moves 40 more leaves and lets
8 of the 50 back inside 1e-9: nine in `scatter_0.05`, more of the `scatter_0.1`
MUPE fit (five spread statistics among them, by 1.3e-9), all three iteration
counts, and the sign of the `Mean bias` the two exact-fit fixtures print as
zero. One GitHub runner moved the OLS iteration count and ten of those signs on
its own, and two lanes of the v1.0.0 tag's run moved 73 leaves where the others
moved 50, two of the 73 being those signs. Nothing else in any golden moves,
the lot engine included.

Off Windows those leaves are compared under
`COMPARE_POLICY.expected_to_move_across_platforms`: every MUPE and ZMPE
statistic in the block under an allowance sized by the same three-times rule as
step 2,
the three iteration counts not at all, and those `Mean bias` cells as equal
when both print a zero, whatever its sign. On the capture machine, which is
Windows outside CI, that entry builds nothing and all of it stays at 1e-9.

A CI runner is never taken for the capture machine, even on Windows. GitHub's
Windows runners, given a different CPU each run, moved the `scatter_0.3` block
by up to 5e-9 relative and the OLS loop from 5 steps to 4 on one of them, on
the pinned stack and on a commit that touched no numerics. They get the same
allowance as Linux. For the same reason the risk model's frozen draws keep
their totals hash on every Windows machine (it held on each runner tried) but
check the element draws' hash on the capture machine only; everywhere else the
element draws are held at 1e-12 relative through their means and percentiles
(`ANALYTIC_ELEMENTS` in `tests/test_monte_carlo.py`).

The
upgrade check above therefore only works on the capture machine, because off it the
allowance absorbs the numpy 2.5.3 and scipy 1.18.1 moves in this block. Under
Linux, with the allowance in place, numpy 2.5.3 and scipy 1.18.1 each pass the
whole suite, so off Windows nothing sees either of them. Measure a cap raise on
Windows. The policy entry records the 32 Linux runs it was measured on; the
maxima the sizes rest on come from the 26 that printed the allowed fields, and
the iteration counts and `Mean bias` signs from those and the one lane that
printed only them.

## What `summary.py` prints, and how it is still pinned at 1e-9

The eight statistics `cost_core/lotmodel/summary.py` computes for itself -- SEE,
R2, Adj R2, CV, MAPE, mean bias, AICc and t -- appear in these goldens only as
the `summary` frame's strings, printed at 2 or 4 decimal places, because that
is all the desktop tool ever emitted. Compared as strings they resolve about
1e-3 relative on MAPE, bias and CV; a change below that is invisible.

That is not left as a gap. `derived_stats`, which the goldens do pin at rtol
1e-9, is a second implementation of the same arithmetic, and
`test_summary_precision` reads `summary.py`'s own numbers back at full
precision to compare against it: move an input smoothly and a printed cell
changes from one string to the next at one exact input value, and *that value*
carries every bit of the number behind the print. Bisecting to the flip point
for `summary.py` and again for `derived_stats` formatted identically, then
comparing the two at rtol 1e-9, is a 1e-9 test of a 2-decimal output. The
selection thresholds are read the same way on every case: bisecting `TGate`
recovers `|t|` and bisecting `AiccTie` recovers the AICc gap.
`golden_support.py`'s "summary.py's own arithmetic" section is the long
version.

## Rebaselined goldens

Three goldens have been rebaselined since capture, in two steps. Everything
else in this directory is the 2026-09-06 capture, byte for byte.

**2026-09-07, step 2, `engine_app_example_tol_1e-14.json.gz`, LC+Rate only.**
That case runs the engine with `Tol` tightened to 1e-14. Under the old solver
the midpoint loop never got inside it: the normal equations settled into a limit
cycle at 1.16e-14, just above the tolerance, so the loop ran out its 100
iterations and reported a fit that had not converged. The accurate solver
converges monotonically and stops at iteration 10 with a delta of 2.05e-15, and
the coefficient it reports agrees with the old one to 1.07e-12. The old `False`
was a statement about the solver, not about the fit, so it was corrected rather
than carved out. Eleven leaves, all of them the same fact written in different
places:

| field | old | new |
| --- | --- | --- |
| `iteration_detail.mdl_lcr.Iter` | 100 | 10 |
| `iteration_detail.mdl_lcr.Delta` | 1.1601830607332886e-14 | 2.0539125955565396e-15 |
| `iteration_detail.mdl_lcr.Converged` | false | true |
| `ctx.mdl_lcr.Converged` | false | true |
| `fit_status` | `LC ok; Rate ok; LC+Rate NOT CONVERGED` | `LC ok; Rate ok; LC+Rate ok` |
| `projections.records[0..5]["Fit Status"]` (6 rows) | `LC ok; Rate ok; LC+Rate NOT CONVERGED` | `LC ok; Rate ok; LC+Rate ok` |

The `Fit Status` column is `engine.py` writing the same `fit_status` string on
to every projected row, so the six rows are not six decisions. Nothing else in
the file changed, and no other golden changed at all. The file was rewritten
with `json.dumps(indent=1, allow_nan=False)` and
`gzip.compress(compresslevel=9, mtime=0)`, which reproduces every untouched
golden byte for byte.

**2026-09-07, step 3, `wbs_TEST_PROGRAM.json.gz` and `wbs_FULL_WBS.json.gz`,
`program_level_percentiles` only.** Each fitted element used to reach the risk
model as a two-parameter lognormal fitted to its own simulated buy totals,
because `RiskModel` took distributions and not draws. An element total is a sum
of correlated lognormal lot costs and is not itself lognormal, so that summary
never reproduced it: measured at the 99 percentile levels, 0 of 99 agreed and
the worst level was 0.79% to 1.23% out. The element's draws now go into the
copula unchanged, and because the draw count equals the programme's iteration
count the column comes back a permutation of them, so all 99 levels agree
exactly. That identity is what `test_program.py::TestDistributionHandoff`
asserts, with `==` rather than a tolerance.

The programme percentiles therefore fall through the working range and rise in
the far tail, because the lognormal was too fat in the shoulder and too thin at
the extreme. Both programmes, before and after:

| figure | TEST_PROGRAM | | FULL_WBS | |
| --- | --- | --- | --- | --- |
| | before -> after | change | before -> after | change |
| P50 | 197,359,861.00 -> 197,248,653.81 | -0.056% | 235,806,082.37 -> 235,679,973.43 | -0.053% |
| P80 | 199,334,505.70 -> 198,764,987.35 | -0.286% | 238,045,329.47 -> 237,399,495.65 | -0.271% |
| P90 | 200,431,772.55 -> 199,833,437.80 | -0.299% | 239,289,630.07 -> 238,611,118.46 | -0.284% |
| P99 | 202,989,641.04 -> 204,171,353.96 | +0.582% | 242,190,252.94 -> 243,530,315.40 | +0.553% |
| mean | 197,367,931.16 -> 197,350,332.99 | -0.009% | 235,815,233.94 -> 235,795,277.61 | -0.009% |
| std | 2,361,702.58 -> 2,337,709.99 | -1.016% | 2,678,170.73 -> 2,650,963.13 | -1.016% |
| cv | 0.011966 -> 0.011845 | -1.007% | 0.011357 -> 0.011243 | -1.008% |
| point estimate at | P46.3875 -> P47.9625 | +1.575 pts | P46.3875 -> P47.9625 | +1.575 pts |
| reserve to P80 | 2,168,821.67 -> 1,599,303.32 | -26.3% | 2,459,443.78 -> 1,813,609.96 | -26.3% |
| `independence_understates_sd_by` | 1.208202 -> 1.170682 | -3.11% | same | same |
| `variance_ratio_analytic` | 1.461729 -> 1.460931 | -0.055% | same | same |
| `p80_understatement` | 0.001665 -> 0.001000 | -40.0% | same | same |
| `reserve_understatement` | 0.153046 -> 0.124241 | -18.8% | same | same |

The four independence figures are identical across the two programmes because
`correlation_impact` runs on the fitted elements alone, and both programmes
carry the same three; `FULL_WBS`'s factor and amount elements are derived from
that draw rather than sampled.

The standard deviation falls 1.02% although the element variances rise about
4%. A Gaussian copula attenuates a heavier-tailed marginal more, so the
achieved Pearson correlation between the three fitted elements drops from a
mean of 0.262 to 0.216 while Spearman does not move: pair by pair, 0.279,
0.241 and 0.267 become 0.225, 0.189 and 0.235. Variance rose, covariance fell
further, and the total narrowed. That is also why the measured
`independence_understates_sd_by` falls further than the closed-form
`variance_ratio_analytic`, which is computed from the marginal variances and
the requested correlation rather than from the sample.
`test_the_sampler_agrees_with_the_algebra_on_correlation` still holds the two
together at 5%. On the standard deviation scale that assertion compares, the
disagreement went from 0.07% to 3.14%, so 1.86 of the 5 percentage points
allowed are left where 4.93 were left before; its comment says so.

162 leaves moved in `wbs_TEST_PROGRAM` and 186 in `wbs_FULL_WBS`, every one of
them inside `program_level_percentiles` and one of each being the block's own
`_note`. Those are the comparator's counts. A plain byte-for-byte diff of
`wbs_FULL_WBS` finds 188, because two further leaves moved in the last bit
only and the comparator's tolerance for that column absorbs them:
`tornado/records[3]/variance_share` (Systems Engineering) went
0.07054673721340389 -> 0.07054673721340388 and `records[4]` (Program
Management) went 0.04761904761904761 -> 0.04761904761904762, one unit in the
last place each, 2e-16 and 3e-16 relative. `COMPARE_POLICY.exclude_after_step3`
records the same thing. Nothing outside that block moved by either count,
checked by running the comparator over a fresh capture against the old file
before writing anything, and confirmed by the raw diff.
`wbs_extra_programs.json.gz` has no such block -- its programmes are captured
with `simulate=False` -- and was not regenerated. Both files were rewritten
through the same path as step 2's, `json.dumps(indent=1, allow_nan=False)` then
`gzip` at level 9 with `mtime=0`, which was proved to reproduce all 42 goldens
byte for byte before either was opened for writing.

## Two known, documented gaps

* **One golden predates one row.** `lots_cost_core.json` was captured off
  `cost_core` at `d7cdea2`, before `21cde45` gave `provenance()` its
  `Rate projection` row, so its 27-row `fit.summary` frames meet a 28-row frame
  today. `test_lots_cost_core` allows that one absence, by name, and nothing
  else does: the row is compared in all 36 engine goldens and both wbs element
  summaries, where it is not provenance at all -- it says whether the
  projections are the corrected ones or the `LegacyRateOmission` ones, and the
  legacy goldens carry different text because of it. The only Items the
  comparator ever skips are `Tool version` and `Run timestamp`, which
  `split_summary` has already stripped from both sides.
* **CSV bytes are no longer asserted.** The CLI writes its CSVs at full repr
  precision, so their sha256 carries the last bit of a double. That was always
  lane-dependent -- numpy 2.0.2 and numpy 2.4.4 disagree in the sixteenth digit
  of `exp()`, so off the capture lane the test already dropped `sha256` and
  `csv_text` -- and step 2 moves the same digits on every lane, because the
  coefficients themselves moved in the thirteenth. Both are excluded now. The
  numbers are compared through the parsed frames at rtol 1e-9, which is the
  tolerance that means something; the sha256 of the matplotlib PNG was never
  compared for the same kind of reason.
