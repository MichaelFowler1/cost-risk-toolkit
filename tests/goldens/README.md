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
1 + 1e-6. And at `GOLDEN_STAGE = 3`, where the programme-level percentiles stop
being golden, moving `monte_carlo.DEFAULT_CORRELATION` from 0.25 to 0.26 still
turns 2 red.

## The stage switch

`test_goldens.py` carries `GOLDEN_STAGE`.

* **Stage 2** (today) applies `COMPARE_POLICY.exclude_from_step2` only. That
  list is provenance, loop counters and informational fields -- nothing the
  engine computes. `iteration_detail` (the `Iter` / `Delta` of the midpoint
  fixed-point loop) is excluded, *except* for `app_example_maxiter_1`,
  `app_example_maxiter_2` and `app_example_tol_1e-14`, which exist to pin the
  bookkeeping itself and keep it golden.
* **Stage 3** additionally applies `exclude_after_step3`: the WBS
  `program_level_percentiles` block, which step 3 changes on purpose when the
  lognormal handoff becomes a raw-draw handoff. Step 3 sets `GOLDEN_STAGE = 3`
  and rebaselines that block. The three roll-up knobs inside it -- `n_iter`,
  `seed` and `correlation` -- are asserted directly in `test_wbs_program`,
  outside the policy walk, so they stay golden at stage 3 as well. They are
  inputs, not results: `correlation` is `monte_carlo.DEFAULT_CORRELATION`, and
  changing that default must not be able to move numbers with nothing going
  red.

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

**Step 3** (raw-draw handoff) rebaselines the WBS `program_level_percentiles`
block and the risk sheets of the programme workbook. It must not move
`element_standalone_simulate_buy`, `simulated_run_point_total`, or any
`simulate`/risk block in the engine goldens -- those stay step-2 goldens.

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

Exactly one golden has been rebaselined since capture. Everything else in this
directory is the 2026-09-06 capture, byte for byte.

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
