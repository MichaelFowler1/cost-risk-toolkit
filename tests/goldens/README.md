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

**Step 2** (three lot models onto `cost_core.fitting`) may move only what
`COMPARE_POLICY.expected_to_move_beyond_rtol_in_step_2` already names:

* the MUPE and ZMPE rows of `compare_fitting_methods` (about 1e-8 on the
  coefficients, 2e-9 to 6e-9 on the error statistics). The OLS row,
  `log_residual_variance`, `theoretical_factor`, `smearing_factor`,
  `mupe_over_ols`, `zmpe_over_ols` and `percent_understated` stay inside 1e-9;
* the `Fit_Methods` sheet in `enrich_run_*.sheets`, for the same reason;
* `enrich_run_*.warnings_raised`, which may gain the `fitting.fit` df<3 warning
  text on the four- and five-lot cases.

Nothing else. `golden_support.OVERRIDES` is the hook for the first two: it maps
a path glob to the rtol that **numeric** leaf may move by, and it is empty today
on purpose. Widening it is how step 2 records the two numeric moves, in one
place, reviewably.

It is not the hook for the third. `warnings_raised` is a list of strings, and a
list that gains an entry is a structural mismatch -- no rtol can absorb it, and
setting an override on that path does nothing. Step 2 has to record that one in
the comparator itself: add the path to `golden_support._EXCLUDE_STEP2` with the
reason, or allow the one extra string explicitly. Whichever it is, it is a code
change with a diff to review, which is the right weight for it.

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
* **CSV bytes are only asserted on the capture lane.** The CLI writes its CSVs
  at full repr precision, so their sha256 carries the last bit of a double, and
  numpy 2.0.2 and numpy 2.4.4 disagree in the sixteenth digit of `exp()`. Off
  the captured numpy/pandas the test drops `sha256` and `csv_text` (saying so)
  and compares the same numbers through the parsed frames at rtol 1e-9.
