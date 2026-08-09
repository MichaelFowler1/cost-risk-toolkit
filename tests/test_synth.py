"""The synthetic CSDR/SRDR generator.

Two things have to be true for this generator to be worth anything.

First, **the same seed reproduces the same program exactly**. A synthetic
dataset that drifts between runs cannot support a repeatable demonstration,
and every downstream test would be chasing a moving target.

Second, **the reports reconcile to the generating truth when the pathologies
are switched off**. That is what makes the truth usable as a closed-form
answer: if a clean program's six reports all sum back to the same numbers, then
any disagreement seen later was introduced by a pathology on purpose, and the
ingest pipeline can be held to recovering the truth exactly rather than
approximately.

The pathology tests then check the other direction -- that the mess actually
appears when asked for, since a generator that quietly produces clean data
would make the whole ingest layer look better than it is.
"""
import numpy as np
import pandas as pd
import pytest

from cost_core.synth import (BASE_WRAP_RATES, DEFAULT_NAME_VARIANTS,
                             DEFAULT_WBS, FUNCTIONAL_MIX, REPORT_NAMES,
                             TRUE_AIRFRAME_CER, TRUE_SOFTWARE_CER,
                             InflationAssumption, PathologyConfig, ProgramSpec,
                             crawford_lot_cost, generate_portfolio,
                             generate_program)

CLEAN = PathologyConfig.clean()


@pytest.fixture(scope="module")
def clean_program():
    return generate_program(seed=3, pathologies=CLEAN)


@pytest.fixture(scope="module")
def messy_program():
    return generate_program(seed=3)


# ============================================================== determinism
def test_the_same_seed_reproduces_the_program_exactly():
    a = generate_program(seed=11)
    b = generate_program(seed=11)
    pd.testing.assert_frame_equal(a.truth.cells, b.truth.cells)
    for name in REPORT_NAMES:
        pd.testing.assert_frame_equal(a[name], b[name])
    assert a.truth.lot_quantities == b.truth.lot_quantities
    assert a.truth.outlier_lots == b.truth.outlier_lots


def test_different_seeds_give_different_programs():
    a = generate_program(seed=11)
    b = generate_program(seed=12)
    assert a.truth.total_dollars_by != b.truth.total_dollars_by


def test_every_report_shape_is_produced_and_non_empty(messy_program):
    assert set(messy_program.reports) == set(REPORT_NAMES)
    for name in REPORT_NAMES:
        assert len(messy_program[name]) > 0, name


def test_an_unknown_report_name_is_refused(messy_program):
    with pytest.raises(KeyError, match="No report"):
        messy_program["dd1922"]


# ==================================================== the underlying cost model
@pytest.mark.parametrize("slope", [0.75, 0.85, 0.95])
def test_crawford_lot_cost_is_the_exact_sum_of_its_unit_costs(slope):
    """Lot cost is summed unit by unit, not approximated with a midpoint, so
    it must equal the sum of the individual Crawford unit costs exactly."""
    t1, first, last = 1000.0, 7, 19
    units = np.arange(first, last + 1, dtype=float)
    expected = float(np.sum(t1 * units ** np.log2(slope)))
    assert crawford_lot_cost(t1, slope, first, last) == pytest.approx(
        expected, rel=1e-14
    )


def test_a_single_unit_lot_costs_exactly_its_unit_cost():
    assert crawford_lot_cost(1000.0, 0.85, 1, 1) == pytest.approx(1000.0, rel=1e-14)
    # The 8th unit of an 85% curve is 0.85^3 of the first.
    assert crawford_lot_cost(1000.0, 0.85, 8, 8) == pytest.approx(
        1000.0 * 0.85**3, rel=1e-12
    )


def test_clean_lot_costs_lie_exactly_on_the_generating_curve(clean_program):
    """With no scatter and no rate breaks, the recurring cost of each lot must
    be the curve's own answer -- this is what lets a learning-curve fit on
    clean 1921-2 data recover the slope to machine precision."""
    truth = clean_program.truth
    lots = truth.recurring_lot_costs()
    first = 1
    for _, row in lots.iterrows():
        qty = int(row["quantity"])
        expected = crawford_lot_cost(
            truth.t1_cost, truth.learning_slope, first, first + qty - 1
        )
        assert row["lot_cost_by"] == pytest.approx(expected, rel=1e-9)
        first += qty


def test_nonrecurring_cost_is_concentrated_in_the_early_lots(clean_program):
    """Front-loaded nonrecurring cost is the reason a naive fit that ignores
    the recurring/nonrecurring split reads a steeper slope than the truth."""
    cells = clean_program.truth.cells
    nonrec = cells[~cells["recurring_flag"]].groupby("lot")["dollars_by"].sum()
    assert nonrec.is_monotonic_decreasing
    # The first lot should carry a large share of the whole nonrecurring pool.
    assert nonrec.iloc[0] / nonrec.sum() > 0.30


# ================================================ reconciliation on clean data
def test_functional_categories_sum_to_the_wbs_element_total(clean_program):
    cells = clean_program.truth.cells
    per_element = cells.groupby(["lot", "wbs_name", "recurring_flag"])[
        "dollars_by"
    ].sum()
    assert per_element.sum() == pytest.approx(
        clean_program.truth.total_dollars_by, rel=1e-12
    )


def test_hours_times_the_wrap_rate_equals_dollars_exactly(clean_program):
    """The reconciliation an analyst does by hand. Material is dollars-only,
    so it is excluded -- dividing all dollars by all hours would give a
    blended rate that matches nothing."""
    cells = clean_program.truth.cells
    hour_bearing = cells[cells["functional_category"] != "material"]
    for category, rate in BASE_WRAP_RATES.items():
        subset = hour_bearing[hour_bearing["functional_category"] == category]
        assert len(subset) > 0
        assert (subset["hours"] * rate).to_numpy() == pytest.approx(
            subset["dollars_by"].to_numpy(), rel=1e-12
        )


def test_material_carries_dollars_but_no_hours(clean_program):
    material = clean_program.truth.cells.query("functional_category == 'material'")
    assert (material["hours"] == 0.0).all()
    assert (material["dollars_by"] > 0.0).all()
    # And in the reported 1921-1 it is a null, not a zero, so nobody divides by it.
    reported = clean_program["dd1921_1"].query("functional_category == 'material'")
    assert reported["hours"].isna().all()


def test_dd1921_period_costs_sum_to_the_program_total(clean_program):
    """Every dollar in the truth has to appear in the summary report."""
    reported = clean_program["dd1921"]["cost_incurred_period"].sum()
    assert reported == pytest.approx(clean_program.truth.total_dollars_by, rel=1e-10)


def test_dd1921_to_date_is_the_running_sum_of_the_period_column(clean_program):
    """The cumulative column has to be consistent with the incremental one, or
    the reconciliation gate in the ingest layer is checking nothing."""
    df = clean_program["dd1921"].sort_values(["wbs_element_name", "recurring_flag", "lot"])
    running = df.groupby(["wbs_element_name", "recurring_flag"])[
        "cost_incurred_period"
    ].cumsum()
    assert running.to_numpy() == pytest.approx(df["cost_to_date"].to_numpy(), rel=1e-9)


def test_dd1921_1_dollars_sum_to_the_dd1921_totals(clean_program):
    """Two report shapes, same underlying dollars."""
    summary = clean_program["dd1921"]["cost_incurred_period"].sum()
    functional = clean_program["dd1921_1"]["dollars"].sum()
    assert functional == pytest.approx(summary, rel=1e-10)


def test_flexfile_dollar_elements_sum_to_the_dd1921_1_dollars(clean_program):
    """The FlexFile splits a burdened dollar into direct labour plus overhead;
    the split must be exhaustive, not lossy."""
    flex = clean_program["flexfile"]
    flex_dollars = flex.query("unit == 'dollars'")["value"].sum()
    assert flex_dollars == pytest.approx(
        clean_program["dd1921_1"]["dollars"].sum(), rel=1e-10
    )


def test_flexfile_hours_match_the_dd1921_1_hours(clean_program):
    flex_hours = clean_program["flexfile"].query("unit == 'hours'")["value"].sum()
    assert flex_hours == pytest.approx(
        clean_program["dd1921_1"]["hours"].sum(), rel=1e-10
    )


def test_dd1921_2_recurring_cost_matches_the_recurring_truth(clean_program):
    reported = clean_program["dd1921_2"]["recurring_lot_cost"].sum()
    cells = clean_program.truth.cells
    expected = cells[cells["recurring_flag"]]["dollars_by"].sum()
    assert reported == pytest.approx(expected, rel=1e-10)


def test_dd1921_2_unit_cost_is_lot_cost_over_quantity(clean_program):
    df = clean_program["dd1921_2"]
    assert (df["recurring_lot_cost"] / df["lot_quantity"]).to_numpy() == pytest.approx(
        df["unit_cost"].to_numpy(), rel=1e-12
    )


def test_quantity_report_cumulative_quantity_is_the_running_total(clean_program):
    """Checked within one submission: the cumulative column resets per report."""
    df = clean_program["quantity_report"]
    latest = df[df["report_date"] == df["report_date"].max()].sort_values("lot")
    assert latest["cumulative_quantity"].to_numpy() == pytest.approx(
        np.cumsum(latest["lot_quantity"].to_numpy())
    )


# ============================================================ the pathologies
def test_clean_config_produces_no_mess_at_all(clean_program):
    """The control case. If any of these fired, the exact-recovery assertions
    elsewhere would be testing the wrong thing."""
    assert clean_program.truth.outlier_lots == {}
    assert clean_program.truth.lot_quantities == clean_program.truth.planned_quantities
    for name in ("dd1921", "dd1921_1", "dd1921_2", "flexfile"):
        assert (clean_program[name]["basis"] == "BY").all(), name
    # No aliases: every reported name is canonical.
    canonical = {w.name for w in DEFAULT_WBS}
    assert set(clean_program["dd1921"]["wbs_element_name"]) <= canonical
    # No resubmissions: one report date per period.
    per_period = clean_program["dd1921"].groupby("period_fy")["report_date"].nunique()
    assert (per_period == 1).all()


def test_names_drift_across_periods_when_asked(messy_program):
    """The reason a crosswalk is needed at all."""
    canonical = {w.name for w in DEFAULT_WBS}
    reported = set(messy_program["dd1921"]["wbs_element_name"])
    aliases = reported - canonical
    assert aliases, "expected at least one drifted element name"
    known = {a for variants in DEFAULT_NAME_VARIANTS.values() for a in variants}
    assert aliases <= known, "every alias must be resolvable through the crosswalk"


def test_both_dollar_bases_appear_in_a_messy_program(messy_program):
    bases = set(messy_program["dd1921"]["basis"])
    assert bases == {"BY", "TY"}


def test_then_year_rows_deflate_back_to_base_year_exactly():
    """The pathology has to be reversible, or the ingest layer could not be
    held to recovering the truth."""
    ty_only = PathologyConfig.clean().__class__(
        **{**PathologyConfig.clean().__dict__, "then_year_prob": 1.0}
    )
    program = generate_program(seed=5, pathologies=ty_only)
    index = program.truth.inflation_index
    df = program["dd1921_2"]
    assert (df["basis"] == "TY").all()
    deflated = df["recurring_lot_cost"] / df["period_fy"].map(index)
    truth = program.truth.recurring_lot_costs().set_index("lot")["lot_cost_by"]
    assert deflated.to_numpy() == pytest.approx(
        truth.loc[df["lot"]].to_numpy(), rel=1e-12
    )


def test_resubmitted_periods_appear_twice_with_the_later_one_correct():
    """Dedup by latest report date has to land on the truth, so the *later*
    submission is the accurate one and the earlier is the mistake."""
    resub = PathologyConfig(
        name_drift_prob=0.0,
        then_year_prob=0.0,
        resubmission_prob=1.0,
        missing_period_prob=0.0,
        outlier_lot_prob=0.0,
        quantity_change=False,
        noise_cv=0.0,
        eac_optimism=0.0,
    )
    program = generate_program(seed=5, pathologies=resub)
    df = program["dd1921_2"]
    assert df.groupby("lot")["report_date"].nunique().eq(2).all()

    latest = df.sort_values("report_date").groupby("lot").tail(1).sort_values("lot")
    truth = program.truth.recurring_lot_costs().set_index("lot")["lot_cost_by"]
    assert latest["recurring_lot_cost"].to_numpy() == pytest.approx(
        truth.loc[latest["lot"]].to_numpy(), rel=1e-12
    )
    # And the superseded submission genuinely disagrees.
    earliest = df.sort_values("report_date").groupby("lot").head(1)
    assert not np.allclose(
        earliest.sort_values("lot")["recurring_lot_cost"].to_numpy(),
        truth.loc[earliest.sort_values("lot")["lot"]].to_numpy(),
    )


def test_missing_periods_are_actually_absent():
    """Unlike the other pathologies this one is not reversible. The pipeline
    is expected to surface the gap, so the generator must really drop it."""
    gappy = PathologyConfig(
        name_drift_prob=0.0,
        then_year_prob=0.0,
        resubmission_prob=0.0,
        missing_period_prob=0.5,
        outlier_lot_prob=0.0,
        quantity_change=False,
        noise_cv=0.0,
        eac_optimism=0.0,
    )
    program = generate_program(seed=2, pathologies=gappy)
    reported = set(program["dd1921_2"]["lot"])
    all_lots = set(range(1, program.spec.n_lots + 1))
    assert reported, "the whole program should not vanish"
    assert reported < all_lots, "some periods should be missing"
    # The gap is in every report, not patched from one shape to another.
    assert set(program["dd1921"]["lot"]) == reported


def test_a_program_with_every_period_missing_is_refused():
    """Rather than emitting empty reports that look like a clean program."""
    allgone = PathologyConfig(missing_period_prob=1.0)
    with pytest.raises(ValueError, match="Every reporting period"):
        generate_program(seed=1, pathologies=allgone)


def test_outlier_lots_sit_above_the_underlying_curve():
    shocked = PathologyConfig(
        name_drift_prob=0.0,
        then_year_prob=0.0,
        resubmission_prob=0.0,
        missing_period_prob=0.0,
        outlier_lot_prob=1.0,
        outlier_magnitude=1.30,
        quantity_change=False,
        noise_cv=0.0,
        eac_optimism=0.0,
    )
    program = generate_program(seed=4, pathologies=shocked)
    assert len(program.truth.outlier_lots) == program.spec.n_lots
    actual = program.truth.recurring_lot_costs().set_index("lot")["lot_cost_by"]
    curve = program.truth.lot_recurring_by
    assert (actual / curve > 1.0).all()


def test_a_quantity_change_shows_up_against_the_original_plan():
    program = generate_program(seed=7)
    assert program.truth.lot_quantities != program.truth.planned_quantities
    qty = program["quantity_report"]
    assert qty["rebaselined"].any()


def test_scatter_is_present_when_asked_and_absent_when_not():
    noisy = generate_program(seed=8, pathologies=PathologyConfig(
        name_drift_prob=0.0, then_year_prob=0.0, resubmission_prob=0.0,
        missing_period_prob=0.0, outlier_lot_prob=0.0, quantity_change=False,
        noise_cv=0.10, eac_optimism=0.0,
    ))
    ratio = (
        noisy.truth.recurring_lot_costs().set_index("lot")["lot_cost_by"]
        / noisy.truth.lot_recurring_by
    )
    assert ratio.std() > 0.0
    assert not np.allclose(ratio.to_numpy(), 1.0)


def test_early_at_completion_estimates_are_optimistic(messy_program):
    """A real program's EAC creeps up. The first submission should sit below
    the last for the same element."""
    df = messy_program["dd1921"].query("basis == 'BY' and recurring_flag")
    if df.empty:  # pragma: no cover - depends on the seed's basis draws
        pytest.skip("no base-year recurring rows in this seed")
    by_period = df.groupby("period_fy")["cost_at_completion"].sum()
    assert by_period.iloc[0] < by_period.iloc[-1]


# ==================================================================== SRDR
def test_srdr_activity_hours_sum_to_total_effort(messy_program):
    df = messy_program["srdr"]
    activity_cols = [c for c in df.columns if c.startswith("hours_")]
    assert len(activity_cols) == 6
    assert df[activity_cols].sum(axis=1).to_numpy() == pytest.approx(
        df["total_effort_hours"].to_numpy(), rel=1e-12
    )


def test_srdr_equivalent_sloc_is_the_adaptation_weighted_size():
    program = generate_program(seed=6, pathologies=CLEAN)
    sw = program.spec.software
    expected = (
        sw.sloc_new + 0.35 * sw.sloc_modified + 0.03 * sw.sloc_reused
        + 0.10 * sw.sloc_autogen
    )
    assert sw.equivalent_sloc == pytest.approx(expected, rel=1e-12)
    # Reused code is nearly free, so equivalent size is well below raw size.
    raw = sw.sloc_new + sw.sloc_modified + sw.sloc_reused + sw.sloc_autogen
    assert sw.equivalent_sloc < raw


def test_srdr_effort_follows_the_true_software_cer():
    program = generate_program(seed=6, pathologies=CLEAN)
    a, b = TRUE_SOFTWARE_CER
    sw = program.spec.software
    assert program["srdr"]["total_effort_hours"].sum() == pytest.approx(
        a * sw.equivalent_ksloc**b, rel=1e-9
    )


# =============================================================== portfolios
def test_a_portfolio_is_reproducible_and_varied():
    a = generate_portfolio(n_programs=5, seed=2, pathologies=CLEAN)
    b = generate_portfolio(n_programs=5, seed=2, pathologies=CLEAN)
    assert len(a) == 5
    pd.testing.assert_frame_equal(a.cer_table(), b.cer_table())
    assert a.cer_table()["empty_weight_lb"].nunique() == 5


def test_skipping_report_generation_leaves_the_truth_identical():
    """``with_reports=False`` exists so a large CER portfolio does not pay for
    thousands of submission rows per program. It must be a pure saving: the
    reports are derived from the truth, so omitting them cannot change it."""
    full = generate_portfolio(n_programs=4, seed=7, pathologies=CLEAN)
    lean = generate_portfolio(
        n_programs=4, seed=7, pathologies=CLEAN, with_reports=False
    )
    pd.testing.assert_frame_equal(full.cer_table(), lean.cer_table())
    for a, b in zip(full, lean):
        pd.testing.assert_frame_equal(a.truth.cells, b.truth.cells)
    assert not lean.programs[0].reports
    with pytest.raises(KeyError, match="without reports"):
        lean.report("dd1921")


def test_portfolio_first_unit_costs_follow_the_true_airframe_cer():
    """Before scatter, T1 must be exactly the CER's answer -- this is the
    closed-form target the CER module has to recover."""
    portfolio = generate_portfolio(
        n_programs=8, seed=1, pathologies=CLEAN, with_reports=False
    )
    a, b = TRUE_AIRFRAME_CER
    drivers = portfolio.cer_table()
    expected = a * (drivers["empty_weight_lb"] / 1000.0) ** b
    assert drivers["t1_cost_noiseless"].to_numpy() == pytest.approx(
        expected.to_numpy(), rel=1e-12
    )
    assert portfolio.truth.airframe_a == a
    assert portfolio.truth.airframe_b == b


def test_portfolio_scatter_is_present_around_the_true_cer():
    portfolio = generate_portfolio(
        n_programs=10, seed=1, pathologies=CLEAN, with_reports=False
    )
    drivers = portfolio.cer_table()
    ratio = drivers["t1_cost_observed"] / drivers["t1_cost_noiseless"]
    assert ratio.std() > 0.05
    assert not np.allclose(ratio.to_numpy(), 1.0)


def test_a_portfolio_of_one_is_refused():
    """One point cannot define a relationship, and pretending otherwise is
    how a CER with no degrees of freedom gets briefed."""
    with pytest.raises(ValueError, match="at least two programs"):
        generate_portfolio(n_programs=1, seed=0)


def test_concatenating_a_report_across_the_portfolio_keeps_every_program():
    portfolio = generate_portfolio(n_programs=4, seed=3, pathologies=CLEAN)
    combined = portfolio.report("dd1921_2")
    assert combined["program"].nunique() == 4


# ============================================================ specification
def test_the_default_wbs_shares_sum_to_one():
    """A silent edit here would change every program total in the library."""
    assert sum(w.cost_share for w in DEFAULT_WBS) == pytest.approx(1.0, abs=1e-12)
    assert sum(w.nonrecurring_share for w in DEFAULT_WBS) == pytest.approx(
        1.0, abs=1e-12
    )


def test_the_functional_mixes_sum_to_one():
    for label, mix in FUNCTIONAL_MIX.items():
        assert sum(mix.values()) == pytest.approx(1.0, abs=1e-12), label


def test_the_inflation_index_is_one_in_the_base_year():
    index = InflationAssumption(base_year=2020, annual_rate=0.03).index()
    assert index[2020] == pytest.approx(1.0, rel=1e-15)
    assert index[2021] == pytest.approx(1.03, rel=1e-12)
    assert index[2019] == pytest.approx(1.0 / 1.03, rel=1e-12)
    # Monotone: a raw index that fell would deflate the wrong way.
    years = sorted(index)
    assert all(index[a] < index[b] for a, b in zip(years, years[1:]))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"learning_slope": 1.4}, "learning_slope"),
        ({"learning_slope": 0.0}, "learning_slope"),
        ({"t1_cost": -1.0}, "t1_cost"),
        ({"lot_quantities": ()}, "at least one production lot"),
        ({"lot_quantities": (4, 0, 6)}, "positive"),
    ],
)
def test_an_inconsistent_spec_is_refused(kwargs, message):
    """Bad configuration raises rather than generating a program whose numbers
    would look perfectly reasonable on a chart."""
    with pytest.raises(ValueError, match=message):
        generate_program(seed=1, spec=ProgramSpec(**kwargs))


def test_wbs_shares_that_do_not_sum_to_one_are_refused():
    from cost_core.synth import WBSElement

    broken = (
        WBSElement("1.1", "Airframe", None, 0.50, 1.0),
        WBSElement("1.2", "Propulsion", None, 0.30, 0.0),
    )
    with pytest.raises(ValueError, match="cost shares sum to"):
        generate_program(seed=1, spec=ProgramSpec(wbs=broken))


def test_a_wbs_element_naming_a_missing_parent_is_refused():
    from cost_core.synth import WBSElement

    orphan = (WBSElement("1.1", "Airframe", "9.9", 1.0, 1.0),)
    with pytest.raises(ValueError, match="not in the hierarchy"):
        generate_program(seed=1, spec=ProgramSpec(wbs=orphan))


# ================================================================== output
def test_reports_write_to_csv_and_round_trip(tmp_path, clean_program):
    written = clean_program.write_csvs(tmp_path)
    assert set(written) == set(REPORT_NAMES)
    for name, path in written.items():
        assert path.exists()
        back = pd.read_csv(path)
        assert len(back) == len(clean_program[name])


def test_no_report_carries_a_wbs_code(messy_program):
    """Deliberate: a reliable key in every file would make the crosswalk
    decoration rather than the load-bearing artifact it is in practice."""
    for name in ("dd1921", "dd1921_1", "flexfile", "srdr"):
        assert "wbs_code" not in messy_program[name].columns, name
