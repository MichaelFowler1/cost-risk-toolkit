"""The ingest and normalisation pipeline.

The generator hands us something rare: messy data whose correct answer is
known. So these are not "does the ETL run" tests. The central assertion is that
a program carrying every pathology at once -- drifted names, mixed then-year
and base-year dollars, resubmitted periods, a mid-program quantity change,
rate breaks -- normalises back to the generating truth **exactly**, to the
cent. Every reversible pathology has to be reversed, and the arithmetic that
reverses it is arithmetic, so anything past floating-point noise is a defect.

The other half is what the pipeline does when it *cannot* recover something. A
missing period is not reconstructible, so it must be named. A WBS element the
crosswalk has never seen must stop the run, not quietly roll up to nothing.
Those cases get as much attention here as the happy path, because they are the
ones where a silent pipeline puts an indefensible number in front of someone.
"""
import numpy as np
import pandas as pd
import pytest

from cost_core.ingest import (CATEGORY_ALL, NORMALIZED_COLUMNS,
                              PROVENANCE_COLUMNS, WBS_PROGRAM_LEVEL,
                              Crosswalk, CrosswalkError, InflationError,
                              InflationTable, IngestError, normalize,
                              normalize_program)
from cost_core.synth import PathologyConfig, generate_program

CLEAN = PathologyConfig.clean()

#: A seed whose messy program actually exercises every pathology at once.
MESSY_SEED = 3


@pytest.fixture(scope="module")
def clean_data():
    program = generate_program(seed=MESSY_SEED, pathologies=CLEAN)
    return program, normalize_program(program)


@pytest.fixture(scope="module")
def messy_data():
    program = generate_program(seed=MESSY_SEED)
    return program, normalize_program(program)


def _index_table(program):
    return InflationTable.from_mapping(
        program.truth.inflation_index, source="test fixture"
    )


# ==================================================== exact truth recovery
def test_a_clean_program_normalises_to_the_truth_exactly(clean_data):
    program, data = clean_data
    assert data.total_dollars("DD1921") == pytest.approx(
        program.truth.total_dollars_by, rel=1e-12
    )


def test_a_messy_program_also_normalises_to_the_truth_exactly(messy_data):
    """The headline test of this module. Name drift, then-year dollars and
    resubmissions are all reversible by construction, so the pipeline that
    reverses them has no excuse for landing anywhere but the exact truth."""
    program, data = messy_data
    for report_type in ("DD1921", "DD1921-1", "FLEXFILE"):
        assert data.total_dollars(report_type) == pytest.approx(
            program.truth.total_dollars_by, rel=1e-12
        ), report_type


def test_the_messy_fixture_really_is_messy(messy_data):
    """Guards the test above: if the seed stopped producing pathologies, the
    exact-recovery assertion would be passing for the wrong reason."""
    program, _ = messy_data
    from cost_core.synth import DEFAULT_WBS

    canonical = {w.name for w in DEFAULT_WBS}
    reported = set(program["dd1921"]["wbs_element_name"])
    assert reported - canonical, "expected drifted element names"
    assert set(program["dd1921"]["basis"]) == {"BY", "TY"}, "expected mixed bases"
    assert (
        program.truth.lot_quantities != program.truth.planned_quantities
    ), "expected a quantity rebaseline"


def test_recurring_lot_costs_survive_normalisation(messy_data):
    program, data = messy_data
    lots = data.learning_curve_input()
    truth = program.truth.recurring_lot_costs().set_index("lot")
    assert lots["lot_cost"].to_numpy() == pytest.approx(
        truth.loc[lots["lot"], "lot_cost_by"].to_numpy(), rel=1e-12
    )
    assert lots["lot_quantity"].to_numpy() == pytest.approx(
        truth.loc[lots["lot"], "quantity"].to_numpy()
    )


def test_element_level_totals_match_the_truth_element_by_element(messy_data):
    """Not just the grand total: a crosswalk that merged two elements into one
    would still balance in aggregate."""
    program, data = messy_data
    got = (
        data.by_report("DD1921")
        .groupby("wbs_element")["dollars"]
        .sum()
        .sort_index()
    )
    want = (
        program.truth.cells.groupby("wbs_name")["dollars_by"].sum().sort_index()
    )
    assert list(got.index) == list(want.index)
    assert got.to_numpy() == pytest.approx(want.to_numpy(), rel=1e-12)


def test_hours_survive_normalisation_unescalated(messy_data):
    """Hours are hours. Escalating them would be a category error, and it is
    an easy one to make when the same loop handles both columns."""
    program, data = messy_data
    got = data.by_report("DD1921-1")["hours"].sum()
    assert got == pytest.approx(program.truth.total_hours, rel=1e-12)


# ============================================== the individual pathologies
def test_then_year_dollars_are_deflated_and_base_year_ones_are_not():
    ty = PathologyConfig(
        name_drift_prob=0.0, then_year_prob=1.0, resubmission_prob=0.0,
        missing_period_prob=0.0, outlier_lot_prob=0.0, quantity_change=False,
        noise_cv=0.0, eac_optimism=0.0,
    )
    program = generate_program(seed=5, pathologies=ty)
    data = normalize_program(program)
    assert data.total_dollars("DD1921") == pytest.approx(
        program.truth.total_dollars_by, rel=1e-12
    )
    # Every source row was then-year, so every factor is a real deflation.
    prov = data.provenance[data.provenance["source_report"] == "dd1921"]
    assert (prov["basis_raw"] == "TY").all()
    assert (prov["inflation_factor"] != 1.0).any()


def test_resubmitted_periods_collapse_to_the_latest_report_date():
    resub = PathologyConfig(
        name_drift_prob=0.0, then_year_prob=0.0, resubmission_prob=1.0,
        missing_period_prob=0.0, outlier_lot_prob=0.0, quantity_change=False,
        noise_cv=0.0, eac_optimism=0.0,
    )
    program = generate_program(seed=5, pathologies=resub)
    data = normalize_program(program)

    assert data.total_dollars("DD1921") == pytest.approx(
        program.truth.total_dollars_by, rel=1e-12
    )
    # Exactly half the source rows were superseded, and they are kept, not
    # deleted -- an auditor has to be able to see the earlier submission.
    prov = data.provenance
    assert prov["superseded"].sum() > 0
    assert prov["superseded"].mean() == pytest.approx(0.5, abs=0.01)


def test_superseded_rows_keep_their_original_values_in_provenance():
    resub = PathologyConfig(
        name_drift_prob=0.0, then_year_prob=0.0, resubmission_prob=1.0,
        missing_period_prob=0.0, outlier_lot_prob=0.0, quantity_change=False,
        noise_cv=0.0, eac_optimism=0.0,
    )
    data = normalize_program(generate_program(seed=5, pathologies=resub))
    superseded = data.provenance[data.provenance["superseded"]]
    live = data.provenance[~data.provenance["superseded"]]
    # The wrong submission really did carry different numbers.
    assert superseded["dollars_raw"].sum() != pytest.approx(
        live["dollars_raw"].sum(), rel=1e-6
    )


def test_drifted_names_are_resolved_and_the_rule_is_recorded(messy_data):
    _, data = messy_data
    rules = set(data.provenance["crosswalk_rule"])
    assert "exact" in rules
    # Aliases resolved, and how they resolved is on the record.
    assert rules <= {"exact", "casefold", "unmatched", "n/a"}
    element_rows = data.by_report("DD1921")
    assert element_rows["wbs_element"].notna().all()


def test_missing_periods_are_surfaced_as_a_warning_not_filled():
    gappy = PathologyConfig(
        name_drift_prob=0.0, then_year_prob=0.0, resubmission_prob=0.0,
        missing_period_prob=0.5, outlier_lot_prob=0.0, quantity_change=False,
        noise_cv=0.0, eac_optimism=0.0,
    )
    program = generate_program(seed=2, pathologies=gappy)
    data = normalize_program(program)          # warning, not an error

    gate = next(g for g in data.validation.gates if g.name == "reporting_gaps")
    assert not gate.passed
    assert gate.severity == "warning"
    assert gate.metrics["missing"], "the gate must name which lots are gone"
    # And nothing was invented to cover the hole.
    cost_rows = data.rows[data.rows["report_type"].isin(
        ["DD1921", "DD1921-1", "DD1921-2", "FLEXFILE"]
    )]
    present = set(cost_rows["lot"].dropna().astype(int))
    assert set(gate.metrics["missing"]).isdisjoint(present)


def test_the_quantity_report_cannot_mask_a_cost_reporting_gap():
    """The Quantity Data Report restates every prior lot in each submission,
    so a lot whose cost period was never filed still shows up there. The gap
    gate must look only at cost-bearing reports or it would pass on data that
    is genuinely missing."""
    gappy = PathologyConfig(
        name_drift_prob=0.0, then_year_prob=0.0, resubmission_prob=0.0,
        missing_period_prob=0.5, outlier_lot_prob=0.0, quantity_change=False,
        noise_cv=0.0, eac_optimism=0.0,
    )
    data = normalize_program(generate_program(seed=2, pathologies=gappy))
    gate = next(g for g in data.validation.gates if g.name == "reporting_gaps")
    missing = set(gate.metrics["missing"])
    quantity_lots = set(
        data.by_report("QUANTITY")["lot"].dropna().astype(int)
    )
    # The quantity report does cover the missing lots -- that is the trap.
    assert missing & quantity_lots
    # The gate reported them missing anyway.
    assert missing


def test_a_quantity_rebaseline_carries_through_to_the_lot_table(messy_data):
    program, data = messy_data
    lots = data.learning_curve_input()
    assert lots["lot_quantity"].tolist() == list(program.truth.lot_quantities)
    assert lots["lot_quantity"].tolist() != list(program.truth.planned_quantities)


# ================================================== unmatched names are fatal
def test_an_unknown_wbs_name_fails_the_run_and_is_named():
    """The single most dangerous silent failure in a cost pipeline: an element
    nobody recognises, quietly dropped, and a total that looks fine."""
    program = generate_program(seed=1, pathologies=CLEAN)
    reports = {k: v.copy() for k, v in program.reports.items()}
    reports["dd1921"].loc[0, "wbs_element_name"] = "Mystery Widget Assembly"

    with pytest.raises(IngestError, match="Mystery Widget Assembly"):
        normalize(
            reports,
            crosswalk=Crosswalk.default(),
            inflation=_index_table(program),
            base_year=program.truth.base_year,
        )


def test_an_unmatched_row_is_kept_not_dropped():
    """With strict off, the row still has to be in the output -- surfaced as
    unresolved, never absorbed."""
    program = generate_program(seed=1, pathologies=CLEAN)
    reports = {k: v.copy() for k, v in program.reports.items()}
    reports["dd1921"].loc[0, "wbs_element_name"] = "Mystery Widget Assembly"

    data = normalize(
        reports,
        crosswalk=Crosswalk.default(),
        inflation=_index_table(program),
        base_year=program.truth.base_year,
        strict=False,
    )
    assert not data.validation.ok
    # The dollars are still there, attached to an unresolved element.
    assert data.total_dollars("DD1921") == pytest.approx(
        program.truth.total_dollars_by, rel=1e-12
    )
    unresolved = data.rows[data.rows["wbs_element"].isna()]
    assert len(unresolved) == 1


def test_the_failure_message_suggests_a_correction():
    program = generate_program(seed=1, pathologies=CLEAN)
    reports = {k: v.copy() for k, v in program.reports.items()}
    reports["dd1921"].loc[0, "wbs_element_name"] = "Airframe Struktur"

    with pytest.raises(IngestError, match="airframe"):
        normalize(
            reports,
            crosswalk=Crosswalk.default(),
            inflation=_index_table(program),
            base_year=program.truth.base_year,
        )


# ============================================ reconciliation gates actually bite
def test_a_corrupted_report_breaks_the_cross_report_gate():
    """Three views of the same dollars must agree. If one is tampered with,
    the gate has to catch it -- otherwise it is checking nothing."""
    program = generate_program(seed=1, pathologies=CLEAN)
    reports = {k: v.copy() for k, v in program.reports.items()}
    reports["dd1921"]["cost_incurred_period"] *= 1.05

    with pytest.raises(IngestError, match="cross_report_reconciliation|disagree"):
        normalize(
            reports,
            crosswalk=Crosswalk.default(),
            inflation=_index_table(program),
            base_year=program.truth.base_year,
        )


def test_a_negative_cost_is_refused():
    program = generate_program(seed=1, pathologies=CLEAN)
    reports = {k: v.copy() for k, v in program.reports.items()}
    reports["dd1921_2"].loc[0, "recurring_lot_cost"] = -1_000.0

    with pytest.raises(IngestError, match="negative"):
        normalize(
            reports,
            crosswalk=Crosswalk.default(),
            inflation=_index_table(program),
            base_year=program.truth.base_year,
        )


def test_every_gate_runs_and_reports_its_verdict(clean_data):
    _, data = clean_data
    frame = data.validation.to_frame()
    expected = {
        "rows_extracted",
        "extraction_covers_every_source_row",
        "wbs_names_resolved",
        "resubmissions_deduplicated",
        "dollars_normalised",
        "row_count_accounting",
        "provenance_complete",
        "dollars_reconcile_to_source",
        "cross_report_reconciliation",
        "reporting_gaps",
        "no_negative_costs",
    }
    assert expected <= set(frame["gate"])
    assert data.validation.ok


def test_an_empty_or_unknown_report_is_refused():
    program = generate_program(seed=1, pathologies=CLEAN)
    with pytest.raises(IngestError, match="No extractor"):
        normalize(
            {"dd1922": pd.DataFrame({"a": [1]})},
            crosswalk=Crosswalk.default(),
            inflation=_index_table(program),
            base_year=2020,
        )
    with pytest.raises(IngestError, match="nothing to normalise"):
        normalize(
            {},
            crosswalk=Crosswalk.default(),
            inflation=_index_table(program),
            base_year=2020,
        )


# ============================================================== provenance
def test_every_output_row_traces_back_to_source_rows(clean_data):
    _, data = clean_data
    for uid in data.rows["row_uid"].head(20):
        trace = data.trace(uid)
        assert len(trace) >= 1
        assert set(trace.columns) == set(PROVENANCE_COLUMNS)
        assert (trace["source_row"] >= 0).all()


def test_provenance_preserves_the_raw_value_and_its_stated_year(messy_data):
    """The normalised number must always be walkable back to what the
    contractor actually wrote down."""
    _, data = messy_data
    prov = data.provenance[
        (data.provenance["source_report"] == "dd1921")
        & (~data.provenance["superseded"])
    ]
    assert prov["dollars_raw"].notna().all()
    assert prov["dollar_year_raw"].notna().all()
    assert prov["basis_raw"].isin(["BY", "TY"]).all()
    # raw * factor == normalised, exactly.
    reconstructed = prov["dollars_raw"] * prov["inflation_factor"]
    assert reconstructed.sum() == pytest.approx(
        data.total_dollars("DD1921"), rel=1e-12
    )


def test_srdr_provenance_points_at_real_source_rows(messy_data):
    """The SRDR melt turns one source row into one per activity. Provenance
    must still name the original row, not a position in the melted frame."""
    program, data = messy_data
    n_source = len(program["srdr"])
    srdr_prov = data.provenance[data.provenance["source_report"] == "srdr"]
    assert srdr_prov["source_row"].max() == n_source - 1
    assert set(srdr_prov["source_row"].unique()) == set(range(n_source))


def test_an_unknown_row_uid_is_refused(clean_data):
    _, data = clean_data
    with pytest.raises(KeyError, match="No provenance"):
        data.trace("not-a-real-uid")


def test_the_normalised_table_has_exactly_the_documented_columns(clean_data):
    _, data = clean_data
    assert tuple(data.rows.columns) == NORMALIZED_COLUMNS
    assert tuple(data.provenance.columns) == PROVENANCE_COLUMNS


# ============================================================== base years
def test_the_base_year_is_reselectable_and_scales_exactly():
    """A raw index means re-basing is a ratio, not a regeneration. Restating
    in FY2026 must multiply the FY2020 total by exactly index(2026)/index(2020)."""
    program = generate_program(seed=1, pathologies=CLEAN)
    index = _index_table(program)

    at_2020 = normalize_program(program, base_year=2020)
    at_2026 = normalize_program(program, base_year=2026, inflation=index)

    ratio = index.factor(2020, 2026)
    assert at_2026.total_dollars("DD1921") == pytest.approx(
        at_2020.total_dollars("DD1921") * ratio, rel=1e-12
    )
    assert (at_2026.rows["dollar_year"] == 2026).all()


def test_authoritative_view_does_not_double_count(clean_data):
    """Summing the whole table triples every dollar, because three report
    types describe the same costs. The authoritative view exists so that is
    not the default thing to do."""
    program, data = clean_data
    naive = float(data.rows["dollars"].sum())
    authoritative = float(data.authoritative()["dollars"].sum())
    assert authoritative == pytest.approx(program.truth.total_dollars_by, rel=1e-12)
    assert naive > authoritative * 2


def test_learning_curve_input_derives_contiguous_unit_ranges(clean_data):
    program, data = clean_data
    lots = data.learning_curve_input()
    assert lots["first_unit"].iloc[0] == 1
    # Lots must tile the unit sequence with no gaps and no overlap.
    assert lots["first_unit"].tolist()[1:] == (lots["last_unit"] + 1).tolist()[:-1]
    assert lots["last_unit"].iloc[-1] == sum(program.truth.lot_quantities)
    assert (lots["unit_cost"] * lots["lot_quantity"]).to_numpy() == pytest.approx(
        lots["lot_cost"].to_numpy(), rel=1e-12
    )


def test_software_input_aggregates_srdr_effort(messy_data):
    program, data = messy_data
    software = data.software_input()
    assert len(software) == 1
    assert software["effort_hours"].iloc[0] == pytest.approx(
        program["srdr"]["total_effort_hours"].sum(), rel=1e-9
    )


# ============================================================== crosswalk
def test_the_crosswalk_round_trips_through_disk(tmp_path):
    original = Crosswalk.default()
    path = original.save(tmp_path / "wbs_crosswalk.csv")
    assert path.exists()
    reloaded = Crosswalk.load(path)
    assert reloaded.mapping == original.mapping
    assert reloaded.codes == original.codes


def test_a_missing_crosswalk_is_refused_rather_than_defaulted(tmp_path):
    """An absent crosswalk is a missing decision. Inventing one would mean
    guessing at where cost rolls up."""
    with pytest.raises(FileNotFoundError, match="No crosswalk artifact"):
        Crosswalk.load(tmp_path / "nope.csv")


def test_a_crosswalk_mapping_one_name_to_two_targets_is_refused(tmp_path):
    path = tmp_path / "conflict.csv"
    pd.DataFrame(
        {
            "reported_name": ["Air Frame", "Air Frame"],
            "canonical_name": ["Airframe", "Propulsion"],
            "wbs_code": ["1.1", "1.2"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(CrosswalkError, match="cannot roll up to two places"):
        Crosswalk.load(path)


def test_a_crosswalk_missing_its_key_columns_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": ["a"], "bar": ["b"]}).to_csv(path, index=False)
    with pytest.raises(CrosswalkError, match="missing required column"):
        Crosswalk.load(path)


def test_casefold_matching_resolves_case_variants_and_says_so():
    cw = Crosswalk.default()
    assert cw.resolve("Airframe").rule == "exact"
    # "AIRFRAME" is a listed alias, so it matches exactly; a case variant that
    # is *not* listed is what exercises the fallback.
    assert cw.resolve("AIRFRAME").rule == "exact"
    for variant in ("  airframe  ", "air frame", "AiRfRaMe"):
        resolved = cw.resolve(variant)
        assert resolved.canonical == "Airframe", variant
        assert resolved.rule == "casefold", variant


def test_casefold_matching_can_be_switched_off():
    strict = Crosswalk.default(allow_casefold=False)
    assert not strict.resolve("airframe").matched
    assert strict.resolve("Airframe").matched


def test_the_crosswalk_never_invents_a_target():
    cw = Crosswalk.default()
    assert not cw.resolve("Quantum Flux Capacitor").matched
    # Suggestions are offered but never applied.
    assert cw.resolve("Airframe Struktur").rule == "unmatched"
    assert cw.suggest("Airframe Struktur")


def test_adding_a_mapping_to_an_unknown_target_is_refused():
    cw = Crosswalk.default()
    with pytest.raises(CrosswalkError, match="unknown canonical name"):
        cw.with_additions({"Widget": "Nonexistent Element"})


def test_adding_a_valid_mapping_resolves_the_name():
    cw = Crosswalk.default().with_additions({"Airframe Struktur": "Airframe"})
    resolved = cw.resolve("Airframe Struktur")
    assert resolved.canonical == "Airframe"
    assert resolved.rule == "exact"


# ============================================================== inflation
def test_the_inflation_table_round_trips_through_disk(tmp_path):
    table = InflationTable.from_rate(0.03, base_year=2020, first_year=2015,
                                     last_year=2030)
    path = table.save(tmp_path / "inflation.csv")
    reloaded = InflationTable.load(path)
    assert reloaded.value(2025) == pytest.approx(table.value(2025), rel=1e-12)
    assert reloaded.coverage["composite"] == (2015, 2030)


def test_index_factors_are_ratios_and_compose():
    table = InflationTable.from_rate(0.025, base_year=2020)
    assert table.factor(2020, 2020) == pytest.approx(1.0, rel=1e-15)
    assert table.factor(2020, 2025) == pytest.approx(1.025**5, rel=1e-12)
    # A round trip is exactly the identity, which is what makes re-basing safe.
    assert table.factor(2020, 2025) * table.factor(2025, 2020) == pytest.approx(
        1.0, rel=1e-12
    )


def test_a_year_outside_the_index_is_refused_not_extrapolated():
    """Extending an index past its published range is a judgement call and
    making it silently is not defensible."""
    table = InflationTable.from_rate(0.025, first_year=2018, last_year=2024)
    with pytest.raises(InflationError, match="FY2018-FY2024"):
        table.value(2030)


def test_an_unknown_index_name_is_refused():
    table = InflationTable.from_rate(0.025)
    with pytest.raises(InflationError, match="No index named"):
        table.value(2020, name="labour_only")


def test_a_missing_index_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="No inflation index"):
        InflationTable.load(tmp_path / "nope.csv")


def test_a_non_positive_index_value_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "index_name": ["composite"] * 2,
            "fiscal_year": [2020, 2021],
            "index_value": [1.0, -0.5],
        }
    ).to_csv(path, index=False)
    with pytest.raises(InflationError, match="non-positive"):
        InflationTable.load(path)


def test_a_duplicated_year_in_an_index_is_refused(tmp_path):
    path = tmp_path / "dupe.csv"
    pd.DataFrame(
        {
            "index_name": ["composite"] * 2,
            "fiscal_year": [2020, 2020],
            "index_value": [1.0, 1.1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(InflationError, match="duplicate year"):
        InflationTable.load(path)


def test_to_base_year_rejects_mismatched_lengths():
    table = InflationTable.from_rate(0.025)
    with pytest.raises(InflationError, match="one to one"):
        table.to_base_year([1.0, 2.0, 3.0], [2020, 2021], 2020)


# ============================================================= determinism
def test_normalisation_is_deterministic():
    program = generate_program(seed=9)
    a = normalize_program(program)
    b = normalize_program(program)
    pd.testing.assert_frame_equal(a.rows, b.rows)
    assert a.total_dollars() == b.total_dollars()
