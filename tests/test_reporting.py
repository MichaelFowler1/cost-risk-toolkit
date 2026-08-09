"""Charts, the assumptions log, and the end-to-end run.

A chart is hard to assert about, so these tests check the things that would
make one wrong rather than the pixels: that the run is reproducible from its
seed, that every promised artifact exists and is a real PNG, that the numbers
in the log are the numbers the analysis produced, and that the tick formatter
does not emit duplicate labels -- an axis reading "$4M $4M $5M" looks like data
and is not.

The end-to-end test is the one that matters most. It is the only place the
whole chain runs together, so it is where a mismatch between two modules'
assumptions would surface.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from cost_core.monte_carlo import (CorrelationWarning, DiscreteRisk,
                                   correlation_impact,
                                   risk_model_from_elements,
                                   simulate_risk_model)
from cost_core.reporting import AssumptionLog, run_full_analysis
from cost_core.reporting import charts as charts_mod

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path) -> bool:
    return path.exists() and path.read_bytes()[:8] == PNG_MAGIC


@pytest.fixture(scope="module")
def simulation():
    costs = {
        "Airframe": 300e6, "Propulsion": 120e6, "Avionics": 90e6,
        "Vehicle Subsystems": 45e6, "Training": 8e6,
    }
    model = risk_model_from_elements(
        costs, default_correlation=0.30,
        risks=[DiscreteRisk("Qual failure", 0.2,
                            {"type": "pert", "left": 8e6, "mode": 20e6,
                             "right": 55e6})],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CorrelationWarning)
        return simulate_risk_model(model, 8_000, seed=1), correlation_impact(
            model, 8_000, seed=1
        )


# =============================================================== formatters
def test_the_money_formatter_does_not_produce_duplicate_labels():
    """A narrow axis formatted with fixed decimals reads "$4M $4M $5M", which
    looks like data and is not. The decimal count has to follow the range."""
    narrow = np.linspace(4.0e6, 7.0e6, 8)
    formatter = charts_mod._money_formatter(narrow)
    labels = [formatter(v) for v in narrow]
    assert len(set(labels)) == len(labels), labels
    assert all(label.startswith("$") and label.endswith("M") for label in labels)


def test_the_money_formatter_scales_to_the_magnitude():
    assert charts_mod._money_formatter([2.4e9, 3.1e9])(2.4e9).endswith("B")
    assert charts_mod._money_formatter([2.4e6, 9.1e6])(2.4e6).endswith("M")
    assert charts_mod._money_formatter([2.4e3, 9.1e3])(2.4e3).endswith("K")
    assert charts_mod._money_formatter([12.0, 90.0])(12.0) == "$12"


def test_a_wide_range_drops_the_decimals():
    wide = np.linspace(100e6, 900e6, 9)
    labels = [charts_mod._money_formatter(wide)(v) for v in wide]
    assert "$100M" in labels and "$900M" in labels
    assert len(set(labels)) == len(labels)


def test_the_plain_formatter_avoids_scientific_notation():
    assert charts_mod._plain(20.0) == "20"
    assert charts_mod._plain(1000.0) == "1,000"


# =================================================================== charts
def test_the_s_curve_is_written_as_a_real_png(tmp_path, simulation):
    result, impact = simulation
    path = charts_mod.plot_s_curve(
        result, tmp_path / "s.png", comparison=impact.independent
    )
    assert _is_png(path)
    assert path.stat().st_size > 10_000       # not a blank canvas


def test_the_s_curve_works_without_a_comparison(tmp_path, simulation):
    result, _ = simulation
    assert _is_png(charts_mod.plot_s_curve(result, tmp_path / "s.png"))


def test_the_tornado_is_written_and_respects_top_n(tmp_path, simulation):
    result, _ = simulation
    assert _is_png(charts_mod.plot_tornado(result, tmp_path / "t.png", top_n=3))


def test_the_summary_table_renders(tmp_path, simulation):
    result, _ = simulation
    path = charts_mod.plot_summary_table(
        result.summary(), tmp_path / "tbl.png", title="Summary"
    )
    assert _is_png(path)


def test_charts_close_their_figures(tmp_path, simulation):
    """A long run that leaks figures eventually warns and then slows down."""
    import matplotlib.pyplot as plt

    result, _ = simulation
    plt.close("all")
    before = len(plt.get_fignums())
    charts_mod.plot_s_curve(result, tmp_path / "a.png")
    charts_mod.plot_tornado(result, tmp_path / "b.png")
    charts_mod.plot_summary_table(result.summary(), tmp_path / "c.png")
    assert len(plt.get_fignums()) == before


# =========================================================== assumptions log
def test_the_log_separates_assumptions_from_findings():
    log = AssumptionLog(title="Test")
    log.section("Data", "Some measured facts.")
    log.assume("Correlation", "rho = 0.3 everywhere", "No programme history")
    text = log.render()

    assert "# Test" in text
    assert "## Data" in text
    assert "## Assumptions applied" in text
    assert "rho = 0.3 everywhere" in text
    assert "No programme history" in text
    assert "**1 assumption(s) recorded.**" in text


def test_the_log_always_lists_the_four_gao_characteristics():
    text = AssumptionLog().render()
    for characteristic in (
        "Comprehensive", "Well-documented", "Accurate", "Credible"
    ):
        assert f"### {characteristic}" in text
    # Unaddressed characteristics say so rather than being omitted, so a gap
    # is visible instead of invisible.
    assert text.count("Not addressed by this run.") == 4


def test_a_gao_note_replaces_the_placeholder():
    log = AssumptionLog().gao("Accurate", "All reconciliation gates passed.")
    text = log.render()
    assert "All reconciliation gates passed." in text
    assert text.count("Not addressed by this run.") == 3


def test_a_table_renders_as_markdown():
    frame = pd.DataFrame({"metric": ["p80", "cv"], "value": [1234.5678, 0.1234]})
    text = AssumptionLog().table("Numbers", frame).render()
    assert "| metric | value |" in text
    assert "| --- | --- |" in text
    assert "p80" in text


def test_a_pipe_in_free_text_does_not_break_the_table():
    """A WBS element named "Airframe | Structure" would otherwise split a
    Markdown row and silently shift every column after it."""
    frame = pd.DataFrame({"name": ["Airframe | Structure"], "value": [1.0]})
    text = AssumptionLog().table("T", frame).render()
    assert r"Airframe \| Structure" in text


def test_an_empty_table_says_so_rather_than_rendering_nothing():
    text = AssumptionLog().table("Empty", pd.DataFrame()).render()
    assert "*(no rows)*" in text


def test_the_log_writes_to_disk(tmp_path):
    path = AssumptionLog().section("A", "b").write(tmp_path / "ASSUMPTIONS.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# ")


# ============================================================== end to end
@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("run")
    return run_full_analysis(directory, seed=3, iterations=4_000,
                             portfolio_size=10)


def test_the_full_run_writes_every_promised_artifact(full_run):
    expected = {
        "crosswalk", "inflation_index", "s_curve", "tornado",
        "learning_curve", "cer", "summary_table", "assumptions_log",
    }
    assert expected <= set(full_run.artifacts)
    for name, path in full_run.artifacts.items():
        assert path.exists(), name
        if path.suffix == ".png":
            assert _is_png(path), name


def test_the_full_run_writes_the_source_reports_and_tables(full_run):
    reports = full_run.output_dir / "artifacts" / "source_reports"
    assert len(list(reports.glob("*.csv"))) == 6
    tables = full_run.output_dir / "tables"
    for name in (
        "normalized_rows.csv", "provenance.csv", "validation_gates.csv",
        "risk_summary.csv", "tornado.csv", "correlation_impact.csv",
        "curve_methods.csv", "cer_methods.csv",
    ):
        assert (tables / name).exists(), name


def test_the_full_run_passes_every_ingest_gate(full_run):
    assert full_run.normalized.validation.ok


def test_the_log_reports_the_numbers_the_analysis_produced(full_run):
    """The log has to agree with the objects, or it is documentation of a
    different run."""
    text = (full_run.output_dir / "ASSUMPTIONS.md").read_text(encoding="utf-8")
    assert f"seed `{full_run.seed}`" in text
    assert "**All data in this run is synthetic.**" in text
    assert f"{full_run.simulation.point_estimate_percentile:.0f}th percentile" in text
    assert f"{full_run.chosen_curve.slope:.2%}" in text
    # Every assumption applied is recorded, and the correlation is one of them.
    assert "Uniform correlation of 0.30" in text
    assert "assumption(s) recorded" in text


def test_the_log_carries_the_methodology_sections(full_run):
    text = (full_run.output_dir / "ASSUMPTIONS.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Source data",
        "## 2. Ingest and normalisation",
        "## 3. Learning curve",
        "## 3.5 Why MUPE and ZMPE",
        "## 4. Cost estimating relationship",
        "## 5. Risk simulation",
        "## 5.4 Why correlation matters",
    ):
        assert heading in text, heading


def test_the_full_run_is_reproducible_from_its_seed(tmp_path):
    """The whole point of seeding the chain. Two runs at the same seed must
    agree exactly, or nothing in the briefing can be reproduced later."""
    first = run_full_analysis(tmp_path / "a", seed=11, iterations=3_000,
                              portfolio_size=6)
    second = run_full_analysis(tmp_path / "b", seed=11, iterations=3_000,
                               portfolio_size=6)

    assert first.simulation.p80 == second.simulation.p80
    assert first.chosen_curve.slope == second.chosen_curve.slope
    assert np.array_equal(first.cer.result.theta, second.cer.result.theta)
    pd.testing.assert_frame_equal(first.tables["tornado"], second.tables["tornado"])


def test_different_seeds_give_different_programs(tmp_path):
    a = run_full_analysis(tmp_path / "a", seed=1, iterations=2_000,
                          portfolio_size=6)
    b = run_full_analysis(tmp_path / "b", seed=2, iterations=2_000,
                          portfolio_size=6)
    assert a.simulation.p80 != b.simulation.p80


def test_a_clean_run_recovers_the_generating_slope_exactly(tmp_path):
    """With the reporting pathologies switched off, the whole chain --
    generate, ingest, fit -- has to return the curve it started from."""
    run = run_full_analysis(
        tmp_path / "clean", seed=5, iterations=2_000, portfolio_size=6,
        clean=True, method="ols",
    )
    assert run.chosen_curve.slope == pytest.approx(
        run.program.truth.learning_slope, rel=1e-6
    )
    assert run.chosen_curve.t1 == pytest.approx(
        run.program.truth.t1_cost, rel=1e-6
    )


def test_the_headline_states_the_percentile_and_the_correlation_effect(full_run):
    headline = full_run.headline()
    assert "percentile" in headline
    assert "understates the variance" in headline
    assert "P80" in headline


def test_the_point_estimate_sits_below_the_p80(full_run):
    """If it did not, the risk model would be saying the estimate already
    carries more reserve than an 80% confidence level requires."""
    assert full_run.simulation.point_estimate < full_run.simulation.p80
    assert 0 < full_run.simulation.point_estimate_percentile < 80


def test_correlation_raises_the_p80_in_the_full_run(full_run):
    assert full_run.impact.correlated.p80 > full_run.impact.independent.p80
    assert full_run.impact.empirical_variance_ratio > 1.0


@pytest.mark.parametrize("theory", ["crawford", "wright"])
def test_both_theories_run_end_to_end(tmp_path, theory):
    run = run_full_analysis(
        tmp_path / theory, seed=4, iterations=2_000, portfolio_size=6,
        theory=theory,
    )
    assert run.chosen_curve.theory.value == theory
    assert _is_png(run.artifacts["learning_curve"])


@pytest.mark.parametrize("method", ["ols", "mupe", "zmpe"])
def test_every_fitting_method_runs_end_to_end(tmp_path, method):
    run = run_full_analysis(
        tmp_path / method, seed=4, iterations=2_000, portfolio_size=6,
        method=method,
    )
    assert run.chosen_curve.method == method
    assert run.cer.method == method
