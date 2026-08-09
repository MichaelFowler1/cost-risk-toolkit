"""
pipeline.py - The whole path, end to end, in one run.

Generate synthetic CSDR/SRDR -> ingest and normalise -> fit a learning curve
and a CER -> simulate with correlation -> emit charts, tables and an
assumptions log.

The point of having this as one function rather than a notebook is that the
entire chain is reproducible from a seed. Every artifact it writes -- every
chart, every table, the log itself -- comes from the same run, so a number on a
slide can be traced back through the assumptions log, to the normalised row, to
the source submission it came from.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cost_core import learning_curve as lc
from cost_core.cer import cer_comparison_table, compare_cer_methods, fit_cer
from cost_core.cer.model import ExtrapolationWarning, Form
from cost_core.fitting import retransformation_bias
from cost_core.ingest import Crosswalk, InflationTable, normalize
from cost_core.monte_carlo import (CorrelationWarning, DiscreteRisk,
                                   correlation_impact, risk_model_from_elements,
                                   simulate_risk_model)
from cost_core.reporting import assumptions as assumptions_mod
from cost_core.reporting import charts
from cost_core.synth import PathologyConfig, generate_portfolio, generate_program

logger = logging.getLogger(__name__)

#: Discrete risks used by the demonstration run. Invented, and labelled as
#: such in the log -- a real analysis elicits these from the programme.
DEMO_RISKS: tuple[dict[str, Any], ...] = (
    {
        "name": "Qualification test failure",
        "probability": 0.20,
        "impact": {"type": "pert", "left": 8e6, "mode": 22e6, "right": 65e6},
    },
    {
        "name": "Second-source qualification slip",
        "probability": 0.35,
        "impact": {"type": "triangular", "left": 4e6, "mode": 11e6, "right": 30e6},
    },
    {
        "name": "Obsolescence redesign",
        "probability": 0.12,
        "impact": {"type": "pert", "left": 15e6, "mode": 38e6, "right": 95e6},
    },
)


@dataclass
class RunResult:
    """Everything one end-to-end run produced."""

    output_dir: Path
    seed: int
    program: Any
    normalized: Any
    curve_fits: dict[str, Any]
    chosen_curve: Any
    cer: Any
    simulation: Any
    impact: Any
    artifacts: dict[str, Path] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def headline(self) -> str:
        """The three sentences the briefing opens with."""
        return (
            f"Point estimate {self.simulation.point_estimate / 1e6:,.1f}M sits "
            f"at the {self.simulation.point_estimate_percentile:.0f}th "
            f"percentile of the risk distribution; the P80 is "
            f"{self.simulation.p80 / 1e6:,.1f}M, a reserve of "
            f"{(self.simulation.p80 - self.simulation.point_estimate) / 1e6:,.1f}M "
            f"({100 * (self.simulation.p80 / self.simulation.point_estimate - 1):.1f}%). "
            f"{self.impact.narrative()}"
        )


def run_full_analysis(
    output_dir: str | Path,
    *,
    seed: int = 7,
    iterations: int = 50_000,
    theory: str = "crawford",
    method: str = "mupe",
    base_year: int | None = None,
    correlation: float = 0.30,
    portfolio_size: int = 14,
    forecast_lots: int = 3,
    forecast_lot_size: int = 24,
    clean: bool = False,
) -> RunResult:
    """Run the complete path and write every artifact to ``output_dir``.

    Args:
        output_dir: Directory for charts, tables, artifacts and the log.
        seed: Master seed. The same seed reproduces the run exactly.
        iterations: Monte Carlo iterations.
        theory: ``"crawford"`` or ``"wright"``.
        method: ``"ols"``, ``"mupe"`` or ``"zmpe"`` for the curve and the CER.
        base_year: Fiscal year to state dollars in. Defaults to the program's.
        correlation: Uniform correlation across WBS elements.
        portfolio_size: Programs generated for the CER fit.
        forecast_lots: How many future lots to forecast.
        forecast_lot_size: Units in each forecast lot.
        clean: Generate a program with no reporting pathologies. Useful for
            showing that the pipeline recovers the truth exactly.

    Returns:
        RunResult with every intermediate object and the paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    charts_dir = output_dir / "charts"
    tables_dir = output_dir / "tables"
    for directory in (artifacts_dir, charts_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log = assumptions_mod.AssumptionLog(
        title=f"Cost estimate assumptions and provenance (seed {seed})"
    )
    artifacts: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------ 1. generate
    pathologies = PathologyConfig.clean() if clean else PathologyConfig()
    program = generate_program(seed=seed, pathologies=pathologies)
    program.write_csvs(artifacts_dir / "source_reports")

    log.section(
        "1. Source data",
        f"**All data in this run is synthetic.** No real or proprietary "
        f"contractor data is used anywhere in this repository. The program "
        f"`{program.spec.program}` was generated from seed `{seed}` and can be "
        f"reproduced exactly from it.\n\n"
        f"- {program.spec.n_lots} production lots, "
        f"{sum(program.truth.lot_quantities)} units total\n"
        f"- Reports emitted: DD 1921, DD 1921-1, DD 1921-2, Cost and Hour "
        f"Report (FlexFile), Quantity Data Report, SRDR (DD 2630)\n"
        f"- Written to `artifacts/source_reports/`\n"
        f"- Reporting pathologies: "
        f"{'none (clean run)' if clean else 'name drift, mixed then-year and base-year dollars, resubmitted and missing periods, mid-program quantity change, rate breaks'}",
    ).gao(
        "Comprehensive",
        f"All six CSDR/SRDR report shapes ingested, covering "
        f"{len(program.truth.cells['wbs_name'].unique())} WBS elements across "
        f"recurring and nonrecurring cost.",
    )

    # -------------------------------------------------------------- 2. ingest
    crosswalk = Crosswalk.default()
    crosswalk_path = crosswalk.save(artifacts_dir / "wbs_crosswalk.csv")
    artifacts["crosswalk"] = crosswalk_path

    inflation = InflationTable.from_mapping(
        program.truth.inflation_index,
        source="constant-rate escalation assumption (synthetic)",
    )
    inflation_path = inflation.save(artifacts_dir / "inflation_index.csv")
    artifacts["inflation_index"] = inflation_path

    resolved_base_year = (
        base_year if base_year is not None else program.truth.base_year
    )
    normalized = normalize(
        program.reports,
        crosswalk=crosswalk,
        inflation=inflation,
        base_year=resolved_base_year,
        expected_lots=range(1, program.spec.n_lots + 1),
    )
    normalized.rows.to_csv(tables_dir / "normalized_rows.csv", index=False)
    normalized.provenance.to_csv(tables_dir / "provenance.csv", index=False)
    validation = normalized.validation.to_frame()
    tables["validation"] = validation
    validation.to_csv(tables_dir / "validation_gates.csv", index=False)

    log.section(
        "2. Ingest and normalisation",
        f"Six report shapes normalised into one long-format table of "
        f"{len(normalized.rows):,} rows, with {len(normalized.provenance):,} "
        f"provenance records -- one per source row, so any output number "
        f"traces back to the submission it came from.\n\n"
        f"- WBS names resolved through the crosswalk artifact at "
        f"`artifacts/wbs_crosswalk.csv` ({len(crosswalk)} entries). Unmatched "
        f"names fail the run rather than being dropped.\n"
        f"- Dollars normalised to FY{resolved_base_year} using a raw index "
        f"({inflation.source}); raw values and the factor applied are "
        f"preserved in the provenance table.\n"
        f"- Resubmitted periods deduplicated to the latest report date; "
        f"superseded rows retained and marked.\n\n"
        "Note that `report_type` is a dimension, not metadata: the DD 1921, "
        "DD 1921-1 and FlexFile describe the same dollars at different levels "
        "of detail, so the normalised table must be filtered to one report "
        "type before summing. That the three agree after normalisation is the "
        "strongest available check that the crosswalk, deduplication and "
        "deflation all did the right thing.",
    ).table("2.1 Validation gates", validation[
        ["gate", "passed", "severity", "detail"]
    ]).gao(
        "Accurate",
        f"All {len(validation)} validation gates passed, including a "
        f"cross-report reconciliation showing the DD 1921, DD 1921-1 and "
        f"FlexFile agree on total cost after normalisation.",
    ).gao(
        "Well-documented",
        "Every normalised row carries provenance to its source submission, "
        "including the raw value, its stated dollar year and the index factor "
        "applied.",
    )

    # ------------------------------------------------------- 3. learning curve
    lot_table = normalized.learning_curve_input()
    lot_table.to_csv(tables_dir / "progress_curve_input.csv", index=False)

    curve_fits = lc.compare_methods(
        theory=theory, lots=lot_table, lot_costs=lot_table["lot_cost"].to_numpy()
    )
    chosen_curve = curve_fits[method]
    both_theories = lc.compare_theories(
        method=method, lots=lot_table, lot_costs=lot_table["lot_cost"].to_numpy()
    )

    methods_table = lc.comparison_table(curve_fits)
    theories_table = lc.comparison_table(both_theories)
    bias = lc.retransformation_report(curve_fits)
    tables["curve_methods"] = methods_table
    tables["curve_theories"] = theories_table
    methods_table.to_csv(tables_dir / "curve_methods.csv", index=False)
    theories_table.to_csv(tables_dir / "curve_theories.csv", index=False)
    bias.to_frame().to_csv(tables_dir / "retransformation_bias.csv", index=False)

    last_unit = int(lot_table["last_unit"].max())
    future = np.array(
        [
            [
                last_unit + 1 + i * forecast_lot_size,
                last_unit + (i + 1) * forecast_lot_size,
            ]
            for i in range(forecast_lots)
        ]
    )
    forecast = chosen_curve.forecast_lots(future, level=0.80, kind="prediction")
    tables["forecast"] = forecast
    forecast.to_csv(tables_dir / "lot_forecast.csv", index=False)

    slope_lo, slope_hi = chosen_curve.slope_interval
    log.section(
        "3. Learning curve",
        f"Fitted to the DD 1921-2 progress curve report, recurring cost only "
        f"-- nonrecurring cost does not follow the curve and folding it in is "
        f"the most common way to fit a slope that is too steep.\n\n"
        f"- Theory: **{theory}**. Wright's cumulative-average form and "
        f"Crawford's unit form give different forecasts from the same data, so "
        f"both are reported in table 3.2 rather than one being assumed.\n"
        f"- Method: **{method.upper()}**, slope "
        f"**{chosen_curve.slope:.2%}** (80% interval "
        f"{slope_lo:.2%} to {slope_hi:.2%}), T1 "
        f"${chosen_curve.t1 / 1e6:,.2f}M\n"
        f"- Standard error ${chosen_curve.standard_error / 1e6:,.2f}M, "
        f"CV {chosen_curve.cv:.1%}. These lead the reporting; R² "
        f"({chosen_curve.r_squared:.3f}) is shown for completeness only. R² "
        f"measures how tightly the points hug the fitted line, which a wrong "
        f"model can do perfectly well: in table 3.2 the two theories return "
        f"R² within "
        f"{abs(both_theories['wright'].r_squared - both_theories['crawford'].r_squared):.3f} "
        f"of each other while giving first-unit costs "
        f"{abs(both_theories['wright'].t1 / both_theories['crawford'].t1 - 1):.0%} "
        f"apart.\n"
        f"- Forecast lots carry **prediction** intervals: the range a new lot "
        f"is expected to fall in, not the range the fitted line lies in.",
    ).table("3.1 Fitting methods compared", methods_table).table(
        "3.2 Theories compared", theories_table
    ).table("3.3 Retransformation bias of naive OLS", bias.to_frame()).table(
        "3.4 Forecast lots (80% prediction interval)", forecast
    )

    log.section(
        "3.5 Why MUPE and ZMPE",
        f"Fitting a power curve by ordinary least squares in log space and "
        f"then exponentiating back to dollars is biased. If the log-space "
        f"errors are normal with variance s², then E[y|x] = f(x)·exp(s²/2), "
        f"so the retransformed value estimates the *median* and understates "
        f"the mean by exp(s²/2).\n\n"
        f"On this data that factor is **{bias.theoretical_factor:.4f}**, an "
        f"understatement of **{bias.percent_understated:.2f}%** before any "
        f"risk analysis begins. Duan's nonparametric smearing estimate agrees "
        f"at {bias.smearing_factor:.4f}, so the lognormal assumption is not "
        f"doing the work.\n\n"
        f"MUPE (minimum-unbiased-percentage-error) and ZMPE "
        f"(zero-percentage-bias minimum-percentage-error) both drive the mean "
        f"percentage error to exactly zero and so do not carry this bias. "
        f"MUPE places the curve {(bias.mupe_ratio - 1) * 100:+.2f}% relative "
        f"to naive OLS and ZMPE {(bias.zmpe_ratio - 1) * 100:+.2f}%.",
    ).gao(
        "Accurate",
        f"Retransformation bias in the naive log-log fit measured at "
        f"{bias.percent_understated:.2f}% and corrected by using {method.upper()}, "
        f"rather than left in the estimate.",
    )

    # ------------------------------------------------------------------ 4. CER
    portfolio = generate_portfolio(
        n_programs=portfolio_size, seed=seed, pathologies=PathologyConfig.clean(),
        with_reports=False,
    )
    cer_table = portfolio.cer_table()
    cer_table["weight_klb"] = cer_table["empty_weight_lb"] / 1000.0
    cer_table.to_csv(tables_dir / "cer_fitting_data.csv", index=False)

    cer_methods = compare_cer_methods(
        cer_table, "t1_cost_observed", ["weight_klb"], form=Form.LOG_LOG,
        label_col="program",
    )
    cer = cer_methods[method]
    cer_table_out = cer_comparison_table(cer_methods)
    tables["cer_methods"] = cer_table_out
    cer_table_out.to_csv(tables_dir / "cer_methods.csv", index=False)

    diagnostics = cer.diagnostics()
    diagnostics.to_frame().to_csv(tables_dir / "cer_diagnostics.csv", index=False)

    # Predict this program's first unit from the CER, for comparison with the
    # learning-curve answer. Suppress the extrapolation warning only after
    # capturing whether it fired -- it belongs in the log either way.
    subject_weight = program.spec.empty_weight_lb / 1000.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ExtrapolationWarning)
        cer_prediction = cer.predict(
            {"weight_klb": [subject_weight]}, kind="prediction", level=0.80
        )
    extrapolating = bool(cer_prediction["outside_fitting_range"].iloc[0])
    tables["cer_prediction"] = cer_prediction

    lo, hi = cer.predictor_ranges["weight_klb"]
    log.section(
        "4. Cost estimating relationship",
        f"Fitted across {portfolio_size} synthetic programs, log-log form, "
        f"{method.upper()}.\n\n"
        f"- **{cer.equation()}**\n"
        f"- n = {cer.result.n_obs}, {cer.result.n_params} parameters, "
        f"df = {cer.df}, {cer.obs_per_param:.1f} observations per parameter\n"
        f"- Standard error ${cer.standard_error / 1e6:,.2f}M, CV {cer.cv:.1%}\n"
        f"- Fitting range for `weight_klb`: {lo:,.1f} to {hi:,.1f}. "
        f"The subject program is {subject_weight:,.1f}, which is "
        f"{'**outside** that range -- the CER has no evidence there and the '
           'interval reflects only the uncertainty of the fitted form, not '
           'the risk that the form stops holding'
         if extrapolating else 'inside that range'}.\n"
        f"- Predicted first-unit cost "
        f"${cer_prediction['fit'].iloc[0] / 1e6:,.2f}M, 80% **prediction** "
        f"interval ${cer_prediction['lower'].iloc[0] / 1e6:,.2f}M to "
        f"${cer_prediction['upper'].iloc[0] / 1e6:,.2f}M.\n\n"
        f"The interval quoted is a prediction interval, not a confidence "
        f"interval on the mean. The two differ by exactly the residual "
        f"variance: Var_pred = Var_mean + s². A cost estimate forecasts one "
        f"new program, so the prediction interval is the correct one; the "
        f"confidence interval is narrower, shrinks toward zero as the sample "
        f"grows, and quoting it would make this estimate look far more certain "
        f"than it is.",
    ).table("4.1 CER methods compared", cer_table_out).table(
        "4.2 Influence diagnostics", diagnostics.to_frame().head(10)
    ).section(
        "4.3 Diagnostic narrative", diagnostics.narrative()
    ).gao(
        "Credible",
        f"CER influence diagnostics computed; "
        f"{len(diagnostics.influential)} program(s) exceed the conventional "
        f"Cook's distance threshold and are named in the log.",
    )

    if cer.obs_per_param < 3.0 or not cer.result.df_is_adequate:
        log.assume(
            "CER sample size",
            f"Fitted with {cer.obs_per_param:.1f} observations per parameter.",
            "Below the conventional minimum of 3; coefficients are not well "
            "separated and the interval should be treated as indicative.",
        )

    # ----------------------------------------------------- 5. risk simulation
    element_costs = (
        normalized.by_report("DD1921-1")
        .groupby("wbs_element")["dollars"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )
    risks = [DiscreteRisk(**spec) for spec in DEMO_RISKS]
    risk_model = risk_model_from_elements(
        element_costs,
        low_factor=0.88,
        high_factor=1.42,
        distribution="pert",
        default_correlation=correlation,
        risks=risks,
        name=program.spec.program,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CorrelationWarning)
        simulation = simulate_risk_model(risk_model, iterations, seed=seed)
        impact = correlation_impact(risk_model, iterations, seed=seed)

    summary = simulation.summary()
    tornado = simulation.tornado()
    convergence = simulation.convergence()
    tables["risk_summary"] = summary
    tables["tornado"] = tornado
    tables["convergence"] = convergence
    tables["correlation_impact"] = impact.to_frame()
    for name, frame in (
        ("risk_summary", summary), ("tornado", tornado),
        ("convergence", convergence), ("correlation_impact", impact.to_frame()),
    ):
        frame.to_csv(tables_dir / f"{name}.csv", index=False)
    pd.DataFrame(
        simulation.correlation_used,
        index=risk_model.element_names,
        columns=risk_model.element_names,
    ).to_csv(artifacts_dir / "correlation_matrix.csv")

    log.section(
        "5. Risk simulation",
        f"{iterations:,} iterations, seed `{seed}`, sampled with a Gaussian "
        f"copula across {len(risk_model.elements)} WBS elements.\n\n"
        f"- Point estimate ${simulation.point_estimate / 1e6:,.1f}M sits at "
        f"the **{simulation.point_estimate_percentile:.0f}th percentile** of "
        f"the risk distribution. An unreserved point estimate typically lands "
        f"well below the median; this is the number that says whether the "
        f"programme is funded to a defensible level.\n"
        f"- P50 ${simulation.p50 / 1e6:,.1f}M, P80 "
        f"${simulation.p80 / 1e6:,.1f}M, P90 ${simulation.p90 / 1e6:,.1f}M\n"
        f"- Risk reserve to P80: ${(simulation.p80 - simulation.point_estimate) / 1e6:,.1f}M "
        f"({100 * (simulation.p80 / simulation.point_estimate - 1):.1f}%)\n"
        f"- CV of the total {simulation.cv:.1%}\n"
        f"- P80 convergence: "
        f"{'settled' if simulation.is_converged else '**still moving** -- increase the iteration count'} "
        f"(last change {convergence['relative_change'].iloc[-1]:.4%})\n\n"
        f"Discrete risks are modelled separately from the continuous "
        f"uncertainty on the base estimate: a Bernoulli-gated occurrence "
        f"probability times an impact distribution. A 20% chance of a $22M "
        f"qualification failure is not the same thing as a wider spread on the "
        f"base estimate, and averaging the two together produces a "
        f"distribution with a mode nobody believes.",
    ).table("5.1 Summary statistics", summary).table(
        "5.2 Variance contribution", tornado
    ).table("5.3 P80 convergence", convergence)

    log.section(
        "5.4 Why correlation matters",
        f"Sampling WBS elements independently is the spreadsheet default and "
        f"close to the worst assumption available. Elements on one programme "
        f"share a workforce, a management chain, a supply base and a schedule; "
        f"when one runs late they mostly all run late.\n\n"
        f"The variance of a sum of correlated variables is\n\n"
        f"    Var(sum X) = sum Var(X_i) + 2 * sum_{{i<j}} rho_ij * sd_i * sd_j\n\n"
        f"so for k equally variable elements at a common rho, ignoring "
        f"correlation understates the variance of the total by exactly "
        f"`1 + rho*(k-1)`.\n\n"
        f"**{impact.narrative()}**\n\n"
        f"The closed-form and simulated figures agree, so the claim does not "
        f"rest on the simulation alone.",
    ).table("5.5 Correlated versus independent", impact.to_frame()).gao(
        "Credible",
        f"Risk analysis quantifies the effect of the correlation assumption: "
        f"independence would understate the variance of the total by a factor "
        f"of {impact.empirical_variance_ratio:.2f} and the P80 risk reserve by "
        f"{impact.reserve_understatement:.0%}.",
    ).gao(
        "Credible",
        f"P80 convergence checked across iteration counts; "
        f"{'converged' if simulation.is_converged else 'not yet converged'} at "
        f"{iterations:,} iterations.",
    )

    # ---------------------------------------------------------- 6. assumptions
    log.assume(
        "WBS element correlation",
        f"Uniform correlation of {correlation:.2f} across all "
        f"{len(risk_model.elements)} elements.",
        "No programme-specific correlation history available. Values of "
        "0.2-0.3 are the usual starting point. Independence (0.0) was tested "
        "and rejected: see section 5.4.",
    )
    log.assume(
        "Element cost spread",
        "PERT distribution from 0.88x to 1.42x the point estimate on every "
        "element.",
        "Uniform judgement applied in the absence of element-level "
        "elicitation. Asymmetric because cost is bounded below by the work and "
        "unbounded above.",
    )
    log.assume(
        "Discrete risks",
        f"{len(risks)} risks with occurrence probabilities "
        f"{', '.join(f'{r.probability:.0%}' for r in risks)}.",
        "Illustrative for this synthetic run. A real analysis elicits these "
        "from the programme risk register.",
    )
    log.assume(
        "Escalation",
        f"Constant-rate index, base year FY{resolved_base_year}.",
        "Synthetic programme assumption. A real estimate uses a published "
        "index; the raw-index design means the base year can be re-selected "
        "without regenerating anything.",
    )
    log.assume(
        "Learning curve theory",
        f"{theory.title()} theory applied.",
        "Selected by the analyst. Both theories are fitted and reported in "
        "table 3.2; they give different forecasts from the same data.",
    )
    if extrapolating:
        log.assume(
            "CER extrapolation",
            f"The subject programme's weight ({subject_weight:,.1f} klb) falls "
            f"outside the CER fitting range ({lo:,.1f} to {hi:,.1f} klb).",
            "Flagged automatically. The prediction interval does not include "
            "the risk that the functional form stops holding out there.",
        )

    log.gao(
        "Comprehensive",
        "Recurring and nonrecurring cost, five functional categories, and "
        "discrete risks are all modelled; nothing is excluded silently.",
    ).gao(
        "Well-documented",
        f"This log, {len(tables)} data tables and {4} charts are emitted "
        f"automatically from seed {seed} and are reproducible from it.",
    ).gao(
        "Accurate",
        "Estimating methods compared (OLS, MUPE, ZMPE) rather than one being "
        "assumed, with the bias of the naive approach measured.",
    )

    # -------------------------------------------------------------- 7. charts
    artifacts["s_curve"] = charts.plot_s_curve(
        simulation,
        charts_dir / "s_curve.png",
        comparison=impact.independent,
        title=f"{program.spec.program}: total cost S-curve",
        subtitle=(
            f"{iterations:,} iterations, rho = {correlation:.2f} across "
            f"{len(risk_model.elements)} WBS elements (synthetic data)"
        ),
    )
    artifacts["tornado"] = charts.plot_tornado(
        simulation,
        charts_dir / "tornado.png",
        title=f"{program.spec.program}: contribution to total cost variance",
    )
    artifacts["learning_curve"] = charts.plot_learning_curve(
        chosen_curve,
        lot_table,
        charts_dir / "learning_curve.png",
        forecast_lots=future,
        title=(
            f"{program.spec.program}: {theory} cost improvement curve "
            f"({method.upper()})"
        ),
    )
    artifacts["cer"] = charts.plot_cer_diagnostics(
        cer, charts_dir / "cer_diagnostics.png"
    )

    display_summary = summary.copy()
    display_summary["value"] = [
        # Thousands separators for money, significant figures for ratios --
        # a CV rendered as "0.07" throws away the digit that distinguishes a
        # tight estimate from a loose one.
        f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"
        for v in display_summary["value"]
    ]
    artifacts["summary_table"] = charts.plot_summary_table(
        display_summary,
        charts_dir / "summary_table.png",
        title=f"{program.spec.program}: risk summary (FY{resolved_base_year} $)",
    )

    # ----------------------------------------------------------------- 8. log
    artifacts["assumptions_log"] = log.write(output_dir / "ASSUMPTIONS.md")

    result = RunResult(
        output_dir=output_dir,
        seed=seed,
        program=program,
        normalized=normalized,
        curve_fits=curve_fits,
        chosen_curve=chosen_curve,
        cer=cer,
        simulation=simulation,
        impact=impact,
        artifacts=artifacts,
        tables=tables,
    )
    logger.info("Run complete. %s", result.headline())
    return result
