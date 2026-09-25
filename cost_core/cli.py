# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Cost-Estimation Core CLI
Bridge between the math engine and the terminal.

Installed as the ``ce-core`` command, and ``python -m cost_core.cli`` runs the
same thing. It lives inside the package rather than as a top-level ``cli``
module, because a wheel that installs a package called ``cli`` shares that
name with every other distribution that does.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NoReturn, Optional

import pandas as pd

from cost_core import data_io, learning_curve, monte_carlo

# Setup minimalist logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def abort(message: str) -> NoReturn:
    """Standardized exit for fatal errors."""
    log.error(message)
    sys.exit(1)

def write_brief(make, *args) -> Optional[str]:
    """Write the PowerPoint briefing, or say how to get one if python-pptx
    (the plots extra) is not installed. Returns the file name written."""
    try:
        make(*args)
    except ImportError:
        print('  (For a PowerPoint briefing as well: pip install "cost-core[plots]")')
        return None
    return "brief.pptx"


def need_file(path, what: str, topic: str) -> None:
    """Stop with a message a first-time user can act on when an input file is
    not there: where it looked, and how to get one."""
    p = Path(path)
    if p.exists():
        return
    here = Path.cwd()
    hint = ("How to save one from Microsoft Project: ce-core template schedule"
            if topic == "schedule" else f"For a file to fill in: ce-core template {topic}")
    abort(f"Can't find {what} '{path}' (looked in {here if not p.is_absolute() else p.parent}).\n"
          f"  To see this command work first: ce-core demo {topic}\n"
          f"  {hint}")


def run_fit(args: argparse.Namespace) -> None:
    path = Path(args.csv)
    if not path.is_file():
        abort(f"CSV not found: {path}")

    try:
        log.info(f"Processing {path}")
        df = data_io.load_cost_csv(path)
        
        model = learning_curve.fit_learning_curve(df, args.qty_col, args.cost_col)
        
        params = {
            "slope": model.slope,
            "reference_quantity": model.reference_quantity,
            "reference_cost": model.reference_cost
        }
        
        Path(args.out).write_text(json.dumps(params, indent=4))
        log.info(f"Model serialized to {args.out}")
        
    except Exception as e:
        abort(f"Fitting pipeline failed: {e}")

def run_forecast(args: argparse.Namespace) -> None:
    model_file = Path(args.model)
    if not model_file.is_file():
        abort(f"Model file missing: {model_file}")

    try:
        # Robust parsing for '10, 20, 50' or '10,20,50'
        qtys = [float(x) for x in args.qtys.replace(" ", "").split(",") if x]

        data = json.loads(model_file.read_text())
        model = learning_curve.LearningCurveModel(**data)
        
        log.info(f"Forecasting {len(qtys)} points")
        results = learning_curve.forecast_costs(model, qtys)
        
        results.to_csv(args.out, index=False)
        log.info(f"Results written to {args.out}")
        
    except (ValueError, json.JSONDecodeError) as e:
        abort(f"Input validation error: {e}")
    except Exception as e:
        abort(f"Forecast execution failed: {e}")

def run_simulate(args: argparse.Namespace) -> None:
    try:
        log.info(f"Starting MC simulation ({args.iters} iterations)")
        res = monte_carlo.run_monte_carlo(
            n_iter=args.iters,
            unit_cost_dist=json.loads(args.cost_dist),
            quantity_dist=json.loads(args.qty_dist)
        )
        
        print(f"\n--- Simulation Results ---\nMean: ${res.mean:,.2f}\nP80:  ${res.p80:,.2f}\n")
        
        pd.DataFrame({"total_cost": res.samples}).to_csv(args.out, index=False)
        log.info(f"Samples saved to {args.out}")
        
    except json.JSONDecodeError:
        abort("Invalid JSON passed to distribution arguments.")
    except Exception as e:
        abort(f"Simulation engine error: {e}")

def run_fit_lots(args: argparse.Namespace) -> None:
    """Fit the lot cost model to a two-column lot file: units and cost."""
    path = Path(args.csv)
    if not path.is_file():
        abort(f"Lot file not found: {path}")

    try:
        from cost_core.lots import LotSeries, analyse_lots, build_assumption_log

        series = LotSeries.read(
            path,
            dollar_year=args.dollar_year,
            cost_basis=args.cost_basis,
            first_unit=args.first_unit,
            quantity_definition=args.quantity_definition,
            program=args.program or path.stem,
            units_col=args.units_col,
            cost_col=args.cost_col,
        )

        forecast = None
        if args.forecast:
            forecast = [int(x) for x in args.forecast.replace(" ", "").split(",") if x]

        report = analyse_lots(
            series, forecast=forecast, complexity=args.complexity,
            level=args.level, t_gate=args.t_gate, aicc_tie=args.aicc_tie,
            legacy_rate_omission=args.legacy_rate_omission,
        )
        if args.legacy_rate_omission:
            print()
            print("[WARN] --legacy-rate-omission is on. Rate and LC+Rate lots "
                  "are priced without the rate regressor, so the projected "
                  "costs will not satisfy the equation printed below and will "
                  "read high. Only for reproducing a legacy workbook.")

        print()
        print(f"--- {series.program} ---")
        # The engine is scale free: it fits whatever cost numbers it is given
        # and hands them back in the same units. Its column names say ($K)
        # because they come from the desktop tool, whose input column is
        # "AUC ($K)", and the charts format the same numbers as dollars. Two
        # conventions over one set of numbers is a 1,000x misread waiting to
        # happen, so say which one this run is in: the file's.
        print(f"Costs are in the units {path.name} states them in. Nothing here "
              f"converts them, so a heading that reads ($K) is the engine's "
              f"column name rather than a claim about scale, and the S-curve "
              f"formats the same numbers as dollars.")
        print(report.narrative())

        print()
        print("Models compared (the engine fits all three):")
        print(report.model_comparison().to_string(index=False))

        print()
        print(f"Selected: {report.selected_model}")
        print(f"  {report.equation()}")
        print()
        print("Coefficients:")
        print(report.fit.equation_detail().to_string(index=False))

        print()
        print("Analogy lots as fitted:")
        print(report.per_lot[[
            "lot", "units", "lot_midpoint", "lot_average_cost",
            "fitted_unit_cost", "percent_error",
        ]].to_string(index=False))

        print()
        print("Retransformation bias, measured on this data:")
        methods = report.methods()
        print(f"  OLS understates the mean by {methods.percent_understated:.3f}%"
              f"   exp(s2/2) = {methods.theoretical_factor:.5f}"
              f"   Duan smearing = {methods.smearing_factor:.5f}")
        print(methods.frame.to_string(index=False))

        print()
        print("Influence on the analogy lots:")
        print(report.influence().to_string(index=False))

        label = "Forecast" if forecast else "Back-cast of the fitted lots"
        print()
        print(f"{label} ({args.level:.0%} prediction interval):")
        print(report.intervals().to_string(index=False))

        simulation = None
        if args.simulate:
            simulation = report.simulate(n_iter=args.simulate, seed=args.seed)
            print()
            print(f"Monte Carlo of the buy ({args.simulate:,} iterations):")
            print(simulation.narrative())
            print()
            print(simulation.summary().to_string(index=False))

        priced = None
        if args.price_lots:
            plan = [int(x) for x in args.price_lots.replace(" ", "").split(",") if x]
            priced = report.price_lot_plan(plan, first_unit=args.price_from_unit)
            print()
            print(f"This model applied to a lot plan of {plan}, from unit "
                  f"{args.price_from_unit} (analogy for a programme with no "
                  f"history of its own):")
            print(priced.to_string(index=False))

        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            report.per_lot.to_csv(out / "lots_fitted.csv", index=False)
            report.summary().to_csv(out / "summary.csv", index=False)
            report.model_comparison().to_csv(out / "models_compared.csv", index=False)
            report.fit.equation_detail().to_csv(out / "equation.csv", index=False)
            report.methods().frame.to_csv(out / "fit_methods.csv", index=False)
            report.influence().to_csv(out / "influence.csv", index=False)
            report.intervals().to_csv(out / "prediction_intervals.csv", index=False)
            report.fit.projections.to_csv(out / "projections.csv", index=False)
            if priced is not None:
                priced.to_csv(out / "lot_plan_priced.csv", index=False)
            if simulation is not None:
                simulation.summary().to_csv(out / "buy_risk.csv", index=False)
            build_assumption_log(
                report, source=path, priced_plan=priced,
                priced_from_unit=args.price_from_unit,
            ).write(out / "ASSUMPTIONS.md")

            if simulation is not None:
                # Inside the branch, not above it. The only chart this command
                # writes is the buy S-curve, and that needs `--simulate`.
                # matplotlib is an optional extra, so importing charts
                # unconditionally would have made `--out` fail on an install
                # without it even when nothing was going to be drawn.
                from cost_core.reporting import charts

                charts.plot_s_curve(
                    simulation, out / "buy_s_curve.png",
                    title=f"{series.program}: cost of the priced lots",
                    subtitle=(f"{args.simulate:,} iterations from the "
                              f"{report.selected_model} fit, constant "
                              f"FY{args.dollar_year} dollars"),
                )
            log.info(f"Wrote results and ASSUMPTIONS.md to {out}")

    except Exception as e:
        abort(f"Lot cost model failed: {e}")


def run_full(args: argparse.Namespace) -> None:
    """Generate, ingest, fit, simulate and report, in one command."""
    try:
        # Imported here rather than at module scope so that `fit-curve` and
        # `simulate` do not pay for matplotlib and the reporting stack.
        from cost_core.reporting import run_full_analysis

        log.info("Running the full path into %s (seed %d)", args.out, args.seed)
        result = run_full_analysis(
            args.out,
            seed=args.seed,
            iterations=args.iters,
            theory=args.theory,
            method=args.method,
            correlation=args.correlation,
            base_year=args.base_year,
            portfolio_size=args.programs,
            clean=args.clean,
        )

        print("\n--- Cost estimate ---")
        print(result.headline())
        print("\nArtifacts:")
        for name, path in result.artifacts.items():
            print(f"  {name:18s} {path}")
        print(f"\nAssumptions log: {result.artifacts['assumptions_log']}\n")

    except Exception as e:
        abort(f"Full run failed: {e}")


def run_aoa(args) -> None:
    """Evaluate an AoA spec file and write the comparison."""
    import json
    from pathlib import Path

    from cost_core.aoa import AoAError
    from cost_core.aoa.spec import run_spec

    need_file(args.spec, "the AoA spec", "aoa")
    try:
        result = run_spec(args.spec)
    except (AoAError, OSError, KeyError, ValueError) as e:
        abort(f"AoA failed: {e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(out / "summary.csv", index=False)
    result.lines.to_csv(out / "lines.csv", index=False)
    result.s_curves().to_csv(out / "s_curves.csv")
    (out / "assumptions.json").write_text(json.dumps(result.assumptions, indent=1, default=str),
                                          encoding="utf-8")
    try:
        from cost_core.reporting.charts import plot_aoa_s_curves
        plot_aoa_s_curves(result, out / "aoa_s_curves.png")
        chart = "aoa_s_curves.png"
    except ImportError:
        chart = None
    from cost_core.reporting.excel_report import aoa_workbook
    aoa_workbook(result, out / "report.xlsx")
    basis = result.assumptions["basis"].upper()
    print(f"\nLife-cycle cost, {basis}, {result.assumptions['n_iter']:,} draws each:\n")
    cols = [c for c in ("alternative", "by", "ty", "pv", "p50", "p80", "p_cheapest",
                        "effectiveness", "cost_per_effectiveness", "dominated_by")
            if c in result.summary]
    print(result.summary[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    from cost_core import plain
    print(plain.show(plain.aoa(result, str(result.assumptions.get("units") or ""))))
    from cost_core.reporting.brief import aoa_brief
    brief = write_brief(aoa_brief, result, out)
    print(f"Wrote report.xlsx, {brief + ', ' if brief else ''}summary.csv, lines.csv, "
          f"s_curves.csv, assumptions.json"
          f"{', ' + chart if chart else ''} to {out}")


def run_portfolio(args) -> None:
    """Solve a portfolio spec and write the choice and its analyses."""
    from pathlib import Path

    try:
        from cost_core.portfolio import (PortfolioError, budget_risk, frontier,
                                         marginal_value, solve)
        from cost_core.portfolio.spec import load_portfolio
    except ImportError as e:
        abort(str(e))
    need_file(args.spec, "the portfolio spec", "portfolio")
    try:
        portfolio, settings = load_portfolio(args.spec)
        result = solve(portfolio)
    except ImportError:
        abort("Choosing a portfolio needs a solver, which comes with the optimize extra:\n"
              '  pip install "cost-core[optimize]"')
    except (PortfolioError, OSError, KeyError, ValueError) as e:
        abort(f"Portfolio failed: {e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    units = settings.get("units", "as entered")
    money = lambda v: f"{v:,.2f}"  # noqa: E731
    result.selected.to_csv(out / "selected.csv", index=False)
    result.spend.to_csv(out / "spend.csv", index=False)
    written = ["selected.csv", "spend.csv"]
    print(f"\nValue {result.value:,.2f}, funding {len(result.funded)} of "
          f"{len(portfolio.candidates)} candidates. Costs in {units}.\n")
    print(result.selected.to_string(index=False, float_format=money))
    print("\nSpend by year:")
    print(result.spend.to_string(index=False, float_format=money))
    delta = settings.get("delta")
    if delta:
        mv = marginal_value(portfolio, float(delta))
        mv.to_csv(out / "marginal_value.csv", index=False)
        written.append("marginal_value.csv")
        print(f"\nWhat {float(delta):,.0f} more in one year would buy:")
        print(mv.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    fr = frontier(portfolio, settings.get("frontier_scales", [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    fr.to_csv(out / "frontier.csv", index=False)
    written.append("frontier.csv")
    print("\nBest value at other budget levels:")
    print(fr.to_string(index=False, float_format=money))
    risk = None
    if "growth" in settings:
        risk = budget_risk(portfolio, result.choice, settings["growth"],
                           n_iter=int(settings.get("n_iter", 20000)),
                           seed=settings.get("seed", 0),
                           correlation=float(settings.get("growth_correlation", 0.0)))
        risk.to_csv(out / "budget_risk.csv", index=False)
        written.append("budget_risk.csv")
        print("\nChance each year breaks its budget once costs grow:")
        print(risk.to_string(index=False, float_format=money))
    from cost_core import plain
    print(plain.show(plain.portfolio(result, len(portfolio.candidates), risk, units)))
    from cost_core.reporting.excel_report import portfolio_workbook
    extra = {"Value against budget": fr}
    if delta:
        extra["Marginal value"] = mv
    portfolio_workbook(result, portfolio, out / "report.xlsx", units, extra, risk)
    written.insert(0, "report.xlsx")
    from cost_core.reporting.brief import portfolio_brief
    written[1:1] = [w for w in [write_brief(portfolio_brief, result, portfolio, out, units, risk)]
                    if w]
    print(f"Wrote {', '.join(written)} to {out}")


def run_jcl(args) -> None:
    """Simulate a JCL spec and write the joint confidence tables and chart."""
    import json
    from pathlib import Path

    from cost_core.schedule import ScheduleError, critical_path, simulate
    from cost_core.schedule.spec import load_project

    need_file(args.spec, "the JCL spec", "jcl")
    try:
        project, settings = load_project(args.spec)
        result = simulate(project, n_iter=int(settings.get("n_iter", 20000)),
                          seed=settings.get("seed", 0))
    except (ScheduleError, OSError, KeyError, ValueError) as e:
        abort(f"JCL failed: {e}")
    conf = float(args.confidence or settings.get("confidence", 0.7))
    units = settings.get("units", "as entered")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = result.summary(conf)
    summary.to_csv(out / "summary.csv", index=False)
    critical_path(project).to_csv(out / "critical_path.csv", index=False)
    result.criticality().to_csv(out / "criticality.csv", index=False)
    result.frontier(conf, points=40).to_csv(out / "frontier.csv", index=False)
    import pandas as pd
    pd.DataFrame({"finish_months": result.finish, "cost": result.cost}).to_csv(
        out / "draws.csv", index=False)
    (out / "assumptions.json").write_text(json.dumps({
        "project": project.name, "units": units, "confidence": conf,
        "n_iter": result.n_iter, "seed": result.seed,
        "standing_army_per_month": project.standing_army,
        "duration_correlation": project.duration_correlation,
        "cost_correlation": project.cost_correlation,
        "cross_correlation": project.cross_correlation,
        "risks": [r.name for r in project.risks], "notes": result.notes,
        **({"schedule_import_notes": settings["notes"]} if settings.get("notes") else {}),
    }, indent=1), encoding="utf-8")
    written = ["summary.csv", "critical_path.csv", "criticality.csv", "frontier.csv",
               "draws.csv", "assumptions.json"]
    from cost_core.reporting.excel_report import jcl_workbook
    jcl_workbook(result, project, conf, out / "report.xlsx", units,
                 critical_path=critical_path(project))
    written.insert(0, "report.xlsx")
    try:
        from cost_core.reporting.charts import plot_jcl
        plot_jcl(result, out / "jcl.png", confidence=conf, units=units)
        written.append("jcl.png")
    except ImportError:
        pass
    print(f"\n{project.name}: {result.n_iter:,} simulated projects, costs in {units}.\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\nWhere the schedule risk is:")
    print(result.criticality().to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    from cost_core import plain
    print(plain.show(plain.jcl(result, conf, units)))
    from cost_core.reporting.brief import jcl_brief
    written[1:1] = [w for w in [write_brief(jcl_brief, result, project, conf, out, units)] if w]
    print(f"Wrote {', '.join(written)} to {out}")


def run_evm(args) -> None:
    """EVM metrics, warning signs and the forecast at completion."""
    import json
    from pathlib import Path

    import pandas as pd

    from cost_core.evm import EvmData, EvmError, forecast

    from cost_core.plain import units_label
    args.units = units_label(args.units)
    try:
        if args.ipmdar:
            from cost_core.evm.ipmdar import read_ipmdar
            for p in args.ipmdar:
                need_file(p, "the IPMDAR dataset", "evm")
            data = read_ipmdar(args.ipmdar, bac=args.bac)
        elif args.data:
            need_file(args.data, "the EVM data", "evm")
            data = EvmData.read(args.data, cumulative=args.cumulative, bac=args.bac)
        else:
            abort("Which data? Give --data my_evm.xlsx (a spreadsheet or CSV) or --ipmdar "
                  "delivery.zip.\n  To see it work first: ce-core demo evm\n"
                  "  For a spreadsheet to fill in: ce-core template evm")
        fc = forecast(data, n_iter=args.iters, seed=args.seed)
    except (EvmError, OSError, KeyError, ValueError) as e:
        abort(f"EVM failed: {e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data.metrics().to_csv(out / "metrics.csv", index=False)
    data.flags().to_csv(out / "flags.csv", index=False)
    summary = pd.concat([data.summary(), fc.summary()], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    fc.percentiles().to_csv(out / "forecast_percentiles.csv", index=False)
    pd.DataFrame({"eac": fc.eac, "finish_periods": fc.finish}).to_csv(
        out / "forecast_draws.csv", index=False)
    if data.accounts:
        rows = []
        for name, acct in data.accounts.items():
            r = acct.metrics().iloc[-1]
            rows.append({"account": name, "bac": acct.bac, **{k: r[k] for k in (
                "bcws", "bcwp", "acwp", "cv", "sv", "cpi", "spi", "spi_t", "tcpi_bac",
                "ieac_cpi", "eac")}})
        pd.DataFrame(rows).to_csv(out / "accounts.csv", index=False)
    (out / "assumptions.json").write_text(json.dumps({
        "data": [str(p) for p in args.ipmdar] if args.ipmdar else str(args.data),
        "cumulative_input": args.cumulative, "bac": data.bac, "import_notes": data.notes,
        "status_period": str(data.periods[data.status - 1]), "n_iter": fc.n_iter,
        "seed": fc.seed, "notes": fc.notes}, indent=1), encoding="utf-8")
    written = ["metrics.csv", "flags.csv", "summary.csv", "forecast_percentiles.csv",
               "forecast_draws.csv", "assumptions.json"] + (["accounts.csv"] if data.accounts else [])
    from cost_core.reporting.excel_report import evm_workbook
    evm_workbook(data, fc, out / "report.xlsx", args.units)
    written.insert(0, "report.xlsx")
    try:
        from cost_core.reporting.charts import plot_evm
        plot_evm(data, fc, out / "evm.png", units=args.units)
        written.append("evm.png")
    except ImportError:
        pass
    def fmt(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return "" if v != v else f"{v:,.3f}"
        return str(v)

    print(f"\n{data.name}: status period {data.periods[data.status - 1]}, "
          f"costs in {args.units}.\n")
    shown = summary.assign(value=summary["value"].map(fmt))
    print(shown.to_string(index=False, justify="left"))
    raised = data.flags()
    raised = raised[raised["raised"]]
    if len(raised):
        print("\nWarning signs:")
        for r in raised.itertuples():
            print(f"  - {r.flag}. {r.detail}")
    if data.notes:
        title = ("How the IPMDAR was read (a preview reader: check these against the "
                 "delivery's own totals):" if args.ipmdar else "How the data was read:")
        print(f"\n{title}")
        for note in data.notes:
            print(f"  - {note}")
    from cost_core import plain
    print(plain.show(plain.evm(data, fc, args.units)))
    from cost_core.reporting.brief import evm_brief
    written[1:1] = [w for w in [write_brief(evm_brief, data, fc, out, args.units)] if w]
    print(f"Wrote {', '.join(written)} to {out}")


def run_schedule_check(args) -> None:
    """Read a Microsoft Project XML schedule and run the DCMA 14-point check."""
    from pathlib import Path

    import pandas as pd

    from cost_core.schedule import ScheduleError
    from cost_core.schedule.dcma import dcma_14_point
    from cost_core.schedule.mspdi import read_mspdi

    need_file(args.mspdi, "the schedule", "schedule")
    try:
        sched = read_mspdi(args.mspdi)
        result = dcma_14_point(sched)
    except (ScheduleError, OSError, ValueError) as e:
        abort(f"Schedule check failed: {e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "dcma.csv", index=False)
    pd.DataFrame([{"check": k, "task": n} for k, v in result.tasks.items() for n in v],
                 columns=["check", "task"]).to_csv(out / "dcma_tasks.csv", index=False)
    sched.tasks.to_csv(out / "tasks.csv", index=False)
    sched.links.to_csv(out / "links.csv", index=False)
    from cost_core.reporting.excel_report import dcma_workbook
    dcma_workbook(result, sched, out / "report.xlsx")
    shown = result.table.assign(
        result=result.table["passed"].map({True: "pass", False: "FAIL"}).fillna("n/a"),
        value=result.table["value"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
        count=result.table["count"].map(lambda v: "" if pd.isna(v) else f"{int(v)}"))
    print(f"\n{sched.name}: {len(sched.detail)} detail tasks, {len(sched.links)} links.\n")
    print(shown[["check", "name", "count", "value", "threshold", "result"]]
          .to_string(index=False))
    print(f"\n{result.passed} passed, {result.failed} failed, "
          f"{14 - result.passed - result.failed} not assessable from this file.")
    for note in sched.notes:
        print(f"  note: {note}")
    from cost_core import plain
    print(plain.show(plain.dcma(result, sched)))
    from cost_core.reporting.brief import dcma_brief
    brief = write_brief(dcma_brief, result, sched, out)
    print(f"Wrote report.xlsx, {brief + ', ' if brief else ''}dcma.csv, dcma_tasks.csv, "
          f"tasks.csv and links.csv to {out}")


#: What each demo runs, as the command line a user would type, with the
#: example's path and the demo's output folder filled in.
DEMOS = {
    "evm": ["evm", "--data", "{path}", "--units", "thousands", "--out", "{out}"],
    "schedule": ["schedule-check", "--mspdi", "{path}", "--out", "{out}"],
    "jcl": ["jcl", "--spec", "{path}", "--out", "{out}"],
    "aoa": ["aoa", "--spec", "{path}", "--out", "{out}"],
    "portfolio": ["portfolio", "--spec", "{path}", "--out", "{out}"],
}


def run_demo(args) -> None:
    """Run one command end to end on its bundled example."""
    from pathlib import Path

    from cost_core.examples import EXAMPLES, example_path

    topic = args.topic
    path = example_path(topic)
    out = Path(args.out or Path("ce-core-demo") / topic)
    argv = [a.format(path=path, out=out) for a in DEMOS[topic]]
    print(f"Demo: {EXAMPLES[topic][1]}\nExample data: {path}\n"
          f"Running: ce-core {' '.join(_quote(a) for a in argv)}")
    main(argv)
    mine = {"evm": "ce-core evm --data my_evm.xlsx",
            "schedule": "ce-core schedule-check --mspdi my_schedule.xml",
            "jcl": "ce-core jcl --spec my_jcl.json",
            "aoa": "ce-core aoa --spec my_aoa.json",
            "portfolio": "ce-core portfolio --spec my_portfolio.json"}[topic]
    first = ("ce-core template schedule   (how to save one from Microsoft Project)"
             if topic == "schedule" else f"ce-core template {topic}")
    print(f"Everything above is in {out}.\n\nNow with your own data:\n  {first}\n  {mine}")


def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or "$" in arg) else arg


TEMPLATE_FILES = {"evm": "my_evm.xlsx", "jcl": "my_jcl.json", "aoa": "my_aoa.json",
                  "portfolio": "my_portfolio.json", "lots": "my_lots.csv"}


def run_template(args) -> None:
    """Write a file to fill in, laid out the way the command reads it."""
    from pathlib import Path

    from cost_core import templates

    topic = args.topic
    if topic == "schedule":
        print(templates.SCHEDULE_HOWTO)
        return
    out = Path(args.out or TEMPLATE_FILES[topic])
    if out.exists() and not args.force:
        abort(f"{out} already exists; choose another name with --out, or add --force "
              "to replace it.")
    written = templates.write(topic, out)
    print(f"Wrote {written}.\n")
    print(templates.NEXT_STEPS[topic].format(path=written))


def run_sar_panel(args) -> None:
    """Build a unit cost panel from public SARs and write it as CSV."""
    try:
        from cost_core.public import build_sar_panel, local_catalog, sar_catalog
    except ImportError as e:
        abort(str(e))
    from cost_core.public import FetchError
    try:
        cat = local_catalog(args.dir) if args.dir else sar_catalog()
    except FileNotFoundError as e:
        abort(str(e))
    except FetchError:
        abort("Couldn't reach the Internet Archive, where the SARs are listed and fetched "
              "from.\n  No internet on this machine, or a proxy in the way? Copy the SAR "
              "PDFs into a folder and run:\n    ce-core sar-panel --dir that_folder")
    if args.list_cycles:
        counts = cat[cat["kind"] != "combined"].groupby("cycle", sort=False, dropna=False).size()
        for cycle, n in counts.items():
            print(f"  {cycle if isinstance(cycle, str) else 'no cycle':10s} {n:4d} reports")
        return
    panel = build_sar_panel(catalog=cat, cycles=args.cycles, programs=args.programs,
                            limit=args.limit, progress=print)
    paths = panel.to_csv(args.out)
    read = int((panel.reports["status"] == "read").sum())
    print(f"\nRead {read} of {len(panel.reports)} reports, "
          f"{len(panel.unit_cost)} unit cost rows.")
    for name, path in paths.items():
        print(f"  {name:10s} {path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="ce-core",
        description="Cost estimating, EVM and schedule analysis. Run with no arguments for "
                    "a guide to what it can do.")
    sub = parser.add_subparsers(dest="cmd")

    # Subcommand: fit
    # Long-form aliases (--quantity-col, --quantities, --n-iter,
    # --unit-cost-dist, --quantity-dist) are accepted alongside the short
    # names. The README documented the long forms while the parser only
    # accepted the short ones, so every example in it failed; adding aliases
    # fixes the documented interface without breaking the existing one.
    p_fit = sub.add_parser("fit-curve", help="Analyze historical cost trends")
    p_fit.add_argument("--csv", required=True)
    p_fit.add_argument("--out", required=True)
    p_fit.add_argument("--qty-col", "--quantity-col", dest="qty_col",
                       default="unit_quantity")
    p_fit.add_argument("--cost-col", dest="cost_col", default="unit_cost")

    # Subcommand: forecast
    p_fcst = sub.add_parser("forecast", help="Predict costs for target lots")
    p_fcst.add_argument("--model", required=True)
    p_fcst.add_argument("--out", required=True)
    p_fcst.add_argument("--qtys", "--quantities", dest="qtys", required=True,
                        help="Lots e.g. '50,100,500'")

    # Subcommand: simulate
    p_sim = sub.add_parser("simulate", help="Run probabilistic risk models")
    p_sim.add_argument("--out", required=True)
    p_sim.add_argument("--iters", "--n-iter", dest="iters", type=int,
                       default=10000)
    p_sim.add_argument("--cost-dist", "--unit-cost-dist", dest="cost_dist",
                       required=True)
    p_sim.add_argument("--qty-dist", "--quantity-dist", dest="qty_dist",
                       required=True)

    # Subcommand: fit-lots -- the simple front door for real production data
    p_lots = sub.add_parser(
        "fit-lots",
        help="Fit a curve to a two-column lot file (units, cost)",
    )
    p_lots.add_argument("--csv", required=True,
                        help="CSV or XLSX with a units column and a cost column, "
                             "in whatever cost units it states; nothing is converted")
    p_lots.add_argument("--dollar-year", type=int, required=True,
                        help="Fiscal year the constant dollars are stated in")
    p_lots.add_argument("--out", default=None,
                        help="Directory for results, chart and ASSUMPTIONS.md")
    p_lots.add_argument("--cost-basis", default="recurring",
                        choices=["recurring", "total"],
                        help="'total' includes nonrecurring and warns")
    p_lots.add_argument("--first-unit", type=int, default=1,
                        help="Unit number the first lot starts at")
    p_lots.add_argument("--complexity", type=float, default=1.0,
                        help="Complexity factor applied to the priced lots")
    p_lots.add_argument("--t-gate", type=float, default=2.0,
                        help="Significance cutoff on the rate coefficient")
    p_lots.add_argument("--aicc-tie", type=float, default=2.0,
                        help="How much better on AICc Rate must be to beat LC")
    p_lots.add_argument("--quantity-definition", default="unspecified",
                        help="What a unit means: delivered, completed, accepted")
    p_lots.add_argument("--program", default=None, help="Program name for reports")
    p_lots.add_argument("--units-col", default=None,
                        help="Name the quantity column if it is not recognised")
    p_lots.add_argument("--cost-col", default=None,
                        help="Name the cost column if it is not recognised")
    p_lots.add_argument("--forecast", default=None,
                        help="Future lot sizes, e.g. '30,40'")
    p_lots.add_argument("--level", type=float, default=0.80,
                        help="Prediction interval coverage")
    p_lots.add_argument("--simulate", type=int, default=0, metavar="N",
                        help="Monte Carlo the forecast buy over N iterations "
                             "(needs --forecast); writes an S-curve")
    p_lots.add_argument("--seed", type=int, default=0,
                        help="Seed for --simulate, so the P80 is reproducible")
    p_lots.add_argument("--price-lots", default=None, metavar="Q1,Q2,...",
                        help="Apply the fitted curve to this lot plan from "
                             "unit 1, with lot midpoints. For pricing an "
                             "analogous program that has no history of its own")
    p_lots.add_argument("--price-from-unit", type=int, default=1,
                        help="Unit the priced plan starts at (default 1)")
    p_lots.add_argument("--legacy-rate-omission", action="store_true",
                        help="Reproduce the original desktop tool, which "
                             "priced Rate and LC+Rate lots without the rate "
                             "regressor. Its projections do not satisfy the "
                             "equation it prints and read high. Only for "
                             "reproducing a workbook built by it")

    # Subcommand: full-run
    p_run = sub.add_parser(
        "full-run",
        help="End to end: synthesise CSDR/SRDR, ingest, fit, simulate, report",
    )
    p_run.add_argument("--out", required=True,
                       help="Output directory for charts, tables and the log")
    p_run.add_argument("--seed", type=int, default=7,
                       help="Master seed; the same seed reproduces the run")
    p_run.add_argument("--iters", "--n-iter", dest="iters", type=int,
                       default=50000, help="Monte Carlo iterations")
    p_run.add_argument("--theory", default="crawford",
                       choices=["crawford", "wright"],
                       help="Learning curve theory")
    p_run.add_argument("--method", default="mupe",
                       choices=["ols", "mupe", "zmpe"],
                       help="Fitting method for the curve and the CER")
    p_run.add_argument("--correlation", type=float, default=0.30,
                       help="Uniform correlation across WBS elements")
    p_run.add_argument("--base-year", type=int, default=None,
                       help="Fiscal year to state dollars in")
    p_run.add_argument("--programs", type=int, default=14,
                       help="Programs generated for the CER fit")
    p_run.add_argument("--clean", action="store_true",
                       help="Generate data with no reporting pathologies")

    # Subcommand: aoa
    p_aoa = sub.add_parser(
        "aoa",
        help="Life-cycle cost of alternatives, compared under uncertainty",
    )
    p_aoa.add_argument("--spec", required=True,
                       help="JSON spec of the alternatives (ce-core template aoa writes one)")
    p_aoa.add_argument("--out", default="aoa",
                       help="Directory for the tables, assumptions and chart")

    # Subcommand: portfolio
    p_port = sub.add_parser(
        "portfolio",
        help="Which programs to fund within each year's budget (needs [optimize])",
    )
    p_port.add_argument("--spec", required=True,
                        help="JSON spec of the candidates and budget "
                             "(ce-core template portfolio writes one)")
    p_port.add_argument("--out", default="portfolio",
                        help="Directory for the choice, spend, frontier and risk tables")

    # Subcommand: jcl
    p_jcl = sub.add_parser(
        "jcl",
        help="Schedule risk and joint cost and schedule confidence (JCL)",
    )
    p_jcl.add_argument("--spec", required=True,
                       help="JSON spec of the network (ce-core template jcl writes one), or one "
                            "naming a Microsoft Project XML file under \"mspdi\"")
    p_jcl.add_argument("--out", default="jcl",
                       help="Directory for the tables, the draws and the JCL chart")
    p_jcl.add_argument("--confidence", type=float, default=None,
                       help="Joint confidence to report against (default: the spec's, else 0.7)")

    # Subcommand: evm
    p_evm = sub.add_parser(
        "evm",
        help="Earned value metrics, warning signs and the forecast at completion",
    )
    p_evm.add_argument("--data", default=None,
                       help="CSV or Excel: period, bcws, bcwp, acwp, optional eac and wbs "
                            "(ce-core template evm writes one)")
    p_evm.add_argument("--ipmdar", nargs="+", default=None, metavar="DATASET",
                       help="IPMDAR Contract Performance Dataset(s) (preview): a folder, ZIP "
                            "or JSON file; several monthly ones rebuild the history")
    p_evm.add_argument("--cumulative", action="store_true",
                       help="The values are cumulative to date, not per period")
    p_evm.add_argument("--bac", type=float, default=None,
                       help="Budget at completion (default: the baseline's total)")
    p_evm.add_argument("--iters", "--n-iter", dest="iters", type=int, default=20000,
                       help="Simulated completions")
    p_evm.add_argument("--seed", type=int, default=0, help="Random seed")
    p_evm.add_argument("--units", default="as entered",
                       help="What the money is in, for labels: dollars, thousands or millions "
                            "(or any text); nothing is converted")
    p_evm.add_argument("--out", default="evm",
                       help="Directory for the tables, the draws and the chart")

    # Subcommand: schedule-check
    p_dcma = sub.add_parser(
        "schedule-check",
        help="DCMA 14-point assessment of a Microsoft Project XML schedule",
    )
    p_dcma.add_argument("--mspdi", required=True,
                        help="Schedule saved from Microsoft Project as XML (MSPDI)")
    p_dcma.add_argument("--out", default="schedule_check",
                        help="Directory for dcma.csv, dcma_tasks.csv, tasks.csv and links.csv")

    # Subcommand: sar-panel
    p_sar = sub.add_parser(
        "sar-panel",
        help="Unit cost of real programs from public Selected Acquisition Reports",
    )
    p_sar.add_argument("--dir", default=None, metavar="FOLDER",
                       help="Read SAR PDFs already in this folder instead of downloading; "
                            "no network access")
    p_sar.add_argument("--out", default="sar_panel",
                       help="Directory for unit_cost.csv, reports.csv and checks.csv")
    p_sar.add_argument("--cycles", nargs="+", default=None, metavar="CYCLE",
                       help="Reporting cycles, e.g. 'Dec 2019' 'PB 2027' (default: all)")
    p_sar.add_argument("--programs", nargs="+", default=None, metavar="NAME",
                       help="Only programs whose file name contains one of these")
    p_sar.add_argument("--limit", type=int, default=None,
                       help="Stop after this many reports")
    p_sar.add_argument("--list-cycles", action="store_true",
                       help="List the cycles and report counts, fetch nothing")

    # Subcommand: demo
    p_demo = sub.add_parser(
        "demo", help="See a command working on bundled example data, no files needed")
    p_demo.add_argument("topic", choices=sorted(DEMOS),
                        help="Which one: evm, schedule, jcl, aoa or portfolio")
    p_demo.add_argument("--out", default=None,
                        help="Folder for the results (default: ce-core-demo/<topic>)")

    # Subcommand: template
    p_tmpl = sub.add_parser(
        "template", help="Write a file to fill in with your own data")
    p_tmpl.add_argument("topic", choices=sorted(list(TEMPLATE_FILES) + ["schedule"]),
                        help="Which one: evm, jcl, aoa, portfolio, lots or schedule")
    p_tmpl.add_argument("--out", default=None,
                        help="File to write (default: my_<topic> in this folder)")
    p_tmpl.add_argument("--force", action="store_true", help="Replace the file if it exists")

    args = parser.parse_args(argv)
    if args.cmd is None:
        from cost_core.plain import menu
        print(menu())
        return

    # Dispatcher map
    dispatch = {
        "fit-curve": run_fit,
        "forecast": run_forecast,
        "simulate": run_simulate,
        "fit-lots": run_fit_lots,
        "full-run": run_full,
        "sar-panel": run_sar_panel,
        "schedule-check": run_schedule_check,
        "evm": run_evm,
        "aoa": run_aoa,
        "portfolio": run_portfolio,
        "jcl": run_jcl,
        "demo": run_demo,
        "template": run_template,
    }

    dispatch[args.cmd](args)

if __name__ == "__main__":
    main()