"""
Cost-Estimation Core CLI
Bridge between the math engine and the terminal.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NoReturn

import pandas as pd

try:
    from cost_core import data_io, learning_curve, monte_carlo
except ImportError as e:
    sys.exit(f"Critical: cost_core modules missing ({e}). Run from project root.")

# Setup minimalist logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def abort(message: str) -> NoReturn:
    """Standardized exit for fatal errors."""
    log.error(message)
    sys.exit(1)

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

def run_gui(args: argparse.Namespace) -> None:
    """Launch the desktop lot cost model."""
    try:
        from cost_core.gui import main as gui_main
    except ImportError as e:
        abort(f"The GUI needs tkinter, which is not available ({e}).")
    raise SystemExit(gui_main())


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

            from cost_core.reporting import charts

            if simulation is not None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="CE Core CLI: Regression & Risk Engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

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

    # Subcommand: gui -- the desktop lot cost model
    sub.add_parser(
        "gui",
        help="Launch the desktop lot cost model (analogy + estimate lots)",
    )

    # Subcommand: fit-lots -- the simple front door for real production data
    p_lots = sub.add_parser(
        "fit-lots",
        help="Fit a curve to a two-column lot file (units, cost)",
    )
    p_lots.add_argument("--csv", required=True,
                        help="CSV or XLSX with a units column and a cost column")
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

    args = parser.parse_args()

    # Dispatcher map
    dispatch = {
        "fit-curve": run_fit,
        "forecast": run_forecast,
        "simulate": run_simulate,
        "gui": run_gui,
        "fit-lots": run_fit_lots,
        "full-run": run_full,
    }

    dispatch[args.cmd](args)

if __name__ == "__main__":
    main()