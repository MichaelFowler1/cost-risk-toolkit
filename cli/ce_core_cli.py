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
        "full-run": run_full,
    }

    dispatch[args.cmd](args)

if __name__ == "__main__":
    main()