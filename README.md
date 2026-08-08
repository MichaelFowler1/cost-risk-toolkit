# Cost-Estimation Core (cost_core)

[![tests](https://github.com/MichaelFowler1/Cost_AI_v1/actions/workflows/tests.yml/badge.svg)](https://github.com/MichaelFowler1/Cost_AI_v1/actions/workflows/tests.yml)

This repository contains the foundational Python library and command-line interface (CLI) for our cost-estimation and risk analysis workflows.

The core is designed to help analysts fit log-log learning curves to historical production data, project future unit costs, and run Monte Carlo simulations to understand cost risk and confidence intervals (e.g., P50, P80).

![Learning-curve forecast and Monte Carlo cost risk](docs/hero.png)

*Real output — `cost_core` fits an 85% Wright learning curve to the bundled `data.csv` and forecasts future lots (left), then runs a 10,000-iteration Monte Carlo total-cost simulation with P50/P80/P90 thresholds (right). Regenerate with `python make_hero.py`.*

## Features

* **Data Ingestion & Fitting:** Safely load historical cost data and calculate theoretical learning curve slopes (log-log regression).
* **Forecasting:** Project unit and total costs for future production lots based on fitted models.
* **Risk Simulation:** Run custom Monte Carlo simulations using standard statistical distributions to generate defensible risk thresholds.

## Installation

This project requires Python 3.11 or higher. It is recommended to install the library within a virtual environment.

1. Create and activate a virtual environment:

```text
python -m venv .venv
.venv\Scripts\activate

```

2. Install the package in editable mode. This will automatically pull in required dependencies like pandas, numpy, and scipy:

```text
pip install -e .

```

Once installed, the CLI tool `ce-core` will be globally available within your virtual environment.

## Usage Guide

The library exposes a single terminal command `ce-core` with three primary subcommands. You can view the help menu at any time by running `ce-core --help`.

### 1. Fit a Learning Curve

Calculate the learning curve slope from a CSV of historical data. The resulting model parameters are saved as a JSON file.

```text
ce-core fit-curve --csv data.csv --out model_params.json

```

### 2. Forecast Future Costs

Use a fitted model to project the costs for upcoming production lots. Pass the targeted unit quantities as a comma-separated list.

```text
ce-core forecast --model model_params.json --quantities "32,64,128" --out forecast.csv

```

### 3. Run a Monte Carlo Simulation

Run a probabilistic simulation to find the P50, P80, etc. You can pass JSON strings to define the distributions for unit cost and quantity.

*Note: If using PowerShell, wrap the JSON arguments in single quotes to prevent parsing errors.*

```text
ce-core simulate --n-iter 10000 --unit-cost-dist '{"type": "lognormal", "mean": 5.0, "sigma": 0.2}' --quantity-dist '{"type": "triangular", "left": 40, "mode": 50, "right": 75}' --out sim_results.csv

```

## Project Structure

* `/cost_core`: The core mathematical Python modules (data_io, learning_curve, monte_carlo).
* `/cli`: The command-line interface wrappers that expose the core to the terminal.
* `pyproject.toml`: The modern build system configuration defining dependencies and CLI entrypoints.
* `/tests`: Property tests for the maths — see below.

## Tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -q
```

35 tests, run on Python 3.11 and 3.12 on every push. They check the maths
against known answers rather than checking that the code runs:

- **Doubling quantity multiplies unit cost by the slope.** That is the
  definition of a Wright curve, so it is asserted to machine precision across
  five different slopes.
- **A round trip through the fit.** Data is generated from a curve of known
  slope, fitted, and the fit has to return that slope — and then reprice the
  original data. A broken log-log regression or exponent conversion cannot
  survive that.
- **The simulation is deterministic.** The same seed gives byte-identical
  samples, because a P80 that moves between runs is not a number you can put
  in front of anyone.
- **Statistics against closed forms.** A triangular distribution's mean is
  `(left + mode + right) / 3`; a fixed quantity times a normal unit cost has a
  known mean total. Both are checked against the analytic answer, not a
  recorded one.
- **Bad input is refused, not absorbed.** Non-positive costs, a single data
  point, unknown distributions, missing parameters, negative quantities and
  missing schema columns all raise rather than silently producing a
  plausible-looking number.
