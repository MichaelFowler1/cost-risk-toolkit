#!/usr/bin/env python3
# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Generate docs/hero.png - the README image.

Runs the REAL cost_core pipeline on the bundled data.csv: fits a Wright learning
curve (log-log regression), forecasts future lots, and runs the Monte Carlo cost
simulation. Both panels are actual library output, not a mock-up.

Run:  .venv\\Scripts\\python make_hero.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cost_core.learning_curve import fit_learning_curve, forecast_costs
from cost_core.monte_carlo import run_monte_carlo

BG, INK, DIM = "#0b1017", "#e6ebf2", "#8493a6"
C_HIST, C_FIT, C_FCAST = "#39c2ff", "#ffb020", "#28c76f"

# ---------- learning curve (real fit) ----------
df = pd.read_csv("data.csv")
model = fit_learning_curve(df)
fc_q = [32, 64, 128]
fc = forecast_costs(model, fc_q)
qs = np.logspace(0, np.log10(160), 200)
fit_line = np.array([model.predict_unit_cost(q) for q in qs])

# ---------- monte carlo (real sim) ----------
sim = run_monte_carlo(
    n_iter=10000,
    unit_cost_dist={"type": "lognormal", "mean": float(np.log(150)), "sigma": 0.2},
    quantity_dist={"type": "triangular", "left": 40, "mode": 50, "right": 75},
    seed=42,
)

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": "#1b2740"})
fig = plt.figure(figsize=(13, 6.6), facecolor=BG)
fig.text(0.05, 0.94, "COST_CORE  ·  LEARNING-CURVE FORECAST + MONTE-CARLO COST RISK",
         fontsize=15, fontweight="bold")
fig.text(0.05, 0.895, "historical lots  →  log-log learning-curve fit  →  unit-cost forecast   |   "
                      "probabilistic total-cost simulation → P50 / P80 / P90",
         fontsize=9.5, color=DIM)

# ---- left: learning curve ----
ax1 = fig.add_axes([0.06, 0.12, 0.42, 0.68], facecolor="#0a0f18")
ax1.plot(qs, fit_line, color=C_FIT, lw=2,
         label=f"fitted curve  ({model.slope:.0%} learning)")
ax1.scatter(df["unit_quantity"], df["unit_cost"], s=70, color=C_HIST,
            edgecolor="white", linewidth=1, zorder=5, label="historical lots")
ax1.scatter(fc["quantity"], fc["unit_cost"], s=90, marker="D", color=C_FCAST,
            edgecolor="white", linewidth=1, zorder=5, label="forecast lots")
for _, row in fc.iterrows():
    ax1.annotate(f"{int(row['quantity'])}u\n${row['unit_cost']:.0f}",
                 (row["quantity"], row["unit_cost"]), textcoords="offset points",
                 xytext=(6, 8), fontsize=8, color=C_FCAST)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("cumulative unit quantity (log)", fontsize=9)
ax1.set_ylabel("unit cost (log)", fontsize=9)
ax1.tick_params(colors=DIM, labelsize=8)
ax1.grid(which="both", color="#12203a", lw=0.6)
ax1.legend(loc="upper right", fontsize=8.2, facecolor="#0a0f18",
           edgecolor="#1b2740", labelcolor=INK)
ax1.set_title(f"Wright learning curve — T1 ${model.reference_cost/(model.reference_quantity**model.learning_exponent):.0f}, "
              f"slope {model.slope:.1%}", fontsize=9.5, color=DIM, loc="left", pad=6)

# ---- right: monte carlo ----
ax2 = fig.add_axes([0.55, 0.12, 0.41, 0.68], facecolor="#0a0f18")
ax2.hist(sim.samples, bins=55, color="#2b3f66", edgecolor="#1b2740", alpha=0.95)
ytop = ax2.get_ylim()[1]
for val, col, lab, yf in [(sim.p50, "#39c2ff", "P50", 0.97),
                          (sim.p80, "#ffb020", "P80", 0.74),
                          (sim.p90, "#ff4d4d", "P90", 0.97)]:
    ax2.axvline(val, color=col, lw=1.8, ls="--")
    ax2.text(val, ytop * yf, f" {lab}\n ${val:,.0f}", color=col,
             fontsize=8.5, fontweight="bold", va="top")
ax2.set_xlabel("simulated total cost ($)", fontsize=9)
ax2.set_ylabel("frequency", fontsize=9)
ax2.tick_params(colors=DIM, labelsize=8)
ax2.grid(axis="y", color="#12203a", lw=0.6)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax2.set_title(f"Monte Carlo — {len(sim.samples):,} runs, mean ${sim.mean:,.0f}",
              fontsize=9.5, color=DIM, loc="left", pad=6)

fig.text(0.05, 0.025, "Real output — cost_core.learning_curve + cost_core.monte_carlo on the bundled "
                      "data.csv (seed 42). Regenerate: python make_hero.py",
         fontsize=8, color=DIM)

os.makedirs("docs", exist_ok=True)
fig.savefig("docs/hero.png", dpi=140, facecolor=BG)
print(f"[+] wrote docs/hero.png  (slope {model.slope:.1%}, P80 ${sim.p80:,.0f})")
