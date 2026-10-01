# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
cost_core.examples - Worked examples that install with the package.

Every number in them is invented. They are what ``ce-core demo`` runs and
what ``ce-core template`` starts from, so anyone who has installed the
package has something to run before they have data of their own::

    from cost_core.examples import example_path
    example_path("evm")          # the path to evm_example.csv
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: Each example: its file, and what it is, in a line.
EXAMPLES = {
    "evm": ("evm_example.csv",
            "An invented three-account program, 14 months into a 30-month plan, "
            "with the contractor's EAC."),
    "cost-risk": ("cost_risk_example.xlsx",
                  "A ground station upgrade estimated in $M: eight WBS elements with "
                  "ranges, three risks and the correlations between elements."),
    "cer": ("cer_example.xlsx",
            "Twelve invented radar programs, cost in BY2026 $M against weight and power, "
            "and two new radars to price (one outside the data)."),
    "phase": ("phase_example.xlsx",
              "An invented BY2026 estimate in $M, five lines across RDT&E, procurement "
              "and O&M, spread by fiscal year with an illustrative 2% index."),
    "inflate": ("inflate_example.csv",
                "An invented estimate in BY2026 $M, phased by fiscal year, converted to "
                "then-year dollars with an illustrative 2% index (inflate_index.csv)."),
    "schedule": ("example_ims.xml",
                 "A small schedule saved from Microsoft Project as XML: ten tasks, every "
                 "link type, progress, a baseline and a status date."),
    "jcl": ("jcl_example.json",
            "A small science spacecraft from design to launch, in months and $M."),
    "aoa": ("aoa_example.json",
            "Three ways to replace an ageing sensor and the status quo of keeping it, "
            "their costs phased by year, with the economic analysis against the status "
            "quo."),
    "portfolio": ("portfolio_example.json",
                  "Candidate programs with funding options and a budget by year."),
}


def example_path(topic: str) -> Path:
    """The installed file for an example, by topic (a key of :data:`EXAMPLES`)."""
    try:
        name = EXAMPLES[topic][0]
    except KeyError:
        raise KeyError(f"No example called {topic!r}; there are: "
                       f"{', '.join(EXAMPLES)}.") from None
    return _HERE / name
