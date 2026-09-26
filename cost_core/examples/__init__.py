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
    "inflate": ("inflate_example.csv",
                "An invented estimate in BY2026 $M, phased by fiscal year, converted to "
                "then-year dollars with an illustrative 2% index (inflate_index.csv)."),
    "schedule": ("example_ims.xml",
                 "A small schedule saved from Microsoft Project as XML: ten tasks, every "
                 "link type, progress, a baseline and a status date."),
    "jcl": ("jcl_example.json",
            "A small science spacecraft from design to launch, in months and $M."),
    "aoa": ("aoa_example.json",
            "Three alternatives for a capability, their costs phased by year."),
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
