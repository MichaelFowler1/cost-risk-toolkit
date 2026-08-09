"""
assumptions.py - The written assumptions and provenance log.

An estimate is only as defensible as the record of how it was made. This module
builds that record: a Markdown document emitted with every run, listing the
data it used, every methodological choice, every assumption that was applied
rather than measured, and every validation gate with its verdict.

The organising principle is the GAO Cost Estimating and Assessment Guide's four
characteristics of a reliable estimate, and each section says which one it
serves:

**Comprehensive** -- the WBS is complete, the ground rules are stated, and
nothing has been left out silently.
**Well-documented** -- the source data, the methods and the assumptions are
written down in enough detail that someone else could reproduce the result.
That is what this file is for.
**Accurate** -- the arithmetic reconciles, the estimating methods are unbiased,
and known biases are measured rather than ignored.
**Credible** -- the sensitivity of the answer to its assumptions is quantified,
and a risk analysis says how confident the number is.

The log distinguishes throughout between things that were *measured* and things
that were *assumed*. An assumption presented as a finding is the failure mode
this document exists to prevent, so assumptions get their own section and are
counted.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AssumptionLog:
    """Accumulates the record of one analysis run.

    Attributes:
        title: Document title.
        sections: Ordered heading -> body Markdown.
        assumptions: Things applied by judgement rather than measured. Each is
            a (topic, statement, basis) triple.
        gao_notes: Characteristic -> what this run did about it.
    """

    title: str = "Cost estimate assumptions and provenance"
    sections: list[tuple[str, str]] = field(default_factory=list)
    assumptions: list[tuple[str, str, str]] = field(default_factory=list)
    gao_notes: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    )

    # ----------------------------------------------------------------- build
    def section(self, heading: str, body: str) -> "AssumptionLog":
        self.sections.append((heading, body.rstrip()))
        return self

    def assume(self, topic: str, statement: str, basis: str) -> "AssumptionLog":
        """Record something applied by judgement rather than derived.

        ``basis`` is not optional by accident. An assumption without a stated
        basis is indistinguishable from a guess, and the reviewer's first
        question is always where the number came from.
        """
        self.assumptions.append((topic, statement, basis))
        return self

    def gao(self, characteristic: str, note: str) -> "AssumptionLog":
        self.gao_notes.setdefault(characteristic, []).append(note)
        return self

    def table(self, heading: str, frame: pd.DataFrame, floatfmt: str = ",.4g"):
        """Add a DataFrame as a Markdown table."""
        return self.section(heading, _markdown_table(frame, floatfmt))

    # ---------------------------------------------------------------- render
    def render(self) -> str:
        parts = [
            f"# {self.title}",
            "",
            f"*Generated {self.created_at} by `cost_core` on "
            f"Python {sys.version.split()[0]} ({platform.system()}).*",
            "",
            "This document is emitted automatically with every run. It records "
            "what was measured, what was assumed, and every validation gate "
            "that was applied. Numbers in the accompanying charts and tables "
            "come from the same run.",
            "",
        ]

        for heading, body in self.sections:
            parts += [f"## {heading}", "", body, ""]

        parts += ["## Assumptions applied", ""]
        if self.assumptions:
            parts += [
                "Each of these was applied by judgement rather than derived "
                "from the data. They are the first things a reviewer should "
                "push on, and the first things to revisit if the answer looks "
                "wrong.",
                "",
                "| Topic | Assumption | Basis |",
                "| --- | --- | --- |",
            ]
            for topic, statement, basis in self.assumptions:
                parts.append(
                    f"| {_escape(topic)} | {_escape(statement)} | {_escape(basis)} |"
                )
            parts += ["", f"**{len(self.assumptions)} assumption(s) recorded.**", ""]
        else:
            parts += ["None recorded.", ""]

        parts += [
            "## GAO Cost Estimating and Assessment Guide",
            "",
            "How this run addresses the four characteristics of a reliable "
            "estimate.",
            "",
        ]
        for characteristic in (
            "Comprehensive", "Well-documented", "Accurate", "Credible"
        ):
            parts += [f"### {characteristic}", ""]
            notes = self.gao_notes.get(characteristic, [])
            if notes:
                parts += [f"- {note}" for note in notes]
            else:
                parts.append("- Not addressed by this run.")
            parts.append("")

        return "\n".join(parts).rstrip() + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        logger.info(
            "Wrote assumptions log to %s (%d sections, %d assumptions)",
            path, len(self.sections), len(self.assumptions),
        )
        return path


def _escape(text: Any) -> str:
    """Keep a pipe in free text from breaking a Markdown table."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ",.4g") -> str:
    """Render a DataFrame as a Markdown table."""
    if frame.empty:
        return "*(no rows)*"

    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return ""
            return format(float(value), floatfmt)
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        return _escape(value)

    header = "| " + " | ".join(str(c).replace("_", " ") for c in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])
