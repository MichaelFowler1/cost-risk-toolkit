# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
output.py - How this run's reports should look: marking and slide template.

A briefing that leaves a lab or a program office usually has to carry a
marking at the top and bottom of every page (CUI, a distribution statement,
a proprietary legend), and is expected on the organisation's own slide
master. Both are set once for a run, from ``--marking`` and ``--template`` or
``ce-core.toml``, and every workbook and deck the run writes picks them up.

cost-core stamps exactly the text it's given. It doesn't decide, check or
validate markings; what a document needs is for the organisation and the
person releasing it to say.

From Python::

    from cost_core.reporting import output
    output.configure(marking="CUI", slide_template="house.potx")
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

_state = {"marking": None, "slide_template": None}


def configure(marking: Optional[str] = None, slide_template=None) -> None:
    """Set the marking and slide template for everything written from now on.
    ``None`` clears each."""
    text = (marking or "").strip()
    _state["marking"] = text or None
    _state["slide_template"] = Path(slide_template) if slide_template else None
    if _state["slide_template"] is not None and not _state["slide_template"].is_file():
        path = _state["slide_template"]
        _state["slide_template"] = None
        raise FileNotFoundError(f"The slide template {path} isn't there.")


@contextlib.contextmanager
def configured(marking: Optional[str] = None, slide_template=None):
    """``configure`` for the length of a ``with`` block."""
    before = dict(_state)
    configure(marking, slide_template)
    try:
        yield
    finally:
        _state.update(before)


def marking() -> Optional[str]:
    return _state["marking"]


def slide_template() -> Optional[Path]:
    return _state["slide_template"]


def mark_workbook(wb) -> None:
    """Put the marking in the header and footer of every sheet of an openpyxl
    workbook, where it prints at the top and bottom of every page."""
    text = marking()
    if not text:
        return
    for ws in wb.worksheets:
        for hf in (ws.oddHeader, ws.evenHeader, ws.firstHeader):
            hf.center.text = text
            hf.center.size = 11
        for hf in (ws.oddFooter, ws.evenFooter, ws.firstFooter):
            hf.center.text = text
            hf.center.size = 11
