# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
provenance.py - What produced a run, for the analyst summary.

The desktop tool once hardcoded its version string, so a saved workbook could
not be dated from the inside. These rows record the library version, the git
revision of the checkout when there is one, the run time and whether the rate
projection was the corrected or the legacy one.
"""

from __future__ import annotations

import functools
import os
from datetime import datetime

from cost_core import __version__
from cost_core.lotmodel.config import SETTINGS


@functools.lru_cache(maxsize=None)
def _source_revision() -> str:
    """Short git revision of this checkout, when it is one.

    An installed copy or a downloaded zip has no repository, which is not an
    error: the version string still identifies the release.
    """
    import subprocess

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(
                ["git", "-C", here, "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            suffix = "+modified" if dirty.stdout.strip() else ""
            return out.stdout.strip() + suffix
    except Exception:
        pass
    return ""


def provenance(
    cfg: dict | None = None,
    *,
    tool_version: str | None = None,
    extra: dict | None = None,
) -> dict:
    """What produced this run, so a saved workbook can be identified later.

    A workbook that cannot say which build made it also cannot say whether it
    predates a correction.

    ``tool_version`` replaces the library's own "cost_core <version> (<rev>)"
    string, used as given; the desktop app passes its own. ``extra`` rows, if
    any, follow the tool version.
    """
    cfg = cfg or SETTINGS
    if tool_version is None:
        rev = _source_revision()
        tool_version = f"cost_core {__version__}" + (f" ({rev})" if rev else "")
    legacy = bool(cfg.get("LegacyRateOmission", False))
    rows = {"Tool version": tool_version}
    if extra:
        rows.update(extra)
    rows["Run timestamp"] = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )
    rows["Rate projection"] = (
        "LEGACY - Rate and LC+Rate are overstated; for reconciling a "
        "pre-2.1.0 workbook only"
        if legacy
        else "corrected (projections satisfy the fitted equation)"
    )
    return rows
