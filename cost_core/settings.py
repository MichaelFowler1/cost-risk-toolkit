# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
settings.py - Defaults you set once, in ce-core.toml.

Every command takes its options on the command line. The ones people type
every time (the marking, the slide template, the units, the seed) can live
in a ``ce-core.toml`` instead: first the one in the folder you run from,
then the one in your home folder. A flag on the command line always wins, and
a workbook's own Settings sheet wins over the file for that workbook.

``ce-core settings`` shows what's in effect and where each value came from;
``ce-core settings --write`` starts a commented file to edit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

FILE_NAME = "ce-core.toml"

#: Each setting: its type, its built-in default and what it's for.
KNOWN: Dict[str, Tuple[type, Any, str]] = {
    "marking": (str, None, "Text stamped at the top and bottom of every slide and "
                           "sheet, exactly as written (for example CUI)"),
    "slide_template": (str, None, "Your organisation's PowerPoint template (.pptx or "
                                  ".potx) for brief.pptx"),
    "units": (str, None, "Money label when a command doesn't get one: dollars, "
                         "thousands, millions or any word"),
    "seed": (int, None, "Random seed for the simulations, so a rerun gives the same answer"),
    "iterations": (int, None, "Simulations per run, 1,000 or more"),
    "fiscal_year_start": (int, 10, "Month the fiscal year starts: 10 for the U.S. "
                                   "government's October"),
}


class SettingsError(ValueError):
    """A ce-core.toml that can't be used; the message says which file and key."""


def _toml():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    import tomli  # the same parser, before it joined the standard library
    return tomli


def _read(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            data = _toml().load(fh)
    except Exception as e:  # noqa: BLE001 - any parse failure is the same problem
        raise SettingsError(f"{path} isn't valid TOML ({e}). Each line is name = value, "
                            'with text in quotes: marking = "CUI".') from None
    unknown = sorted(set(data) - set(KNOWN))
    if unknown:
        raise SettingsError(f"{path}: {', '.join(map(repr, unknown))} "
                            f"{'is' if len(unknown) == 1 else 'are'} not a setting. The "
                            f"settings are {', '.join(KNOWN)}.")
    for key, value in data.items():
        kind = KNOWN[key][0]
        if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise SettingsError(f"{path}: {key} is a whole number, not {value!r}.")
        if kind is str and not isinstance(value, str):
            raise SettingsError(f'{path}: {key} is text, in quotes: {key} = "{value}".')
    if "seed" in data and data["seed"] < 0:
        raise SettingsError(f"{path}: seed is 0 or more, not {data['seed']}.")
    if "iterations" in data and data["iterations"] < 1000:
        raise SettingsError(f"{path}: iterations is 1,000 or more, or the P80 is mostly "
                            f"noise; got {data['iterations']}.")
    if "fiscal_year_start" in data and not 1 <= data["fiscal_year_start"] <= 12:
        raise SettingsError(f"{path}: fiscal_year_start is a month, 1 to 12; got "
                            f"{data['fiscal_year_start']}.")
    if "slide_template" in data:
        template = Path(data["slide_template"]).expanduser()
        if not template.is_absolute():
            template = path.parent / template  # relative to the file that names it
        data["slide_template"] = str(template)
    return data


def files(cwd: Optional[Path] = None, home: Optional[Path] = None):
    """The settings files that apply, most specific first."""
    cwd = Path.cwd() if cwd is None else Path(cwd)
    home = Path.home() if home is None else Path(home)
    found = []
    for folder in (cwd, home):
        path = folder / FILE_NAME
        if path.is_file() and path.resolve() not in [p.resolve() for p in found]:
            found.append(path)
    return found


def load(cwd: Optional[Path] = None, home: Optional[Path] = None
         ) -> Dict[str, Tuple[Any, str]]:
    """Every setting as (value, where it came from)."""
    out = {key: (default, "built-in default") for key, (_, default, _) in KNOWN.items()}
    for path in reversed(files(cwd, home)):  # the home file first, the folder's over it
        for key, value in _read(path).items():
            out[key] = (value, str(path))
    return out


def values(cwd: Optional[Path] = None, home: Optional[Path] = None) -> Dict[str, Any]:
    return {k: v for k, (v, _) in load(cwd, home).items()}


TEMPLATE = """\
# ce-core.toml: defaults for every ce-core command run from this folder.
# A flag on the command line always wins. Delete the # to use a line.

# Text stamped at the top and bottom of every slide and every sheet, exactly as
# written. cost-core doesn't decide or check markings; your organisation does.
# marking = "CUI"

# Your organisation's PowerPoint template for brief.pptx (.pptx or .potx), relative
# to this file.
# slide_template = "house.potx"

# Money label when a command isn't given one: dollars, thousands, millions or any word.
# units = "millions"

# Random seed, so a rerun gives exactly the same answer, and simulations per run.
# seed = 1
# iterations = 20000

# Month the fiscal year starts: 10 is the U.S. government's (1 October).
# fiscal_year_start = 10
"""


def write_template(folder: Optional[Path] = None) -> Path:
    path = (Path.cwd() if folder is None else Path(folder)) / FILE_NAME
    if path.exists():
        raise SettingsError(f"{path} already exists; edit it, or delete it first.")
    path.write_text(TEMPLATE, encoding="utf-8")
    return path
