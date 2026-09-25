# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
sbom_licences.py - Check every dependency in a CycloneDX SBOM is permissively licensed.

A software approval review at a lab or a government office asks two things of
a Python package: what it installs, and under what terms. The SBOM answers the
first; this answers the second, from the SBOM itself, and fails when a
dependency's licence is copyleft or unknown, so nobody learns that from the
reviewer.

    python tools/sbom_licences.py sbom.json [--markdown licences.md]

Each component's licence is taken from the SBOM (an SPDX id, an SPDX
expression, or a licence name or trove classifier where the package declares
no SPDX form). An ``OR`` expression passes when any branch is permissive, an
``AND`` expression only when every part is. A package that declares nothing
usable is looked up in :data:`VERIFIED`, which records what its source says
and where; anything else unknown fails. The project's own licence is not
checked, since it is what the reviewer is being asked about.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Licences that put no conditions on how the code using them is licensed.
PERMISSIVE = {
    "MIT", "MIT-0", "MIT-CMU", "BSD-2-Clause", "BSD-3-Clause", "0BSD", "ISC",
    "Apache-2.0", "PSF-2.0", "Python-2.0", "Zlib", "CC0-1.0", "HPND", "Unlicense",
}

#: Trove classifiers and free-text names, for packages that give no SPDX id.
_PERMISSIVE_NAMES = re.compile(
    r"\b(MIT|BSD|Apache|Python Software Foundation|PSF|ISC|zlib|Public Domain|HPND)\b", re.I)
_COPYLEFT_NAMES = re.compile(r"\b(GPL|LGPL|AGPL|GNU|Mozilla|MPL|Eclipse|EPL|CDDL|EUPL)\b", re.I)

#: Packages whose metadata declares no licence the SBOM can carry, with what
#: their source says. Each entry was read at the place named, not assumed.
VERIFIED = {
    # PuLP 4's log parser. Its PyPI metadata has no licence field or
    # classifier; the repository's LICENSE.txt is the MIT licence.
    "orloge": ("MIT", "LICENSE.txt in https://github.com/pchtsp/orloge"),
    # pdfplumber's renderer. Its metadata says "BSD-3-Clause, Apache-2.0,
    # dependency licenses" in a form the SBOM tool does not read; the PDFium
    # binary it bundles carries permissive licences (BSD, Apache-2.0, FTL,
    # IJG, zlib, libpng, ICU), listed under BUILD_LICENSES in the wheel.
    "pypdfium2": ("BSD-3-Clause OR Apache-2.0",
                  "METADATA and LICENSES/ in the pypdfium2 wheel"),
}


def component_licences(component: dict) -> "list[str]":
    """The licence strings a CycloneDX component declares, in any form."""
    out = []
    for entry in component.get("licenses", []):
        if "expression" in entry:
            out.append(entry["expression"])
        elif "license" in entry:
            lic = entry["license"]
            out.append(lic.get("id") or lic.get("name") or "")
    return [s for s in out if s.strip()]


def _term_permissive(term: str) -> bool:
    term = term.strip().strip("()").strip()
    if term in PERMISSIVE:
        return True
    if _COPYLEFT_NAMES.search(term):
        return False
    return bool(_PERMISSIVE_NAMES.search(term))


def is_permissive(licence: str) -> bool:
    """True when an SPDX expression or licence name is permissive.

    >>> is_permissive("Apache-2.0 OR BSD-3-Clause")
    True
    >>> is_permissive("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0")
    True
    >>> is_permissive("MIT AND LGPL-2.1-only")
    False
    >>> is_permissive("EPL-2.0")
    False
    >>> is_permissive("License :: OSI Approved :: BSD License")
    True
    """
    if re.search(r"\sOR\s", licence):
        return any(is_permissive(part) for part in re.split(r"\s+OR\s+", licence))
    return all(_term_permissive(part) for part in re.split(r"\s+AND\s+|\s+WITH\s+", licence))


def review(sbom: dict) -> "list[dict]":
    """One row per dependency: name, version, licence, where it came from, ok."""
    own = sbom.get("metadata", {}).get("component", {}).get("name")
    rows = []
    for c in sorted(sbom.get("components", []), key=lambda c: c["name"].lower()):
        if c["name"] == own:
            continue
        found = component_licences(c)
        key = c["name"].lower().replace("_", "-")
        if found:
            licence, source = " / ".join(found), "package metadata"
            ok = all(is_permissive(f) for f in found)
        elif key in VERIFIED:
            licence, source = VERIFIED[key]
            ok = is_permissive(licence)
        else:
            licence, source, ok = "not declared", "", False
        rows.append({"name": c["name"], "version": c.get("version", ""),
                     "licence": licence, "source": source, "ok": ok})
    return rows


def markdown(rows: "list[dict]") -> str:
    lines = ["| Package | Version | Licence | Source | |", "|---|---|---|---|---|"]
    for r in rows:
        mark = "ok" if r["ok"] else "**REVIEW**"
        lines.append(f"| {r['name']} | {r['version']} | {r['licence']} | {r['source']} | {mark} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sbom", type=Path, help="CycloneDX JSON")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Also write the table as Markdown here")
    args = parser.parse_args(argv)
    rows = review(json.loads(args.sbom.read_text(encoding="utf-8")))
    for r in rows:
        print(f"{'ok    ' if r['ok'] else 'REVIEW'} {r['name']:22s} {r['version']:14s} {r['licence']}")
    if args.markdown:
        args.markdown.write_text(markdown(rows), encoding="utf-8")
    bad = [r["name"] for r in rows if not r["ok"]]
    print(f"\n{len(rows) - len(bad)} of {len(rows)} dependencies permissively licensed.")
    if bad:
        print("Needs review: " + ", ".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
