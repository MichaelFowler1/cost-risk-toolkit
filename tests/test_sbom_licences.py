# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""The licence check CI runs on the SBOM: it has to pass what is permissive
and stop what is not, including a package that declares nothing."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "sbom_licences", Path(__file__).parent.parent / "tools" / "sbom_licences.py")
sbom_licences = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sbom_licences)


def _sbom(*components):
    return {"metadata": {"component": {"name": "cost-core"}},
            "components": [{"name": "cost-core", "version": "2.2.0",
                            "licenses": [{"license": {"id": "PolyForm-Noncommercial-1.0.0"}}]},
                           *components]}


def _c(name, *licences, expression=None):
    lic = [{"license": {"id": l}} for l in licences]
    if expression:
        lic.append({"expression": expression})
    return {"name": name, "version": "1", "licenses": lic}


@pytest.mark.parametrize("licence,ok", [
    ("MIT", True), ("Apache-2.0 OR BSD-3-Clause", True), ("PSF-2.0", True),
    ("License :: OSI Approved :: BSD License", True),
    ("EPL-2.0", False), ("GPL-3.0-or-later", False), ("MPL-2.0", False),
    ("MIT AND LGPL-2.1-only", False), ("GPL-2.0-only OR MIT", True),
    ("Eclipse Public License 2.0", False), ("Proprietary", False),
])
def test_licences_are_classified(licence, ok):
    assert sbom_licences.is_permissive(licence) is ok


def test_the_project_itself_is_not_checked_and_permissive_deps_pass():
    rows = sbom_licences.review(_sbom(_c("numpy", expression="BSD-3-Clause AND MIT"),
                                      _c("six", "MIT")))
    assert [r["name"] for r in rows] == ["numpy", "six"] and all(r["ok"] for r in rows)


def test_cbc_would_be_stopped_and_an_undeclared_licence_needs_a_record(tmp_path):
    import json

    rows = sbom_licences.review(_sbom(_c("cbcbox", "EPL-2.0"), _c("mystery"), _c("orloge")))
    by = {r["name"]: r for r in rows}
    assert not by["cbcbox"]["ok"] and not by["mystery"]["ok"]
    assert by["orloge"]["ok"] and "github.com/pchtsp/orloge" in by["orloge"]["source"]
    path = tmp_path / "sbom.json"
    path.write_text(json.dumps(_sbom(_c("cbcbox", "EPL-2.0"))), encoding="utf-8")
    assert sbom_licences.main([str(path), "--markdown", str(tmp_path / "l.md")]) == 1
    assert "**REVIEW**" in (tmp_path / "l.md").read_text(encoding="utf-8")
