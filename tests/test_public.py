# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Reading real Selected Acquisition Reports.

The fixtures are the laid-out text of the pages that matter from five real
SARs (public-domain U.S. government works, from the WHS FOIA reading room), one
or two from each template era, including a scan whose OCR text layer garbles
its headers. The expected numbers below were read off the printed pages by
eye, not produced by the parser, so a test failing here means the parser
misread a real report.

Nothing here touches the network or needs a PDF library: the parser takes page
text, and the fetcher takes an injected opener.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from cost_core.public import (FetchError, SarParseError, check_unit_cost,
                              fetch, parse_sar_pages)
from cost_core.public.catalog import _program_guess, sar_catalog

FIXTURES = Path(__file__).parent / "fixtures" / "sar"


def load(name):
    doc = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    pages = [""] * doc["n_pages"]
    for number, text in doc["pages"].items():
        pages[int(number) - 1] = text
    return parse_sar_pages(pages)


def row(report, comparison, measure, subprogram=None):
    uc = report.unit_cost
    sel = (uc.comparison == comparison) & (uc.measure == measure)
    if subprogram is not None:
        sel &= uc.subprogram == subprogram
    found = uc[sel]
    assert len(found) == 1, f"{len(found)} rows for {comparison} {measure} {subprogram}"
    return found.iloc[0]


# ------------------------------------------------------------ every era ---
@pytest.mark.parametrize("name,program,year,template", [
    ("aehf_dec2014_damir", "AEHF", 2014, "damir"),
    ("sdb2_dec2017_damir_ocr", "SDB II", 2017, "damir"),
    ("aag_dec2021_dave", "AAG", 2021, "dave"),
    ("sdb2_dec2022_dave", "SDB II", 2022, "dave"),
    ("aag_dec2023_msar", "AAG", 2023, "msar"),
])
def test_each_era_reads_whole_and_passes_its_own_arithmetic(name, program, year, template):
    report = load(name)
    assert (report.program, report.report_year, report.template) == (program, year, template)
    # Two baselines, two measures, per program or subprogram.
    assert len(report.unit_cost) % 4 == 0 and len(report.unit_cost) > 0
    assert report.ok, report.checks[~report.checks.ok]
    assert report.unit_cost.base_year.notna().all()


def test_damir_2014_two_subprograms_each_with_its_baselines():
    r = load("aehf_dec2014_damir")
    assert set(r.unit_cost.subprogram) == {"AEHF SV 1-4", "AEHF SV 5-6"}
    pauc = row(r, "current", "PAUC", "AEHF SV 1-4")
    assert (pauc.baseline_cost, pauc.baseline_quantity, pauc.baseline_unit_cost) == (9085.3, 4, 2271.325)
    assert (pauc.current_cost, pauc.current_unit_cost, pauc.reported_pct_change) == (9337.4, 2334.35, 2.77)
    assert pauc.baseline_label == "Mar 2014 APB" and pauc.base_year == 2002
    orig = row(r, "original", "APUC", "AEHF SV 1-4")
    assert orig.revised and orig.baseline_label == "Jun 2011 APB"
    sv56 = row(r, "current", "APUC", "AEHF SV 5-6")
    assert (sv56.baseline_unit_cost, sv56.current_unit_cost, sv56.reported_pct_change) == (1328.0, 1020.05, -23.19)


def test_ocr_scan_reads_despite_garbled_headers():
    # The original-baseline header on this page comes out of OCR as
    # "II BY 2015 $M 1. 1011 BY 2015 $M 1.11", and the current block prints no
    # readable base year at all, so that one comes from the rest of the report.
    r = load("sdb2_dec2017_damir_ocr")
    cur = row(r, "current", "PAUC")
    assert (cur.baseline_cost, cur.current_cost, cur.reported_pct_change) == (4054.9, 4249.0, 5.08)
    assert cur.base_year == 2015 and cur.base_year_source == "document"
    orig = row(r, "original", "APUC")
    assert (orig.baseline_cost, orig.baseline_quantity, orig.current_unit_cost) == (3237.9, 17000, 0.143)
    assert orig.baseline_label == "Oct 2010 APB" and orig.base_year_source == "page"


def test_dave_base_year_comes_from_the_budget_table():
    # The 2021 unit cost page prints no base year; the Acquisition Budget
    # Estimate table carries it as the first number on each row.
    r = load("aag_dec2021_dave")
    assert set(r.unit_cost.base_year) == {2017}
    assert set(r.unit_cost.base_year_source) == {"document"}
    apuc = row(r, "original", "APUC")
    assert (apuc.baseline_unit_cost, apuc.current_unit_cost, apuc.reported_pct_change) == (254.734, 277.218, 8.9)


def test_each_block_keeps_its_own_base_year():
    # SDB II's December 2022 SAR states the current comparison in BY2015 and
    # the original one in BY2010; mixing them would misstate the growth.
    r = load("sdb2_dec2022_dave")
    assert row(r, "current", "PAUC").base_year == 2015
    assert row(r, "original", "PAUC").base_year == 2010
    assert row(r, "original", "PAUC").baseline_quantity == 17163


def test_msar_reads_dates_and_ignores_trailing_breach_text():
    r = load("aag_dec2023_msar")
    apuc = row(r, "current", "APUC")
    # Printed as "17.92% Significant Cost Growth".
    assert (apuc.current_unit_cost, apuc.reported_pct_change) == (328.648, 17.92)
    assert apuc.baseline_label == "02/05/2020"
    assert row(r, "original", "PAUC").baseline_label == "11/17/2017"


# --------------------------------------------------------------- checks ---
def test_a_misread_number_fails_its_check():
    r = load("aag_dec2023_msar")
    uc = r.unit_cost.copy()
    uc.loc[0, "current_cost"] *= 1.1
    checks = check_unit_cost(uc)
    assert not checks.ok.all()
    assert set(checks[~checks.ok].check) >= {"current_unit_times_quantity"}


def test_pages_without_unit_cost_are_refused():
    with pytest.raises(SarParseError):
        parse_sar_pages(["Selected Acquisition Report\nNothing here", "Schedule\n"])


# ---------------------------------------------------------- fetch/catalog ---
def test_fetch_falls_back_to_the_archive_and_records_it(tmp_path):
    asked = []

    def opener(url):
        asked.append(url)
        return (403, b"") if "web.archive.org" not in url else (200, b"%PDF-1.6 test")

    got = fetch("https://www.esd.whs.mil/x/SAR.pdf", cache=tmp_path, opener=opener, delay=0)
    assert got.archived and got.served_from.startswith("https://web.archive.org/web/2026id_/")
    assert got.path.read_bytes() == b"%PDF-1.6 test" and len(got.sha256) == 64
    # A second call is answered from the cache, with the same provenance.
    again = fetch("https://www.esd.whs.mil/x/SAR.pdf", cache=tmp_path, opener=opener, delay=0)
    assert len(asked) == 2 and again.served_from == got.served_from


def test_fetch_names_every_source_it_tried(tmp_path):
    with pytest.raises(FetchError, match="403.*404"):
        fetch("https://gao.gov/a.pdf", cache=tmp_path, delay=0,
              opener=lambda url: (404, b"") if "archive" in url else (403, b""))


def test_catalog_reads_cycles_and_skips_combined_volumes():
    base = "https://www.esd.whs.mil/Portals/54/Documents/FOID/Reading%20Room/Selected_Acquisition_Reports/"
    cdx = "\n".join([
        f"{base}FY_2014_SARS/15-F-0540_AEHF_SAR_Dec_2014.PDF 20211230201010 555840",
        f"{base}FY_2023_SARS/(U)AAG_MSAR_Dec_2023.pdf 20240823155341 945731",
        f"{base}PB_2027_MSARs/AAG_MSAR_FY2027_PBv2.pdf 20260730212345 527129",
        f"{base}FY_2016_SARS/17-F-1350_Navy_FY_2016_SARs_combined.pdf 20211230 1",
        f"{base}15-F-1687_FY1990_SARS.pdf 20210410 1",
    ]).encode()
    cat = sar_catalog(opener=lambda url: (200, cdx))
    assert list(cat.cycle) == ["Dec 2014", "Dec 2016", "Dec 2023", "PB 2027"]
    assert list(cat.kind) == ["SAR", "combined", "MSAR", "MSAR"]
    assert list(cat.program_guess)[::2] == ["AEHF", "AAG"]


@pytest.mark.parametrize("filename,expected", [
    ("DDG_51_December_2012_SAR.pdf", "DDG 51"),
    ("16-F-0402_DOC_42_F-35_DEC_2015_SAR.pdf", "F-35"),
    ("16-F-0402_DOC_15_B-2_EHF_Inc_1_DEC_2015_SAR.pdf", "B-2 EHF Inc 1"),
    ("RQ-4A_B_UAS_GLOBAL_HAWK-SAR_31_DEC_2011.pdf", "RQ-4A B UAS GLOBAL HAWK"),
    ("14-F-0402_DOC_40_JLENSDecember2013SAR.PDF", "JLENS"),
    ("18-F-1016_DOC_37_Army_MQ-1C_Gray_Eagle_SAR_Dec_2017.pdf", "MQ-1C Gray Eagle"),
    ("22-F-0762_LPD_17_SAR_2021.pdf", "LPD 17"),
])
def test_program_guess_from_file_names(filename, expected):
    assert _program_guess(filename) == expected


# ----------------------------------------------------------------- panel ---
class _Got:
    def __init__(self, url, served="https://web.archive.org/x", sha="ab" * 32):
        self.url, self.served_from, self.sha256, self.path = url, served, sha, url


def _catalog(*rows):
    return pd.DataFrame([{"program_guess": g, "cycle": c, "cycle_year": int(c[-4:]),
                          "kind": "SAR", "folder": "f", "filename": u, "url": u,
                          "capture": "2021", "size": 1} for g, c, u in rows])


def test_panel_stacks_reports_dedupes_and_records_failures(monkeypatch):
    import cost_core.public.panel as panel

    by_url = {"a.pdf": "aag_dec2021_dave", "b.pdf": "aag_dec2023_msar",
              "b2.pdf": "aag_dec2023_msar"}
    monkeypatch.setattr(panel, "fetch", lambda url, **kw: _Got(url))

    def read(path, program_hint=None, source=None):
        if path == "broken.pdf":
            raise SarParseError("no unit cost section found")
        return load(by_url[path])

    monkeypatch.setattr(panel, "read_sar", read)
    cat = _catalog(("AAG", "Dec 2021", "a.pdf"), ("AAG", "Dec 2023", "b.pdf"),
                   ("AAG", "Dec 2023", "b2.pdf"), ("XYZ", "Dec 2023", "broken.pdf"))
    lines = []
    p = panel.build_sar_panel(catalog=cat, progress=lines.append)

    # Two copies of the December 2023 report collapse to one set of rows.
    assert len(p.unit_cost) == 8
    assert list(p.reports.status) == ["read", "read", "read", "failed"]
    assert "no unit cost section" in p.reports.error.iloc[3]
    assert p.unit_cost.checks_ok.all() and len(lines) == 4
    orig = p.unit_cost[(p.unit_cost.cycle == "Dec 2023") & (p.unit_cost.measure == "APUC")
                       & (p.unit_cost.comparison == "original")].iloc[0]
    assert orig.unit_cost_growth_pct == pytest.approx((328.648 / 254.733 - 1) * 100)
    assert orig.quantity_change_pct == pytest.approx((4 / 3 - 1) * 100)


def test_panel_retries_a_truncated_capture(monkeypatch):
    import cost_core.public.panel as panel

    stamps = []

    def fake_fetch(url, wayback_timestamp, **kw):
        stamps.append(wayback_timestamp)
        return _Got(url if wayback_timestamp == "2026" else "truncated")

    def read(path, program_hint=None, source=None):
        if path == "truncated":
            raise EOFError("Unexpected EOF")
        return load("aag_dec2023_msar")

    monkeypatch.setattr(panel, "fetch", fake_fetch)
    monkeypatch.setattr(panel, "read_sar", read)
    p = panel.build_sar_panel(catalog=_catalog(("AAG", "Dec 2023", "a.pdf")))
    assert stamps == ["2021", "2026"] and list(p.reports.status) == ["read"]


def test_panel_filters_by_cycle_and_program(monkeypatch):
    import cost_core.public.panel as panel

    monkeypatch.setattr(panel, "fetch", lambda url, **kw: _Got(url))
    monkeypatch.setattr(panel, "read_sar", lambda *a, **k: load("aag_dec2023_msar"))
    cat = _catalog(("AAG", "Dec 2023", "a.pdf"), ("F-35", "Dec 2023", "b.pdf"),
                   ("AAG", "Dec 2021", "c.pdf"))
    p = panel.build_sar_panel(catalog=cat, cycles=["Dec 2023"], programs=["aag"])
    assert list(p.reports.url) == ["a.pdf"]
