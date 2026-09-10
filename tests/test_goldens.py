"""The golden regression: cost_core alone must reproduce the captured engine.

``tests/goldens/`` holds what the desktop tool's engine produced on 2026-09-06,
case by case, at full float precision. Every test here rebuilds its inputs FROM
THE GOLDEN FILE ITSELF -- the analogy and estimate frames, the overrides, the
run info, the programme definition, the lot series, the CLI arguments -- runs
them through ``cost_core``, and compares the result with
:func:`golden_support.compare_with_policy` under ``COMPARE_POLICY.json``.

A disagreement is a failure, never a warning, and the fix is never to
regenerate the golden. See ``tests/goldens/README.md``.

Stage
-----
``GOLDEN_STAGE`` selects which exclusion lists apply, and it is 2. Step 3
(the raw-draw handoff replacing the lognormal one) moved the WBS
``program_level_percentiles`` block on purpose and could have set the stage to
3, which drops that block from the comparison. It rebaselined the block to its
new values instead, so the block is still compared leaf by leaf and nothing
here is unpinned. ``COMPARE_POLICY.exclude_after_step3`` records what was
rebaselined, when and by how much.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import golden_support as GS

#: 2. Step 3 could have moved this to 3, which drops wbs
#: program_level_percentiles from the comparison; it rebaselined that block
#: instead and left the stage here, so the block stays golden.
GOLDEN_STAGE = 2

GOLDENS = Path(__file__).resolve().parent / "goldens"
POLICY = GS.load_policy(GOLDENS / "COMPARE_POLICY.json")

#: Engine golden blocks this test does not compare, and why. They are produced
#: by the desktop tool's own risk.py / workbook writer, which the library does
#: not have and this refactor does not touch; the library-side numbers behind
#: them (ctx, projections, enrich.simulate_buy) ARE compared.
APP_ONLY_ENGINE_BLOCKS = {
    "risk_run_risk": "risk.run_risk lives in the desktop tool (risk.py); cost_core has no port of it",
    "risk_variants": "same: the tests/test_risk.py option sets run the tool's risk.run_risk",
    "test_workbook_fixture": "tests/test_workbook.py fixtures written by the tool's Excel writer",
}

#: error_paths blocks this test does not compare, and why.
APP_ONLY_ERROR_BLOCKS = {
    "risk": "risk.run_risk refusal from the desktop tool's risk.py, which cost_core does not have",
}

#: error_paths.json records the engine refusals twice, once per engine. The
#: golden side is the desktop tool's ("app"), which is what the library has to
#: reproduce; the ".lib" side is the library as it stood at capture time,
#: before "Bring the desktop tool's engine, summary and workbook writer up to
#: date" (21cde45) reworded its ToolMatchProjection message to the tool's.
ERROR_ENGINE_SIDE = "app"

#: sha256 entries not compared: a matplotlib PNG is not byte-reproducible
#: across matplotlib versions, and the CSVs are what carry the numbers.
CLI_SHA_SKIP = {"buy_s_curve.png": "matplotlib PNG bytes are renderer-version dependent"}

#: True on the numpy/pandas the goldens were captured under. The CSVs are
#: written at full repr precision, so their exact bytes carry the last bit of a
#: double: numpy 2.0.2 and numpy 2.4.4 disagree in the 16th digit of exp() and
#: the prediction-interval CSV then hashes differently although every number in
#: it agrees to 1e-16. Byte identity was therefore asserted on the capture lane
#: only; off it, csv_text and sha256 are dropped here and the same numbers are
#: compared through csv_contents (parsed with float_precision='round_trip') at
#: the policy's rtol 1e-9, which is the tolerance that actually matters.
#:
#: Since step 2 the comparator drops both on every lane as well (COMPARE_POLICY
#: expected_to_move_beyond_rtol_in_step_2.full_repr_csv_bytes): moving the
#: engine on to one estimator moved the coefficients in the 13th digit, which
#: is the same digit a numpy version moves. This lane test stays because it
#: says which lane the run is on.
_CAPTURED = POLICY["captured_under"]
CAPTURE_LANE = (np.__version__ == _CAPTURED["numpy"] and pd.__version__ == _CAPTURED["pandas"])
CLI_BYTES_SKIP_REASON = (
    f"numpy {np.__version__} / pandas {pd.__version__} is not the capture lane "
    f"(numpy {_CAPTURED['numpy']} / pandas {_CAPTURED['pandas']}); CSV bytes carry the last bit of a double")

ENGINE_CASES = sorted(p.name[len("engine_"):-len(".json.gz")] for p in GOLDENS.glob("engine_*.json.gz"))
WBS_PROGRAMS = ["TEST_PROGRAM", "FULL_WBS"]

#: The goldens the suite loads by name. A missing one of these already fails
#: loudly, because load_golden opens the file.
NAMED_GOLDENS = ("cli_fit_lots", "error_paths", "lots_cost_core",
                 "wbs_FULL_WBS", "wbs_TEST_PROGRAM", "wbs_extra_programs")

#: How many engine cases were captured on 2026-09-06. The parametrised engine
#: test globs the directory, so without this a golden that is deleted, or that
#: never reaches the commit, would delete its own test and the suite would go
#: green with fewer cases instead of red. The census below is the count; the
#: file list is checked with it, so a rename cannot pass either.
EXPECTED_ENGINE_CASES = 36
EXPECTED_EXTRA_FILES = ("COMPARE_POLICY.json", "README.md")


def _census():
    """(problems, files) -- what tests/goldens must contain, checked."""
    problems = []
    if len(ENGINE_CASES) != EXPECTED_ENGINE_CASES:
        problems.append(f"expected {EXPECTED_ENGINE_CASES} engine goldens, found {len(ENGINE_CASES)}: "
                        f"{ENGINE_CASES}")
    on_disk = sorted(p.name for p in GOLDENS.iterdir() if p.is_file())
    expected = sorted([f"engine_{c}.json.gz" for c in ENGINE_CASES]
                      + [f"{n}.json.gz" for n in NAMED_GOLDENS] + list(EXPECTED_EXTRA_FILES))
    if on_disk != expected:
        missing = sorted(set(expected) - set(on_disk))
        extra = sorted(set(on_disk) - set(expected))
        problems.append(f"tests/goldens contents differ: missing {missing}, unexpected {extra}")
    return problems, on_disk


_CENSUS_PROBLEMS, _GOLDEN_FILES = _census()
# Fails at collection, so it is caught even when the run selects a subset (-k)
# that would not have included test_golden_census.
assert not _CENSUS_PROBLEMS, "tests/goldens census: " + "; ".join(_CENSUS_PROBLEMS)

#: Cases the summary flip probe runs on. It reads summary.py's own statistics
#: back at full precision out of the point where each printed digit flips, so
#: it is about that arithmetic rather than about a particular set of numbers,
#: and it costs about half a second a case. Two are enough: both have all three
#: models fitted, so all three summary columns and all eight statistics exist,
#: and they sit at opposite ends of the selection rule -- LC wins on
#: cc_reference, Rate on rate_driven_8lots.
SUMMARY_PRECISION_CASES = ["cc_reference", "rate_driven_8lots"]

_TIMINGS: dict = {}


def load_golden(name: str):
    """The exact LF bytes captured on 2026-09-06, un-gzipped."""
    with gzip.open(GOLDENS / f"{name}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def _time_case(request):
    started = time.perf_counter()
    yield
    _TIMINGS[request.node.name] = time.perf_counter() - started


@pytest.fixture(scope="module", autouse=True)
def _report_timings():
    yield
    if not _TIMINGS:
        return
    slow = sorted(_TIMINGS.items(), key=lambda kv: -kv[1])
    print("\ngolden case timings (seconds), slowest first:")
    for name, secs in slow[:10]:
        print(f"  {secs:7.2f}  {name}")
    print(f"  {sum(_TIMINGS.values()):7.2f}  TOTAL over {len(_TIMINGS)} cases")


def check(golden, new, *, case=None, new_only_items=()):
    """Compare and fail with every disagreeing leaf spelled out."""
    bad = GS.compare_with_policy(golden, GS.to_json_tree(new), POLICY, stage=GOLDEN_STAGE,
                                 case=case, new_only_items=new_only_items)
    if bad:
        pytest.fail(GS.format_mismatches(bad), pytrace=False)


# --------------------------------------------------------------------------
# the harness itself
# --------------------------------------------------------------------------


def test_golden_census():
    """Exactly the goldens captured on 2026-09-06 are here, and nothing else.

    The engine test globs the directory for its parameters, so a golden that
    goes missing would otherwise take its own test with it and the suite would
    stay green on fewer cases. The same assertion runs at import (above), which
    is what covers a subset run that deselects this test.
    """
    assert not _CENSUS_PROBLEMS, "; ".join(_CENSUS_PROBLEMS)
    assert len(_GOLDEN_FILES) == EXPECTED_ENGINE_CASES + len(NAMED_GOLDENS) + len(EXPECTED_EXTRA_FILES)
    # every parametrised and named golden really is loadable
    for case in ENGINE_CASES:
        assert (GOLDENS / f"engine_{case}.json.gz").is_file()
    for name in NAMED_GOLDENS:
        assert (GOLDENS / f"{name}.json.gz").is_file()


def _projection_frame(value):
    """A one-row projections frame carrying a 4-dp rounded column."""
    return {"columns": ["Complexity Factor"], "shape": [1, 1],
            "records": [{"Complexity Factor": value}]}


def test_rounded_tolerance_is_absolute():
    """The rounded and sum rules are absolute limits, as COMPARE_POLICY states.

    ``rounded_leaves.rule`` is ``one_unit * (1 + 1e-6) + atol`` and
    ``sums_of_rounded`` is ``n_terms * one_unit``; neither carries an rtol
    term, and "a leaf that moves by more than one unit is a real regression"
    only means anything if none is added. On a large cell -- projections run to
    1e8 and the lots ``LC F`` column to 1e26 -- the general rule's
    ``rtol * |value|`` would swamp the unit and make the tolerance thousands of
    units wide, so it is dropped exactly where one of these two rules fires.
    """
    one_unit = 1e-4  # 'Complexity Factor', dp 4
    big = 1e8        # a realistic lot cost; rtol * big = 0.1, a thousand units
    # the move the general rule would have absorbed at that magnitude and the
    # rounded rule must not: 500 units, and still inside rtol * big
    drift = 0.05
    assert one_unit < drift < POLICY["tolerance"]["rtol"] * big

    # a genuine one-unit boundary flip still passes, at any magnitude
    for base in (100.0, big):
        assert not GS.compare_with_policy(
            {"projections": _projection_frame(base)},
            {"projections": _projection_frame(base + one_unit)}, POLICY, stage=GOLDEN_STAGE)

    # 500 units does not, however large the value
    bad = GS.compare_with_policy({"projections": _projection_frame(big)},
                                 {"projections": _projection_frame(big + drift)}, POLICY, stage=GOLDEN_STAGE)
    assert [m["path"] for m in bad] == ["/projections/records[0]/Complexity Factor"], bad
    assert "rounded_leaves.atol_by_column" in bad[0]["rule"]
    assert "absolute" in bad[0]["tolerance"]

    # and the same for a sum of rounded leaves: three rows, so three 0.01 units
    def sums(total, rows=3):
        return {"projections": dict(_projection_frame(1.0), shape=[rows, 1]),
                "projections_column_sums": {"Complexity Factor": total}}

    assert not GS.compare_with_policy(sums(big), sums(big + 0.02), POLICY, stage=GOLDEN_STAGE)
    bad = GS.compare_with_policy(sums(big), sums(big + drift), POLICY, stage=GOLDEN_STAGE)
    assert [m["path"] for m in bad] == ["/projections_column_sums/Complexity Factor"], bad
    assert bad[0]["rule"].startswith("sums_of_rounded (n_terms=3"), bad[0]["rule"]


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", ENGINE_CASES)
def test_engine_case(case):
    golden = load_golden(f"engine_{case}")
    inputs = golden["inputs"]
    analogy = GS.frame_from_json(inputs["analogy"])
    estimate = GS.frame_from_json(inputs["estimate"])
    data, _prov, live = GS.capture_engine_lib(analogy, estimate, inputs["overrides"], inputs["run_info"])
    if "test_lotmodel_fixtures" in golden:
        # the tests/test_lotmodel.py simulate_buy seeds; captured only for the
        # two cases that suite runs against (cc_reference and app_example)
        projections, ctx = live[0], live[1]
        data["test_lotmodel_fixtures"] = GS.test_lotmodel_fixtures(ctx, projections)
    for block, why in APP_ONLY_ENGINE_BLOCKS.items():
        if block in golden:
            assert why  # the reason is the documentation; keep it visible in the source
            golden.pop(block)
    check(golden, data, case=case)
    # The eight statistics summary.py computes for itself reach the golden only
    # as strings printed at 2 or 4 dp, so a sub-printed-digit change in its own
    # arithmetic moves no golden byte. This reads |t| and the AICc gap back out
    # of summary.py's selection thresholds at full precision and holds every
    # printed stat to half a unit of derived_stats, which the golden above does
    # pin at 1e-9. golden_support's "summary.py's own arithmetic" section says
    # what that reaches and what stays at printed resolution.
    bad = GS.check_summary_arithmetic(live[1], live[2], POLICY)
    if bad:
        pytest.fail("summary.py disagrees with derived_stats:\n" + GS.format_mismatches(bad), pytrace=False)


@pytest.mark.parametrize("case", SUMMARY_PRECISION_CASES)
def test_summary_precision(case):
    """summary.py's own eight statistics, at 1e-9 rather than at 2 decimals.

    The golden can only hold what the desktop tool printed, which is 2 or 4
    decimal places, so the summary frame alone cannot tell a correct SEE from
    one multiplied by 1 + 1e-6. This reads each statistic back out of the knob
    value at which its printed digit flips, which does carry full precision,
    and compares it with derived_stats -- the block the golden pins at rtol
    1e-9. golden_support's "summary.py's own arithmetic" section explains it.

    The ctx comes straight off the golden: it is input to summary.py here, and
    test_engine_case is what proves cost_core still computes that same ctx.
    """
    golden = load_golden(f"engine_{case}")
    bad = GS.check_summary_precision(golden["ctx"], POLICY)
    if bad:
        pytest.fail("summary.py's arithmetic disagrees with derived_stats:\n"
                    + GS.format_mismatches(bad), pytrace=False)


# --------------------------------------------------------------------------
# the programme roll-up
# --------------------------------------------------------------------------


@pytest.mark.parametrize("program", WBS_PROGRAMS)
def test_wbs_program(program, tmp_path):
    golden = load_golden(f"wbs_{program}")
    # the roll-up arguments the capture used, read back off the golden so a
    # changed seed or iteration count cannot slip past unnoticed
    assert golden["program_level_percentiles"]["n_iter"] == 8000
    assert golden["program_level_percentiles"]["seed"] == 11
    assert list(golden["overrides"]) == ["LegacyRateOmission_True", "FcstPriorUnits_40"]
    for entry in golden["element_standalone_simulate_buy"].values():
        assert (entry["n_iter"], entry["seed"], entry["lot_correlation"]) == (8000, 11, 0.30)
    prog = GS.program_from_inputs(golden)
    new = GS.capture_wbs(prog, tmp_path, {})
    # The three roll-up knobs are inputs, not numbers step 3 rebaselined, and
    # correlation is a defaulted argument (rollup.roll_up correlation=
    # monte_carlo.DEFAULT_CORRELATION). They are asserted here, outside the
    # policy walk, so that they stay pinned even if a later step does drop the
    # percentiles around them; otherwise changing the default correlation
    # could move numbers with nothing going red.
    for knob in ("n_iter", "seed", "correlation"):
        assert new["program_level_percentiles"][knob] == golden["program_level_percentiles"][knob], (
            f"roll-up {knob} changed: golden {golden['program_level_percentiles'][knob]!r}, "
            f"now {new['program_level_percentiles'][knob]!r}")
    check(golden, new)


def test_wbs_extra_programs():
    golden = load_golden("wbs_extra_programs")
    for name, entry in golden["programs"].items():
        rebuilt = GS.program_from_inputs(entry["inputs"])
        assert rebuilt.name == name or name.startswith(rebuilt.name)
    check(golden, GS.capture_wbs_extras({}))


# --------------------------------------------------------------------------
# cost_core.lots
# --------------------------------------------------------------------------


def test_lots_cost_core(tmp_path):
    # This golden was captured off the library at d7cdea2, before 21cde45 gave
    # provenance() its "Rate projection" row, so its 27-row fit.summary frames
    # meet a 28-row frame today. That one absence is allowed here and nowhere
    # else: the row IS compared in all 36 engine goldens and both wbs element
    # summaries, where it also witnesses LegacyRateOmission.
    check(load_golden("lots_cost_core"), GS.capture_lots(tmp_path),
          new_only_items=GS.LOTS_GOLDEN_PREDATES_ITEMS)


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------


@pytest.mark.parametrize("run", sorted(GS.CLI_RUNS))
def test_cli_fit_lots(run, tmp_path):
    golden = load_golden("cli_fit_lots")[run]
    # the arguments come off the golden's own masked command line. It was
    # captured when the command line was a top-level cli package; it has been
    # cost_core.cli since, so the module differs and the arguments, which are
    # what the numbers depend on, must not.
    assert golden["cmd"][:4] == ["<python>", "-m", "cli.ce_core_cli", "fit-lots"]
    assert golden["cmd"][4:] == GS.CLI_RUNS[run] + ["--out", f"<workbooks>/{run}"]
    new = GS.run_cli_one(run, GS.CLI_RUNS[run], tmp_path)
    assert new["cmd"][:4] == ["<python>", "-m", "cost_core.cli", "fit-lots"]
    assert new["cmd"][4:] == golden["cmd"][4:]
    golden.pop("cmd")
    new.pop("cmd")
    assert new["returncode"] == golden["returncode"] == 0
    for fname, why in CLI_SHA_SKIP.items():
        assert why
        golden["sha256"].pop(fname, None)
        new["sha256"].pop(fname, None)
    assert set(golden["sha256"]) and all(k.endswith(".csv") for k in golden["sha256"])
    if not CAPTURE_LANE:
        print(f"  [{run}] CSV bytes not compared: {CLI_BYTES_SKIP_REASON}")
        for side in (golden, new):
            side.pop("sha256")
            side.pop("csv_text")
    check(golden, new)


# --------------------------------------------------------------------------
# the refusal texts
# --------------------------------------------------------------------------


def test_error_paths(tmp_path):
    golden = load_golden("error_paths")
    for block, why in APP_ONLY_ERROR_BLOCKS.items():
        assert why
        golden.pop(block, None)
    # the library capture tags its own engine "lib"; the golden side it has to
    # reproduce is the tool's, so the tool's texts are read under that tag
    golden["engine"] = {"lib": golden["engine"][ERROR_ENGINE_SIDE]}
    check(golden, GS.capture_error_paths(tmp_path))
