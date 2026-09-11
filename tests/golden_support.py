"""Shared builders and the policy comparator behind the golden regression test.

Two callers use this module and they must agree to the byte:

* ``tests/test_goldens.py`` rebuilds every captured case from ``tests/goldens/``
  using ``cost_core`` alone and compares the result with the golden.
* the out-of-tree capture script (``baseline/capture_goldens.py``) runs the same
  builders against BOTH the desktop tool's engine and this library, and is what
  wrote those goldens in the first place.

Keeping one implementation is the point: if the capture script and the test
built their frames differently, a passing test would prove nothing. So every
builder that does not need the desktop tool lives here, and the capture script
imports them.

This module never imports ``lot_cost_model``, ``risk`` or ``wbs`` (the desktop
tool). Where a builder has to reach an engine or a roll-up module, the module is
a parameter: ``capture_engine(mod, ...)`` takes it explicitly and the WBS
builders read the module-level ``PROGRAM_MOD``, which defaults to
``cost_core.program`` and which the capture script repoints at the tool's
``wbs`` with :func:`use_program_module`.

Comparator
----------
:func:`compare_with_policy` walks a golden and a freshly computed tree together
and returns one record per disagreeing leaf. It reads ``COMPARE_POLICY.json``
rather than hard-coding tolerances, and it applies, per leaf:

* the stage exclusion list ``exclude_from_step2`` at stage 2, which is the
  stage the suite runs at. Stage 3 exists in the walk but excludes nothing:
  step 3 rebaselined the programme-level percentiles rather than dropping them
  from the comparison. With the documented exception that
  ``iteration_detail`` IS golden for ``app_example_maxiter_1``,
  ``app_example_maxiter_2`` and ``app_example_tol_1e-14``;
* ``ctx.cfg`` compared for exact equality, so a changed engine default is caught
  rather than tolerated (``ctx.cfg.ToolVersion`` is the one excluded key);
* ``rounded_leaves.atol_by_column`` by column name, scoped to the frames the
  policy block names (see ``_BLOCK_SCOPES``), at the policy's both-sides-rounded
  tolerance ``one_unit * (1 + 1e-6) + atol``. Both sides ARE rounded here: the
  golden and the new value come out of the same rounding code;
* ``rounded_leaves.sums_of_rounded`` at ``n_terms * one_unit``, with ``n_terms``
  taken from the golden itself (see ``_SUM_RULES``);

  Both of those limits are ABSOLUTE, as the policy writes them: the general
  rule's ``rtol * max(|new|, |golden|)`` is NOT added on top of them, because
  on a large cell it would swallow the unit whole -- 1e-9 of $1e8 is ten 2-dp
  units, and of the lots ``LC F`` column (1e26) it is 1e19 of them -- and
  "a leaf that moves by more than one unit is a real regression" would stop
  meaning anything. What IS added is :func:`_grid_slack`, four ulps of the
  value (1e-15 relative), which is the policy's own ``(1 + 1e-6)`` factor made
  to work above 1e5 or so, where the float grid is coarser than 1e-6 of a
  decimal unit. See that function for the arithmetic;
* ``rounded_leaves.derived_from_rounded`` at rtol 1e-9. The policy's rule for
  these leaves is "compared at rtol 1e-9" followed by a MANUAL traceability
  instruction (find the one-unit flip in the named base leaf and check the
  derived leaf moved by the same factor or the same absolute amount). A
  comparator cannot carry out the manual half, so this module applies the
  numeric half only -- rtol 1e-9, which is also the default -- and leaves the
  traceability to whoever reads the failure. Recorded here because the policy
  asked for it to be recorded;
* the encoding rules: inside DataFrame record leaves ``null``, ``"nan"`` and
  ``"NA"`` are all a missing value and compare equal, while everywhere else
  (``ctx`` scalars above all) None-versus-NaN is a real difference;
* ``tolerance`` (rtol 1e-9, atol 1e-12) everywhere else.

Structural differences -- a key or list element on one side only, a length
mismatch, a type mismatch -- are mismatches, never silently skipped.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

import cost_core.program as _COST_CORE_PROGRAM
from cost_core import lots as LOTS
from cost_core import lotmodel as LIB
from cost_core.lotmodel import enrich as ENRICH
from cost_core.lotmodel.config import SETTINGS

#: Repository root (``tests/`` sits directly under it).
LIB_ROOT = Path(__file__).resolve().parent.parent

PY = sys.executable

#: Roll-up module the WBS builders construct elements and programmes with.
#: Defaults to the library. The capture script repoints it at the desktop
#: tool's ``wbs`` so both sides are built by this one implementation.
PROGRAM_MOD = _COST_CORE_PROGRAM


def use_program_module(mod) -> None:
    """Point the WBS builders at ``mod`` (``cost_core.program`` or the tool's
    ``wbs``). Both expose Element / Program / roll_up with the same shape."""
    global PROGRAM_MOD
    PROGRAM_MOD = mod


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def jsonable(obj):
    """Convert numpy / pandas objects to plain JSON types at full precision.

    None -> null; float NaN -> "nan"; pd.NA -> "NA"; pd.NaT -> "NaT";
    +/-inf -> "inf" / "-inf". Everything else round-trips exactly.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(obj, str):
        return obj
    if obj is pd.NA:
        return "NA"
    if obj is pd.NaT:
        return "NaT"
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()] if obj.dtype != object else [jsonable(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return frame(obj)
    if isinstance(obj, pd.Series):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return repr(obj)


def frame(df: pd.DataFrame) -> dict:
    """A DataFrame as columns + list-of-dict records, full precision.

    Records are built from each column's own .tolist() (positional, so
    duplicate column names survive) rather than iterrows(), which would
    upcast every all-numeric row to float64 and lose int-ness.
    """
    cols = [str(c) for c in df.columns]
    col_vals = [df.iloc[:, j].tolist() for j in range(df.shape[1])]
    records = [{cols[j]: jsonable(col_vals[j][i]) for j in range(len(cols))} for i in range(len(df))]
    return {
        "columns": cols,
        "dtypes": {cols[j]: str(df.iloc[:, j].dtype) for j in range(len(cols))},
        "shape": list(df.shape),
        "records": records,
    }


def to_json_tree(obj):
    """jsonable() followed by a JSON round trip, so the new side is compared in
    exactly the representation the golden was read back in (tuples become
    lists, integer dict keys become strings)."""
    return json.loads(json.dumps(jsonable(obj), allow_nan=False))


#: The encoded spellings of a missing value (see COMPARE_POLICY "encoding").
MISSING_TOKENS = (None, "nan", "NA", "NaT")

#: pandas dtypes that only exist on the capture lane (pandas 3), mapped to what
#: an older pandas calls the same thing, so a rebuilt input frame keeps working.
_DTYPE_FALLBACK = {"str": "object", "string": "object"}


def _unjson(value):
    """Inverse of jsonable() for a scalar leaf."""
    if value == "nan":
        return float("nan")
    if value == "NA":
        return pd.NA
    if value == "NaT":
        return pd.NaT
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return value


def frame_from_json(fj: dict) -> pd.DataFrame:
    """Rebuild a DataFrame from what frame() wrote, dtypes included.

    Duplicate column names cannot survive the records encoding and do not occur
    in any input frame the goldens carry.
    """
    cols = list(fj["columns"])
    recs = fj.get("records") or []
    df = pd.DataFrame({c: [_unjson(r.get(c)) for r in recs] for c in cols}, columns=cols)
    for col, want in (fj.get("dtypes") or {}).items():
        if col not in df.columns:
            continue
        want = _DTYPE_FALLBACK.get(want, want)
        try:
            df[col] = df[col].astype(want)
        except (TypeError, ValueError):
            pass  # a dtype this pandas cannot spell; the values are what matter
    return df


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(data), indent=1, allow_nan=False, sort_keys=False)
    # newline="\n": Path.write_text on Windows would otherwise turn every "\n"
    # into "\r\n" and the byte-identity check would hold on this machine only
    path.write_text(text, encoding="utf-8", newline="\n")


def attempt(fn, *args, **kwargs):
    """Run fn; return {"ok": value} or {"error": text}. Warnings are recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}", "warnings": [str(w.message) for w in caught]}
    return {"ok": value, "warnings": [str(w.message) for w in caught]}


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_dir(s: str, d, token: str) -> str:
    """Replace every spelling of directory d (backslash, posix, and the
    doubled-backslash form an exception repr carries) with token."""
    d = Path(d)
    for needle in (str(d), d.as_posix(), str(d).replace("\\", "\\\\")):
        s = s.replace(needle, token)
    return s


#: (directory, token) pairs mask_paths() applies, in order. The capture script
#: appends the desktop-tool checkout and its own scratch base; the order it ends
#: up with -- python, lib, app, baseline -- is the order the goldens were
#: written under, and changing it would change masked text.
MASK_DIRS = [(Path(PY), "<python>"), (LIB_ROOT, "<lib>")]


def mask_paths(s: str) -> str:
    """Replace the interpreter and every registered checkout root (backslash,
    posix and repr forms) so no golden carries a machine path."""
    for d, token in MASK_DIRS:
        s = mask_dir(s, d, token)
    return s


#: A mask token and the separator that followed the directory it replaced.
_TOKEN_SEPARATOR = re.compile(r"(<[a-z_]+>)[\\/]")


def neutral_separators(tree):
    """tree with the separator after every mask token written as '/'.

    Masking replaces the directory and keeps the separator after it, so one
    missing file reads ``<error_inputs>\\nope.csv`` in the goldens, which were
    captured on Windows, and ``<error_inputs>/nope.csv`` everywhere else. That
    separator is the platform's spelling of the path, not part of the text a
    test pins, so off the capture platform test_error_paths applies this to
    both sides of its comparison. On the capture platform it is not applied, so
    the separator is still compared there, and it is never applied to a golden
    on disk.
    """
    if isinstance(tree, dict):
        return {k: neutral_separators(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [neutral_separators(v) for v in tree]
    if isinstance(tree, str):
        return _TOKEN_SEPARATOR.sub(r"\1/", tree)
    return tree


# --------------------------------------------------------------------------
# input data
# --------------------------------------------------------------------------

PROVENANCE_ROWS = ("Tool version", "Run timestamp")

#: Summary Items the comparator skips when aligning an Item-keyed frame. It is
#: exactly PROVENANCE_ROWS, the two rows :func:`split_summary` has already
#: stripped out of every golden, and it is deliberately no longer than that.
#:
#: "Rate projection" used to be here on the grounds that it was the desktop
#: tool's own row. That was wrong: ``cost_core.lotmodel.provenance`` emits it
#: too, with the same two texts, and which text it carries says whether the
#: projections are the corrected ones or the ``LegacyRateOmission`` ones -- so
#: it is behaviour, not provenance, and the legacy goldens differ from the rest
#: because of it. It is compared like any other row now.
#:
#: Nor is a row reserved for a "cost_core version" stamp the library might
#: grow. If it grows one, the goldens should go red and a human should decide;
#: a skip written in advance would hide the day it appears.
PROVENANCE_ITEMS = PROVENANCE_ROWS

#: Summary Items that ONE named golden may legitimately lack, with the reason.
#: ``lots_cost_core.json`` was captured off the library at ``d7cdea2``, before
#: ``21cde45`` gave ``provenance()`` its "Rate projection" row, so its 27-row
#: ``fit.summary`` frames meet a 28-row frame today. Nothing else is in this
#: position: the 36 engine goldens and both wbs element summaries carry the row
#: on both sides and compare it. :func:`compare_with_policy` takes this as an
#: argument rather than reading it here, so the allowance is visible at the one
#: call site that needs it.
LOTS_GOLDEN_PREDATES_ITEMS = ("Rate projection",)

#: lot_cost_model.EXAMPLE_ANALOGY / EXAMPLE_ESTIMATE, as literals. The capture
#: script asserts these still equal the tool's own constants; they are repeated
#: here so the test can rebuild the example without importing the tool.
EXAMPLE_ANALOGY_ROWS = [
    ("2015", "5", "857.91"),
    ("2016", "9", "645.57"),
    ("2017", "14", "531.74"),
    ("2018", "22", "437.51"),
    ("2019", "34", "380.10"),
    ("2020", "50", "332.21"),
]
EXAMPLE_ESTIMATE_ROWS = [
    ("2028", "12", "1.15"),
    ("2029", "20", "1.15"),
    ("2030", "30", "1.15"),
    ("2031", "40", "1.15"),
    ("2032", "25", "1.15"),
    ("2033", "10", "1.15"),
]


def app_example_frames():
    """EXAMPLE_ANALOGY / EXAMPLE_ESTIMATE as the frames run_lot_cost_model expects
    (the shape of tests/conftest.py analogy_df / estimate_df)."""
    an = pd.DataFrame(
        {
            "Lot": list(range(1, len(EXAMPLE_ANALOGY_ROWS) + 1)),
            "Lot FY": [int(r[0]) for r in EXAMPLE_ANALOGY_ROWS],
            "Qty": [float(r[1]) for r in EXAMPLE_ANALOGY_ROWS],
            "AUC ($K)": [float(r[2]) for r in EXAMPLE_ANALOGY_ROWS],
        }
    )
    es = pd.DataFrame(
        {
            "Lot": list(range(1, len(EXAMPLE_ESTIMATE_ROWS) + 1)),
            "Lot FY": [int(r[0]) for r in EXAMPLE_ESTIMATE_ROWS],
            "Qty": [float(r[1]) for r in EXAMPLE_ESTIMATE_ROWS],
            "Complexity": [float(r[2]) for r in EXAMPLE_ESTIMATE_ROWS],
        }
    )
    return an, es


def gui_analogy(rows):
    """What LotCostApp._collect_analogy builds from grid rows (lot_cost_model.py:4254-4309):
    Lot FY via parse_float (float), Qty float, blank AUC -> NaN."""
    return pd.DataFrame({
        "Lot": list(range(1, len(rows) + 1)),
        "Lot FY": [float(r[0]) if r[0] else np.nan for r in rows],
        "Qty": [float(r[1]) for r in rows],
        "AUC ($K)": [float(r[2]) if r[2] else np.nan for r in rows],
    })


def gui_estimate(rows):
    """LotCostApp._collect_estimate (lot_cost_model.py:4311-4355)."""
    return pd.DataFrame({
        "Lot": list(range(1, len(rows) + 1)),
        "Lot FY": [float(r[0]) if r[0] else np.nan for r in rows],
        "Qty": [float(r[1]) for r in rows],
        "Complexity": [float(r[2]) if r[2] else np.nan for r in rows],
    })


def gui_overrides(**changes):
    """LotCostApp._collect_overrides at the GUI defaults (lot_cost_model.py:4357-4374,
    defaults from SETTINGS at 2482-2496), with the given fields changed.

    Read off cost_core.lotmodel.config.SETTINGS; the capture script asserts the
    tool's SETTINGS still carries the same values for these seven keys."""
    ov = {
        "CostUnitScale": float(SETTINGS["CostUnitScale"]),
        "TotalScale": float(SETTINGS["TotalScale"]),
        "DefaultCF": float(SETTINGS["DefaultCF"]),
        "TGate": float(SETTINGS["TGate"]),
        "FitPriorUnits": int(SETTINGS["FitPriorUnits"]),
        "FcstPriorUnits": int(SETTINGS["FcstPriorUnits"]),
        "LegacyRateOmission": bool(SETTINGS["LegacyRateOmission"]),
    }
    ov.update(changes)
    return ov


def gui_run_info(**changes):
    """LotCostApp._run_info at the GUI defaults (lot_cost_model.py:4140-4148)."""
    ri = {"RunID": SETTINGS["DefaultRunID"], "Program": SETTINGS["DefaultProgram"],
          "RunLabel": SETTINGS["DefaultRunLabel"], "BaseYear": ""}
    ri.update(changes)
    return ri


# tests/test_equation_conformance.py:24-26
RATE_QTY = [22.0, 18.0, 25.0, 30.0, 30.0, 36.0]
RATE_AUC = [4400.0, 3900.0, 3600.0, 3350.0, 3200.0, 3100.0]
RATE_FY = [2018, 2019, 2020, 2021, 2022, 2023]

# Cost_AI_v1/tests/test_lotmodel.py:33-46
CC_ANALOGY = pd.DataFrame({
    "Lot": [1, 2, 3, 4, 5, 6],
    "Lot FY": [2018, 2019, 2020, 2021, 2022, 2023],
    "Qty": [8, 16, 24, 24, 18, 18],
    "AUC ($K)": [3120.00, 2585.50, 2402.75, 2438.10, 2310.40, 2266.85],
})
CC_ESTIMATE = pd.DataFrame({
    "Lot": list(range(1, 9)),
    "Lot FY": [2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037],
    "Qty": [6, 12, 12, 12, 12, 12, 12, 6],
    "Complexity": [1.0] * 8,
})
CC_RUN_INFO = {"RunID": "R001", "Program": "TEST", "RunLabel": "golden", "BaseYear": ""}

# tests/test_wbs.py:17-19, 22-67
FY_HIST = [2015, 2016, 2017, 2018, 2019, 2020]
FY_BUY = [2028, 2029, 2030, 2031, 2032, 2033]
AIRCRAFT = [12.0, 20.0, 30.0, 40.0, 25.0, 10.0]


def wbs_analogy(qty, auc):
    return pd.DataFrame({
        "Lot": range(1, len(qty) + 1),
        "Lot FY": FY_HIST,
        "Qty": [float(q) for q in qty],
        "AUC ($K)": auc,
    })


def airframe(**kw):
    return PROGRAM_MOD.Element(name="1.1 Airframe",
                               analogy=wbs_analogy([5, 9, 14, 22, 34, 50],
                                                   [857.91, 645.57, 531.74, 437.51, 380.10, 332.21]),
                               quantities=list(AIRCRAFT), complexity=1.15, **kw)


def propulsion(**kw):
    return PROGRAM_MOD.Element(name="1.2 Propulsion",
                               analogy=wbs_analogy([12, 20, 30, 44, 68, 100],
                                                   [402.10, 331.55, 288.90, 254.30, 228.75, 210.40]),
                               quantities=[round(q * 2 * 1.10) for q in AIRCRAFT], complexity=1.0, **kw)


def avionics(**kw):
    return PROGRAM_MOD.Element(name="1.3 Avionics kit",
                               analogy=wbs_analogy([6, 11, 16, 26, 38, 55],
                                                   [610.00, 486.20, 421.30, 366.10, 330.55, 302.80]),
                               quantities=[q + 2 for q in AIRCRAFT], complexity=1.05, **kw)


def test_program():
    return PROGRAM_MOD.Program(name="TEST_PROGRAM", fiscal_years=FY_BUY,
                               elements=[airframe(), propulsion(), avionics()])


def se_pm_program():
    """tests/test_wbs.py:510-525."""
    W = PROGRAM_MOD
    return W.Program(
        name="FULL_WBS", fiscal_years=FY_BUY,
        elements=[
            airframe(), propulsion(), avionics(),
            W.factor_of("1.4 Systems Engineering", 0.08),
            W.factor_of("1.5 Program Management", 0.05,
                        basis=["1.1 Airframe", "1.2 Propulsion", "1.3 Avionics kit",
                               "1.4 Systems Engineering"]),
            W.flat_amount("1.6 Tooling", [8e6, 4e6, 0, 0, 0, 0]),
        ],
    )


def late_avionics():
    """tests/test_wbs.py:158-176, 498-507: zero-quantity lots keep their place."""
    late = avionics()
    late.quantities = [0.0, 0.0] + list(AIRCRAFT[2:])
    return late


def extra_wbs_programs() -> dict:
    """name -> (Program, note). Deterministic-only wbs fixtures from tests/test_wbs.py."""
    W = PROGRAM_MOD
    dearer = airframe()
    dearer.complexity = 2.30  # tests/test_wbs.py:143-155
    many = []
    for i in range(12):  # tests/test_wbs.py:386-401
        el = airframe()
        el.name = f"1.{i + 1} Element {i + 1}"
        el.quantities = [q + i for q in AIRCRAFT]
        many.append(el)
    nre_amounts = W.phase_total(12e6, 6, "percentages", percentages=[50, 50], lots=[1, 2])  # :832-844
    progs = {
        "ZERO_QTY_LOTS": (W.Program("ZERO_QTY_LOTS", list(FY_BUY), [airframe(), late_avionics()]),
                          "tests/test_wbs.py:158-176 avionics quantities [0,0,30,40,25,10]"),
        "TWO_LOTS_PER_YEAR": (W.Program("TWO_LOTS_PER_YEAR", [2028, 2028, 2029, 2029, 2030, 2030],
                                        [airframe(), propulsion()]),
                              "tests/test_wbs.py:869-880"),
        "GAP_YEARS": (W.Program("GAP_YEARS", [2028, 2029, 2032, 2033, 2034, 2035], [airframe(), propulsion()]),
                      "tests/test_wbs.py:882-890 gap years appear as zero rows"),
        "DOZEN_ELEMENTS": (W.Program("DOZEN_ELEMENTS", list(FY_BUY), many),
                           "tests/test_wbs.py:386-401 twelve airframes with quantities [q+i]"),
        "AIRFRAME_CF_2.30": (W.Program("AIRFRAME_CF_2.30", list(FY_BUY), [dearer, propulsion(), avionics()]),
                             "tests/test_wbs.py:143-155 airframe complexity 2.30"),
        "NRE_PHASED_50_50": (W.Program("NRE_PHASED_50_50", list(FY_BUY),
                                       [airframe(), W.flat_amount("1.6 NRE", nre_amounts)]),
                             "tests/test_wbs.py:832-844 phase_total percentages [50,50] lots [1,2]"),
    }
    # Element.settings (wbs.py:93) is merged over the call's overrides at wbs.py:493-494
    # (cfg.update(element.settings) AFTER cfg = dict(overrides)), so per-element settings win.
    # No test and no other golden sets it.
    n_air_units = int(sum(q for q in [5, 9, 14, 22, 34, 50]))
    progs["ELEMENT_SETTINGS"] = (
        W.Program("ELEMENT_SETTINGS", list(FY_BUY),
                  [airframe(settings={"FitPriorUnits": 10, "FcstPriorUnits": 10 + n_air_units}),
                   propulsion(settings={"TGate": 1.5}),
                   avionics()]),
        "Element.settings: airframe {FitPriorUnits 10, FcstPriorUnits 10+134=144} (moves its total from "
        "66857453.10 to 123887489.31), propulsion {TGate 1.5} (selection unchanged: LC+Rate; cfg.TGate records it); "
        "also rolled with overrides {FcstPriorUnits 40, TGate 2.0} to pin that element settings win")
    for f in (0.6, 1.5):  # tests/test_wbs.py:722-733, 745-781
        progs[f"FULL_WBS_SCALED_{f}"] = (
            W.Program(f"FULL_WBS_SCALED_{f}", list(FY_BUY),
                      [W._scale_element(e, f) for e in se_pm_program().elements]),
            f"tests/test_wbs.py:722-781 _scale_element({f}) on every FULL_WBS element")
    return progs


def phase_total_cases() -> dict:
    """tests/test_wbs.py:787-844: every phasing profile."""
    W = PROGRAM_MOD
    return {
        "even_12e6_6": attempt(W.phase_total, 12e6, 6),
        "single_lot2": attempt(W.phase_total, 12e6, 6, "single", lots=[2]),
        "even_lots_1_2": attempt(W.phase_total, 12e6, 6, "even", lots=[1, 2]),
        "percentages_40_30_20_10": attempt(W.phase_total, 10e6, 6, "percentages",
                                           percentages=[40, 30, 20, 10], lots=[1, 2, 3, 4]),
        "percentages_50_50_lots_1_2": attempt(W.phase_total, 12e6, 6, "percentages",
                                              percentages=[50, 50], lots=[1, 2]),
        "err_percentages_not_100": attempt(W.phase_total, 10e6, 6, "percentages",
                                           percentages=[40, 30, 20], lots=[1, 2, 3]),
        "err_lot_outside": attempt(W.phase_total, 1e6, 6, "single", lots=[9]),
        "err_unknown_profile": attempt(W.phase_total, 1e6, 6, "sometime"),
        "err_negative": attempt(W.phase_total, -1.0, 6),
    }


def rate_selected_frames():
    """A series the summary selects 'Rate' on (lot_cost_model.py:1032-1046 branch 3).

    Geometric lot sizes with a pure rate effect, c=-0.25, lognormal(0, 0.02)
    noise from default_rng(2) drawn in list order.
    """
    qty = [5, 10, 20, 40, 80, 160]
    rng = np.random.default_rng(2)
    auc = [3000.0 * q ** -0.25 * rng.lognormal(0, 0.02) for q in qty]
    an = pd.DataFrame({"Lot": range(1, 7), "Lot FY": range(2010, 2016),
                       "Qty": [float(q) for q in qty], "AUC ($K)": auc})
    es = pd.DataFrame({"Lot": [1, 2], "Lot FY": [2030, 2031], "Qty": [10.0, 20.0], "Complexity": [1.0, 1.0]})
    return an, es


def rate_driven_test_series():
    """tests/test_lots.py:385-388 exactly: default_rng(3), lognormal(0, 0.01) drawn in list order."""
    rng = np.random.default_rng(3)
    quantities = [40, 5, 38, 6, 36, 8, 34, 10]
    costs = [q * (3_000.0 * q ** -0.35) * rng.lognormal(0, 0.01) for q in quantities]
    return quantities, costs


# --------------------------------------------------------------------------
# per-run capture
# --------------------------------------------------------------------------

MODELS = {"LC": "mdl_lc", "Rate": "mdl_rt", "LC+Rate": "mdl_lcr"}
K_OF = {"LC": 2, "Rate": 2, "LC+Rate": 3}
RATE_IDX = {"LC": None, "Rate": 1, "LC+Rate": 2}
INTERVAL_LEVELS = (0.80, 0.50, 0.95)


def derived_stats(ctx: dict) -> dict:
    """SEE/R2/Adj/CV/MAPE/Bias/AICc/t exactly as generate_analyst_summary computes
    them (lot_cost_model.py:950-1009), but unformatted."""
    n_keep = ctx["n_keep"]
    fit_c = np.asarray(ctx["fit_c"], dtype=float)
    ln_y = np.log(fit_c)
    sst = float(np.sum((ln_y - ln_y.mean()) ** 2))
    out = {"sst_log": sst}
    for name, key in MODELS.items():
        m = ctx.get(key)
        if m is None:
            out[name] = None
            continue
        k = K_OF[name]
        sse0 = m["SSE"]
        dfe = n_keep - k
        see = np.sqrt(sse0 / dfe) if dfe > 0 else None
        r2 = (1.0 - sse0 / sst) if sst > 0 else None
        adj = (1.0 - (1.0 - r2) * (n_keep - 1) / dfe) if (r2 is not None and dfe > 0) else None
        cv = np.sqrt(np.exp(see * see) - 1.0) if see is not None else None
        fit_u = np.exp(np.asarray(m["Fitted"]))
        mape = float(np.mean(np.abs(fit_c - fit_u) / fit_c))
        bias = float(np.mean(fit_c / fit_u - 1.0))
        kp = k + 1
        sseg = max(sse0, 1e-30)
        aicc = (n_keep * np.log(sseg / n_keep) + 2 * kp + 2 * kp * (kp + 1) / (n_keep - kp - 1)) \
            if (n_keep - kp - 1 > 0) else None
        ridx = RATE_IDX[name]
        sec = see * np.sqrt(m["InvDiag"][ridx]) if (ridx is not None and see is not None) else None
        tc = (m["Beta"][ridx] / sec) if (sec is not None and sec >= 1e-15) else None
        # t statistic for every coefficient from the ols SEs
        t_all = [(b / s) if (s is not None and not np.isnan(s) and s > 0) else None
                 for b, s in zip(m["Beta"], m["SE"])]
        out[name] = {"SEE": see, "R2": r2, "AdjR2": adj, "CV": cv, "MAPE": mape, "Bias": bias,
                     "AICc": aicc, "t_rate": tc, "se_rate": sec, "t_all": t_all}
    aiccs = {n: out[n]["AICc"] for n in MODELS if out.get(n) and out[n]["AICc"] is not None}
    if aiccs:
        best = min(aiccs.values())
        out["dAICc"] = {n: aiccs[n] - best for n in aiccs}
    return out


def midpoint_block(mod, ctx: dict, projections: pd.DataFrame) -> dict:
    """Unrounded midpoints and unit costs recomputed with the module's own lmp_func."""
    lmp = mod.lmp_func
    fit_q = np.asarray(ctx["fit_q"], dtype=float)
    se = ctx["fit_se"]
    cfg = ctx["cfg"]
    out = {}
    for name, bkey in (("LC", "b_lc"), ("Rate", "b_rt"), ("LC+Rate", "b_br")):
        b = ctx.get(bkey)
        if b is None or (isinstance(b, float) and np.isnan(b)):
            out[f"analogy_midpoints_{name}"] = None
        else:
            out[f"analogy_midpoints_{name}"] = [lmp(s["S"], s["E"], q, b) for s, q in zip(se, fit_q)]
    S = projections["First Unit in Lot"].to_numpy(dtype=float)
    E = projections["Last Unit in Lot"].to_numpy(dtype=float)
    Q = projections["Lot Quantity"].to_numpy(dtype=float)
    CF = projections["Complexity Factor"].to_numpy(dtype=float)
    scale = float(cfg["CostUnitScale"])
    tot = float(cfg["TotalScale"])
    legacy = bool(cfg.get("LegacyRateOmission", False))
    unr = {}
    if not np.isnan(ctx["t1_lc"]):
        mid = np.array([lmp(s, e, q, ctx["b_lc"]) for s, e, q in zip(S, E, Q)])
        uc = ctx["t1_lc"] * mid ** ctx["b_lc"] * scale
        unr["LC"] = {"midpoint": mid, "unit_cost": uc, "lot_cost_before": uc * Q * tot,
                     "lot_cost_after": uc * Q * tot * CF}
    if not np.isnan(ctx["t1_rt"]):
        mid = np.array([lmp(s, e, q, ctx["b_rt"]) for s, e, q in zip(S, E, Q)])
        uc = (ctx["t1_rt"] * mid ** ctx["b_rt"] * scale) if legacy else (ctx["t1_rt"] * Q ** ctx["b_rt"] * scale)
        unr["Rate"] = {"midpoint": mid, "unit_cost": uc, "lot_cost_before": uc * Q * tot,
                       "lot_cost_after": uc * Q * tot * CF}
    if not np.isnan(ctx["t1_br"]):
        mid = np.array([lmp(s, e, q, ctx["b_br"]) for s, e, q in zip(S, E, Q)])
        rf = 1.0 if legacy else Q ** ctx["c_br"]
        uc = ctx["t1_br"] * mid ** ctx["b_br"] * rf * scale
        unr["LC+Rate"] = {"midpoint": mid, "unit_cost": uc, "lot_cost_before": uc * Q * tot,
                          "lot_cost_after": uc * Q * tot * CF}
    out["forecast_unrounded"] = unr
    return out


def split_summary(summary: pd.DataFrame):
    """Golden rows (everything except timestamp/tool version) and the stripped rows."""
    mask = summary["Item"].isin(PROVENANCE_ROWS)
    return summary.loc[~mask].reset_index(drop=True), summary.loc[mask].reset_index(drop=True)


def split_ctx(ctx: dict):
    """ctx without the iteration bookkeeping, plus that bookkeeping on its own.

    Iter and Delta describe how the fixed-point loop ran, not what it found;
    re-wrapping the loop around fitting.fit changes them while the converged
    coefficients stay put. Converged stays in ctx (it drives Fit Status).
    """
    ctx_json = {}
    detail = {}
    for k, v in ctx.items():
        if k in ("mdl_lc", "mdl_rt", "mdl_lcr") and isinstance(v, dict) and "Iter" in v:
            m = dict(v)
            detail[k] = {"Iter": m.pop("Iter"), "Delta": m.pop("Delta"), "Converged": m.get("Converged")}
            ctx_json[k] = m
        else:
            ctx_json[k] = v
    return ctx_json, detail


def selected_model_of(summary: pd.DataFrame):
    try:
        return ENRICH.selected_model_name(summary)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def buy_json(buy, **extra) -> dict:
    tot = np.asarray(buy.totals, dtype=float)
    return {
        "n_iter": buy.n_iter, "seed": buy.seed, "lot_correlation": buy.lot_correlation,
        "dof": buy.dof, "n_history_lots": buy.n_history_lots, "clipped": buy.clipped,
        "point_estimate": buy.point_estimate,
        "percentiles": {f"p{p}": float(np.percentile(tot, p)) for p in (5, 10, 25, 50, 80, 90, 95)},
        "mean": buy.mean, "std": buy.std, "cv": buy.cv,
        "point_estimate_percentile": buy.point_estimate_percentile,
        "totals_first_10": tot[:10], "totals_sum": float(tot.sum()),
        "per_lot_mean": np.asarray(buy.per_lot).mean(axis=0),
        "summary": buy.summary(), **extra,
    }


def methods_json(mc) -> dict:
    return {
        "model": mc.model, "frame": mc.frame,
        "log_residual_variance": mc.log_residual_variance,
        "theoretical_factor": mc.theoretical_factor, "smearing_factor": mc.smearing_factor,
        "mupe_over_ols": mc.mupe_over_ols, "zmpe_over_ols": mc.zmpe_over_ols,
        "percent_understated": mc.percent_understated,
    }


def enrich_block(ctx: dict, projections: pd.DataFrame, summary: pd.DataFrame, selected) -> dict:
    """projection_intervals / simulate_buy / influence / methods for every fitted
    model, plus enrich_run (the assembled public layer) on the selected model."""
    out = {"selected_model": selected}
    for name, key in MODELS.items():
        if ctx.get(key) is None:
            out[name] = None
            continue
        blk = {}
        for level in INTERVAL_LEVELS:
            blk[f"projection_intervals_{level:.2f}"] = attempt(ENRICH.projection_intervals, ctx, projections, name,
                                                               level=level)
        r = attempt(ENRICH.simulate_buy, ctx, projections, name, n_iter=20000, seed=11, lot_correlation=0.30)
        if "ok" in r:
            r["ok"] = buy_json(r["ok"])
        blk["simulate_buy_20000_seed11_rho0.30"] = r
        blk["influence_diagnostics"] = attempt(ENRICH.influence_diagnostics, ctx, name)
        r = attempt(ENRICH.compare_fitting_methods, ctx, name)
        if "ok" in r:
            r["ok"] = methods_json(r["ok"])
        blk["compare_fitting_methods"] = r
        out[name] = blk
    # Cost_AI_v1/tests/test_lotmodel.py:495-505 and 519-525
    for label, kw in (("enrich_run_n5000_seed0", {"n_iter": 5000, "seed": 0}),
                      ("enrich_run_n2000_seed0", {"n_iter": 2000, "seed": 0})):
        r = attempt(ENRICH.enrich_run, ctx, projections, summary, **kw)
        if "ok" in r:
            en = r["ok"]
            r["ok"] = {"kwargs": kw, "selected_model": en.selected_model,
                       "sheets": en.sheets(), "warnings_raised": list(en.warnings_raised),
                       "risk": buy_json(en.risk), "methods": methods_json(en.methods)}
        out[label] = r
    return out


def test_lotmodel_fixtures(ctx: dict, projections: pd.DataFrame) -> dict:
    """The simulate_buy calls Cost_AI_v1/tests/test_lotmodel.py makes on the LC
    model, at their own seeds, and the one-lot slice of :449-455."""
    specs = {
        "LC_n20000_seed1": dict(n_iter=20000, seed=1),  # :438
        "LC_n30000_seed3_rho0.0": dict(n_iter=30000, seed=3, lot_correlation=0.0),  # :463-466
        "LC_n30000_seed3_rho0.6": dict(n_iter=30000, seed=3, lot_correlation=0.6),
        "LC_n5000_seed42": dict(n_iter=5000, seed=42),  # :474-478
        "LC_n5000_seed43": dict(n_iter=5000, seed=43),
        "LC_n2000_seed1": dict(n_iter=2000, seed=1),  # :483 (per_lot shape (2000, n_lots))
    }
    out = {"_note": "tests/test_lotmodel.py simulate_buy seeds on the LC model; per_lot_shape pins :483",
           "simulate_buy_variants": {}}
    for label, kw in specs.items():
        r = attempt(ENRICH.simulate_buy, ctx, projections, "LC", **kw)
        if "ok" in r:
            b = r["ok"]
            r["ok"] = buy_json(b, kwargs=kw, per_lot_shape=list(np.asarray(b.per_lot).shape))
        out["simulate_buy_variants"][label] = r
    if len(projections) >= 4:  # :449-455 on projections.iloc[[3]]
        one = projections.iloc[[3]].copy()
        blk = {"row": one,
               "projection_intervals_0.80": attempt(ENRICH.projection_intervals, ctx, one, "LC", level=0.80)}
        r = attempt(ENRICH.simulate_buy, ctx, one, "LC", n_iter=120_000, seed=2)
        if "ok" in r:
            b = r["ok"]
            tot = np.asarray(b.totals, dtype=float)
            r["ok"] = buy_json(b, per_lot_shape=list(np.asarray(b.per_lot).shape),
                               p10=float(np.percentile(tot, 10)), p90=float(np.percentile(tot, 90)))
        blk["simulate_buy_n120000_seed2"] = r
        out["one_lot_iloc3"] = blk
    return out


def capture_engine(mod, analogy, estimate, overrides, run_info):
    """One engine case through `mod` (the tool's lot_cost_model or cost_core.lotmodel).

    Returns (data, provenance_rows, (projections, ctx, summary, chart)). `data`
    is the golden dict minus the capture_warnings key the caller adds.
    """
    projections, ctx = mod.run_lot_cost_model(analogy, estimate, dict(overrides) if overrides else None)
    summary = mod.generate_analyst_summary(ctx, dict(run_info))
    chart = mod.generate_fit_chart_data(ctx)
    golden_summary, prov = split_summary(summary)
    selected = selected_model_of(summary)
    ctx_json, iteration_detail = split_ctx(ctx)
    data = {
        "inputs": {"analogy": analogy, "estimate": estimate, "overrides": overrides, "run_info": run_info},
        "ctx": ctx_json,
        # The _note below still names solve_model, which step 2 deleted; the
        # loop is models.solve_lot_model now. The string is left alone because
        # /iteration_detail/_note is compared exactly for the three
        # ITERATION_GOLDEN_CASES, and rewriting a golden to correct a name
        # would be a rebaseline for a spelling.
        "iteration_detail": {"_note": "Iter/Delta of the midpoint fixed-point loop (solve_model); excluded from "
                                      "the 1e-9 regression except for the maxiter_*/tol_* cases, which pin the "
                                      "bookkeeping itself (see COMPARE_POLICY.json)", **iteration_detail},
        "derived_stats": derived_stats(ctx),
        "midpoints": midpoint_block(mod, ctx, projections),
        "projections": projections,
        "projections_column_sums": {c: float(projections[c].sum()) for c in projections.columns
                                    if pd.api.types.is_numeric_dtype(projections[c])},
        "summary": golden_summary,
        "selected_model": selected,
        "fit_chart_data": chart,
        "fit_status": str(projections["Fit Status"].iloc[0]),
        "enrich": enrich_block(ctx, projections, summary, selected),
    }
    return data, prov, (projections, ctx, summary, chart)


def capture_engine_lib(analogy, estimate, overrides, run_info):
    """capture_engine on cost_core.lotmodel: the exact dict shape of an
    engine_<case>.lib.json golden (minus capture_warnings, which the capture
    script adds from the warnings raised around the whole call)."""
    return capture_engine(LIB, analogy, estimate, overrides, run_info)


# --------------------------------------------------------------------------
# app vs lib comparison (the capture script's own cross-check)
# --------------------------------------------------------------------------

RTOL = 1e-9
ATOL = 1e-12


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _item_keys(records):
    """Item-keyed alignment that survives blank and duplicate Items: the k-th
    occurrence of an Item gets key 'Item#k' (k>0), a blank Item gets '#k'."""
    seen, keys = {}, []
    for r in records:
        item = r.get("Item", "")
        item = item if isinstance(item, str) else str(item)
        k = seen.get(item, 0)
        seen[item] = k + 1
        keys.append(item if (item and k == 0) else f"{item}#{k}")
    return keys


def compare(a, b, path="", acc=None, rtol=RTOL, atol=ATOL):
    """Walk two jsonable structures; collect max relative diff over numeric leaves
    and every non-numeric mismatch. Frames with an Item column are aligned on
    Item (blank/duplicate Items keyed by occurrence). Numeric leaves fail when
    |a-b| > atol + rtol*max(|a|,|b|); the raw max relative difference is
    reported separately."""
    if acc is None:
        acc = {"rtol": rtol, "atol": atol, "max_rel": 0.0, "max_rel_at": None, "n_numeric": 0,
               "mismatches": [], "diffs_over_tol": []}
    if _is_num(a) and _is_num(b):
        acc["n_numeric"] += 1
        d = abs(a - b)
        den = max(abs(a), abs(b))
        rel = 0.0 if d == 0 else (d / den if den > 0 else float("inf"))
        if rel > acc["max_rel"]:
            acc["max_rel"], acc["max_rel_at"] = rel, path
        if d > atol + rtol * den:
            acc["diffs_over_tol"].append({"path": path, "app": a, "lib": b, "rel": rel, "abs": d})
        return acc
    if a is None and b is None:
        return acc
    if isinstance(a, dict) and isinstance(b, dict):
        if "records" in a and "records" in b and "Item" in a.get("columns", []) and "Item" in b.get("columns", []):
            ra = dict(zip(_item_keys(a["records"]), a["records"]))
            rb = dict(zip(_item_keys(b["records"]), b["records"]))
            for k in sorted(set(ra) | set(rb)):
                if k in ra and k in rb:
                    compare(ra[k], rb[k], f"{path}/[Item={k}]", acc, rtol, atol)
                else:
                    acc["mismatches"].append({"path": f"{path}/[Item={k}]", "only_in": "app" if k in ra else "lib"})
            return acc
        for k in sorted(set(a) | set(b)):
            if k in ("dtypes", "warnings", "_note"):
                continue
            if k in a and k in b:
                compare(a[k], b[k], f"{path}/{k}", acc, rtol, atol)
            else:
                acc["mismatches"].append({"path": f"{path}/{k}", "only_in": "app" if k in a else "lib"})
        return acc
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["mismatches"].append({"path": path, "len_app": len(a), "len_lib": len(b)})
            longer, side = (a, "app") if len(a) > len(b) else (b, "lib")
            for i in range(min(len(a), len(b)), len(longer)):
                acc["mismatches"].append({"path": f"{path}[{i}]", "only_in": side,
                                          "value": json.dumps(longer[i])[:200]})
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", acc, rtol, atol)
        return acc
    if a != b:
        acc["mismatches"].append({"path": path, "app": a if not isinstance(a, str) else a[:200],
                                  "lib": b if not isinstance(b, str) else b[:200]})
    return acc


# --------------------------------------------------------------------------
# wbs / programme roll-up
# --------------------------------------------------------------------------


def element_json(r, prov_sink: dict, prov_key: str, include_ctx=True) -> dict:
    d = {
        "name": r.name, "kind": r.kind, "model": r.model, "total": r.total, "by_lot": np.asarray(r.by_lot),
        "n_lots_fitted": r.n_lots_fitted, "basis": list(r.basis),
        "projections": r.projections,
    }
    if r.kind == "fitted":
        gs, prov = split_summary(r.summary)
        d["summary"] = gs
        prov_sink[f"{prov_key}.{r.name}"] = prov
        if include_ctx:
            ctx_json, detail = split_ctx(r.ctx)
            d["ctx"] = ctx_json
            d["iteration_detail"] = detail
            d["derived_stats"] = derived_stats(r.ctx)
    return d


def pct(tot):
    tot = np.asarray(tot, dtype=float)
    return {f"p{p}": float(np.percentile(tot, p)) for p in (5, 10, 25, 50, 80, 90, 95)} | {
        "mean": float(tot.mean()), "std": float(tot.std(ddof=1)), "n": int(tot.size)}


def program_inputs(program) -> dict:
    return {"program": program.name, "fiscal_years": program.fiscal_years,
            "elements": [{"name": e.name, "kind": e.kind, "quantities": e.quantities, "complexity": e.complexity,
                          "factor": e.factor, "basis": e.basis, "amounts": e.amounts, "settings": e.settings,
                          "analogy": e.analogy} for e in program.elements]}


def _same_number(x):
    """`x` as the number it already is: a JSON integer stays an integer.

    program_from_inputs is the inverse of program_inputs, so the Program it
    rebuilds has to hand the inputs block back with the types the golden holds
    -- the tool's fixtures wrote lot quantities as ints. Widening them to
    floats on the way in is a real difference in the echoed inputs, and the
    comparator reports one.
    """
    if isinstance(x, bool) or x is None:
        return x
    return x if isinstance(x, int) else float(x)


def program_from_inputs(inputs: dict, mod=None):
    """Rebuild a Program from what program_inputs() wrote.

    The inverse of program_inputs: it takes the JSON block straight out of a
    wbs golden -- analogy frames included -- and hands back a live Program, so
    the regression test never restates a fixture the golden already carries.
    """
    W = mod or PROGRAM_MOD
    elements = []
    for e in inputs["elements"]:
        analogy = e.get("analogy")
        if isinstance(analogy, dict) and "records" in analogy:
            analogy = frame_from_json(analogy)
        complexity = e.get("complexity")
        if isinstance(complexity, list):
            complexity = [_same_number(x) for x in complexity]
        elif complexity is not None:
            complexity = _same_number(complexity)
        quantities = e.get("quantities")
        if quantities is not None:
            quantities = [_same_number(q) for q in quantities]
        amounts = e.get("amounts")
        if amounts is not None:
            amounts = [_same_number(a) for a in amounts]
        kw = {"name": e["name"], "kind": e["kind"], "analogy": analogy, "quantities": quantities,
              "settings": dict(e.get("settings") or {}),
              "factor": _same_number(e.get("factor")),
              "basis": None if e.get("basis") is None else list(e["basis"]),
              "amounts": amounts}
        if complexity is not None:
            kw["complexity"] = complexity
        elements.append(W.Element(**kw))
    return W.Program(name=inputs["program"], fiscal_years=[int(y) for y in inputs["fiscal_years"]],
                     elements=elements)


def det_block(det, prov_sink: dict, prov_key: str, *, influence=True) -> dict:
    """Everything deterministic that roll_up(simulate=False) and its views produce."""
    W = PROGRAM_MOD
    out = {
        "total": det.total, "by_lot": det.by_lot,
        "order": [e.name for e in det.order],
        "elements": [element_json(r, prov_sink, prov_key) for r in det.elements],
        "element_summary": W.element_summary(det),
        "program_summary": W.program_summary(det),
        "by_fiscal_year": W.by_fiscal_year(det),
        "notes": det.notes, "warnings": det.warnings,
    }
    if influence:
        out["influence_table"] = attempt(W.influence_table, det)
    return out


def capture_wbs(program, workbooks, prov_sink: dict) -> dict:
    """The roll-up goldens for one programme.

    ``workbooks`` may be None, in which case the three workbooks are not
    written and the "workbooks" block is left out. The regression test passes
    None: it regresses numbers, and the workbook check is compare_workbooks.py's
    job.
    """
    W = PROGRAM_MOD
    out = program_inputs(program)
    det = W.roll_up(program, simulate=False)
    out["deterministic"] = det_block(det, prov_sink, f"wbs_{program.name}.deterministic")
    sens = attempt(W.buy_profile_sensitivity, program)
    out["deterministic"]["buy_profile_sensitivity"] = sens
    # tests/test_wbs.py:459-465 (0.8,1.0,1.25), :489 (0.5,2.0), :494-497 reference_element (error path;
    # the positive path is never run by a test), and the overrides argument the GUI always passes
    # (lot_cost_model.py:2840-2842 passes self._collect_overrides())
    out["deterministic"]["buy_profile_sensitivity_variants"] = {
        "factors_0.8_1.0_1.25": attempt(W.buy_profile_sensitivity, program, factors=(0.8, 1.0, 1.25)),
        "factors_0.5_2.0": attempt(W.buy_profile_sensitivity, program, factors=(0.5, 2.0)),
        "reference_1.2_Propulsion_0.8_1.0_1.25": attempt(W.buy_profile_sensitivity, program,
                                                         factors=(0.8, 1.0, 1.25),
                                                         reference_element="1.2 Propulsion"),
        "overrides_FcstPriorUnits_40_0.8_1.0_1.25": attempt(W.buy_profile_sensitivity, program,
                                                            factors=(0.8, 1.0, 1.25),
                                                            overrides={"FcstPriorUnits": 40}),
        "overrides_gui_defaults_0.8_1.0_1.25": attempt(W.buy_profile_sensitivity, program,
                                                       factors=(0.8, 1.0, 1.25), overrides=gui_overrides()),
        "err_reference_not_an_element": attempt(W.buy_profile_sensitivity, program, factors=(1.0,),
                                                reference_element="1.9 Nope"),
    }
    # the GUI passes _collect_overrides() as the second positional argument (lot_cost_model.py:3033-3035)
    out["overrides"] = {}
    for label, ov in (("LegacyRateOmission_True", {"LegacyRateOmission": True}),
                      ("FcstPriorUnits_40", {"FcstPriorUnits": 40})):
        r = attempt(W.roll_up, program, ov, simulate=False)
        out["overrides"][label] = {"overrides": ov,
                                   **({"ok": det_block(r["ok"], prov_sink, f"wbs_{program.name}.overrides.{label}",
                                                       influence=False)} if "ok" in r else {"error": r["error"]}),
                                   "warnings": r["warnings"]}

    sim = W.roll_up(program, n_iter=8000, seed=11)
    standalone = {}
    for r in sim.elements:
        if r.kind == "fitted":
            b = ENRICH.simulate_buy(r.ctx, r.projections, r.model, n_iter=8000, seed=11, lot_correlation=0.30)
            standalone[r.name] = {"seed": 11, "n_iter": 8000, "lot_correlation": 0.30,
                                  "point_estimate": b.point_estimate, "percentiles": pct(b.totals),
                                  "cv": b.cv, "point_estimate_percentile": b.point_estimate_percentile,
                                  "clipped": b.clipped}
    out["element_standalone_simulate_buy"] = standalone
    # The block below is what step 3 changed on purpose (lognormal handoff ->
    # raw draws). It was rebaselined to the new values rather than excluded, so
    # it is still compared at the general rule like everything else.
    out["program_level_percentiles"] = {
        "_note": "rebaselined by step 3 (raw-draw handoff); compared at the general rule",
        "n_iter": sim.n_iter, "seed": 11, "correlation": sim.correlation,
        "p50": sim.p50, "p80": sim.p80, "p90": sim.p90, "mean": sim.mean, "std": sim.std, "cv": sim.cv,
        "point_percentile": sim.point_percentile,
        "independence_understates_sd_by": sim.independence_understates_sd_by,
        "variance_ratio_analytic": sim.variance_ratio_analytic,
        "p80_understatement": sim.p80_understatement,
        "reserve_understatement": sim.reserve_understatement,
        "scurve": sim.scurve, "tornado": sim.tornado,
        "element_in_program_totals": {r.name: pct(r.totals) for r in sim.elements if r.totals is not None},
        "element_summary_with_risk": W.element_summary(sim),
        "program_summary_with_risk": W.program_summary(sim),
        "notes": sim.notes, "warnings": sim.warnings,
    }
    # the point-estimate side of the simulated roll-up must equal the deterministic one
    out["simulated_run_point_total"] = sim.total
    if workbooks is None:
        return out
    # workbooks: risk variant (as before), plain variant with the sensitivity
    # frame, and plain variant with sensitivity=None (what the GUI writes unless
    # the analyst ran the sensitivity first, lot_cost_model.py:3134-3136)
    workbooks = Path(workbooks)
    sens_frame = sens["ok"] if "ok" in sens else None
    out["workbooks"] = {}
    for label, result, sframe in (("risk", sim, sens_frame), ("plain", det, sens_frame), ("plain_nosens", det, None)):
        wb = workbooks / (f"wbs_{program.name}.xlsx" if label == "risk" else f"wbs_{program.name}_{label}.xlsx")
        r = attempt(W.save_program_workbook, str(wb), result, sframe)
        out["workbooks"][label] = {"file": wb.name, "error": r.get("error"),
                                   "sensitivity_sheet": sframe is not None,
                                   "risk_sheets": label == "risk"}
    out["workbooks"]["_note"] = ("wbs_<program>.xlsx carries Program_SCurve, Program_Tornado and the risk rows of "
                                 "Program_Summary / Program_Elements, which step 3 changes on purpose; regress the "
                                 "_plain / _plain_nosens variants at 1e-9 and use the risk variant only for "
                                 "determinism (compare_workbooks.py --skip-sheets Program_SCurve,Program_Tornado "
                                 "still sees the risk rows)")
    return out


def capture_wbs_extras(prov_sink: dict) -> dict:
    W = PROGRAM_MOD
    out = {"_note": "deterministic roll_up(simulate=False) only; no workbook, no program-level risk",
           "phase_total": phase_total_cases(), "programs": {}}
    for name, (prog, note) in extra_wbs_programs().items():
        r = attempt(W.roll_up, prog, simulate=False)
        entry = {"note": note, "inputs": program_inputs(prog), "warnings": r["warnings"]}
        if "ok" in r:
            entry["deterministic"] = det_block(r["ok"], prov_sink, f"wbs_extra.{name}")
        else:
            entry["error"] = r["error"]
        if name == "ELEMENT_SETTINGS":
            ov = {"FcstPriorUnits": 40, "TGate": 2.0}
            r2 = attempt(W.roll_up, prog, ov, simulate=False)
            entry["overrides_FcstPriorUnits_40_TGate_2.0"] = {
                "overrides": ov, "warnings": r2["warnings"],
                **({"ok": det_block(r2["ok"], prov_sink, f"wbs_extra.{name}.overrides", influence=False)}
                   if "ok" in r2 else {"error": r2["error"]})}
        out["programs"][name] = entry
    # tests/test_wbs.py:498-507: sensitivity on a program with a zero-quantity lot
    out["buy_profile_sensitivity_zero_qty_factor_1.4"] = attempt(
        W.buy_profile_sensitivity, W.Program("P", list(FY_BUY), [airframe(), late_avionics()]), factors=(1.4,))
    # tests/test_wbs.py:715-720, 745-750
    out["buy_profile_sensitivity_full_wbs_0.6_1.0_1.5"] = attempt(
        W.buy_profile_sensitivity, se_pm_program(), factors=(0.6, 1.0, 1.5))
    return out


# --------------------------------------------------------------------------
# cost_core.lots
# --------------------------------------------------------------------------

REF_QTY = [8, 16, 24, 24, 18, 18]
REF_AUC = [3120.00, 2585.50, 2402.75, 2438.10, 2310.40, 2266.85]


def reference_series(**kw):
    kw.setdefault("dollar_year", 2026)
    kw.setdefault("program", "TEST")
    return LOTS.LotSeries(quantities=REF_QTY, costs=[q * a for q, a in zip(REF_QTY, REF_AUC)], **kw)


def clean_series(**kw):
    """tests/test_lots.py:42-56."""
    from cost_core.lotmodel.mathx import lmp_func
    t1, b = 5_000.0, np.log2(0.85)
    quantities = [20, 20, 25, 25, 30, 30]
    cursor, costs = 1, []
    for q in quantities:
        mid = lmp_func(cursor, cursor + q - 1, q, b)
        costs.append(t1 * mid ** b * q)
        cursor += q
    kw.setdefault("dollar_year", 2026)
    return LOTS.LotSeries(quantities=quantities, costs=costs, **kw)


def escalate(series_in, rate: float):
    """tests/test_lots.py:616-623."""
    return LOTS.LotSeries(
        quantities=series_in.quantities,
        costs=series_in.costs * np.array([(1.0 + rate) ** i for i in range(series_in.n_lots)]),
        dollar_year=series_in.dollar_year,
    )


def series_json(s) -> dict:
    return {"to_frame": s.to_frame(), "labels": list(s.labels), "program": s.program,
            "cost_basis": s.cost_basis, "first_unit": s.first_unit, "dollar_year": s.dollar_year,
            "quantity_definition": s.quantity_definition,
            "cumulative_average": s.cumulative_average(), "unit_ranges": s.unit_ranges(),
            "constant_dollar_findings": s.check_constant_dollars(warn=False)}


def fit_json(fit) -> dict:
    gs, _ = split_summary(fit.summary)
    ctx_json, detail = split_ctx(fit.ctx)
    return {
        "selected_model": fit.selected_model, "selection_note": fit.selection_note,
        "t1": fit.t1, "b": fit.b, "c": fit.c, "slope": fit.slope, "rate_slope": fit.rate_slope,
        "n_obs": fit.n_obs, "n_params": fit.n_params, "df": fit.df, "sigma": fit.sigma, "cv": fit.cv,
        "r_squared": fit.r_squared, "equation": fit.equation(), "equation_detail": fit.equation_detail(),
        "model_comparison": fit.model_comparison(), "lot_midpoints": fit.lot_midpoints(),
        "projections": fit.projections, "summary": gs, "chart": fit.chart,
        "ctx": ctx_json, "iteration_detail": detail, "derived_stats": derived_stats(fit.ctx),
        "forecast_quantities": list(fit.forecast_quantities), "is_forecast": fit.is_forecast,
        "complexity": fit.complexity,
    }


def report_json(rep, *, simulate_specs=(), plans=(), forecasts=()) -> dict:
    d = {"fit": fit_json(rep.fit), "level": rep.level, "equation": rep.equation()}
    d["per_lot"] = attempt(lambda: rep.per_lot)
    r = attempt(rep.methods)
    if "ok" in r:
        r["ok"] = methods_json(r["ok"])
    d["methods"] = r
    d["influence"] = attempt(rep.influence)
    d["intervals"] = attempt(rep.intervals)
    d["intervals_0.50"] = attempt(rep.intervals, level=0.50)
    d["intervals_0.95"] = attempt(rep.intervals, level=0.95)
    d["curvature"] = attempt(lambda: list(rep.curvature()))
    d["check_curve_shape"] = attempt(rep.check_curve_shape, warn=False)
    d["diagnostics"] = attempt(rep.diagnostics)
    d["summary"] = attempt(rep.summary)
    d["narrative"] = attempt(rep.narrative)
    d["price_lot_plan"] = {}
    for label, q, kw in plans:
        d["price_lot_plan"][label] = attempt(rep.price_lot_plan, q, **kw)
    d["forecast"] = {}
    for label, q, kw in forecasts:
        d["forecast"][label] = attempt(rep.forecast, q, **kw)
    d["simulate"] = {}
    for label, kw in simulate_specs:
        r = attempt(rep.simulate, **kw)
        if "ok" in r:
            b = r["ok"]
            r["ok"] = {"kwargs": kw, "point_estimate": b.point_estimate, "percentiles": pct(b.totals),
                       "cv": b.cv, "point_estimate_percentile": b.point_estimate_percentile,
                       "dof": b.dof, "clipped": b.clipped, "totals_first_10": np.asarray(b.totals)[:10],
                       "summary": b.summary()}
        d["simulate"][label] = r
    return d


def learning_curve_block(scatter: float) -> dict:
    """tests/test_lots.py:270-290 exactly."""
    from cost_core.learning_curve import compare_methods, comparison_table
    rng = np.random.default_rng(4)
    costs = np.array([q * a for q, a in zip(REF_QTY, REF_AUC)], dtype=float)
    lots = LOTS.LotSeries(quantities=REF_QTY, costs=costs * rng.lognormal(0.0, scatter, len(costs)),
                          dollar_year=2026)
    fits = compare_methods(lots=lots.unit_ranges(), lot_costs=lots.costs)
    per = {}
    for method, cf in fits.items():
        res = cf.result
        per[method] = {
            "t1": cf.t1, "slope": cf.slope, "theory": cf.theory.value, "granularity": cf.granularity,
            "theta": res.theta, "cov": res.cov, "sigma": res.sigma, "n_obs": res.n_obs,
            "n_params": res.n_params, "df": res.df, "n_iter": res.n_iter, "converged": res.converged,
            "observed": res.observed, "fitted": res.fitted,
            "mean_percent_error": res.mean_percent_error, "standard_error": res.standard_error,
            "cv": res.cv, "r_squared": res.r_squared,
        }
    return {"scatter": scatter, "series": series_json(lots), "unit_ranges": lots.unit_ranges(),
            "fits": per, "comparison_table": comparison_table(fits)}


def render_log(log) -> dict:
    """AssumptionLog.render() without its '*Generated <timestamp> by cost_core on
    Python x.y.z (OS)*' line (assumptions.py:61-65, 95): the timestamp changes
    every run and the interpreter text changes per machine."""
    log.created_at = "<created_at stripped: non-golden>"
    lines = log.render().splitlines()
    kept = [ln for ln in lines if not ln.startswith("*Generated ")]
    text = "\n".join(kept)
    return {"text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "n_lines": len(kept), "n_lines_dropped": len(lines) - len(kept)}


def capture_lots(inputs_dir) -> dict:
    out = {}
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = LOTS.analyse_lots(reference_series())
        out["reference"] = report_json(
            ref,
            simulate_specs=[("n20000_seed1", {"n_iter": 20000, "seed": 1}),
                            ("n5000_seed42", {"n_iter": 5000, "seed": 42}),
                            ("n5000_seed43", {"n_iter": 5000, "seed": 43}),
                            ("n20000_seed11_rho0.30", {"n_iter": 20000, "seed": 11, "lot_correlation": 0.30})],
            plans=[("10_15_20", [10, 15, 20], {}), ("10_15_20_25_30", [10, 15, 20, 25, 30], {}),
                   ("20_from_1", [20], {"first_unit": 1}), ("20_from_201", [20], {"first_unit": 201}),
                   ("10_20_30", [10, 20, 30], {}), ("10_15_20_cf1.2", [10, 15, 20], {"complexity": 1.2})],
            forecasts=[("14_14", [14, 14], {}), ("14_14_level0.95", [14, 14], {"level": 0.95})],
        )
        out["reference_forecast_6_12x6_6"] = report_json(
            LOTS.analyse_lots(reference_series(), forecast=[6, 12, 12, 12, 12, 12, 12, 6]),
            simulate_specs=[("n20000_seed11", {"n_iter": 20000, "seed": 11})])
        out["reference_first_unit_101"] = fit_json(reference_series(first_unit=101).fit())
        out["reference_complexity_1.2"] = report_json(LOTS.analyse_lots(reference_series(), complexity=1.2))
        out["reference_legacy"] = report_json(LOTS.analyse_lots(reference_series(), legacy_rate_omission=True))
        out["reference_series_frame"] = series_json(reference_series())
        # tests/test_lots.py:246-258: PROGRAM Z, accepted, total basis, first_unit=7
        s7 = reference_series(program="PROGRAM Z", quantity_definition="accepted", cost_basis="total", first_unit=7)
        rep7 = LOTS.analyse_lots(s7)
        out["reference_first_unit_7_total_basis"] = {"series": series_json(s7), "report": report_json(rep7)}
        # assumption logs (tests/test_lots.py:246-258, 686-706)
        out["assumption_logs"] = {
            "_note": "AssumptionLog.render() with created_at replaced by a fixed marker (assumptions.py:61-65, 95)",
            "reference_source_history_csv": render_log(LOTS.build_assumption_log(ref, source="history.csv")),
            "reference_priced_plan_10_20_30": render_log(
                LOTS.build_assumption_log(ref, priced_plan=ref.price_lot_plan([10, 20, 30]))),
            "first_unit_7_total_basis_source_history_csv": render_log(
                LOTS.build_assumption_log(rep7, source="history.csv")),
        }
        # clean series and its escalated variants (tests/test_lots.py:42-56, 616-656)
        out["clean_series"] = report_json(LOTS.analyse_lots(clean_series()))
        out["clean_series_inputs"] = series_json(clean_series())
        for rate in (0.02, 0.04, 0.06, 0.15):
            s = escalate(clean_series(), rate)
            entry = {"rate": rate, "series": series_json(s)}
            r = attempt(lambda: report_json(LOTS.analyse_lots(s)))
            entry["report"] = r
            out[f"clean_series_escalated_{rate}"] = entry
        # tests/test_lots.py:60-66 escalated(rate) on the REFERENCE series; :118-124 uses escalated(0.15)
        for rate in (0.02, 0.04, 0.06, 0.15):
            s = escalate(reference_series(), rate)
            out[f"reference_escalated_{rate}"] = {
                "rate": rate, "series": series_json(s),
                "check_constant_dollars": s.check_constant_dollars(warn=False),
                "report": attempt(lambda: report_json(LOTS.analyse_lots(s)))}
        # tests/test_lots.py:359-374: 85% curve, t1 5000, quantities [10,15,20,25,30,35] priced at their
        # own lmp_func midpoints, through LotSeries.fit() (t1 and slope must come back to rel 1e-6)
        out["midpoint_curve_recovery"] = {}
        for label, qs in (("10_15_20_25_30_35", [10, 15, 20, 25, 30, 35]),):
            from cost_core.lotmodel.mathx import lmp_func
            t1, b = 5_000.0, np.log2(0.85)
            cursor, costs = 1, []
            for q in qs:
                mid = lmp_func(cursor, cursor + q - 1, q, b)
                costs.append(t1 * mid ** b * q)
                cursor += q
            s = LOTS.LotSeries(quantities=qs, costs=costs, dollar_year=2026)
            out["midpoint_curve_recovery"][label] = {
                "truth": {"t1": t1, "b": b, "slope": 0.85}, "series": series_json(s),
                "fit": attempt(lambda: fit_json(s.fit())),
                "report": attempt(lambda: report_json(LOTS.analyse_lots(s)))}
        # tests/test_lots.py:608-613: rising cumulative average, costs q*1000*1.20**i (RISES warning through fit())
        qs = [20, 20, 25, 25, 30, 30]
        s = LOTS.LotSeries(quantities=qs, costs=[q * 1_000.0 * (1.20 ** i) for i, q in enumerate(qs)],
                           dollar_year=2026)
        out["rising_cumulative_average_1.20"] = {
            "series": series_json(s), "check_constant_dollars": s.check_constant_dollars(warn=False),
            "fit": attempt(lambda: fit_json(s.fit())),
            "report": attempt(lambda: report_json(LOTS.analyse_lots(s)))}
        # tests/test_lots.py:270-290: cost_core.learning_curve.compare_methods on the reference lots with
        # lognormal scatter from default_rng(4); this is the fitting-side MUPE/ZMPE the refactor consolidates onto
        out["learning_curve_compare_methods"] = {
            "_note": "compare_methods(lots=series.unit_ranges(), lot_costs=series.costs); costs = reference lot "
                     "totals * default_rng(4).lognormal(0, scatter, 6); fitting.fit results per method",
        }
        for scatter in (0.05, 0.10, 0.30):
            out["learning_curve_compare_methods"][f"scatter_{scatter}"] = attempt(
                learning_curve_block, scatter)
        # rate-driven series, tests/test_lots.py:385-388 and 402-410 exactly
        qty, costs = rate_driven_test_series()
        out["rate_driven_inputs_note"] = ("quantities [40,5,38,6,36,8,34,10], costs = q*(3000*q**-0.35)*"
                                          "default_rng(3).lognormal(0, 0.01) drawn in list order (tests/test_lots.py:385-388)")
        rd = LOTS.LotSeries(quantities=qty, costs=costs, dollar_year=2026)
        out["rate_driven_inputs"] = series_json(rd)
        out["rate_driven"] = report_json(LOTS.analyse_lots(rd))
        out["rate_driven_fit_t_gate_2.0"] = fit_json(rd.fit(t_gate=2.0))
        out["rate_driven_fit_t_gate_500"] = fit_json(rd.fit(t_gate=500.0))
        # three lots (tests/test_lots.py:576-586, 675-683)
        s3 = LOTS.LotSeries(quantities=[20, 25, 30], costs=[6.0e4, 7.0e4, 8.0e4], dollar_year=2026)
        out["three_lots"] = {"series": series_json(s3), "fit": attempt(lambda: fit_json(s3.fit())),
                             "report": attempt(lambda: report_json(LOTS.analyse_lots(s3)))}
        # labelled 4-lot CSV via LotSeries.read (tests/test_lots.py:731-742)
        csv_path = inputs_dir / "labelled_lots.csv"
        pd.DataFrame({"lot": ["LRIP 1", "LRIP 2", "FRP 1", "FRP 2"],
                      "units": [10, 12, 20, 25],
                      "cost": [5e4, 5.4e4, 8e4, 9.4e4]}).to_csv(csv_path, index=False)
        lab = LOTS.LotSeries.read(csv_path, dollar_year=2026)
        out["labelled_csv"] = {"file": csv_path.name, "csv_text": csv_path.read_text(encoding="utf-8"),
                               "series": series_json(lab),
                               "report": attempt(lambda: report_json(LOTS.analyse_lots(lab)))}
        # Excel front door (cost_core/lots.py:462-464)
        xlsx_path = inputs_dir / "reference_lots.xlsx"
        pd.DataFrame({"units": REF_QTY, "cost": [q * a for q, a in zip(REF_QTY, REF_AUC)]}).to_excel(
            xlsx_path, index=False)
        sx = LOTS.LotSeries.read(xlsx_path, dollar_year=2026)
        out["reference_xlsx_read"] = {"file": xlsx_path.name, "series": series_json(sx),
                                      "fit": attempt(lambda: fit_json(sx.fit()))}
        # example_lots.csv
        ex = LOTS.LotSeries.read(str(LIB_ROOT / "example_lots.csv"), dollar_year=2026)
        out["example_lots_csv_series"] = series_json(ex)
        out["example_lots_csv"] = report_json(
            LOTS.analyse_lots(ex, forecast=[30, 40]),
            simulate_specs=[("n20000_seed0", {"n_iter": 20000, "seed": 0}),
                            ("n5000_seed0", {"n_iter": 5000, "seed": 0})],
            plans=[("10_15_20_25_30", [10, 15, 20, 25, 30], {}), ("10_15_20", [10, 15, 20], {})],
        )
        out["example_lots_csv_nofcst"] = report_json(LOTS.analyse_lots(ex))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

# --csv is given relative to LIB_ROOT (the subprocess cwd) because the CLI
# copies the argument verbatim into ASSUMPTIONS.md's "Source:" line.
CLI_RUNS = {
    "cli_fit_lots": ["--csv", "example_lots.csv", "--dollar-year", "2026",
                     "--forecast", "30,40", "--simulate", "5000", "--seed", "0", "--price-lots", "10,15,20"],
    # the argument plumbing: cost_core/cli.py, the fit-lots parser
    "cli_fit_lots_flags": ["--csv", "example_lots.csv", "--dollar-year", "2026",
                           "--forecast", "30,40", "--level", "0.95", "--legacy-rate-omission",
                           "--first-unit", "101", "--price-lots", "10,15,20", "--price-from-unit", "201",
                           "--complexity", "1.2", "--program", "CLI_FLAGS", "--quantity-definition", "accepted",
                           "--units-col", "units", "--cost-col", "cost", "--simulate", "2000", "--seed", "7"],
    "cli_fit_lots_total_strict": ["--csv", "example_lots.csv", "--dollar-year", "2026",
                                  "--cost-basis", "total", "--t-gate", "500", "--aicc-tie", "3.0"],
}


def run_cli_one(name: str, extra: list, workbooks) -> dict:
    workbooks = Path(workbooks)
    workbooks.mkdir(parents=True, exist_ok=True)
    outdir = workbooks / name
    if outdir.exists():
        shutil.rmtree(outdir)
    cmd = [PY, "-m", "cost_core.cli", "fit-lots"] + extra + ["--out", str(outdir)]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, cwd=str(LIB_ROOT), capture_output=True, text=True, env=env, encoding="utf-8",
                          errors="replace")
    (workbooks / f"{name}_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (workbooks / f"{name}_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    files = sorted(p.name for p in outdir.iterdir()) if outdir.exists() else []
    csvs, texts, hashes = {}, {}, {}
    assumptions = None
    for fname in files:
        p = outdir / fname
        hashes[fname] = sha256_of(p)
        if fname.endswith(".csv"):
            csvs[fname] = pd.read_csv(p, float_precision="round_trip")
            texts[fname] = mask_paths(p.read_text(encoding="utf-8"))
        elif fname == "ASSUMPTIONS.md":
            lines = p.read_text(encoding="utf-8").splitlines()
            kept = [mask_paths(ln) for ln in lines if not ln.startswith("*Generated ")]
            assumptions = {"text_without_generated_line": "\n".join(kept),
                           "n_lines_dropped": len(lines) - len(kept)}
    shown = [mask_paths("<workbooks>/" + name if c == str(outdir) else c) for c in cmd]
    stdout_lines = [mask_paths(ln) for ln in proc.stdout.splitlines()]
    return {"cmd": shown, "returncode": proc.returncode, "files": files,
            "sha256": {k: v for k, v in hashes.items() if k != "ASSUMPTIONS.md"},
            "csv_contents": csvs, "csv_text": texts, "assumptions_md": assumptions,
            "stdout_lines": stdout_lines,
            "note": "ASSUMPTIONS.md line 3 carries a timestamp (reporting/assumptions.py:62); its sha256 is "
                    "omitted and the text is stored without that line. CSVs parsed with float_precision="
                    "'round_trip'; the raw text and sha256 are stored too."}


def run_cli(workbooks) -> dict:
    return {name: run_cli_one(name, extra, workbooks) for name, extra in CLI_RUNS.items()}


# --------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------


def error_paths_engine(mod) -> dict:
    """Every engine refusal a test asserts a substring of, through `mod`."""
    an, es = app_example_frames()
    thin = pd.DataFrame({"Lot": [1, 2], "Lot FY": [2015, 2016], "Qty": [10.0, 20.0], "AUC ($K)": [800.0, 640.0]})
    empty_est = pd.DataFrame(columns=["Lot", "Lot FY", "Qty", "Complexity"])
    return {
        # tests/test_model.py:189-203
        "two_costed_lots_at_least_3": attempt(mod.run_lot_cost_model, thin, es, None),
        "empty_estimate_table": attempt(mod.run_lot_cost_model, an, empty_est, None),
        # Cost_AI_v1/tests/test_lotmodel.py:167-181
        "empty_analogy_iloc0_0": attempt(mod.run_lot_cost_model, CC_ANALOGY.iloc[0:0], CC_ESTIMATE, {}),
        "empty_estimate_iloc0_0": attempt(mod.run_lot_cost_model, CC_ANALOGY, CC_ESTIMATE.iloc[0:0], {}),
        "missing_cost_column": attempt(mod.run_lot_cost_model, CC_ANALOGY.drop(columns=["AUC ($K)"]),
                                       CC_ESTIMATE, {}),
        # tests/test_equation_conformance.py:262-271, Cost_AI_v1/tests/test_lotmodel.py:297-301
        "ToolMatchProjection_True": attempt(mod.run_lot_cost_model, an, es, {"ToolMatchProjection": True}),
        "ToolMatchProjection_False": attempt(mod.run_lot_cost_model, an, es, {"ToolMatchProjection": False}),
    }


def error_paths_enrich() -> dict:
    """enrich refusals (Cost_AI_v1/tests/test_lotmodel.py:334, :432, :491, :513, _design's 'not fitted')."""
    proj, ctx = LIB.run_lot_cost_model(CC_ANALOGY, CC_ESTIMATE, {})
    _, ctx3 = LIB.run_lot_cost_model(CC_ANALOGY.iloc[:3], CC_ESTIMATE, {})
    empty_summary = pd.DataFrame({"Item": ["SELECTED"], "Value": [""], "LC": [""], "Rate": [""], "LC+Rate": [""]})
    return {
        "projection_intervals_level_1.4": attempt(ENRICH.projection_intervals, ctx, proj, "LC", level=1.4),
        "projection_intervals_level_0": attempt(ENRICH.projection_intervals, ctx, proj, "LC", level=0.0),
        "simulate_buy_n_iter_1": attempt(ENRICH.simulate_buy, ctx, proj, "LC", n_iter=1),
        "compare_fitting_methods_Quadratic": attempt(ENRICH.compare_fitting_methods, ctx, "Quadratic"),
        "selected_model_name_nothing_selected": attempt(ENRICH.selected_model_name, empty_summary),
        "rate_model_not_fitted_three_lots": attempt(ENRICH.compare_fitting_methods, ctx3, "Rate"),
        "influence_unknown_model": attempt(ENRICH.influence_diagnostics, ctx, "Cubic"),
    }


def error_paths_wbs() -> dict:
    """ProgramError texts (tests/test_wbs.py:180-211, :600-645, :494-497)."""
    W = PROGRAM_MOD

    def prog(*els):
        return W.Program("P", list(FY_BUY), list(els))

    bad_len = airframe()
    bad_len.quantities = [1.0, 2.0]
    idle = airframe()
    idle.quantities = [0.0] * 6
    neg = airframe()
    neg.quantities = [-1.0] + list(AIRCRAFT[1:])
    thin_el = airframe()
    thin_el.analogy = thin_el.analogy.head(2)

    def full(mutate):
        p = se_pm_program()
        mutate(p)
        return p

    def set_amounts(p):
        p.elements[-1].amounts = [1.0, 2.0]

    def set_factor_none(p):
        p.elements[3].factor = None

    def set_unknown_basis(p):
        p.elements[3].basis = ["1.9 Imaginary"]

    def set_self_basis(p):
        p.elements[3].basis = ["1.4 Systems Engineering"]

    def set_circle(p):
        p.elements[3].basis = ["1.5 Program Management"]
        p.elements[4].basis = ["1.4 Systems Engineering"]

    def set_kind(p):
        p.elements[-1].kind = "guesswork"

    return {
        "quantity_vector_wrong_length": attempt(W.roll_up, prog(bad_len), simulate=False),
        "duplicate_element_names": attempt(W.roll_up, prog(airframe(), airframe()), simulate=False),
        "element_bought_in_no_lot": attempt(W.roll_up, prog(idle), simulate=False),
        "negative_quantity": attempt(W.roll_up, prog(neg), simulate=False),
        "program_with_no_elements": attempt(W.roll_up, W.Program("P", list(FY_BUY), []), simulate=False),
        "element_that_cannot_be_fitted": attempt(W.roll_up, prog(thin_el), simulate=False),
        "wrong_number_of_amounts": attempt(W.roll_up, full(set_amounts), simulate=False),
        "factor_element_needs_a_factor": attempt(W.roll_up, full(set_factor_none), simulate=False),
        "basis_naming_unknown_element": attempt(W.roll_up, full(set_unknown_basis), simulate=False),
        "percentage_of_itself": attempt(W.roll_up, full(set_self_basis), simulate=False),
        "circle_between_factors": attempt(W.roll_up, full(set_circle), simulate=False),
        "only_derived_elements": attempt(W.roll_up, W.Program("P", list(FY_BUY),
                                                              [W.flat_amount("Only tooling", [1e6] * 6)]),
                                         simulate=False),
        "unknown_kind": attempt(W.roll_up, full(set_kind), simulate=False),
        "no_lots_at_all": attempt(W.roll_up, W.Program("P", [], [airframe()]), simulate=False),
        "sensitivity_reference_not_an_element": attempt(W.buy_profile_sensitivity, test_program(), factors=(1.0,),
                                                        reference_element="1.9 Nope"),
    }


def error_paths_lots(inputs_dir) -> dict:
    """LotInputError texts (tests/test_lots.py:98-180, :557-566, :570-586, :236-244, :193-225)."""
    q, c = np.array(REF_QTY), np.array([q * a for q, a in zip(REF_QTY, REF_AUC)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = LOTS.analyse_lots(reference_series())
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    err_dir = inputs_dir / "error_inputs"
    err_dir.mkdir(exist_ok=True)
    pd.DataFrame({"widgets": REF_QTY, "spend": c}).to_csv(err_dir / "widgets_spend.csv", index=False)
    pd.DataFrame({"units": REF_QTY, "cost": c}).to_csv(err_dir / "units_cost.csv", index=False)
    pd.DataFrame({"units": [20, 25, 30], "cost": ["1200000", "see note 4", "1500000"]}).to_csv(
        err_dir / "bad_cost_text.csv", index=False)
    (err_dir / "lots.json").write_text("{}", encoding="utf-8")
    lots_err = {
        "dollar_year_required": attempt(LOTS.LotSeries, quantities=q, costs=c),
        "cost_basis_whatever": attempt(LOTS.LotSeries, quantities=q, costs=c, dollar_year=2026, cost_basis="whatever"),
        "mismatched_lengths": attempt(LOTS.LotSeries, quantities=[10, 20, 30], costs=[1e6, 2e6], dollar_year=2026),
        "no_lots": attempt(LOTS.LotSeries, quantities=[], costs=[], dollar_year=2026),
        "zero_quantity_lot2": attempt(LOTS.LotSeries, quantities=[20, 0, 30], costs=[1e6, 2e6, 3e6], dollar_year=2026),
        "negative_quantity_lot2": attempt(LOTS.LotSeries, quantities=[20, -5, 30], costs=[1e6, 2e6, 3e6],
                                          dollar_year=2026),
        "fractional_quantity": attempt(LOTS.LotSeries, quantities=[20, 22.5, 30], costs=[1e6, 2e6, 3e6],
                                       dollar_year=2026),
        "zero_cost_lot3": attempt(LOTS.LotSeries, quantities=[20, 25, 30], costs=[1e6, 2e6, 0.0], dollar_year=2026),
        "first_unit_0": attempt(LOTS.LotSeries, quantities=q, costs=c, dollar_year=2026, first_unit=0),
        "mismatched_labels": attempt(LOTS.LotSeries, quantities=q, costs=c, dollar_year=2026, labels=("only", "two")),
        "lot_plan_[0]": attempt(ref.price_lot_plan, [0]),
        "lot_plan_[10,-5]": attempt(ref.price_lot_plan, [10, -5]),
        "lot_plan_[]": attempt(ref.price_lot_plan, []),
        "price_from_unit_0": attempt(ref.price_lot_plan, [10], first_unit=0),
        "two_lots_fit": attempt(lambda: LOTS.LotSeries(quantities=[20, 25], costs=[1e6, 1.2e6],
                                                       dollar_year=2026).fit()),
        "three_lots_allow_small_sample_False": attempt(
            lambda: LOTS.LotSeries(quantities=[20, 25, 30], costs=[6.0e4, 7.0e4, 8.0e4],
                                   dollar_year=2026).fit(allow_small_sample=False)),
        "read_missing_file": attempt(LOTS.LotSeries.read, err_dir / "nope.csv", dollar_year=2026),
        "read_unsupported_json": attempt(LOTS.LotSeries.read, err_dir / "lots.json", dollar_year=2026),
        "read_unrecognisable_header": attempt(LOTS.LotSeries.read, err_dir / "widgets_spend.csv", dollar_year=2026),
        "read_named_column_missing": attempt(LOTS.LotSeries.read, err_dir / "units_cost.csv", dollar_year=2026,
                                             units_col="nope"),
        "read_unparseable_cost_text": attempt(LOTS.LotSeries.read, err_dir / "bad_cost_text.csv", dollar_year=2026),
        "estimate_frame_non_positive": attempt(reference_series().estimate_frame, [10, 0]),
    }
    for bad in ("not a year", None, 12, 3500):
        lots_err[f"implausible_dollar_year_{bad!r}"] = attempt(LOTS.LotSeries, quantities=q, costs=c,
                                                                dollar_year=bad)
    for v in lots_err.values():
        if "error" in v:
            v["error"] = mask_paths(mask_dir(v["error"], err_dir, "<error_inputs>"))
    return lots_err


def capture_error_paths(inputs_dir, *, engine_mods=None, after_enrich=None) -> dict:
    """Every refusal text a test asserts a substring of. The suites catch a
    changed wording through pytest; this pins the full text so a rewrite of
    mathx/enrich/wbs can be diffed against it. Each entry is attempt()'s
    {"error": "<Type>: <message>"}; an "ok" here means the refusal is gone.

    `engine_mods` maps a tag to an engine module (default {"lib": cost_core});
    the capture script passes both the tool's engine and the library's.
    `after_enrich` is spliced in between the enrich and wbs blocks -- it is how
    the capture script adds its app-only "risk" section without changing the
    key order of the golden.
    """
    out = {"_note": "attempt() results: {'error': 'Type: text'}; an 'ok' entry means the refusal no longer fires"}
    out["engine"] = {tag: error_paths_engine(mod) for tag, mod in (engine_mods or {"lib": LIB}).items()}
    out["enrich"] = error_paths_enrich()
    for key, value in (after_enrich or {}).items():
        out[key] = value
    out["wbs"] = error_paths_wbs()
    out["lots"] = error_paths_lots(inputs_dir)
    return out


# --------------------------------------------------------------------------
# summary.py's own arithmetic
# --------------------------------------------------------------------------
#
# THE PROBLEM. ``derived_stats`` above is a RE-IMPLEMENTATION of the eight
# statistics ``cost_core/lotmodel/summary.py`` computes in its ``stat_for``
# closure (SEE, R2, AdjR2, CV, MAPE, Bias, AICc, T). The goldens pin
# ``derived_stats`` at rtol 1e-9, but summary.py's own copy of that arithmetic
# reaches a golden only through the ``summary`` frame, whose cells are strings
# printed at 2 or 4 decimals and compared for exact string equality. Anything
# below half a printed digit is therefore invisible there: multiplying
# summary.py's MAPE or SEE by 1 + 1e-6 moves no golden byte. In a commit whose
# whole claim is 1e-9, that is a hole, and it is the reason this section exists.
#
# WHAT CLOSES IT. summary.py never returns a float, but a printed digit still
# carries the full precision of the number behind it -- not in its value, in
# the POINT AT WHICH IT FLIPS. Move an input continuously and the printed cell
# changes from one string to the next at one exact input value, and that value
# is a full-precision measurement of the arithmetic behind the print. So:
#
#   1. take a real ctx and a knob that moves the statistic smoothly (scaling
#      the model's SSE moves SEE, R2, AdjR2, CV, AICc and T; shifting its
#      Fitted vector moves MAPE and Bias);
#   2. bisect the knob to the point where SUMMARY.PY's printed cell flips;
#   3. bisect the same knob to the point where DERIVED_STATS, formatted the
#      same way, flips;
#   4. evaluate derived_stats at both points and compare the two values at the
#      policy's own rtol 1e-9.
#
# If the two implementations agree as floats they flip at the same knob value
# -- bit for bit, because both bisections walk the same midpoints -- and the
# difference is exactly zero. If summary.py's statistic is off by a relative
# eps, the flip moves, and the two derived values differ by about eps times the
# statistic. A part in a million is then five hundred times the tolerance.
# :func:`check_summary_precision` does this for all eight statistics on all
# three model columns.
#
# Two further probes read summary.py's SELECTION thresholds the same way, on
# every case rather than one, and they check something the flip probe does not:
# that the gate compares the right number, not merely that the number is right.
#
#   * ``|T|`` for LC+Rate -- the SELECTED row is LC+Rate exactly while
#     ``TGate <= |T|`` (summary.py:127-141), so bisecting ``cfg["TGate"]``
#     recovers ``|T|``;
#   * ``AICc(LC+Rate) - AICc(LC)`` -- with TGate 0 the selection is LC+Rate and
#     the Selection basis row gains "AICc favors LC" exactly while
#     ``AiccTie < AICc(LC+Rate) - AICc(LC)`` (summary.py:181-190, 204-211).
#
# And :func:`check_summary_arithmetic` holds every printed stat cell to within
# half a printed unit of ``derived_stats`` on every case, which is the cheap
# check that the two implementations are the same function at all.
#
# The chain that results is golden -> derived_stats (rtol 1e-9, authority_1e-9)
# -> summary.py (rtol 1e-9, here). Nothing in summary.py's arithmetic is now
# pinned only to two decimal places.

#: The stat rows of the summary frame. Per row: the Item, the derived_stats
#: key, half a printed unit in the units of the derived value, how to format
#: the derived value exactly as summary.py prints it (fmt_n / fmt_p / fmt_ps,
#: summary.py:236-268, as applied in mk_col :297-356), how to read that print
#: back, and which knob moves the statistic for the flip probe.
def _fmt4(v):
    return f"{v:.4f}"


def _fmt2(v):
    return f"{v:.2f}"


def _fmtp(v):
    return f"{v * 100:.2f}%"


def _fmtps(v):
    return f"{v * 100:+.2f}%"


_SUMMARY_STAT_ROWS = (
    # Item, derived key, half printed unit, formatter, parse kind, knob
    ("R2 (log)", "R2", 5e-5, _fmt4, "num", "sse"),
    ("Adj R2", "AdjR2", 5e-5, _fmt4, "num", "sse"),
    ("SEE (log)", "SEE", 5e-5, _fmt4, "num", "sse"),
    ("CV", "CV", 5e-5, _fmtp, "pct", "sse"),
    ("MAPE", "MAPE", 5e-5, _fmtp, "pct", "fitted"),
    ("Mean bias", "Bias", 5e-5, _fmtps, "pct_signed", "fitted"),
    ("AICc", "AICc", 5e-3, _fmt2, "num", "sse"),
    ("t (rate coefficient)", "t_rate", 5e-3, _fmt2, "num", "sse"),
)

#: The note summary.py appends to the Selection basis when AICc disagrees with
#: the significance gate (summary.py:204-211). The probe reads its presence.
_AICC_DISAGREE_NEEDLE = "AICc favors LC"


def _summary_probe(ctx: dict, **cfg_over):
    """generate_analyst_summary on `ctx` with cfg keys overridden.

    Neither `ctx` nor its cfg is mutated, and `provenance={}` keeps the two
    provenance rows -- and the git call behind them -- out of the loop.
    """
    probe = dict(ctx)
    probe["cfg"] = {**ctx["cfg"], **cfg_over}
    return LIB.generate_analyst_summary(probe, None, provenance={})


def _summary_cell(frame: pd.DataFrame, item: str, column: str):
    for rec in frame.to_dict("records"):
        if rec.get("Item") == item:
            return rec.get(column)
    return None


def _selected_in(frame: pd.DataFrame):
    for name in MODELS:
        if _summary_cell(frame, "SELECTED", name) == "YES":
            return name
    return None


def _bisect_flip(pred, lo: float, hi: float, *, rel=1e-14, max_iter=200) -> float:
    """The x in [lo, hi] where `pred` flips from True to False.

    `pred(lo)` must be True and `pred(hi)` False. Returns the largest x still
    True. Two bisections run over the same bracket with predicates that agree
    everywhere visit the same midpoints and return the same float exactly,
    which is what makes the paired flip probe below a zero-difference test.
    """
    for _ in range(max_iter):
        if hi - lo <= rel * max(1.0, abs(lo), abs(hi)):
            break
        mid = lo + (hi - lo) / 2.0
        if mid <= lo or mid >= hi:
            break
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _bracket_flip(pred, lo: float, hi: float, *, doublings=64):
    """Widen [lo, hi] until pred(lo) is True and pred(hi) is False.

    Returns (lo, hi), or None if the boundary is not in reach. `lo` is walked
    away from zero in its own direction, so a negative `lo` goes more negative.
    """
    for _ in range(doublings):
        if pred(lo):
            break
        lo *= 2.0
    else:
        return None
    for _ in range(doublings):
        if not pred(hi):
            break
        hi *= 2.0
    else:
        return None
    return lo, hi


def _probe_mismatch(out, path, reference, measured, policy, what):
    rtol = float(policy["tolerance"]["rtol"])
    atol = float(policy["tolerance"]["atol"])
    limit = atol + rtol * max(abs(reference), abs(measured))
    if not (abs(reference - measured) <= limit):
        out.append({"path": path, "golden": reference, "new": measured, "rule": what,
                    "tolerance": f"|summary.py - derived_stats| <= {limit!r} "
                                 f"(rtol {rtol!r}, atol {atol!r})"})


def _parse_printed(cell, kind: str):
    """The number summary.py printed, back as a float, or None if it is not one."""
    if not isinstance(cell, str):
        return None
    text = cell.strip()
    if kind in ("pct", "pct_signed"):
        if not text.endswith("%"):
            return None
        text = text[:-1]
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None
    return value / 100.0 if kind in ("pct", "pct_signed") else value


def check_summary_arithmetic(ctx: dict, summary: pd.DataFrame, policy: dict) -> list:
    """Check summary.py's statistics against ``derived_stats(ctx)`` on one case.

    Returns the same {"path", "golden", "new", "rule", "tolerance"} records
    :func:`compare_with_policy` returns, so :func:`format_mismatches` prints
    them the same way. "golden" is the ``derived_stats`` value -- the one the
    goldens pin at 1e-9 -- and "new" is what summary.py itself produced.

    Every printed stat cell must agree with ``derived_stats`` to within half a
    printed unit, and ``|T|`` for LC+Rate and the ``AICc(LC+Rate) - AICc(LC)``
    gap are read back out of summary.py's own selection thresholds at full
    precision and compared at rtol 1e-9. :func:`check_summary_precision` is
    what pins the other six statistics at 1e-9; see the section comment.
    """
    out: list = []
    ds = derived_stats(ctx)

    # ---- 1. every printed stat cell, against derived_stats, to half a unit
    rows = _SUMMARY_STAT_ROWS + (("dAICc", None, 5e-3, _fmt2, "num", None),)
    for name in MODELS:
        stats = ds.get(name)
        for item, key, half, _fmt, kind, _knob in rows:
            cell = _summary_cell(summary, item, name)
            value = ((ds.get("dAICc") or {}).get(name) if key is None
                     else (None if stats is None else stats.get(key)))
            path = f"/summary/[Item={item}]/{name}"
            if stats is None:
                if cell != "-":
                    out.append({"path": path, "golden": "- (model not fitted)", "new": cell,
                                "rule": "summary.py printed a statistic for a model "
                                        "derived_stats says is not fitted"})
                continue
            if value is None or (isinstance(value, float) and math.isnan(value)):
                if cell not in ("n/a", "-"):
                    out.append({"path": path, "golden": "n/a (derived_stats has no value)", "new": cell,
                                "rule": "summary.py printed a statistic derived_stats does not have"})
                continue
            printed = _parse_printed(cell, kind)
            if printed is None:
                out.append({"path": path, "golden": float(value), "new": cell,
                            "rule": "summary.py did not print a number where derived_stats has one"})
                continue
            limit = half * (1.0 + 1e-9) + 1e-12
            if not (abs(printed - float(value)) <= limit):
                out.append({"path": path, "golden": float(value), "new": printed,
                            "rule": "summary.py's printed value is more than half a printed unit "
                                    "from derived_stats",
                            "tolerance": f"|printed - derived_stats| <= {limit!r}"})

    # ---- 2. |T| for LC+Rate, read back out of the TGate threshold
    lcr = ds.get("LC+Rate")
    t_lcr = None if lcr is None else lcr.get("t_rate")
    has_t = t_lcr is not None and not math.isnan(float(t_lcr))

    def gated_in(g):
        return _selected_in(_summary_probe(ctx, TGate=g)) == "LC+Rate"

    if lcr is not None and has_t:
        if not gated_in(0.0):
            out.append({"path": "/summary/[Item=SELECTED]/LC+Rate", "golden": "YES at TGate 0",
                        "new": _selected_in(_summary_probe(ctx, TGate=0.0)),
                        "rule": "derived_stats has a t for LC+Rate, so summary.py must gate it "
                                "in at TGate 0"})
        else:
            bracket = _bracket_flip(gated_in, 0.0, 1.0)
            if bracket is None:
                out.append({"path": "/summary/(TGate boundary)/abs(t_rate) LC+Rate",
                            "golden": abs(float(t_lcr)),
                            "new": "no TGate large enough to gate LC+Rate out",
                            "rule": "the |t| >= TGate boundary could not be bracketed"})
            else:
                measured = _bisect_flip(gated_in, *bracket)
                _probe_mismatch(out, "/summary/(TGate boundary)/abs(t_rate) LC+Rate",
                                abs(float(t_lcr)), measured, policy,
                                "summary.py's own |t| for LC+Rate, recovered from the TGate "
                                "selection threshold, against derived_stats")
    elif lcr is not None and not has_t and gated_in(0.0):
        out.append({"path": "/summary/[Item=SELECTED]/LC+Rate", "golden": "not YES (no t)",
                    "new": "YES",
                    "rule": "summary.py gated LC+Rate in on a t derived_stats does not have"})

    # ---- 3. AICc(LC+Rate) - AICc(LC), read back out of the AiccTie threshold
    lc = ds.get("LC")
    a_lcr = None if lcr is None else lcr.get("AICc")
    a_lc = None if lc is None else lc.get("AICc")
    if lcr is not None and has_t and a_lcr is not None and a_lc is not None:
        def disagrees(tie):
            cell = _summary_cell(_summary_probe(ctx, TGate=0.0, AiccTie=tie),
                                 "Selection basis", "LC+Rate")
            return isinstance(cell, str) and _AICC_DISAGREE_NEEDLE in cell

        bracket = _bracket_flip(disagrees, -1.0, 1.0)
        path = "/summary/(AiccTie boundary)/AICc LC+Rate minus LC"
        if bracket is None:
            out.append({"path": path, "golden": float(a_lcr) - float(a_lc),
                        "new": "the AICc-disagrees note could not be bracketed",
                        "rule": "the AICc(LC+Rate) > AICc(LC) + AiccTie boundary could not "
                                "be bracketed"})
        else:
            measured = _bisect_flip(disagrees, *bracket)
            _probe_mismatch(out, path, float(a_lcr) - float(a_lc), measured, policy,
                            "summary.py's own AICc gap, recovered from the AiccTie selection "
                            "threshold, against derived_stats")
    return out


# ---- the flip probe: every statistic, at full precision ------------------

def _knobbed_ctx(ctx: dict, key: str, knob: str, x: float) -> dict:
    """`ctx` with one model's SSE scaled by `x`, or its Fitted vector shifted by `x`.

    Scaling SSE moves SEE, R2, AdjR2, CV, AICc and T; shifting Fitted (which
    multiplies the fitted costs by e**x) moves MAPE and Bias. Both leave every
    other input alone, and neither touches the caller's ctx.
    """
    model = dict(ctx[key])
    if knob == "sse":
        model["SSE"] = float(ctx[key]["SSE"]) * x
    else:
        model["Fitted"] = [float(f) + x for f in ctx[key]["Fitted"]]
    out = dict(ctx)
    out[key] = model
    return out


def _fitted_shift_base(ctx: dict, key: str) -> float:
    """A Fitted shift that puts every fitted cost strictly below its actual.

    MAPE is a mean of absolute values, so it is only monotone in the shift
    while no residual changes sign. Below this point they are all one sign and
    both MAPE and Bias are strictly decreasing, which is what the bisection
    needs.
    """
    fit_c = np.asarray(ctx["fit_c"], dtype=float)
    fitted = np.asarray(ctx[key]["Fitted"], dtype=float)
    return float(np.min(np.log(fit_c) - fitted)) - 0.5


def check_summary_precision(ctx: dict, policy: dict) -> list:
    """Read summary.py's eight statistics back at full precision, and check them.

    For each statistic and each fitted model: bisect a knob to the point where
    summary.py's printed cell flips, bisect the same knob to the point where
    ``derived_stats`` formatted identically flips, and compare the derived
    value at the two points at the policy's rtol 1e-9. See the section comment
    for why that measures the arithmetic and not the print.

    Returns the usual mismatch records. Statistics a model does not have (t for
    LC, AICc on a sample too small for it) are skipped -- :func:`check_summary_arithmetic`
    is what checks that summary.py agrees they are absent.
    """
    out: list = []
    base = derived_stats(ctx)
    for name, key in MODELS.items():
        if ctx.get(key) is None or base.get(name) is None:
            continue
        for item, dkey, _half, fmt, _kind, knob in _SUMMARY_STAT_ROWS:
            if base[name].get(dkey) is None:
                continue
            if knob == "sse":
                lo, span = 1.0, 0.05
            else:
                lo, span = _fitted_shift_base(ctx, key), 0.02

            def derived_at(x, _key=key, _knob=knob, _name=name, _dkey=dkey):
                value = derived_stats(_knobbed_ctx(ctx, _key, _knob, x))[_name][_dkey]
                return None if value is None else float(value)

            def printed_summary(x, _key=key, _knob=knob, _item=item, _name=name):
                return _summary_cell(_summary_probe(_knobbed_ctx(ctx, _key, _knob, x)), _item, _name)

            def printed_derived(x, _fmt=fmt):
                value = derived_at(x)
                return "n/a" if value is None or math.isnan(value) else _fmt(value)

            path = f"/summary/(flip probe on {knob})/[Item={item}]/{name}"
            base_s, base_d = printed_summary(lo), printed_derived(lo)
            if base_s != base_d:
                out.append({"path": path, "golden": base_d, "new": base_s,
                            "rule": "summary.py and derived_stats print different values at the "
                                    "start of the flip probe"})
                continue
            if base_d == "n/a":
                continue
            hi = None
            for _ in range(8):
                span *= 2.0
                if printed_summary(lo + span) != base_s and printed_derived(lo + span) != base_d:
                    hi = lo + span
                    break
            if hi is None:
                out.append({"path": path, "golden": "a knob range that flips the printed digit",
                            "new": f"none within {span!r} of {lo!r}",
                            "rule": "the flip probe could not move the printed cell; the bracket "
                                    "needs widening for this statistic"})
                continue
            x_s = _bisect_flip(lambda x: printed_summary(x) == base_s, lo, hi)
            x_d = _bisect_flip(lambda x: printed_derived(x) == base_d, lo, hi)
            if x_s == x_d:
                continue  # identical to the last bit: the two agree as floats
            v_s, v_d = derived_at(x_s), derived_at(x_d)
            _probe_mismatch(out, path, v_d, v_s, policy,
                            f"summary.py's {item} for {name}, recovered from where its printed "
                            f"digit flips against the {knob} knob, against derived_stats")
    return out


# --------------------------------------------------------------------------
# the policy comparator
# --------------------------------------------------------------------------

#: Where the copied policy lives once tests/goldens exists.
POLICY_PATH = Path(__file__).resolve().parent / "goldens" / "COMPARE_POLICY.json"


def _build_overrides(path=None) -> dict:
    """The step-2 carve-out, expanded from COMPARE_POLICY into path globs.

    The policy names the frames, the two rows and the per-column tolerance;
    this turns that into one entry per leaf so the comparator can look a path
    up. Building it here rather than writing the globs out by hand is what
    stops the policy and the test drifting apart: change the number in the
    policy and this table changes with it.

    A value is ``{"rtol": ...}``, ``{"atol": ...}`` or both; whichever is
    absent keeps the general rule's. Every leaf not named here is still
    compared at rtol 1e-9, including the OLS row of the same frames.

    Note the ``[[]`` in the generated globs: :mod:`fnmatch` reads a bare ``[``
    as the start of a character class, so ``records[1]`` as a pattern would
    match the string ``records1`` and never the path ``records[1]``.
    """
    entry = _step2_carve_out(path)
    out = {}
    for frame in entry["frames"]:
        for index in entry["rows"].values():
            for column, tol in entry["columns"].items():
                out[f"{frame}/records[[]{index}]/{column}"] = dict(tol)
    bias = entry["derived_stats_Bias"]
    for glob in bias["paths"]:
        out[glob] = {"atol": bias["atol"]}
    return out


def _step2_carve_out(path=None) -> dict:
    """COMPARE_POLICY's step-2 entry, which every carve-out below is read from."""
    return json.loads(Path(path or POLICY_PATH).read_text(encoding="utf-8"))[
        "expected_to_move_beyond_rtol_in_step_2"]


def _build_exclusions() -> list:
    """The step-2 exclusions the policy spells out, as (pattern, why) pairs.

    Three kinds. The exact-fit fixtures build their data by evaluating the
    curve at each lot midpoint, so the fit passes through every point and every
    statistic derived from the residual scale -- F, the standard errors, AICc,
    the t-statistics, the residuals -- is a ratio of rounding errors. The CLI
    writes its CSVs at full repr precision, so their bytes carry the last digit
    of every coefficient and a solver change moves them, exactly as a numpy
    version change already did. And the sha256 beside each assumption log
    hashes the unmasked document, whose MUPE and ZMPE rows are masked out of
    the text comparison by _MASKED_LINES, so the hash cannot be compared while
    the text it hashes is not.

    What each one gives up, said plainly rather than implied. The fixtures keep
    every number an analyst acts on: the selected model, the printed equation,
    the lot midpoints, unit costs, projections, intervals and risk draws are
    all still compared under the general rule. Model selection is not carved
    out at all, because summary.py stopped forming a t-statistic out of a
    residual scale that is not there. What is given up is the goodness-of-fit
    statistics on data with no scatter to measure them against, and the closing
    sentence of the narrative, which names which of six residuals that are all
    zero to eleven decimals is the largest. The CSV bytes give up byte identity
    and keep the numbers, which are still compared through csv_contents at rtol
    1e-9. The assumption-log hashes give up nothing that the masked text is not
    already compared on, line by line.
    """
    entry = _step2_carve_out()
    out = []
    fixtures = entry["exact_fit_fixtures"]
    for block in fixtures["blocks"]:
        for leaf in fixtures["leaves"]:
            out.append((block + ".*" + leaf,
                        "exact-fit fixture: the residual scale is at the floating-point floor, so this "
                        "leaf is a ratio of rounding errors (COMPARE_POLICY "
                        "expected_to_move_beyond_rtol_in_step_2.exact_fit_fixtures)"))
    for pat in entry["full_repr_csv_bytes"]["paths"]:
        out.append((pat,
                    "full-repr CSV bytes carry the last digit of every coefficient; the numbers are "
                    "compared through csv_contents at rtol 1e-9 (COMPARE_POLICY "
                    "expected_to_move_beyond_rtol_in_step_2.full_repr_csv_bytes)"))
    for pat in entry["printed_mupe_zmpe_rows"]["hashes"]:
        out.append((pat,
                    "hashes the unmasked assumption log, whose MUPE and ZMPE rows move with the "
                    "carve-out (COMPARE_POLICY "
                    "expected_to_move_beyond_rtol_in_step_2.printed_mupe_zmpe_rows)"))
    return out


def _platform_carve_out(path=None) -> dict:
    """COMPARE_POLICY's cross-platform entry."""
    return json.loads(Path(path or POLICY_PATH).read_text(encoding="utf-8"))[
        "expected_to_move_across_platforms"]


def _build_platform_overrides(path=None, *, platform=None) -> dict:
    """The cross-platform allowance, expanded from COMPARE_POLICY into path globs.

    Empty on the platform the goldens were captured on, so there every leaf is
    still compared at 1e-9 and a numpy or scipy upgrade still shows. Anywhere
    else it gives the MUPE and ZMPE results of the lots golden's learning-curve
    block -- the fitting.fit results and their two comparison-table rows -- the
    tolerance the policy sizes for each field. theta, cov and fitted are lists,
    which the second glob reaches element by element.
    """
    entry = _platform_carve_out(path)
    if (platform or sys.platform) == entry["captured_on_platform"]:
        return {}
    block = entry["block"]
    out = {}
    for method in entry["fits"]:
        for field, tol in entry["fit_fields"].items():
            leaf = f"{block}/*/ok/fits/{method}/{field}"
            out[leaf] = dict(tol)
            out[leaf + "[[]*"] = dict(tol)
    for index in entry["table_rows"].values():
        for column, tol in entry["table_columns"].items():
            out[f"{block}/*/ok/comparison_table/records[[]{index}]/{column}"] = dict(tol)
    return out


def _build_platform_exclusions(path=None, *, platform=None) -> list:
    """The leaves the cross-platform entry stops comparing, as (pattern, why)
    pairs. Empty on the capture platform, like the allowance."""
    entry = _platform_carve_out(path)
    if (platform or sys.platform) == entry["captured_on_platform"]:
        return []
    return [(rf"^{re.escape(entry['block'])}/[^/]+/ok/fits/({'|'.join(spec['fits'])})/{re.escape(field)}$",
             f"{spec['why']} (COMPARE_POLICY expected_to_move_across_platforms.not_compared_off_platform)")
            for field, spec in entry["not_compared_off_platform"].items()]


def _build_platform_zero_sign(path=None, *, platform=None) -> list:
    """The printed cells whose zero is compared without its sign, as path
    patterns. Empty on the capture platform, like the allowance."""
    entry = _platform_carve_out(path)
    if (platform or sys.platform) == entry["captured_on_platform"]:
        return []
    return list(entry["printed_zero_sign"]["paths"])


#: The leaves allowed a tolerance of their own, and how much. Two policy
#: entries feed it. COMPARE_POLICY.expected_to_move_beyond_rtol_in_step_2
#: carries the measured sizes and the reasoning for what step 2 moved: the
#: MUPE and ZMPE rows of every compare_fitting_methods frame, which move when
#: the private _fit_mupe / _fit_zmpe are replaced by
#: cost_core.fitting.fit_all_methods because the two estimators stop on
#: different rules, and derived_stats Bias, which is a near-zero quantity that
#: needs an absolute allowance rather than the general one.
#: COMPARE_POLICY.expected_to_move_across_platforms adds the MUPE and ZMPE
#: results of the lots golden's learning-curve block, off the platform the
#: goldens were captured on and only there. Nothing else is given a tolerance
#: of its own; what is dropped from the comparison entirely is in
#: _build_exclusions and _build_platform_exclusions.
OVERRIDES: dict = {**_build_overrides(), **_build_platform_overrides()}

#: Whether this run is on the platform the goldens were captured on. Every
#: cross-platform allowance in this module is empty when it is.
ON_CAPTURE_PLATFORM: bool = sys.platform == _platform_carve_out()["captured_on_platform"]

#: The three cases whose iteration bookkeeping IS golden -- they exist to pin
#: the loop counter itself (COMPARE_POLICY exclude_from_step2, second item).
ITERATION_GOLDEN_CASES = ("app_example_maxiter_1", "app_example_maxiter_2", "app_example_tol_1e-14")

# Exclusions, as (compiled path pattern, why). Everything here is provenance, a
# loop counter, a documented app/lib configuration difference, or an
# informational field -- never a number the engine computes.
_EXCLUDE_STEP2 = [
    (r"(^|/)iteration_detail($|/)",
     "iteration_detail: Iter/Delta of the midpoint loop (COMPARE_POLICY exclude_from_step2); golden only for "
     + ", ".join(ITERATION_GOLDEN_CASES)),
    (r"(^|/)ctx/cfg/ToolVersion$", "ctx.cfg.ToolVersion: the tool leaves it None, the library stamps '2.0-dev'"),
    (r"(^|/)dtypes($|/)", "frame dtypes are informational and differ between pandas 2 and pandas 3"),
    (r"(^|/)warnings($|/|\[)", "warnings lists are informational (COMPARE_POLICY exclude_from_step2, last item)"),
    (r"(^|/)capture_warnings($|/|\[)", "capture_warnings is informational"),
    (r"(^|/)stdout_lines($|\[)", "cli stdout_lines are informational and carry pandas' own frame rendering"),
]
# The step-2 carve-outs the policy spells out, appended so they are visible in
# the same list as everything else the comparator skips.
_EXCLUDE_STEP2 += _build_exclusions()
# And the cross-platform one, which is empty on the platform the goldens were
# captured on.
_EXCLUDE_STEP2 += _build_platform_exclusions()

#: A printed zero, with or without a sign: "-0.00%", "+0.00%", "0.0000".
_PRINTED_ZERO = re.compile(r"^[+-]?0(\.0+)?%?$")

#: The cells COMPARE_POLICY.expected_to_move_across_platforms.printed_zero_sign
#: names; empty on the capture platform.
_PLATFORM_ZERO_SIGN = [re.compile(p) for p in _build_platform_zero_sign()]


def _same_printed_zero(path: str, g, n, patterns=None) -> bool:
    """True when both cells print the same zero, differing at most in its sign,
    and the policy names the cell: the sign of a value below the last printed
    digit is the one thing allowed to differ. A printed digit on either side,
    or any other change in how the zero is written, is still a mismatch."""
    return (isinstance(g, str) and isinstance(n, str)
            and bool(_PRINTED_ZERO.fullmatch(g)) and bool(_PRINTED_ZERO.fullmatch(n))
            and g.lstrip("+-") == n.lstrip("+-")
            and any(p.search(path) for p in (_PLATFORM_ZERO_SIGN if patterns is None else patterns)))

#: DFFITS at fewer than two degrees of freedom: {"column", "accept_new",
#: "golden_abs_at_least"} from the policy. The allowance is only taken when the
#: new value is exactly the accepted one and the golden is larger than a DFFITS
#: can meaningfully be, so a DFFITS that merely moved still fails.
_DFFITS_RULE = _step2_carve_out()["dffits_below_two_df"]

#: Text leaves whose MUPE and ZMPE table rows move with the carve-out, and the
#: lines to drop from both sides before comparing the rest byte for byte.
_MASKED_TEXTS = [re.compile(p) for p in _step2_carve_out()["printed_mupe_zmpe_rows"]["texts"]]
_MASKED_LINES = [re.compile(p) for p in _step2_carve_out()["printed_mupe_zmpe_rows"]["mask_lines"]]


def _mask_printed_rows(text: str) -> str:
    """Drop the carved-out table rows from an assumption log."""
    return "\n".join(line for line in text.splitlines()
                     if not any(p.search(line) for p in _MASKED_LINES))


#: Empty on purpose. Step 3 replaced the lognormal handoff with raw draws and
#: moved the programme-level percentiles, and the choice was to rebaseline that
#: block and keep comparing it rather than to stop comparing it. COMPARE_POLICY
#: exclude_after_step3 now records what was rebaselined and by how much. The
#: stage-3 branch below stays so a later step has somewhere to put an exclusion
#: it can justify, and so the two stages keep meaning what the docstring says.
_EXCLUDE_AFTER_STEP3 = []

# ctx.cfg is the settings dict the run was given. Compared for exact equality so
# a changed default is caught rather than absorbed by a tolerance.
_CFG_EXACT = re.compile(r"(^|/)ctx/cfg/")

# Which frames each rounded_leaves.atol_by_column block applies to. The policy
# names them in prose; these are the same frames as paths. A block whose key
# starts with the given prefix is tried against the frame's path, in policy
# order, and the first block that carries the column name wins.
_BLOCK_SCOPES = [
    ("engine projections (also wbs elements[].projections",
     r"(^|/)projections$"),
    ("engine fit_chart_data (also lots fit.chart",
     r"(^|/)(fit_chart_data|chart)$"),
    ("intervals: engine enrich.<model>.projection_intervals_<level>",
     r"projection_intervals_|(^|/)Prediction_Intervals$|(^|/)intervals$|(^|/)intervals_0\.\d+$"
     r"|(^|/)forecast/[^/]+/ok$"),
    ("risk_run_risk.intervals / risk_variants.*.intervals extra columns",
     r"^/(risk_run_risk|risk_variants/[^/]+|test_workbook_fixture)/.*(^|/)intervals$"),
    ("risk_run_risk / risk_variants.* / test_workbook_fixture.risk scalars",
     r"^/(risk_run_risk|risk_variants/[^/]+|test_workbook_fixture)/"),
    ("risk_run_risk.scurve",
     r"^/(risk_run_risk|risk_variants/[^/]+|test_workbook_fixture)/.*(^|/)scurve$"),
    ("wbs deterministic.by_lot and overrides.*.ok.by_lot",
     r"(^|/)by_lot$"),
    ("wbs elements[].projections stub",
     r"/elements\[\d+\]/projections$"),
    ("wbs element_summary and program_level_percentiles.element_summary_with_risk",
     r"(^|/)element_summary(_with_risk)?$"),
    ("wbs by_fiscal_year",
     r"(^|/)by_fiscal_year$"),
    ("wbs buy_profile_sensitivity and buy_profile_sensitivity_variants",
     r"buy_profile_sensitivity"),
    ("wbs program_level_percentiles.scurve",
     r"(^|/)program_level_percentiles/scurve$"),
    ("lots price_lot_plan.*",
     r"/price_lot_plan/"),
    ("lots fit.lot_midpoints and per_lot",
     r"(^|/)(lot_midpoints|per_lot)($|/ok$)"),
]


def load_policy(path=None) -> dict:
    """Read COMPARE_POLICY.json and check every block in it is one this
    comparator knows how to scope. A policy that grew a block would otherwise
    be silently ignored."""
    policy = json.loads(Path(path or POLICY_PATH).read_text(encoding="utf-8"))
    blocks = list(policy["rounded_leaves"]["atol_by_column"])
    prefixes = [p for p, _ in _BLOCK_SCOPES]
    unrouted = [b for b in blocks if not any(b.startswith(p) for p in prefixes)]
    if unrouted:
        raise AssertionError(
            "COMPARE_POLICY.rounded_leaves.atol_by_column has blocks this comparator does not scope; "
            "add them to _BLOCK_SCOPES: " + repr(unrouted))
    return policy


def _compiled_scopes(policy):
    """[(compiled scope, {column: {dp, half_unit, one_unit}})] in policy order."""
    out = []
    by_col = policy["rounded_leaves"]["atol_by_column"]
    for prefix, pattern in _BLOCK_SCOPES:
        for block, cols in by_col.items():
            if block.startswith(prefix):
                out.append((re.compile(pattern), cols))
    return out


# Leaves that are sums of rounded leaves. Each entry is (pattern, n_terms), where
# n_terms(match, walk) returns the number of rounded terms in the sum, or None to
# leave the leaf at the default rtol -- which is stricter, never looser.
def _n_projection_rows(m, w):
    col = m.group("col")
    if not w.is_rounded_projection_column(col):
        return None
    proj = (w.golden_root or {}).get("projections")
    return proj["shape"][0] if isinstance(proj, dict) and "shape" in proj else None


def _n_element_lots(m, w):
    by_lot = (w.parent_golden or {}).get("by_lot")
    return len(by_lot) if isinstance(by_lot, list) else None


def _n_program_terms(m, w):
    els = (w.parent_golden or {}).get("elements")
    if not isinstance(els, list):
        return None
    return sum(len(e.get("by_lot") or []) for e in els if isinstance(e, dict))


def _n_simulated_run_terms(m, w):
    det = (w.golden_root or {}).get("deterministic") or {}
    els = det.get("elements")
    if not isinstance(els, list):
        return None
    return sum(len(e.get("by_lot") or []) for e in els if isinstance(e, dict))


def _n_buy_lots(m, w):
    per_lot = (w.parent_golden or {}).get("per_lot_mean")
    return len(per_lot) if isinstance(per_lot, list) else None


def _n_interval_rows(m, w):
    iv = (w.parent_golden or {}).get("intervals")
    if isinstance(iv, dict) and isinstance(iv.get("records"), list):
        return len(iv["records"])
    return None


_SUM_RULES = [
    (re.compile(r"^/projections_column_sums/(?P<col>.+)$"), _n_projection_rows),
    (re.compile(r"/elements\[\d+\]/by_lot\[\d+\]$"), lambda m, w: 1),
    (re.compile(r"/elements\[\d+\]/total$"), _n_element_lots),
    (re.compile(r"/(deterministic|ok)/total$"), _n_program_terms),
    (re.compile(r"^/simulated_run_point_total$"), _n_simulated_run_terms),
    (re.compile(r"/point_estimate$"), _n_buy_lots),
    (re.compile(r"/total_point$"), _n_interval_rows),
]

#: Frame columns that are themselves sums of 2-dp values, with how many terms.
#: (frame scope, column) -> n_terms(record index, frame, walk) .
_SUM_COLUMNS = [
    (re.compile(r"/price_lot_plan/"), "cumulative_cost", lambda i, fj, w: i + 1, 0.01),
    (re.compile(r"(^|/)by_fiscal_year$"), "Total ($)", lambda i, fj, w: _n_fy_terms(i, fj), 0.01),
    (re.compile(r"(^|/)by_fiscal_year$"), "Cumulative ($)",
     lambda i, fj, w: sum(_n_fy_terms(j, fj) for j in range(i + 1)) + 1, 0.01),
]

#: by_fiscal_year carries one column per element plus these bookkeeping ones.
#: The names matter: the frame is built by rollup.by_fiscal_year (:1064-1071)
#: with "Fiscal Year" and "Lots", NOT "FY" and "Lot". Naming them wrong counted
#: two bookkeeping columns as elements and made the tolerance below too loose.
_FY_NON_ELEMENT_COLUMNS = ("Fiscal Year", "Lots", "Total ($)", "Cumulative ($)", "Share of Program")


def _n_fy_element_columns(fj) -> int:
    """How many element columns the by_fiscal_year frame carries."""
    return max(1, len([c for c in fj.get("columns", []) if c not in _FY_NON_ELEMENT_COLUMNS]))


def _n_fy_terms(i, fj) -> int:
    """n_terms for by_fiscal_year 'Total ($)' on row `i`.

    ``Total ($)`` is ``round(sum of the year's by_lot 'Program Total ($)'
    cells, 2)`` (rollup.py:1067-1069), and COMPARE_POLICY already treats each
    of those cells as a single rounded leaf worth one unit
    (rounded_leaves.atol_by_column, "wbs deterministic.by_lot"). So the terms
    are the lots awarded in that year -- the frame's own "Lots" column -- plus
    one unit for the outer re-round.

    The lot count is read off the golden's frame rather than guessed. If the
    column is missing (an older or hand-built frame), fall back to the number
    of element columns, which is the other reading of the policy's "n_terms"
    and is never smaller than one.
    """
    recs = fj.get("records") or []
    if 0 <= i < len(recs):
        lots = recs[i].get("Lots")
        if isinstance(lots, (int, float)) and not isinstance(lots, bool) and lots == lots:
            return int(lots) + 1
    return _n_fy_element_columns(fj)


class _Walk:
    """State the tolerance rules read while compare_with_policy walks a tree."""

    def __init__(self, policy, stage, case, new_only_items=()):
        self.policy = policy
        self.stage = stage
        self.case = case
        self.new_only_items = tuple(new_only_items)
        self.rtol = float(policy["tolerance"]["rtol"])
        self.atol = float(policy["tolerance"]["atol"])
        self.scopes = _compiled_scopes(policy)
        self.mismatches = []
        self.golden_root = None
        self.parent_golden = None
        excl = list(_EXCLUDE_STEP2) + (list(_EXCLUDE_AFTER_STEP3) if stage >= 3 else [])
        self.exclusions = [(re.compile(p), why) for p, why in excl]
        proj_block = next((cols for prefix, _ in _BLOCK_SCOPES[:1]
                           for block, cols in policy["rounded_leaves"]["atol_by_column"].items()
                           if block.startswith(prefix)), {})
        self._projection_columns = set(proj_block)

    def is_rounded_projection_column(self, col: str) -> bool:
        return col in self._projection_columns

    def excluded(self, path: str):
        for pat, why in self.exclusions:
            if pat.search(path):
                if pat.pattern.startswith("(^|/)iteration_detail") and self.case in ITERATION_GOLDEN_CASES:
                    continue
                return why
        return None

    def rounded_atol(self, frame_path: str, column: str):
        """(atol, dp) for a rounded frame column, or (None, None)."""
        for scope, cols in self.scopes:
            if not scope.search(frame_path):
                continue
            spec = cols.get(column) or cols.get("<element>")
            if spec is not None:
                # both sides are rounded here: golden and new value come out of
                # the same rounding code, so a boundary flip is a whole unit
                return spec["one_unit"] * (1.0 + 1e-6) + self.atol, spec["dp"]
        return None, None

    def add(self, path, golden, new, rule, tolerance=None):
        self.mismatches.append({"path": path, "golden": golden, "new": new,
                                "rule": rule, "tolerance": tolerance})


def _is_frame(x) -> bool:
    return isinstance(x, dict) and "records" in x and "columns" in x


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_number(x):
    """``x`` as a float when it is a string spelling one, else ``x`` unchanged."""
    if isinstance(x, str):
        try:
            v = float(x)
        except ValueError:
            return x
        return v if math.isfinite(v) else x
    return x


def _dffits_zeroed(column, golden, new) -> bool:
    """True where DFFITS went to zero because the fit has fewer than two df.

    See COMPARE_POLICY expected_to_move_beyond_rtol_in_step_2.dffits_below_two_df.
    Deliberately narrow: the new value has to be exactly zero and the golden
    has to be larger than any DFFITS can meaningfully be, so a DFFITS that
    merely moved is still a failure.
    """
    if column != _DFFITS_RULE["column"] or not (_num(golden) and _num(new)):
        return False
    return (float(new) == float(_DFFITS_RULE["accept_new"])
            and abs(float(golden)) >= float(_DFFITS_RULE["golden_abs_at_least"]))


def _grid_slack(a: float, b: float) -> float:
    """The representation allowance the rounded and sum limits need.

    The policy's limits are absolute -- one unit in the last decimal place, or
    ``n_terms`` of them -- and its ``(1 + 1e-6)`` factor is there because a
    flip is "exactly one unit". It is not exactly one unit as a float: a
    rounded value is ``fl(round(x, dp))``, which sits within half an ulp of the
    decimal grid point, so the measured gap between two adjacent grid points
    differs from ``one_unit`` by a couple of ulps of the value. At $1e8 with
    2-dp rounding that is 1.5e-6 of a unit, already more than the 1e-6 the
    policy allows, and on the ``LC F`` column (1e26, where a 2-dp grid is far
    below the float spacing) a unit does not exist at all and the only
    meaningful tolerance is a few ulps.

    Four ulps of the larger side covers that and nothing else: it is 1e-15
    relative, six orders of magnitude tighter than the general rule's rtol
    1e-9, which is what used to be added here instead.
    """
    v = max(abs(a), abs(b))
    if not math.isfinite(v) or v == 0.0:
        return 0.0
    return 4.0 * math.ulp(v)


def _override_tol(path: str):
    """The carve-out for ``path``, as ``{"rtol": ..., "atol": ...}`` or None.

    A bare number is read as an rtol, which is how the hook was first
    documented.
    """
    for glob, tol in OVERRIDES.items():
        if fnmatch.fnmatch(path, glob):
            return glob, ({"rtol": float(tol)} if _num(tol) else dict(tol))
    return None, None


def _numeric_leaf(path, g, n, w, *, frame_path=None, column=None, rec_index=None, frame=None):
    """Compare one numeric leaf under the policy. Returns None or a mismatch."""
    if isinstance(g, bool) or isinstance(n, bool):
        if g is not n:
            w.add(path, g, n, "exact (boolean)")
        return
    if isinstance(g, int) and isinstance(n, int):
        if g != n:
            w.add(path, g, n, "exact (integer)")
        return
    if isinstance(g, int) != isinstance(n, int) and column is None:
        # An int that turned into a float (or back) is a type change like any
        # other and is reported -- ctx.mdl_*.N / K / DF and ctx.n_keep are ints
        # and an authority leaf turning into a float must be seen. Inside frame
        # records it is tolerated instead, and only there: the goldens were
        # captured on pandas 3.0.3 and the 3.9 lane runs pandas 2.3.3, which
        # disagree about when an integer column stays integer (a column that
        # gains a missing value goes float on one and not the other). The
        # VALUE is still compared, below, at the leaf's own tolerance.
        w.add(path, g, n, "structural: type differs (int vs float)")
        return
    if _CFG_EXACT.search(path):
        if g != n:
            w.add(path, g, n, "exact (ctx.cfg: a changed engine default must be seen, not tolerated)")
        return

    rule = "tolerance rtol/atol"
    rtol, atol = w.rtol, w.atol
    # The policy states the rounded and sum rules as ABSOLUTE limits --
    # "one_unit * (1 + 1e-6) + atol" and "n_terms * one_unit", neither of which
    # carries an rtol term (rounded_leaves.rule, rounded_leaves.sums_of_rounded).
    # Adding the general rule's rtol*|value| on top would make the tolerance on
    # a large rounded cell many units wide (a 1e26 'LC F' would get 1.8e19
    # units), which is exactly what "a leaf that moves by more than one unit is
    # a real regression" forbids. So when either rule fires the limit is
    # absolute, and only then.
    absolute = False
    glob, override = _override_tol(path)
    if override is not None:
        rtol = float(override.get("rtol", rtol))
        atol = float(override.get("atol", atol))
        rule = f"OVERRIDES[{glob!r}]"
    else:
        atol_sum = None
        if column is not None:
            for scope, col, n_terms, unit in _SUM_COLUMNS:
                if col == column and scope.search(frame_path or ""):
                    k = n_terms(rec_index, frame or {}, w)
                    if k:
                        atol_sum, rule = k * unit, f"sums_of_rounded (n_terms={k} x {unit})"
                    break
        if atol_sum is None:
            for pat, n_terms in _SUM_RULES:
                m = pat.search(path)
                if m is None:
                    continue
                k = n_terms(m, w)
                if k:
                    atol_sum, rule = k * 0.01, f"sums_of_rounded (n_terms={k} x 0.01)"
                break
        if atol_sum is not None:
            atol, absolute = atol_sum, True
        elif column is not None:
            a, dp = w.rounded_atol(frame_path or "", column)
            if a is not None:
                atol, absolute = a, True
                rule = f"rounded_leaves.atol_by_column[{column!r}] dp={dp}, both sides rounded"
    gf, nf = float(g), float(n)
    if math.isnan(gf) and math.isnan(nf):
        return
    if absolute:
        slack = _grid_slack(gf, nf)
        limit = atol + slack
        how = (f"absolute, atol {atol!r} + {slack!r} of decimal-grid slack "
               f"(the policy's rounded/sum rule carries no rtol term)")
    else:
        limit = atol + rtol * max(abs(gf), abs(nf))
        how = f"rtol {rtol!r}, atol {atol!r}"
    if not (abs(gf - nf) <= limit):
        w.add(path, g, n, rule, f"|new-golden| <= {limit!r} ({how})")


def _compare_records(path, gr, nr, w, frame_path, frame, index):
    """One record of a DataFrame leaf. Missing values compare equal here and
    only here (COMPARE_POLICY missing_values_in_frames)."""
    for key in sorted(set(gr) | set(nr)):
        p = f"{path}/{key}"
        if w.excluded(p):
            continue
        if key not in gr or key not in nr:
            w.add(p, gr.get(key, "<absent>"), nr.get(key, "<absent>"), "structural: column present on one side only")
            continue
        g, n = gr[key], nr[key]
        if g in MISSING_TOKENS and n in MISSING_TOKENS:
            continue
        if _dffits_zeroed(key, g, n):
            continue
        if not (_num(g) and _num(n)) and "/csv_contents/" in frame_path:
            # A CSV column holding both names and numbers comes back from
            # read_csv as text, and comparing that text byte for byte is not
            # what the policy says csv_contents means. Where both sides parse
            # as a number, compare them as numbers.
            g, n = _as_number(g), _as_number(n)
        if _num(g) and _num(n):
            _numeric_leaf(p, g, n, w, frame_path=frame_path, column=key, rec_index=index, frame=frame)
        elif g != n and not _same_printed_zero(p, g, n):
            w.add(p, gr[key], nr[key], "exact (frame cell)")


def _compare_frame(path, gf, nf, w):
    if gf.get("columns") != nf.get("columns"):
        w.add(f"{path}/columns", gf.get("columns"), nf.get("columns"), "structural: frame columns differ")
        return
    grecs, nrecs = gf.get("records") or [], nf.get("records") or []
    if "Item" in gf.get("columns", []):
        # Item-keyed frames (summary, model_comparison, equation_detail,
        # program_summary) are aligned on Item, not on row number, because a
        # provenance row that is stripped from one side but not the other would
        # otherwise shift every row below it. Only PROVENANCE_ITEMS -- the two
        # rows split_summary strips out of every golden -- are skipped, so the
        # set of Items IS compared: a row that appears, disappears or is
        # renamed is a structural mismatch here, and since every row is keyed
        # (duplicates and blanks included, see _item_keys) that comparison
        # covers the row count as completely as `shape` would.
        gk, nk = _item_keys(grecs), _item_keys(nrecs)
        gmap, nmap = dict(zip(gk, grecs)), dict(zip(nk, nrecs))
        index_of = {k: i for i, k in enumerate(gk)}
        for k in sorted(set(gmap) | set(nmap)):
            base = k.split("#")[0]
            if base in PROVENANCE_ITEMS:
                continue
            p = f"{path}/[Item={k}]"
            if k not in gmap and base in w.new_only_items:
                # this golden predates the row; the caller said which and why
                continue
            if k not in gmap or k not in nmap:
                w.add(p, "<present>" if k in gmap else "<absent>", "<present>" if k in nmap else "<absent>",
                      "structural: summary row present on one side only")
                continue
            _compare_records(p, gmap[k], nmap[k], w, path, gf, index_of.get(k, 0))
        return
    if gf.get("shape") != nf.get("shape"):
        w.add(f"{path}/shape", gf.get("shape"), nf.get("shape"), "structural: frame shape differs")
        return
    if len(grecs) != len(nrecs):
        w.add(f"{path}/records", len(grecs), len(nrecs), "structural: record count differs")
        return
    for i, (gr, nr) in enumerate(zip(grecs, nrecs)):
        _compare_records(f"{path}/records[{i}]", gr, nr, w, path, gf, i)


def _walk(path, g, n, w):
    why = w.excluded(path)
    if why:
        return
    if _is_frame(g) and _is_frame(n):
        _compare_frame(path, g, n, w)
        return
    if isinstance(g, dict) and isinstance(n, dict):
        parent = w.parent_golden
        w.parent_golden = g
        for k in sorted(set(g) | set(n)):
            p = f"{path}/{k}"
            if w.excluded(p):
                continue
            if k not in g or k not in n:
                w.add(p, "<absent>" if k not in g else _short(g[k]), "<absent>" if k not in n else _short(n[k]),
                      "structural: key present on one side only")
                continue
            _walk(p, g[k], n[k], w)
        w.parent_golden = parent
        return
    if isinstance(g, list) and isinstance(n, list):
        if len(g) != len(n):
            w.add(path, len(g), len(n), "structural: list length differs")
        for i, (x, y) in enumerate(zip(g, n)):
            _walk(f"{path}[{i}]", x, y, w)
        return
    if isinstance(g, (dict, list)) != isinstance(n, (dict, list)):
        w.add(path, _short(g), _short(n), "structural: type differs")
        return
    if _num(g) and _num(n):
        _numeric_leaf(path, g, n, w)
        return
    if type(g) is not type(n) and not (g is None and n is None):
        w.add(path, _short(g), _short(n), "structural: type differs")
        return
    if isinstance(g, str) and isinstance(n, str) and any(p.search(path) for p in _MASKED_TEXTS):
        if _mask_printed_rows(g) != _mask_printed_rows(n):
            w.add(path, _short(g), _short(n),
                  "exact, with the carved-out MUPE and ZMPE table rows masked out of both sides")
        return
    if g != n:
        w.add(path, _short(g), _short(n), "exact")


def _short(v, limit=300):
    s = v if isinstance(v, str) else json.dumps(v, default=repr)
    return s if len(s) <= limit else s[:limit] + f"... (+{len(s) - limit} chars)"


def compare_with_policy(golden, new, policy, *, stage, case=None, new_only_items=()):
    """Compare a golden tree with a freshly computed one under COMPARE_POLICY.

    `stage` is 2, which applies exclude_from_step2. Stage 3 adds
    _EXCLUDE_AFTER_STEP3, which is empty: step 3 rebaselined the
    programme-level percentiles instead of excluding them, so nothing stopped
    being compared. `case` names the engine case, which is what lets the three
    maxiter/tol cases keep their iteration_detail.

    `new_only_items` names summary Items this particular golden is allowed to
    lack -- for a golden captured before the library grew the row. It is empty
    everywhere except ``lots_cost_core`` (see LOTS_GOLDEN_PREDATES_ITEMS); an
    Item not named here that is on one side only is a structural mismatch.

    Returns a list of {"path", "golden", "new", "rule", "tolerance"} records,
    empty when the two agree. Nothing is ever silently skipped: a key or list
    element on one side only, a length mismatch and a type mismatch are all
    reported as structural mismatches.
    """
    w = _Walk(policy, stage, case, new_only_items)
    w.golden_root = golden if isinstance(golden, dict) else None
    _walk("", golden, new, w)
    return w.mismatches


def format_mismatches(mismatches, limit=40) -> str:
    """One line per failing leaf: path, golden, new, the rule that was applied."""
    lines = [f"{len(mismatches)} leaf/leaves disagree with the golden:"]
    for m in mismatches[:limit]:
        line = f"  {m['path'] or '/'}\n      golden = {_short(m['golden'], 160)}\n      new    = {_short(m['new'], 160)}\n      rule   = {m['rule']}"
        if m.get("tolerance"):
            line += f"\n      tol    = {m['tolerance']}"
        lines.append(line)
    if len(mismatches) > limit:
        lines.append(f"  ... and {len(mismatches) - limit} more")
    return "\n".join(lines)
