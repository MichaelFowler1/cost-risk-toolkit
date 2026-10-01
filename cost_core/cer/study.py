# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
study.py - A CER study kept in Excel: fit it, compare the methods, estimate.

What ``ce-core cer`` runs. The workbook has a Data sheet (one row per past
program: its name, its cost and the technical drivers), an optional Estimate
sheet (the new programs to price, by the same drivers) and an optional
Settings sheet (which column is the cost, which are the drivers, the form,
the method, the interval). ``ce-core template cer`` writes one with an
invented example in it.

Everything the engine in :mod:`cost_core.cer.model` already does is used as
it stands: the three fitting methods side by side, prediction (not
confidence) intervals, the small-sample guardrails, the leverage and
influence diagnostics and the extrapolation check. This module only turns a
workbook into those calls and their results into tables, and says in plain
words what a reviewer will ask about.

Rows can be left out of the fit with a Use column ("no") and a Note saying
why, which is how an analyst keeps an outlier in the data set without fitting
to it. A row with a blank cost or driver is left out too, and the notes name
it, so nothing disappears without a word.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from cost_core.cer.model import CER, Form, cer_comparison_table, fit_cer
from cost_core.costrisk import (CostRiskError, _blank, _norm, excel_rows, headingless,
                                open_workbook)
from cost_core.costrisk import _number as _costrisk_number
from cost_core.fitting import METHODS, FitError, retransformation_bias


class CerError(ValueError):
    """A CER workbook that can't be used; the message says where and why."""


#: Headings read as each column, as normalised headings.
LABEL_ALIASES = ("program", "name", "system", "item", "label", "project", "data_point",
                 "element")
COST_ALIASES = ("cost", "actual_cost", "actuals", "total_cost", "unit_cost", "auc",
                "first_unit_cost", "t1", "price", "value", "hours", "effort")
USE_ALIASES = ("use", "include", "used", "in_fit")
NOTE_ALIASES = ("note", "notes", "comment", "comments", "why", "reason", "source")

YES = {"yes", "y", "true", "1", "x", "include", "included", "in"}
#: Columns never taken as a driver unless the Settings sheet names them: a
#: year says when, not what, and fitting cost to it is rarely what's meant.
NOT_DRIVERS = {"year", "fy", "fiscal_year", "dollar_year", "base_year", "by", "ty",
               "lot", "id", "row", "number", "no"}
NO = {"no", "n", "false", "0", "exclude", "excluded", "out", "drop"}

#: The Settings sheet's labels, as normalised headings, and what each sets.
SETTING_NAMES = {"cost": "cost", "cost_column": "cost", "response": "cost", "y": "cost",
                 "drivers": "drivers", "driver": "drivers", "predictors": "drivers",
                 "x": "drivers", "form": "form", "method": "method",
                 "confidence": "confidence", "interval": "confidence", "level": "confidence",
                 "units": "units", "dollar_year": "dollar_year", "base_year": "dollar_year",
                 "description": None, "notes": None}
SETTINGS_DEFAULTS = {"cost": None, "drivers": None, "form": "log-log", "method": "mupe",
                     "confidence": 0.8, "units": "", "dollar_year": ""}

FORM_WORDS = {"log_log": Form.LOG_LOG, "loglog": Form.LOG_LOG, "log": Form.LOG_LOG,
              "power": Form.LOG_LOG, "multiplicative": Form.LOG_LOG,
              "linear": Form.LINEAR, "additive": Form.LINEAR}

#: A driver whose coefficient has a p-value above this is called out.
P_FLAG = 0.10


# ------------------------------------------------------------------ reading
def _number(value, sheet: str, row: int, what: str) -> Optional[float]:
    """A cell as a number, None when blank; a CerError naming the cell when
    it holds text."""
    try:
        return _costrisk_number(value, sheet, row, what)
    except CostRiskError as e:
        raise CerError(str(e)) from None


def _first(frame: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    have = {_norm(c): c for c in frame.columns}
    return next((have[a] for a in aliases if a in have), None)


def _heading(frame: pd.DataFrame, name: str, sheet: str) -> str:
    """The sheet's heading for a column named in the Settings or a flag."""
    have = {_norm(c): c for c in frame.columns}
    found = have.get(_norm(name))
    if found is None:
        raise CerError(f"The {sheet} sheet has no column {name!r}. Its headings are "
                       f"{', '.join(repr(str(c)) for c in frame.columns)}.")
    return found


def _flag(value) -> str:
    """A Use cell as text: TRUE, 1, 1.0 and "yes" all read as yes."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _norm(value)


def _form(value) -> Form:
    form = FORM_WORDS.get(_norm(value))
    if form is None:
        raise CerError(f"The form {value!r} isn't one cost-core fits: log-log "
                       "(cost = a * driver^b, the usual one for cost) or linear "
                       "(cost = a + b * driver).")
    return form


def _method(value) -> str:
    method = _norm(value)
    if method not in METHODS:
        raise CerError(f"The method {value!r} isn't one of OLS, MUPE or ZMPE.")
    return method


def _level(value) -> float:
    try:
        text = str(value).strip()
        level = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    except ValueError:
        raise CerError(f"The confidence {value!r} isn't a number; write 0.8 or 80%.") from None
    if level > 1.0:
        level /= 100.0
    if not 0.5 <= level < 1.0:
        raise CerError(f"A confidence of {level:g} isn't usable for a prediction interval; "
                       "give one from 0.5 to 0.99 (0.8 is usual).")
    return level


def _read_settings(frame: Optional[pd.DataFrame]) -> Tuple[Dict, frozenset]:
    out = dict(SETTINGS_DEFAULTS)
    stated = set()
    if frame is None:
        return out, frozenset()
    frame = headingless(frame, SETTING_NAMES).dropna(how="all")
    for row, rec in zip(excel_rows(frame), frame.itertuples(index=False)):
        if len(rec) < 2 or _blank(rec[0]):
            continue
        label = _norm(rec[0])
        if label not in SETTING_NAMES:
            raise CerError(f"Settings sheet, row {row}: {str(rec[0]).strip()!r} is not a "
                           "setting. The settings are Cost, Drivers, Form, Method, "
                           "Confidence, Units and Dollar Year.")
        key, value = SETTING_NAMES[label], rec[1]
        if key is None or _blank(value):
            continue
        out[key] = value if key == "confidence" else str(value).strip()
        stated.add(key)
    return out, frozenset(stated)


@dataclass
class CerInput:
    """What the workbook says: the data, what to fit, and what to price."""

    name: str
    data: pd.DataFrame
    response: str
    predictors: Tuple[str, ...]
    label: Optional[str]
    form: Form = Form.LOG_LOG
    method: str = "mupe"
    level: float = 0.8
    units: str = ""
    dollar_year: str = ""
    estimate: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: List[str] = field(default_factory=list)
    stated: frozenset = frozenset()

    @property
    def used(self) -> pd.DataFrame:
        """The rows the CER is fitted to."""
        return self.data[self.data["_used"]]


def read_workbook(path, cost: Optional[str] = None,
                  drivers: Optional[Sequence[str]] = None) -> CerInput:
    """Read a CER workbook, or a CSV of the data alone.

    ``cost`` and ``drivers`` name columns, overriding the Settings sheet.
    """
    path = Path(path)
    unread: List[str] = []
    if path.suffix.lower() == ".csv":
        sheets = {"data": pd.read_csv(path)}
    else:
        raw = open_workbook(path, CerError, ", or give the data alone as a .csv")
        sheets = {_norm(k): v for k, v in raw.items()}
        titles = {_norm(k): str(k) for k in raw}
        aliases = {"data": ("data", "cer_data", "programs", "history", "historical_data"),
                   "estimate": ("estimate", "estimates", "new_program", "new_programs",
                                "predict", "prediction", "predictions")}
        for key, names in aliases.items():
            found = [a for a in names if a in sheets]
            if found and found[0] != key:
                sheets[key] = sheets.pop(found[0])
                titles[key] = titles.pop(found[0])
        if "data" not in sheets:
            first = next(iter(raw), None)
            if first is None:
                raise CerError(f"{path.name} has no sheets.")
            sheets["data"] = raw[first]
            titles["data"] = titles.pop(_norm(first))
        unread = [t for k, t in titles.items()
                  if k not in ("data", "estimate", "settings", "instructions")]
    settings, stated = _read_settings(sheets.get("settings"))
    if cost:
        settings["cost"], stated = cost, stated | {"cost"}
    if drivers:
        settings["drivers"], stated = list(drivers), stated | {"drivers"}

    frame = sheets["data"].dropna(how="all")
    if frame.empty:
        raise CerError("The Data sheet has no rows: one row per past program, with its "
                       "cost and its drivers.")
    label = _first(frame, LABEL_ALIASES)
    use_col, note_col = _first(frame, USE_ALIASES), _first(frame, NOTE_ALIASES)
    if settings["cost"]:
        response = _heading(frame, settings["cost"], "Data")
    else:
        response = _first(frame, COST_ALIASES)
        if response is None:
            raise CerError("The Data sheet has no Cost column. Head the column Cost, or "
                           "name it on the Settings sheet (Cost: Actual $M, say).")
    taken = {label, use_col, note_col, response}
    if settings["drivers"]:
        names = settings["drivers"]
        if isinstance(names, str):
            names = [n for n in re.split(r"[;,]", names) if n.strip()]
        predictors = tuple(_heading(frame, n.strip(), "Data") for n in names)
    else:
        # Every other column that holds numbers, except a year.
        numeric = [c for c in frame.columns if c not in taken
                   and pd.to_numeric(frame[c], errors="coerce").notna().any()]
        predictors = tuple(c for c in numeric if _norm(c) not in NOT_DRIVERS)
        skipped = [str(c) for c in numeric if _norm(c) in NOT_DRIVERS]
        if skipped:
            notes_early = [f"{', '.join(map(repr, skipped))} {'is' if len(skipped) == 1 else 'are'} "
                           "not used as a driver; name it under Drivers on the Settings "
                           "sheet to fit to it."]
        else:
            notes_early = []
    if not predictors:
        raise CerError("The Data sheet has no driver columns: add one column per technical "
                       "driver (weight, power, lines of code), or name them on the "
                       "Settings sheet.")
    if response in predictors:
        raise CerError(f"{response!r} is both the cost and a driver.")
    dup = sorted({str(p) for p in predictors if list(predictors).count(p) > 1})
    if dup:
        raise CerError(f"The driver(s) {dup} are listed twice.")

    if settings["drivers"]:
        notes_early = []
    form = _form(settings["form"])
    rows, excluded, notes = [], [], list(notes_early)
    for i, rec in zip(excel_rows(frame), frame.to_dict("records")):
        name = str(rec.get(label)).strip() if label and not _blank(rec.get(label)) \
            else f"row {i}"
        values = {c: _number(rec.get(c), "Data", i, str(c)) for c in (response, *predictors)}
        use = True
        why = str(rec.get(note_col)).strip() if note_col and not _blank(rec.get(note_col)) \
            else ""
        if use_col and not _blank(rec.get(use_col)):
            flag = _flag(rec.get(use_col))
            if flag in NO:
                use = False
                excluded.append({"program": name, "row": i,
                                 "why": why or "marked not to use"})
            elif flag not in YES:
                raise CerError(f"Data sheet, row {i} ({name}): Use is yes or no, not "
                               f"{rec.get(use_col)!r}.")
        blank = [str(c) for c, v in values.items() if v is None]
        if use and blank:
            use = False
            excluded.append({"program": name, "row": i,
                             "why": f"no {' or '.join(blank)} given"})
        if use and form is Form.LOG_LOG:
            bad = [(str(c), v) for c, v in values.items() if v <= 0]
            if bad:
                raise CerError(f"Data sheet, row {i} ({name}): a log-log CER needs every "
                               f"value above zero, and {bad[0][0]} is {bad[0][1]:g}. Fix "
                               "the value, set Use to no for this row, or fit the linear "
                               "form.")
        rows.append({"_row": i, "_name": name, "_used": use, "_why": why,
                     **{str(c): values[c] for c in (response, *predictors)}})
    data = pd.DataFrame(rows)
    if excluded:
        shown = "; ".join(f"{e['program']} ({e['why']})" for e in excluded[:6])
        notes.append(f"{len(excluded)} of {len(data)} rows are left out of the fit: {shown}"
                     f"{'; ...' if len(excluded) > 6 else ''}.")
    if unread:
        notes.append(f"The sheet{'s' if len(unread) > 1 else ''} "
                     f"{', '.join(map(repr, unread))} {'were' if len(unread) > 1 else 'was'} "
                     "not read: the sheets read are Data, Estimate and Settings.")

    estimate = _read_estimate(sheets.get("estimate"), predictors, form)
    return CerInput(name=path.stem, data=data, response=str(response),
                    predictors=tuple(str(p) for p in predictors), label=label,
                    form=form, method=_method(settings["method"]),
                    level=_level(settings["confidence"]), units=settings["units"],
                    dollar_year=str(settings["dollar_year"]), estimate=estimate,
                    excluded=pd.DataFrame(excluded, columns=["program", "row", "why"]),
                    notes=notes, stated=stated)


def _read_estimate(frame: Optional[pd.DataFrame], predictors: Sequence[str],
                   form: Form) -> pd.DataFrame:
    """The new programs to price: a name and each driver."""
    empty = pd.DataFrame(columns=["name", *map(str, predictors)])
    if frame is None:
        return empty
    frame = frame.dropna(how="all")
    if frame.empty:
        return empty
    label = _first(frame, LABEL_ALIASES)
    have = {_norm(c): c for c in frame.columns}
    missing = [str(p) for p in predictors if _norm(p) not in have]
    if missing:
        raise CerError(f"The Estimate sheet has no column for {', '.join(map(repr, missing))}; "
                       "it needs the same driver headings as the Data sheet.")
    rows = []
    for i, rec in zip(excel_rows(frame), frame.to_dict("records")):
        name = str(rec.get(label)).strip() if label and not _blank(rec.get(label)) \
            else f"row {i}"
        values = {}
        for p in predictors:
            v = _number(rec.get(have[_norm(p)]), "Estimate", i, str(p))
            if v is None:
                raise CerError(f"Estimate sheet, row {i} ({name}): {p} is blank.")
            if form is Form.LOG_LOG and v <= 0:
                raise CerError(f"Estimate sheet, row {i} ({name}): a log-log CER can't "
                               f"price {p} = {v:g}; it needs a value above zero.")
            values[str(p)] = v
        rows.append({"name": name, **values})
    return pd.DataFrame(rows, columns=empty.columns)


# ---------------------------------------------------------------- analysing
@dataclass
class CerStudy:
    """The CER fitted every way, the one chosen, and what it prices."""

    inputs: CerInput
    fits: Dict[str, CER]
    estimates: pd.DataFrame
    notes: List[str] = field(default_factory=list)
    units: str = ""

    @property
    def cer(self) -> CER:
        """The CER fitted by the chosen method."""
        return self.fits[self.inputs.method]

    def comparison(self) -> pd.DataFrame:
        """OLS, MUPE and ZMPE side by side, the chosen one marked."""
        table = cer_comparison_table(self.fits)
        table.insert(0, "chosen", ["yes" if m == self.inputs.method else ""
                                   for m in self.fits])
        return table.drop(columns=["label"])

    def coefficients(self) -> pd.DataFrame:
        """The chosen fit's coefficients, with what each means in cost."""
        table = self.cer.summary()
        meaning = []
        for name, value in zip(table["parameter"], table["estimate"]):
            driver = name[2:] if name.startswith("b_") else None
            if driver is None:
                meaning.append(f"a = {np.exp(value):,.4g}" if name == "log_a"
                               else "cost at zero for every driver")
            elif self.cer.form is Form.LOG_LOG:
                meaning.append(f"doubling {driver} multiplies cost by {2.0 ** value:,.3f}")
            else:
                meaning.append(f"each unit of {driver} adds {value:,.4g}")
        table["meaning"] = meaning
        return table

    def overall_f(self) -> Tuple[float, float]:
        """The overall F test of the OLS fit on its own scale: (F, p).

        Defined for least squares; reported from the OLS fit whatever the
        chosen method, as a check that the drivers together explain anything.
        """
        ols = self.fits["ols"]
        k = ols.result.n_params - 1
        r2, df = ols.r_squared, ols.df
        if df <= 0 or k <= 0 or not np.isfinite(r2) or r2 >= 1.0:
            return float("nan"), float("nan")
        f = (r2 / k) / ((1.0 - r2) / df)
        return float(f), float(stats.f.sf(f, k, df))

    def bias(self):
        """How far a log-space OLS fit sits below the mean (log-log only)."""
        if self.cer.form is not Form.LOG_LOG:
            return None
        return retransformation_bias(self.fits["ols"].result, self.fits["mupe"].result,
                                     self.fits["zmpe"].result)

    def diagnostics(self) -> pd.DataFrame:
        return self.cer.diagnostics().to_frame()

    def cost_risk_rows(self) -> pd.DataFrame:
        """The estimates as rows for a cost-risk Elements sheet.

        A lognormal whose Low and High are the prediction interval's ends:
        cost-risk reads them as the 80% range by default, so a study at
        another confidence says so in the setting to use alongside.
        """
        est = self.estimates
        return pd.DataFrame({"Element": est["name"], "Point Estimate": est["estimate"],
                             "Low": est["lower"], "Most Likely": np.nan,
                             "High": est["upper"], "Distribution": "lognormal"})

    def data_table(self) -> pd.DataFrame:
        """Every row of the Data sheet, fitted value beside the ones used."""
        d = self.inputs.data
        out = pd.DataFrame({"program": d["_name"], "row": d["_row"],
                            "used": d["_used"].map({True: "yes", False: "no"}),
                            "note": d["_why"]})
        for c in (self.inputs.response, *self.inputs.predictors):
            out[c] = d[c]
        fitted = pd.Series(np.nan, index=d.index)
        fitted[d.index[d["_used"]]] = self.cer.result.fitted
        out["fitted"] = fitted
        out["percent_error"] = (out[self.inputs.response] - fitted) / fitted
        return out

    @property
    def assumptions(self) -> Dict:
        i, c = self.inputs, self.cer
        return {"data": i.name, "cost column": i.response,
                "drivers": ", ".join(i.predictors), "form": c.form.value.replace("_", "-"),
                "method": c.method.upper(), "interval": f"{i.level:.0%} prediction interval",
                "programs used": c.result.n_obs, "rows left out": len(i.excluded),
                "parameters": c.result.n_params, "degrees of freedom": c.df,
                "units": self.units or "as entered",
                "dollar year": i.dollar_year or "not stated"}


#: A variance inflation factor above this is called out: the driver's effect
#: is hard to tell apart from the others'.
VIF_FLAG = 10.0


def _check_drivers(used: pd.DataFrame, predictors: Sequence[str], form: Form) -> List[str]:
    """Refuse drivers a CER can't learn from; note ones it can barely learn from.

    A driver with one value across the programs is indistinguishable from the
    intercept, and drivers that move in lockstep (on the log scale, for a
    log-log CER) can't be told apart: either way the fit would still print
    coefficients and t-statistics, which mean nothing.
    """
    scale = np.log if form is Form.LOG_LOG else (lambda v: v)
    cols = {p: scale(used[p].to_numpy(dtype=float)) for p in predictors}
    words = " on the log scale" if form is Form.LOG_LOG else ""
    flat = [p for p, v in cols.items() if np.ptp(v) == 0]
    if flat:
        raise CerError(f"{', '.join(map(repr, flat))} {'has' if len(flat) == 1 else 'have'} "
                       f"the same value for every program used, so the CER can't learn "
                       f"{'its' if len(flat) == 1 else 'their'} effect. Drop "
                       f"{'it' if len(flat) == 1 else 'them'} or add programs where "
                       f"{'it differs' if len(flat) == 1 else 'they differ'}.")
    if len(predictors) < 2:
        return []
    names = list(cols)
    X = np.column_stack([cols[p] for p in names])
    design = np.column_stack([np.ones(len(X)), X])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                 if abs(np.corrcoef(cols[a], cols[b])[0, 1]) > 1 - 1e-9]
        which = ("; ".join(f"{a!r} and {b!r}" for a, b in pairs) if pairs
                 else ", ".join(map(repr, names)))
        raise CerError(f"The drivers {which} move in lockstep{words}: one is a fixed "
                       "multiple or combination of the others, so the CER can't separate "
                       "their effects. Keep one.")
    notes = []
    for j, p in enumerate(names):
        others = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        resid = X[:, j] - others @ beta
        ss = float(np.sum((X[:, j] - X[:, j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else 1.0
        vif = 1.0 / max(1.0 - r2, 1e-12)
        if vif > VIF_FLAG:
            notes.append(f"{p!r} is largely explained by the other drivers{words} (variance "
                         f"inflation factor {vif:,.0f}, above {VIF_FLAG:g}): its coefficient "
                         "and theirs trade off against each other, so read them together, "
                         "not one at a time.")
    return notes


def analyse(inputs: CerInput, form: Optional[str] = None, method: Optional[str] = None,
            level: Optional[float] = None, units: Optional[str] = None) -> CerStudy:
    """Fit the CER by OLS, MUPE and ZMPE and price the Estimate sheet.

    The keyword arguments override what the workbook says.
    """
    if form is not None:
        inputs.form = _form(form)
    if method is not None:
        inputs.method = _method(method)
    if level is not None:
        inputs.level = _level(level)
    used = inputs.used
    n, p = len(used), len(inputs.predictors) + 1
    if n <= p:
        raise CerError(f"{n} program{'s' if n != 1 else ''} can't fit {p} parameters (an "
                       f"intercept and {p - 1} driver{'s' if p > 2 else ''}): a CER needs "
                       "more programs than parameters, and three per parameter to be "
                       "worth defending. Add programs or drop a driver.")
    frame = used[[inputs.response, *inputs.predictors, "_name"]]
    notes = list(inputs.notes) + _check_drivers(used, inputs.predictors, inputs.form)
    fits = {}
    # The engine logs each fit at INFO; the report says it all once.
    logs = [logging.getLogger(n) for n in ("cost_core.cer.model", "cost_core.fitting")]
    was = [g.level for g in logs]
    for g in logs:
        g.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for m in METHODS:
                try:
                    fits[m] = fit_cer(frame, inputs.response, list(inputs.predictors),
                                      form=inputs.form, method=m, label_col="_name")
                except FitError as e:
                    raise CerError(f"The {m.upper()} fit failed: {e}") from None
    finally:
        for g, level in zip(logs, was):
            g.setLevel(level)
    said = set()
    for w in caught:
        text = str(w.message)
        # The engine's own small-sample warnings, not numpy's arithmetic ones.
        if text.startswith(("overflow", "invalid value", "divide by zero")):
            continue
        if issubclass(w.category, RuntimeWarning) and text not in said:
            said.add(text)
            notes.append(text)
    study = CerStudy(inputs=inputs, fits=fits, estimates=pd.DataFrame(), notes=notes,
                     units=inputs.units if units is None else units)
    study.estimates = _price(study)
    return study


def _price(study: CerStudy) -> pd.DataFrame:
    est = study.inputs.estimate
    cols = ["name", *study.inputs.predictors, "estimate", "lower", "upper", "level",
            "outside_data", "extrapolation_note"]
    if est.empty:
        return pd.DataFrame(columns=cols)
    pred = study.cer.predict(est[list(study.inputs.predictors)], kind="prediction",
                             level=study.inputs.level, warn_on_extrapolation=False)
    out = est.reset_index(drop=True).copy()
    out["estimate"] = pred["fit"].to_numpy()
    out["lower"] = pred["lower"].to_numpy()
    out["upper"] = pred["upper"].to_numpy()
    out["level"] = study.inputs.level
    out["outside_data"] = pred["outside_fitting_range"].map({True: "yes", False: ""}).to_numpy()
    out["extrapolation_note"] = pred["extrapolation_note"].to_numpy()
    # What cost-risk needs to carry the CER's own uncertainty: the spread on
    # the fitting scale and its degrees of freedom.
    out["se_fitting_scale"] = pred["se"].to_numpy()
    out["df"] = study.cer.df
    return out


# ------------------------------------------------------------- the template
INSTRUCTIONS = [
    ("What this is", "The data for a cost estimating relationship (CER) and the new "
     "programs to price with it: ce-core cer --data <this file>. Every number in the "
     "example is invented."),
    ("Data sheet", "One row per past program. Program is its name; Cost is what it cost "
     "(all in one dollar year and one unit); every other column with numbers in it is a "
     "driver. Use (yes or no) leaves a row out of the fit without deleting it, and Note "
     "says why."),
    ("Estimate sheet", "One row per new program to price, with the same driver headings "
     "as the Data sheet. Leave it empty to fit the CER only."),
    ("Settings sheet", "Cost and Drivers name the columns when the headings aren't the "
     "defaults (drivers separated by ;). Form is log-log (cost = a * driver^b, the usual "
     "one for cost) or linear. Method is MUPE (the default), OLS or ZMPE; all three are "
     "fitted and compared either way. Confidence is the prediction interval (0.8 for 80%). "
     "Units and Dollar Year are for the record."),
    ("Reading the result", "The prediction interval is where one new program is expected "
     "to land, which is what an estimate needs; it is wider than the confidence interval "
     "on the line itself, and it never shrinks below the scatter of the data. A new "
     "program outside the range of the data is flagged: the CER has no evidence there."),
]

EXAMPLE_DATA = pd.DataFrame(
    [("Radar A", 27.2, 500, 15.0), ("Radar B", 51.2, 600, 29.7),
     ("Radar C", 63.9, 890, 56.4), ("Radar D", 68.4, 1000, 40.7),
     ("Radar E", 85.9, 1060, 30.0), ("Radar F", 57.3, 1170, 30.7),
     ("Radar G", 97.1, 1440, 41.6), ("Radar H", 96.7, 1750, 43.5),
     ("Radar I", 108.0, 1770, 36.1), ("Radar J", 108.6, 2450, 70.7),
     ("Radar K", 234.1, 2890, 150.0), ("Radar L", 138.1, 2900, 55.9)],
    columns=["Program", "Cost", "Weight (lb)", "Power (kW)"])
EXAMPLE_DATA["Use"] = "yes"
EXAMPLE_DATA["Note"] = ""
EXAMPLE_ESTIMATE = pd.DataFrame(
    [("Next radar", 2000, 60.0), ("Large radar", 5200, 180.0)],
    columns=["Program", "Weight (lb)", "Power (kW)"])
EXAMPLE_SETTINGS = pd.DataFrame(
    [("Cost", "Cost"), ("Drivers", "Weight (lb); Power (kW)"), ("Form", "log-log"),
     ("Method", "MUPE"), ("Confidence", 0.8), ("Units", "millions"),
     ("Dollar Year", "BY2026")], columns=["Setting", "Value"])


def write_workbook(out, data: pd.DataFrame = EXAMPLE_DATA,
                   estimate: pd.DataFrame = EXAMPLE_ESTIMATE,
                   settings: pd.DataFrame = EXAMPLE_SETTINGS) -> Path:
    """Write a workbook laid out the way :func:`read_workbook` reads it."""
    from openpyxl.styles import Alignment, Font

    out = Path(out)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        data.to_excel(xl, sheet_name="Data", index=False)
        estimate.to_excel(xl, sheet_name="Estimate", index=False)
        settings.to_excel(xl, sheet_name="Settings", index=False)
        pd.DataFrame(INSTRUCTIONS, columns=["Sheet or topic", "What to put there"]) \
            .to_excel(xl, sheet_name="Instructions", index=False)
        widths = {"Data": (18, 10, 14, 14, 8, 30), "Estimate": (18, 14, 14),
                  "Settings": (16, 26), "Instructions": (22, 100)}
        for name, cols in widths.items():
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for j, w in enumerate(cols):
                ws.column_dimensions[chr(ord("A") + j)].width = w
        for row in xl.sheets["Instructions"].iter_rows(min_row=2):
            row[0].alignment = Alignment(vertical="top")
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
    return out
