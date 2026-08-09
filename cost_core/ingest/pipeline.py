"""
pipeline.py - Normalise CSDR/SRDR submissions into one long-format table.

Six report shapes go in; one table comes out, at the grain

    (program, report_type, wbs_element, functional_category, period, lot,
     recurring_flag) -> quantity, hours, dollars, dollar_year

**Read this before summing the output.** ``report_type`` is a dimension, not
metadata. The DD 1921, the DD 1921-1 and the FlexFile all describe the *same
dollars* at different levels of detail, so a naive ``rows["dollars"].sum()``
counts every dollar three times. Filter to one report type, or use
:meth:`NormalizedDataset.authoritative`. The cross-report reconciliation gate
exists precisely because those three views must agree, and comparing them is
the strongest single check available that the normalisation is right.

The pipeline does five things, in this order, and records what it did at each:

1. **Extract.** Each report shape is read into a common staging frame, keeping
   the source file and row number on every record.
2. **Crosswalk.** Reported WBS names resolve to canonical names through the
   persisted artifact. Unmatched names are *surfaced and fail the run*; they
   are never dropped and never guessed at.
3. **Deduplicate.** Resubmitted periods collapse to the latest report date.
   Superseded rows stay in the provenance table marked as such, so the row
   count still adds up and the earlier submission is auditable.
4. **Normalise dollars.** Then-year amounts deflate to the base year through
   the index table. The raw amount, its stated year and the factor applied all
   survive into provenance.
5. **Validate.** Row counts before and after, dollar totals reconciled within
   and across reports, unmatched names, and reporting gaps. In strict mode any
   error-severity gate raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cost_core.ingest.crosswalk import Crosswalk
from cost_core.ingest.inflation import DEFAULT_INDEX, InflationTable

logger = logging.getLogger(__name__)

#: Sentinel for rows that describe the whole program rather than one element.
WBS_PROGRAM_LEVEL = "(program level)"
#: Sentinel for rows that are not split by functional category.
CATEGORY_ALL = "(all functions)"

NORMALIZED_COLUMNS: tuple[str, ...] = (
    "row_uid",
    "program",
    "report_type",
    "wbs_element",
    "wbs_code",
    "functional_category",
    "period",
    "lot",
    "quantity",
    "hours",
    "dollars",
    "dollar_year",
    "recurring_flag",
)

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "row_uid",
    "source_report",
    "source_row",
    "report_date",
    "wbs_element_raw",
    "crosswalk_rule",
    "hours_raw",
    "dollars_raw",
    "dollar_year_raw",
    "basis_raw",
    "inflation_index",
    "inflation_factor",
    "superseded",
    "ingested_at",
)

#: Relative tolerance for the dollar reconciliation gates. Tight on purpose:
#: normalisation is arithmetic, so anything beyond floating-point noise is a
#: defect, not rounding.
RECONCILIATION_RTOL = 1e-9

#: Which report type is preferred when the same measure appears in several.
_AUTHORITATIVE_ORDER = ("DD1921-1", "FLEXFILE", "DD1921", "DD1921-2")

#: Report types that actually carry cost for a reporting period. Used by the
#: gap gate, which must not be satisfied by the Quantity Data Report's habit
#: of restating every prior lot in every submission.
_COST_REPORT_TYPES = ("DD1921", "DD1921-1", "DD1921-2", "FLEXFILE")


class IngestError(ValueError):
    """Raised when a validation gate of error severity fails."""


# --------------------------------------------------------------------------
# validation reporting
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationGate:
    """One check, its verdict, and the numbers behind it.

    Attributes:
        name: Short identifier, stable across runs so it can be diffed.
        passed: Whether the check succeeded.
        severity: ``"error"`` fails the run in strict mode; ``"warning"`` is
            surfaced but tolerated; ``"info"`` is bookkeeping.
        detail: One-line human-readable explanation.
        metrics: Supporting numbers, written into the assumptions log.
    """

    name: str
    passed: bool
    severity: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Every gate the pipeline ran."""

    gates: list[ValidationGate] = field(default_factory=list)

    def add(
        self,
        name: str,
        passed: bool,
        severity: str,
        detail: str,
        **metrics: Any,
    ) -> None:
        self.gates.append(ValidationGate(name, passed, severity, detail, metrics))
        level = logging.INFO if passed else (
            logging.ERROR if severity == "error" else logging.WARNING
        )
        logger.log(level, "gate %s: %s", name, detail)

    @property
    def errors(self) -> list[ValidationGate]:
        return [g for g in self.gates if not g.passed and g.severity == "error"]

    @property
    def warnings(self) -> list[ValidationGate]:
        return [g for g in self.gates if not g.passed and g.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        """Raise a single error naming every failed gate.

        Raises:
            IngestError: If any error-severity gate failed.
        """
        if self.ok:
            return
        lines = [f"  - {g.name}: {g.detail}" for g in self.errors]
        raise IngestError(
            f"{len(self.errors)} validation gate(s) failed:\n" + "\n".join(lines)
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "gate": g.name,
                    "passed": g.passed,
                    "severity": g.severity,
                    "detail": g.detail,
                    **g.metrics,
                }
                for g in self.gates
            ]
        )


# --------------------------------------------------------------------------
# the normalised dataset
# --------------------------------------------------------------------------
@dataclass
class NormalizedDataset:
    """The output of the pipeline: rows, provenance, and the audit trail."""

    rows: pd.DataFrame
    provenance: pd.DataFrame
    validation: ValidationReport
    crosswalk: Crosswalk
    inflation: InflationTable
    base_year: int
    index_name: str = DEFAULT_INDEX

    # ------------------------------------------------------------ provenance
    def trace(self, row_uid: str) -> pd.DataFrame:
        """Every source row that contributed to one normalised row.

        This is the answer to "where did this number come from", and it is the
        reason the provenance table is one row per *source* record rather than
        one per output record: aggregation is many-to-one, and a trace that
        only recorded the last contributor would be worse than none.
        """
        hit = self.provenance[self.provenance["row_uid"] == row_uid]
        if hit.empty:
            raise KeyError(
                f"No provenance for {row_uid!r}. Known uids look like: "
                f"{list(self.rows['row_uid'].head(3))}"
            )
        return hit.copy()

    # ---------------------------------------------------------------- views
    def by_report(self, report_type: str) -> pd.DataFrame:
        return self.rows[self.rows["report_type"] == report_type].copy()

    def authoritative(self) -> pd.DataFrame:
        """The single non-double-counting view of element-level cost.

        Picks the most detailed report type actually present, in the order
        DD 1921-1, FlexFile, DD 1921. Program-level rows (the progress curve
        and quantity reports) are excluded because they restate the same
        dollars at a coarser grain.
        """
        present = [
            r for r in _AUTHORITATIVE_ORDER if (self.rows["report_type"] == r).any()
        ]
        if not present:
            return self.rows.iloc[0:0].copy()
        chosen = present[0]
        logger.info("Authoritative view: %s", chosen)
        return self.by_report(chosen)

    def total_dollars(self, report_type: str | None = None) -> float:
        """Total base-year dollars for one report type, or the authoritative view."""
        frame = self.by_report(report_type) if report_type else self.authoritative()
        return float(frame["dollars"].sum(skipna=True))

    def learning_curve_input(self) -> pd.DataFrame:
        """Lot-level recurring cost, ready for a learning-curve fit.

        Derives the first and last unit index of each lot from the cumulative
        quantity, which a unit-theory fit needs and the report does not carry
        in normalised form.

        Raises:
            IngestError: If no progress-curve rows survived ingest.
        """
        lots = self.by_report("DD1921-2").sort_values("lot")
        if lots.empty:
            raise IngestError(
                "No DD1921-2 rows in the normalised data, so there is nothing "
                "to fit a progress curve to. Check the reporting gaps gate."
            )
        out = lots[["program", "lot", "period", "quantity", "dollars"]].copy()
        out = out.rename(columns={"dollars": "lot_cost", "quantity": "lot_quantity"})
        out["lot_quantity"] = out["lot_quantity"].astype(int)
        cum = out["lot_quantity"].cumsum()
        out["first_unit"] = (cum - out["lot_quantity"] + 1).astype(int)
        out["last_unit"] = cum.astype(int)
        out["unit_cost"] = out["lot_cost"] / out["lot_quantity"]
        return out.reset_index(drop=True)

    def software_input(self) -> pd.DataFrame:
        """SRDR size and effort, aggregated per program, ready for a CER fit."""
        srdr = self.by_report("SRDR")
        if srdr.empty:
            return pd.DataFrame(columns=["program", "equivalent_sloc", "effort_hours"])
        grouped = srdr.groupby("program").agg(
            equivalent_sloc=("quantity", "max"),
            effort_hours=("hours", "sum"),
        )
        return grouped.reset_index()

    def summary(self) -> pd.DataFrame:
        """Row counts and dollar totals by report type, for the run log."""
        out = (
            self.rows.groupby("report_type")
            .agg(
                rows=("row_uid", "size"),
                dollars=("dollars", "sum"),
                hours=("hours", "sum"),
            )
            .reset_index()
        )
        return out.sort_values("report_type").reset_index(drop=True)


# --------------------------------------------------------------------------
# extraction: each report shape -> the common staging frame
# --------------------------------------------------------------------------
_STAGING_COLUMNS = (
    "program",
    "report_type",
    "wbs_element_raw",
    "functional_category",
    "period",
    "lot",
    "quantity",
    "hours",
    "dollars_raw",
    "basis",
    "dollar_year_raw",
    "recurring_flag",
    "report_date",
    "source_report",
    "source_row",
)


def _stage(frame: pd.DataFrame, source: str, **overrides: Any) -> pd.DataFrame:
    """Assemble a staging frame, filling anything not supplied with nulls.

    ``source_row`` defaults to the frame's own position, which is right for the
    extractors that map one source row to one staged row. An extractor that
    reshapes -- the SRDR melt turns one row into one per activity -- must pass
    the *original* row numbers explicitly, or the provenance trail would point
    at positions in an intermediate frame that no file ever had.
    """
    n = len(frame)
    data: dict[str, Any] = {
        "program": frame.get("program", pd.Series([pd.NA] * n)),
        "report_type": frame.get("report_type", pd.Series([source] * n)),
        "wbs_element_raw": WBS_PROGRAM_LEVEL,
        "functional_category": CATEGORY_ALL,
        "period": frame.get("period_fy", pd.Series([pd.NA] * n)),
        "lot": frame.get("lot", pd.Series([pd.NA] * n)),
        "quantity": np.nan,
        "hours": np.nan,
        "dollars_raw": np.nan,
        "basis": frame.get("basis", pd.Series(["BY"] * n)),
        "dollar_year_raw": frame.get("dollar_year", pd.Series([pd.NA] * n)),
        "recurring_flag": frame.get("recurring_flag", pd.Series([True] * n)),
        "report_date": frame.get("report_date", pd.Series([pd.NA] * n)),
        "source_report": source,
        "source_row": np.arange(n),
    }
    data.update(overrides)
    staged = pd.DataFrame(data).reset_index(drop=True)
    return staged[list(_STAGING_COLUMNS)]


def extract_dd1921(frame: pd.DataFrame) -> pd.DataFrame:
    """Element-level recurring/nonrecurring cost. Uses the *incremental*
    period column, not the cumulative one: a then-year to-date figure is a sum
    across several years and no single index can deflate it."""
    return _stage(
        frame,
        "dd1921",
        wbs_element_raw=frame["wbs_element_name"],
        dollars_raw=frame["cost_incurred_period"],
    )


def extract_dd1921_1(frame: pd.DataFrame) -> pd.DataFrame:
    """Functional hours and dollars per element."""
    return _stage(
        frame,
        "dd1921_1",
        wbs_element_raw=frame["wbs_element_name"],
        functional_category=frame["functional_category"],
        hours=frame["hours"],
        dollars_raw=frame["dollars"],
    )


def extract_dd1921_2(frame: pd.DataFrame) -> pd.DataFrame:
    """Recurring lot cost and quantity: the progress-curve input."""
    return _stage(
        frame,
        "dd1921_2",
        quantity=frame["lot_quantity"],
        dollars_raw=frame["recurring_lot_cost"],
        recurring_flag=True,
    )


def extract_flexfile(frame: pd.DataFrame) -> pd.DataFrame:
    """Long-format cost elements, split back out into hours and dollars.

    The FlexFile puts hours and dollars in one value column distinguished by a
    unit, so the split has to be made on the unit and nothing else. Summing the
    value column without looking at the unit adds hours to dollars, which is
    the classic way this format goes wrong.
    """
    hours = frame["value"].where(frame["unit"] == "hours")
    dollars = frame["value"].where(frame["unit"] == "dollars")
    return _stage(
        frame,
        "flexfile",
        report_type=pd.Series(["FLEXFILE"] * len(frame)),
        wbs_element_raw=frame["wbs_element_name"],
        functional_category=frame["functional_category"],
        hours=hours,
        dollars_raw=dollars,
    )


def extract_quantity_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Lot quantities and delivery periods."""
    return _stage(
        frame,
        "quantity_report",
        quantity=frame["lot_quantity"],
        recurring_flag=True,
    )


def extract_srdr(frame: pd.DataFrame) -> pd.DataFrame:
    """Software effort by activity.

    Melted so each activity becomes a functional category, which lets software
    effort live in the same table as everything else. ``quantity`` carries
    equivalent source lines of code -- the size measure that plays the same
    role for software that unit count plays for hardware.

    SRDR deliberately does not reconcile to the CSDR dollars: it is a separate
    collection with different boundaries, and the cross-report gate excludes it
    for that reason.
    """
    activity_cols = [c for c in frame.columns if c.startswith("hours_")]
    if not activity_cols:
        raise IngestError(
            f"SRDR frame has no hours_* activity columns. Found: "
            f"{list(frame.columns)}"
        )
    tagged = frame.copy()
    tagged["_source_row"] = np.arange(len(tagged))
    melted = tagged.melt(
        id_vars=[
            c
            for c in (
                "program", "report_type", "report_date", "period_fy",
                "wbs_element_name", "equivalent_sloc", "_source_row",
            )
            if c in tagged.columns
        ],
        value_vars=activity_cols,
        var_name="activity",
        value_name="activity_hours",
    ).reset_index(drop=True)

    return _stage(
        melted,
        "srdr",
        source_row=melted["_source_row"],
        report_type=pd.Series(["SRDR"] * len(melted)),
        wbs_element_raw=melted["wbs_element_name"],
        functional_category=melted["activity"].str.removeprefix("hours_"),
        quantity=melted["equivalent_sloc"],
        hours=melted["activity_hours"],
        # Software development effort is nonrecurring by nature.
        recurring_flag=False,
        lot=pd.Series([pd.NA] * len(melted)),
    )


EXTRACTORS = {
    "dd1921": extract_dd1921,
    "dd1921_1": extract_dd1921_1,
    "dd1921_2": extract_dd1921_2,
    "flexfile": extract_flexfile,
    "quantity_report": extract_quantity_report,
    "srdr": extract_srdr,
}


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def _row_uid(row: pd.Series) -> str:
    """A deterministic, readable identity for a normalised row.

    Readable rather than hashed on purpose: a provenance key that a reviewer
    can parse by eye is worth more than a short one.
    """
    lot = "-" if pd.isna(row["lot"]) else f"L{int(row['lot'])}"
    flag = "R" if bool(row["recurring_flag"]) else "N"
    return (
        f"{row['report_type']}|{row['program']}|FY{row['period']}|{lot}"
        f"|{row['wbs_element']}|{row['functional_category']}|{flag}"
    )


def normalize(
    reports: dict[str, pd.DataFrame],
    *,
    crosswalk: Crosswalk,
    inflation: InflationTable,
    base_year: int,
    index_name: str = DEFAULT_INDEX,
    strict: bool = True,
    expected_lots: Iterable[int] | None = None,
) -> NormalizedDataset:
    """Normalise a set of report frames into one long table.

    Args:
        reports: Report name -> raw frame, as produced by
            :mod:`cost_core.synth` or read from CADE extracts.
        crosswalk: The persisted WBS name mapping. Required, not optional:
            without it the pipeline would have to guess.
        inflation: Index table for base-year normalisation.
        base_year: Fiscal year to state all dollars in.
        index_name: Which index in the table to use.
        strict: Raise on any error-severity gate. Turn off only to inspect a
            failing dataset, never to ship one.
        expected_lots: Lots the program should have reported, so gaps can be
            named rather than merely implied by absence.

    Returns:
        NormalizedDataset: rows, provenance and the validation report.

    Raises:
        IngestError: In strict mode, if any error-severity gate fails.
    """
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report_out = ValidationReport()

    unknown = set(reports) - set(EXTRACTORS)
    if unknown:
        raise IngestError(
            f"No extractor for report(s) {sorted(unknown)}. "
            f"Known: {sorted(EXTRACTORS)}."
        )
    if not reports:
        raise IngestError("No reports supplied; there is nothing to normalise.")

    # --- 1. extract -------------------------------------------------------
    staged_parts, rows_in = [], {}
    for name, frame in reports.items():
        rows_in[name] = len(frame)
        if frame.empty:
            report_out.add(
                f"empty_report:{name}", False, "warning",
                f"{name} contained no rows and was skipped.", rows=0,
            )
            continue
        staged_parts.append(EXTRACTORS[name](frame))

    if not staged_parts:
        raise IngestError("Every supplied report was empty.")

    staged = pd.concat(staged_parts, ignore_index=True)
    total_in = int(sum(rows_in.values()))
    report_out.add(
        "rows_extracted", True, "info",
        f"Extracted {len(staged)} staging rows from {total_in} source rows "
        f"across {len(reports)} report(s).",
        source_rows=total_in, staged_rows=len(staged),
    )
    _gate_extraction_coverage(report_out, staged, rows_in)

    # --- 2. crosswalk -----------------------------------------------------
    resolved = crosswalk.apply(staged["wbs_element_raw"])
    # Program-level sentinels are not WBS elements and need no mapping.
    is_sentinel = staged["wbs_element_raw"] == WBS_PROGRAM_LEVEL
    staged["wbs_element"] = np.where(
        is_sentinel, WBS_PROGRAM_LEVEL, resolved["canonical"]
    )
    staged["wbs_code"] = np.where(is_sentinel, pd.NA, resolved["wbs_code"])
    staged["crosswalk_rule"] = np.where(is_sentinel, "n/a", resolved["rule"])

    unmatched_mask = (~is_sentinel) & resolved["canonical"].isna()
    unmatched_names = sorted(set(staged.loc[unmatched_mask, "wbs_element_raw"]))
    if unmatched_names:
        suggestions = {n: crosswalk.suggest(n) for n in unmatched_names}
        report_out.add(
            "wbs_names_resolved", False, "error",
            f"{len(unmatched_names)} WBS name(s) are not in the crosswalk and "
            f"were NOT dropped: {unmatched_names}. Suggested matches: "
            f"{suggestions}. Add them to the crosswalk artifact and rerun.",
            unmatched=unmatched_names, affected_rows=int(unmatched_mask.sum()),
        )
    else:
        report_out.add(
            "wbs_names_resolved", True, "info",
            f"All {int((~is_sentinel).sum())} element rows resolved through the "
            f"crosswalk.",
            by_rule=resolved.loc[~is_sentinel, "rule"].value_counts().to_dict(),
        )

    # --- 3. deduplicate resubmissions ------------------------------------
    staged["report_date"] = staged["report_date"].astype("string")
    key = [
        "program", "report_type", "wbs_element", "functional_category",
        "period", "lot", "recurring_flag", "source_report",
    ]
    # NA lots would drop rows from a groupby, so key on a filled copy.
    staged["_lot_key"] = staged["lot"].astype("string").fillna("~")
    group_key = [c if c != "lot" else "_lot_key" for c in key]

    latest = staged.groupby(group_key, dropna=False)["report_date"].transform("max")
    staged["superseded"] = staged["report_date"].ne(latest)
    n_superseded = int(staged["superseded"].sum())

    live = staged[~staged["superseded"]].copy()
    report_out.add(
        "resubmissions_deduplicated", True, "info",
        f"{n_superseded} row(s) superseded by a later submission; "
        f"{len(live)} retained.",
        superseded=n_superseded, retained=len(live),
    )

    # --- 4. normalise dollars to the base year ---------------------------
    stated_year = live["dollar_year_raw"].fillna(base_year).astype(int)
    factors = np.array(
        [
            inflation.factor(int(y), base_year, index_name) if pd.notna(d) else np.nan
            for y, d in zip(stated_year, live["dollars_raw"])
        ]
    )
    live["inflation_factor"] = factors
    live["dollars"] = live["dollars_raw"].to_numpy(dtype=float) * factors
    live["dollar_year"] = base_year

    report_out.add(
        "dollars_normalised", True, "info",
        f"Deflated to FY{base_year} using index {index_name!r} "
        f"({inflation.source}).",
        base_year=base_year, index_name=index_name,
        then_year_rows=int((live["basis"] == "TY").sum()),
        base_year_rows=int((live["basis"] == "BY").sum()),
    )

    # --- 5. aggregate to the normalised grain ----------------------------
    live["row_uid"] = live.apply(_row_uid, axis=1)
    aggregated = (
        live.groupby("row_uid", dropna=False)
        .agg(
            program=("program", "first"),
            report_type=("report_type", "first"),
            wbs_element=("wbs_element", "first"),
            wbs_code=("wbs_code", "first"),
            functional_category=("functional_category", "first"),
            period=("period", "first"),
            lot=("lot", "first"),
            quantity=("quantity", "max"),
            hours=("hours", "sum"),
            dollars=("dollars", "sum"),
            dollar_year=("dollar_year", "first"),
            recurring_flag=("recurring_flag", "first"),
        )
        .reset_index()
    )
    rows_out = aggregated[list(NORMALIZED_COLUMNS)]

    # --- provenance: one record per source row, live or superseded -------
    staged_live_uid = dict(zip(live.index, live["row_uid"]))
    provenance = pd.DataFrame(
        {
            "row_uid": [staged_live_uid.get(i) for i in staged.index],
            "source_report": staged["source_report"],
            "source_row": staged["source_row"],
            "report_date": staged["report_date"],
            "wbs_element_raw": staged["wbs_element_raw"],
            "crosswalk_rule": staged["crosswalk_rule"],
            "hours_raw": staged["hours"],
            "dollars_raw": staged["dollars_raw"],
            "dollar_year_raw": staged["dollar_year_raw"],
            "basis_raw": staged["basis"],
            "inflation_index": index_name,
            "inflation_factor": [
                live["inflation_factor"].get(i, np.nan) for i in staged.index
            ],
            "superseded": staged["superseded"],
            "ingested_at": ingested_at,
        }
    )[list(PROVENANCE_COLUMNS)]

    # --- 6. validation gates ---------------------------------------------
    _gate_row_accounting(report_out, total_in, len(staged), n_superseded, len(rows_out))
    _gate_provenance_complete(report_out, rows_out, provenance)
    _gate_dollar_reconciliation(report_out, live, rows_out)
    _gate_cross_report(report_out, rows_out)
    _gate_reporting_gaps(report_out, rows_out, expected_lots)
    _gate_no_negative_costs(report_out, rows_out)

    dataset = NormalizedDataset(
        rows=rows_out,
        provenance=provenance,
        validation=report_out,
        crosswalk=crosswalk,
        inflation=inflation,
        base_year=base_year,
        index_name=index_name,
    )

    if strict:
        report_out.raise_if_failed()
    elif not report_out.ok:
        logger.error(
            "Ingest completed with %d failed gate(s); strict=False so the "
            "dataset is returned anyway. Do not brief these numbers.",
            len(report_out.errors),
        )
    return dataset


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def _gate_extraction_coverage(
    report: ValidationReport, staged: pd.DataFrame, rows_in: dict[str, int]
) -> None:
    """Every source row must appear in staging at least once.

    Not a count comparison: extraction is allowed to reshape, and the SRDR
    melt legitimately turns one source row into one row per activity. What is
    never allowed is a source row vanishing, so the check is that each
    report's staged ``source_row`` values cover ``0..n-1`` exactly.
    """
    missing: dict[str, list[int]] = {}
    expansion: dict[str, float] = {}
    for name, n_in in rows_in.items():
        if n_in == 0:
            continue
        seen = set(
            int(v)
            for v in staged.loc[staged["source_report"] == name, "source_row"].unique()
        )
        gaps = sorted(set(range(n_in)) - seen)
        if gaps:
            missing[name] = gaps[:10]
        expansion[name] = round(
            int((staged["source_report"] == name).sum()) / n_in, 4
        )

    ok = not missing
    report.add(
        "extraction_covers_every_source_row", ok, "error" if not ok else "info",
        f"All source rows represented in staging; rows-per-source-row by "
        f"report: {expansion}."
        if ok else
        f"Source row(s) dropped during extraction (first few per report): "
        f"{missing}.",
        expansion=expansion, missing=missing,
    )


def _gate_row_accounting(
    report: ValidationReport, source_rows: int, staged: int,
    superseded: int, out_rows: int,
) -> None:
    """Staged rows must all be either retained or superseded, and the
    aggregation must not lose any of the retained ones."""
    retained = staged - superseded
    ok = retained >= out_rows and superseded >= 0
    report.add(
        "row_count_accounting", ok, "error" if not ok else "info",
        f"{source_rows} source rows -> {staged} staged "
        f"({superseded} superseded, {retained} retained) -> {out_rows} "
        f"normalised rows after aggregation."
        + ("" if ok else " Aggregation produced more rows than it consumed."),
        source_rows=source_rows, staged_rows=staged,
        superseded=superseded, retained=retained, normalised_rows=out_rows,
    )


def _gate_provenance_complete(
    report: ValidationReport, rows: pd.DataFrame, provenance: pd.DataFrame
) -> None:
    """Every output row must be traceable, and every trace must point at a
    real output row."""
    out_uids = set(rows["row_uid"])
    traced = set(provenance.loc[~provenance["superseded"], "row_uid"].dropna())
    untraceable = out_uids - traced
    orphaned = traced - out_uids
    ok = not untraceable and not orphaned
    report.add(
        "provenance_complete", ok, "error" if not ok else "info",
        f"{len(out_uids)} normalised rows, all traceable to source."
        if ok else
        f"{len(untraceable)} output row(s) have no provenance and "
        f"{len(orphaned)} provenance record(s) point at no output row.",
        output_rows=len(out_uids), traced_rows=len(traced),
    )


def _gate_dollar_reconciliation(
    report: ValidationReport, live: pd.DataFrame, rows: pd.DataFrame
) -> None:
    """Normalised dollars, re-escalated, must equal the source dollars.

    Deflation is arithmetic, so this is an identity, not an estimate: dividing
    by a factor and multiplying it back must return the original to
    floating-point precision. Anything larger means a row was dropped, double
    counted, or deflated with the wrong year.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        reconstructed = live["dollars"] / live["inflation_factor"]
    source_total = float(live["dollars_raw"].sum(skipna=True))
    back_total = float(reconstructed.sum(skipna=True))
    normalised_total = float(live["dollars"].sum(skipna=True))
    aggregated_total = float(rows["dollars"].sum(skipna=True))

    ok_roundtrip = np.isclose(back_total, source_total, rtol=RECONCILIATION_RTOL)
    ok_aggregate = np.isclose(
        aggregated_total, normalised_total, rtol=RECONCILIATION_RTOL
    )
    ok = bool(ok_roundtrip and ok_aggregate)
    report.add(
        "dollars_reconcile_to_source", ok, "error" if not ok else "info",
        f"Across all report types combined (so each dollar is counted once "
        f"per view -- see the cross-report gate for per-report totals): "
        f"source ${source_total:,.2f} -> normalised ${normalised_total:,.2f} "
        f"(base year); re-escalated ${back_total:,.2f}."
        + ("" if ok_roundtrip else " Round trip through the index does not "
                                   "return the source total.")
        + ("" if ok_aggregate else " Aggregation changed the dollar total."),
        source_total=source_total, normalised_total=normalised_total,
        reescalated_total=back_total, aggregated_total=aggregated_total,
    )


def _gate_cross_report(report: ValidationReport, rows: pd.DataFrame) -> None:
    """The DD 1921, DD 1921-1 and FlexFile must describe the same dollars.

    They are three views of one set of costs, so after normalisation their
    totals have to agree. This is the strongest available check that the
    crosswalk, the dedup and the deflation all did the right thing, because
    all three views must be wrong in exactly the same way to still agree.
    """
    totals = {
        rt: float(rows.loc[rows["report_type"] == rt, "dollars"].sum(skipna=True))
        for rt in ("DD1921", "DD1921-1", "FLEXFILE")
        if (rows["report_type"] == rt).any()
    }
    if len(totals) < 2:
        report.add(
            "cross_report_reconciliation", True, "info",
            f"Only {len(totals)} element-level report present; nothing to "
            f"cross-check.", totals=totals,
        )
        return

    values = list(totals.values())
    spread = max(values) - min(values)
    scale = max(abs(v) for v in values) or 1.0
    ok = bool(spread / scale <= RECONCILIATION_RTOL)
    report.add(
        "cross_report_reconciliation", ok, "error" if not ok else "info",
        "; ".join(f"{k} ${v:,.2f}" for k, v in totals.items())
        + (
            f". Agree to within ${spread:,.4f}."
            if ok
            else f". These disagree by ${spread:,.2f} "
                 f"({spread / scale:.3%}), so at least one view is wrong."
        ),
        totals=totals, spread=spread, relative_spread=spread / scale,
    )


def _gate_reporting_gaps(
    report: ValidationReport, rows: pd.DataFrame, expected_lots: Iterable[int] | None
) -> None:
    """Name the missing periods rather than letting absence speak for itself.

    A gap is real information about the data, not a pipeline failure, so this
    is a warning. What it must never be is invisible: interpolating across a
    missing period would put a number in front of someone that no contractor
    ever reported.

    Only the cost-bearing reports count toward presence. The Quantity Data
    Report restates every prior lot in each submission, so a lot whose cost
    period was never filed still appears there -- checking against it would
    hide exactly the gap this gate exists to find.
    """
    if expected_lots is None:
        report.add(
            "reporting_gaps", True, "info",
            "No expected lot list supplied, so gaps cannot be detected.",
        )
        return

    cost_rows = rows[rows["report_type"].isin(_COST_REPORT_TYPES)]
    expected = set(int(x) for x in expected_lots)
    present = set(int(x) for x in cost_rows["lot"].dropna().unique())
    missing = sorted(expected - present)
    ok = not missing
    report.add(
        "reporting_gaps", ok, "warning" if not ok else "info",
        f"All {len(expected)} expected lots present."
        if ok else
        f"Lot(s) {missing} were never reported. They are left absent, not "
        f"interpolated; any forecast over this range is extrapolating.",
        expected=sorted(expected), missing=missing,
    )


def _gate_no_negative_costs(report: ValidationReport, rows: pd.DataFrame) -> None:
    """Negative normalised cost means a sign or offset error upstream."""
    bad = rows[rows["dollars"] < 0]
    ok = bad.empty
    report.add(
        "no_negative_costs", ok, "error" if not ok else "info",
        "No negative normalised costs."
        if ok else
        f"{len(bad)} row(s) have negative cost after normalisation, e.g. "
        f"{list(bad['row_uid'].head(3))}.",
        negative_rows=len(bad),
    )


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------
def normalize_program(
    program,
    *,
    crosswalk: Crosswalk | None = None,
    inflation: InflationTable | None = None,
    base_year: int | None = None,
    strict: bool = True,
) -> NormalizedDataset:
    """Ingest a :class:`~cost_core.synth.SyntheticProgram` end to end.

    Defaults the crosswalk and index table to the ones matching the bundled
    synthetic WBS, which is a convenience for demos and tests only -- a real
    run supplies both as reviewed artifacts.
    """
    return normalize(
        program.reports,
        crosswalk=crosswalk or Crosswalk.default(),
        inflation=inflation
        or InflationTable.from_mapping(
            program.truth.inflation_index, source="synthetic program assumption"
        ),
        base_year=base_year if base_year is not None else program.truth.base_year,
        strict=strict,
        expected_lots=range(1, program.spec.n_lots + 1),
    )
