"""
generator.py - Seeded generation of a synthetic contractor cost program.

The whole generator is built on a single fine-grained truth table, at the grain

    (wbs_code, lot, recurring_flag, functional_category)
        -> hours, base-year dollars

Every report shape in :mod:`cost_core.synth.reports` is a *view* of that one
table. That is deliberate: it means the six reports reconcile to each other by
construction, so when the ingest pipeline finds them disagreeing, the
disagreement was introduced by a pathology that the pipeline is supposed to
resolve, and not by the generator being sloppy.

The generating truth is exposed on the returned object. Tests use it the way
the rest of this library uses closed-form answers: a clean pipeline over a
clean program must reproduce the truth *exactly*, not approximately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cost_core.synth.spec import (BASE_WRAP_RATES, DEFAULT_NAME_VARIANTS,
                                  FUNCTIONAL_CATEGORIES, FUNCTIONAL_MIX,
                                  PathologyConfig, ProgramSpec)

logger = logging.getLogger(__name__)

#: Share of a burdened hour that is booked as direct labour rather than
#: overhead. Only used to split a wrap rate into FlexFile cost elements.
DIRECT_LABOR_FRACTION = 0.38


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProgramTruth:
    """What the generator actually built, before any reporting mess.

    Attributes:
        cells: The fine-grained truth table. Columns: ``wbs_code``,
            ``wbs_name`` (canonical), ``lot``, ``fiscal_year``,
            ``recurring_flag``, ``functional_category``, ``hours``,
            ``dollars_by``.
        lot_quantities: Units actually delivered per lot, after any
            mid-program rebaseline.
        planned_quantities: The original buy profile before rebaseline.
        learning_slope: True Crawford unit-theory slope used to price units.
        t1_cost: True theoretical first-unit recurring cost, base-year.
        lot_recurring_by: Noiseless recurring cost of each lot from the curve,
            before outliers and scatter -- the answer a learning-curve fit on
            clean data has to recover.
        outlier_lots: Lots carrying a rate break or design change, mapped to
            the multiplicative shock applied.
        inflation_index: Raw weighted index by fiscal year.
        wrap_rates: Base-year composite rate per hour-bearing category.
        crosswalk: Reported alias -> canonical WBS name, for every alias that
            was actually emitted.
        base_year: Fiscal year the base-year dollars are stated in.
    """

    cells: pd.DataFrame
    lot_quantities: tuple[int, ...]
    planned_quantities: tuple[int, ...]
    learning_slope: float
    t1_cost: float
    lot_recurring_by: pd.Series
    outlier_lots: dict[int, float]
    inflation_index: dict[int, float]
    wrap_rates: dict[str, float]
    crosswalk: dict[str, str]
    base_year: int

    @property
    def total_dollars_by(self) -> float:
        """Total program cost in base-year dollars. The reconciliation target."""
        return float(self.cells["dollars_by"].sum())

    @property
    def total_hours(self) -> float:
        return float(self.cells["hours"].sum())

    def totals_by_wbs(self) -> pd.Series:
        return self.cells.groupby("wbs_code")["dollars_by"].sum().sort_index()

    def totals_by_lot(self) -> pd.Series:
        return self.cells.groupby("lot")["dollars_by"].sum().sort_index()

    def recurring_lot_costs(self) -> pd.DataFrame:
        """Actual recurring cost and quantity per lot -- the 1921-2 truth."""
        rec = self.cells[self.cells["recurring_flag"]]
        out = rec.groupby("lot")["dollars_by"].sum().reset_index()
        out = out.rename(columns={"dollars_by": "lot_cost_by"})
        out["quantity"] = [self.lot_quantities[int(i) - 1] for i in out["lot"]]
        out["unit_cost_by"] = out["lot_cost_by"] / out["quantity"]
        return out


@dataclass
class SyntheticProgram:
    """A generated program: its truth, and the reports an analyst would pull."""

    spec: ProgramSpec
    seed: int
    truth: ProgramTruth
    reports: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, report: str) -> pd.DataFrame:
        if report not in self.reports:
            raise KeyError(
                f"No report {report!r}. Available: {sorted(self.reports)}."
            )
        return self.reports[report]

    def write_csvs(self, directory: str | Path) -> dict[str, Path]:
        """Write every report to CSV, the way an analyst receives them.

        Returns:
            Mapping of report name to the path written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, frame in self.reports.items():
            path = directory / f"{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = path
        logger.info("Wrote %d synthetic reports to %s", len(written), directory)
        return written


# --------------------------------------------------------------------------
# the learning curve used to price units
# --------------------------------------------------------------------------
def crawford_lot_cost(t1: float, slope: float, first: int, last: int) -> float:
    """Exact Crawford (unit theory) cost of units ``first``..``last``.

    Summed unit by unit rather than approximated with a lot midpoint, so the
    generated data is consistent with an exact theory and a fit on clean data
    can recover the parameters to machine precision.
    """
    units = np.arange(first, last + 1, dtype=float)
    return float(t1 * np.sum(units ** np.log2(slope)))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _effective_quantities(
    spec: ProgramSpec, rng: np.random.Generator
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Apply a mid-program buy rebaseline, if configured.

    Quantity changes are one of the main reasons a learning-curve fit made in
    year three stops matching reality in year six, so the generator does it
    the way programs actually do: a decision partway through cuts or adds to
    every remaining lot.
    """
    planned = tuple(spec.lot_quantities)
    if not spec.pathologies.quantity_change or spec.n_lots < 4:
        return planned, planned

    change_at = int(rng.integers(2, spec.n_lots - 1))
    factor = float(rng.choice([0.70, 0.75, 1.25, 1.40]))
    effective = tuple(
        q if i < change_at else max(1, int(round(q * factor)))
        for i, q in enumerate(planned)
    )
    logger.info(
        "Buy rebaselined at lot %d by factor %.2f: %s -> %s",
        change_at + 1,
        factor,
        planned,
        effective,
    )
    return effective, planned


def _build_truth(spec: ProgramSpec, seed: int) -> ProgramTruth:
    """Construct the fine-grained truth table for one program."""
    rng = np.random.default_rng(seed)
    spec.validate()

    quantities, planned = _effective_quantities(spec, rng)
    index = spec.inflation.index()
    base_year = spec.inflation.base_year

    # --- recurring: price every unit off the Crawford curve, lot by lot
    lot_first_unit, cum = [], 0
    for q in quantities:
        lot_first_unit.append(cum + 1)
        cum += q
    lot_recurring = np.array(
        [
            crawford_lot_cost(spec.t1_cost, spec.learning_slope, f, f + q - 1)
            for f, q in zip(lot_first_unit, quantities)
        ]
    )

    # --- rate breaks and design changes on a few lots
    outliers: dict[int, float] = {}
    p = spec.pathologies
    for i in range(spec.n_lots):
        if rng.random() < p.outlier_lot_prob:
            shock = float(rng.normal(p.outlier_magnitude, 0.06))
            shock = max(shock, 1.02)
            outliers[i + 1] = shock

    # --- nonrecurring: geometric decay front-loads it into the early lots
    weights = spec.nonrecurring_decay ** np.arange(spec.n_lots)
    lot_nonrecurring = spec.nonrecurring_total * weights / weights.sum()

    # --- explode to (wbs, lot, recurring_flag, functional_category)
    rows = []
    for lot_idx in range(spec.n_lots):
        lot = lot_idx + 1
        fy = spec.fiscal_year(lot_idx)
        rec_pool = lot_recurring[lot_idx] * outliers.get(lot, 1.0)
        nonrec_pool = lot_nonrecurring[lot_idx]

        for leaf in spec.leaves:
            for is_recurring, pool, share in (
                (True, rec_pool, leaf.cost_share),
                (False, nonrec_pool, leaf.nonrecurring_share),
            ):
                element_cost = pool * share
                if element_cost <= 0.0:
                    continue
                mix = FUNCTIONAL_MIX["recurring" if is_recurring else "nonrecurring"]
                for category, has_hours in FUNCTIONAL_CATEGORIES:
                    dollars = element_cost * mix[category]
                    if p.noise_cv > 0.0:
                        dollars *= float(rng.lognormal(0.0, p.noise_cv))
                    hours = (
                        dollars / BASE_WRAP_RATES[category] if has_hours else 0.0
                    )
                    rows.append(
                        {
                            "wbs_code": leaf.code,
                            "wbs_name": leaf.name,
                            "lot": lot,
                            "fiscal_year": fy,
                            "recurring_flag": is_recurring,
                            "functional_category": category,
                            "hours": hours,
                            "dollars_by": dollars,
                        }
                    )

    cells = pd.DataFrame(rows)

    # --- which aliases will actually be used, so the crosswalk is complete
    crosswalk: dict[str, str] = {}
    for leaf in spec.leaves:
        crosswalk[leaf.name] = leaf.name
        for alias in DEFAULT_NAME_VARIANTS.get(leaf.name, ()):
            crosswalk[alias] = leaf.name

    return ProgramTruth(
        cells=cells,
        lot_quantities=quantities,
        planned_quantities=planned,
        learning_slope=spec.learning_slope,
        t1_cost=spec.t1_cost,
        lot_recurring_by=pd.Series(
            lot_recurring, index=pd.RangeIndex(1, spec.n_lots + 1, name="lot")
        ),
        outlier_lots=outliers,
        inflation_index=index,
        wrap_rates=dict(BASE_WRAP_RATES),
        crosswalk=crosswalk,
        base_year=base_year,
    )


def generate_program(
    seed: int,
    spec: ProgramSpec | None = None,
    *,
    pathologies: PathologyConfig | None = None,
    with_reports: bool = True,
) -> SyntheticProgram:
    """Generate one synthetic program and all six report shapes.

    Args:
        seed: Any integer. The same seed with the same spec reproduces the
            program exactly, down to the last cent and every alias chosen.
        spec: Program definition. Defaults to the bundled PEGASUS-X spec.
        pathologies: Overrides ``spec.pathologies``; convenient for asking for
            a clean program without rebuilding the whole spec.
        with_reports: Build the six report shapes. Set False when only the
            truth is wanted -- CER work across a large portfolio needs one row
            per program, not thousands of submission rows each, and building
            the reports is the bulk of the cost. The truth is identical either
            way; the reports are derived from it, not the other way round.

    Returns:
        SyntheticProgram: truth plus reports.

    Raises:
        ValueError: If the spec is internally inconsistent.
    """
    from cost_core.synth import reports as _reports  # circular at module level

    spec = spec or ProgramSpec()
    if pathologies is not None:
        spec = ProgramSpec(
            **{
                **{
                    f: getattr(spec, f)
                    for f in spec.__dataclass_fields__
                    if f != "pathologies"
                },
                "pathologies": pathologies,
            }
        )

    truth = _build_truth(spec, seed)
    logger.info(
        "Generated %s (seed=%d): %d lots, %d units, $%.1fM base-year",
        spec.program,
        seed,
        spec.n_lots,
        sum(truth.lot_quantities),
        truth.total_dollars_by / 1e6,
    )

    program = SyntheticProgram(spec=spec, seed=seed, truth=truth)
    if with_reports:
        program.reports = _reports.build_all(program, np.random.default_rng(seed + 1))
    return program


# --------------------------------------------------------------------------
# portfolios: what a CER is actually fitted against
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioTruth:
    """The generating CERs behind a set of programs.

    A CER is fitted *across* programs, not within one, so a portfolio is the
    unit a CER test needs. Both relationships below are exact power laws with
    lognormal scatter, which means a correctly implemented CER fit has a known
    right answer to land on -- the same standard the learning-curve tests hold
    themselves to.

    Attributes:
        airframe_a, airframe_b: ``t1_cost = a * (empty_weight_lb/1000) ** b``.
        software_a, software_b: ``effort_hours = a * equivalent_KSLOC ** b``.
        scatter_cv: Lognormal scatter applied around each relationship.
        drivers: One row per program with its technical parameters and the
            noiseless values the CERs imply.
    """

    airframe_a: float
    airframe_b: float
    software_a: float
    software_b: float
    scatter_cv: float
    drivers: pd.DataFrame


@dataclass
class Portfolio:
    """Several synthetic programs plus the CER truth behind them."""

    programs: list[SyntheticProgram]
    truth: PortfolioTruth

    def __len__(self) -> int:
        return len(self.programs)

    def __iter__(self):
        return iter(self.programs)

    def report(self, name: str) -> pd.DataFrame:
        """Concatenate one report shape across every program.

        Raises:
            KeyError: If the portfolio was built with ``with_reports=False``.
        """
        if not any(p.reports for p in self.programs):
            raise KeyError(
                "This portfolio was generated without reports "
                "(with_reports=False), so only cer_table() is available."
            )
        return pd.concat(
            [p.reports[name] for p in self.programs], ignore_index=True
        )

    def cer_table(self) -> pd.DataFrame:
        """One row per program: technical drivers and observed outcomes.

        This is the table a CER is fitted on. ``t1_cost_observed`` is the
        value that *was* generated (scatter included); a real analyst would
        instead recover it by fitting a learning curve to each program's
        1921-2 data, and ``tests/test_cer.py`` checks that route arrives at
        the same place.
        """
        return self.truth.drivers.copy()


def generate_portfolio(
    n_programs: int = 12,
    seed: int = 0,
    *,
    pathologies: PathologyConfig | None = None,
    scatter_cv: float = 0.18,
    base_spec: ProgramSpec | None = None,
    with_reports: bool = True,
) -> Portfolio:
    """Generate a portfolio of related programs for CER work.

    Each program gets its own empty weight, speed, software size and learning
    slope; its theoretical first-unit cost and software effort are then priced
    off the shared true CERs with lognormal scatter.

    Args:
        n_programs: How many programs to generate. Kept small by default
            because real CER datasets are small, which is the whole reason the
            degrees-of-freedom guardrails matter.
        seed: Master seed. Program ``i`` is generated from ``seed * 1000 + i``,
            so the portfolio is reproducible and individual programs can be
            regenerated in isolation.
        pathologies: Applied to every program.
        scatter_cv: Lognormal scatter around each true CER.
        base_spec: Template to vary. Defaults to the bundled spec.
        with_reports: Build each program's six report shapes. Set False when
            only :meth:`Portfolio.cer_table` is needed -- a CER is fitted on
            one row per program, and skipping report generation makes a large
            portfolio roughly an order of magnitude cheaper to build.

    Raises:
        ValueError: If fewer than two programs are requested, since a single
            point cannot define a relationship.
    """
    from cost_core.synth.spec import (SoftwareSpec, TRUE_AIRFRAME_CER,
                                      TRUE_SOFTWARE_CER)

    if n_programs < 2:
        raise ValueError(
            f"A CER needs at least two programs to be estimable; got {n_programs}."
        )

    base = base_spec or ProgramSpec()
    air_a, air_b = TRUE_AIRFRAME_CER
    sw_a, sw_b = TRUE_SOFTWARE_CER
    rng = np.random.default_rng(seed)

    programs: list[SyntheticProgram] = []
    driver_rows = []

    for i in range(n_programs):
        weight = float(rng.uniform(8_000.0, 46_000.0))
        speed = float(rng.uniform(380.0, 720.0))
        slope = float(rng.uniform(0.78, 0.94))
        n_lots = int(rng.integers(5, 10))
        first_qty = int(rng.integers(3, 8))
        quantities = tuple(
            max(1, int(round(first_qty * (1.0 + 0.22 * k) )))
            for k in range(n_lots)
        )

        # Airframe CER with scatter -> this program's true first-unit cost.
        t1_noiseless = air_a * (weight / 1000.0) ** air_b
        t1 = t1_noiseless * float(rng.lognormal(0.0, scatter_cv))

        sw_scale = float(rng.uniform(0.35, 2.10))
        software = SoftwareSpec(
            sloc_new=int(base.software.sloc_new * sw_scale),
            sloc_modified=int(base.software.sloc_modified * sw_scale),
            sloc_reused=int(base.software.sloc_reused * sw_scale),
            sloc_autogen=int(base.software.sloc_autogen * sw_scale),
            cmmi_level=int(rng.integers(2, 6)),
            peak_staff=int(base.software.peak_staff * sw_scale),
            requirements_volatility=float(rng.uniform(0.05, 0.35)),
            productivity_factor=float(rng.lognormal(0.0, scatter_cv)),
        )

        spec = ProgramSpec(
            program=f"{base.program}-{i + 1:02d}",
            contractor=base.contractor,
            learning_slope=slope,
            t1_cost=t1,
            lot_quantities=quantities,
            first_fiscal_year=base.first_fiscal_year,
            nonrecurring_total=base.nonrecurring_total * (weight / 22_000.0) ** 0.6,
            nonrecurring_decay=base.nonrecurring_decay,
            empty_weight_lb=weight,
            max_speed_kts=speed,
            wbs=base.wbs,
            software=software,
            inflation=base.inflation,
            pathologies=pathologies if pathologies is not None else base.pathologies,
        )

        program = generate_program(
            seed * 1000 + i, spec, with_reports=with_reports
        )
        programs.append(program)

        driver_rows.append(
            {
                "program": spec.program,
                "empty_weight_lb": weight,
                "max_speed_kts": speed,
                "equivalent_ksloc": software.equivalent_ksloc,
                "cmmi_level": software.cmmi_level,
                "requirements_volatility": software.requirements_volatility,
                "learning_slope_true": slope,
                "t1_cost_noiseless": t1_noiseless,
                "t1_cost_observed": t1,
                "software_effort_noiseless": sw_a * software.equivalent_ksloc**sw_b,
                "software_effort_observed": software.true_effort_hours(),
                "total_units": sum(quantities),
            }
        )

    logger.info("Generated a portfolio of %d programs (seed=%d)", n_programs, seed)
    return Portfolio(
        programs=programs,
        truth=PortfolioTruth(
            airframe_a=air_a,
            airframe_b=air_b,
            software_a=sw_a,
            software_b=sw_b,
            scatter_cv=scatter_cv,
            drivers=pd.DataFrame(driver_rows),
        ),
    )


# --------------------------------------------------------------------------
# reporting-period bookkeeping, shared by the report builders
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Submission:
    """One (period, report date) submission of a report.

    Attributes:
        lot: The production lot the period covers.
        fiscal_year: Fiscal year of the period.
        report_date: Date the submission was made. Later wins on dedup.
        is_resubmission: True if this supersedes an earlier submission.
        error_factor: Multiplier applied to the *superseded* first submission,
            so the resubmission carries the correct numbers and dedup by
            latest report date recovers the truth.
    """

    lot: int
    fiscal_year: int
    report_date: date
    is_resubmission: bool = False
    error_factor: float = 1.0


def plan_submissions(
    program: SyntheticProgram, rng: np.random.Generator
) -> list[Submission]:
    """Decide which periods are reported, missed, and resubmitted.

    Missing periods are genuinely gone -- nothing downstream can reconstruct
    them, and the pipeline is expected to surface the gap rather than
    interpolate over it. Resubmissions are recoverable: the later submission
    carries the correct figures.
    """
    p = program.spec.pathologies
    submissions: list[Submission] = []

    for lot_idx in range(program.spec.n_lots):
        lot = lot_idx + 1
        fy = program.spec.fiscal_year(lot_idx)
        # Submissions land about 90 days after the close of the fiscal year.
        due = date(fy + 1, 1, 1) - timedelta(days=1) + timedelta(days=90)

        if rng.random() < p.missing_period_prob:
            logger.debug("Lot %d (FY%d) period is missing from the data", lot, fy)
            continue

        if rng.random() < p.resubmission_prob:
            # First, wrong submission, then a corrected one six months later.
            wrong_by = float(rng.normal(1.0, 0.09))
            wrong_by = wrong_by if abs(wrong_by - 1.0) > 0.02 else 1.06
            submissions.append(
                Submission(lot, fy, due, False, error_factor=wrong_by)
            )
            submissions.append(
                Submission(lot, fy, due + timedelta(days=182), True, 1.0)
            )
        else:
            submissions.append(Submission(lot, fy, due))

    return submissions


def reported_name(
    canonical: str, rng: np.random.Generator, drift_prob: float
) -> str:
    """Return the canonical name, or one of its aliases."""
    variants = DEFAULT_NAME_VARIANTS.get(canonical, ())
    if variants and rng.random() < drift_prob:
        return str(rng.choice(np.array(variants, dtype=object)))
    return canonical


def dollar_basis(
    rng: np.random.Generator, then_year_prob: float
) -> str:
    """Choose whether a submission reports in base-year or then-year dollars."""
    return "TY" if rng.random() < then_year_prob else "BY"
