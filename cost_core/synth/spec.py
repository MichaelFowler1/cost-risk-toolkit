"""
spec.py - Configuration for the synthetic CSDR/SRDR generator.

Everything here is invented. There is no real or proprietary contractor data in
this repository, and none is required to run any part of it. The WBS skeleton
follows the *shape* of MIL-STD-881 aircraft-system reporting because that is
what makes the ingest problem realistic; the numbers attached to it are made
up and were chosen to look plausible on a chart, nothing more.

The generator is built around one rule: **every pathology it introduces must
be either reversible or detectable.** A name that drifts across reporting
periods has a crosswalk entry that maps it back. A then-year dollar has an
index that deflates it. A resubmitted period has a report date that orders it.
A missing period is genuinely gone and the pipeline is expected to say so
rather than interpolate. That rule is what lets ``tests/test_ingest.py`` assert
that a clean pipeline recovers the generating truth *exactly*, instead of
asserting that today's output matches yesterday's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# WBS skeleton
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WBSElement:
    """One reporting element.

    Attributes:
        code: MIL-STD-881-style number, e.g. ``"1.1"``.
        name: Canonical name. The generator will emit drifted variants of this
            across periods; the crosswalk maps them back to this string.
        parent: Parent code, or None at the top.
        cost_share: Share of the total program *recurring* pool. Only
            meaningful for leaf elements; parents roll up from children.
        nonrecurring_share: Share of the total *nonrecurring* pool.

    An element's own recurring/nonrecurring split is therefore derived from
    these two shares rather than stated separately -- carrying a third field
    would let the three disagree, and a spec that can contradict itself is
    exactly what this generator exists to help find.
    """

    code: str
    name: str
    parent: str | None
    cost_share: float
    nonrecurring_share: float

    @property
    def is_leaf(self) -> bool:
        return self.cost_share > 0.0 or self.nonrecurring_share > 0.0


#: A synthetic aircraft-system WBS. Shares sum to 1.0 across leaves, which
#: ``tests/test_synth.py`` asserts so a careless edit here cannot silently
#: change program totals.
DEFAULT_WBS: tuple[WBSElement, ...] = (
    WBSElement("1.0", "Air Vehicle", None, 0.0, 0.0),
    WBSElement("1.1", "Airframe", "1.0", 0.340, 0.180),
    WBSElement("1.2", "Propulsion", "1.0", 0.210, 0.060),
    WBSElement("1.3", "Vehicle Subsystems", "1.0", 0.115, 0.070),
    WBSElement("1.4", "Avionics", "1.0", 0.190, 0.150),
    WBSElement("1.5", "Air Vehicle Software", "1.0", 0.035, 0.190),
    WBSElement("2.0", "Systems Engineering", None, 0.045, 0.130),
    WBSElement("3.0", "Program Management", None, 0.030, 0.070),
    WBSElement("4.0", "System Test and Evaluation", None, 0.020, 0.110),
    WBSElement("5.0", "Training", None, 0.010, 0.025),
    WBSElement("6.0", "Data", None, 0.005, 0.015),
)

#: Alternative names a contractor might use for the same element in different
#: submissions. This is the single most common reason an analyst's row counts
#: do not reconcile across periods, and the crosswalk is the artifact that
#: fixes it. Every alias here resolves to the canonical name in DEFAULT_WBS.
DEFAULT_NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    "Airframe": ("Air Frame", "AIRFRAME", "Airframe Structure", "Airframe (Struct)"),
    "Propulsion": ("Propulsion System", "PROPULSION", "Engine / Propulsion"),
    "Vehicle Subsystems": (
        "Veh Subsystems",
        "Vehicle Sub-Systems",
        "VEHICLE SUBSYSTEMS",
    ),
    "Avionics": ("Avionics Suite", "AVIONICS", "Avionics Systems"),
    "Air Vehicle Software": ("AV Software", "Software", "Air Vehicle S/W"),
    "Systems Engineering": ("Sys Engineering", "SE", "Systems Engr"),
    "Program Management": ("Prog Management", "PM", "Program Mgmt"),
    "System Test and Evaluation": ("Sys Test & Eval", "ST&E", "System Test and Eval"),
    "Training": ("Trng", "TRAINING"),
    "Data": ("Data Deliverables", "DATA"),
}


# --------------------------------------------------------------------------
# functional categories and wrap rates
# --------------------------------------------------------------------------
#: The DD 1921-1 functional breakdown. ``has_hours`` is the detail that makes
#: wrap-rate reconciliation non-trivial: material is a dollars-only category,
#: so an analyst who divides total dollars by total hours to get a blended rate
#: gets a wrong answer, and a pipeline that assumes every row has hours will
#: silently drop material or produce an infinite rate.
FUNCTIONAL_CATEGORIES: tuple[tuple[str, bool], ...] = (
    ("engineering", True),
    ("manufacturing", True),
    ("tooling", True),
    ("quality", True),
    ("material", False),
)

#: Share of a WBS element's cost going to each functional category, by whether
#: the cost is recurring. Nonrecurring work is engineering-heavy; recurring
#: production is manufacturing- and material-heavy. Each column sums to 1.0.
FUNCTIONAL_MIX: dict[str, dict[str, float]] = {
    "recurring": {
        "engineering": 0.10,
        "manufacturing": 0.38,
        "tooling": 0.06,
        "quality": 0.09,
        "material": 0.37,
    },
    "nonrecurring": {
        "engineering": 0.55,
        "manufacturing": 0.12,
        "tooling": 0.18,
        "quality": 0.07,
        "material": 0.08,
    },
}

#: Base-year composite wrap rates in dollars per hour: direct labour plus
#: overhead, fringe and G&A. Escalated year over year by the labour index.
BASE_WRAP_RATES: dict[str, float] = {
    "engineering": 148.50,
    "manufacturing": 96.25,
    "tooling": 112.00,
    "quality": 89.75,
}


# --------------------------------------------------------------------------
# inflation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InflationAssumption:
    """A raw-index escalation assumption.

    Attributes:
        base_year: Fiscal year in which the index equals 1.000.
        annual_rate: Compound annual escalation applied to build the index.
        first_year: First year present in the generated table.
        last_year: Last year present in the generated table.
    """

    base_year: int = 2020
    annual_rate: float = 0.0235
    first_year: int = 2016
    last_year: int = 2036

    def index(self) -> dict[int, float]:
        """Weighted index by fiscal year, normalised to 1.000 at ``base_year``.

        A raw index rather than a set of factors, because the base year has to
        be re-selectable downstream without regenerating anything: a raw index
        divided by its own value in any chosen year gives the factors for that
        year.
        """
        return {
            year: (1.0 + self.annual_rate) ** (year - self.base_year)
            for year in range(self.first_year, self.last_year + 1)
        }


# --------------------------------------------------------------------------
# pathologies
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PathologyConfig:
    """Which kinds of mess to introduce, and how much.

    Set every probability to zero for a clean program -- useful in tests that
    want to isolate one behaviour, and the basis for the "clean pipeline
    recovers truth exactly" assertions.

    Attributes:
        name_drift_prob: Chance that a given element in a given period is
            reported under an alias rather than its canonical name.
        then_year_prob: Chance that a period reports in then-year dollars
            rather than the program base year. Real submissions mix these and
            label them inconsistently.
        resubmission_prob: Chance a period is submitted twice, the second
            submission superseding the first with corrected numbers.
        missing_period_prob: Chance a period is simply absent.
        outlier_lot_prob: Chance a lot carries a rate break or design-change
            cost shock.
        outlier_magnitude: Multiplicative size of that shock, drawn around
            this value.
        quantity_change: Whether the buy profile is rebaselined mid-program.
        noise_cv: Multiplicative lognormal scatter on each reported cost.
        eac_optimism: How much the early at-completion estimates understate
            the eventual total, decaying to zero as the program completes.
    """

    name_drift_prob: float = 0.30
    then_year_prob: float = 0.45
    resubmission_prob: float = 0.15
    missing_period_prob: float = 0.08
    outlier_lot_prob: float = 0.15
    outlier_magnitude: float = 1.28
    quantity_change: bool = True
    noise_cv: float = 0.06
    eac_optimism: float = 0.12

    @classmethod
    def clean(cls) -> "PathologyConfig":
        """No mess at all: reported values equal the generating truth."""
        return cls(
            name_drift_prob=0.0,
            then_year_prob=0.0,
            resubmission_prob=0.0,
            missing_period_prob=0.0,
            outlier_lot_prob=0.0,
            quantity_change=False,
            noise_cv=0.0,
            eac_optimism=0.0,
        )


# --------------------------------------------------------------------------
# software / SRDR
# --------------------------------------------------------------------------
#: Weights turning raw counts into equivalent source lines. Reused code is
#: nearly free, modified code costs about a third of new, auto-generated code
#: costs a tenth. Invented, but in the range the adaptation-adjustment
#: literature uses.
ADAPTATION_WEIGHTS: dict[str, float] = {
    "new": 1.00,
    "modified": 0.35,
    "reused": 0.03,
    "autogen": 0.10,
}

#: Share of total software effort by SRDR activity. Sums to 1.0.
SRDR_ACTIVITY_MIX: dict[str, float] = {
    "requirements_analysis": 0.10,
    "architecture_design": 0.20,
    "coding": 0.26,
    "unit_test": 0.14,
    "integration_test": 0.22,
    "management_quality": 0.08,
}

#: The *true* software CER the portfolio is generated from:
#: ``effort_hours = a * (equivalent_KSLOC ** b)``. Exposed so a CER fit across
#: a generated portfolio can be checked against the answer it should find.
TRUE_SOFTWARE_CER: tuple[float, float] = (2000.0, 1.05)

#: The *true* airframe CER the portfolio is generated from:
#: ``t1_cost = a * ((empty_weight_lb / 1000) ** b)``. The theoretical first
#: unit of each program is priced off this, so fitting a learning curve to get
#: T1 and then a CER across programs to get weight sensitivity should recover
#: these two numbers.
TRUE_AIRFRAME_CER: tuple[float, float] = (465_000.0, 0.72)


@dataclass(frozen=True)
class SoftwareSpec:
    """Software size and process characteristics for the SRDR.

    Attributes:
        sloc_new: Newly written source lines.
        sloc_modified: Pre-existing lines modified.
        sloc_reused: Lines carried across unchanged.
        sloc_autogen: Auto-generated lines.
        primary_language: Reported implementation language.
        application_domain: Reported domain.
        development_process: Reported lifecycle model.
        cmmi_level: Reported process maturity, 1-5.
        peak_staff: Peak full-time-equivalent staff.
        team_experience: Reported experience band.
        requirements_volatility: Fraction of requirements churned.
        productivity_factor: Program-specific multiplier on the true CER --
            this is the scatter a CER fit has to see through.
    """

    sloc_new: int = 420_000
    sloc_modified: int = 160_000
    sloc_reused: int = 310_000
    sloc_autogen: int = 90_000
    primary_language: str = "C++"
    application_domain: str = "Vehicle Control"
    development_process: str = "Incremental"
    cmmi_level: int = 3
    peak_staff: int = 145
    team_experience: str = "Nominal"
    requirements_volatility: float = 0.18
    productivity_factor: float = 1.0

    @property
    def equivalent_sloc(self) -> float:
        """Adaptation-adjusted size, the CER's actual driver."""
        return (
            ADAPTATION_WEIGHTS["new"] * self.sloc_new
            + ADAPTATION_WEIGHTS["modified"] * self.sloc_modified
            + ADAPTATION_WEIGHTS["reused"] * self.sloc_reused
            + ADAPTATION_WEIGHTS["autogen"] * self.sloc_autogen
        )

    @property
    def equivalent_ksloc(self) -> float:
        return self.equivalent_sloc / 1000.0

    def true_effort_hours(self) -> float:
        """Effort implied by the true CER, before reporting scatter."""
        a, b = TRUE_SOFTWARE_CER
        return a * (self.equivalent_ksloc**b) * self.productivity_factor


# --------------------------------------------------------------------------
# program specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProgramSpec:
    """The generating truth for one synthetic program.

    Attributes:
        program: Program name used on every emitted row.
        contractor: Reporting contractor name (invented).
        learning_slope: True Crawford unit-theory slope, e.g. 0.85.
        t1_cost: True theoretical first-unit recurring cost, base-year dollars.
        lot_quantities: Units delivered in each production lot.
        first_fiscal_year: Fiscal year of the first lot.
        nonrecurring_total: Total nonrecurring budget, base-year dollars.
        nonrecurring_decay: Per-lot decay of the nonrecurring spend profile;
            0.55 means each lot books 55% of what the previous lot did, which
            is what concentrates nonrecurring cost in the early lots.
        empty_weight_lb: Technical driver for the airframe CER. The program's
            ``t1_cost`` is priced off this via :data:`TRUE_AIRFRAME_CER` when
            a portfolio is generated.
        max_speed_kts: A second technical parameter, included so a CER can be
            fitted with more than one predictor and the degrees-of-freedom
            guardrails have something to bite on.
        wbs: The reporting hierarchy.
        software: Software size and process characteristics for the SRDR.
        inflation: Escalation assumption.
        pathologies: What mess to introduce.
    """

    program: str = "PEGASUS-X"
    contractor: str = "Meridian Aerostructures"
    learning_slope: float = 0.85
    t1_cost: float = 4_250_000.0
    lot_quantities: tuple[int, ...] = (4, 6, 10, 12, 12, 16, 18, 20)
    first_fiscal_year: int = 2020
    nonrecurring_total: float = 310_000_000.0
    nonrecurring_decay: float = 0.55
    empty_weight_lb: float = 22_000.0
    max_speed_kts: float = 540.0
    wbs: tuple[WBSElement, ...] = DEFAULT_WBS
    software: SoftwareSpec = field(default_factory=SoftwareSpec)
    inflation: InflationAssumption = field(default_factory=InflationAssumption)
    pathologies: PathologyConfig = field(default_factory=PathologyConfig)

    @property
    def n_lots(self) -> int:
        return len(self.lot_quantities)

    @property
    def leaves(self) -> tuple[WBSElement, ...]:
        return tuple(w for w in self.wbs if w.is_leaf)

    def fiscal_year(self, lot_index: int) -> int:
        """Fiscal year for a zero-based lot index."""
        return self.first_fiscal_year + lot_index

    def validate(self) -> None:
        """Fail loudly on an inconsistent specification.

        Raises:
            ValueError: If shares do not sum to one, the slope is outside
                (0, 1], or quantities are not positive integers.
        """
        if not 0.0 < self.learning_slope <= 1.0:
            raise ValueError(
                f"learning_slope must be in (0, 1]; got {self.learning_slope}. "
                f"A slope above 1 would mean cost rising with every unit built."
            )
        if self.t1_cost <= 0:
            raise ValueError(f"t1_cost must be positive; got {self.t1_cost}.")
        if not self.lot_quantities:
            raise ValueError("A program needs at least one production lot.")
        if any(q <= 0 for q in self.lot_quantities):
            raise ValueError(
                f"Lot quantities must all be positive; got {self.lot_quantities}."
            )

        rec = sum(w.cost_share for w in self.wbs)
        nonrec = sum(w.nonrecurring_share for w in self.wbs)
        if abs(rec - 1.0) > 1e-9:
            raise ValueError(f"WBS recurring cost shares sum to {rec}, not 1.0.")
        if abs(nonrec - 1.0) > 1e-9:
            raise ValueError(
                f"WBS nonrecurring shares sum to {nonrec}, not 1.0."
            )

        for label, mix in FUNCTIONAL_MIX.items():
            total = sum(mix.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"Functional mix for {label} sums to {total}, not 1.0."
                )

        codes = {w.code for w in self.wbs}
        for w in self.wbs:
            if w.parent is not None and w.parent not in codes:
                raise ValueError(
                    f"WBS element {w.code} names parent {w.parent}, "
                    f"which is not in the hierarchy."
                )
