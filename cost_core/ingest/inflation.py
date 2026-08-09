"""
inflation.py - Base-year normalisation from a configurable index table.

Mixing then-year and base-year dollars in one regression is the quickest way to
fit a learning curve that is really an inflation curve: costs that rise 2.4% a
year while unit cost falls 15% per doubling produce a slope that is wrong in a
direction nobody notices, because the answer still looks like a learning curve.

Two rules here, both aimed at the "well-documented" and "accurate"
characteristics of the GAO guide:

**Raw indices, not factors.** The table stores an index normalised to 1.000 in
some reference year. Factors between any two years are derived on demand as a
ratio. That means the base year can be re-selected downstream -- from BY2020 to
BY2026 for a new briefing -- without regenerating anything, and the arithmetic
is visible rather than baked in.

**Raw values are always preserved.** Normalisation never overwrites; the
original amount, its stated year, and the index applied all survive into the
provenance table, so any normalised number can be walked back to what the
contractor actually reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INDEX_COLUMNS = ("index_name", "fiscal_year", "index_value")

#: Index used when a caller does not name one.
DEFAULT_INDEX = "composite"


class InflationError(ValueError):
    """Raised when an index is missing, malformed, or asked for a year it
    does not cover."""


@dataclass
class InflationTable:
    """Named raw indices by fiscal year.

    Attributes:
        indices: Index name -> {fiscal year -> raw index value}.
        source: Free-text provenance for the table itself, carried into the
            assumptions log so a reader knows where the escalation came from.
    """

    indices: dict[str, dict[int, float]] = field(default_factory=dict)
    source: str = "unspecified"

    # ------------------------------------------------------------- building
    @classmethod
    def from_rate(
        cls,
        annual_rate: float = 0.0235,
        base_year: int = 2020,
        first_year: int = 2010,
        last_year: int = 2045,
        name: str = DEFAULT_INDEX,
        source: str = "constant-rate assumption",
    ) -> "InflationTable":
        """Build a single index from a compound annual rate.

        A placeholder for a real published index. Named as an assumption in
        the run log precisely so nobody mistakes it for one.
        """
        if last_year < first_year:
            raise InflationError(
                f"last_year {last_year} precedes first_year {first_year}."
            )
        values = {
            year: (1.0 + annual_rate) ** (year - base_year)
            for year in range(first_year, last_year + 1)
        }
        return cls(indices={name: values}, source=f"{source} ({annual_rate:.2%}/yr)")

    @classmethod
    def from_mapping(
        cls, values: dict[int, float], name: str = DEFAULT_INDEX, source: str = "supplied"
    ) -> "InflationTable":
        return cls(indices={name: dict(values)}, source=source)

    # ---------------------------------------------------------- persistence
    @classmethod
    def load(cls, path: str | Path, source: str | None = None) -> "InflationTable":
        """Read an index table from CSV with columns index_name, fiscal_year,
        index_value.

        Raises:
            FileNotFoundError: If the artifact is missing.
            InflationError: On missing columns, non-positive index values, or
                duplicate year entries within one index.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No inflation index at {path}. Write one with "
                f"InflationTable.from_rate(...).save(path), and replace it "
                f"with a published index before the numbers leave the room."
            )

        frame = pd.read_csv(path)
        missing = [c for c in INDEX_COLUMNS if c not in frame.columns]
        if missing:
            raise InflationError(
                f"Index table at {path} is missing column(s): {missing}."
            )

        indices: dict[str, dict[int, float]] = {}
        for name, group in frame.groupby("index_name"):
            years = group["fiscal_year"].astype(int)
            if years.duplicated().any():
                dupes = sorted(years[years.duplicated()].unique())
                raise InflationError(
                    f"Index {name!r} in {path} has duplicate year(s): {dupes}."
                )
            values = group["index_value"].astype(float)
            if (values <= 0).any():
                raise InflationError(
                    f"Index {name!r} in {path} has non-positive values, which "
                    f"cannot be a price index."
                )
            indices[str(name)] = dict(zip(years, values))

        logger.info("Loaded %d inflation index/indices from %s", len(indices), path)
        return cls(indices=indices, source=source or f"loaded from {path}")

    def save(self, path: str | Path) -> Path:
        """Write the index table to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"index_name": name, "fiscal_year": year, "index_value": value}
            for name, values in sorted(self.indices.items())
            for year, value in sorted(values.items())
        ]
        pd.DataFrame(rows, columns=list(INDEX_COLUMNS)).to_csv(path, index=False)
        logger.info("Wrote inflation index artifact to %s", path)
        return path

    # ------------------------------------------------------------- querying
    def _index(self, name: str) -> dict[int, float]:
        if name not in self.indices:
            raise InflationError(
                f"No index named {name!r}. Available: {sorted(self.indices)}."
            )
        return self.indices[name]

    def value(self, year: int, name: str = DEFAULT_INDEX) -> float:
        """Raw index value for one year.

        Raises:
            InflationError: If the year is outside the table. Extrapolating an
                index past its range is a judgement call, and making it
                silently is not defensible -- extend the table on purpose.
        """
        values = self._index(name)
        year = int(year)
        if year not in values:
            lo, hi = min(values), max(values)
            raise InflationError(
                f"Index {name!r} covers FY{lo}-FY{hi} and was asked for "
                f"FY{year}. Extend the index table rather than extrapolating "
                f"silently."
            )
        return float(values[year])

    def factor(self, from_year: int, to_year: int, name: str = DEFAULT_INDEX) -> float:
        """Multiplier converting ``from_year`` dollars into ``to_year`` dollars."""
        return self.value(to_year, name) / self.value(from_year, name)

    def to_base_year(
        self,
        amounts,
        from_years,
        base_year: int,
        name: str = DEFAULT_INDEX,
    ) -> np.ndarray:
        """Convert amounts stated in various years into one base year.

        Args:
            amounts: Values to convert.
            from_years: The year each amount is stated in, same length.
            base_year: Target year.
            name: Which index to use.

        Raises:
            InflationError: On a length mismatch or an out-of-range year.
        """
        amounts = np.asarray(amounts, dtype=float)
        years = np.asarray(from_years)
        if amounts.shape != years.shape:
            raise InflationError(
                f"Got {amounts.size} amounts and {years.size} years; these "
                f"must correspond one to one."
            )
        base_value = self.value(base_year, name)
        factors = np.array([base_value / self.value(int(y), name) for y in years])
        return amounts * factors

    @property
    def coverage(self) -> dict[str, tuple[int, int]]:
        """First and last year covered by each index, for the run log."""
        return {
            name: (min(values), max(values)) for name, values in self.indices.items()
        }

    def describe(self) -> pd.DataFrame:
        """Human-readable summary for the assumptions log."""
        return pd.DataFrame(
            [
                {
                    "index_name": name,
                    "first_year": lo,
                    "last_year": hi,
                    "n_years": hi - lo + 1,
                    "source": self.source,
                }
                for name, (lo, hi) in sorted(self.coverage.items())
            ]
        )
