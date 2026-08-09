"""
crosswalk.py - The WBS name crosswalk, as a persisted artifact.

Contractors rename WBS elements between submissions. "Airframe" becomes "Air
Frame" becomes "AIRFRAME" becomes "Airframe Structure", and an analyst who
joins on the name gets four elements where there is one. Fixing that inline --
a chain of ``str.replace`` calls buried in a loading function -- is how a
pipeline becomes impossible to review, because the mapping that decides what
rolls up where is not written down anywhere a reviewer can see it.

So the mapping lives in a file. It is loaded, applied, and saved back out with
every run, and the rule used to resolve each individual name is recorded in the
provenance table. Three rules exist, in order of preference:

``exact``
    The reported name is in the crosswalk verbatim.
``casefold``
    The reported name matches an entry after lowercasing and collapsing
    whitespace. Enabled by default and recorded as such -- it is a lookup
    policy, not a mapping decision, and it never invents a target that is not
    already in the file.
``unmatched``
    Nothing matched. The name is *surfaced*, never dropped and never guessed
    at. :meth:`Crosswalk.suggest` will propose candidates for a human to
    approve, but nothing auto-applies: silently absorbing an unrecognised
    element is how cost disappears from an estimate without anyone noticing.
"""

from __future__ import annotations

import csv
import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CROSSWALK_COLUMNS = ("reported_name", "canonical_name", "wbs_code", "rule", "notes")

_WHITESPACE = re.compile(r"\s+")


def normalize_key(name: str) -> str:
    """Lowercase and collapse whitespace, for the ``casefold`` rule."""
    return _WHITESPACE.sub(" ", str(name).strip()).casefold()


@dataclass(frozen=True)
class ResolvedName:
    """One name resolution and how it was reached.

    Attributes:
        reported: The name exactly as it appeared in the source file.
        canonical: The canonical name, or None if unmatched.
        wbs_code: Code for the canonical element, if the crosswalk knows one.
        rule: ``"exact"``, ``"casefold"`` or ``"unmatched"``.
    """

    reported: str
    canonical: str | None
    wbs_code: str | None
    rule: str

    @property
    def matched(self) -> bool:
        return self.canonical is not None


class CrosswalkError(ValueError):
    """Raised when a crosswalk artifact is malformed or cannot be applied."""


@dataclass
class Crosswalk:
    """An explicit reported-name to canonical-name mapping.

    Attributes:
        mapping: Reported name -> canonical name, as written in the artifact.
        codes: Canonical name -> WBS code.
        allow_casefold: Whether to fall back to case-insensitive matching.
        notes: Free-text per reported name, carried through save/load so the
            reason for a mapping survives.
    """

    mapping: dict[str, str] = field(default_factory=dict)
    codes: dict[str, str] = field(default_factory=dict)
    allow_casefold: bool = True
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._folded = {normalize_key(k): v for k, v in self.mapping.items()}

    # ------------------------------------------------------------- building
    @classmethod
    def from_wbs(
        cls,
        wbs,
        variants: dict[str, tuple[str, ...]] | None = None,
        *,
        allow_casefold: bool = True,
    ) -> "Crosswalk":
        """Build a crosswalk from a WBS definition and its known aliases.

        Args:
            wbs: Iterable of objects with ``code`` and ``name`` attributes.
            variants: Canonical name -> aliases.
        """
        mapping: dict[str, str] = {}
        codes: dict[str, str] = {}
        for element in wbs:
            mapping[element.name] = element.name
            codes[element.name] = element.code
            for alias in (variants or {}).get(element.name, ()):
                mapping[alias] = element.name
        return cls(mapping=mapping, codes=codes, allow_casefold=allow_casefold)

    @classmethod
    def default(cls, *, allow_casefold: bool = True) -> "Crosswalk":
        """The crosswalk for the bundled synthetic WBS."""
        from cost_core.synth.spec import DEFAULT_NAME_VARIANTS, DEFAULT_WBS

        return cls.from_wbs(
            [w for w in DEFAULT_WBS if w.is_leaf],
            DEFAULT_NAME_VARIANTS,
            allow_casefold=allow_casefold,
        )

    # ---------------------------------------------------------- persistence
    @classmethod
    def load(cls, path: str | Path, *, allow_casefold: bool = True) -> "Crosswalk":
        """Read a crosswalk artifact from CSV.

        Raises:
            FileNotFoundError: If the artifact is missing. The pipeline will
                not invent one -- an absent crosswalk is a missing decision,
                not a default.
            CrosswalkError: If required columns are absent, or a reported name
                appears twice with different canonical targets.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No crosswalk artifact at {path}. Generate one with "
                f"Crosswalk.default().save(path) and review it before use."
            )

        frame = pd.read_csv(path, dtype=str).fillna("")
        missing = [c for c in ("reported_name", "canonical_name") if c not in frame]
        if missing:
            raise CrosswalkError(
                f"Crosswalk at {path} is missing required column(s): {missing}. "
                f"Found: {list(frame.columns)}."
            )

        mapping: dict[str, str] = {}
        codes: dict[str, str] = {}
        notes: dict[str, str] = {}
        for _, row in frame.iterrows():
            reported = str(row["reported_name"]).strip()
            canonical = str(row["canonical_name"]).strip()
            if not reported or not canonical:
                continue
            if reported in mapping and mapping[reported] != canonical:
                raise CrosswalkError(
                    f"Crosswalk at {path} maps {reported!r} to both "
                    f"{mapping[reported]!r} and {canonical!r}. A name cannot "
                    f"roll up to two places; resolve the conflict in the file."
                )
            mapping[reported] = canonical
            code = str(row.get("wbs_code", "")).strip()
            if code:
                codes[canonical] = code
            note = str(row.get("notes", "")).strip()
            if note:
                notes[reported] = note

        logger.info("Loaded crosswalk from %s: %d entries", path, len(mapping))
        return cls(
            mapping=mapping, codes=codes, allow_casefold=allow_casefold, notes=notes
        )

    def save(self, path: str | Path) -> Path:
        """Write the crosswalk artifact to CSV, sorted for a readable diff."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CROSSWALK_COLUMNS)
            for reported in sorted(self.mapping):
                canonical = self.mapping[reported]
                writer.writerow(
                    [
                        reported,
                        canonical,
                        self.codes.get(canonical, ""),
                        "exact" if reported == canonical else "alias",
                        self.notes.get(reported, ""),
                    ]
                )
        logger.info("Wrote crosswalk artifact to %s (%d entries)", path, len(self.mapping))
        return path

    # ------------------------------------------------------------ resolving
    def resolve(self, name: str) -> ResolvedName:
        """Resolve one reported name, recording which rule was used."""
        reported = str(name)
        stripped = reported.strip()

        if stripped in self.mapping:
            canonical = self.mapping[stripped]
            return ResolvedName(reported, canonical, self.codes.get(canonical), "exact")

        if self.allow_casefold:
            folded = self._folded.get(normalize_key(stripped))
            if folded is not None:
                return ResolvedName(
                    reported, folded, self.codes.get(folded), "casefold"
                )

        return ResolvedName(reported, None, None, "unmatched")

    def apply(self, names) -> pd.DataFrame:
        """Resolve a column of reported names.

        Returns:
            DataFrame with ``reported``, ``canonical``, ``wbs_code`` and
            ``rule``, aligned to the input order. Unmatched rows carry a null
            canonical rather than being dropped.
        """
        resolutions = [self.resolve(n) for n in names]
        return pd.DataFrame(
            {
                "reported": [r.reported for r in resolutions],
                "canonical": [r.canonical for r in resolutions],
                "wbs_code": [r.wbs_code for r in resolutions],
                "rule": [r.rule for r in resolutions],
            },
            index=getattr(names, "index", None),
        )

    def unmatched(self, names) -> list[str]:
        """Distinct reported names this crosswalk cannot resolve, sorted."""
        return sorted({str(n) for n in names if not self.resolve(n).matched})

    def suggest(self, name: str, n: int = 3, cutoff: float = 0.6) -> list[str]:
        """Propose canonical names for an unmatched reported name.

        Suggestions only. Nothing here is ever applied automatically: a wrong
        auto-match moves real cost into the wrong element and nothing
        downstream would flag it.
        """
        return difflib.get_close_matches(
            normalize_key(name),
            [normalize_key(c) for c in set(self.mapping.values())],
            n=n,
            cutoff=cutoff,
        )

    def with_additions(self, additions: dict[str, str], note: str = "") -> "Crosswalk":
        """Return a new crosswalk with extra mappings, validating the targets.

        Raises:
            CrosswalkError: If an addition points at a canonical name the
                crosswalk does not already know, which almost always means a
                typo that would create a phantom WBS element.
        """
        known = set(self.mapping.values())
        unknown = {v for v in additions.values() if v not in known}
        if unknown:
            raise CrosswalkError(
                f"Cannot map to unknown canonical name(s): {sorted(unknown)}. "
                f"Known: {sorted(known)}."
            )
        merged = {**self.mapping, **additions}
        stamp = note or f"added {date.today().isoformat()}"
        return Crosswalk(
            mapping=merged,
            codes=dict(self.codes),
            allow_casefold=self.allow_casefold,
            notes={**self.notes, **{k: stamp for k in additions}},
        )

    def __len__(self) -> int:
        return len(self.mapping)

    @property
    def canonical_names(self) -> set[str]:
        return set(self.mapping.values())
