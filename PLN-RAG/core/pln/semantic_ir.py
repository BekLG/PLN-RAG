from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Literal:
    """Typed predicate occurrence used before rendering a PLN atom."""

    head: str
    args: tuple[str, ...]
    negated: bool = False


@dataclass(frozen=True)
class NumericValue:
    value: float
    unit: str = ""


@dataclass(frozen=True)
class Measurement:
    """A measured property, optionally awaiting an unambiguous owner."""

    entity: str
    property_terms: frozenset[str]
    numeric: NumericValue
    source: Literal
    owner_missing: bool = False


@dataclass(frozen=True)
class NumericConstraint:
    target: Literal
    comparator: str
    threshold: NumericValue
    property_terms: frozenset[str]
    secondary: tuple[NumericValue, ...] = field(default_factory=tuple)

