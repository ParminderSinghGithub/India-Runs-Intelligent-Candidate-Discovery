"""Certification domain model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Certification:
    """Represents a candidate's certification."""

    name: str
    issuer: str
    year: int
