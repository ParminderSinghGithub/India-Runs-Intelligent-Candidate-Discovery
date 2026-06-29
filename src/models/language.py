"""Language domain model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """Represents a candidate's language proficiency."""

    language: str
    proficiency: str
