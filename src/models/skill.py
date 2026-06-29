"""Skill domain model."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Skill:
    """Represents a candidate's skill."""

    name: str
    proficiency: str
    endorsements: int
    duration_months: Optional[int] = None
