"""Skill domain model."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Skill:
    """Represents a candidate's skill."""

    name: str
    proficiency: Optional[str] = None
    years_experience: Optional[float] = None
    last_used: Optional[str] = None
