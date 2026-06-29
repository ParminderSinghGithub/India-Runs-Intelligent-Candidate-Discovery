"""Education domain model."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Education:
    """Represents a candidate's education history."""

    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: Optional[str] = None
