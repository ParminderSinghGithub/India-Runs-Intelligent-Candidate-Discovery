"""Education domain model."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Education:
    """Represents a candidate's education history."""

    institution: str
    degree: str
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[float] = None
