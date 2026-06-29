"""Job description domain model."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class JobDescription:
    """Represents a job description."""

    job_id: str
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    required_skills: List[str] = None
    required_experience_years: Optional[float] = None
    required_education: Optional[str] = None
    responsibilities: List[str] = None
    behaviors: List[str] = None

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.required_skills is None:
            object.__setattr__(self, "required_skills", [])
        if self.responsibilities is None:
            object.__setattr__(self, "responsibilities", [])
        if self.behaviors is None:
            object.__setattr__(self, "behaviors", [])
