"""Candidate domain model."""

from dataclasses import dataclass
from typing import List, Optional

from .career import Career
from .skill import Skill
from .behavior import Behavior
from .education import Education


@dataclass(frozen=True)
class Candidate:
    """Represents a candidate profile."""

    candidate_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    careers: List[Career] = None
    skills: List[Skill] = None
    behaviors: List[Behavior] = None
    education: List[Education] = None
    summary: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.careers is None:
            object.__setattr__(self, "careers", [])
        if self.skills is None:
            object.__setattr__(self, "skills", [])
        if self.behaviors is None:
            object.__setattr__(self, "behaviors", [])
        if self.education is None:
            object.__setattr__(self, "education", [])
