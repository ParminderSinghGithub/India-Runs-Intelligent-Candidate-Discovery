"""Candidate domain model."""

from dataclasses import dataclass
from typing import List, Optional

from .career import Career
from .skill import Skill
from .behavior import Behavior
from .education import Education
from .profile import Profile
from .certification import Certification
from .language import Language
from .redrob_signals import RedrobSignals


@dataclass(frozen=True)
class Candidate:
    """Represents a candidate profile."""

    candidate_id: str
    profile: Profile
    career_history: List[Career]
    education: List[Education]
    skills: List[Skill]
    redrob_signals: RedrobSignals
    certifications: List[Certification] = None
    languages: List[Language] = None
    behaviors: List[Behavior] = None

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.certifications is None:
            object.__setattr__(self, "certifications", [])
        if self.languages is None:
            object.__setattr__(self, "languages", [])
        if self.behaviors is None:
            object.__setattr__(self, "behaviors", [])
