"""Behavior domain model."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Behavior:
    """Represents a candidate's behavioral attributes."""

    trait: str
    description: Optional[str] = None
    score: Optional[float] = None
