"""Career domain model."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Career:
    """Represents a candidate's career history."""

    company: str
    position: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    is_current: bool = False
