"""Career domain model."""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class Career:
    """Represents a candidate's career history."""

    company: str
    title: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    duration_months: Optional[int] = None
    is_current: bool = False
    industry: Optional[str] = None
    company_size: Optional[str] = None
    description: Optional[str] = None
