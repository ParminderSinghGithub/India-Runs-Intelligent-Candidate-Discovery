"""Profile domain model."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Profile:
    """Represents a candidate's profile information."""

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
