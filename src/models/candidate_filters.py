"""Candidate filters domain model for retrieval."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class CandidateFilters:
    """Filters for candidate retrieval.

    These filters are applied to narrow down the candidate pool
    before or during retrieval.
    """

    minimum_experience_years: Optional[float] = None
    """Minimum years of experience required."""

    maximum_experience_years: Optional[float] = None
    """Maximum years of experience allowed."""

    required_location: Optional[str] = None
    """Required location (city, country, or region)."""

    required_industries: Optional[List[str]] = None
    """List of required industries."""

    required_work_mode: Optional[str] = None
    """Required work mode (remote, on-site, hybrid)."""

    open_to_work: Optional[bool] = None
    """Whether candidate must be open to work."""

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.required_industries is None:
            object.__setattr__(self, "required_industries", [])

    def has_filters(self) -> bool:
        """Check if any filters are set.

        Returns:
            bool: True if any filter is set, False otherwise.
        """
        return (
            self.minimum_experience_years is not None
            or self.maximum_experience_years is not None
            or self.required_location is not None
            or self.required_industries
            or self.required_work_mode is not None
            or self.open_to_work is not None
        )
