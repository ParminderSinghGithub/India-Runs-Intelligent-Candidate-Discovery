"""Search query domain model for retrieval."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SearchQuery:
    """Structured search query for candidate retrieval.

    Each field represents a different query dimension that can be
    embedded separately for multi-vector retrieval.
    """

    identity_query: str
    """Query focused on role identity (title, seniority, domain)."""

    career_query: str
    """Query focused on career history and experience."""

    skills_query: str
    """Query focused on technical skills and technologies."""

    combined_query: str
    """Combined query for general similarity search."""

    def __str__(self) -> str:
        """String representation of the search query."""
        return f"SearchQuery(identity={self.identity_query[:50]}..., career={self.career_query[:50]}..., skills={self.skills_query[:50]}...)"
