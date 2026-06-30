"""Parsed job domain model combining all job-related data."""

from dataclasses import dataclass

from src.models.candidate_filters import CandidateFilters
from src.models.job_description import JobDescription
from src.models.search_query import SearchQuery


@dataclass(frozen=True)
class ParsedJob:
    """Complete parsed job information for retrieval.

    This is the single input object that will be passed to the retrieval engine.
    It contains the structured job description, search queries, and filters.
    """

    job_description: JobDescription
    """Structured job description with all extracted fields."""

    search_query: SearchQuery
    """Search queries for different retrieval dimensions."""

    candidate_filters: CandidateFilters
    """Filters to apply during candidate retrieval."""

    def __str__(self) -> str:
        """String representation of the parsed job."""
        return f"ParsedJob(title={self.job_description.title}, company={self.job_description.company})"
