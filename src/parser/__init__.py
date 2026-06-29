"""Parser interfaces for loading and validating JSON data."""

from .candidate_parser import CandidateParser
from .job_description_parser import JobDescriptionParser

__all__ = [
    "CandidateParser",
    "JobDescriptionParser",
]
