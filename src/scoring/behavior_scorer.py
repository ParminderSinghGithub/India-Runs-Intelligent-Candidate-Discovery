"""Behavior scorer interface."""

from abc import abstractmethod
from typing import Any

from src.models.candidate import Candidate
from src.models.job_description import JobDescription
from .base_scorer import BaseScorer


class BehaviorScorer(BaseScorer):
    """Abstract scorer for evaluating behavioral match."""

    @abstractmethod
    def score(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        **kwargs: Any,
    ) -> float:
        """Calculate behavioral match score between candidate and job.

        Args:
            candidate: Candidate object containing behavioral traits.
            job_description: Job description object with required behaviors.
            **kwargs: Additional scoring parameters.

        Returns:
            float: Behavioral match score between 0.0 and 1.0.
        """
        pass

    @abstractmethod
    def validate_inputs(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        **kwargs: Any,
    ) -> bool:
        """Validate inputs for behavioral scoring.

        Args:
            candidate: Candidate object.
            job_description: Job description object.
            **kwargs: Additional parameters to validate.

        Returns:
            bool: True if inputs are valid.
        """
        pass
