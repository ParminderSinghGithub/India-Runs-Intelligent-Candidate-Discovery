"""Hybrid ranker interface."""

from abc import abstractmethod
from typing import Any, Dict, List

from src.models.candidate import Candidate
from src.models.job_description import JobDescription


class HybridRanker:
    """Abstract ranker for combining multiple scoring dimensions."""

    @abstractmethod
    def rank(
        self,
        candidates: List[Candidate],
        job_description: JobDescription,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rank candidates based on hybrid scoring.

        Args:
            candidates: List of candidate objects to rank.
            job_description: Job description to match against.
            **kwargs: Additional ranking parameters.

        Returns:
            List[Dict[str, Any]]: Ranked list of candidates with scores.
                Each dict should contain candidate_id and total_score.

        Raises:
            ScorerError: If ranking fails.
        """
        pass

    @abstractmethod
    def calculate_hybrid_score(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        **kwargs: Any,
    ) -> float:
        """Calculate hybrid score for a single candidate.

        Args:
            candidate: Candidate object.
            job_description: Job description object.
            **kwargs: Additional scoring parameters.

        Returns:
            float: Hybrid score between 0.0 and 1.0.
        """
        pass
