"""Behavior scorer interface."""

from abc import abstractmethod

from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from .base_scorer import BaseScorer


class BehaviorScorer(BaseScorer):
    """Abstract scorer for evaluating behavioral match."""

    @abstractmethod
    def score(self, context: ScoringContext) -> ScoreResult:
        """Calculate behavioral match score between candidate and job.

        Args:
            context: ScoringContext containing candidate and job description.

        Returns:
            ScoreResult: Behavioral match score with details.
        """
        pass
