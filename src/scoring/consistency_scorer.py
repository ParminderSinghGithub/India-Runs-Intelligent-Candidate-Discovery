"""Consistency scorer interface."""

from abc import abstractmethod

from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from .base_scorer import BaseScorer


class ConsistencyScorer(BaseScorer):
    """Abstract scorer for evaluating profile consistency."""

    @abstractmethod
    def score(self, context: ScoringContext) -> ScoreResult:
        """Calculate consistency score for candidate profile.

        Args:
            context: ScoringContext containing candidate and job description.

        Returns:
            ScoreResult: Consistency score with details.
        """
        pass
