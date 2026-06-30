"""Semantic scorer interface."""

from abc import abstractmethod

from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from .base_scorer import BaseScorer


class SemanticScorer(BaseScorer):
    """Abstract scorer for evaluating semantic similarity."""

    @abstractmethod
    def score(self, context: ScoringContext) -> ScoreResult:
        """Calculate semantic similarity score between candidate and job.

        Args:
            context: ScoringContext containing candidate and job description.

        Returns:
            ScoreResult: Semantic similarity score with details.
        """
        pass
