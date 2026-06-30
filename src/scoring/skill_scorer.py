"""Skill scorer interface."""

from abc import abstractmethod

from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from .base_scorer import BaseScorer


class SkillScorer(BaseScorer):
    """Abstract scorer for evaluating skill match."""

    @abstractmethod
    def score(self, context: ScoringContext) -> ScoreResult:
        """Calculate skill match score between candidate and job.

        Args:
            context: ScoringContext containing candidate and job description.

        Returns:
            ScoreResult: Skill match score with details.
        """
        pass
