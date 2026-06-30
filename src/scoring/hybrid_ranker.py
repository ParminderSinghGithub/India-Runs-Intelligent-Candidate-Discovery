"""Hybrid ranker interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext


class HybridRanker(ABC):
    """Abstract ranker for combining multiple scoring dimensions.

    Orchestrates multiple scorers and merges their results into a final
    ranking. Does not implement scoring logic itself - that's handled
    by individual scorers.
    """

    @abstractmethod
    def rank(self, context: ScoringContext) -> List[Dict[str, Any]]:
        """Rank candidates based on hybrid scoring.

        Args:
            context: ScoringContext containing candidates and job description.
                    The context should contain a list of candidates in metadata
                    or config.

        Returns:
            List[Dict[str, Any]]: Ranked list of candidates with scores.
                Each dict should contain candidate_id and total_score.

        Raises:
            ScorerError: If ranking fails.
        """
        pass

    @abstractmethod
    def calculate_hybrid_score(self, context: ScoringContext) -> ScoreResult:
        """Calculate hybrid score for a single candidate.

        Args:
            context: ScoringContext containing candidate and job description.

        Returns:
            ScoreResult: Hybrid score with merged details from all scorers.
        """
        pass
