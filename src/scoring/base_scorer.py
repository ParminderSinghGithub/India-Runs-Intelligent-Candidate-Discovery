"""Base scorer interface."""

from abc import ABC, abstractmethod

from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext


class BaseScorer(ABC):
    """Abstract base class for all scorers.

    All scorers must implement the score method that accepts ScoringContext
    and returns ScoreResult. This ensures consistent interface across all
    scoring dimensions.
    """

    @abstractmethod
    def score(self, context: ScoringContext) -> ScoreResult:
        """Calculate a score based on the provided context.

        Args:
            context: ScoringContext containing candidate, job description,
                    and configuration.

        Returns:
            ScoreResult: Standardized scoring result with score, confidence,
                         reasons, matched/missing items, and metadata.

        Raises:
            ScorerError: If scoring fails.
        """
        pass

    def validate_inputs(self, context: ScoringContext) -> bool:
        """Validate inputs before scoring.

        Default implementation checks that candidate and job_description
        are present in context. Subclasses can override for specific validation.

        Args:
            context: ScoringContext to validate.

        Returns:
            bool: True if inputs are valid, False otherwise.
        """
        return (
            context.candidate is not None
            and context.job_description is not None
        )
