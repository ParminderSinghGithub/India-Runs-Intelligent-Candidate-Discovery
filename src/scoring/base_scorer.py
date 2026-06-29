"""Base scorer interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseScorer(ABC):
    """Abstract base class for all scorers."""

    @abstractmethod
    def score(self, **kwargs: Any) -> float:
        """Calculate a score based on provided inputs.

        Args:
            **kwargs: Arbitrary keyword arguments specific to the scorer.

        Returns:
            float: Calculated score between 0.0 and 1.0.

        Raises:
            ScorerError: If scoring fails.
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs: Any) -> bool:
        """Validate inputs before scoring.

        Args:
            **kwargs: Arbitrary keyword arguments to validate.

        Returns:
            bool: True if inputs are valid, False otherwise.
        """
        pass
