"""Score result domain model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ScoreResult:
    """Standardized result object returned by all scorers.

    Every scorer must return a ScoreResult instead of primitive floats.
    This provides consistent structure across all scoring dimensions.
    """

    score: float
    """Primary score value between 0.0 and 1.0."""

    confidence: float = 0.0
    """Confidence in the score between 0.0 and 1.0.
    Higher values indicate more reliable scoring."""

    reasons: List[str] = field(default_factory=list)
    """Human-readable explanations for the score."""

    matched_items: List[str] = field(default_factory=list)
    """Items that matched positively (e.g., skills, keywords)."""

    missing_items: List[str] = field(default_factory=list)
    """Items that were missing or didn't match."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional scorer-specific metadata."""

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if score confidence exceeds threshold.

        Args:
            threshold: Confidence threshold (default 0.7).

        Returns:
            bool: True if confidence >= threshold.
        """
        return self.confidence >= threshold

    def has_match(self, item: str) -> bool:
        """Check if a specific item was matched.

        Args:
            item: Item to check.

        Returns:
            bool: True if item is in matched_items.
        """
        return item in self.matched_items

    def has_missing(self, item: str) -> bool:
        """Check if a specific item was missing.

        Args:
            item: Item to check.

        Returns:
            bool: True if item is in missing_items.
        """
        return item in self.missing_items

    def add_reason(self, reason: str) -> "ScoreResult":
        """Add a reason to the result (returns new instance).

        Args:
            reason: Reason to add.

        Returns:
            ScoreResult: New ScoreResult with added reason.
        """
        new_reasons = list(self.reasons)
        new_reasons.append(reason)
        object.__setattr__(self, "reasons", new_reasons)
        return self

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Any: Metadata value or default.
        """
        return self.metadata.get(key, default)
