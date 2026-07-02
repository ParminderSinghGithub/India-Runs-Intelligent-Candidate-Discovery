"""Submission row domain model."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class SubmissionRow:
    """Represents one row in the final submission CSV."""

    candidate_id: str
    rank: int
    score: float
    reasoning: str
    serializable: bool = True

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert the row to a CSV-ready dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "score": self.score,
            "reasoning": self.reasoning,
        }
