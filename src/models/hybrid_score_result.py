"""Hybrid score result domain model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class HybridScoreResult:
    """Final ranking result for a single retrieved candidate."""

    candidate_id: str
    semantic_score: float
    career_score: float
    skill_score: float
    behavior_score: float
    education_score: float
    consistency_score: float
    weighted_final_score: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    matched_items: List[str] = field(default_factory=list)
    missing_items: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a serializable dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "semantic_score": self.semantic_score,
            "career_score": self.career_score,
            "skill_score": self.skill_score,
            "behavior_score": self.behavior_score,
            "education_score": self.education_score,
            "consistency_score": self.consistency_score,
            "weighted_final_score": self.weighted_final_score,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "matched_items": self.matched_items,
            "missing_items": self.missing_items,
            "reasons": self.reasons,
        }
