"""Lightweight deterministic profile consistency checks."""

from typing import Dict, List

from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from .base_scorer import BaseScorer


class ConsistencyScorer(BaseScorer):
    """Penalize implausible or very incomplete profiles without ID rules."""

    SENIORITY_MINIMUM_YEARS = {
        "senior": 3.0,
        "lead": 4.0,
        "manager": 4.0,
        "staff": 5.0,
        "principal": 6.0,
        "architect": 5.0,
        "distinguished": 8.0,
    }

    def score(self, context: ScoringContext) -> ScoreResult:
        if not self.validate_inputs(context):
            return ScoreResult(0.0, 0.0, ["Invalid consistency inputs"], [], [], {"available": False})

        candidate = context.candidate
        penalties: List[Dict[str, object]] = []
        declared_years = max(float(candidate.profile.years_of_experience or 0.0), 0.0)
        history_months = sum(max(c.duration_months or 0, 0) for c in candidate.career_history or [])
        history_years = history_months / 12.0

        if history_years > 0:
            discrepancy = abs(declared_years - history_years)
            tolerance = max(1.5, declared_years * 0.30)
            if discrepancy > tolerance:
                severity = min(0.18, 0.06 + ((discrepancy - tolerance) / max(declared_years, history_years, 1.0)) * 0.20)
                penalties.append({"item": "Experience chronology", "penalty": severity,
                                  "reason": f"Declared experience ({declared_years:.1f}y) differs from career durations ({history_years:.1f}y)"})

        title = (candidate.profile.current_title or "").lower()
        required_years = max(
            (years for label, years in self.SENIORITY_MINIMUM_YEARS.items() if label in title),
            default=0.0,
        )
        if required_years and declared_years + 0.5 < required_years:
            penalties.append({"item": "Seniority plausibility", "penalty": 0.18,
                              "reason": f"Current seniority is unusually high for {declared_years:.1f} years of experience"})

        completeness = float(candidate.redrob_signals.profile_completeness_score or 0.0)
        if completeness < 0.35:
            penalties.append({"item": "Profile completeness", "penalty": min(0.20, (0.35 - completeness) * 0.50),
                              "reason": f"Profile completeness is extremely low ({completeness:.2f})"})

        dated_roles = [c for c in candidate.career_history or [] if c.duration_months is not None]
        short_roles = sum(1 for c in dated_roles if 0 < (c.duration_months or 0) < 4)
        if len(dated_roles) >= 3 and short_roles / len(dated_roles) >= 0.60:
            penalties.append({"item": "Career transitions", "penalty": 0.10,
                              "reason": "Most recorded roles are shorter than four months"})

        total_penalty = min(sum(float(item["penalty"]) for item in penalties), 0.50)
        score = 1.0 - total_penalty
        reasons = [str(item["reason"]) for item in penalties]
        if not reasons:
            reasons = ["Career chronology, seniority, and profile completeness are internally consistent"]

        return ScoreResult(
            score=score,
            confidence=0.90 if candidate.career_history else 0.55,
            reasons=reasons,
            matched_items=[] if penalties else ["Profile consistency"],
            missing_items=[str(item["item"]) for item in penalties],
            metadata={"available": True, "component": "consistency", "total_penalty": total_penalty,
                      "checks": penalties, "declared_years": declared_years, "history_years": history_years},
        )
