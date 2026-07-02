"""Behavior scorer for recruiter-centric candidate signals."""

import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    BEHAVIOR_ACTIVITY_APPLICATIONS_WEIGHT,
    BEHAVIOR_ACTIVITY_GITHUB_WEIGHT,
    BEHAVIOR_ACTIVITY_RECENCY_HALF_LIFE_DAYS,
    BEHAVIOR_ACTIVITY_RECENCY_WEIGHT,
    BEHAVIOR_AVAILABILITY_OPEN_WEIGHT,
    BEHAVIOR_AVAILABILITY_NOTICE_WEIGHT,
    BEHAVIOR_AVAILABILITY_RELOCATE_WEIGHT,
    BEHAVIOR_AVAILABILITY_WORK_MODE_WEIGHT,
    BEHAVIOR_AVAILABILITY_WEIGHT,
    BEHAVIOR_CANDIDATE_ACTIVITY_WEIGHT,
    BEHAVIOR_PROFILE_QUALITY_WEIGHT,
    BEHAVIOR_GITHUB_ACTIVITY_THRESHOLD,
    BEHAVIOR_HIRING_RELIABILITY_WEIGHT,
    BEHAVIOR_INACTIVE_PROFILE_PENALTY_WEIGHT,
    BEHAVIOR_INCOMPLETE_PROFILE_PENALTY_WEIGHT,
    BEHAVIOR_INTERVIEW_COMPLETION_THRESHOLD,
    BEHAVIOR_LOW_ENGAGEMENT_PENALTY_WEIGHT,
    BEHAVIOR_LOG_COUNT_NORMALIZATION_CAP,
    BEHAVIOR_MAX_TOTAL_PENALTY,
    BEHAVIOR_MIN_AVAILABILITY_SCORE,
    BEHAVIOR_MIN_NOT_OPEN_SCORE,
    BEHAVIOR_NETWORK_CONNECTIONS_WEIGHT,
    BEHAVIOR_NETWORK_ENDORSEMENTS_WEIGHT,
    BEHAVIOR_NOTICE_PERIOD_MAX_DAYS,
    BEHAVIOR_NOTICE_PERIOD_RELIEF_FACTOR,
    BEHAVIOR_NOT_RELOCATE_SCORE,
    BEHAVIOR_OFFER_ACCEPTANCE_THRESHOLD,
    BEHAVIOR_OPEN_TO_WORK_SCORE,
    BEHAVIOR_POOR_RELIABILITY_PENALTY_WEIGHT,
    BEHAVIOR_PROFILE_COMPLETENESS_WEIGHT,
    BEHAVIOR_PROFILE_EMAIL_WEIGHT,
    BEHAVIOR_PROFILE_QUALITY_INCOMPLETE_THRESHOLD,
    BEHAVIOR_PROFILE_PHONE_WEIGHT,
    BEHAVIOR_PROFILE_LINKEDIN_WEIGHT,
    BEHAVIOR_PROFESSIONAL_NETWORK_WEIGHT,
    BEHAVIOR_RECRUITER_INTEREST_WEIGHT,
    BEHAVIOR_RECRUITER_SEARCH_WEIGHT,
    BEHAVIOR_RECRUITER_SAVES_WEIGHT,
    BEHAVIOR_RECRUITER_VIEWS_WEIGHT,
    BEHAVIOR_RELIABILITY_INTERVIEW_WEIGHT,
    BEHAVIOR_RELIABILITY_OFFER_WEIGHT,
    BEHAVIOR_RELOCATE_SCORE,
    BEHAVIOR_TOTAL_SIGNALS_EXPECTED,
    BEHAVIOR_UNVERIFIED_PROFILE_PENALTY_WEIGHT,
    BEHAVIOR_WORK_MODE_SCORES,
)
from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from .base_scorer import BaseScorer


class BehaviorScorer(BaseScorer):
    """Deterministic scorer for recruiter-centric behavioral signals.

    This scorer evaluates only `Candidate.redrob_signals` and answers whether
    the candidate is likely to attract recruiter interest and be easy to hire.
    """

    def score(self, context: ScoringContext) -> ScoreResult:
        """Calculate behavioral match score between candidate and job."""
        if not self.validate_inputs(context):
            return ScoreResult(
                score=0.0,
                confidence=0.0,
                reasons=["Invalid inputs for behavior scoring"],
                matched_items=[],
                missing_items=[],
                metadata={
                    "component": "behavior",
                    "partial_scores": {},
                    "evidence_count": 0,
                },
            )

        signals = context.candidate.redrob_signals

        profile_quality_score, profile_evidence, profile_available = self._score_profile_quality(signals)
        recruiter_interest_score, recruiter_evidence, recruiter_available = self._score_recruiter_interest(signals)
        activity_score, activity_evidence, activity_available = self._score_candidate_activity(signals)
        reliability_score, reliability_evidence, reliability_available = self._score_hiring_reliability(signals)
        availability_score, availability_evidence, availability_available = self._score_availability(signals)
        network_score, network_evidence, network_available = self._score_professional_network(signals)

        all_evidence = self._collect_evidence(
            profile_evidence,
            recruiter_evidence,
            activity_evidence,
            reliability_evidence,
            availability_evidence,
            network_evidence,
        )

        weighted_score = (
            profile_quality_score * BEHAVIOR_PROFILE_QUALITY_WEIGHT
            + recruiter_interest_score * BEHAVIOR_RECRUITER_INTEREST_WEIGHT
            + activity_score * BEHAVIOR_CANDIDATE_ACTIVITY_WEIGHT
            + reliability_score * BEHAVIOR_HIRING_RELIABILITY_WEIGHT
            + availability_score * BEHAVIOR_AVAILABILITY_WEIGHT
            + network_score * BEHAVIOR_PROFESSIONAL_NETWORK_WEIGHT
        )

        penalty_score, penalty_evidence = self._calculate_penalties(
            signals,
            profile_quality_score,
            recruiter_interest_score,
            reliability_score,
        )
        all_evidence = self._collect_evidence(all_evidence, penalty_evidence)

        final_score = max(0.0, min(weighted_score - penalty_score, 1.0))

        matched_items = self._extract_unique_items(all_evidence, "match")
        missing_items = self._extract_unique_items(all_evidence, "missing")
        reasons = [item["reason"] for item in all_evidence]

        available_signal_count = sum(
            [
                profile_available,
                recruiter_available,
                activity_available,
                reliability_available,
                availability_available,
                network_available,
            ]
        )
        confidence = self._calculate_confidence(
            available_signal_count=available_signal_count,
            component_scores=[
                profile_quality_score,
                recruiter_interest_score,
                activity_score,
                reliability_score,
                availability_score,
                network_score,
            ],
            evidence_count=len(all_evidence),
        )

        partial_scores = {
            "profile_quality": profile_quality_score,
            "recruiter_interest": recruiter_interest_score,
            "candidate_activity": activity_score,
            "hiring_reliability": reliability_score,
            "availability": availability_score,
            "professional_network": network_score,
            "risk_penalties": penalty_score,
            "score_before_penalty": weighted_score,
        }

        return ScoreResult(
            score=final_score,
            confidence=confidence,
            matched_items=matched_items,
            missing_items=missing_items,
            reasons=reasons,
            metadata={
                "component": "behavior",
                "partial_scores": partial_scores,
                "evidence_count": len(all_evidence),
            },
        )

    def _score_profile_quality(
        self,
        signals: Any,
    ) -> Tuple[float, List[Dict[str, str]], int]:
        """Score trustworthiness and completeness of the candidate profile."""
        evidence: List[Dict[str, str]] = []
        available = 0

        completeness = self._clamp_unit(self._to_float(signals.profile_completeness_score, 0.0))
        available += 1
        if completeness >= BEHAVIOR_PROFILE_QUALITY_INCOMPLETE_THRESHOLD:
            evidence.append(self._evidence("match", "Profile completeness", f"Profile completeness score is strong ({completeness:.2f})"))
        else:
            evidence.append(self._evidence("missing", "Profile completeness", f"Very incomplete profile ({completeness:.2f})"))

        email_score = 1.0 if signals.verified_email else 0.0
        available += 1
        if signals.verified_email:
            evidence.append(self._evidence("match", "Verified email", "Profile is verified by email"))
        else:
            evidence.append(self._evidence("missing", "Missing email verification", "Email verification is missing"))

        phone_score = 1.0 if signals.verified_phone else 0.0
        available += 1
        if signals.verified_phone:
            evidence.append(self._evidence("match", "Verified phone", "Profile is verified by phone"))
        else:
            evidence.append(self._evidence("missing", "Missing phone verification", "Phone verification is missing"))

        linkedin_score = 1.0 if signals.linkedin_connected else 0.0
        available += 1
        if signals.linkedin_connected:
            evidence.append(self._evidence("match", "LinkedIn connected", "LinkedIn account is connected"))
        else:
            evidence.append(self._evidence("missing", "LinkedIn not connected", "LinkedIn account is not connected"))

        score = (
            completeness * BEHAVIOR_PROFILE_COMPLETENESS_WEIGHT
            + email_score * BEHAVIOR_PROFILE_EMAIL_WEIGHT
            + phone_score * BEHAVIOR_PROFILE_PHONE_WEIGHT
            + linkedin_score * BEHAVIOR_PROFILE_LINKEDIN_WEIGHT
        )

        return score, evidence, available

    def _score_recruiter_interest(
        self,
        signals: Any,
    ) -> Tuple[float, List[Dict[str, str]], int]:
        """Score recruiter engagement and visibility."""
        evidence: List[Dict[str, str]] = []
        available = 3

        views = self._normalize_log_count(signals.profile_views_received_30d)
        saved = self._normalize_log_count(signals.saved_by_recruiters_30d)
        search = self._normalize_log_count(signals.search_appearance_30d)

        if signals.profile_views_received_30d > 0:
            evidence.append(self._evidence("match", "Recruiter profile views", f"Profile received {signals.profile_views_received_30d} recruiter views"))
        else:
            evidence.append(self._evidence("missing", "Low recruiter visibility", "No recruiter profile views received"))

        if signals.saved_by_recruiters_30d > 0:
            evidence.append(self._evidence("match", "Saved by recruiters", f"Saved by recruiters {signals.saved_by_recruiters_30d} times in 30 days"))
        else:
            evidence.append(self._evidence("missing", "Low recruiter saves", "No recruiter saves in 30 days"))

        if signals.search_appearance_30d > 0:
            evidence.append(self._evidence("match", "Search appearance", f"Appeared in recruiter searches {signals.search_appearance_30d} times"))
        else:
            evidence.append(self._evidence("missing", "Low search appearance", "No recruiter search appearances in 30 days"))

        score = (
            views * BEHAVIOR_RECRUITER_VIEWS_WEIGHT
            + saved * BEHAVIOR_RECRUITER_SAVES_WEIGHT
            + search * BEHAVIOR_RECRUITER_SEARCH_WEIGHT
        )

        if score >= 0.65:
            evidence.append(self._evidence("match", "High recruiter engagement", f"Strong recruiter engagement ({score:.2f})"))
        elif score <= 0.20:
            evidence.append(self._evidence("missing", "Low recruiter engagement", f"Extremely low recruiter engagement ({score:.2f})"))

        return score, evidence, available

    def _score_candidate_activity(
        self,
        signals: Any,
    ) -> Tuple[float, List[Dict[str, str]], int]:
        """Score candidate activity and recency."""
        evidence: List[Dict[str, str]] = []
        available = 0
        scores: List[float] = []

        recency_score, recency_available = self._score_last_active(signals.last_active_date)
        recency_weight = BEHAVIOR_ACTIVITY_RECENCY_WEIGHT if recency_available else 0.0
        available += recency_available
        if recency_available:
            if recency_score >= 0.70:
                evidence.append(self._evidence("match", "Recently active", f"Candidate was recently active ({recency_score:.2f})"))
            else:
                evidence.append(self._evidence("missing", "Inactive profile", f"Candidate appears inactive ({recency_score:.2f})"))
        else:
            evidence.append(self._evidence("missing", "Last active date unavailable", "Last active date is missing"))

        applications_score = self._normalize_log_count(signals.applications_submitted_30d)
        applications_weight = BEHAVIOR_ACTIVITY_APPLICATIONS_WEIGHT
        available += 1
        if signals.applications_submitted_30d > 0:
            evidence.append(self._evidence("match", "Recent applications", f"Submitted {signals.applications_submitted_30d} applications in 30 days"))
        else:
            evidence.append(self._evidence("missing", "No recent applications", "No recent applications submitted"))

        github_available = self._is_missing_optional(signals.github_activity_score)
        if github_available:
            github_score = self._clamp_unit(self._to_float(signals.github_activity_score, 0.0))
            github_weight = BEHAVIOR_ACTIVITY_GITHUB_WEIGHT
            available += 1
            if github_score >= BEHAVIOR_GITHUB_ACTIVITY_THRESHOLD:
                evidence.append(self._evidence("match", "Strong GitHub activity", f"GitHub activity score is strong ({github_score:.2f})"))
            else:
                evidence.append(self._evidence("missing", "Weak GitHub activity", f"GitHub activity score is weak ({github_score:.2f})"))
        else:
            github_score = 0.0
            github_weight = 0.0
            evidence.append(self._evidence("missing", "GitHub activity unavailable", "GitHub activity score is unavailable"))

        weighted_total = (
            recency_score * recency_weight
            + applications_score * applications_weight
            + github_score * github_weight
        )
        total_weight = recency_weight + applications_weight + github_weight
        score = weighted_total / total_weight if total_weight > 0 else 0.5
        return score, evidence, available

    def _score_hiring_reliability(
        self,
        signals: Any,
    ) -> Tuple[float, List[Dict[str, str]], int]:
        """Score interview completion and offer acceptance reliability."""
        evidence: List[Dict[str, str]] = []
        available = 1

        interview_score = self._clamp_unit(self._to_float(signals.interview_completion_rate, 0.0))
        interview_weight = BEHAVIOR_RELIABILITY_INTERVIEW_WEIGHT
        if interview_score >= BEHAVIOR_INTERVIEW_COMPLETION_THRESHOLD:
            evidence.append(self._evidence("match", "High interview completion rate", f"Interview completion rate is strong ({interview_score:.2f})"))
        else:
            evidence.append(self._evidence("missing", "Low interview completion rate", f"Interview completion rate is low ({interview_score:.2f})"))

        offer_missing = self._is_missing_optional(signals.offer_acceptance_rate)
        if offer_missing:
            offer_score = 0.0
            offer_weight = 0.0
            evidence.append(self._evidence("missing", "Offer acceptance unavailable", "Offer acceptance rate is unavailable"))
        else:
            offer_score = self._clamp_unit(self._to_float(signals.offer_acceptance_rate, 0.0))
            offer_weight = BEHAVIOR_RELIABILITY_OFFER_WEIGHT
            available += 1
            if offer_score >= BEHAVIOR_OFFER_ACCEPTANCE_THRESHOLD:
                evidence.append(self._evidence("match", "High offer acceptance rate", f"Offer acceptance rate is strong ({offer_score:.2f})"))
            else:
                evidence.append(self._evidence("missing", "Low offer acceptance rate", f"Offer acceptance rate is low ({offer_score:.2f})"))

        weighted_total = interview_score * interview_weight + offer_score * offer_weight
        total_weight = interview_weight + offer_weight
        score = weighted_total / total_weight if total_weight > 0 else 0.5
        return score, evidence, available

    def _score_availability(
        self,
        signals: Any,
    ) -> Tuple[float, List[Dict[str, str]], int]:
        """Score candidate availability and hiring ease."""
        evidence: List[Dict[str, str]] = []
        available = 4

        open_score = BEHAVIOR_OPEN_TO_WORK_SCORE if signals.open_to_work_flag else BEHAVIOR_MIN_NOT_OPEN_SCORE
        if signals.open_to_work_flag:
            evidence.append(self._evidence("match", "Open to work", "Candidate is open to work"))
        else:
            evidence.append(self._evidence("missing", "Not open to work", "Candidate is not explicitly open to work"))

        notice_score = self._normalize_notice_period(signals.notice_period_days)
        if signals.notice_period_days <= 30:
            evidence.append(self._evidence("match", "Short notice period", f"Short notice period ({signals.notice_period_days} days)"))
        elif signals.notice_period_days <= 90:
            evidence.append(self._evidence("match", "Moderate notice period", f"Moderate notice period ({signals.notice_period_days} days)"))
        else:
            evidence.append(self._evidence("missing", "Long notice period", f"Long notice period ({signals.notice_period_days} days)"))

        work_mode_score = self._score_work_mode(signals.preferred_work_mode)
        if work_mode_score >= 0.90:
            evidence.append(self._evidence("match", "Flexible work mode", f"Preferred work mode is flexible ({signals.preferred_work_mode})"))
        else:
            evidence.append(self._evidence("missing", "Restrictive work mode", f"Preferred work mode is less flexible ({signals.preferred_work_mode})"))

        relocate_score = BEHAVIOR_RELOCATE_SCORE if signals.willing_to_relocate else BEHAVIOR_NOT_RELOCATE_SCORE
        if signals.willing_to_relocate:
            evidence.append(self._evidence("match", "Willing to relocate", "Candidate is willing to relocate"))
        else:
            evidence.append(self._evidence("missing", "Not willing to relocate", "Candidate is not willing to relocate"))

        score = (
            open_score * BEHAVIOR_AVAILABILITY_OPEN_WEIGHT
            + notice_score * BEHAVIOR_AVAILABILITY_NOTICE_WEIGHT
            + work_mode_score * BEHAVIOR_AVAILABILITY_WORK_MODE_WEIGHT
            + relocate_score * BEHAVIOR_AVAILABILITY_RELOCATE_WEIGHT
        )

        return score, evidence, available

    def _score_professional_network(
        self,
        signals: Any,
    ) -> Tuple[float, List[Dict[str, str]], int]:
        """Score professional network depth and endorsements."""
        evidence: List[Dict[str, str]] = []
        available = 2

        connections_score = self._normalize_log_count(signals.connection_count)
        endorsements_score = self._normalize_log_count(signals.endorsements_received)

        if signals.connection_count > 0:
            evidence.append(self._evidence("match", "Professional connections", f"Has {signals.connection_count} professional connections"))
        else:
            evidence.append(self._evidence("missing", "Zero connections", "No professional connections recorded"))

        if signals.endorsements_received > 0:
            evidence.append(self._evidence("match", "Received endorsements", f"Received {signals.endorsements_received} endorsements"))
        else:
            evidence.append(self._evidence("missing", "Zero endorsements", "No endorsements received"))

        score = (
            connections_score * BEHAVIOR_NETWORK_CONNECTIONS_WEIGHT
            + endorsements_score * BEHAVIOR_NETWORK_ENDORSEMENTS_WEIGHT
        )
        return score, evidence, available

    def _calculate_penalties(
        self,
        signals: Any,
        profile_quality_score: float,
        recruiter_interest_score: float,
        reliability_score: float,
    ) -> Tuple[float, List[Dict[str, str]]]:
        """Calculate bounded risk penalties."""
        evidence: List[Dict[str, str]] = []

        incomplete_profile_penalty = (1.0 - self._clamp_unit(self._to_float(signals.profile_completeness_score, 0.0))) * BEHAVIOR_INCOMPLETE_PROFILE_PENALTY_WEIGHT
        if incomplete_profile_penalty > 0:
            evidence.append(self._evidence("missing", "Incomplete profile risk", f"Profile completeness is incomplete (penalty {incomplete_profile_penalty:.2f})"))

        inactivity_score, _ = self._score_last_active(signals.last_active_date)
        inactive_profile_penalty = (1.0 - inactivity_score) * BEHAVIOR_INACTIVE_PROFILE_PENALTY_WEIGHT
        if inactive_profile_penalty > 0:
            evidence.append(self._evidence("missing", "Inactive profile risk", f"Candidate appears inactive (penalty {inactive_profile_penalty:.2f})"))

        unverified_count = sum([0 if signals.verified_email else 1, 0 if signals.verified_phone else 1, 0 if signals.linkedin_connected else 1])
        unverified_profile_penalty = (unverified_count / 3.0) * BEHAVIOR_UNVERIFIED_PROFILE_PENALTY_WEIGHT
        if unverified_profile_penalty > 0:
            evidence.append(self._evidence("missing", "Unverified profile risk", f"Profile lacks verification signals (penalty {unverified_profile_penalty:.2f})"))

        low_engagement_penalty = (1.0 - recruiter_interest_score) * BEHAVIOR_LOW_ENGAGEMENT_PENALTY_WEIGHT
        if low_engagement_penalty > 0:
            evidence.append(self._evidence("missing", "Low recruiter engagement risk", f"Recruiter engagement is low (penalty {low_engagement_penalty:.2f})"))

        poor_reliability_penalty = (1.0 - reliability_score) * BEHAVIOR_POOR_RELIABILITY_PENALTY_WEIGHT
        if poor_reliability_penalty > 0:
            evidence.append(self._evidence("missing", "Poor hiring reliability risk", f"Hiring reliability is weak (penalty {poor_reliability_penalty:.2f})"))

        total_penalty = min(
            incomplete_profile_penalty
            + inactive_profile_penalty
            + unverified_profile_penalty
            + low_engagement_penalty
            + poor_reliability_penalty,
            BEHAVIOR_MAX_TOTAL_PENALTY,
        )

        return total_penalty, evidence

    def _calculate_confidence(
        self,
        available_signal_count: int,
        component_scores: List[float],
        evidence_count: int,
    ) -> float:
        """Calculate confidence from signal coverage and signal quality."""
        total = max(BEHAVIOR_TOTAL_SIGNALS_EXPECTED, 1)
        available_ratio = min(max(available_signal_count / float(total), 0.0), 1.0)
        missing_ratio = 1.0 - available_ratio
        quality_score = sum(component_scores) / len(component_scores) if component_scores else 0.0
        evidence_ratio = min(evidence_count / float(total), 1.0)

        confidence = (
            (available_ratio * 0.40)
            + (quality_score * 0.40)
            + (evidence_ratio * 0.20)
            - (missing_ratio * 0.10)
        )

        return max(0.0, min(confidence, 1.0))

    def _score_last_active(self, last_active: Optional[date]) -> Tuple[float, int]:
        """Score recency using an exponential decay curve."""
        if last_active is None:
            return 0.5, 0

        days_since = max((date.today() - last_active).days, 0)
        score = math.exp(-days_since / float(BEHAVIOR_ACTIVITY_RECENCY_HALF_LIFE_DAYS))
        return self._clamp_unit(score), 1

    def _normalize_log_count(self, value: Optional[int]) -> float:
        """Normalize count-based values logarithmically into [0, 1]."""
        safe_value = max(int(value or 0), 0)
        if safe_value <= 0:
            return 0.0
        return self._clamp_unit(math.log1p(safe_value) / math.log1p(BEHAVIOR_LOG_COUNT_NORMALIZATION_CAP))

    def _normalize_notice_period(self, notice_period_days: Optional[int]) -> float:
        """Normalize notice period with a mild penalty for longer waits."""
        safe_value = max(int(notice_period_days or 0), 0)
        capped_value = min(safe_value, BEHAVIOR_NOTICE_PERIOD_MAX_DAYS)
        normalized = self._normalize_log_count(capped_value)
        score = 1.0 - (normalized * BEHAVIOR_NOTICE_PERIOD_RELIEF_FACTOR)
        return max(BEHAVIOR_MIN_AVAILABILITY_SCORE, self._clamp_unit(score))

    def _score_work_mode(self, preferred_work_mode: Optional[str]) -> float:
        """Score preferred work mode using a deterministic lookup."""
        if not preferred_work_mode:
            return 0.85
        normalized_mode = preferred_work_mode.strip().lower()
        return self._clamp_unit(BEHAVIOR_WORK_MODE_SCORES.get(normalized_mode, 0.85))

    def _collect_evidence(self, *evidence_groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Flatten and deduplicate evidence items while preserving order."""
        collected: List[Dict[str, str]] = []
        seen = set()

        for group in evidence_groups:
            if not group:
                continue
            for item in group:
                key = (item.get("type"), item.get("item"))
                if key in seen:
                    continue
                seen.add(key)
                collected.append(item)

        return collected

    def _extract_unique_items(self, evidence: List[Dict[str, str]], item_type: str) -> List[str]:
        """Extract unique evidence item labels by type."""
        items: List[str] = []
        seen = set()
        for item in evidence:
            if item.get("type") != item_type:
                continue
            label = item.get("item")
            if label and label not in seen:
                seen.add(label)
                items.append(label)
        return items

    def _evidence(self, item_type: str, item: str, reason: str) -> Dict[str, str]:
        """Create a normalized evidence record."""
        return {"type": item_type, "item": item, "reason": reason}

    def _clamp_unit(self, value: float) -> float:
        """Clamp a numeric value to the [0, 1] interval."""
        return max(0.0, min(float(value), 1.0))

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        """Safely coerce a value to float."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _is_missing_optional(self, value: Any) -> bool:
        """Check whether an optional signal is missing or sentinel encoded."""
        return value is None or value == -1
