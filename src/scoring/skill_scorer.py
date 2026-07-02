"""Skill scorer for evaluating candidate skill match."""

import logging
import re
from typing import Dict, List, Tuple

from src.config import (
    SKILL_CONFIDENCE_DURATION_THRESHOLD_MONTHS,
    SKILL_CONFIDENCE_DURATION_WEIGHT,
    SKILL_CONFIDENCE_ENDORSEMENT_THRESHOLD,
    SKILL_CONFIDENCE_ENDORSEMENT_WEIGHT,
    SKILL_CONFIDENCE_EVIDENCE_THRESHOLD,
    SKILL_CONFIDENCE_EVIDENCE_WEIGHT,
    SKILL_CONFIDENCE_MATCH_WEIGHT,
    SKILL_DIVERSITY_CATEGORIES,
    SKILL_DIVERSITY_MAX_CATEGORIES,
    SKILL_REQUIRED_MATCH_WEIGHT,
    SKILL_PREFERRED_MATCH_WEIGHT,
    SKILL_PROFICIENCY_WEIGHT,
    SKILL_DURATION_MAX_MONTHS,
    SKILL_DURATION_WEIGHT,
    SKILL_ENDORSEMENT_MAX_COUNT,
    SKILL_ENDORSEMENT_WEIGHT,
    SKILL_TECHNOLOGY_BREADTH_WEIGHT,
    SKILL_TECHNOLOGY_MATCH_WEIGHT,
    SKILL_TECHNOLOGY_COVERAGE_WEIGHT,
    SKILL_DIVERSITY_WEIGHT,
    SKILL_PROFICIENCY_MAPPING,
    SKILL_SYNONYMS,
)
from src.scoring.base_scorer import BaseScorer
from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from src.models.skill import Skill

logger = logging.getLogger(__name__)


class SkillScorer(BaseScorer):
    """Production-quality skill scorer for evaluating candidate skill match.

    Evaluates how well a candidate's skills match job requirements using:
    - Required skill matching (exact normalized matching)
    - Preferred skill matching (lower weight)
    - Skill proficiency (beginner to expert)
    - Skill duration (long-term usage)
    - Endorsements (normalized counts)
    - Technology coverage (multiple technologies)
    - Skill diversity (multiple categories)

    All matching is deterministic with synonym support.
    """

    def __init__(self):
        """Initialize the skill scorer."""
        self._build_reverse_synonym_map()
        self._build_diversity_lookup()

    def _build_reverse_synonym_map(self) -> None:
        """Build reverse synonym map for normalization."""
        self._synonym_map: Dict[str, str] = {}
        self._display_map: Dict[str, str] = {}

        for canonical, synonyms in SKILL_SYNONYMS.items():
            canonical_normalized = self._normalize_skill(canonical)
            self._synonym_map[canonical_normalized] = canonical_normalized
            self._display_map[canonical_normalized] = canonical

            for synonym in synonyms:
                synonym_normalized = self._normalize_skill(synonym)
                self._synonym_map[synonym_normalized] = canonical_normalized

    def _build_diversity_lookup(self) -> None:
        """Build normalized keyword lookup for diversity scoring."""
        self._diversity_lookup: Dict[str, str] = {}
        for category, keywords in SKILL_DIVERSITY_CATEGORIES.items():
            for keyword in keywords:
                normalized_keyword = self._normalize_skill(keyword)
                self._diversity_lookup[normalized_keyword] = category

    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill name for matching.

        Args:
            skill: Raw skill name.

        Returns:
            Normalized skill name (lowercase, no spaces/symbols).
        """
        # Convert to lowercase
        normalized = skill.lower()

        # Remove spaces, hyphens, underscores, dots, and slashes
        normalized = re.sub(r'[\s\-_\./]', '', normalized)

        return normalized

    def _get_canonical_skill(self, skill: str) -> str:
        """Get canonical skill name using synonym map.

        Args:
            skill: Raw skill name.

        Returns:
            Canonical skill name.
        """
        normalized = self._normalize_skill(skill)
        return self._synonym_map.get(normalized, normalized)

    def _get_skill_display(self, skill_key: str) -> str:
        """Return a display label for a normalized skill key."""
        return self._display_map.get(skill_key, skill_key)

    def _normalize_skill_list(self, skills: List[str]) -> List[str]:
        """Normalize a list of skill strings to canonical keys."""
        return [self._get_canonical_skill(skill) for skill in skills if skill]

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        """Remove duplicates while preserving original order."""
        seen = set()
        deduped = []
        for item in items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _skill_strength_tuple(self, skill: Skill) -> Tuple[float, float, float]:
        """Build a comparable strength tuple for duplicate skill records."""
        proficiency = (skill.proficiency or "intermediate").lower()
        proficiency_score = SKILL_PROFICIENCY_MAPPING.get(proficiency, SKILL_PROFICIENCY_MAPPING["intermediate"])
        duration_score = self._normalize_duration(skill.duration_months)
        endorsement_score = self._normalize_endorsements(skill.endorsements)
        return proficiency_score, duration_score, endorsement_score

    def _is_better_skill(self, incoming: Skill, existing: Skill) -> bool:
        """Return True when an incoming skill record carries stronger evidence."""
        return self._skill_strength_tuple(incoming) > self._skill_strength_tuple(existing)

    def _normalize_duration(self, duration_months: int) -> float:
        """Normalize duration into [0, 1]."""
        safe_duration = max(duration_months or 0, 0)
        return min(safe_duration / float(SKILL_DURATION_MAX_MONTHS), 1.0)

    def _normalize_endorsements(self, endorsements: int) -> float:
        """Normalize endorsement counts into [0, 1]."""
        safe_endorsements = max(endorsements or 0, 0)
        return min(safe_endorsements / float(SKILL_ENDORSEMENT_MAX_COUNT), 1.0)

    def score(self, context: ScoringContext) -> ScoreResult:
        """Calculate skill match score between candidate and job.

        Args:
            context: ScoringContext containing candidate and job description.

        Returns:
            ScoreResult: Skill match score with details.
        """
        if not self.validate_inputs(context):
            return ScoreResult(
                score=0.0,
                confidence=0.0,
                matched_items=[],
                missing_items=[],
                reasons=["Invalid inputs for skill scoring"],
                metadata={
                    "component": "skill",
                    "partial_scores": {},
                    "evidence_count": 0,
                },
            )

        candidate = context.candidate
        job_description = context.job_description

        required_skills = job_description.required_skills or []
        preferred_skills = job_description.preferred_skills or []
        technologies = job_description.technologies or []

        # Get candidate skills
        candidate_skills = candidate.skills or []

        # Normalize all skills
        required_normalized = self._normalize_skill_list(required_skills)
        preferred_normalized = self._normalize_skill_list(preferred_skills)
        technology_normalized = self._normalize_skill_list(technologies)
        candidate_skill_map = self._build_candidate_skill_map(candidate_skills)

        # Calculate component scores
        required_score, required_matched, required_missing = self._score_required_match(
            required_normalized, candidate_skill_map
        )

        preferred_score, preferred_matched, preferred_missing = self._score_preferred_match(
            preferred_normalized, candidate_skill_map
        )

        proficiency_score = self._score_proficiency(candidate_skill_map, required_normalized)
        duration_score = self._score_duration(candidate_skill_map, required_normalized)
        endorsement_score = self._score_endorsements(candidate_skill_map, required_normalized)
        tech_coverage_score = self._score_technology_coverage(technology_normalized, candidate_skill_map)
        diversity_score = self._score_diversity(candidate_skill_map)

        # Calculate weighted final score
        final_score = (
            required_score * SKILL_REQUIRED_MATCH_WEIGHT
            + preferred_score * SKILL_PREFERRED_MATCH_WEIGHT
            + proficiency_score * SKILL_PROFICIENCY_WEIGHT
            + duration_score * SKILL_DURATION_WEIGHT
            + endorsement_score * SKILL_ENDORSEMENT_WEIGHT
            + tech_coverage_score * SKILL_TECHNOLOGY_COVERAGE_WEIGHT
            + diversity_score * SKILL_DIVERSITY_WEIGHT
        )

        # Clamp score to [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        # Calculate confidence
        confidence = self._calculate_confidence(
            required_matched,
            required_normalized,
            candidate_skill_map,
        )

        # Build reasons
        reasons = self._build_reasons(
            required_matched,
            required_missing,
            preferred_matched,
            candidate_skill_map,
            technology_normalized,
            tech_coverage_score,
            diversity_score,
        )

        # Build matched and missing items
        matched_items = [self._get_skill_display(skill) for skill in self._dedupe_preserve_order(required_matched + preferred_matched)]
        missing_items = [self._get_skill_display(skill) for skill in self._dedupe_preserve_order(required_missing)]

        # Build partial scores (store in metadata)
        partial_scores = {
            "required_match": required_score,
            "preferred_match": preferred_score,
            "proficiency": proficiency_score,
            "duration": duration_score,
            "endorsement": endorsement_score,
            "technology_coverage": tech_coverage_score,
            "diversity": diversity_score,
        }

        return ScoreResult(
            score=final_score,
            confidence=confidence,
            matched_items=matched_items,
            missing_items=missing_items,
            reasons=reasons,
            metadata={
                "component": "skill",
                "partial_scores": partial_scores,
                "evidence_count": len(reasons),
            },
        )

    def _build_candidate_skill_map(self, skills: List[Skill]) -> Dict[str, Skill]:
        """Build normalized skill map from candidate skills.

        Args:
            skills: List of Skill objects.

        Returns:
            Dictionary mapping normalized skill names to Skill objects.
        """
        skill_map: Dict[str, Skill] = {}
        for skill in skills:
            if skill.name:
                canonical = self._get_canonical_skill(skill.name)
                existing = skill_map.get(canonical)
                if existing is None or self._is_better_skill(skill, existing):
                    skill_map[canonical] = skill
        return skill_map

    def _score_required_match(
        self,
        required_skills: List[str],
        candidate_skill_map: Dict[str, Skill],
    ) -> Tuple[float, List[str], List[str]]:
        """Score required skill matching.

        Args:
            required_skills: List of normalized required skills.
            candidate_skill_map: Map of candidate skills.

        Returns:
            Tuple of (score, matched_skills, missing_skills).
        """
        if not required_skills:
            return 1.0, [], []

        matched = []
        missing = []

        for skill in required_skills:
            if skill in candidate_skill_map:
                matched.append(skill)
            else:
                missing.append(skill)

        score = len(matched) / len(required_skills) if required_skills else 1.0
        return score, matched, missing

    def _score_preferred_match(
        self,
        preferred_skills: List[str],
        candidate_skill_map: Dict[str, Skill],
    ) -> Tuple[float, List[str], List[str]]:
        """Score preferred skill matching.

        Args:
            preferred_skills: List of normalized preferred skills.
            candidate_skill_map: Map of candidate skills.

        Returns:
            Tuple of (score, matched_skills, missing_skills).
        """
        if not preferred_skills:
            return 1.0, [], []

        matched = []
        missing = []

        for skill in preferred_skills:
            if skill in candidate_skill_map:
                matched.append(skill)
            else:
                missing.append(skill)

        score = len(matched) / len(preferred_skills) if preferred_skills else 1.0
        return score, matched, missing

    def _score_proficiency(
        self,
        candidate_skill_map: Dict[str, Skill],
        required_skills: List[str],
    ) -> float:
        """Score skill proficiency for matched required skills.

        Args:
            candidate_skill_map: Map of candidate skills.
            required_skills: List of required skills.

        Returns:
            Proficiency score in [0, 1].
        """
        if not required_skills:
            return 1.0

        proficiency_scores = []

        for skill in required_skills:
            if skill in candidate_skill_map:
                skill_obj = candidate_skill_map[skill]
                proficiency = skill_obj.proficiency or "intermediate"
                score = SKILL_PROFICIENCY_MAPPING.get(proficiency.lower(), 0.5)
                proficiency_scores.append(score)

        if not proficiency_scores:
            return 0.0

        return sum(proficiency_scores) / len(proficiency_scores)

    def _score_duration(
        self,
        candidate_skill_map: Dict[str, Skill],
        required_skills: List[str],
    ) -> float:
        """Score skill duration for matched required skills.

        Args:
            candidate_skill_map: Map of candidate skills.
            required_skills: List of required skills.

        Returns:
            Duration score in [0, 1].
        """
        if not required_skills:
            return 1.0

        duration_scores = []

        for skill in required_skills:
            if skill in candidate_skill_map:
                skill_obj = candidate_skill_map[skill]
                normalized = self._normalize_duration(skill_obj.duration_months or 0)
                duration_scores.append(normalized)

        if not duration_scores:
            return 0.0

        return sum(duration_scores) / len(duration_scores)

    def _score_endorsements(
        self,
        candidate_skill_map: Dict[str, Skill],
        required_skills: List[str],
    ) -> float:
        """Score skill endorsements for matched required skills.

        Args:
            candidate_skill_map: Map of candidate skills.
            required_skills: List of required skills.

        Returns:
            Endorsement score in [0, 1].
        """
        if not required_skills:
            return 1.0

        endorsement_scores = []

        for skill in required_skills:
            if skill in candidate_skill_map:
                skill_obj = candidate_skill_map[skill]
                normalized = self._normalize_endorsements(skill_obj.endorsements or 0)
                endorsement_scores.append(normalized)

        if not endorsement_scores:
            return 0.0

        return sum(endorsement_scores) / len(endorsement_scores)

    def _score_technology_coverage(
        self,
        technologies: List[str],
        candidate_skill_map: Dict[str, Skill],
    ) -> float:
        """Score technology coverage.

        Args:
            technologies: List of required technologies.
            candidate_skill_map: Map of candidate skills.

        Returns:
            Technology coverage score in [0, 1].
        """
        if not technologies:
            return 1.0

        matched = 0
        for tech in technologies:
            canonical = self._get_canonical_skill(tech)
            if canonical in candidate_skill_map:
                matched += 1

        coverage = matched / len(technologies) if technologies else 1.0

        # Reward broader overlap with a capped bonus based on the number of matched technologies.
        breadth_bonus = min(matched / float(max(len(technologies), 1)), 1.0)

        return min(
            (coverage * SKILL_TECHNOLOGY_MATCH_WEIGHT)
            + (breadth_bonus * SKILL_TECHNOLOGY_BREADTH_WEIGHT),
            1.0,
        )

    def _score_diversity(self, candidate_skill_map: Dict[str, Skill]) -> float:
        """Score skill diversity using deterministic category heuristics.

        Args:
            candidate_skill_map: Map of candidate skills.

        Returns:
            Diversity score in [0, 1].
        """
        if not candidate_skill_map:
            return 0.0

        matched_categories = set()
        for skill_key in candidate_skill_map:
            category = self._diversity_lookup.get(skill_key)
            if category:
                matched_categories.add(category)

        category_count = len(matched_categories)
        if category_count == 0:
            return 0.0

        return min(category_count / float(SKILL_DIVERSITY_MAX_CATEGORIES), 1.0)

    def _calculate_confidence(
        self,
        required_matched: List[str],
        required_skills: List[str],
        candidate_skill_map: Dict[str, Skill],
    ) -> float:
        """Calculate confidence score.

        Confidence increases with:
        - Number of matched required skills
        - Skill duration evidence
        - Endorsement evidence

        Args:
            required_matched: List of matched required skills.
            required_skills: List of all required skills.
            candidate_skill_map: Map of candidate skills.

        Returns:
            Confidence score in [0, 1].
        """
        if not required_skills:
            return 0.5

        match_ratio = len(required_matched) / len(required_skills) if required_skills else 0.0

        duration_scores = []
        endorsement_scores = []
        evidence_hits = 0

        for skill in required_matched:
            skill_obj = candidate_skill_map.get(skill)
            if skill_obj is None:
                continue

            duration_score = self._normalize_duration(skill_obj.duration_months or 0)
            endorsement_score = self._normalize_endorsements(skill_obj.endorsements or 0)
            proficiency_score = SKILL_PROFICIENCY_MAPPING.get(
                (skill_obj.proficiency or "intermediate").lower(),
                SKILL_PROFICIENCY_MAPPING["intermediate"],
            )

            duration_scores.append(duration_score)
            endorsement_scores.append(endorsement_score)

            signal_count = 0
            if (skill_obj.duration_months or 0) >= SKILL_CONFIDENCE_DURATION_THRESHOLD_MONTHS:
                signal_count += 1
            if (skill_obj.endorsements or 0) >= SKILL_CONFIDENCE_ENDORSEMENT_THRESHOLD:
                signal_count += 1
            if proficiency_score >= 0.75:
                signal_count += 1

            if signal_count >= SKILL_CONFIDENCE_EVIDENCE_THRESHOLD:
                evidence_hits += 1

        duration_strength = sum(duration_scores) / len(duration_scores) if duration_scores else 0.0
        endorsement_strength = sum(endorsement_scores) / len(endorsement_scores) if endorsement_scores else 0.0
        evidence_quality = evidence_hits / len(required_matched) if required_matched else 0.0

        confidence = (
            (match_ratio * SKILL_CONFIDENCE_MATCH_WEIGHT)
            + (duration_strength * SKILL_CONFIDENCE_DURATION_WEIGHT)
            + (endorsement_strength * SKILL_CONFIDENCE_ENDORSEMENT_WEIGHT)
            + (evidence_quality * SKILL_CONFIDENCE_EVIDENCE_WEIGHT)
        )

        return max(0.0, min(confidence, 1.0))

    def _build_reasons(
        self,
        required_matched: List[str],
        required_missing: List[str],
        preferred_matched: List[str],
        candidate_skill_map: Dict[str, Skill],
        technologies: List[str],
        tech_coverage_score: float,
        diversity_score: float,
    ) -> List[str]:
        """Build list of scoring reasons.

        Args:
            required_matched: Matched required skills.
            required_missing: Missing required skills.
            preferred_matched: Matched preferred skills.
            candidate_skill_map: Map of candidate skills.

        Returns:
            List of reason strings.
        """
        reasons = []

        # Required skills
        for skill in required_matched[:5]:  # Limit to top 5
            if skill in candidate_skill_map:
                skill_obj = candidate_skill_map[skill]
                proficiency = skill_obj.proficiency or "unknown"
                reasons.append(f"Matched required skill: {self._get_skill_display(skill)} (proficiency: {proficiency})")

        for skill in required_missing[:3]:  # Limit to top 3
            reasons.append(f"Missing required skill: {self._get_skill_display(skill)}")

        # Preferred skills
        for skill in preferred_matched[:3]:  # Limit to top 3
            reasons.append(f"Matched preferred skill: {self._get_skill_display(skill)}")

        # Strong evidence
        for skill in required_matched:
            if skill in candidate_skill_map:
                skill_obj = candidate_skill_map[skill]
                if skill_obj.duration_months and skill_obj.duration_months >= SKILL_DURATION_MAX_MONTHS:
                    reasons.append(
                        f"Strong duration evidence for {self._get_skill_display(skill)} ({skill_obj.duration_months} months)"
                    )
                if skill_obj.endorsements and skill_obj.endorsements >= SKILL_ENDORSEMENT_MAX_COUNT:
                    reasons.append(
                        f"High endorsements for {self._get_skill_display(skill)} ({skill_obj.endorsements})"
                    )

        if technologies:
            tech_count = sum(1 for tech in technologies if self._get_canonical_skill(tech) in candidate_skill_map)
            reasons.append(
                f"Matched {tech_count}/{len(technologies)} target technologies (coverage: {tech_coverage_score:.2f})"
            )

        if diversity_score > 0.0:
            reasons.append(f"Skill diversity across technical domains: {diversity_score:.2f}")

        return reasons
