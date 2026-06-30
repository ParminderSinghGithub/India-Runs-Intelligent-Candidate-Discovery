"""Career scorer implementation."""

import re
from typing import Dict, List, Tuple

from src.config import (
    CAREER_INDUSTRY_MATCH_WEIGHT,
    CAREER_PROGRESSION_WEIGHT,
    CAREER_RELEVANT_EXPERIENCE_WEIGHT,
    CAREER_RESPONSIBILITIES_WEIGHT,
    CAREER_ROLE_RELEVANCE_WEIGHT,
)
from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from .base_scorer import BaseScorer


class CareerScorer(BaseScorer):
    """Evidence-based scorer for evaluating career match.

    Evaluates career history relevance through multiple dimensions:
    - Role relevance (title matching)
    - Responsibility analysis (evidence of key responsibilities)
    - Career progression (consistent growth)
    - Industry match (relevant industry experience)
    - Relevant experience (weighted by role relevance)
    """

    # Responsibility keywords to search for in descriptions
    RESPONSIBILITY_KEYWORDS = {
        "building systems": ["building systems", "built systems", "system building"],
        "designing pipelines": ["designing pipelines", "pipeline design", "data pipelines"],
        "production": ["production", "production environment", "production systems"],
        "recommendation": ["recommendation", "recommendation system", "recommender"],
        "retrieval": ["retrieval", "information retrieval", "retrieval systems"],
        "ranking": ["ranking", "rank", "learning to rank"],
        "machine learning": ["machine learning", "ml", "machine learning models"],
        "backend engineering": ["backend", "backend engineering", "back-end"],
        "APIs": ["api", "apis", "rest api", "graphql"],
        "distributed systems": ["distributed systems", "distributed", "scalable"],
        "feature engineering": ["feature engineering", "feature", "features"],
        "evaluation": ["evaluation", "evaluate", "metrics", "performance"],
    }

    # Career level hierarchy for progression scoring
    CAREER_LEVELS = {
        "intern": 1,
        "internship": 1,
        "trainee": 1,
        "junior": 2,
        "associate": 2,
        "engineer": 3,
        "developer": 3,
        "software engineer": 3,
        "senior": 4,
        "senior engineer": 4,
        "lead": 5,
        "lead engineer": 5,
        "tech lead": 5,
        "principal": 6,
        "principal engineer": 6,
        "staff": 6,
        "manager": 5,
        "engineering manager": 5,
    }

    # Relevant industries (higher weight)
    RELEVANT_INDUSTRIES = {
        "software",
        "ai",
        "ml",
        "machine learning",
        "artificial intelligence",
        "fintech",
        "financial technology",
        "saas",
        "technology",
        "tech",
        "information technology",
        "it services",
        "data",
        "analytics",
    }

    def score(self, context: ScoringContext) -> ScoreResult:
        """Calculate career match score between candidate and job.

        Args:
            context: ScoringContext containing candidate and job description.

        Returns:
            ScoreResult: Career match score with detailed evidence.
        """
        if not self.validate_inputs(context):
            return ScoreResult(
                score=0.0,
                confidence=0.0,
                reasons=["Invalid inputs for career scoring"],
                matched_items=[],
                missing_items=[],
                metadata={"component": "career", "partial_scores": {}, "evidence_count": 0},
            )

        # Calculate component scores
        role_relevance_score, role_evidence = self._score_role_relevance(context)
        responsibilities_score, resp_evidence = self._score_responsibilities(context)
        progression_score, prog_evidence = self._score_career_progression(context)
        industry_score, ind_evidence = self._score_industry_match(context)
        experience_score, exp_evidence = self._score_relevant_experience(context)

        # Combine scores using weights from config
        total_score = (
            role_relevance_score * CAREER_ROLE_RELEVANCE_WEIGHT
            + responsibilities_score * CAREER_RESPONSIBILITIES_WEIGHT
            + progression_score * CAREER_PROGRESSION_WEIGHT
            + industry_score * CAREER_INDUSTRY_MATCH_WEIGHT
            + experience_score * CAREER_RELEVANT_EXPERIENCE_WEIGHT
        )

        # Collect all evidence
        all_evidence = self._collect_evidence(
            role_evidence, resp_evidence, prog_evidence, ind_evidence, exp_evidence
        )

        # Build reasons
        reasons = []
        matched_items = []
        missing_items = []

        for evidence in all_evidence:
            if evidence["type"] == "match":
                reasons.append(evidence["reason"])
                matched_items.append(evidence["item"])
            elif evidence["type"] == "missing":
                reasons.append(evidence["reason"])
                missing_items.append(evidence["item"])

        # Calculate confidence based on evidence count and data completeness
        evidence_count = len(all_evidence)
        confidence = min(0.5 + (evidence_count * 0.05), 1.0)

        # Build partial scores metadata
        partial_scores = {
            "role_relevance": role_relevance_score,
            "responsibilities": responsibilities_score,
            "progression": progression_score,
            "industry_match": industry_score,
            "relevant_experience": experience_score,
        }

        return ScoreResult(
            score=total_score,
            confidence=confidence,
            reasons=reasons,
            matched_items=matched_items,
            missing_items=missing_items,
            metadata={
                "component": "career",
                "partial_scores": partial_scores,
                "evidence_count": evidence_count,
            },
        )

    def _score_role_relevance(self, context: ScoringContext) -> Tuple[float, List[Dict]]:
        """Score relevance of candidate titles to job title.

        Args:
            context: ScoringContext.

        Returns:
            Tuple of (score, evidence_list).
        """
        candidate = context.candidate
        job_title = context.job_description.title.lower()
        evidence = []

        if not candidate.career_history:
            return 0.0, evidence

        # Normalize job title for comparison
        job_title_normalized = self._normalize_title(job_title)
        job_title_words = set(job_title_normalized.split())

        relevance_scores = []
        matched_titles = []

        for career in candidate.career_history:
            if not career.title:
                continue

            title_normalized = self._normalize_title(career.title.lower())
            title_words = set(title_normalized.split())

            # Calculate overlap
            overlap = len(job_title_words & title_words)
            total_unique = len(job_title_words | title_words)

            if total_unique > 0:
                similarity = overlap / total_unique
                relevance_scores.append(similarity)

                if similarity > 0.3:
                    matched_titles.append(career.title)
                    evidence.append({
                        "type": "match",
                        "item": career.title,
                        "reason": f"Title '{career.title}' has {similarity:.2f} similarity to job title",
                    })

        # Current title gets bonus weight
        if candidate.career_history and candidate.career_history[0].is_current:
            current_title = candidate.career_history[0].title
            if current_title in matched_titles:
                relevance_scores = [s * 1.2 for s in relevance_scores]

        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            # Normalize to 0-1 range
            score = min(avg_relevance * 1.5, 1.0)
        else:
            score = 0.0

        return score, evidence

    def _score_responsibilities(self, context: ScoringContext) -> Tuple[float, List[Dict]]:
        """Score evidence of relevant responsibilities in career descriptions.

        Args:
            context: ScoringContext.

        Returns:
            Tuple of (score, evidence_list).
        """
        candidate = context.candidate
        evidence = []

        if not candidate.career_history:
            return 0.0, evidence

        matched_responsibilities = set()
        total_descriptions = 0

        for career in candidate.career_history:
            if not career.description:
                continue

            total_descriptions += 1
            description_lower = career.description.lower()

            for resp_name, keywords in self.RESPONSIBILITY_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in description_lower:
                        matched_responsibilities.add(resp_name)
                        if resp_name not in [e["item"] for e in evidence]:
                            evidence.append({
                                "type": "match",
                                "item": resp_name,
                                "reason": f"Evidence of {resp_name} in role at {career.company}",
                            })
                        break

        # Score based on proportion of responsibilities found
        total_responsibilities = len(self.RESPONSIBILITY_KEYWORDS)
        if total_responsibilities > 0:
            score = len(matched_responsibilities) / total_responsibilities
        else:
            score = 0.0

        # Boost score if multiple descriptions contain evidence
        if total_descriptions > 1 and len(matched_responsibilities) > 0:
            score = min(score * 1.2, 1.0)

        return score, evidence

    def _score_career_progression(self, context: ScoringContext) -> Tuple[float, List[Dict]]:
        """Score career progression (consistent growth vs erratic transitions).

        Args:
            context: ScoringContext.

        Returns:
            Tuple of (score, evidence_list).
        """
        candidate = context.candidate
        evidence = []

        if not candidate.career_history or len(candidate.career_history) < 2:
            return 0.5, evidence  # Neutral score for insufficient data

        # Sort careers by start date (most recent first)
        careers_sorted = sorted(
            candidate.career_history,
            key=lambda c: c.start_date if c.start_date else None,
            reverse=True,
        )

        progression_scores = []
        previous_level = None

        for career in careers_sorted:
            if not career.title:
                continue

            title_lower = career.title.lower()
            current_level = self._get_career_level(title_lower)

            if current_level:
                if previous_level is not None:
                    # Reward upward progression
                    if current_level > previous_level:
                        progression_scores.append(1.0)
                        evidence.append({
                            "type": "match",
                            "item": career.title,
                            "reason": f"Progression from previous level to {career.title}",
                        })
                    # Penalize significant downward moves
                    elif current_level < previous_level - 1:
                        progression_scores.append(0.3)
                        evidence.append({
                            "type": "missing",
                            "item": career.title,
                            "reason": f"Significant level change to {career.title}",
                        })
                    else:
                        progression_scores.append(0.7)  # Neutral for similar levels
                else:
                    progression_scores.append(0.7)  # First role gets neutral score

                previous_level = current_level

        if progression_scores:
            score = sum(progression_scores) / len(progression_scores)
        else:
            score = 0.5  # Neutral if no levels detected

        return score, evidence

    def _score_industry_match(self, context: ScoringContext) -> Tuple[float, List[Dict]]:
        """Score industry relevance to target role.

        Args:
            context: ScoringContext.

        Returns:
            Tuple of (score, evidence_list).
        """
        candidate = context.candidate
        evidence = []

        if not candidate.career_history:
            return 0.0, evidence

        matched_industries = set()
        total_careers = 0

        for career in candidate.career_history:
            if not career.industry:
                continue

            total_careers += 1
            industry_lower = career.industry.lower()

            for relevant_ind in self.RELEVANT_INDUSTRIES:
                if relevant_ind in industry_lower:
                    matched_industries.add(career.industry)
                    evidence.append({
                        "type": "match",
                        "item": career.industry,
                        "reason": f"Relevant industry experience in {career.industry}",
                    })
                    break

        if total_careers > 0:
            score = len(matched_industries) / total_careers
        else:
            score = 0.0

        return score, evidence

    def _score_relevant_experience(self, context: ScoringContext) -> Tuple[float, List[Dict]]:
        """Score relevant experience (weighted by role relevance).

        Args:
            context: ScoringContext.

        Returns:
            Tuple of (score, evidence_list).
        """
        candidate = context.candidate
        evidence = []

        if not candidate.career_history:
            return 0.0, evidence

        job_title = context.job_description.title.lower()
        job_title_normalized = self._normalize_title(job_title)
        job_title_words = set(job_title_normalized.split())

        total_relevant_months = 0
        total_months = 0

        for career in candidate.career_history:
            if not career.title or not career.duration_months:
                continue

            total_months += career.duration_months

            # Calculate relevance of this role
            title_normalized = self._normalize_title(career.title.lower())
            title_words = set(title_normalized.split())
            overlap = len(job_title_words & title_words)
            total_unique = len(job_title_words | title_words)

            if total_unique > 0:
                similarity = overlap / total_unique
                # Weight duration by relevance
                if similarity > 0.3:
                    weighted_duration = career.duration_months * similarity
                    total_relevant_months += weighted_duration
                    evidence.append({
                        "type": "match",
                        "item": career.title,
                        "reason": f"{career.duration_months} months relevant experience as {career.title}",
                    })
                elif similarity > 0.1:
                    # Partial relevance
                    weighted_duration = career.duration_months * (similarity * 0.5)
                    total_relevant_months += weighted_duration

        # Convert to years and normalize
        relevant_years = total_relevant_months / 12.0
        total_years = total_months / 12.0 if total_months > 0 else 1.0

        # Score based on proportion of relevant experience
        if total_years > 0:
            score = min(relevant_years / total_years, 1.0)
        else:
            score = 0.0

        return score, evidence

    def _collect_evidence(self, *evidence_lists: List[Dict]) -> List[Dict]:
        """Collect and deduplicate evidence from multiple sources.

        Args:
            *evidence_lists: Variable number of evidence lists.

        Returns:
            Combined and deduplicated evidence list.
        """
        all_evidence = []
        seen = set()

        for evidence_list in evidence_lists:
            for evidence in evidence_list:
                key = (evidence["type"], evidence["item"])
                if key not in seen:
                    seen.add(key)
                    all_evidence.append(evidence)

        return all_evidence

    def _normalize_title(self, title: str) -> str:
        """Normalize job title for comparison.

        Args:
            title: Raw title string.

        Returns:
            Normalized title.
        """
        # Remove common suffixes and prefixes
        title = re.sub(r"\b(senior(s)?|lead|principal|staff|jr\.|sr\.|i|ii|iii)\b", "", title)
        # Remove special characters
        title = re.sub(r"[^\w\s]", " ", title)
        # Normalize whitespace
        title = " ".join(title.split())
        return title

    def _get_career_level(self, title: str) -> int:
        """Get career level from title.

        Args:
            title: Job title.

        Returns:
            Career level (1-6), or 0 if not recognized.
        """
        title_lower = title.lower()
        for level_name, level_value in self.CAREER_LEVELS.items():
            if level_name in title_lower:
                return level_value
        return 0
