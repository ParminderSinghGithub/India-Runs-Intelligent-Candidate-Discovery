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
        "building systems": ["building systems", "built systems", "system building", "build systems"],
        "designing pipelines": ["designing pipelines", "pipeline design", "data pipelines"],
        "production": ["production", "production environment", "production systems", "deploy", "deployment"],
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
        "staff engineer": 6,
        "architect": 6,
        "distinguished": 7,
        "fellow": 7,
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
        "machine learning platform",
        "recommendation",
        "search",
        "ads",
        "marketplace",
    }

    ROLE_DOMAINS = {
        "machine_learning": (
            "machine learning", "ml engineer", "ai engineer", "applied scientist",
            "data scientist", "deep learning", "computer vision", "nlp",
        ),
        "data": ("data engineer", "analytics engineer", "data platform"),
        "backend": ("backend", "back end", "server side", "api engineer"),
        "software": ("software engineer", "software developer", "full stack", "developer"),
        "platform": ("platform engineer", "infrastructure", "distributed systems", "site reliability"),
        "product": ("product manager", "program manager"),
    }

    DOMAIN_RELATEDNESS = {
        ("machine_learning", "data"): 0.65,
        ("machine_learning", "platform"): 0.50,
        ("machine_learning", "software"): 0.35,
        ("machine_learning", "backend"): 0.28,
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
        relevance_scores = []
        ordered_careers = sorted(
            candidate.career_history,
            key=lambda c: (bool(c.is_current), c.start_date is not None, c.start_date),
            reverse=True,
        )

        for index, career in enumerate(ordered_careers):
            if not career.title:
                continue

            similarity = self._role_similarity(job_title, career.title)
            recency_weight = 0.68 ** index
            relevance_scores.append((similarity, recency_weight))
            if similarity >= 0.45:
                evidence.append({
                    "type": "match",
                    "item": career.title,
                    "reason": f"Role '{career.title}' is relevant to {context.job_description.title} ({similarity:.2f})",
                })

        score = (
            sum(value * weight for value, weight in relevance_scores)
            / sum(weight for _, weight in relevance_scores)
            if relevance_scores else 0.0
        )

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

        target_text = " ".join(
            [context.job_description.description or ""]
            + list(context.job_description.responsibilities or [])
        ).lower()
        target_responsibilities = {
            name for name, keywords in self.RESPONSIBILITY_KEYWORDS.items()
            if any(keyword in target_text for keyword in keywords)
        }
        if not target_responsibilities:
            return 0.5, evidence

        matched_responsibilities = set()
        total_descriptions = 0

        for career in candidate.career_history:
            if not career.description:
                continue

            total_descriptions += 1
            description_lower = career.description.lower()

            for resp_name in target_responsibilities:
                keywords = self.RESPONSIBILITY_KEYWORDS[resp_name]
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
        total_responsibilities = len(target_responsibilities)
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

        # Evaluate transitions chronologically.  The previous implementation
        # sorted newest-first, inadvertently treating normal promotions as drops.
        careers_sorted = sorted(
            candidate.career_history,
            key=lambda c: (c.start_date is not None, c.start_date),
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
                previous_level = current_level

        current_career = next((c for c in candidate.career_history if c.is_current), candidate.career_history[-1])
        current_level = self._get_career_level(current_career.title or "")
        hierarchy_score = min(0.55 + (0.075 * current_level), 1.0) if current_level else 0.55
        transition_score = sum(progression_scores) / len(progression_scores) if progression_scores else 0.65
        score = (transition_score * 0.60) + (hierarchy_score * 0.40)
        if current_level >= 4:
            evidence.append({
                "type": "match",
                "item": current_career.title,
                "reason": f"Current role demonstrates level-{current_level} seniority",
            })

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

        parsed_job = context.get_config("parsed_job")
        targets = []
        if parsed_job is not None:
            targets = list(parsed_job.candidate_filters.required_industries or [])
        target_text = " ".join(targets).lower()
        industry_scores = []

        for career in candidate.career_history:
            if not career.industry:
                continue

            industry_lower = career.industry.lower()
            if any(target.lower() in industry_lower or industry_lower in target.lower() for target in targets):
                relevance = 1.0
            elif any(term in industry_lower for term in ("ai", "machine learning", "ml platform", "recommendation", "search")):
                relevance = 0.95
            elif any(term in industry_lower for term in ("ads", "marketplace", "fintech", "saas", "technology", "software", "data", "analytics")):
                relevance = 0.75
            elif any(term in industry_lower for term in self.RELEVANT_INDUSTRIES):
                relevance = 0.65
            else:
                relevance = 0.30
            industry_scores.append(relevance)
            if relevance >= 0.65:
                evidence.append({
                    "type": "match",
                    "item": career.industry,
                    "reason": f"Relevant industry experience in {career.industry}",
                })

        score = max(industry_scores) if industry_scores else (0.5 if not target_text else 0.0)

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

            similarity = self._role_similarity(job_title, career.title)
            if similarity > 0:
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

    def _role_domain(self, title: str) -> str:
        normalized = self._normalize_title(title.lower())
        for domain, phrases in self.ROLE_DOMAINS.items():
            if any(phrase in normalized for phrase in phrases):
                return domain
        return "other"

    def _role_similarity(self, target_title: str, candidate_title: str) -> float:
        target = self._normalize_title(target_title.lower())
        candidate = self._normalize_title(candidate_title.lower())
        if not target or not candidate:
            return 0.0
        if target == candidate or target in candidate or candidate in target:
            return 1.0

        target_domain = self._role_domain(target)
        candidate_domain = self._role_domain(candidate)
        if target_domain == candidate_domain and target_domain != "other":
            domain_score = 0.88
        else:
            domain_score = self.DOMAIN_RELATEDNESS.get(
                (target_domain, candidate_domain),
                self.DOMAIN_RELATEDNESS.get((candidate_domain, target_domain), 0.12),
            )

        generic = {"engineer", "engineering", "developer", "manager", "lead", "specialist"}
        target_words = set(target.split()) - generic
        candidate_words = set(candidate.split()) - generic
        lexical = len(target_words & candidate_words) / max(len(target_words | candidate_words), 1)
        return min(max(domain_score, lexical), 1.0)

    def _get_career_level(self, title: str) -> int:
        """Get career level from title.

        Args:
            title: Job title.

        Returns:
            Career level (1-6), or 0 if not recognized.
        """
        title_lower = title.lower()
        matches = [
            level_value for level_name, level_value in self.CAREER_LEVELS.items()
            if re.search(rf"\b{re.escape(level_name)}\b", title_lower)
        ]
        return max(matches, default=0)
