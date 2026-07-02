"""Hybrid ranker for final candidate ordering."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.utils.logging import stage_log

from src.config import (
    BEHAVIOR_WEIGHT,
    CAREER_WEIGHT,
    CONSISTENCY_WEIGHT,
    EDUCATION_WEIGHT,
    MAX_BEHAVIOR_SCORE,
    MAX_CAREER_SCORE,
    MAX_CONSISTENCY_SCORE,
    MAX_SEMANTIC_SCORE,
    MAX_SKILL_SCORE,
    MIN_BEHAVIOR_SCORE,
    MIN_CAREER_SCORE,
    MIN_CONSISTENCY_SCORE,
    MIN_SEMANTIC_SCORE,
    MIN_SKILL_SCORE,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
)
from src.models.candidate import Candidate
from src.models.hybrid_score_result import HybridScoreResult
from src.models.parsed_job import ParsedJob
from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from src.retrieval.retriever import Retriever
from .behavior_scorer import BehaviorScorer
from .career_scorer import CareerScorer
from .consistency_scorer import ConsistencyScorer
from .skill_scorer import SkillScorer

logger = logging.getLogger(__name__)


class HybridRanker:
    """Production ranker that combines retrieval similarity and scorer outputs."""

    def __init__(
        self,
        retriever: Retriever,
        candidate_lookup: Optional[Dict[str, Candidate]] = None,
        candidate_resolver: Optional[Callable[[str], Optional[Candidate]]] = None,
        career_scorer: Optional[CareerScorer] = None,
        skill_scorer: Optional[SkillScorer] = None,
        behavior_scorer: Optional[BehaviorScorer] = None,
        education_scorer: Optional[Any] = None,
        consistency_scorer: Optional[Any] = None,
    ):
        self.retriever = retriever
        self.candidate_lookup = candidate_lookup or {}
        self.candidate_resolver = candidate_resolver
        self.career_scorer = career_scorer or CareerScorer()
        self.skill_scorer = skill_scorer or SkillScorer()
        self.behavior_scorer = behavior_scorer or BehaviorScorer()
        self.education_scorer = education_scorer
        self.consistency_scorer = consistency_scorer or ConsistencyScorer()
        self.last_retrieved_candidates: List[Dict[str, Any]] = []

    def rank(self, context: ScoringContext) -> List[Dict[str, Any]]:
        """Rank candidates based on a parsed job and retriever search."""
        parsed_job = self._get_parsed_job(context)
        top_k = int(context.get_config("top_k", 10))
        ranked_results = self.rank_candidates(parsed_job, top_k=top_k)
        return [result.to_dict() for result in ranked_results]

    def rank_candidates(self, parsed_job: ParsedJob, top_k: int = 10) -> List[HybridScoreResult]:
        """Search for top candidates and score only the retrieved set."""
        if top_k <= 0:
            return []

        query_text = self._get_query_text(parsed_job)
        logger.info(
            "[HybridRanker] START rank_candidates -- job='%s', top_k=%d",
            parsed_job.job_description.job_id,
            top_k,
        )
        with stage_log(logger, "FAISS retrieval", count_label=f"top_k={top_k}"):
            retrieval_results = self.retriever.search(query_text, k=top_k)
        self.last_retrieved_candidates = retrieval_results
        logger.info("[HybridRanker] Retrieved %d candidates from FAISS", len(retrieval_results))

        with stage_log(logger, "Hybrid re-ranking", count_label=f"{len(retrieval_results)} candidates"):
            results = self.rank_retrieval_results(parsed_job, retrieval_results)

        logger.info(
            "[HybridRanker] END rank_candidates -- ranked %d candidates",
            len(results),
        )
        return results

    def rank_retrieval_results(
        self,
        parsed_job: ParsedJob,
        retrieval_results: List[Dict[str, Any]],
    ) -> List[HybridScoreResult]:
        """Score an already-retrieved candidate list without calling FAISS again."""
        logger.info(
            "[HybridRanker] START rank_retrieval_results -- %d candidates to score",
            len(retrieval_results),
        )

        ranked_results: List[HybridScoreResult] = []
        for retrieval_result in retrieval_results:
            candidate = self._resolve_candidate(retrieval_result["candidate_id"])
            if candidate is None:
                logger.warning("Unable to resolve candidate %s", retrieval_result["candidate_id"])
                continue

            candidate_context = ScoringContext(
                candidate=candidate,
                job_description=parsed_job.job_description,
                config={
                    "parsed_job": parsed_job,
                    "retrieval_result": retrieval_result,
                    "semantic_score": retrieval_result.get("similarity", 0.0),
                },
            )
            ranked_results.append(self.calculate_hybrid_score(candidate_context))

        ranked_results.sort(key=self._sort_key)
        logger.info(
            "[HybridRanker] END rank_retrieval_results -- scored and sorted %d candidates",
            len(ranked_results),
        )
        return ranked_results

    def calculate_hybrid_score(self, context: ScoringContext) -> HybridScoreResult:
        """Calculate hybrid score for a single retrieved candidate."""
        candidate = context.candidate
        parsed_job = self._get_parsed_job(context)
        retrieval_result = context.get_config("retrieval_result", {})

        semantic_score = self._clamp(
            float(context.get_config("semantic_score", retrieval_result.get("similarity", 0.0))),
            MIN_SEMANTIC_SCORE,
            MAX_SEMANTIC_SCORE,
        )

        career_result = self._score_with(self.career_scorer, context)
        skill_result = self._score_with(self.skill_scorer, context)
        behavior_result = self._score_with(self.behavior_scorer, context)
        education_result = self._score_with_optional(self.education_scorer, context)
        consistency_result = self._score_with_optional(self.consistency_scorer, context)

        component_scores = {
            "semantic": semantic_score,
            "career": career_result.score,
            "skill": skill_result.score,
            "behavior": behavior_result.score,
            "education": education_result.score,
            "consistency": consistency_result.score,
        }

        raw_weighted_score = self._clamp(
            semantic_score * SEMANTIC_WEIGHT
            + career_result.score * CAREER_WEIGHT
            + skill_result.score * SKILL_WEIGHT
            + behavior_result.score * BEHAVIOR_WEIGHT
            + education_result.score * EDUCATION_WEIGHT
            + consistency_result.score * CONSISTENCY_WEIGHT,
            0.0,
            1.0,
        )
        weighted_final_score, calibration_adjustment = self._calibrate_score(
            raw_weighted_score,
            career_result.score,
            skill_result.score,
        )

        confidence = self._aggregate_confidence(
            [
                (semantic_score, SEMANTIC_WEIGHT),
                (career_result.confidence, CAREER_WEIGHT),
                (skill_result.confidence, SKILL_WEIGHT),
                (behavior_result.confidence, BEHAVIOR_WEIGHT),
                (education_result.confidence, EDUCATION_WEIGHT, self._is_available(education_result)),
                (consistency_result.confidence, CONSISTENCY_WEIGHT, self._is_available(consistency_result)),
            ]
        )

        matched_items = self._dedupe_preserve_order(
            list(career_result.matched_items)
            + list(skill_result.matched_items)
            + list(behavior_result.matched_items)
            + list(education_result.matched_items)
            + list(consistency_result.matched_items)
        )
        missing_items = self._dedupe_preserve_order(
            list(career_result.missing_items)
            + list(skill_result.missing_items)
            + list(behavior_result.missing_items)
            + list(education_result.missing_items)
            + list(consistency_result.missing_items)
        )
        matched_keys = {self._evidence_key(item) for item in matched_items}
        missing_items = [item for item in missing_items if self._evidence_key(item) not in matched_keys]
        reasons = self._dedupe_preserve_order(
            list(career_result.reasons)
            + list(skill_result.reasons)
            + list(behavior_result.reasons)
            + list(education_result.reasons)
            + list(consistency_result.reasons)
            + [f"Semantic similarity score: {semantic_score:.2f}"]
        )

        metadata = {
            "parsed_job_id": parsed_job.job_description.job_id,
            "retrieval_rank": retrieval_result.get("rank"),
            "retrieval_similarity": semantic_score,
            "component_scores": component_scores,
            "raw_weighted_score": raw_weighted_score,
            "calibration_adjustment": calibration_adjustment,
            "component_confidences": {
                "semantic": semantic_score,
                "career": career_result.confidence,
                "skill": skill_result.confidence,
                "behavior": behavior_result.confidence,
                "education": education_result.confidence,
                "consistency": consistency_result.confidence,
            },
            "partial_scores": {
                "semantic": semantic_score,
                "career": career_result.score,
                "skill": skill_result.score,
                "behavior": behavior_result.score,
                "education": education_result.score,
                "consistency": consistency_result.score,
            },
            "scorer_metadata": {
                "career": career_result.metadata,
                "skill": skill_result.metadata,
                "behavior": behavior_result.metadata,
                "education": education_result.metadata,
                "consistency": consistency_result.metadata,
            },
        }

        return HybridScoreResult(
            candidate_id=candidate.candidate_id,
            semantic_score=semantic_score,
            career_score=career_result.score,
            skill_score=skill_result.score,
            behavior_score=behavior_result.score,
            education_score=education_result.score,
            consistency_score=consistency_result.score,
            weighted_final_score=weighted_final_score,
            confidence=confidence,
            metadata=metadata,
            matched_items=matched_items,
            missing_items=missing_items,
            reasons=reasons,
        )

    def _get_parsed_job(self, context: ScoringContext) -> ParsedJob:
        parsed_job = context.get_config("parsed_job")
        if isinstance(parsed_job, ParsedJob):
            return parsed_job
        raise ValueError("HybridRanker requires parsed_job in context config")

    def _get_query_text(self, parsed_job: ParsedJob) -> str:
        search_query = getattr(parsed_job, "search_query", None)
        if search_query and getattr(search_query, "combined_query", None):
            return search_query.combined_query
        return parsed_job.job_description.title

    def _resolve_candidate(self, candidate_id: str) -> Optional[Candidate]:
        if candidate_id in self.candidate_lookup:
            return self.candidate_lookup[candidate_id]
        if self.candidate_resolver is not None:
            candidate = self.candidate_resolver(candidate_id)
            if candidate is not None:
                self.candidate_lookup[candidate_id] = candidate
            return candidate
        return None

    def _score_with(self, scorer: Any, context: ScoringContext) -> ScoreResult:
        return scorer.score(context)

    def _score_with_optional(self, scorer: Any, context: ScoringContext) -> ScoreResult:
        if scorer is None:
            return ScoreResult(
                score=0.0,
                confidence=0.0,
                reasons=[],
                matched_items=[],
                missing_items=[],
                metadata={"available": False},
            )
        return scorer.score(context)

    def _aggregate_confidence(self, weighted_components: Sequence[Sequence[Any]]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0

        for component in weighted_components:
            if len(component) == 2:
                score, weight = component
                available = True
            else:
                score, weight, available = component

            if not available or weight <= 0:
                continue

            total_weight += float(weight)
            weighted_sum += float(score) * float(weight)

        if total_weight <= 0:
            return 0.0

        return self._clamp(weighted_sum / total_weight, 0.0, 1.0)

    def _is_available(self, result: ScoreResult) -> bool:
        return result.get_metadata("available", True) is not False

    def _sort_key(self, result: HybridScoreResult) -> Tuple[float, float, float, float, float, str]:
        return (
            -result.weighted_final_score,
            -result.career_score,
            -result.skill_score,
            -result.consistency_score,
            -result.behavior_score,
            result.candidate_id,
        )

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _calibrate_score(self, raw_score: float, career_score: float, skill_score: float) -> Tuple[float, float]:
        """Reward corroborated recruiter evidence and separate weak technical fits.

        The adjustment depends on the weaker of career and skill evidence, so a
        high semantic score or one excellent component cannot trigger it alone.
        """
        joint_technical_evidence = min(career_score, skill_score)
        if joint_technical_evidence >= 0.60:
            adjustment = min(0.10, 0.02 + ((joint_technical_evidence - 0.60) * 0.45))
        elif joint_technical_evidence < 0.40:
            adjustment = -min(0.08, (0.40 - joint_technical_evidence) * 0.35)
        else:
            adjustment = 0.0
        return self._clamp(raw_score + adjustment, 0.0, 1.0), adjustment

    def _evidence_key(self, item: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", item.lower())

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))
