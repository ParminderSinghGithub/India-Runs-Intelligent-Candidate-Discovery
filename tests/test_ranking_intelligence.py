"""Regression tests for Sprint 13 ranking intelligence."""

import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

from src.models.career import Career
from src.models.job_description import JobDescription
from src.models.scoring_context import ScoringContext
from src.models.skill import Skill
from src.parser.candidate_parser import CandidateParser
from src.parser.job_description_parser import JobDescriptionParser
from src.scoring.career_scorer import CareerScorer
from src.scoring.consistency_scorer import ConsistencyScorer
from src.scoring.skill_scorer import SkillScorer
from src.scoring.hybrid_ranker import HybridRanker


class RankingIntelligenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = CandidateParser().parse_many(Path("sample_candidates.json"))[0]

    def test_ml_role_outranks_backend_role_for_ml_job(self):
        scorer = CareerScorer()
        self.assertGreater(
            scorer._role_similarity("Machine Learning Engineer", "Senior ML Engineer"),
            scorer._role_similarity("Machine Learning Engineer", "Senior Backend Engineer"),
        )

    def test_progression_is_evaluated_oldest_to_newest(self):
        careers = [
            Career("B", "Senior Machine Learning Engineer", date(2022, 1, 1), None, 36, True, "AI/ML", "", "production ML"),
            Career("A", "Associate Machine Learning Engineer", date(2020, 1, 1), date(2021, 12, 1), 24, False, "Technology", "", "ML"),
        ]
        candidate = replace(self.candidate, career_history=careers)
        context = ScoringContext(candidate, JobDescription("J", "Machine Learning Engineer"))
        score, _ = CareerScorer()._score_career_progression(context)
        self.assertGreater(score, 0.80)

    def test_skill_aliases_and_related_database_match(self):
        scorer = SkillScorer()
        self.assertEqual(scorer._get_canonical_skill("Torch"), scorer._get_canonical_skill("PyTorch"))
        self.assertEqual(scorer._get_canonical_skill("Neural Networks"), scorer._get_canonical_skill("Deep Learning"))
        skill_map = scorer._build_candidate_skill_map([Skill("Postgres", "advanced", 4, 24)])
        strength, _ = scorer._match_strength(scorer._get_canonical_skill("SQL"), skill_map, set())
        self.assertGreaterEqual(strength, 0.60)

    def test_parser_preserves_preferred_skills_and_technologies(self):
        parsed = JobDescriptionParser().parse({
            "job_id": "J", "title": "ML Engineer", "preferred_skills": ["Docker"],
            "nice_to_have_skills": ["NLP"], "technologies": ["PyTorch"],
        })
        self.assertEqual(parsed.job_description.preferred_skills, ["Docker", "NLP"])
        self.assertEqual(parsed.job_description.technologies, ["PyTorch"])

    def test_consistency_scorer_is_concrete_and_bounded(self):
        context = ScoringContext(self.candidate, JobDescription("J", "ML Engineer"))
        result = ConsistencyScorer().score(context)
        self.assertTrue(0.5 <= result.score <= 1.0)
        self.assertTrue(result.metadata["available"])

    def test_calibration_requires_joint_career_and_skill_evidence(self):
        ranker = object.__new__(HybridRanker)
        strong, strong_adjustment = ranker._calibrate_score(0.80, 0.88, 0.78)
        weak, weak_adjustment = ranker._calibrate_score(0.50, 0.80, 0.20)
        self.assertGreaterEqual(strong, 0.85)
        self.assertGreater(strong_adjustment, 0.0)
        self.assertLess(weak_adjustment, 0.0)
        self.assertLess(weak, 0.50)


if __name__ == "__main__":
    unittest.main()
