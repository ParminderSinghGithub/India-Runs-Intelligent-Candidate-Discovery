"""Main entry point for the India Runs Data & AI Challenge pipeline."""

import argparse
import sys
from pathlib import Path

from src.config import (
    FAISS_DIR,
    PROJECT_ROOT,
    SRC_DIR,
    TESTS_DIR,
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
)
from src.utils import setup_logging
from src.parser.candidate_parser import CandidateParser
from src.parser.job_description_parser import JobDescriptionParser
from src.models.score_result import ScoreResult
from src.models.scoring_context import ScoringContext
from src.scoring.hybrid_ranker import HybridRanker
from src.scoring.career_scorer import CareerScorer
from src.scoring.skill_scorer import SkillScorer
from src.scoring.behavior_scorer import BehaviorScorer
from src.retrieval.retriever import Retriever
from src.pipeline.offline_pipeline import OfflineIndexBuilder


def print_banner() -> None:
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        Redrob AI INDIA.RUNS Data & AI Challenge          ║
    ║                    Candidate Ranking System               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def verify_project_structure() -> bool:
    """Verify that required directories exist.

    Returns:
        bool: True if all directories exist, False otherwise.
    """
    required_dirs = [
        SRC_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR,
        OUTPUTS_DIR,
    ]

    all_exist = True
    for directory in required_dirs:
        if not directory.exists():
            print(f"Missing directory: {directory}")
            all_exist = False
        else:
            print(f"✓ Directory exists: {directory}")

    return all_exist


def _offline_artifacts_exist() -> bool:
    """Check whether the production FAISS artifacts are already present."""
    required_artifacts = [
        FAISS_DIR / "faiss.index",
        FAISS_DIR / "candidate_lookup.pkl",
        FAISS_DIR / "embedding_metadata.pkl",
    ]
    return all(path.exists() for path in required_artifacts)


def _build_candidate_resolver(candidates_jsonl_path: Path):
    """Build a lazy resolver that loads candidate objects from the source JSONL once."""
    parser = CandidateParser()
    cache = {}
    loaded = False

    def resolve(candidate_id: str):
        nonlocal loaded
        if not loaded:
            with candidates_jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    candidate = parser.from_jsonl_line(line)
                    cache[candidate.candidate_id] = candidate
            loaded = True
        return cache.get(candidate_id)

    return resolve


def main(force_rebuild: bool = False) -> int:
    """Main entry point for the pipeline.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    print_banner()

    # Initialize logging
    setup_logging()
    print("✓ Logging initialized")

    # Verify project structure
    print("\nVerifying project structure...")
    if not verify_project_structure():
        print("\n✗ Project structure verification failed")
        return 1

    print("\n✓ Project structure verified")

    # Test parser with sample data
    print("\nTesting CandidateParser...")
    sample_file = PROJECT_ROOT / "sample_candidates.json"

    if not sample_file.exists():
        print(f"⚠ Sample file not found: {sample_file}")
        print("Pipeline ready. Parser module implemented.")
        return 0

    try:
        parser = CandidateParser()
        candidates = parser.parse_many(sample_file)

        if candidates:
            candidate = candidates[0]
            print(f"\n✓ Successfully parsed {len(candidates)} candidate(s)")
            print(f"\nSample Candidate Summary:")
            print(f"  ID: {candidate.candidate_id}")
            print(f"  Name: {candidate.profile.anonymized_name}")
            print(f"  Title: {candidate.profile.current_title}")
            print(f"  Company: {candidate.profile.current_company}")
            print(f"  Experience: {candidate.profile.years_of_experience} years")
            print(f"  Jobs: {len(candidate.career_history)}")
            print(f"  Skills: {len(candidate.skills)}")
            print(f"  Education: {len(candidate.education)}")
            print(f"  Languages: {len(candidate.languages)}")
            print(f"  Certifications: {len(candidate.certifications)}")
            print(f"\n✓ Parser module working correctly")
        else:
            print("⚠ No candidates parsed from sample file")

    except Exception as e:
        print(f"✗ Parser test failed: {e}")
        return 1

    # Test scoring infrastructure
    print("\nTesting scoring infrastructure...")
    try:
        # Create a mock job description for testing
        from src.models.job_description import JobDescription
        job_desc = JobDescription(
            job_id="JOB_001",
            title="Software Engineer",
            company="Tech Corp",
            required_skills=["Python", "SQL"],
        )

        # Create ScoringContext
        context = ScoringContext(
            candidate=candidate,
            job_description=job_desc,
            config={"test": True},
        )
        print("✓ ScoringContext instantiated")

        # Create ScoreResult
        score_result = ScoreResult(
            score=0.85,
            confidence=0.9,
            reasons=["Good match", "Required skills present"],
            matched_items=["Python", "SQL"],
            missing_items=["Machine Learning"],
            metadata={"scorer": "test"},
        )
        print(f"✓ ScoreResult instantiated (score: {score_result.score})")

        # Verify ScoreResult helper methods
        print(f"  - High confidence: {score_result.is_high_confidence()}")
        print(f"  - Has match 'Python': {score_result.has_match('Python')}")
        print(f"  - Has missing 'ML': {score_result.has_missing('Machine Learning')}")

        # Note: HybridRanker is abstract, so we can't instantiate it directly
        print("✓ HybridRanker interface imported (abstract class)")
        print("\n✓ Scoring infrastructure working correctly")

    except Exception as e:
        print(f"✗ Scoring infrastructure test failed: {e}")
        return 1

    # Test CareerScorer
    print("\nTesting CareerScorer...")
    try:
        from src.models.job_description import JobDescription
        job_desc = JobDescription(
            job_id="JOB_001",
            title="Backend Engineer",
            company="Tech Corp",
            required_skills=["Python", "SQL"],
            responsibilities=["Build backend systems", "Design data pipelines"],
        )

        # Create ScoringContext for career scoring
        context = ScoringContext(
            candidate=candidate,
            job_description=job_desc,
            config={
                "career_role_relevance_weight": 0.30,
                "career_responsibilities_weight": 0.25,
                "career_progression_weight": 0.15,
                "career_industry_match_weight": 0.15,
                "career_relevant_experience_weight": 0.15,
            },
        )

        # Instantiate and run CareerScorer
        career_scorer = CareerScorer()
        career_result = career_scorer.score(context)

        print(f"\nCareer Score: {career_result.score:.2f}")
        print(f"Confidence: {career_result.confidence:.2f}")
        print(f"\nReasons ({len(career_result.reasons)}):")
        for reason in career_result.reasons[:5]:  # Show first 5 reasons
            print(f"  - {reason}")
        if len(career_result.reasons) > 5:
            print(f"  ... and {len(career_result.reasons) - 5} more")

        print(f"\nMatched Evidence ({len(career_result.matched_items)}):")
        for item in career_result.matched_items[:5]:
            print(f"  - {item}")
        if len(career_result.matched_items) > 5:
            print(f"  ... and {len(career_result.matched_items) - 5} more")

        print(f"\nPartial Scores:")
        partial_scores = career_result.get_metadata("partial_scores", {})
        for component, score in partial_scores.items():
            print(f"  - {component}: {score:.2f}")

        print(f"\nEvidence Count: {career_result.get_metadata('evidence_count', 0)}")
        print("\n✓ CareerScorer working correctly")

    except Exception as e:
        print(f"✗ CareerScorer test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test SkillScorer
    print("\nTesting SkillScorer...")
    try:
        # Use the sample candidate and parsed job from CareerScorer test
        candidates_file = PROJECT_ROOT / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "sample_candidates.json"

        if not candidates_file.exists():
            print(f"⚠ Candidates file not found: {candidates_file}")
        else:
            import json

            # Load candidates from JSON array
            with open(candidates_file, "r", encoding="utf-8") as f:
                candidates_data = json.load(f)

            parser = CandidateParser()
            candidates = [parser.from_dict(cand_data) for cand_data in candidates_data]

            # Use first candidate
            candidate = candidates[0]

            # Parse job description
            job_file = PROJECT_ROOT / "job_description.json"

            if not job_file.exists():
                print(f"⚠ Job description file not found: {job_file}")
            else:
                job_parser = JobDescriptionParser()
                parsed_job = job_parser.parse_from_file(job_file)

                # Create scoring context
                context = ScoringContext(
                    candidate=candidate,
                    job_description=parsed_job.job_description,
                    config={"parsed_job": parsed_job},
                )

                # Score with SkillScorer
                skill_scorer = SkillScorer()
                skill_result = skill_scorer.score(context)

                print(f"\nSkill Score: {skill_result.score:.2f}")
                print(f"Confidence: {skill_result.confidence:.2f}")

                print(f"\nMatched Skills ({len(skill_result.matched_items)}):")
                for skill in skill_result.matched_items[:10]:
                    print(f"  - {skill}")

                print(f"\nMissing Skills ({len(skill_result.missing_items)}):")
                for skill in skill_result.missing_items[:10]:
                    print(f"  - {skill}")

                print(f"\nReasons ({len(skill_result.reasons)}):")
                for reason in skill_result.reasons[:10]:
                    print(f"  - {reason}")

                print(f"\nPartial Scores:")
                partial_scores = skill_result.get_metadata("partial_scores", {})
                for name, value in partial_scores.items():
                    print(f"  - {name}: {value:.2f}")

                print(f"\nEvidence Count: {skill_result.get_metadata('evidence_count', 0)}")

                print("\n✓ SkillScorer working correctly")

    except Exception as e:
        print(f"✗ SkillScorer test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test BehaviorScorer
    print("\nTesting BehaviorScorer...")
    try:
        job_file = PROJECT_ROOT / "job_description.json"

        if not job_file.exists():
            print(f"⚠ Job description file not found: {job_file}")
        else:
            job_parser = JobDescriptionParser()
            parsed_job = job_parser.parse_from_file(job_file)

            context = ScoringContext(
                candidate=candidate,
                job_description=parsed_job.job_description,
                config={"parsed_job": parsed_job},
            )

            behavior_scorer = BehaviorScorer()
            behavior_result = behavior_scorer.score(context)

            print(f"\nBehavior Score: {behavior_result.score:.2f}")
            print(f"Confidence: {behavior_result.confidence:.2f}")

            print(f"\nMatched Signals ({len(behavior_result.matched_items)}):")
            for signal in behavior_result.matched_items[:10]:
                print(f"  - {signal}")
            if len(behavior_result.matched_items) > 10:
                print(f"  ... and {len(behavior_result.matched_items) - 10} more")

            print(f"\nMissing Signals ({len(behavior_result.missing_items)}):")
            for signal in behavior_result.missing_items[:10]:
                print(f"  - {signal}")
            if len(behavior_result.missing_items) > 10:
                print(f"  ... and {len(behavior_result.missing_items) - 10} more")

            print(f"\nReasons ({len(behavior_result.reasons)}):")
            for reason in behavior_result.reasons[:10]:
                print(f"  - {reason}")
            if len(behavior_result.reasons) > 10:
                print(f"  ... and {len(behavior_result.reasons) - 10} more")

            print(f"\nPartial Scores:")
            partial_scores = behavior_result.get_metadata("partial_scores", {})
            for name, value in partial_scores.items():
                print(f"  - {name}: {value:.2f}")

            print(f"\nEvidence Count: {behavior_result.get_metadata('evidence_count', 0)}")

            print("\n✓ BehaviorScorer working correctly")

    except Exception as e:
        print(f"✗ BehaviorScorer test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test JobDescriptionParser
    print("\nTesting JobDescriptionParser...")
    try:
        job_file = PROJECT_ROOT / "job_description.json"

        if not job_file.exists():
            print(f"⚠ Job description file not found: {job_file}")
        else:
            jd_parser = JobDescriptionParser()
            parsed_job = jd_parser.parse_from_file(job_file)

            jd = parsed_job.job_description
            sq = parsed_job.search_query
            cf = parsed_job.candidate_filters

            print(f"\nRole: {jd.title}")
            print(f"Company: {jd.company}")
            print(f"Location: {jd.location}")
            print(f"Experience Required: {jd.required_experience_years} years")
            print(f"\nRequired Skills ({len(jd.required_skills)}):")
            for skill in jd.required_skills:
                print(f"  - {skill}")
            print(f"\nResponsibilities ({len(jd.responsibilities)}):")
            for resp in jd.responsibilities:
                print(f"  - {resp}")

            print(f"\nSearch Queries:")
            print(f"  Identity: {sq.identity_query[:80]}...")
            print(f"  Career: {sq.career_query[:80]}...")
            print(f"  Skills: {sq.skills_query[:80]}...")
            print(f"  Combined: {sq.combined_query[:80]}...")

            print(f"\nCandidate Filters:")
            print(f"  Min Experience: {cf.minimum_experience_years}")
            print(f"  Max Experience: {cf.maximum_experience_years}")
            print(f"  Location: {cf.required_location}")
            print(f"  Industries: {cf.required_industries}")
            print(f"  Work Mode: {cf.required_work_mode}")
            print(f"  Has Filters: {cf.has_filters()}")

            print("\n✓ JobDescriptionParser working correctly")

    except Exception as e:
        print(f"✗ JobDescriptionParser test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test retrieval pipeline using existing artifacts by default.
    print("\nTesting retrieval pipeline...")
    try:
        candidates_jsonl = PROJECT_ROOT / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
        artifacts_exist = _offline_artifacts_exist()

        if artifacts_exist and not force_rebuild:
            print("✓ Existing FAISS artifacts detected. Skipping offline rebuild.")
            retriever = Retriever(force_rebuild=False)
            retriever.load_index()
            print(f"Loaded index with {retriever.get_index_size()} candidates")
        else:
            if not candidates_jsonl.exists():
                print(f"⚠ Candidates JSONL file not found: {candidates_jsonl}")
                return 1

            if force_rebuild:
                print("Rebuilding offline artifacts because --rebuild-index was requested.")
            else:
                print("Offline artifacts are missing. Rebuilding the index to restore pipeline state.")

            builder = OfflineIndexBuilder()
            result = builder.build_candidate_index(
                candidates_jsonl_path=candidates_jsonl,
                force_rebuild=True,
            )

            print("\n" + str(result))

            retriever = Retriever(force_rebuild=False)
            retriever.load_index()
            print(f"Loaded index with {retriever.get_index_size()} candidates")

        query = "machine learning engineer python tensorflow"
        print(f"\nSearching for: {query}")
        results = retriever.search(query, k=5)

        print(f"\nTop 5 Results:")
        for result in results:
            rank = result["rank"]
            candidate_id = result["candidate_id"]
            similarity = result["similarity"]
            metadata = result["metadata"]
            title = metadata.get("title", "N/A")
            experience = metadata.get("experience_years", "N/A")

            print(f"\n  Rank {rank}:")
            print(f"    Candidate ID: {candidate_id}")
            print(f"    Similarity: {similarity:.4f}")
            print(f"    Current Title: {title}")
            print(f"    Years Experience: {experience}")

        print("\n✓ Retrieval pipeline validated")

    except Exception as e:
        print(f"✗ Retrieval pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test HybridRanker
    print("\nTesting HybridRanker...")
    try:
        if "parsed_job" not in locals():
            print("⚠ Parsed job unavailable; skipping HybridRanker demo")
        else:
            candidates_jsonl = PROJECT_ROOT / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
            if not candidates_jsonl.exists():
                print(f"⚠ Candidates JSONL file not found: {candidates_jsonl}")
            else:
                candidate_resolver = _build_candidate_resolver(candidates_jsonl)
                ranker = HybridRanker(
                    retriever=retriever,
                    candidate_resolver=candidate_resolver,
                )

                ranked_results = ranker.rank_candidates(parsed_job, top_k=10)

                print(f"\nRetrieved Candidates ({len(ranker.last_retrieved_candidates)}):")
                for item in ranker.last_retrieved_candidates[:10]:
                    metadata = item.get("metadata", {})
                    print(f"  - {item['rank']}: {item['candidate_id']} | semantic={item['similarity']:.4f} | {metadata.get('title', 'N/A')}")

                print(f"\nTop 10 Ranking:")
                for rank, result in enumerate(ranked_results[:10], start=1):
                    partial_scores = result.metadata.get("partial_scores", {})
                    print(f"\nRank {rank}")
                    print(f"  Candidate: {result.candidate_id}")
                    print(f"  Semantic: {result.semantic_score:.2f}")
                    print(f"  Career: {result.career_score:.2f}")
                    print(f"  Skill: {result.skill_score:.2f}")
                    print(f"  Behavior: {result.behavior_score:.2f}")
                    print(f"  Education: {result.education_score:.2f}")
                    print(f"  Consistency: {result.consistency_score:.2f}")
                    print(f"  Final: {result.weighted_final_score:.2f}")
                    print(f"  Confidence: {result.confidence:.2f}")
                    print(f"  Matched Evidence: {', '.join(result.matched_items[:8]) if result.matched_items else 'None'}")
                    print(f"  Missing Evidence: {', '.join(result.missing_items[:8]) if result.missing_items else 'None'}")
                    print(f"  Partial Scores: {partial_scores}")

                print("\n✓ HybridRanker working correctly")

    except Exception as e:
        print(f"✗ HybridRanker test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nPipeline ready. Parser, CareerScorer, SkillScorer, BehaviorScorer, HybridRanker, JobDescriptionParser, and retrieval artifact loading are implemented.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the India Runs challenge pipeline")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of offline FAISS artifacts.",
    )
    args = parser.parse_args()
    sys.exit(main(force_rebuild=args.rebuild_index))
