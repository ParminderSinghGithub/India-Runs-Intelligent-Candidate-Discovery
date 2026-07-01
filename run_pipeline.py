"""Main entry point for the India Runs Data & AI Challenge pipeline."""

import sys
from pathlib import Path

from src.config import (
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
from src.retrieval.document_builder import RetrievalDocumentBuilder
from src.embeddings.embedder import EmbeddingEngine


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


def main() -> int:
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

    # Test RetrievalDocumentBuilder
    print("\nTesting RetrievalDocumentBuilder...")
    try:
        doc_builder = RetrievalDocumentBuilder()
        retrieval_doc = doc_builder.build(candidate)

        print(f"\nCandidate ID: {retrieval_doc.candidate_id}")
        print(f"\nDocument Preview (first 500 chars):")
        print(retrieval_doc.document[:500])
        if len(retrieval_doc.document) > 500:
            print("...")

        print(f"\nMetadata:")
        print(f"  Title: {retrieval_doc.get_metadata('title')}")
        print(f"  Location: {retrieval_doc.get_metadata('location')}")
        print(f"  Experience Years: {retrieval_doc.get_metadata('experience_years')}")
        print(f"  Number of Skills: {retrieval_doc.get_metadata('number_of_skills')}")
        print(f"  Number of Career Entries: {retrieval_doc.get_metadata('number_of_career_entries')}")
        print(f"  Document Length: {retrieval_doc.get_metadata('document_length')} chars")
        print(f"  Document Word Count: {retrieval_doc.get_metadata('document_word_count')} words")
        print(f"  Section Count: {retrieval_doc.get_metadata('section_count')}")
        print(f"  Sections: {retrieval_doc.get_metadata('sections')}")

        print(f"\nFull Document:")
        print(retrieval_doc.document)

        print("\n✓ RetrievalDocumentBuilder working correctly")

    except Exception as e:
        print(f"✗ RetrievalDocumentBuilder test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test EmbeddingEngine
    print("\nTesting EmbeddingEngine...")
    try:
        import time

        # Parse 20 sample candidates
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

            # Limit to 20 candidates for testing
            candidates = candidates[:20]
            print(f"Parsed {len(candidates)} candidates for embedding test")

            # Generate RetrievalDocument objects
            doc_builder = RetrievalDocumentBuilder()
            retrieval_docs = [doc_builder.build(cand) for cand in candidates]

            print(f"Generated {len(retrieval_docs)} retrieval documents")

            # Initialize embedding engine
            embedder = EmbeddingEngine()

            # Load model
            embedder.load_model()
            print(f"Embedding model: {embedder.model_name}")
            print(f"Embedding dimension: {embedder.get_embedding_dimension()}")

            # Embed documents
            start_time = time.time()
            embeddings = embedder.embed_documents(retrieval_docs, show_progress=True)
            embedding_time = time.time() - start_time

            print(f"\nNumber of documents: {len(retrieval_docs)}")
            print(f"Embedding matrix shape: {embeddings.shape}")
            print(f"Embedding time: {embedding_time:.2f}s")
            print(f"Average time per document: {embedding_time / len(retrieval_docs):.3f}s")

            # Save embeddings
            candidate_ids = [doc.candidate_id for doc in retrieval_docs]
            metadata = [doc.metadata for doc in retrieval_docs]
            cache_path = embedder.save_embeddings(embeddings, candidate_ids, metadata, cache_name="test_embeddings")

            print(f"Cache location: {cache_path}")

            # Test loading embeddings
            loaded_embeddings, loaded_ids, loaded_metadata = embedder.load_embeddings(cache_name="test_embeddings")
            print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
            print(f"Loaded candidate IDs: {len(loaded_ids)}")

            print("\n✓ EmbeddingEngine working correctly")

    except Exception as e:
        print(f"✗ EmbeddingEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nPipeline ready. Parser, CareerScorer, JobDescriptionParser, RetrievalDocumentBuilder, and EmbeddingEngine implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
