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

    print("\nPipeline ready. Parser module implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
