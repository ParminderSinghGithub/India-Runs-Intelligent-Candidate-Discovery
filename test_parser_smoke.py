"""Smoke test for CandidateParser."""

from pathlib import Path

from src.parser.candidate_parser import CandidateParser
from src.utils import setup_logging


def main() -> None:
    """Run smoke test for CandidateParser."""
    setup_logging()

    print("=== CandidateParser Smoke Test ===\n")

    parser = CandidateParser()
    sample_file = Path("sample_candidates.json")

    if not sample_file.exists():
        print(f"Error: {sample_file} not found")
        return

    try:
        candidates = parser.parse_many(sample_file)
        
        if not candidates:
            print("No candidates parsed")
            return

        candidate = candidates[0]

        print(f"Candidate ID: {candidate.candidate_id}")
        print(f"Current Title: {candidate.profile.current_title}")
        print(f"Number of Jobs: {len(candidate.career_history)}")
        print(f"Number of Skills: {len(candidate.skills)}")
        print(f"Years of Experience: {candidate.profile.years_of_experience}")
        print(f"Redrob Signals: Parsed successfully")
        print(f"  - Profile Completeness: {candidate.redrob_signals.profile_completeness_score}")
        print(f"  - GitHub Activity Score: {candidate.redrob_signals.github_activity_score}")
        print(f"  - Offer Acceptance Rate: {candidate.redrob_signals.offer_acceptance_rate}")
        print(f"\n✓ Smoke test passed - parsing successful")

    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        raise


if __name__ == "__main__":
    main()
