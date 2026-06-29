"""Main entry point for the India Runs Data & AI Challenge pipeline."""

import sys
from pathlib import Path

import src
from src.config import (
    PROJECT_ROOT,
    SRC_DIR,
    TESTS_DIR,
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
)
from src.utils import setup_logging


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
    print("\nPipeline foundation ready. Awaiting further implementation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
