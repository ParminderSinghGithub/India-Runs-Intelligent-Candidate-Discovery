"""Configuration module for India Runs Data & AI Challenge."""

import logging
from pathlib import Path
from typing import Final

# Project paths
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
TESTS_DIR: Final[Path] = PROJECT_ROOT / "tests"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"

# Data paths (placeholders - to be configured)
CANDIDATES_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "candidates"
JOB_DESCRIPTIONS_PATH: Final[Path] = PROJECT_ROOT / "data" / "job_descriptions"

# Random seed for reproducibility
RANDOM_SEED: Final[int] = 42

# Logging configuration
LOG_LEVEL: Final[int] = logging.INFO
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Scoring weights (placeholders for now)
CAREER_WEIGHT: Final[float] = 0.25
SKILL_WEIGHT: Final[float] = 0.25
BEHAVIOR_WEIGHT: Final[float] = 0.25
SEMANTIC_WEIGHT: Final[float] = 0.15
CONSISTENCY_WEIGHT: Final[float] = 0.10

# Validation thresholds
MIN_CAREER_SCORE: Final[float] = 0.0
MAX_CAREER_SCORE: Final[float] = 1.0
MIN_SKILL_SCORE: Final[float] = 0.0
MAX_SKILL_SCORE: Final[float] = 1.0
MIN_BEHAVIOR_SCORE: Final[float] = 0.0
MAX_BEHAVIOR_SCORE: Final[float] = 1.0
MIN_SEMANTIC_SCORE: Final[float] = 0.0
MAX_SEMANTIC_SCORE: Final[float] = 1.0
MIN_CONSISTENCY_SCORE: Final[float] = 0.0
MAX_CONSISTENCY_SCORE: Final[float] = 1.0

# Output configuration
OUTPUT_FILE_FORMAT: Final[str] = "json"
OUTPUT_ENCODING: Final[str] = "utf-8"
