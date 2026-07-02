"""Configuration module for India Runs Data & AI Challenge."""

import logging
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Project directory paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
TESTS_DIR: Final[Path] = PROJECT_ROOT / "tests"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"

# Embedding artifact directory (under artifacts/)
EMBEDDINGS_DIR: Final[Path] = ARTIFACTS_DIR / "embeddings"

# FAISS artifact directory (under artifacts/)
FAISS_DIR: Final[Path] = ARTIFACTS_DIR / "faiss"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: Final[int] = logging.INFO
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Top-level hybrid scorer weights
# Must sum to 1.0.  Career and skill evidence are the primary signals;
# behavior and consistency serve as tie-breakers only.
# ---------------------------------------------------------------------------
CAREER_WEIGHT: Final[float] = 0.38
SKILL_WEIGHT: Final[float] = 0.34
BEHAVIOR_WEIGHT: Final[float] = 0.05
SEMANTIC_WEIGHT: Final[float] = 0.18
EDUCATION_WEIGHT: Final[float] = 0.00
CONSISTENCY_WEIGHT: Final[float] = 0.05
CONSISTENCY_PENALTY: Final[float] = 0.1

# ---------------------------------------------------------------------------
# Career scorer component weights (must sum to 1.0)
# ---------------------------------------------------------------------------
CAREER_ROLE_RELEVANCE_WEIGHT: Final[float] = 0.38
CAREER_RESPONSIBILITIES_WEIGHT: Final[float] = 0.25
CAREER_PROGRESSION_WEIGHT: Final[float] = 0.15
CAREER_INDUSTRY_MATCH_WEIGHT: Final[float] = 0.10
CAREER_RELEVANT_EXPERIENCE_WEIGHT: Final[float] = 0.12

# ---------------------------------------------------------------------------
# Score range validation bounds (for internal clamping only)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Behavior scorer — component weights
# ---------------------------------------------------------------------------
BEHAVIOR_PROFILE_QUALITY_WEIGHT: Final[float] = 0.18
BEHAVIOR_RECRUITER_INTEREST_WEIGHT: Final[float] = 0.18
BEHAVIOR_CANDIDATE_ACTIVITY_WEIGHT: Final[float] = 0.18
BEHAVIOR_HIRING_RELIABILITY_WEIGHT: Final[float] = 0.16
BEHAVIOR_AVAILABILITY_WEIGHT: Final[float] = 0.15
BEHAVIOR_PROFESSIONAL_NETWORK_WEIGHT: Final[float] = 0.15

# ---------------------------------------------------------------------------
# Behavior scorer — signal weights (within each component)
# ---------------------------------------------------------------------------
BEHAVIOR_PROFILE_COMPLETENESS_WEIGHT: Final[float] = 0.40
BEHAVIOR_PROFILE_EMAIL_WEIGHT: Final[float] = 0.20
BEHAVIOR_PROFILE_PHONE_WEIGHT: Final[float] = 0.20
BEHAVIOR_PROFILE_LINKEDIN_WEIGHT: Final[float] = 0.20
BEHAVIOR_RECRUITER_VIEWS_WEIGHT: Final[float] = 0.40
BEHAVIOR_RECRUITER_SAVES_WEIGHT: Final[float] = 0.35
BEHAVIOR_RECRUITER_SEARCH_WEIGHT: Final[float] = 0.25
BEHAVIOR_ACTIVITY_RECENCY_WEIGHT: Final[float] = 0.45
BEHAVIOR_ACTIVITY_APPLICATIONS_WEIGHT: Final[float] = 0.35
BEHAVIOR_ACTIVITY_GITHUB_WEIGHT: Final[float] = 0.20
BEHAVIOR_RELIABILITY_INTERVIEW_WEIGHT: Final[float] = 0.55
BEHAVIOR_RELIABILITY_OFFER_WEIGHT: Final[float] = 0.45
BEHAVIOR_AVAILABILITY_OPEN_WEIGHT: Final[float] = 0.30
BEHAVIOR_AVAILABILITY_NOTICE_WEIGHT: Final[float] = 0.25
BEHAVIOR_AVAILABILITY_WORK_MODE_WEIGHT: Final[float] = 0.25
BEHAVIOR_AVAILABILITY_RELOCATE_WEIGHT: Final[float] = 0.20
BEHAVIOR_NETWORK_CONNECTIONS_WEIGHT: Final[float] = 0.60
BEHAVIOR_NETWORK_ENDORSEMENTS_WEIGHT: Final[float] = 0.40

# ---------------------------------------------------------------------------
# Behavior scorer — penalty weights and total penalty cap
# ---------------------------------------------------------------------------
BEHAVIOR_INCOMPLETE_PROFILE_PENALTY_WEIGHT: Final[float] = 0.08
BEHAVIOR_INACTIVE_PROFILE_PENALTY_WEIGHT: Final[float] = 0.08
BEHAVIOR_UNVERIFIED_PROFILE_PENALTY_WEIGHT: Final[float] = 0.07
BEHAVIOR_LOW_ENGAGEMENT_PENALTY_WEIGHT: Final[float] = 0.04
BEHAVIOR_POOR_RELIABILITY_PENALTY_WEIGHT: Final[float] = 0.08
BEHAVIOR_MAX_TOTAL_PENALTY: Final[float] = 0.30

# ---------------------------------------------------------------------------
# Behavior scorer — normalization parameters
# ---------------------------------------------------------------------------
BEHAVIOR_LOG_COUNT_NORMALIZATION_CAP: Final[int] = 1000
BEHAVIOR_ACTIVITY_RECENCY_HALF_LIFE_DAYS: Final[int] = 45
BEHAVIOR_NOTICE_PERIOD_MAX_DAYS: Final[int] = 180
BEHAVIOR_NOTICE_PERIOD_RELIEF_FACTOR: Final[float] = 0.30
BEHAVIOR_MIN_NOT_OPEN_SCORE: Final[float] = 0.35
BEHAVIOR_OPEN_TO_WORK_SCORE: Final[float] = 1.00
BEHAVIOR_RELOCATE_SCORE: Final[float] = 1.00
BEHAVIOR_NOT_RELOCATE_SCORE: Final[float] = 0.72
BEHAVIOR_GITHUB_ACTIVITY_THRESHOLD: Final[float] = 0.60
BEHAVIOR_INTERVIEW_COMPLETION_THRESHOLD: Final[float] = 0.75
BEHAVIOR_OFFER_ACCEPTANCE_THRESHOLD: Final[float] = 0.55
BEHAVIOR_PROFILE_QUALITY_INCOMPLETE_THRESHOLD: Final[float] = 0.65
BEHAVIOR_MIN_AVAILABILITY_SCORE: Final[float] = 0.35
BEHAVIOR_TOTAL_SIGNALS_EXPECTED: Final[int] = 18

# ---------------------------------------------------------------------------
# Behavior scorer — preferred work-mode scores
# ---------------------------------------------------------------------------
BEHAVIOR_WORK_MODE_SCORES: Final[dict] = {
    "remote": 1.00,
    "work from home": 1.00,
    "wfh": 1.00,
    "hybrid": 0.90,
    "flexible": 0.95,
    "onsite": 0.75,
    "on-site": 0.75,
    "office": 0.75,
    "wfo": 0.75,
}

# ---------------------------------------------------------------------------
# Output encoding
# ---------------------------------------------------------------------------
OUTPUT_FILE_FORMAT: Final[str] = "json"
OUTPUT_ENCODING: Final[str] = "utf-8"

# ---------------------------------------------------------------------------
# Embedding model configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE: Final[int] = 32
EMBEDDING_DEVICE: Final[str] = "auto"  # "auto", "cuda", "cpu"
EMBEDDING_NORMALIZE: Final[bool] = True
EMBEDDING_CACHE_ENABLED: Final[bool] = True

# ---------------------------------------------------------------------------
# FAISS index configuration
# ---------------------------------------------------------------------------
FAISS_INDEX_TYPE: Final[str] = "IndexFlatIP"  # Inner product (requires L2-normalized embeddings)
FAISS_NPROBE: Final[int] = 10  # For IVF variants; not used with IndexFlatIP

# ---------------------------------------------------------------------------
# Skill scorer — component weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SKILL_REQUIRED_MATCH_WEIGHT: Final[float] = 0.45
SKILL_PREFERRED_MATCH_WEIGHT: Final[float] = 0.08
SKILL_PROFICIENCY_WEIGHT: Final[float] = 0.18
SKILL_DURATION_WEIGHT: Final[float] = 0.10
SKILL_ENDORSEMENT_WEIGHT: Final[float] = 0.06
SKILL_TECHNOLOGY_COVERAGE_WEIGHT: Final[float] = 0.08
SKILL_DIVERSITY_WEIGHT: Final[float] = 0.05

# ---------------------------------------------------------------------------
# Skill scorer — normalization and confidence tuning
# ---------------------------------------------------------------------------
SKILL_DURATION_MAX_MONTHS: Final[int] = 24
SKILL_ENDORSEMENT_MAX_COUNT: Final[int] = 10
SKILL_DIVERSITY_MAX_CATEGORIES: Final[int] = 5
SKILL_TECHNOLOGY_MATCH_WEIGHT: Final[float] = 0.75
SKILL_TECHNOLOGY_BREADTH_WEIGHT: Final[float] = 0.25
SKILL_CONFIDENCE_MATCH_WEIGHT: Final[float] = 0.40
SKILL_CONFIDENCE_DURATION_WEIGHT: Final[float] = 0.20
SKILL_CONFIDENCE_ENDORSEMENT_WEIGHT: Final[float] = 0.20
SKILL_CONFIDENCE_EVIDENCE_WEIGHT: Final[float] = 0.20
SKILL_CONFIDENCE_DURATION_THRESHOLD_MONTHS: Final[int] = 12
SKILL_CONFIDENCE_ENDORSEMENT_THRESHOLD: Final[int] = 5
SKILL_CONFIDENCE_EVIDENCE_THRESHOLD: Final[int] = 2

# ---------------------------------------------------------------------------
# Skill diversity — categories used for heuristic scoring
# ---------------------------------------------------------------------------
SKILL_DIVERSITY_CATEGORIES: Final[dict] = {
    "programming": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "go",
        "rust",
        "ruby",
        "php",
        "scala",
        "kotlin",
        "swift",
    ],
    "machine learning": [
        "machine learning",
        "ml",
        "scikit-learn",
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
    ],
    "deep learning": [
        "deep learning",
        "dl",
        "tensorflow",
        "pytorch",
        "keras",
        "torch",
    ],
    "cloud": ["aws", "gcp", "azure", "cloud", "cloud computing"],
    "data engineering": ["spark", "hadoop", "kafka", "airflow", "etl", "elt", "dbt"],
    "databases": ["sql", "nosql", "mongodb", "postgresql", "postgres", "mysql", "redis"],
    "visualization": ["tableau", "power bi", "powerbi", "matplotlib", "seaborn", "plotly", "visualization"],
    "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "cicd", "terraform", "ansible"],
    "mlops": ["mlops", "model deployment", "feature store", "experiment tracking", "mlflow"],
    "soft skills": ["communication", "leadership", "collaboration", "stakeholder", "mentoring", "presentation"],
}

# ---------------------------------------------------------------------------
# Skill proficiency — normalized score mapping
# ---------------------------------------------------------------------------
SKILL_PROFICIENCY_MAPPING: Final[dict] = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.75,
    "expert": 1.00,
}

# ---------------------------------------------------------------------------
# Skill synonyms — canonical form → accepted aliases
# ---------------------------------------------------------------------------
SKILL_SYNONYMS: Final[dict] = {
    # ML/AI
    "machine learning": ["ml", "machinelearning", "machine learning engineering", "ml engineering"],
    "deep learning": ["dl", "deeplearning", "neural networks", "neural network"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "computer vision": ["cv"],
    # Frameworks
    "tensorflow": ["tf"],
    "pytorch": ["torch"],
    "keras": [],
    "scikit-learn": ["sklearn"],
    # Languages
    "javascript": ["js"],
    "typescript": ["ts"],
    "python": ["py"],
    "java": [],
    "c++": ["cpp"],
    "c#": ["csharp"],
    # Cloud
    "aws": ["amazon web services"],
    "gcp": ["google cloud platform"],
    "azure": ["microsoft azure"],
    # Databases
    "sql": ["structured query language"],
    "nosql": [],
    "mongodb": ["mongo"],
    "postgresql": ["postgres"],
    # DevOps
    "docker": [],
    "kubernetes": ["k8s"],
    "jenkins": [],
    "ci/cd": ["cicd", "continuous integration"],
    # Data
    "spark": ["apache spark"],
    "hadoop": [],
    "kafka": ["apache kafka"],
    "airflow": ["apache airflow"],
    # Web
    "react": ["reactjs"],
    "vue": ["vuejs"],
    "angular": [],
    "node.js": ["nodejs", "node"],
    # Tools
    "git": [],
    "github": [],
    "gitlab": [],
    "linux": [],
    "unix": [],
}

# ---------------------------------------------------------------------------
# Hugging Face deployment configuration
# ---------------------------------------------------------------------------
HF_DATASET_REPO: Final[str] = "ParminderzHuggingFace/india-runs-faiss-artifacts"
HF_DATASET_REVISION: Final[str] = "main"
HF_ARTIFACT_DIR: Final[Path] = ARTIFACTS_DIR / "faiss"
HF_DOWNLOAD_TIMEOUT: Final[int] = 300
HF_FORCE_DOWNLOAD: Final[bool] = False
