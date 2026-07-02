"""Pipeline artifact and configuration validator.

Validates that all required artifacts, output files, and configuration
values are present and internally consistent before a ranking run begins.
This is a read-only diagnostic tool and makes no changes to the filesystem.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import (
    ARTIFACTS_DIR,
    EMBEDDING_MODEL_NAME,
    FAISS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValidationCheck:
    """A single pass/fail check with an optional detail message."""

    name: str
    passed: bool
    message: str = ""


@dataclass
class PipelineValidationReport:
    """Aggregated results of all pipeline validation checks."""

    checks: List[ValidationCheck] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> List[ValidationCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def num_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def summary_lines(self) -> List[str]:
        """Return a human-readable list of check results."""
        lines: List[str] = []
        for check in self.checks:
            icon = "OK  " if check.passed else "FAIL"
            line = f"  [{icon}] {check.name}"
            if check.message:
                line += f": {check.message}"
            lines.append(line)
        return lines

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        header = f"Pipeline Validation: {status} ({self.num_passed}/{self.total} checks, {self.elapsed_seconds:.2f}s)"
        return "\n".join([header] + self.summary_lines())


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class PipelineValidator:
    """Validate pipeline artifacts and configuration without modifying anything.

    Usage::

        validator = PipelineValidator()
        report = validator.validate()
        if not report.passed:
            for check in report.failed_checks:
                print(check.name, check.message)
    """

    # Required FAISS artifact filenames
    FAISS_ARTIFACTS = [
        "faiss.index",
        "candidate_lookup.pkl",
        "embedding_metadata.pkl",
    ]

    # Required output filenames produced by a successful ranking run
    OUTPUT_FILES = [
        "submission.csv",
        "submission.xlsx",
        "ranking.json",
        "pipeline_report.json",
    ]

    # Configuration constants that must be within sensible ranges
    WEIGHT_CONSTANTS_SUM_KEY = "scorer_weights_sum"

    def validate(
        self,
        *,
        check_outputs: bool = True,
        check_source_data: bool = True,
    ) -> PipelineValidationReport:
        """Run all validation checks and return a structured report.

        Args:
            check_outputs: Whether to validate output files (skip on first run).
            check_source_data: Whether to validate raw candidate JSONL source.

        Returns:
            PipelineValidationReport with individual check results.
        """
        start = time.perf_counter()
        logger.info("[PipelineValidator] START validation")

        checks: List[ValidationCheck] = []

        # --- FAISS artifacts --------------------------------------------------
        checks.extend(self._check_faiss_artifacts())

        # --- Pickle file integrity ---------------------------------------------
        checks.extend(self._check_pickle_integrity())

        # --- Outputs directory ------------------------------------------------
        if check_outputs:
            checks.extend(self._check_output_files())

        # --- Configuration ----------------------------------------------------
        checks.extend(self._check_configuration())

        # --- Source data (optional) -------------------------------------------
        if check_source_data:
            checks.extend(self._check_source_files())

        elapsed = time.perf_counter() - start
        report = PipelineValidationReport(checks=checks, elapsed_seconds=elapsed)

        logger.info(
            "[PipelineValidator] END validation -- %d/%d checks passed in %.2fs",
            report.num_passed,
            report.total,
            elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # FAISS artifact checks
    # ------------------------------------------------------------------

    def _check_faiss_artifacts(self) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []

        # Parent directory
        checks.append(
            ValidationCheck(
                name="faiss_directory_exists",
                passed=FAISS_DIR.exists() and FAISS_DIR.is_dir(),
                message=str(FAISS_DIR),
            )
        )

        for filename in self.FAISS_ARTIFACTS:
            artifact_path = FAISS_DIR / filename
            exists = artifact_path.exists() and artifact_path.is_file()
            size_bytes = artifact_path.stat().st_size if exists else 0
            checks.append(
                ValidationCheck(
                    name=f"faiss_artifact_{filename}",
                    passed=exists and size_bytes > 0,
                    message=f"{artifact_path} ({size_bytes} bytes)" if exists else f"MISSING: {artifact_path}",
                )
            )

        return checks

    # ------------------------------------------------------------------
    # Pickle integrity checks
    # ------------------------------------------------------------------

    def _check_pickle_integrity(self) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []

        # candidate_lookup
        lookup_path = FAISS_DIR / "candidate_lookup.pkl"
        check = self._load_pickle_check(
            name="candidate_lookup_readable",
            path=lookup_path,
            expected_type=dict,
        )
        checks.append(check)

        if check.passed:
            # Verify it has at least some entries
            try:
                with lookup_path.open("rb") as fh:
                    data: Dict[str, Any] = pickle.load(fh)
                count = len(data)
                checks.append(
                    ValidationCheck(
                        name="candidate_lookup_non_empty",
                        passed=count > 0,
                        message=f"{count} entries",
                    )
                )
            except Exception as exc:  # pragma: no cover
                checks.append(
                    ValidationCheck(
                        name="candidate_lookup_non_empty",
                        passed=False,
                        message=str(exc),
                    )
                )

        # embedding_metadata
        meta_path = FAISS_DIR / "embedding_metadata.pkl"
        meta_check = self._load_pickle_check(
            name="embedding_metadata_readable",
            path=meta_path,
            expected_type=None,  # can be list or dict
        )
        checks.append(meta_check)

        if meta_check.passed:
            try:
                with meta_path.open("rb") as fh:
                    meta = pickle.load(fh)
                meta_len = len(meta) if hasattr(meta, "__len__") else -1
                checks.append(
                    ValidationCheck(
                        name="embedding_metadata_non_empty",
                        passed=meta_len > 0,
                        message=f"{meta_len} records",
                    )
                )
            except Exception as exc:  # pragma: no cover
                checks.append(
                    ValidationCheck(
                        name="embedding_metadata_non_empty",
                        passed=False,
                        message=str(exc),
                    )
                )

        return checks

    def _load_pickle_check(
        self,
        name: str,
        path: Path,
        expected_type: Optional[type],
    ) -> ValidationCheck:
        if not path.exists():
            return ValidationCheck(name=name, passed=False, message=f"MISSING: {path}")
        try:
            with path.open("rb") as fh:
                obj = pickle.load(fh)
            if expected_type is not None and not isinstance(obj, expected_type):
                return ValidationCheck(
                    name=name,
                    passed=False,
                    message=f"Expected {expected_type.__name__}, got {type(obj).__name__}",
                )
            return ValidationCheck(name=name, passed=True, message=str(path))
        except Exception as exc:
            return ValidationCheck(name=name, passed=False, message=f"Pickle error: {exc}")

    # ------------------------------------------------------------------
    # Output file checks
    # ------------------------------------------------------------------

    def _check_output_files(self) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []

        checks.append(
            ValidationCheck(
                name="outputs_directory_exists",
                passed=OUTPUTS_DIR.exists() and OUTPUTS_DIR.is_dir(),
                message=str(OUTPUTS_DIR),
            )
        )

        for filename in self.OUTPUT_FILES:
            output_path = OUTPUTS_DIR / filename
            exists = output_path.exists() and output_path.is_file()
            size_bytes = output_path.stat().st_size if exists else 0
            checks.append(
                ValidationCheck(
                    name=f"output_{filename.replace('.', '_')}",
                    passed=exists and size_bytes > 0,
                    message=f"{output_path} ({size_bytes} bytes)" if exists else f"MISSING: {output_path}",
                )
            )

        # Validate submission.csv row count
        submission_path = OUTPUTS_DIR / "submission.csv"
        if submission_path.exists():
            try:
                import csv

                with submission_path.open("r", encoding="utf-8", newline="") as fh:
                    rows = list(csv.reader(fh))
                # First row is header; data rows follow
                data_rows = len(rows) - 1 if rows else 0
                checks.append(
                    ValidationCheck(
                        name="submission_csv_row_count",
                        passed=data_rows == 100,
                        message=f"{data_rows} data rows (expected 100)",
                    )
                )
            except Exception as exc:
                checks.append(
                    ValidationCheck(
                        name="submission_csv_row_count",
                        passed=False,
                        message=str(exc),
                    )
                )

        # Validate submission.xlsx row count
        xlsx_path = OUTPUTS_DIR / "submission.xlsx"
        if xlsx_path.exists():
            try:
                import openpyxl

                wb = openpyxl.load_workbook(xlsx_path, read_only=True)
                ws = wb.active
                # In read_only mode, we can count rows. Header row + 100 candidate rows = 101 max_row or manual iteration
                row_count = 0
                for row in ws.iter_rows(values_only=True):
                    row_count += 1
                data_rows = row_count - 1 if row_count > 0 else 0
                wb.close()

                checks.append(
                    ValidationCheck(
                        name="submission_xlsx_row_count",
                        passed=data_rows == 100,
                        message=f"{data_rows} data rows (expected 100)",
                    )
                )
            except Exception as exc:
                checks.append(
                    ValidationCheck(
                        name="submission_xlsx_row_count",
                        passed=False,
                        message=str(exc),
                    )
                )

        return checks

    # ------------------------------------------------------------------
    # Configuration checks
    # ------------------------------------------------------------------

    def _check_configuration(self) -> List[ValidationCheck]:
        from src.config import (
            BEHAVIOR_WEIGHT,
            CAREER_WEIGHT,
            CONSISTENCY_WEIGHT,
            EDUCATION_WEIGHT,
            EMBEDDING_MODEL_NAME,
            SEMANTIC_WEIGHT,
            SKILL_WEIGHT,
        )

        checks: List[ValidationCheck] = []

        # Scorer weights must sum to 1.0 (±0.005 tolerance for float rounding)
        weight_sum = round(
            CAREER_WEIGHT
            + SKILL_WEIGHT
            + BEHAVIOR_WEIGHT
            + SEMANTIC_WEIGHT
            + EDUCATION_WEIGHT
            + CONSISTENCY_WEIGHT,
            6,
        )
        checks.append(
            ValidationCheck(
                name="scorer_weights_sum_to_one",
                passed=abs(weight_sum - 1.0) < 0.005,
                message=f"sum={weight_sum}",
            )
        )

        # Individual weights must be in [0, 1]
        named_weights = {
            "CAREER_WEIGHT": CAREER_WEIGHT,
            "SKILL_WEIGHT": SKILL_WEIGHT,
            "BEHAVIOR_WEIGHT": BEHAVIOR_WEIGHT,
            "SEMANTIC_WEIGHT": SEMANTIC_WEIGHT,
            "EDUCATION_WEIGHT": EDUCATION_WEIGHT,
            "CONSISTENCY_WEIGHT": CONSISTENCY_WEIGHT,
        }
        for name, value in named_weights.items():
            checks.append(
                ValidationCheck(
                    name=f"weight_{name.lower()}_valid",
                    passed=0.0 <= value <= 1.0,
                    message=f"{value}",
                )
            )

        # Embedding model name must be set
        checks.append(
            ValidationCheck(
                name="embedding_model_name_set",
                passed=bool(EMBEDDING_MODEL_NAME),
                message=EMBEDDING_MODEL_NAME,
            )
        )

        return checks

    # ------------------------------------------------------------------
    # Source file checks
    # ------------------------------------------------------------------

    def _check_source_files(self) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []

        job_description = PROJECT_ROOT / "job_description.json"
        checks.append(
            ValidationCheck(
                name="job_description_json_exists",
                passed=job_description.exists(),
                message=str(job_description),
            )
        )

        candidates_jsonl = (
            PROJECT_ROOT
            / "[PUB] India_runs_data_and_ai_challenge"
            / "[PUB] India_runs_data_and_ai_challenge"
            / "India_runs_data_and_ai_challenge"
            / "candidates.jsonl"
        )
        checks.append(
            ValidationCheck(
                name="candidates_jsonl_exists",
                passed=candidates_jsonl.exists(),
                message=str(candidates_jsonl),
            )
        )

        return checks
