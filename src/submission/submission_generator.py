"""Submission generation and export utilities."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.config import (
    ARTIFACTS_DIR,
    BEHAVIOR_WEIGHT,
    CAREER_WEIGHT,
    CONSISTENCY_WEIGHT,
    EDUCATION_WEIGHT,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
)
from src.models.candidate import Candidate
from src.models.hybrid_score_result import HybridScoreResult
from src.models.parsed_job import ParsedJob
from src.models.submission_row import SubmissionRow
from src.parser.candidate_parser import CandidateParser
from src.utils.exceptions import ValidationError
from .reason_generator import ReasonGenerator
from .submission_validator import SubmissionValidator

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class CandidateResolver:
    """Lazy JSONL-backed candidate resolver for ranking and validation."""

    def __init__(self, candidates_jsonl_path: Path):
        self.candidates_jsonl_path = Path(candidates_jsonl_path)
        self._parser = CandidateParser()
        self._cache: Dict[str, Candidate] = {}
        self._loaded = False

    def resolve(self, candidate_id: str) -> Optional[Candidate]:
        """Resolve a candidate by ID, loading the JSONL source on first use."""
        self._ensure_loaded()
        return self._cache.get(candidate_id)

    def has(self, candidate_id: str) -> bool:
        """Check whether a candidate exists in the source dataset."""
        return self.resolve(candidate_id) is not None

    @property
    def candidate_ids(self) -> List[str]:
        """Return the loaded candidate IDs."""
        self._ensure_loaded()
        return list(self._cache.keys())

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.candidates_jsonl_path.exists():
            raise FileNotFoundError(f"Candidates JSONL not found: {self.candidates_jsonl_path}")

        with self.candidates_jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                candidate = self._parser.from_jsonl_line(line)
                self._cache[candidate.candidate_id] = candidate
        self._loaded = True


class SubmissionGenerator:
    """Builds submission artifacts from ranked candidates."""

    def __init__(
        self,
        output_dir: Path = OUTPUTS_DIR,
        reason_generator: Optional[ReasonGenerator] = None,
        validator: Optional[SubmissionValidator] = None,
    ):
        self.output_dir = Path(output_dir)
        self.reason_generator = reason_generator or ReasonGenerator()
        self.validator = validator or SubmissionValidator()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_submission(
        self,
        parsed_job: ParsedJob,
        ranked_results: Sequence[HybridScoreResult],
        candidate_resolver: Callable[[str], Optional[Candidate]],
        *,
        top_n: int = 100,
        submission_csv_path: Optional[Path] = None,
        ranking_json_path: Optional[Path] = None,
        pipeline_report_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        candidate_exists: Optional[Callable[[str], bool]] = None,
        retrieval_results: Optional[Sequence[Dict[str, Any]]] = None,
        timings: Optional[Dict[str, float]] = None,
        artifact_paths: Optional[Dict[str, str]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        ai_tools_used: Optional[List[str]] = None,
        ai_usage_summary: Optional[str] = None,
        methodology_summary: Optional[str] = None,
        pipeline_summary: Optional[str] = None,
        compute_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate all submission artifacts and validate them."""
        submission_csv_path = Path(submission_csv_path or (self.output_dir / "submission.csv"))
        ranking_json_path = Path(ranking_json_path or (self.output_dir / "ranking.json"))
        pipeline_report_path = Path(pipeline_report_path or (self.output_dir / "pipeline_report.json"))
        metadata_path = Path(metadata_path or (PROJECT_ROOT / "submission_metadata.yaml"))

        selected_results = list(ranked_results[:top_n])
        rows = self.build_rows(parsed_job, selected_results, candidate_resolver)

        validation_errors = self.validator.validate_rows(
            rows,
            candidate_exists=candidate_exists,
            expected_rows=top_n,
        )
        if validation_errors:
            raise ValidationError("Submission validation failed: " + "; ".join(validation_errors))

        csv_seconds = self._write_csv(rows, submission_csv_path)
        csv_errors = self.validator.validate_csv_file(
            submission_csv_path,
            candidate_exists=candidate_exists,
            expected_rows=top_n,
        )
        if csv_errors:
            raise ValidationError("Submission CSV validation failed: " + "; ".join(csv_errors))

        ranking_entries = self._build_ranking_entries(selected_results, rows)
        ranking_seconds = self._write_json(
            ranking_json_path,
            {
                "job": {
                    "job_id": parsed_job.job_description.job_id,
                    "title": parsed_job.job_description.title,
                    "company": parsed_job.job_description.company,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "results": ranking_entries,
            },
        )

        summary = self._build_pipeline_report(
            parsed_job=parsed_job,
            ranked_results=selected_results,
            ranking_entries=ranking_entries,
            timing_breakdown=timings or {},
            artifact_paths=artifact_paths or {},
            configuration=configuration or {},
            retrieval_results=list(retrieval_results or []),
            submission_csv_path=submission_csv_path,
            ranking_json_path=ranking_json_path,
            pipeline_report_path=pipeline_report_path,
            metadata_path=metadata_path,
        )
        report_seconds = self._write_json(pipeline_report_path, summary)

        metadata_seconds = self._write_text(
            metadata_path,
            self._build_metadata_yaml(
                parsed_job=parsed_job,
                submission_csv_path=submission_csv_path,
                ranking_json_path=ranking_json_path,
                pipeline_report_path=pipeline_report_path,
                artifact_paths=artifact_paths or {},
                configuration=configuration or {},
                ai_tools_used=ai_tools_used or ["GitHub Copilot"],
                ai_usage_summary=ai_usage_summary or "Used GitHub Copilot for implementation support. No candidate data was sent to any external LLM service during ranking.",
                methodology_summary=methodology_summary or self._default_methodology_summary(),
                pipeline_summary=pipeline_summary or self._default_pipeline_summary(),
                compute_summary=compute_summary or self._default_compute_summary(),
            ),
        )

        return {
            "rows": rows,
            "ranking_entries": ranking_entries,
            "submission_csv_path": submission_csv_path,
            "ranking_json_path": ranking_json_path,
            "pipeline_report_path": pipeline_report_path,
            "metadata_path": metadata_path,
            "validation_errors": [],
            "timings": {
                "csv_generation_seconds": csv_seconds,
                "ranking_json_seconds": ranking_seconds,
                "pipeline_report_seconds": report_seconds,
                "metadata_yaml_seconds": metadata_seconds,
            },
            "pipeline_report": summary,
        }

    def build_rows(
        self,
        parsed_job: ParsedJob,
        ranked_results: Sequence[HybridScoreResult],
        candidate_resolver: Callable[[str], Optional[Candidate]],
    ) -> List[SubmissionRow]:
        """Create submission rows from ranked results."""
        rows: List[SubmissionRow] = []
        for rank, result in enumerate(ranked_results, start=1):
            candidate = candidate_resolver(result.candidate_id)
            if candidate is None:
                raise ValidationError(f"Unable to resolve candidate '{result.candidate_id}' for submission generation.")

            reasoning = self.reason_generator.generate(candidate, result, parsed_job)
            rows.append(
                SubmissionRow(
                    candidate_id=result.candidate_id,
                    rank=rank,
                    score=float(result.weighted_final_score),
                    reasoning=reasoning,
                )
            )
        return rows

    def _write_csv(self, rows: Sequence[SubmissionRow], path: Path) -> float:
        path.parent.mkdir(parents=True, exist_ok=True)
        started = self._now()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "candidate_id": row.candidate_id,
                        "rank": row.rank,
                        "score": f"{row.score:.6f}",
                        "reasoning": row.reasoning,
                    }
                )
        return self._now() - started

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> float:
        path.parent.mkdir(parents=True, exist_ok=True)
        started = self._now()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return self._now() - started

    def _write_text(self, path: Path, content: str) -> float:
        path.parent.mkdir(parents=True, exist_ok=True)
        started = self._now()
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return self._now() - started

    def _build_ranking_entries(
        self,
        ranked_results: Sequence[HybridScoreResult],
        rows: Sequence[SubmissionRow],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for row, result in zip(rows, ranked_results):
            entries.append(
                {
                    "candidate_id": row.candidate_id,
                    "rank": row.rank,
                    "score": row.score,
                    "reasoning": row.reasoning,
                    "semantic_score": result.semantic_score,
                    "career_score": result.career_score,
                    "skill_score": result.skill_score,
                    "behavior_score": result.behavior_score,
                    "education_score": result.education_score,
                    "consistency_score": result.consistency_score,
                    "weighted_final_score": result.weighted_final_score,
                    "confidence": result.confidence,
                    "matched_evidence": list(result.matched_items),
                    "missing_evidence": list(result.missing_items),
                    "component_scores": result.metadata.get("component_scores", {}),
                    "component_confidences": result.metadata.get("component_confidences", {}),
                    "metadata": result.metadata,
                }
            )
        return entries

    def _build_pipeline_report(
        self,
        *,
        parsed_job: ParsedJob,
        ranked_results: Sequence[HybridScoreResult],
        ranking_entries: Sequence[Dict[str, Any]],
        timing_breakdown: Dict[str, float],
        artifact_paths: Dict[str, str],
        configuration: Dict[str, Any],
        retrieval_results: Sequence[Dict[str, Any]],
        submission_csv_path: Path,
        ranking_json_path: Path,
        pipeline_report_path: Path,
        metadata_path: Path,
    ) -> Dict[str, Any]:
        retrieval_scores = [float(item.get("similarity", 0.0)) for item in retrieval_results]
        weights = {
            "semantic": SEMANTIC_WEIGHT,
            "career": CAREER_WEIGHT,
            "skill": SKILL_WEIGHT,
            "behavior": BEHAVIOR_WEIGHT,
            "education": EDUCATION_WEIGHT,
            "consistency": CONSISTENCY_WEIGHT,
        }

        total_runtime = sum(timing_breakdown.values())
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": total_runtime,
            "retrieval_latency_seconds": timing_breakdown.get("retrieval_latency_seconds", 0.0),
            "reranking_latency_seconds": timing_breakdown.get("reranking_latency_seconds", 0.0),
            "csv_generation_seconds": timing_breakdown.get("csv_generation_seconds", 0.0),
            "validation_seconds": timing_breakdown.get("validation_seconds", 0.0),
            "candidate_counts": {
                "retrieved": len(retrieval_results),
                "ranked": len(ranked_results),
                "submitted": len(ranking_entries),
            },
            "retrieval_statistics": {
                "average_similarity": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0,
                "max_similarity": max(retrieval_scores) if retrieval_scores else 0.0,
                "min_similarity": min(retrieval_scores) if retrieval_scores else 0.0,
            },
            "weights": weights,
            "configuration": configuration,
            "top10": list(ranking_entries[:10]),
            "artifact_paths": {
                "submission_csv": str(submission_csv_path),
                "ranking_json": str(ranking_json_path),
                "pipeline_report": str(pipeline_report_path),
                "submission_metadata": str(metadata_path),
                "faiss_index": str(ARTIFACTS_DIR / "faiss" / "faiss.index"),
                "candidate_lookup": str(ARTIFACTS_DIR / "faiss" / "candidate_lookup.pkl"),
                "embedding_metadata": str(ARTIFACTS_DIR / "faiss" / "embedding_metadata.pkl"),
            },
        }
        return report

    def _build_metadata_yaml(
        self,
        *,
        parsed_job: ParsedJob,
        submission_csv_path: Path,
        ranking_json_path: Path,
        pipeline_report_path: Path,
        artifact_paths: Dict[str, str],
        configuration: Dict[str, Any],
        ai_tools_used: List[str],
        ai_usage_summary: str,
        methodology_summary: str,
        pipeline_summary: str,
        compute_summary: Dict[str, Any],
    ) -> str:
        job_title = parsed_job.job_description.title
        compute = {
            "platform": compute_summary.get("platform", platform.platform()),
            "cpu_cores": compute_summary.get("cpu_cores", os.cpu_count() or 1),
            "ram_gb": compute_summary.get("ram_gb"),
            "python_version": compute_summary.get("python_version", sys.version.split()[0]),
            "os": compute_summary.get("os", platform.platform()),
            "uses_gpu_for_inference": compute_summary.get("uses_gpu_for_inference", False),
            "has_network_during_ranking": compute_summary.get("has_network_during_ranking", False),
            "pre_computation_required": compute_summary.get("pre_computation_required", True),
            "pre_computation_time_minutes": compute_summary.get("pre_computation_time_minutes", 0),
        }

        artifact_summary = {
            "submission_csv": str(submission_csv_path),
            "ranking_json": str(ranking_json_path),
            "pipeline_report": str(pipeline_report_path),
            "faiss_index": artifact_paths.get("faiss_index", str(ARTIFACTS_DIR / "faiss" / "faiss.index")),
            "candidate_lookup": artifact_paths.get("candidate_lookup", str(ARTIFACTS_DIR / "faiss" / "candidate_lookup.pkl")),
            "embedding_metadata": artifact_paths.get("embedding_metadata", str(ARTIFACTS_DIR / "faiss" / "embedding_metadata.pkl")),
        }

        def dump_list(items: Sequence[str], indent: int = 2) -> str:
            pad = " " * indent
            return "\n".join(f"{pad}- {item}" for item in items)

        yaml_lines = [
            "team_name: \"\"",
            "",
            "primary_contact:",
            "  name: \"\"",
            "  email: \"\"",
            "  phone: \"\"",
            "",
            "team_members: []",
            "",
            f"github_repo: \"\"",
            f"sandbox_link: \"\"",
            f"reproduce_command: \"python rank.py --job job_description.json --top-k 100 --output outputs/submission.csv\"",
            "",
            "compute:",
        ]
        for key, value in compute.items():
            yaml_lines.append(f"  {key}: {self._yaml_scalar(value)}")
        yaml_lines.extend([
            "",
            "ai_tools_used:",
        ])
        yaml_lines.extend(dump_list(ai_tools_used, indent=2).splitlines())
        yaml_lines.extend([
            "",
            "ai_usage_summary: |",
        ])
        yaml_lines.extend([f"  {line}" if line else "" for line in ai_usage_summary.strip().splitlines()])
        yaml_lines.extend([
            "",
            "methodology_summary: |",
        ])
        yaml_lines.extend([f"  {line}" if line else "" for line in methodology_summary.strip().splitlines()])
        yaml_lines.extend([
            "",
            "pipeline_summary: |",
        ])
        yaml_lines.extend([f"  {line}" if line else "" for line in pipeline_summary.strip().splitlines()])
        yaml_lines.extend([
            "",
            "artifact_summary:",
        ])
        for key, value in artifact_summary.items():
            yaml_lines.append(f"  {key}: {self._yaml_scalar(value)}")
        yaml_lines.extend([
            "",
            "configuration:",
            f"  job_id: {self._yaml_scalar(parsed_job.job_description.job_id)}",
            f"  job_title: {self._yaml_scalar(job_title)}",
            f"  top_k: {self._yaml_scalar(configuration.get('top_k', 100))}",
            f"  output_csv: {self._yaml_scalar(str(submission_csv_path))}",
            f"  ranking_json: {self._yaml_scalar(str(ranking_json_path))}",
            f"  pipeline_report: {self._yaml_scalar(str(pipeline_report_path))}",
            "",
            "declarations:",
            "  read_submission_spec: true",
            "  code_is_original_work: true",
            "  no_collusion: true",
            "  honeypot_check_done: false",
            "  reproduction_tested: true",
        ])

        return "\n".join(yaml_lines).strip() + "\n"

    def _default_methodology_summary(self) -> str:
        return (
            "Deterministic retrieval-plus-ranking pipeline. FAISS retrieves the top-K candidates from precomputed embeddings, then HybridRanker combines semantic similarity with CareerScorer, SkillScorer, and BehaviorScorer. Submission rows are generated from live candidate evidence and validated before export."
        )

    def _default_pipeline_summary(self) -> str:
        return (
            "The pipeline loads existing FAISS artifacts, parses the job description, retrieves top-K candidates, reranks them with the hybrid scorer stack, generates submission CSV and JSON artifacts, validates the output, and writes the pipeline report plus metadata YAML."
        )

    def _default_compute_summary(self) -> Dict[str, Any]:
        return {
            "platform": platform.platform(),
            "cpu_cores": os.cpu_count() or 1,
            "ram_gb": self._ram_gb(),
            "python_version": sys.version.split()[0],
            "os": platform.platform(),
            "uses_gpu_for_inference": False,
            "has_network_during_ranking": False,
            "pre_computation_required": True,
            "pre_computation_time_minutes": 0,
        }

    def _yaml_scalar(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        if text == "":
            return '""'
        if any(ch in text for ch in [":", "#", "\n", "\r", "\t", '"', "{"]):
            return json.dumps(text)
        return f'"{text}"'

    def _ram_gb(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            return round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            return None

    def _now(self) -> float:
        import time

        return time.perf_counter()
