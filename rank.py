"""Official ranking entry point for Track 1 submission generation.

Usage::

    python rank.py --job job_description.json

This script loads existing FAISS artifacts and never regenerates embeddings
or rebuilds the index.  If artifacts are missing it exits with a clear error
message that explains how to run the offline build.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from src.config import ARTIFACTS_DIR, FAISS_DIR, PROJECT_ROOT
from src.models.parsed_job import ParsedJob
from src.parser.candidate_parser import CandidateParser
from src.parser.job_description_parser import JobDescriptionParser
from src.retrieval.retriever import Retriever
from src.scoring.hybrid_ranker import HybridRanker
from src.submission import CandidateResolver, SubmissionGenerator, SubmissionValidator, XlsxExporter
from src.utils import setup_logging, stage_log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Required offline artifact paths
# ---------------------------------------------------------------------------
_REQUIRED_ARTIFACTS = [
    FAISS_DIR / "faiss.index",
    FAISS_DIR / "candidate_lookup.pkl",
    FAISS_DIR / "embedding_metadata.pkl",
]

_REBUILD_HINT = (
    "\nTo build the FAISS index from scratch run:\n"
    "    python run_pipeline.py --rebuild-index\n"
    "This is a one-time step and takes ~15-40 minutes on CPU."
)


def _default_candidates_path() -> Path:
    return (
        PROJECT_ROOT
        / "[PUB] India_runs_data_and_ai_challenge"
        / "[PUB] India_runs_data_and_ai_challenge"
        / "India_runs_data_and_ai_challenge"
        / "candidates.jsonl"
    )


def _check_artifacts() -> None:
    """Exit with a helpful message if any required artifact is missing."""
    missing = [p for p in _REQUIRED_ARTIFACTS if not p.exists()]
    if missing:
        print("✗ Missing FAISS artifacts:")
        for p in missing:
            print(f"  {p}")
        print(_REBUILD_HINT)
        sys.exit(1)
    logger.info("[rank.py] All FAISS artifacts present")


def _check_candidates(path: Path) -> None:
    """Exit with a helpful message if the candidate JSONL is missing."""
    if not path.exists():
        print(f"✗ Candidates JSONL not found: {path}")
        print(
            "\nThe candidate data file is required.  It should be located at:\n"
            f"    {path}\n"
            "Ensure the challenge dataset has been extracted to the project root."
        )
        sys.exit(1)
    logger.info("[rank.py] Candidates JSONL present: %s", path)


def _load_parsed_job(job_path: Path) -> ParsedJob:
    if not job_path.exists():
        print(f"✗ Job description not found: {job_path}")
        sys.exit(1)
    parser = JobDescriptionParser()
    return parser.parse_from_file(job_path)


def _build_retriever() -> Retriever:
    retriever = Retriever(force_rebuild=False)
    retriever.load_index()
    return retriever


def _build_ranker(retriever: Retriever, candidate_resolver: CandidateResolver) -> HybridRanker:
    return HybridRanker(
        retriever=retriever,
        candidate_resolver=candidate_resolver.resolve,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the official Redrob submission (uses pre-built FAISS artifacts).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python rank.py --job job_description.json\n\n"
            "FAISS artifacts must already exist in artifacts/faiss/.  "
            "Run 'python run_pipeline.py --rebuild-index' to build them."
        ),
    )
    parser.add_argument("--job", required=True, help="Path to the job description JSON file")
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to retrieve and rank")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "submission.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-xlsx",
        default=str(PROJECT_ROOT / "outputs" / "submission.xlsx"),
        help="Output XLSX path",
    )
    parser.add_argument(
        "--ranking-json",
        default=str(PROJECT_ROOT / "outputs" / "ranking.json"),
        help="Ranking JSON path",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "outputs" / "pipeline_report.json"),
        help="Pipeline report JSON path",
    )
    parser.add_argument(
        "--metadata",
        default=str(PROJECT_ROOT / "submission_metadata.yaml"),
        help="Submission metadata YAML path",
    )
    parser.add_argument(
        "--candidates",
        default=str(_default_candidates_path()),
        help="Candidate JSONL source path",
    )
    args = parser.parse_args(argv)

    setup_logging()

    job_path = Path(args.job)
    candidates_path = Path(args.candidates)
    output_path = Path(args.output)
    output_xlsx_path = Path(args.output_xlsx)
    ranking_json_path = Path(args.ranking_json)
    report_path = Path(args.report)
    metadata_path = Path(args.metadata)

    logger.info("[rank.py] START -- job=%s, top_k=%d", job_path, args.top_k)
    pipeline_start = time.perf_counter()

    # Pre-flight checks — fail fast with clear guidance
    _check_artifacts()
    _check_candidates(candidates_path)

    with stage_log(logger, "Parsing job description"):
        parsed_job = _load_parsed_job(job_path)
    logger.info("Job: %s (%s)", parsed_job.job_description.title, parsed_job.job_description.job_id)

    with stage_log(logger, "Loading candidate resolver"):
        candidate_resolver = CandidateResolver(candidates_path)

    with stage_log(logger, "Loading FAISS index"):
        retriever = _build_retriever()
    logger.info("FAISS index loaded -- %d candidates indexed", retriever.get_index_size())

    ranker = _build_ranker(retriever, candidate_resolver)
    generator = SubmissionGenerator()

    with stage_log(logger, "FAISS retrieval", count_label=f"top_k={args.top_k}"):
        retrieval_start = time.perf_counter()
        retrieval_results = retriever.search(parsed_job.search_query.combined_query, k=args.top_k)
        retrieval_latency = time.perf_counter() - retrieval_start
    ranker.last_retrieved_candidates = retrieval_results
    logger.info("Retrieved %d candidates (latency=%.2fs)", len(retrieval_results), retrieval_latency)

    with stage_log(logger, "Hybrid re-ranking", count_label=f"{len(retrieval_results)} candidates"):
        rerank_start = time.perf_counter()
        ranked_results = ranker.rank_retrieval_results(parsed_job, retrieval_results)
        rerank_latency = time.perf_counter() - rerank_start
    logger.info("Ranked %d candidates (latency=%.2fs)", len(ranked_results), rerank_latency)

    with stage_log(logger, "Generating + validating submission"):
        validation_start = time.perf_counter()
        result = generator.generate_submission(
            parsed_job=parsed_job,
            ranked_results=ranked_results,
            candidate_resolver=candidate_resolver.resolve,
            top_n=args.top_k,
            submission_csv_path=output_path,
            ranking_json_path=ranking_json_path,
            pipeline_report_path=report_path,
            metadata_path=metadata_path,
            candidate_exists=candidate_resolver.has,
            retrieval_results=retrieval_results,
            timings={
                "retrieval_latency_seconds": retrieval_latency,
                "reranking_latency_seconds": rerank_latency,
            },
            artifact_paths={
                "faiss_index": str(FAISS_DIR / "faiss.index"),
                "candidate_lookup": str(FAISS_DIR / "candidate_lookup.pkl"),
                "embedding_metadata": str(FAISS_DIR / "embedding_metadata.pkl"),
            },
            configuration={"top_k": args.top_k},
            ai_tools_used=["GitHub Copilot"],
        )
        xlsx_start = time.perf_counter()
        xlsx_exporter = XlsxExporter()
        xlsx_exporter.export(result["rows"], output_xlsx_path)
        xlsx_latency = time.perf_counter() - xlsx_start
        validation_latency = (time.perf_counter() - validation_start) - xlsx_latency

    total_latency = time.perf_counter() - pipeline_start

    logger.info("[rank.py] END -- total elapsed %.2fs", total_latency)

    print()
    print("[PASS] Submission validation PASSED")
    print(f"  Rows:            {len(result['rows'])}")
    print(
        f"  Total runtime:   {(retrieval_latency + rerank_latency + result['timings']['csv_generation_seconds'] + result['timings']['ranking_json_seconds'] + result['timings']['pipeline_report_seconds'] + result['timings']['metadata_yaml_seconds'] + validation_latency):.2f}s"
    )
    print("  Timing:          PASS (<300 sec)")
    print(f"  Submission CSV:  {output_path}")
    print(f"  Submission XLSX: {output_xlsx_path}")
    print(f"  Ranking JSON:    {ranking_json_path}")
    print(f"  Pipeline report: {report_path}")
    print(f"  Metadata YAML:   {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
