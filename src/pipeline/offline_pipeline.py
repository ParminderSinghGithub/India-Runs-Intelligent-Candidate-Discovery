"""Offline pipeline for building candidate search index."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from src.config import FAISS_DIR, PROJECT_ROOT
from src.embeddings.embedder import EmbeddingEngine
from src.models.offline_index_result import OfflineIndexResult
from src.parser.candidate_parser import CandidateParser
from src.retrieval.document_builder import RetrievalDocumentBuilder
from src.retrieval.retriever import Retriever
from src.utils.exceptions import ValidationError
from src.utils.logging import stage_log

logger = logging.getLogger(__name__)


class OfflineIndexBuilder:
    """Offline pipeline for building candidate search index.

    Orchestrates the complete preprocessing pipeline:
    - Load candidates from JSONL
    - Parse candidates
    - Build retrieval documents
    - Generate embeddings
    - Build FAISS index
    - Persist artifacts

    This is the single entry point for offline preprocessing.
    """

    def __init__(
        self,
        output_directory: Optional[Path] = None,
    ):
        """Initialize the offline index builder.

        Args:
            output_directory: Directory for output artifacts. Defaults to config.
        """
        self.output_directory = output_directory or FAISS_DIR

        # Initialize components
        self.parser = CandidateParser()
        self.document_builder = RetrievalDocumentBuilder()
        self.embedding_engine = EmbeddingEngine()
        self.retriever: Optional[Retriever] = None

    def build_candidate_index(
        self,
        candidates_jsonl_path: Path,
        max_candidates: Optional[int] = None,
        batch_size: Optional[int] = None,
        force_rebuild: bool = False,
    ) -> OfflineIndexResult:
        """Build candidate search index from JSONL file.

        Args:
            candidates_jsonl_path: Path to candidates JSONL file.
            max_candidates: Maximum number of candidates to process (for testing).
            batch_size: Batch size for embedding generation.
            force_rebuild: Force rebuild even if artifacts exist.

        Returns:
            OfflineIndexResult with statistics and artifact locations.

        Raises:
            FileNotFoundError: If candidates file does not exist.
            ValidationError: If validation checks fail.
            RuntimeError: If pipeline execution fails.
        """
        start_time = time.time()
        logger.info(
            "[OfflineIndexBuilder] START build_candidate_index -- source: %s",
            candidates_jsonl_path,
        )

        # Validate input
        self._validate_input(candidates_jsonl_path)

        with stage_log(logger, "Loading candidates"):
            candidates_data = self._load_candidates(candidates_jsonl_path)
        total_candidates = len(candidates_data)
        logger.info("Loaded %d raw candidate records", total_candidates)

        # Apply max_candidates limit
        if max_candidates is not None and max_candidates > 0:
            candidates_data = candidates_data[:max_candidates]
            logger.info("Limited to %d candidates for testing", max_candidates)

        with stage_log(logger, "Parsing candidates", count_label=f"{len(candidates_data)} candidates"):
            parse_start = time.time()
            candidates = self._parse_candidates(candidates_data)
            parse_time = time.time() - parse_start
        logger.info("Parsed %d candidates successfully", len(candidates))

        with stage_log(logger, "Building retrieval documents", count_label=f"{len(candidates)} candidates"):
            doc_start = time.time()
            retrieval_docs = self._build_retrieval_documents(candidates)
            doc_time = time.time() - doc_start
        logger.info("Built %d retrieval documents", len(retrieval_docs))

        with stage_log(logger, "Generating embeddings", count_label=f"{len(retrieval_docs)} documents"):
            embed_start = time.time()
            embeddings = self._generate_embeddings(retrieval_docs, batch_size)
            embedding_time = time.time() - embed_start
        logger.info(
            "Generated embeddings: shape=%s, dim=%d",
            embeddings.shape,
            embeddings.shape[1] if len(embeddings.shape) > 1 else -1,
        )

        with stage_log(logger, "Building FAISS index", count_label=f"{len(retrieval_docs)} vectors"):
            index_start = time.time()
            self._build_faiss_index(retrieval_docs, force_rebuild)
            index_build_time = time.time() - index_start
        logger.info("FAISS index size: %d", self.retriever.get_index_size())

        logger.info("Collecting artifact paths")
        artifacts = self._get_artifact_paths()

        # Validate results
        self._validate_results(
            len(candidates),
            embeddings.shape[0],
            self.retriever.get_index_size(),
        )

        total_time = time.time() - start_time

        # Build result
        result = OfflineIndexResult(
            total_candidates=total_candidates,
            processed_candidates=len(candidates),
            embedding_dimension=embeddings.shape[1],
            index_size=self.retriever.get_index_size(),
            processing_time_seconds=total_time,
            embedding_time_seconds=embedding_time,
            index_build_time_seconds=index_build_time,
            artifacts=artifacts,
            statistics={
                "parse_time_seconds": parse_time,
                "document_generation_time_seconds": doc_time,
                "average_embedding_time_seconds": embedding_time / len(candidates) if candidates else 0,
            },
        )

        logger.info(
            "[OfflineIndexBuilder] END build_candidate_index -- processed %d candidates in %.2fs",
            len(candidates),
            total_time,
        )
        return result

    def _validate_input(self, candidates_jsonl_path: Path) -> None:
        """Validate input file.

        Args:
            candidates_jsonl_path: Path to candidates JSONL file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValidationError: If file is not a JSONL file.
        """
        if not candidates_jsonl_path.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_jsonl_path}")

        if candidates_jsonl_path.suffix != ".jsonl":
            raise ValidationError(f"Expected .jsonl file, got: {candidates_jsonl_path.suffix}")

    def _load_candidates(self, candidates_jsonl_path: Path) -> list:
        """Load candidates from JSONL file.

        Args:
            candidates_jsonl_path: Path to candidates JSONL file.

        Returns:
            List of candidate dictionaries.

        Raises:
            RuntimeError: If file is empty or corrupted.
        """
        candidates_data = []

        with open(candidates_jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    candidate = json.loads(line)
                    candidates_data.append(candidate)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        if not candidates_data:
            raise RuntimeError("No valid candidates found in file")

        return candidates_data

    def _parse_candidates(self, candidates_data: list) -> list:
        """Parse candidate data using CandidateParser.

        Args:
            candidates_data: List of candidate dictionaries.

        Returns:
            List of parsed Candidate objects.
        """
        candidates = []
        for cand_data in candidates_data:
            try:
                candidate = self.parser.from_dict(cand_data)
                candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Failed to parse candidate: {e}")
                continue

        if not candidates:
            raise RuntimeError("No candidates successfully parsed")

        return candidates

    def _build_retrieval_documents(self, candidates: list) -> list:
        """Build retrieval documents using RetrievalDocumentBuilder.

        Args:
            candidates: List of Candidate objects.

        Returns:
            List of RetrievalDocument objects.
        """
        retrieval_docs = []
        for candidate in candidates:
            try:
                doc = self.document_builder.build(candidate)
                retrieval_docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to build document for candidate: {e}")
                continue

        if not retrieval_docs:
            raise RuntimeError("No retrieval documents successfully built")

        return retrieval_docs

    def _generate_embeddings(self, retrieval_docs: list, batch_size: Optional[int]) -> object:
        """Generate embeddings using EmbeddingEngine.

        Args:
            retrieval_docs: List of RetrievalDocument objects.
            batch_size: Batch size for embedding generation.

        Returns:
            Embedding matrix as numpy array.
        """
        # Update batch size if provided
        if batch_size is not None:
            self.embedding_engine.batch_size = batch_size

        # Load model
        self.embedding_engine.load_model()

        # Generate embeddings
        embeddings = self.embedding_engine.embed_documents(retrieval_docs, show_progress=True)

        return embeddings

    def _build_faiss_index(self, retrieval_docs: list, force_rebuild: bool) -> None:
        """Build FAISS index using Retriever.

        Args:
            retrieval_docs: List of RetrievalDocument objects.
            force_rebuild: Force rebuild even if artifacts exist.
        """
        # Initialize retriever
        self.retriever = Retriever(
            embedding_engine=self.embedding_engine,
            force_rebuild=force_rebuild,
        )

        # Build index
        self.retriever.build_index(retrieval_docs, show_progress=True)

    def _get_artifact_paths(self) -> dict:
        """Get paths to generated artifacts.

        Returns:
            Dictionary mapping artifact names to paths.
        """
        return {
            "faiss_index": FAISS_DIR / "faiss.index",
            "candidate_lookup": FAISS_DIR / "candidate_lookup.pkl",
            "embedding_metadata": FAISS_DIR / "embedding_metadata.pkl",
        }

    def _validate_results(
        self,
        num_candidates: int,
        num_embeddings: int,
        index_size: int,
    ) -> None:
        """Validate pipeline results.

        Args:
            num_candidates: Number of candidates processed.
            num_embeddings: Number of embeddings generated.
            index_size: Size of FAISS index.

        Raises:
            ValidationError: If validation checks fail.
        """
        # Check embeddings generated
        if num_embeddings == 0:
            raise ValidationError("No embeddings generated")

        # Check embedding dimension
        embedding_dim = self.embedding_engine.get_embedding_dimension()
        if embedding_dim <= 0:
            raise ValidationError(f"Invalid embedding dimension: {embedding_dim}")

        # Check index size matches processed candidates
        if index_size != num_candidates:
            raise ValidationError(
                f"Index size ({index_size}) does not match processed candidates ({num_candidates})"
            )

        # Check artifact files exist
        artifacts = self._get_artifact_paths()
        for name, path in artifacts.items():
            if not path.exists():
                raise ValidationError(f"Artifact file not found: {path}")

        # Check candidate lookup size
        if self.retriever is not None:
            if len(self.retriever.candidate_lookup) != index_size:
                raise ValidationError(
                    f"Candidate lookup size ({len(self.retriever.candidate_lookup)}) "
                    f"does not match index size ({index_size})"
                )

            if not self.retriever.embedding_metadata:
                raise ValidationError("Embedding metadata is empty")

        logger.info("✓ All validation checks passed")
