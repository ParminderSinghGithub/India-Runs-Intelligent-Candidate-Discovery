"""Retriever for semantic search using FAISS index."""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.config import FAISS_DIR
from src.embeddings.embedder import EmbeddingEngine
from src.models.retrieval_document import RetrievalDocument
from src.retrieval.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class Retriever:
    """Semantic retriever using FAISS index for candidate search.

    Orchestrates the complete retrieval pipeline:
    - Build index from RetrievalDocuments
    - Load existing artifacts
    - Embed queries
    - Search top-k candidates
    - Return results with metadata
    """

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        force_rebuild: bool = False,
    ):
        """Initialize the retriever.

        Args:
            embedding_engine: EmbeddingEngine instance. If None, creates new instance.
            force_rebuild: Force rebuild of index even if artifacts exist.
        """
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.force_rebuild = force_rebuild

        self.faiss_index: Optional[FaissIndex] = None
        self.candidate_lookup: Dict[int, str] = {}  # index -> candidate_id
        self.embedding_metadata: Dict[str, Dict[str, Any]] = {}  # candidate_id -> metadata

        # Artifact paths
        self.index_path = FAISS_DIR / "faiss.index"
        self.lookup_path = FAISS_DIR / "candidate_lookup.pkl"
        self.metadata_path = FAISS_DIR / "embedding_metadata.pkl"

        # Ensure FAISS directory exists
        FAISS_DIR.mkdir(parents=True, exist_ok=True)

    def build_index(
        self,
        retrieval_documents: List[RetrievalDocument],
        show_progress: bool = True,
    ) -> None:
        """Build FAISS index from retrieval documents.

        Args:
            retrieval_documents: List of RetrievalDocument objects.
            show_progress: Whether to show progress bar.

        Raises:
            RuntimeError: If artifacts exist and force_rebuild is False.
        """
        # Check if artifacts already exist
        if not self.force_rebuild and self._artifacts_exist():
            logger.info("Artifacts already exist. Use load_index() or set force_rebuild=True.")
            return

        logger.info(f"Building index from {len(retrieval_documents)} documents")

        # Load embedding model
        self.embedding_engine.load_model()
        embedding_dim = self.embedding_engine.get_embedding_dimension()

        # Initialize FAISS index
        self.faiss_index = FaissIndex(embedding_dim=embedding_dim)

        # Generate embeddings
        embeddings = self.embedding_engine.embed_documents(
            retrieval_documents, show_progress=show_progress
        )

        # Build candidate lookup and metadata
        self.candidate_lookup = {}
        self.embedding_metadata = {}

        for idx, doc in enumerate(tqdm(retrieval_documents, desc="Building lookup", disable=not show_progress)):
            self.candidate_lookup[idx] = doc.candidate_id
            self.embedding_metadata[doc.candidate_id] = doc.metadata

        # Build FAISS index
        self.faiss_index.build(embeddings)

        # Save artifacts
        self._save_artifacts()

        logger.info(f"Index built successfully with {len(self.candidate_lookup)} candidates")

    def load_index(self) -> None:
        """Load existing FAISS index and artifacts.

        Raises:
            FileNotFoundError: If artifacts do not exist.
        """
        if not self._artifacts_exist():
            raise FileNotFoundError("FAISS artifacts do not exist. Build index first.")

        logger.info("Loading FAISS index and artifacts")

        # Load FAISS index
        self.faiss_index = FaissIndex(embedding_dim=0)  # Dimension will be set on load
        self.faiss_index.load(self.index_path)

        # Load candidate lookup
        with open(self.lookup_path, "rb") as f:
            self.candidate_lookup = pickle.load(f)

        # Load embedding metadata
        with open(self.metadata_path, "rb") as f:
            self.embedding_metadata = pickle.load(f)

        logger.info(f"Loaded index with {len(self.candidate_lookup)} candidates")

    def search(
        self,
        query_text: str,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for top-k similar candidates.

        Args:
            query_text: Query text to search for.
            k: Number of results to return.

        Returns:
            List of result dictionaries containing:
                - rank: Result rank (1-indexed)
                - candidate_id: Candidate identifier
                - similarity: Similarity score
                - metadata: Candidate metadata

        Raises:
            RuntimeError: If index has not been built or loaded.
        """
        if self.faiss_index is None or not self.faiss_index.is_built():
            raise RuntimeError("Index has not been built or loaded. Call build_index() or load_index() first.")

        logger.debug(f"Searching for top-{k} candidates with query: {query_text[:50]}...")

        # Load embedding model if needed
        self.embedding_engine.load_model()

        # Embed query
        query_embedding = self.embedding_engine.embed_query(query_text)

        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, k=k)

        # Format results
        results = []
        for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            candidate_id = self.candidate_lookup.get(int(idx))
            if candidate_id is None:
                logger.warning(f"No candidate found for index {idx}")
                continue

            result = {
                "rank": rank,
                "candidate_id": candidate_id,
                "similarity": float(similarity),
                "metadata": self.embedding_metadata.get(candidate_id, {}),
            }
            results.append(result)

        logger.debug(f"Found {len(results)} results")
        return results

    def _artifacts_exist(self) -> bool:
        """Check if FAISS artifacts exist.

        Returns:
            True if all artifacts exist, False otherwise.
        """
        return (
            self.index_path.exists()
            and self.lookup_path.exists()
            and self.metadata_path.exists()
        )

    def _save_artifacts(self) -> None:
        """Save FAISS artifacts to disk."""
        logger.info("Saving FAISS artifacts")

        # Save FAISS index
        self.faiss_index.save(self.index_path)

        # Save candidate lookup
        with open(self.lookup_path, "wb") as f:
            pickle.dump(self.candidate_lookup, f)

        # Save embedding metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.embedding_metadata, f)

        logger.info("FAISS artifacts saved successfully")

    def get_index_size(self) -> int:
        """Get the number of candidates in the index.

        Returns:
            Number of candidates in the index.

        Raises:
            RuntimeError: If index has not been built or loaded.
        """
        if self.faiss_index is None or not self.faiss_index.is_built():
            raise RuntimeError("Index has not been built or loaded.")

        return self.faiss_index.get_index_size()
