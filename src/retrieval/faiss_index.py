"""FAISS index management for semantic retrieval."""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np

from src.config import FAISS_DIR, FAISS_INDEX_TYPE

logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS index wrapper for efficient similarity search.

    Uses IndexFlatIP for inner product search on normalized embeddings.
    Provides methods for building, saving, loading, and searching.
    """

    def __init__(self, embedding_dim: int, index_type: Optional[str] = None):
        """Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embeddings.
            index_type: Type of FAISS index. Defaults to config.
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type or FAISS_INDEX_TYPE
        self.index: Optional[faiss.Index] = None
        self._is_built = False

        # Ensure FAISS directory exists
        FAISS_DIR.mkdir(parents=True, exist_ok=True)

    def build(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Embedding matrix (n_docs, embedding_dim).

        Raises:
            ValueError: If embeddings dimension mismatch.
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embeddings dimension {embeddings.shape[1]} does not match "
                f"expected dimension {self.embedding_dim}"
            )

        logger.info(f"Building {self.index_type} index with {embeddings.shape[0]} vectors")

        # Create index based on type
        if self.index_type == "IndexFlatIP":
            # Inner product index for normalized embeddings
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexFlatL2":
            # L2 distance index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))

        self._is_built = True
        logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")

    def save(self, index_path: Path) -> None:
        """Save FAISS index to disk.

        Args:
            index_path: Path to save index file.

        Raises:
            RuntimeError: If index has not been built.
        """
        if not self._is_built:
            raise RuntimeError("Cannot save index: index has not been built")

        logger.info(f"Saving FAISS index to {index_path}")

        # Ensure parent directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, str(index_path))

        logger.info(f"Index saved successfully to {index_path}")

    def load(self, index_path: Path) -> None:
        """Load FAISS index from disk.

        Args:
            index_path: Path to index file.

        Raises:
            FileNotFoundError: If index file does not exist.
        """
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        logger.info(f"Loading FAISS index from {index_path}")

        # Load index
        self.index = faiss.read_index(str(index_path))

        # Update embedding dimension from loaded index
        self.embedding_dim = self.index.d

        self._is_built = True
        logger.info(f"Index loaded successfully. Total vectors: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search index for top-k similar vectors.

        Args:
            query_embedding: Query embedding vector (embedding_dim,) or (n_queries, embedding_dim).
            k: Number of results to return.

        Returns:
            Tuple of (similarities, indices) where:
                - similarities: (n_queries, k) array of similarity scores
                - indices: (n_queries, k) array of vector indices

        Raises:
            RuntimeError: If index has not been built.
        """
        if not self._is_built:
            raise RuntimeError("Cannot search: index has not been built")

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure correct dtype
        query_embedding = query_embedding.astype(np.float32)

        # Search
        similarities, indices = self.index.search(query_embedding, k)

        return similarities, indices

    def get_index_size(self) -> int:
        """Get the number of vectors in the index.

        Returns:
            Number of vectors in the index.

        Raises:
            RuntimeError: If index has not been built.
        """
        if not self._is_built:
            raise RuntimeError("Cannot get size: index has not been built")

        return self.index.ntotal

    def is_built(self) -> bool:
        """Check if index has been built.

        Returns:
            True if index is built, False otherwise.
        """
        return self._is_built
