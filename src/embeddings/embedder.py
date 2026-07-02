"""Embedding engine for semantic search using SentenceTransformers."""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_ENABLED,
    EMBEDDING_DEVICE,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
)
from src.models.retrieval_document import RetrievalDocument

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Production-quality embedding engine for semantic search.

    Uses BAAI/bge-base-en-v1.5 model from SentenceTransformers.
    Supports GPU acceleration with automatic CPU fallback.
    Implements caching to avoid recomputation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
    ):
        """Initialize the embedding engine.

        Args:
            model_name: Model name from HuggingFace. Defaults to config.
            batch_size: Batch size for encoding. Defaults to config.
            device: Device to use ("auto", "cuda", "cpu"). Defaults to config.
            normalize: Whether to normalize embeddings. Defaults to config.
            cache_enabled: Whether to enable caching. Defaults to config.
        """
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or EMBEDDING_BATCH_SIZE
        self.device = device or EMBEDDING_DEVICE
        self.normalize = normalize if normalize is not None else EMBEDDING_NORMALIZE
        self.cache_enabled = cache_enabled if cache_enabled is not None else EMBEDDING_CACHE_ENABLED

        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

        # Ensure embeddings directory exists
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> SentenceTransformer:
        """Load the embedding model.

        Returns:
            Loaded SentenceTransformer model.

        Raises:
            RuntimeError: If model fails to load.
        """
        if self._model is not None:
            logger.debug("Model already loaded")
            return self._model

        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            # Determine device
            if self.device == "auto":
                try:
                    import torch

                    self._actual_device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    self._actual_device = "cpu"
            else:
                self._actual_device = self.device

            logger.info(f"Using device: {self._actual_device}")

            # Load model from local cache first to avoid unnecessary network checks.
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self._actual_device,
                    local_files_only=True,
                )
            except Exception:
                logger.info("Local model cache unavailable; falling back to online resolution")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self._actual_device,
                )

            # Get embedding dimension
            if hasattr(self._model, "get_embedding_dimension"):
                self._embedding_dim = self._model.get_embedding_dimension()
            else:
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")

            return self._model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

    def embed_document(self, retrieval_document: RetrievalDocument) -> np.ndarray:
        """Embed a single retrieval document.

        Args:
            retrieval_document: RetrievalDocument to embed.

        Returns:
            Embedding vector as numpy array.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            self.load_model()

        logger.debug(f"Embedding document: {retrieval_document.candidate_id}")

        # Extract document text
        text = retrieval_document.document

        # Encode
        embedding = self._model.encode(
            text,
            batch_size=1,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embedding

    def embed_documents(
        self,
        retrieval_documents: List[RetrievalDocument],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed multiple retrieval documents.

        Args:
            retrieval_documents: List of RetrievalDocument objects.
            show_progress: Whether to show progress bar.

        Returns:
            Embedding matrix as numpy array (n_docs, embedding_dim).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            self.load_model()

        if not retrieval_documents:
            logger.warning("No documents to embed")
            return np.array([]).reshape(0, self._embedding_dim or 768)

        logger.info(f"Embedding {len(retrieval_documents)} documents")
        start_time = time.time()

        # Extract document texts
        texts = [doc.document for doc in retrieval_documents]

        # Encode with progress bar
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Embedded {len(retrieval_documents)} documents in {elapsed_time:.2f}s")

        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed a query text.

        Args:
            query_text: Query text to embed.

        Returns:
            Embedding vector as numpy array.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            self.load_model()

        logger.debug(f"Embedding query: {query_text[:50]}...")

        # Encode
        embedding = self._model.encode(
            query_text,
            batch_size=1,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embedding

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        candidate_ids: List[str],
        metadata: Optional[List[dict]] = None,
        cache_name: str = "candidate_embeddings",
    ) -> Path:
        """Save embeddings to disk in numpy format.

        Args:
            embeddings: Embedding matrix (n_docs, embedding_dim).
            candidate_ids: List of candidate IDs.
            metadata: Optional list of metadata dictionaries.
            cache_name: Name for the cache file.

        Returns:
            Path to saved cache file.

        Raises:
            ValueError: If embeddings and candidate_ids length mismatch.
        """
        if len(embeddings) != len(candidate_ids):
            raise ValueError(
                f"Embeddings shape {embeddings.shape} does not match "
                f"candidate_ids length {len(candidate_ids)}"
            )

        cache_path = EMBEDDINGS_DIR / f"{cache_name}.npz"

        logger.info(f"Saving embeddings to {cache_path}")

        # Save to numpy format
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            candidate_ids=np.array(candidate_ids, dtype=object),
            metadata=np.array(metadata, dtype=object) if metadata else np.array(None),
        )

        logger.info(f"Saved {len(candidate_ids)} embeddings to {cache_path}")

        return cache_path

    def load_embeddings(
        self,
        cache_name: str = "candidate_embeddings",
    ) -> Tuple[np.ndarray, List[str], Optional[List[dict]]]:
        """Load embeddings from disk.

        Args:
            cache_name: Name of the cache file.

        Returns:
            Tuple of (embeddings, candidate_ids, metadata).

        Raises:
            FileNotFoundError: If cache file does not exist.
        """
        cache_path = EMBEDDINGS_DIR / f"{cache_name}.npz"

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        logger.info(f"Loading embeddings from {cache_path}")

        # Load from numpy format
        data = np.load(cache_path, allow_pickle=True)

        embeddings = data["embeddings"]
        candidate_ids = data["candidate_ids"].tolist()
        metadata = data["metadata"].tolist() if data["metadata"] is not None else None

        logger.info(f"Loaded {len(candidate_ids)} embeddings from {cache_path}")

        return embeddings, candidate_ids, metadata

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model.

        Returns:
            Embedding dimension.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            self.load_model()

        return self._embedding_dim

    def cache_exists(self, cache_name: str = "candidate_embeddings") -> bool:
        """Check if a cache file exists.

        Args:
            cache_name: Name of the cache file.

        Returns:
            True if cache exists, False otherwise.
        """
        cache_path = EMBEDDINGS_DIR / f"{cache_name}.npz"
        return cache_path.exists()
