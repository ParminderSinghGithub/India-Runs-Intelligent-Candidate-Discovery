"""Result dataclass for offline index building."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class OfflineIndexResult:
    """Result of offline index building pipeline.

    Contains statistics, timing information, and artifact locations.
    """

    total_candidates: int
    """Total number of candidates loaded from source."""

    processed_candidates:int
    """Number of candidates successfully processed."""

    embedding_dimension: int
    """Dimension of embedding vectors."""

    index_size: int
    """Number of vectors in FAISS index."""

    processing_time_seconds: float
    """Total pipeline execution time in seconds."""

    embedding_time_seconds: float
    """Time spent generating embeddings in seconds."""

    index_build_time_seconds: float
    """Time spent building FAISS index in seconds."""

    artifacts: Dict[str, Path]
    """Dictionary mapping artifact names to their file paths."""

    statistics: Dict[str, Any] = field(default_factory=dict)
    """Additional statistics about the pipeline run."""

    def __str__(self) -> str:
        """Return a formatted string representation."""
        lines = [
            "OfflineIndexResult",
            "=" * 50,
            f"Total Candidates: {self.total_candidates}",
            f"Processed Candidates: {self.processed_candidates}",
            f"Embedding Dimension: {self.embedding_dimension}",
            f"Index Size: {self.index_size}",
            f"Processing Time: {self.processing_time_seconds:.2f}s",
            f"  - Embedding Time: {self.embedding_time_seconds:.2f}s",
            f"  - Index Build Time: {self.index_build_time_seconds:.2f}s",
            "",
            "Artifacts:",
        ]
        for name, path in self.artifacts.items():
            lines.append(f"  {name}: {path}")

        if self.statistics:
            lines.append("")
            lines.append("Statistics:")
            for key, value in self.statistics.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)
