"""Retrieval document domain model for embedding."""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass(frozen=True)
class RetrievalDocument:
    """Optimized text representation of a candidate for embedding.

    This document is engineered for semantic retrieval quality.
    It contains the final text to be embedded, the individual sections,
    and metadata for filtering and analysis.
    """

    candidate_id: str
    """Unique identifier for the candidate."""

    document: str
    """Final optimized text for embedding."""

    sections: Dict[str, str]
    """Individual sections before joining (e.g., title, summary, career_history)."""

    metadata: Dict[str, Any]
    """Useful metadata not embedded (e.g., experience_years, skill_count)."""

    def __str__(self) -> str:
        """String representation of the retrieval document."""
        return f"RetrievalDocument(candidate_id={self.candidate_id}, length={len(self.document)})"

    def get_section(self, section_name: str) -> str:
        """Get a specific section by name.

        Args:
            section_name: Name of the section to retrieve.

        Returns:
            str: Section content, or empty string if not found.
        """
        return self.sections.get(section_name, "")

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Metadata value or default.
        """
        return self.metadata.get(key, default)
