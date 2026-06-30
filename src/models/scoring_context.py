"""Scoring context domain model."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.models.candidate import Candidate
from src.models.job_description import JobDescription


@dataclass(frozen=True)
class ScoringContext:
    """Context object passed to all scorers.

    Encapsulates all data needed for scoring to avoid large parameter lists.
    Can be extended with cache objects, configuration, etc.
    """

    candidate: Candidate
    """Candidate to score."""

    job_description: JobDescription
    """Job description to match against."""

    config: Dict[str, Any] = field(default_factory=dict)
    """Scoring configuration (weights, thresholds, etc.)."""

    cache: Dict[str, Any] = field(default_factory=dict)
    """Shared cache for expensive computations (embeddings, lookups, etc.)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata for scoring context."""

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Any: Configuration value or default.
        """
        return self.config.get(key, default)

    def get_cache(self, key: str, default: Any = None) -> Any:
        """Get cached value by key.

        Args:
            key: Cache key.
            default: Default value if key not found.

        Returns:
            Any: Cached value or default.
        """
        return self.cache.get(key, default)

    def with_config(self, **kwargs: Any) -> "ScoringContext":
        """Create new context with additional config values.

        Args:
            **kwargs: Configuration key-value pairs to add.

        Returns:
            ScoringContext: New context with merged config.
        """
        new_config = dict(self.config)
        new_config.update(kwargs)
        object.__setattr__(self, "config", new_config)
        return self

    def with_cache(self, **kwargs: Any) -> "ScoringContext":
        """Create new context with additional cache values.

        Args:
            **kwargs: Cache key-value pairs to add.

        Returns:
            ScoringContext: New context with merged cache.
        """
        new_cache = dict(self.cache)
        new_cache.update(kwargs)
        object.__setattr__(self, "cache", new_cache)
        return self
