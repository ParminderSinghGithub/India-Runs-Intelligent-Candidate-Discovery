"""Candidate parser interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from src.models.candidate import Candidate


class CandidateParser(ABC):
    """Abstract base class for parsing candidate data."""

    @abstractmethod
    def parse(self, data: Dict[str, Any]) -> Candidate:
        """Parse raw dictionary data into a Candidate object.

        Args:
            data: Raw dictionary containing candidate data.

        Returns:
            Candidate: Parsed candidate object.

        Raises:
            ValidationError: If data structure is invalid.
        """
        pass

    @abstractmethod
    def parse_from_file(self, file_path: Path) -> Candidate:
        """Load and parse candidate data from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Candidate: Parsed candidate object.

        Raises:
            FileNotFoundError: If file does not exist.
            ParserError: If file cannot be parsed.
            ValidationError: If data structure is invalid.
        """
        pass
