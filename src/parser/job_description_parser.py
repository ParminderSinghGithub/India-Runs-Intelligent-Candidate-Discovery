"""Job description parser interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from src.models.job_description import JobDescription


class JobDescriptionParser(ABC):
    """Abstract base class for parsing job description data."""

    @abstractmethod
    def parse(self, data: Dict[str, Any]) -> JobDescription:
        """Parse raw dictionary data into a JobDescription object.

        Args:
            data: Raw dictionary containing job description data.

        Returns:
            JobDescription: Parsed job description object.

        Raises:
            ValidationError: If data structure is invalid.
        """
        pass

    @abstractmethod
    def parse_from_file(self, file_path: Path) -> JobDescription:
        """Load and parse job description data from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            JobDescription: Parsed job description object.

        Raises:
            FileNotFoundError: If file does not exist.
            ParserError: If file cannot be parsed.
            ValidationError: If data structure is invalid.
        """
        pass
