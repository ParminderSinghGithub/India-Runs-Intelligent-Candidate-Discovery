"""Custom exceptions for the application."""


class IndiaRunsError(Exception):
    """Base exception for India Runs application."""

    pass


class ParserError(IndiaRunsError):
    """Exception raised when parsing fails."""

    pass


class ValidationError(IndiaRunsError):
    """Exception raised when data validation fails."""

    pass


class ScorerError(IndiaRunsError):
    """Exception raised when scoring fails."""

    pass


class FileNotFoundError(IndiaRunsError):
    """Exception raised when a required file is not found."""

    pass
