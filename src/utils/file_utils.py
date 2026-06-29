"""File operation utilities."""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.
    """
    path.mkdir(parents=True, exist_ok=True)


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON data from a file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON data.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to save.
        file_path: Path to the output file.
    """
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
