"""Strict validation for final submission artifacts."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from src.models.submission_row import SubmissionRow
from src.utils.exceptions import ValidationError

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
EXPECTED_ROWS = 100
MAX_REASONING_CHARACTERS = 1000


class SubmissionValidator:
    """Validate ranked rows and generated submission CSVs."""

    def validate_rows(
        self,
        rows: Sequence[SubmissionRow],
        candidate_exists: Optional[Callable[[str], bool]] = None,
        expected_rows: int = EXPECTED_ROWS,
    ) -> List[str]:
        """Validate in-memory submission rows before writing."""
        errors: List[str] = []

        if len(rows) != expected_rows:
            errors.append(f"Expected exactly {expected_rows} rows, found {len(rows)}.")

        seen_ids = set()
        seen_ranks = set()
        seen_reasoning = set()
        previous_score = None
        previous_candidate_id = None

        for position, row in enumerate(rows, start=1):
            row_num = position + 1

            if not row.serializable:
                errors.append(f"Row {row_num}: row is marked non-serializable.")

            candidate_id = (row.candidate_id or "").strip()
            if not candidate_id:
                errors.append(f"Row {row_num}: candidate_id is required.")
            elif not CANDIDATE_ID_PATTERN.match(candidate_id):
                errors.append(f"Row {row_num}: candidate_id must match CAND_XXXXXXX.")
            elif candidate_id in seen_ids:
                errors.append(f"Row {row_num}: duplicate candidate_id '{candidate_id}'.")
            else:
                seen_ids.add(candidate_id)

            if candidate_exists is not None and candidate_id and not candidate_exists(candidate_id):
                errors.append(f"Row {row_num}: candidate_id '{candidate_id}' does not exist in the candidate source.")

            if not isinstance(row.rank, int):
                errors.append(f"Row {row_num}: rank must be an integer.")
            elif not 1 <= row.rank <= expected_rows:
                errors.append(f"Row {row_num}: rank must be between 1 and {expected_rows}.")
            elif row.rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {row.rank}.")
            else:
                seen_ranks.add(row.rank)
                if row.rank != position:
                    errors.append(
                        f"Row {row_num}: rank must be strictly increasing; expected {position}, found {row.rank}."
                    )

            if not self._is_numeric_score(row.score):
                errors.append(f"Row {row_num}: score must be numeric.")
            elif not math.isfinite(float(row.score)):
                errors.append(f"Row {row_num}: score must be finite.")
            elif not 0.0 <= float(row.score) <= 1.0:
                errors.append(f"Row {row_num}: score must be within [0, 1].")

            reasoning = (row.reasoning or "").strip()
            if not reasoning:
                errors.append(f"Row {row_num}: reasoning must not be empty.")
            elif len(reasoning) > MAX_REASONING_CHARACTERS:
                errors.append(
                    f"Row {row_num}: reasoning exceeds the {MAX_REASONING_CHARACTERS}-character safety limit."
                )
            elif reasoning in seen_reasoning:
                errors.append(f"Row {row_num}: reasoning must be unique across candidates.")
            else:
                seen_reasoning.add(reasoning)

            if previous_score is not None and self._is_numeric_score(row.score):
                current_score = float(row.score)
                if current_score > previous_score + 1e-12:
                    errors.append(
                        f"Rows must be sorted by non-increasing score: rank {row.rank} has {current_score:.6f} after {previous_score:.6f}."
                    )
                elif abs(current_score - previous_score) <= 1e-12 and previous_candidate_id is not None:
                    if previous_candidate_id > candidate_id:
                        errors.append(
                            f"Tie-break ordering failed for equal scores: '{previous_candidate_id}' should come after '{candidate_id}'."
                        )

            if self._is_numeric_score(row.score):
                previous_score = float(row.score)
                previous_candidate_id = candidate_id

        missing_ranks = set(range(1, expected_rows + 1)) - seen_ranks
        if missing_ranks:
            errors.append(f"Each rank from 1 to {expected_rows} must appear exactly once; missing: {sorted(missing_ranks)}.")

        return errors

    def validate_csv_file(
        self,
        csv_path: Path,
        candidate_exists: Optional[Callable[[str], bool]] = None,
        expected_rows: int = EXPECTED_ROWS,
    ) -> List[str]:
        """Validate a CSV file after it has been written."""
        errors: List[str] = []
        path = Path(csv_path)

        if path.suffix.lower() != ".csv":
            errors.append("Filename must use a .csv extension.")

        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    return ["Submission CSV is empty."]

                if header != REQUIRED_HEADER:
                    errors.append(
                        f"CSV header must be exactly {','.join(REQUIRED_HEADER)}; found {','.join(header)}."
                    )

                rows = []
                for row in reader:
                    if any(cell.strip() for cell in row):
                        rows.append(row)
        except UnicodeDecodeError:
            return ["Submission CSV must be UTF-8 encoded."]
        except OSError as exc:
            return [f"Unable to read submission CSV: {exc}"]

        if len(rows) != expected_rows:
            errors.append(f"Submission CSV must contain exactly {expected_rows} data rows; found {len(rows)}.")

        submission_rows: List[SubmissionRow] = []
        for row_num, cells in enumerate(rows, start=2):
            if len(cells) != len(REQUIRED_HEADER):
                errors.append(f"Row {row_num}: expected {len(REQUIRED_HEADER)} columns, found {len(cells)}.")
                continue

            candidate_id, rank_s, score_s, reasoning = cells
            try:
                rank = int(rank_s)
            except ValueError:
                errors.append(f"Row {row_num}: rank must be an integer.")
                rank = -1

            try:
                score = float(score_s)
            except ValueError:
                errors.append(f"Row {row_num}: score must be a float.")
                score = -1.0

            submission_rows.append(
                SubmissionRow(
                    candidate_id=candidate_id,
                    rank=rank,
                    score=score,
                    reasoning=reasoning,
                )
            )

        errors.extend(self.validate_rows(submission_rows, candidate_exists=candidate_exists, expected_rows=expected_rows))
        return self._dedupe(errors)

    def _is_numeric_score(self, value: object) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _dedupe(self, errors: Sequence[str]) -> List[str]:
        seen = set()
        deduped = []
        for error in errors:
            if error not in seen:
                seen.add(error)
                deduped.append(error)
        return deduped
