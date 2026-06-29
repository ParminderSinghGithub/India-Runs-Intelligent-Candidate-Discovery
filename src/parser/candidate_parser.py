"""Candidate parser implementation."""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.candidate import Candidate
from src.models.profile import Profile
from src.models.career import Career
from src.models.education import Education
from src.models.skill import Skill
from src.models.certification import Certification
from src.models.language import Language
from src.models.redrob_signals import RedrobSignals, SalaryRange
from src.utils.exceptions import ParserError, ValidationError

logger = logging.getLogger(__name__)


class CandidateParser:
    """Parser for converting raw candidate JSON into domain objects."""

    def from_dict(self, data: Dict[str, Any]) -> Candidate:
        """Parse raw dictionary data into a Candidate object.

        Args:
            data: Raw dictionary containing candidate data.

        Returns:
            Candidate: Parsed candidate object.

        Raises:
            ValidationError: If required fields are missing or invalid.
            ParserError: If parsing fails.
        """
        try:
            self._validate_required_fields(data)
            return self._parse_candidate(data)
        except (ValidationError, ParserError):
            raise
        except Exception as e:
            raise ParserError(f"Failed to parse candidate: {e}") from e

    def from_json(self, json_str: str) -> Candidate:
        """Parse JSON string into a Candidate object.

        Args:
            json_str: JSON string containing candidate data.

        Returns:
            Candidate: Parsed candidate object.

        Raises:
            ParserError: If JSON is invalid or parsing fails.
            ValidationError: If required fields are missing or invalid.
        """
        try:
            data = json.loads(json_str)
            return self.from_dict(data)
        except json.JSONDecodeError as e:
            raise ParserError(f"Invalid JSON: {e}") from e

    def from_jsonl_line(self, jsonl_line: str) -> Candidate:
        """Parse a single JSONL line into a Candidate object.

        Args:
            jsonl_line: Single line from a JSONL file.

        Returns:
            Candidate: Parsed candidate object.

        Raises:
            ParserError: If JSON is invalid or parsing fails.
            ValidationError: If required fields are missing or invalid.
        """
        return self.from_json(jsonl_line.strip())

    def parse_file(self, file_path: Path) -> Candidate:
        """Load and parse a single candidate from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Candidate: Parsed candidate object.

        Raises:
            FileNotFoundError: If file does not exist.
            ParserError: If file cannot be parsed.
            ValidationError: If required fields are missing or invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return self.from_dict(data)
        except json.JSONDecodeError as e:
            raise ParserError(f"Invalid JSON in file {file_path}: {e}") from e

    def parse_many(self, file_path: Path) -> List[Candidate]:
        """Load and parse multiple candidates from a JSON array file.

        Args:
            file_path: Path to the JSON file containing an array of candidates.

        Returns:
            List[Candidate]: List of parsed candidate objects.

        Raises:
            FileNotFoundError: If file does not exist.
            ParserError: If file cannot be parsed.
            ValidationError: If required fields are missing or invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ParserError(f"Expected JSON array in {file_path}, got {type(data).__name__}")

            candidates = []
            for i, item in enumerate(data):
                try:
                    candidate = self.from_dict(item)
                    candidates.append(candidate)
                except (ValidationError, ParserError) as e:
                    logger.warning(f"Failed to parse candidate at index {i}: {e}")
                    continue

            return candidates
        except json.JSONDecodeError as e:
            raise ParserError(f"Invalid JSON in file {file_path}: {e}") from e

    def _validate_required_fields(self, data: Dict[str, Any]) -> None:
        """Validate that all required fields are present.

        Args:
            data: Raw candidate data dictionary.

        Raises:
            ValidationError: If required fields are missing.
        """
        required_fields = [
            "candidate_id",
            "profile",
            "career_history",
            "education",
            "skills",
            "redrob_signals",
        ]

        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")

        # Validate nested profile fields
        profile_required = [
            "anonymized_name",
            "headline",
            "summary",
            "location",
            "country",
            "years_of_experience",
            "current_title",
            "current_company",
            "current_company_size",
            "current_industry",
        ]
        for field in profile_required:
            if field not in data["profile"]:
                raise ValidationError(f"Missing required profile field: {field}")

    def _parse_candidate(self, data: Dict[str, Any]) -> Candidate:
        """Parse candidate data dictionary into Candidate object.

        Args:
            data: Raw candidate data dictionary.

        Returns:
            Candidate: Parsed candidate object.
        """
        profile = self._parse_profile(data["profile"])
        career_history = self._parse_career_history(data["career_history"])
        education = self._parse_education(data["education"])
        skills = self._parse_skills(data["skills"])
        redrob_signals = self._parse_redrob_signals(data["redrob_signals"])
        certifications = self._parse_certifications(data.get("certifications", []))
        languages = self._parse_languages(data.get("languages", []))

        return Candidate(
            candidate_id=self._normalize_string(data["candidate_id"]),
            profile=profile,
            career_history=career_history,
            education=education,
            skills=skills,
            redrob_signals=redrob_signals,
            certifications=certifications,
            languages=languages,
        )

    def _parse_profile(self, profile_data: Dict[str, Any]) -> Profile:
        """Parse profile data into Profile object.

        Args:
            profile_data: Raw profile data dictionary.

        Returns:
            Profile: Parsed profile object.
        """
        return Profile(
            anonymized_name=self._normalize_string(profile_data["anonymized_name"]),
            headline=self._normalize_string(profile_data["headline"]),
            summary=self._normalize_string(profile_data["summary"]),
            location=self._normalize_string(profile_data["location"]),
            country=self._normalize_string(profile_data["country"]),
            years_of_experience=float(profile_data["years_of_experience"]),
            current_title=self._normalize_string(profile_data["current_title"]),
            current_company=self._normalize_string(profile_data["current_company"]),
            current_company_size=self._normalize_string(profile_data["current_company_size"]),
            current_industry=self._normalize_string(profile_data["current_industry"]),
        )

    def _parse_career_history(self, career_data: List[Dict[str, Any]]) -> List[Career]:
        """Parse career history data into list of Career objects.

        Args:
            career_data: Raw career history data list.

        Returns:
            List[Career]: List of parsed Career objects.
        """
        careers = []
        for item in career_data:
            careers.append(
                Career(
                    company=self._normalize_string(item["company"]),
                    title=self._normalize_string(item["title"]),
                    start_date=self._parse_date(item.get("start_date")),
                    end_date=self._parse_date(item.get("end_date")),
                    duration_months=item.get("duration_months"),
                    is_current=bool(item.get("is_current", False)),
                    industry=self._normalize_string(item.get("industry")),
                    company_size=self._normalize_string(item.get("company_size")),
                    description=self._normalize_string(item.get("description")),
                )
            )
        return careers

    def _parse_education(self, education_data: List[Dict[str, Any]]) -> List[Education]:
        """Parse education data into list of Education objects.

        Args:
            education_data: Raw education data list.

        Returns:
            List[Education]: List of parsed Education objects.
        """
        educations = []
        for item in education_data:
            educations.append(
                Education(
                    institution=self._normalize_string(item["institution"]),
                    degree=self._normalize_string(item["degree"]),
                    field_of_study=self._normalize_string(item["field_of_study"]),
                    start_year=int(item["start_year"]),
                    end_year=int(item["end_year"]),
                    grade=self._normalize_string(item.get("grade")),
                    tier=self._normalize_string(item.get("tier")),
                )
            )
        return educations

    def _parse_skills(self, skills_data: List[Dict[str, Any]]) -> List[Skill]:
        """Parse skills data into list of Skill objects.

        Args:
            skills_data: Raw skills data list.

        Returns:
            List[Skill]: List of parsed Skill objects.
        """
        skills = []
        for item in skills_data:
            skills.append(
                Skill(
                    name=self._normalize_string(item["name"]),
                    proficiency=self._normalize_string(item["proficiency"]),
                    endorsements=int(item.get("endorsements", 0)),
                    duration_months=item.get("duration_months"),
                )
            )
        return skills

    def _parse_certifications(self, cert_data: List[Dict[str, Any]]) -> List[Certification]:
        """Parse certifications data into list of Certification objects.

        Args:
            cert_data: Raw certifications data list.

        Returns:
            List[Certification]: List of parsed Certification objects.
        """
        certifications = []
        for item in cert_data:
            certifications.append(
                Certification(
                    name=self._normalize_string(item["name"]),
                    issuer=self._normalize_string(item["issuer"]),
                    year=int(item["year"]),
                )
            )
        return certifications

    def _parse_languages(self, lang_data: List[Dict[str, Any]]) -> List[Language]:
        """Parse languages data into list of Language objects.

        Args:
            lang_data: Raw languages data list.

        Returns:
            List[Language]: List of parsed Language objects.
        """
        languages = []
        for item in lang_data:
            languages.append(
                Language(
                    language=self._normalize_string(item["language"]),
                    proficiency=self._normalize_string(item["proficiency"]),
                )
            )
        return languages

    def _parse_redrob_signals(self, signals_data: Dict[str, Any]) -> RedrobSignals:
        """Parse Redrob signals data into RedrobSignals object.

        Args:
            signals_data: Raw Redrob signals data dictionary.

        Returns:
            RedrobSignals: Parsed RedrobSignals object.
        """
        salary_range_data = signals_data.get("expected_salary_range_inr_lpa", {})
        salary_range = SalaryRange(
            min=float(salary_range_data.get("min", 0)),
            max=float(salary_range_data.get("max", 0)),
        )

        github_score = signals_data.get("github_activity_score")
        if github_score == -1:
            github_score = None

        offer_acceptance = signals_data.get("offer_acceptance_rate")
        if offer_acceptance == -1:
            offer_acceptance = None

        return RedrobSignals(
            profile_completeness_score=float(signals_data["profile_completeness_score"]),
            signup_date=self._parse_date(signals_data["signup_date"]),
            last_active_date=self._parse_date(signals_data["last_active_date"]),
            open_to_work_flag=bool(signals_data["open_to_work_flag"]),
            profile_views_received_30d=int(signals_data["profile_views_received_30d"]),
            applications_submitted_30d=int(signals_data["applications_submitted_30d"]),
            recruiter_response_rate=float(signals_data["recruiter_response_rate"]),
            avg_response_time_hours=float(signals_data["avg_response_time_hours"]),
            skill_assessment_scores=dict(signals_data.get("skill_assessment_scores", {})),
            connection_count=int(signals_data["connection_count"]),
            endorsements_received=int(signals_data["endorsements_received"]),
            notice_period_days=int(signals_data["notice_period_days"]),
            expected_salary_range_inr_lpa=salary_range,
            preferred_work_mode=self._normalize_string(signals_data["preferred_work_mode"]),
            willing_to_relocate=bool(signals_data["willing_to_relocate"]),
            github_activity_score=github_score,
            search_appearance_30d=int(signals_data["search_appearance_30d"]),
            saved_by_recruiters_30d=int(signals_data["saved_by_recruiters_30d"]),
            interview_completion_rate=float(signals_data["interview_completion_rate"]),
            offer_acceptance_rate=offer_acceptance,
            verified_email=bool(signals_data["verified_email"]),
            verified_phone=bool(signals_data["verified_phone"]),
            linkedin_connected=bool(signals_data["linkedin_connected"]),
        )

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string into date object.

        Args:
            date_str: Date string in YYYY-MM-DD format, or None.

        Returns:
            Optional[date]: Parsed date object, or None if input is None or empty.
        """
        if not date_str:
            return None

        try:
            year, month, day = map(int, date_str.split("-"))
            return date(year, month, day)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None

    def _normalize_string(self, value: Any) -> Optional[str]:
        """Normalize string value by trimming whitespace and converting empty to None.

        Args:
            value: Value to normalize.

        Returns:
            Optional[str]: Normalized string, or None if empty after trimming.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            return str(value)

        trimmed = value.strip()
        return trimmed if trimmed else None
