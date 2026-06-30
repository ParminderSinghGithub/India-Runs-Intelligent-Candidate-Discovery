"""Job description parser implementation."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.models.candidate_filters import CandidateFilters
from src.models.job_description import JobDescription
from src.models.parsed_job import ParsedJob
from src.models.search_query import SearchQuery


class JobDescriptionParser:
    """Parser for converting job description data into structured retrieval inputs.

    This parser extracts structured information from job descriptions and creates:
    - JobDescription: Standardized job description object
    - SearchQuery: Multi-dimensional queries for retrieval
    - CandidateFilters: Filters for candidate filtering
    """

    def parse(self, data: Dict[str, Any]) -> ParsedJob:
        """Parse raw dictionary data into a ParsedJob object.

        Args:
            data: Raw dictionary containing job description data.

        Returns:
            ParsedJob: Complete parsed job information.

        Raises:
            ValueError: If required fields are missing.
        """
        # Extract required fields
        job_id = data.get("job_id", "UNKNOWN")
        title = data.get("title")
        company = data.get("company")

        if not title:
            raise ValueError("Job title is required")

        # Extract optional fields
        location = data.get("location")
        description = data.get("description")
        required_experience_years = data.get("required_experience_years")
        required_skills = data.get("required_skills", [])
        preferred_skills = data.get("preferred_skills", [])
        responsibilities = data.get("responsibilities", [])
        target_industries = data.get("target_industries", [])
        technologies = data.get("technologies", [])
        nice_to_have_skills = data.get("nice_to_have_skills", [])
        work_mode = data.get("work_mode")
        minimum_experience = data.get("minimum_experience")
        maximum_experience = data.get("maximum_experience")

        # Create JobDescription
        job_description = JobDescription(
            job_id=job_id,
            title=title,
            company=company,
            location=location,
            description=description,
            required_skills=required_skills,
            required_experience_years=required_experience_years,
            responsibilities=responsibilities,
            behaviors=[],  # Not extracted from JD
        )

        # Create SearchQuery
        search_query = self._build_search_query(
            title=title,
            company=company,
            description=description,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            responsibilities=responsibilities,
            technologies=technologies,
        )

        # Create CandidateFilters
        candidate_filters = self._build_candidate_filters(
            minimum_experience=minimum_experience,
            maximum_experience=maximum_experience,
            required_location=location,
            required_industries=target_industries,
            required_work_mode=work_mode,
        )

        return ParsedJob(
            job_description=job_description,
            search_query=search_query,
            candidate_filters=candidate_filters,
        )

    def parse_from_file(self, file_path: Path) -> ParsedJob:
        """Load and parse job description data from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            ParsedJob: Complete parsed job information.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file cannot be parsed or data is invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Job description file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.parse(data)

    def _build_search_query(
        self,
        title: str,
        company: Optional[str],
        description: Optional[str],
        required_skills: List[str],
        preferred_skills: List[str],
        responsibilities: List[str],
        technologies: List[str],
    ) -> SearchQuery:
        """Build search queries from job description components.

        Args:
            title: Job title.
            company: Company name.
            description: Job description.
            required_skills: Required skills list.
            preferred_skills: Preferred skills list.
            responsibilities: Responsibilities list.
            technologies: Technologies list.

        Returns:
            SearchQuery: Multi-dimensional search queries.
        """
        # Identity query: Focus on role and seniority
        identity_parts = [title]
        if company:
            identity_parts.append(f"at {company}")
        identity_query = " ".join(identity_parts)

        # Career query: Focus on experience and responsibilities
        career_parts = [title]
        if description:
            career_parts.append(description)
        if responsibilities:
            career_parts.extend(responsibilities)
        career_query = " ".join(career_parts)

        # Skills query: Focus on technical skills
        all_skills = required_skills + preferred_skills + technologies
        skills_query = " ".join(all_skills)

        # Combined query: All information together
        combined_parts = [title]
        if description:
            combined_parts.append(description)
        if required_skills:
            combined_parts.extend(required_skills)
        if responsibilities:
            combined_parts.extend(responsibilities)
        combined_query = " ".join(combined_parts)

        return SearchQuery(
            identity_query=identity_query,
            career_query=career_query,
            skills_query=skills_query,
            combined_query=combined_query,
        )

    def _build_candidate_filters(
        self,
        minimum_experience: Optional[float],
        maximum_experience: Optional[float],
        required_location: Optional[str],
        required_industries: Optional[List[str]],
        required_work_mode: Optional[str],
    ) -> CandidateFilters:
        """Build candidate filters from job description.

        Args:
            minimum_experience: Minimum years of experience.
            maximum_experience: Maximum years of experience.
            required_location: Required location.
            required_industries: Required industries.
            required_work_mode: Required work mode.

        Returns:
            CandidateFilters: Filters for candidate retrieval.
        """
        return CandidateFilters(
            minimum_experience_years=minimum_experience,
            maximum_experience_years=maximum_experience,
            required_location=required_location,
            required_industries=required_industries,
            required_work_mode=required_work_mode,
            open_to_work=None,  # Not specified in JD
        )
