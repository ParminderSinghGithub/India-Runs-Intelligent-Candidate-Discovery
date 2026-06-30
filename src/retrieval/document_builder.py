"""Retrieval document builder for candidate embedding."""

import re
from typing import Dict, Any

from src.models.candidate import Candidate
from src.models.retrieval_document import RetrievalDocument


class RetrievalDocumentBuilder:
    """Builder for creating optimized retrieval documents from candidate data.

    The document is engineered for semantic retrieval quality with:
    - Career history as the strongest signal (largest portion)
    - Current title appearing near the beginning
    - Skills after career history
    - Education with small weight
    - No behavior signals embedded
    """

    def build(self, candidate: Candidate) -> RetrievalDocument:
        """Build a retrieval document from candidate data.

        Args:
            candidate: Candidate dataclass.

        Returns:
            RetrievalDocument: Optimized document for embedding.
        """
        # Build individual sections
        sections = self._build_sections(candidate)

        # Combine sections into final document
        document = self._combine_sections(sections)

        # Build metadata
        metadata = self._build_metadata(candidate, sections, document)

        return RetrievalDocument(
            candidate_id=candidate.candidate_id,
            document=document,
            sections=sections,
            metadata=metadata,
        )

    def _build_sections(self, candidate: Candidate) -> Dict[str, str]:
        """Build individual document sections from candidate data.

        Args:
            candidate: Candidate dataclass.

        Returns:
            Dict of section names to section content.
        """
        sections = {}

        # Title section (appears near beginning for strong signal)
        sections["title"] = self._build_title_section(candidate)

        # Summary section (brief overview)
        sections["summary"] = self._build_summary_section(candidate)

        # Current role section (current title appears again for emphasis)
        sections["current_role"] = self._build_current_role_section(candidate)

        # Career history section (largest portion - strongest signal)
        sections["career_history"] = self._build_career_history_section(candidate)

        # Skills section (after career history)
        sections["skills"] = self._build_skills_section(candidate)

        # Industries section
        sections["industries"] = self._build_industries_section(candidate)

        # Experience section
        sections["experience"] = self._build_experience_section(candidate)

        # Education section (small weight)
        sections["education"] = self._build_education_section(candidate)

        # Filter out empty sections
        return {k: v for k, v in sections.items() if v}

    def _build_title_section(self, candidate: Candidate) -> str:
        """Build title section.

        Args:
            candidate: Candidate dataclass.

        Returns:
            Title section text.
        """
        if candidate.profile and candidate.profile.current_title:
            return f"TITLE\n{self._clean_text(candidate.profile.current_title)}"
        elif candidate.career_history and candidate.career_history[0].title:
            return f"TITLE\n{self._clean_text(candidate.career_history[0].title)}"
        return ""

    def _build_summary_section(self, candidate: Candidate) -> str:
        """Build summary section.

        Args:
            candidate: Candidate dataclass.

        Returns:
            Summary section text.
        """
        parts = []
        if candidate.profile:
            if candidate.profile.current_title:
                parts.append(candidate.profile.current_title)
            if candidate.profile.location:
                parts.append(f"based in {candidate.profile.location}")

        if parts:
            return f"SUMMARY\n{self._clean_text('. '.join(parts))}"
        return ""

    def _build_current_role_section(self, candidate: Candidate) -> str:
        """Build current role section.

        Args:
            candidate: Candidate dataclass.

        Returns:
            Current role section text.
        """
        if not candidate.career_history:
            return ""

        # Get current role (most recent with is_current=True)
        current_role = None
        for career in candidate.career_history:
            if career.is_current:
                current_role = career
                break

        if not current_role and candidate.career_history:
            # Fallback to most recent
            current_role = candidate.career_history[0]

        if not current_role:
            return ""

        parts = [f"CURRENT ROLE\n{self._clean_text(current_role.title)}"]
        if current_role.company:
            parts.append(f"at {self._clean_text(current_role.company)}")
        if current_role.industry:
            parts.append(f"in {self._clean_text(current_role.industry)}")

        return " ".join(parts)

    def _build_career_history_section(self, candidate: Candidate) -> str:
        """Build career history section (largest portion).

        Args:
            candidate: Candidate dataclass.

        Returns:
            Career history section text.
        """
        if not candidate.career_history:
            return ""

        parts = ["CAREER HISTORY"]

        for career in candidate.career_history:
            career_parts = []
            if career.title:
                career_parts.append(f"Role: {self._clean_text(career.title)}")
            if career.company:
                career_parts.append(f"Company: {self._clean_text(career.company)}")
            if career.industry:
                career_parts.append(f"Industry: {self._clean_text(career.industry)}")
            if career.description:
                career_parts.append(f"Description: {self._clean_text(career.description)}")

            if career_parts:
                parts.append(". ".join(career_parts))

        return "\n".join(parts)

    def _build_skills_section(self, candidate: Candidate) -> str:
        """Build skills section.

        Args:
            candidate: Candidate dataclass.

        Returns:
            Skills section text.
        """
        if not candidate.skills:
            return ""

        skill_names = [skill.name for skill in candidate.skills if skill.name]
        if not skill_names:
            return ""

        return f"SKILLS\n{self._clean_text(', '.join(skill_names))}"

    def _build_industries_section(self, candidate: Candidate) -> str:
        """Build industries section.

        Args:
            candidate: Candidate dataclass.

        Returns:
            Industries section text.
        """
        if not candidate.career_history:
            return ""

        industries = set()
        for career in candidate.career_history:
            if career.industry:
                industries.add(career.industry)

        if not industries:
            return ""

        return f"INDUSTRIES\n{self._clean_text(', '.join(sorted(industries)))}"

    def _build_experience_section(self, candidate: Candidate) -> str:
        """Build experience section.

        Args:
            candidate: Candidate dataclass.

        Returns:
            Experience section text.
        """
        if candidate.profile and candidate.profile.years_of_experience is not None:
            return f"YEARS OF EXPERIENCE\n{candidate.profile.years_of_experience}"
        return ""

    def _build_education_section(self, candidate: Candidate) -> str:
        """Build education section (small weight).

        Args:
            candidate: Candidate dataclass.

        Returns:
            Education section text.
        """
        if not candidate.education:
            return ""

        parts = ["EDUCATION"]
        for edu in candidate.education[:2]:  # Limit to 2 most recent
            edu_parts = []
            if edu.degree:
                edu_parts.append(edu.degree)
            if edu.field_of_study:
                edu_parts.append(edu.field_of_study)
            if edu.institution:
                edu_parts.append(f"at {edu.institution}")

            if edu_parts:
                parts.append(self._clean_text(" ".join(edu_parts)))

        return "\n".join(parts)

    def _combine_sections(self, sections: Dict[str, str]) -> str:
        """Combine sections into final document.

        Args:
            sections: Dict of section names to content.

        Returns:
            Combined document text.
        """
        # Order sections for optimal embedding
        section_order = [
            "title",
            "summary",
            "current_role",
            "career_history",
            "skills",
            "industries",
            "experience",
            "education",
        ]

        document_parts = []
        for section_name in section_order:
            if section_name in sections and sections[section_name]:
                document_parts.append(sections[section_name])

        # Join with double newlines for clear separation
        return "\n\n".join(document_parts)

    def _build_metadata(
        self, candidate: Candidate, sections: Dict[str, str], document: str
    ) -> Dict[str, Any]:
        """Build metadata for the retrieval document.

        Args:
            candidate: Candidate dataclass.
            sections: Document sections.
            document: Final document text.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "candidate_id": candidate.candidate_id,
            "title": candidate.profile.current_title if candidate.profile else None,
            "location": candidate.profile.location if candidate.profile else None,
            "experience_years": (
                candidate.profile.years_of_experience if candidate.profile else None
            ),
            "number_of_skills": len(candidate.skills) if candidate.skills else 0,
            "number_of_career_entries": (
                len(candidate.career_history) if candidate.career_history else 0
            ),
            "number_of_education_entries": (
                len(candidate.education) if candidate.education else 0
            ),
            "document_length": len(document),
            "document_word_count": len(document.split()),
            "section_count": len(sections),
            "sections": list(sections.keys()),
        }

        # Add current role if available
        if candidate.career_history:
            for career in candidate.career_history:
                if career.is_current:
                    metadata["current_title"] = career.title
                    metadata["current_company"] = career.company
                    metadata["current_industry"] = career.industry
                    break

        return metadata

    def _clean_text(self, text: str) -> str:
        """Clean text by normalizing whitespace and removing artifacts.

        Args:
            text: Raw text.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Remove special characters that might interfere with embedding
        text = re.sub(r"[^\w\s\.,\-]", " ", text)
        # Normalize whitespace again after special char removal
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text
