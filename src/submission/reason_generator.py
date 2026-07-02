"""Diverse, deterministic, evidence-backed recruiter reasoning."""

from __future__ import annotations

import re
from typing import List, Sequence

from src.models.candidate import Candidate
from src.models.hybrid_score_result import HybridScoreResult
from src.models.parsed_job import ParsedJob


class ReasonGenerator:
    """Render recruiter explanations without introducing unsupported claims."""

    LEAD_TEMPLATES = {
        "excellent": (
            "{title} brings {years:.1f} years of experience and unusually strong career and technical alignment for this role.",
            "The combination of {title} experience, relevant skills, and {years:.1f} years in the field makes this a leading profile.",
            "This is a high-confidence technical fit: the candidate is a {title} with {years:.1f} years of directly useful experience.",
            "Career trajectory and hands-on evidence both strongly support this {title} profile for the opening.",
        ),
        "strong": (
            "Strong career relevance from a {title} profile with {years:.1f} years of experience in {industry}.",
            "The candidate's {title} background provides credible role alignment, backed by {years:.1f} years of experience.",
            "A relevant {title} trajectory and solid technical coverage place this profile among the stronger matches.",
            "This profile aligns well through its {title} experience and sustained work in {industry}.",
        ),
        "technical": (
            "Technical evidence is stronger than the title match, but the profile still covers much of the role's core work.",
            "The candidate shows a useful technical fit, with career relevance that is credible though not exact.",
            "Skills and production evidence carry this profile, while title alignment is more moderate.",
            "This is primarily a capability-led match rather than a perfect title-for-title match.",
        ),
        "partial": (
            "The profile has partial alignment for the role, with {years:.1f} years of experience and some relevant technical evidence.",
            "There is a workable foundation here, although career and skill coverage are mixed.",
            "The candidate offers transferable experience, but the evidence is less direct than for the higher-ranked profiles.",
            "This is a moderate match: relevant signals are present, alongside material role-specific gaps.",
        ),
    }

    def generate(self, candidate: Candidate, result: HybridScoreResult, parsed_job: ParsedJob) -> str:
        parts = [self._lead_clause(candidate, result)]
        theme = self._specialism_clause(candidate, result)
        if theme:
            parts.append(theme)
        skills = self._skill_clause(result, parsed_job)
        if skills:
            parts.append(skills)
        behavior = self._behavior_clause(candidate, result)
        if behavior:
            parts.append(behavior)
        gap = self._gap_clause(result, parsed_job)
        if gap:
            parts.append(gap)
        return " ".join(parts)

    def _pick(self, candidate_id: str, choices: Sequence[str], salt: int = 0) -> str:
        index = (sum(ord(char) for char in candidate_id) + salt) % len(choices)
        return choices[index]

    def _lead_clause(self, candidate: Candidate, result: HybridScoreResult) -> str:
        if result.career_score >= 0.80 and result.skill_score >= 0.76:
            band = "excellent"
        elif result.career_score >= 0.68 and result.skill_score >= 0.65:
            band = "strong"
        elif result.skill_score >= 0.68:
            band = "technical"
        else:
            band = "partial"
        template = self._pick(candidate.candidate_id, self.LEAD_TEMPLATES[band])
        return template.format(
            title=candidate.profile.current_title or "technical professional",
            years=float(candidate.profile.years_of_experience or 0.0),
            industry=candidate.profile.current_industry or "the relevant industry",
        )

    def _specialism_clause(self, candidate: Candidate, result: HybridScoreResult) -> str:
        evidence = " ".join(result.matched_items + result.reasons + [
            career.description or "" for career in candidate.career_history or []
        ]).lower()
        options = []
        if "recommend" in evidence:
            options.append("Recommendation-systems experience adds useful relevance for personalization work.")
        if "retrieval" in evidence:
            options.append("Retrieval-heavy experience strengthens the fit for search and candidate-matching systems.")
        if re.search(r"\branking\b|learning to rank", evidence):
            options.append("Ranking-systems evidence is directly useful for relevance-oriented ML work.")
        if "production" in evidence or "deploy" in evidence:
            options.append("Production deployment evidence reduces the gap between model development and operating ML systems.")
        if "platform" in evidence:
            options.append("ML or data-platform experience supports the systems side of the role.")
        return self._pick(candidate.candidate_id, options, 7) if options else ""

    def _skill_clause(self, result: HybridScoreResult, parsed_job: ParsedJob) -> str:
        required = list(parsed_job.job_description.required_skills or [])
        matched_keys = {self._normalize(item) for item in result.matched_items}
        missing_keys = {self._normalize(item) for item in result.missing_items}
        matched = [
            skill for skill in required
            if self._normalize(skill) in matched_keys and self._normalize(skill) not in missing_keys
        ]
        missing = [skill for skill in required if self._normalize(skill) in missing_keys]
        clauses = []
        if matched:
            clauses.append(f"Core matches include {self._format_list(matched[:4])}.")
        if missing:
            clauses.append(f"Explicit gaps remain in {self._format_list(missing[:3])}.")
        return " ".join(clauses)

    def _behavior_clause(self, candidate: Candidate, result: HybridScoreResult) -> str:
        signals = candidate.redrob_signals
        facts = []
        if signals.open_to_work_flag:
            facts.append("open to work")
        if signals.verified_email and signals.verified_phone:
            facts.append("email and phone verified")
        if signals.recruiter_response_rate is not None and signals.recruiter_response_rate >= 0.70:
            facts.append(f"a {signals.recruiter_response_rate:.0%} recruiter response rate")
        if signals.notice_period_days is not None and signals.notice_period_days <= 30:
            facts.append(f"a {signals.notice_period_days}-day notice period")
        if signals.profile_completeness_score >= 0.85:
            facts.append("a highly complete profile")
        if not facts:
            return ""
        templates = (
            "As a tie-breaker, recruiter-readiness is supported by {facts}.",
            "Recruiter-facing signals add confidence through {facts}.",
            "For outreach, the profile also offers {facts}.",
            "Secondary behavioral evidence is favorable: {facts}.",
        )
        return self._pick(candidate.candidate_id, templates, 13).format(facts=self._format_list(facts[:3]))

    def _gap_clause(self, result: HybridScoreResult, parsed_job: ParsedJob) -> str:
        required_keys = {self._normalize(skill) for skill in parsed_job.job_description.required_skills or []}
        gaps = [item for item in result.missing_items if self._normalize(item) in required_keys]
        if not gaps:
            return "No major required-skill gap is evident from the structured profile."
        if result.career_score < 0.55:
            return "The main reservation is that both direct career relevance and required-skill coverage need validation."
        return "The remaining technical gaps should be verified during screening."

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _format_list(self, items: List[str]) -> str:
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"
