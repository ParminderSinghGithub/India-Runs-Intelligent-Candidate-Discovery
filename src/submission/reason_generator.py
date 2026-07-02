"""Deterministic recruiter reasoning for final submissions."""

from __future__ import annotations

from typing import List

from src.models.candidate import Candidate
from src.models.hybrid_score_result import HybridScoreResult
from src.models.parsed_job import ParsedJob


class ReasonGenerator:
    """Generate natural-language recruiter reasoning from real evidence."""

    def generate(
        self,
        candidate: Candidate,
        result: HybridScoreResult,
        parsed_job: ParsedJob,
    ) -> str:
        """Create a concise, evidence-backed reasoning string."""
        parts: List[str] = []
        parts.append(self._lead_clause(candidate, result, parsed_job))

        skill_clause = self._skill_clause(candidate, result)
        if skill_clause:
            parts.append(skill_clause)

        behavior_clause = self._behavior_clause(candidate, result)
        if behavior_clause:
            parts.append(behavior_clause)

        gap_clause = self._gap_clause(result)
        if gap_clause:
            parts.append(gap_clause)

        return " ".join(part.strip() for part in parts if part).strip()

    def _lead_clause(
        self,
        candidate: Candidate,
        result: HybridScoreResult,
        parsed_job: ParsedJob,
    ) -> str:
        profile = candidate.profile
        title = profile.current_title or parsed_job.job_description.title
        years = profile.years_of_experience
        industry = profile.current_industry or "industry"
        semantic = result.semantic_score
        career = result.career_score
        skill = result.skill_score
        behavior = result.behavior_score

        if career >= 0.70 and skill >= 0.70 and semantic >= 0.70:
            return (
                f"Strong {title} fit with {years:.1f} years in {industry} and tightly aligned career, skill, and semantic signals."
            )
        if skill >= 0.72 and behavior >= 0.60:
            return (
                f"Excellent technical match with {title} experience, supported by solid recruiter-ready behavioral signals."
            )
        if career >= 0.70:
            return (
                f"Strong career history in {industry} through {title}, giving this profile credible role alignment."
            )
        if semantic >= 0.75:
            return (
                f"High semantic alignment with the job description and a credible {title} background across {years:.1f} years."
            )
        return (
            f"Candidate shows partial alignment for the {parsed_job.job_description.title} role, with {years:.1f} years of experience in {industry}."
        )

    def _skill_clause(self, candidate: Candidate, result: HybridScoreResult) -> str:
        matched = self._dedupe_preserve_order(result.matched_items)
        missing = self._dedupe_preserve_order(result.missing_items)

        matched_skills = [item for item in matched if self._looks_like_skill(item)]
        missing_skills = [item for item in missing if self._looks_like_skill(item)]

        clauses = []
        if matched_skills:
            clauses.append(f"Important matches include {self._format_list(matched_skills[:3])}.")
        elif result.skill_score >= 0.70:
            clauses.append("The skill profile is broad enough to support the role requirements.")

        if missing_skills:
            clauses.append(f"Remaining skill gaps are {self._format_list(missing_skills[:3])}.")

        return " ".join(clauses).strip()

    def _behavior_clause(self, candidate: Candidate, result: HybridScoreResult) -> str:
        signals = candidate.redrob_signals
        clauses = []

        if signals.open_to_work_flag:
            clauses.append("Open to work, which makes outreach easier.")
        else:
            clauses.append("The candidate is not explicitly open to work.")

        if signals.notice_period_days is not None:
            if signals.notice_period_days <= 30:
                clauses.append(f"Notice period is short at {signals.notice_period_days} days.")
            elif signals.notice_period_days <= 90:
                clauses.append(f"Notice period is moderate at {signals.notice_period_days} days.")
            else:
                clauses.append(f"Notice period is long at {signals.notice_period_days} days.")

        if signals.verified_email and signals.verified_phone:
            clauses.append("Profile is verified by email and phone.")
        elif signals.verified_email or signals.verified_phone:
            verified_field = "email" if signals.verified_email else "phone"
            clauses.append(f"Only {verified_field} verification is present.")
        else:
            clauses.append("The profile lacks direct verification signals.")

        if signals.recruiter_response_rate is not None:
            clauses.append(f"Recruiter response rate is {signals.recruiter_response_rate:.2f}.")

        if signals.github_activity_score not in (None, -1):
            clauses.append(f"GitHub activity score is {float(signals.github_activity_score):.2f}.")

        if result.behavior_score >= 0.70:
            clauses.append("Behavioral signals support a low-friction recruiting conversation.")
        elif result.behavior_score <= 0.45:
            clauses.append("Behavioral signals are weaker, so outreach confidence is lower.")

        return " ".join(clauses).strip()

    def _gap_clause(self, result: HybridScoreResult) -> str:
        missing = self._dedupe_preserve_order(result.missing_items)
        if not missing:
            return ""

        if result.skill_score < 0.60 and result.career_score < 0.60:
            return "The main concern is missing key role-specific skills and only partial career alignment."
        if result.skill_score < 0.60:
            return "The candidate is still missing some key technical requirements from the job description."
        if result.behavior_score < 0.50:
            return "Recruiter-facing behavioral signals are not as strong as the technical fit."
        return "There are still a few gaps to review before outreach."

    def _looks_like_skill(self, item: str) -> bool:
        return any(char.isalpha() for char in item)

    def _format_list(self, items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for item in items:
            normalized = item.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(item)
        return ordered
