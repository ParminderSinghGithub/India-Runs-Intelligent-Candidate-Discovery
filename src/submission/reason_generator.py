"""Diverse, deterministic, evidence-backed recruiter reasoning."""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

from src.models.candidate import Candidate
from src.models.hybrid_score_result import HybridScoreResult
from src.models.parsed_job import ParsedJob


class ReasonGenerator:
    """Render concise recruiter explanations without unsupported claims."""

    # Six variants per score band provide 24 distinct opening templates.
    OPENING_TEMPLATES = {
        "excellent": (
            "{title} brings {years:.1f} years of experience and strong career and technical alignment for this role.",
            "The combination of {title} experience and {years:.1f} years in the field makes this a leading profile.",
            "This profile pairs relevant {title} experience with convincing structured technical evidence.",
            "Career trajectory and skill evidence both strongly support this {title} profile for the opening.",
            "With {years:.1f} years of experience, this {title} is among the clearest matches in the retrieved group.",
            "The candidate's {title} background in {industry} provides a particularly strong foundation for the role.",
        ),
        "strong": (
            "Strong career relevance comes from a {title} profile with {years:.1f} years of experience in {industry}.",
            "The candidate's {title} background provides credible role alignment, backed by {years:.1f} years of experience.",
            "A relevant {title} trajectory and solid technical coverage place this profile among the stronger matches.",
            "This profile aligns well through its {title} experience and sustained work in {industry}.",
            "The structured profile shows a strong fit through {years:.1f} years of relevant {title} experience.",
            "Relevant career progression and technical breadth make this {title} a competitive candidate for review.",
        ),
        "technical": (
            "Technical evidence is stronger than the exact title match, while still covering much of the role's core work.",
            "The candidate shows a useful technical fit, with career relevance that is credible though not exact.",
            "Skills and delivery evidence carry this profile, while title alignment is more moderate.",
            "This is primarily a capability-led match rather than a direct title-for-title match.",
            "The profile offers relevant technical depth, supported by transferable career experience.",
            "Structured skill evidence makes this candidate worth consideration despite a less exact career match.",
        ),
        "partial": (
            "The profile has partial alignment for the role, with {years:.1f} years of experience and some relevant evidence.",
            "There is a workable foundation here, although career and skill coverage are mixed.",
            "The candidate offers transferable experience, but the evidence is less direct than for higher-ranked profiles.",
            "This is a moderate match: relevant signals are present, with some areas requiring confirmation.",
            "The {title} background offers adjacent experience that may transfer to the target role.",
            "Some useful role signals are present, although the structured evidence is not comprehensive.",
        ),
    }

    # Five evidence families with five variants each provide 25 middle templates.
    MIDDLE_TEMPLATES = {
        "recommendation": (
            "Recommendation-systems evidence adds relevance for personalization work.",
            "The profile includes recommendation work that is useful for personalization-oriented ML.",
            "Experience with recommender systems strengthens the applied-ML fit.",
            "Recommendation-related work provides a relevant product-ML signal.",
            "The recorded recommendation experience is useful for user-facing ML systems.",
        ),
        "retrieval": (
            "Retrieval-heavy experience strengthens the fit for search and candidate-matching systems.",
            "The profile's retrieval work is relevant to relevance-driven ML products.",
            "Information-retrieval evidence supports the search side of the role.",
            "Recorded retrieval experience provides a useful signal for matching systems.",
            "Search and retrieval work adds directly relevant systems context.",
        ),
        "ranking": (
            "Ranking-systems evidence is useful for relevance-oriented ML work.",
            "The profile includes ranking work that maps well to relevance optimization.",
            "Experience with ranking systems adds an applicable ML signal.",
            "Recorded ranking work supports the relevance-focused aspects of the role.",
            "Ranking-related experience provides useful context for search quality problems.",
        ),
        "production": (
            "Production deployment evidence connects model development with operating ML systems.",
            "The profile indicates experience moving ML work toward production use.",
            "Recorded production work supports the delivery side of the ML role.",
            "Deployment evidence suggests exposure beyond model experimentation.",
            "Production-oriented experience adds confidence in practical ML delivery.",
        ),
        "platform": (
            "ML or data-platform experience supports the systems side of the role.",
            "The profile's platform work is relevant to dependable ML infrastructure.",
            "Platform-oriented experience adds useful context for production ML systems.",
            "Recorded data-platform work supports the operational requirements of the role.",
            "Systems and platform evidence complements the candidate's ML background.",
        ),
    }

    MATCH_TEMPLATES = (
        "Structured matches include {skills}.",
        "The profile explicitly identifies {skills}.",
        "Relevant recorded skills include {skills}.",
        "Skill evidence is strongest around {skills}.",
        "The structured skill set covers {skills}.",
        "Named technical strengths include {skills}.",
    )

    UNCERTAINTY_TEMPLATES = (
        "{skills} {verb} not explicitly identified in the structured profile.",
        "Hands-on experience with {skills} should be confirmed during screening.",
        "The profile suggests relevant experience, although explicit evidence for {skills} is limited.",
        "Structured evidence for {skills} is limited and merits follow-up.",
        "A screening conversation should clarify the candidate's depth with {skills}.",
        "The available fields do not clearly establish experience with {skills}.",
        "Direct evidence for {skills} is not prominent in the structured profile.",
        "The candidate may have adjacent experience, but {skills} should be verified.",
    )

    BEHAVIOR_TEMPLATES = (
        "As a tie-breaker, recruiter-readiness is supported by {facts}.",
        "Recruiter-facing signals add confidence through {facts}.",
        "For outreach, the profile also offers {facts}.",
        "Secondary behavioral evidence is favorable: {facts}.",
        "Outreach may be easier given {facts}.",
        "Recruiting practicality is helped by {facts}.",
        "The profile also carries useful outreach signals, including {facts}.",
        "As a secondary consideration, {facts} supports recruiter confidence.",
    )

    # Six variants per tone band provide 18 closing templates.
    CLOSING_TEMPLATES = {
        "high": (
            "This profile appears technically competitive for the next stage.",
            "The structured profile indicates a promising match.",
            "Remaining uncertainties are limited to evidence that can be checked in interview.",
            "An interview can now focus on depth and production ownership.",
            "The evidence supports advancing this profile for technical discussion.",
            "Technical depth should be validated in interview, but the overall fit is strong.",
        ),
        "medium": (
            "Technical depth should be validated during interview.",
            "Further discussion should verify the scope of production ownership.",
            "A focused screen can confirm the remaining tool and delivery details.",
            "The profile is worth screening, with a few evidence points still to confirm.",
            "Interview discussion should test how the recorded experience transfers to this role.",
            "The structured evidence supports consideration, subject to technical validation.",
        ),
        "cautious": (
            "A screening conversation would clarify whether the adjacent experience is sufficient.",
            "The remaining uncertainties warrant a targeted technical screen.",
            "Further review should focus on direct ownership of the role's core requirements.",
            "The profile may merit discussion, but technical depth needs clearer evidence.",
            "A short screen can determine whether the transferable experience is deep enough.",
            "The available evidence is mixed, so advancement should depend on technical confirmation.",
        ),
    }

    def generate(self, candidate: Candidate, result: HybridScoreResult, parsed_job: ParsedJob) -> str:
        parts = [self._opening_clause(candidate, result)]
        specialism = self._specialism_clause(candidate, result)
        if specialism:
            parts.append(specialism)
        skill_clause = self._skill_clause(candidate.candidate_id, result, parsed_job)
        if skill_clause:
            parts.append(skill_clause)
        behavior = self._behavior_clause(candidate, result)
        if behavior:
            parts.append(behavior)
        parts.append(self._closing_clause(candidate.candidate_id, result))
        return " ".join(part.strip() for part in parts if part).strip()

    def _pick(self, candidate_id: str, choices: Sequence[str], salt: str = "") -> str:
        digest = hashlib.sha256(f"{candidate_id}:{salt}".encode("utf-8")).digest()
        return choices[int.from_bytes(digest[:4], "big") % len(choices)]

    def _opening_clause(self, candidate: Candidate, result: HybridScoreResult) -> str:
        if result.weighted_final_score >= 0.82:
            band = "excellent"
        elif result.weighted_final_score >= 0.72:
            band = "strong"
        elif result.weighted_final_score >= 0.60:
            band = "technical"
        else:
            band = "partial"
        template = self._pick(candidate.candidate_id, self.OPENING_TEMPLATES[band], "opening")
        return template.format(
            title=candidate.profile.current_title or "technical professional",
            years=float(candidate.profile.years_of_experience or 0.0),
            industry=candidate.profile.current_industry or "the relevant industry",
        )

    def _specialism_clause(self, candidate: Candidate, result: HybridScoreResult) -> str:
        evidence = " ".join(result.matched_items + result.reasons + [
            career.description or "" for career in candidate.career_history or []
        ]).lower()
        families = []
        if "recommend" in evidence:
            families.append("recommendation")
        if "retrieval" in evidence:
            families.append("retrieval")
        if re.search(r"\branking\b|learning to rank", evidence):
            families.append("ranking")
        if "production" in evidence or "deploy" in evidence:
            families.append("production")
        if "platform" in evidence:
            families.append("platform")
        if not families:
            return ""
        family = self._pick(candidate.candidate_id, families, "middle-family")
        return self._pick(candidate.candidate_id, self.MIDDLE_TEMPLATES[family], f"middle-{family}")

    def _skill_clause(self, candidate_id: str, result: HybridScoreResult, parsed_job: ParsedJob) -> str:
        required = list(parsed_job.job_description.required_skills or [])
        matched_keys = {self._normalize(item) for item in result.matched_items}
        missing_keys = {self._normalize(item) for item in result.missing_items}
        matched = [
            skill for skill in required
            if self._normalize(skill) in matched_keys and self._normalize(skill) not in missing_keys
        ]
        uncertain = [skill for skill in required if self._normalize(skill) in missing_keys]
        clauses = []
        if matched:
            template = self._pick(candidate_id, self.MATCH_TEMPLATES, "skill-match")
            clauses.append(template.format(skills=self._format_list(matched[:4])))
        if uncertain:
            template = self._pick(candidate_id, self.UNCERTAINTY_TEMPLATES, "skill-uncertainty")
            clauses.append(template.format(
                skills=self._format_list(uncertain[:3]),
                verb="was" if len(uncertain[:3]) == 1 else "were",
            ))
        return " ".join(clauses)

    def _behavior_clause(self, candidate: Candidate, result: HybridScoreResult) -> str:
        signals = candidate.redrob_signals
        facts = []
        if signals.open_to_work_flag:
            facts.append("open-to-work status")
        if signals.verified_email and signals.verified_phone:
            facts.append("verified email and phone")
        if signals.recruiter_response_rate is not None and signals.recruiter_response_rate >= 0.70:
            facts.append(f"a {signals.recruiter_response_rate:.0%} recruiter response rate")
        if signals.notice_period_days is not None and signals.notice_period_days <= 30:
            facts.append(f"a {signals.notice_period_days}-day notice period")
        if signals.profile_completeness_score >= 0.85:
            facts.append("a highly complete profile")
        if not facts:
            return ""
        template = self._pick(candidate.candidate_id, self.BEHAVIOR_TEMPLATES, "behavior")
        return template.format(facts=self._format_list(facts[:3]))

    def _closing_clause(self, candidate_id: str, result: HybridScoreResult) -> str:
        if result.weighted_final_score >= 0.75:
            band = "high"
        elif result.weighted_final_score >= 0.60:
            band = "medium"
        else:
            band = "cautious"
        return self._pick(candidate_id, self.CLOSING_TEMPLATES[band], "closing")

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _format_list(self, items: List[str]) -> str:
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"
