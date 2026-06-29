"""Redrob signals domain model."""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional


@dataclass(frozen=True)
class SalaryRange:
    """Represents expected salary range."""

    min: float
    max: float


@dataclass(frozen=True)
class RedrobSignals:
    """Represents Redrob platform signals for a candidate."""

    profile_completeness_score: float
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: Optional[float]
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: Optional[float]
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool
