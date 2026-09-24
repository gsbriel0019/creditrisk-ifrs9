"""Tests for AdverseActionEngine."""

import pytest
import pandas as pd
from src.models.explainability import AdverseActionEngine, AdverseActionReport


def test_adverse_action_approval():
    engine = AdverseActionEngine(approval_score_cutoff=580, referral_score_cutoff=540)
    features = pd.Series({"bureau_score": 750, "debt_to_income": 0.20})
    points = pd.Series({"bureau_score_points": 180.0, "debt_to_income_points": 120.0})

    report = engine.generate_report(
        application_id="APP-001",
        applicant_features=features,
        scorecard_points_breakdown=points,
        total_score=680,
        predicted_pd=0.015
    )

    assert isinstance(report, AdverseActionReport)
    assert report.decision == "APPROVED"
    assert len(report.top_adverse_reasons) == 0


def test_adverse_action_rejection_reasons():
    engine = AdverseActionEngine(approval_score_cutoff=580, referral_score_cutoff=540)
    features = pd.Series({"bureau_score": 450, "debt_to_income": 0.65, "delinquencies_2yrs": 3})
    points = pd.Series({
        "bureau_score_points": 45.0,
        "debt_to_income_points": 50.0,
        "delinquencies_2yrs_points": 40.0
    })

    report = engine.generate_report(
        application_id="APP-002",
        applicant_features=features,
        scorecard_points_breakdown=points,
        total_score=480,
        predicted_pd=0.25
    )

    assert report.decision == "REJECTED"
    # ECOA compliance requires adverse reasons when denied
    assert len(report.top_adverse_reasons) > 0
    # Top negative reason should correspond to lowest points feature (delinquencies)
    assert any("delinquencies" in r.lower() or "bureau" in r.lower() for r in report.top_adverse_reasons)
