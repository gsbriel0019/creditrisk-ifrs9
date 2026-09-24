"""Tests for SurvivalPDEngine and Lifetime PD curves."""

import pytest
import numpy as np
import pandas as pd
from src.models.survival_pd import SurvivalPDEngine, LifetimePDCurve


def test_survival_curve_properties():
    engine = SurvivalPDEngine(baseline_aging_shape=1.05, max_years=5)
    curve = engine.generate_lifetime_curve(pd_12m=0.08, loan_id="TEST-01", tenor_months=60)

    assert isinstance(curve, LifetimePDCurve)
    assert len(curve.projection_years) == 5

    # Survival probabilities should be strictly decreasing
    for i in range(len(curve.survival_probabilities) - 1):
        assert curve.survival_probabilities[i] >= curve.survival_probabilities[i + 1]

    # Cumulative default probabilities should be strictly increasing
    for i in range(len(curve.cumulative_default_probabilities) - 1):
        assert curve.cumulative_default_probabilities[i] <= curve.cumulative_default_probabilities[i + 1]

    # Cumulative PD + Survival Prob == 1.0 (approx)
    for s, f in zip(curve.survival_probabilities, curve.cumulative_default_probabilities):
        assert abs((s + f) - 1.0) < 1e-4

    # Marginal PDs should be positive
    for m in curve.marginal_default_probabilities:
        assert m >= 0.0


def test_survival_macro_adjustment():
    engine = SurvivalPDEngine(max_years=5)
    # Severe downturn macro adjustment (factor 1.5) vs expansion (factor 0.7)
    curve_base = engine.generate_lifetime_curve(pd_12m=0.05, macro_adjustment_factor=1.0)
    curve_stress = engine.generate_lifetime_curve(pd_12m=0.05, macro_adjustment_factor=1.5)

    assert curve_stress.cumulative_default_probabilities[-1] > curve_base.cumulative_default_probabilities[-1]


def test_batch_portfolio_lifetime_pds():
    engine = SurvivalPDEngine(max_years=5)
    df_loans = pd.DataFrame([
        {"application_id": "LOAN-1", "tenor_months": 36},
        {"application_id": "LOAN-2", "tenor_months": 60}
    ])
    pds = pd.Series([0.04, 0.08])
    batch_df = engine.generate_portfolio_lifetime_pds(df_loans, pds)

    assert len(batch_df) == 2
    assert "marg_pd_y1" in batch_df.columns
    assert "lifetime_pd" in batch_df.columns
    assert (batch_df["lifetime_pd"] > batch_df["pd_12m"]).all()
