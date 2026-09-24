"""Tests for ScorecardTransformer (WoE, IV, and Points Scaling)."""

import pytest
import numpy as np
import pandas as pd
from src.features.scorecard import ScorecardTransformer
from src.data.generator import CreditPortfolioGenerator


@pytest.fixture
def sample_credit_data():
    gen = CreditPortfolioGenerator(random_seed=123)
    portfolio = gen.generate(n_samples=600)
    acc = portfolio.accepted_loans
    features = ["bureau_score", "debt_to_income", "delinquencies_2yrs", "annual_income"]
    return acc[features], acc["default_12m"]


def test_scorecard_fit_and_iv_calculation(sample_credit_data):
    X, y = sample_credit_data
    sc = ScorecardTransformer(target_score=600, target_odds=50, pdo=20)
    sc.fit(X, y)

    assert len(sc.iv_summary) == 4
    # All features should have non-negative IV
    for summary in sc.iv_summary:
        assert summary.information_value >= 0.0
        assert summary.num_bins >= 2

    # High predictive features like bureau_score or dti should have IV > 0.05
    bureau_summary = next(s for s in sc.iv_summary if s.feature_name == "bureau_score")
    assert bureau_summary.information_value > 0.05


def test_scorecard_woe_transformation(sample_credit_data):
    X, y = sample_credit_data
    sc = ScorecardTransformer()
    sc.fit(X, y)
    X_woe = sc.transform_woe(X)

    assert len(X_woe) == len(X)
    assert not X_woe.isnull().any().any()
    # WoE columns should have suffix _woe
    for col in X_woe.columns:
        assert col.endswith("_woe")


def test_scorecard_points_scaling(sample_credit_data):
    X, y = sample_credit_data
    sc = ScorecardTransformer(target_score=600, target_odds=50, pdo=20)
    sc.fit(X, y)

    # Mock logistic coefficients
    coefs = {f"{col}_woe": -0.8 for col in sc.selected_features}
    sc.set_model_weights(intercept=-1.5, coefficients=coefs)

    scores, pts_df = sc.transform_score(X)
    assert len(scores) == len(X)
    # Check score bounds (300 to 850)
    assert (scores >= 300).all()
    assert (scores <= 850).all()

    table = sc.get_scorecard_table()
    assert not table.empty
    assert "Assigned Points" in table.columns
