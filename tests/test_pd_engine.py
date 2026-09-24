"""Tests for ProbabilityOfDefaultEngine and Probability Calibration."""

import pytest
import numpy as np
import pandas as pd
from src.data.generator import CreditPortfolioGenerator
from src.features.scorecard import ScorecardTransformer
from src.models.pd_engine import ProbabilityOfDefaultEngine


@pytest.fixture
def credit_dataset():
    gen = CreditPortfolioGenerator(random_seed=99)
    portfolio = gen.generate(n_samples=800)
    acc = portfolio.accepted_loans
    features = ["bureau_score", "debt_to_income", "delinquencies_2yrs"]

    sc = ScorecardTransformer()
    sc.fit(acc[features], acc["default_12m"])
    X_woe = sc.transform_woe(acc[features])
    return X_woe, acc[features], acc["default_12m"]


def test_pd_engine_fit_and_metrics(credit_dataset):
    X_woe, X_raw, y = credit_dataset
    engine = ProbabilityOfDefaultEngine(random_state=42)
    engine.fit(X_woe, X_raw, y, calibration_method="isotonic")

    m_log, m_lgb = engine.evaluate(X_woe, X_raw, y)

    # Discrimination test
    assert m_log.roc_auc >= 0.70
    assert m_lgb.roc_auc >= 0.70
    assert m_log.gini_coefficient >= 0.40
    assert m_log.ks_statistic >= 0.30

    # Calibration test
    assert m_log.brier_score < 0.25
    assert m_log.expected_calibration_error < 0.15


def test_pd_engine_predictions_bounded(credit_dataset):
    X_woe, X_raw, y = credit_dataset
    engine = ProbabilityOfDefaultEngine(random_state=42)
    engine.fit(X_woe, X_raw, y)

    probs_log = engine.predict_pd_logistic(X_woe, calibrated=True)
    probs_lgb = engine.predict_pd_lgb(X_raw, calibrated=True)

    assert len(probs_log) == len(y)
    assert len(probs_lgb) == len(y)
    assert (probs_log >= 0.0).all() and (probs_log <= 1.0).all()
    assert (probs_lgb >= 0.0).all() and (probs_lgb <= 1.0).all()
