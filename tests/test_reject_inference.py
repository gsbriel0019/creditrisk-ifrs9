"""Tests for RejectInferenceEngine."""

import pytest
import pandas as pd
import numpy as np
from src.data.generator import CreditPortfolioGenerator
from src.models.reject_inference import RejectInferenceEngine


@pytest.fixture
def portfolio_data():
    gen = CreditPortfolioGenerator(random_seed=77)
    portfolio = gen.generate(n_samples=500, reject_rate=0.25)
    return portfolio.accepted_loans, portfolio.rejected_applications


def test_reject_inference_hard_cutoff(portfolio_data):
    acc, rej = portfolio_data
    features = ["bureau_score", "debt_to_income"]

    engine = RejectInferenceEngine(method="hard_cutoff")
    aug_X, aug_y, weights = engine.infer_and_augment(
        accepted_X=acc[features],
        accepted_y=acc["default_12m"],
        rejected_X=rej[features],
        feature_cols=features
    )

    assert len(aug_X) == len(acc) + len(rej)
    assert len(aug_y) == len(aug_X)
    assert len(weights) == len(aug_X)
    assert aug_y.isin([0, 1]).all()


def test_reject_inference_parceling(portfolio_data):
    acc, rej = portfolio_data
    features = ["bureau_score", "debt_to_income"]

    engine = RejectInferenceEngine(method="parceling", parceling_multiplier=1.5)
    aug_X, aug_y, weights = engine.infer_and_augment(
        accepted_X=acc[features],
        accepted_y=acc["default_12m"],
        rejected_X=rej[features],
        feature_cols=features
    )

    assert len(aug_X) == len(acc) + len(rej)
    assert len(aug_y) == len(aug_X)
    # The inferred rejected default rate should be reasonable
    rej_inferred_y = aug_y.iloc[len(acc):]
    assert rej_inferred_y.mean() > 0.0


def test_reject_inference_fuzzy_augmentation(portfolio_data):
    acc, rej = portfolio_data
    features = ["bureau_score", "debt_to_income"]

    engine = RejectInferenceEngine(method="fuzzy_augmentation")
    aug_X, aug_y, weights = engine.infer_and_augment(
        accepted_X=acc[features],
        accepted_y=acc["default_12m"],
        rejected_X=rej[features],
        feature_cols=features
    )

    # In fuzzy augmentation, rejected are duplicated with soft weights
    assert len(aug_X) == len(acc) + 2 * len(rej)
    assert len(weights) == len(aug_X)
    assert (weights >= 0.0).all() and (weights <= 1.0).all()
