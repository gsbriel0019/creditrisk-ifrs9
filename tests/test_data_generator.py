"""Tests for CreditPortfolioGenerator."""

import pytest
import numpy as np
import pandas as pd
from src.data.generator import CreditPortfolioGenerator, LoanPortfolio


def test_generator_structure_and_counts():
    gen = CreditPortfolioGenerator(random_seed=42)
    portfolio = gen.generate(n_samples=500, reject_rate=0.20)

    assert isinstance(portfolio, LoanPortfolio)
    assert len(portfolio.all_applications) == 500
    assert len(portfolio.accepted_loans) + len(portfolio.rejected_applications) == 500
    assert 0.15 <= len(portfolio.rejected_applications) / 500 <= 0.25


def test_generator_fields_and_ranges():
    gen = CreditPortfolioGenerator(random_seed=42)
    portfolio = gen.generate(n_samples=300)
    df = portfolio.all_applications

    required_cols = [
        "application_id", "loan_type", "loan_amount", "annual_income",
        "debt_to_income", "bureau_score", "exposure_at_default",
        "loss_given_default", "default_12m", "days_past_due"
    ]
    for col in required_cols:
        assert col in df.columns

    # Check realistic boundaries
    assert df["bureau_score"].min() >= 300
    assert df["bureau_score"].max() <= 850
    assert df["debt_to_income"].min() > 0.0
    assert df["loss_given_default"].between(0.0, 1.0).all()
    assert df["default_12m"].isin([0, 1]).all()
