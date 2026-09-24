"""Tests for IFRS 9 Staging and ECL Engine."""

import pytest
import numpy as np
import pandas as pd
from src.simulation.ecl_engine import IFRS9ECLEngine, PortfolioECLSummary
from src.data.generator import CreditPortfolioGenerator


def test_ifrs9_staging_logic():
    engine = IFRS9ECLEngine(sicr_pd_ratio_threshold=2.0)

    # Performing loan (0 DPD, low PD increase) -> Stage 1
    s1, sicr1 = engine.classify_stage(days_past_due=0, current_pd_12m=0.03, origination_pd_12m=0.025)
    assert s1 == 1
    assert not sicr1

    # SICR by DPD (35 DPD) -> Stage 2
    s2_dpd, sicr2_dpd = engine.classify_stage(days_past_due=35, current_pd_12m=0.03, origination_pd_12m=0.03)
    assert s2_dpd == 2
    assert sicr2_dpd

    # SICR by PD deterioration (ratio = 0.08 / 0.02 = 4.0 >= 2.0) -> Stage 2
    s2_pd, sicr2_pd = engine.classify_stage(days_past_due=10, current_pd_12m=0.08, origination_pd_12m=0.02)
    assert s2_pd == 2
    assert sicr2_pd

    # Default / Credit Impaired (90+ DPD) -> Stage 3
    s3, sicr3 = engine.classify_stage(days_past_due=95, current_pd_12m=0.20, origination_pd_12m=0.05)
    assert s3 == 3
    assert sicr3


def test_ecl_calculation_monotonicity_across_scenarios():
    engine = IFRS9ECLEngine()
    ecl_dict = engine.calculate_loan_ecl(
        loan_id="TEST-ECL",
        stage=1,
        ead=20000.0,
        lgd=0.45,
        base_pd_12m=0.05,
        tenor_months=36
    )

    # Severe downturn ECL should be higher than Baseline, and Favorable should be lower
    assert ecl_dict["adverse"] > ecl_dict["baseline"]
    assert ecl_dict["favorable"] < ecl_dict["baseline"]
    assert ecl_dict["weighted_ecl"] > 0.0


def test_portfolio_processing_and_summary():
    gen = CreditPortfolioGenerator(random_seed=42)
    portfolio = gen.generate(n_samples=400)
    acc = portfolio.accepted_loans

    calibrated_pds = np.clip(acc["fundamental_pd"].values, 0.01, 0.50)

    engine = IFRS9ECLEngine()
    df_res, summary = engine.process_portfolio(acc, calibrated_pds)

    assert isinstance(summary, PortfolioECLSummary)
    assert len(df_res) == len(acc)
    assert summary.total_exposure > 0.0
    assert summary.total_weighted_ecl > 0.0
    assert 0.0 < summary.total_coverage_ratio < 1.0

    # Stages must sum to total
    assert summary.stage1_count + summary.stage2_count + summary.stage3_count == len(acc)
    assert summary.adverse_ecl_total > summary.baseline_ecl_total
