"""Tests for IFRS9ReportGenerator."""

import pytest
from src.reporting.report_generator import IFRS9ReportGenerator
from src.simulation.ecl_engine import PortfolioECLSummary
from src.models.pd_engine import ModelBenchmarkMetrics
from src.features.scorecard import ScorecardSummary


def test_report_generation(tmp_path):
    reporter = IFRS9ReportGenerator(author="Gabriel Proaño", institution="Quantitative Credit Risk")

    summary = PortfolioECLSummary(
        total_exposure=15000000.0,
        total_weighted_ecl=450000.0,
        total_coverage_ratio=0.03,
        stage1_exposure=12000000.0,
        stage1_ecl=120000.0,
        stage1_coverage_ratio=0.01,
        stage1_count=2000,
        stage2_exposure=2500000.0,
        stage2_ecl=200000.0,
        stage2_coverage_ratio=0.08,
        stage2_count=350,
        stage3_exposure=500000.0,
        stage3_ecl=130000.0,
        stage3_coverage_ratio=0.26,
        stage3_count=60,
        baseline_ecl_total=400000.0,
        adverse_ecl_total=580000.0,
        favorable_ecl_total=320000.0,
        adverse_ecl_delta_pct=45.0
    )

    m_log = ModelBenchmarkMetrics(
        model_name="Regulatory Logistic (WoE)",
        roc_auc=0.825,
        gini_coefficient=0.650,
        ks_statistic=0.510,
        ks_optimal_threshold=0.08,
        brier_score=0.045,
        log_loss=0.18,
        expected_calibration_error=0.015
    )

    m_lgb = ModelBenchmarkMetrics(
        model_name="Calibrated LightGBM",
        roc_auc=0.855,
        gini_coefficient=0.710,
        ks_statistic=0.550,
        ks_optimal_threshold=0.075,
        brier_score=0.041,
        log_loss=0.16,
        expected_calibration_error=0.012
    )

    sc_summaries = [
        ScorecardSummary(feature_name="bureau_score", information_value=0.45, predictive_power="Strong", num_bins=5),
        ScorecardSummary(feature_name="debt_to_income", information_value=0.32, predictive_power="Strong", num_bins=5)
    ]

    report = reporter.build_markdown_report(
        portfolio_summary=summary,
        metrics_logistic=m_log,
        metrics_lgb=m_lgb,
        scorecard_summaries=sc_summaries,
        portfolio_metadata={"n_total": 2410}
    )

    assert "IFRS 9 Expected Credit Loss" in report
    assert "Gabriel Proaño" in report
    assert "$15,000,000.00" in report

    out_file = tmp_path / "test_report.md"
    reporter.save_report(report, str(out_file))
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8")) > 500
