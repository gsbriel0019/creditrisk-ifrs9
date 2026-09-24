"""End-to-End Enterprise Credit Risk, Scorecard & IFRS 9 CLI Pipeline.

Usage:
    python run_credit_pipeline.py --portfolio-size 3500 --output-dir reports
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data.generator import CreditPortfolioGenerator
from src.features.scorecard import ScorecardTransformer
from src.models.pd_engine import ProbabilityOfDefaultEngine
from src.models.reject_inference import RejectInferenceEngine
from src.models.survival_pd import SurvivalPDEngine
from src.simulation.ecl_engine import IFRS9ECLEngine
from src.reporting.report_generator import IFRS9ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="CreditRisk-IFRS9 Analytics Pipeline")
    parser.add_argument("--portfolio-size", type=int, default=3000, help="Number of credit applications to generate")
    parser.add_argument("--reject-rate", type=float, default=0.25, help="Underwriting rejection rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save audit filings")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("🏛️  CREDITRISK-IFRS9: ENTERPRISE CREDIT RISK & IMPAIRMENT ENGINE")
    print("    Author: Gabriel Proaño | Compliance: IFRS 9 & BCBS Basel III/IV")
    print("=" * 75 + "\n")

    # Step 1: Data Generation
    print("📥 [1/6] Simulating Multi-Asset Credit Portfolio & Selection Process...")
    gen = CreditPortfolioGenerator(random_seed=args.seed)
    portfolio = gen.generate(n_samples=args.portfolio_size, reject_rate=args.reject_rate)
    acc_df = portfolio.accepted_loans
    rej_df = portfolio.rejected_applications
    print(f"    ✓ Total Applications : {args.portfolio_size:,}")
    print(f"    ✓ Accepted Loans     : {len(acc_df):,} (12m Default Rate: {acc_df['default_12m'].mean()*100:.2f}%)")
    print(f"    ✓ Rejected Applicants: {len(rej_df):,}")

    feature_cols = [
        "bureau_score",
        "debt_to_income",
        "delinquencies_2yrs",
        "revolving_utilization",
        "annual_income",
        "employment_years",
        "loan_to_value"
    ]

    # Step 2: Scorecard Engineering (WoE & IV)
    print("\n🎯 [2/6] Engineering Regulatory Scorecard (WoE & Information Value)...")
    sc = ScorecardTransformer(target_score=600, target_odds=50, pdo=20, max_bins=5)
    sc.fit(acc_df[feature_cols], acc_df["default_12m"])
    X_woe = sc.transform_woe(acc_df[feature_cols])

    top_ivs = [[s.feature_name, f"{s.information_value:.4f}", s.predictive_power] for s in sc.iv_summary[:5]]
    print(tabulate(top_ivs, headers=["Feature", "Information Value", "Predictive Power"], tablefmt="simple"))

    # Step 3: Reject Inference Debiasing
    print("\n⚖️ [3/6] Applying Reject Inference (Parceling & Proportional Assignment)...")
    ri_engine = RejectInferenceEngine(method="parceling", parceling_multiplier=1.5, random_state=args.seed)
    aug_X, aug_y, weights = ri_engine.infer_and_augment(
        accepted_X=acc_df[feature_cols],
        accepted_y=acc_df["default_12m"],
        rejected_X=rej_df[feature_cols],
        feature_cols=feature_cols
    )
    print(f"    ✓ Augmented Training Set: {len(aug_X):,} observations (debaised sample)")

    # Step 4: Probability of Default Modeling & Calibration
    print("\n📈 [4/6] Training & Calibrating Probability of Default (PD) Engines...")
    pd_engine = ProbabilityOfDefaultEngine(random_state=args.seed)
    pd_engine.fit(X_woe, acc_df[feature_cols], acc_df["default_12m"], calibration_method="isotonic")

    # Map model weights to Scorecard
    intercept, coefs = pd_engine.get_logistic_coefficients()
    sc.set_model_weights(intercept, coefs)

    m_log, m_lgb = pd_engine.evaluate(X_woe, acc_df[feature_cols], acc_df["default_12m"])
    eval_table = [
        ["Regulatory WoE Logistic", f"{m_log.roc_auc:.4f}", f"{m_log.gini_coefficient:.4f}", f"{m_log.ks_statistic*100:.1f}%", f"{m_log.expected_calibration_error:.4f}"],
        ["Calibrated LightGBM", f"{m_lgb.roc_auc:.4f}", f"{m_lgb.gini_coefficient:.4f}", f"{m_lgb.ks_statistic*100:.1f}%", f"{m_lgb.expected_calibration_error:.4f}"]
    ]
    print(tabulate(eval_table, headers=["Model", "ROC-AUC", "Gini", "KS Stat", "ECE"], tablefmt="simple"))

    # Step 5: IFRS 9 Staging & Forward-Looking ECL Simulation
    print("\n🏦 [5/6] Executing IFRS 9 Three-Stage Impairment & Forward-Looking Stress Engine...")
    calibrated_pds = pd_engine.predict_pd_logistic(X_woe, calibrated=True)
    ecl_engine = IFRS9ECLEngine(effective_interest_rate=0.065, asset_correlation=0.15)
    ecl_df, ecl_summary = ecl_engine.process_portfolio(acc_df, calibrated_pds)

    staging_table = [
        ["Stage 1 (Performing)", f"{ecl_summary.stage1_count:,}", f"${ecl_summary.stage1_exposure:,.2f}", f"${ecl_summary.stage1_ecl:,.2f}", f"{ecl_summary.stage1_coverage_ratio*100:.2f}%"],
        ["Stage 2 (SICR)", f"{ecl_summary.stage2_count:,}", f"${ecl_summary.stage2_exposure:,.2f}", f"${ecl_summary.stage2_ecl:,.2f}", f"{ecl_summary.stage2_coverage_ratio*100:.2f}%"],
        ["Stage 3 (Default)", f"{ecl_summary.stage3_count:,}", f"${ecl_summary.stage3_exposure:,.2f}", f"${ecl_summary.stage3_ecl:,.2f}", f"{ecl_summary.stage3_coverage_ratio*100:.2f}%"],
        ["TOTAL IFRS 9 PROVISION", f"{len(acc_df):,}", f"${ecl_summary.total_exposure:,.2f}", f"${ecl_summary.total_weighted_ecl:,.2f}", f"{ecl_summary.total_coverage_ratio*100:.2f}%"]
    ]
    print(tabulate(staging_table, headers=["Classification", "Count", "Exposure (EAD)", "ECL Provision", "Coverage %"], tablefmt="github"))

    # Step 6: Generate Executive Filing
    print("\n📝 [6/6] Compiling Audit-Ready Filing & Executive Memorandum...")
    reporter = IFRS9ReportGenerator(author="Gabriel Proaño", institution="Enterprise Quantitative Risk")
    report_md = reporter.build_markdown_report(
        portfolio_summary=ecl_summary,
        metrics_logistic=m_log,
        metrics_lgb=m_lgb,
        scorecard_summaries=sc.iv_summary,
        portfolio_metadata=portfolio.metadata
    )

    out_file = Path(args.output_dir) / "IFRS9_Credit_Risk_Audit.md"
    reporter.save_report(report_md, str(out_file))
    print(f"    ✓ Audit Report saved successfully to: {out_file.resolve()}")

    print("\n" + "=" * 75)
    print("✅ CREDITRISK-IFRS9 PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
