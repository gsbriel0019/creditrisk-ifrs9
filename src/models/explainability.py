"""Adverse Action and Model Explainability Engine (ECOA / FCRA Compliant).

Provides local and global feature attribution for credit decisions. Generates
regulatory Adverse Action Reason Codes mandated by ECOA / Regulation B when an applicant
is rejected or subject to risk-based interest rate markups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class AdverseActionReport:
    """Regulatory adverse action disclosure statement for an applicant."""
    application_id: str
    credit_score: int
    predicted_pd: float
    decision: str  # 'APPROVED', 'REFERRED', 'REJECTED'
    top_adverse_reasons: List[str]
    feature_contributions: Dict[str, float]


class AdverseActionEngine:
    """Computes feature attribution and formats compliance reason codes."""

    # Standard FCRA / ECOA regulatory reason codes mapping
    REASON_CODE_CATALOG = {
        "debt_to_income": "Excessive Debt-to-Income (DTI) ratio relative to disposable cash flow",
        "debt_to_income_woe": "Excessive Debt-to-Income (DTI) ratio relative to disposable cash flow",
        "bureau_score": "Insufficient or derogatory historical credit bureau score",
        "bureau_score_woe": "Insufficient or derogatory historical credit bureau score",
        "delinquencies_2yrs": "Recent history of 30+ or 60+ days past due delinquencies",
        "delinquencies_2yrs_woe": "Recent history of 30+ or 60+ days past due delinquencies",
        "revolving_utilization": "High revolving credit line utilization rate (>50%)",
        "revolving_utilization_woe": "High revolving credit line utilization rate (>50%)",
        "annual_income": "Annual income is insufficient for the requested debt service",
        "annual_income_woe": "Annual income is insufficient for the requested debt service",
        "employment_years": "Short duration of current employment / job stability",
        "employment_years_woe": "Short duration of current employment / job stability",
        "loan_to_value": "Elevated Loan-to-Value (LTV) ratio or insufficient collateral margin",
        "loan_to_value_woe": "Elevated Loan-to-Value (LTV) ratio or insufficient collateral margin",
        "credit_lines_count": "Limited or thin credit file history",
        "credit_lines_count_woe": "Limited or thin credit file history"
    }

    def __init__(
        self,
        approval_score_cutoff: int = 580,
        referral_score_cutoff: int = 540
    ):
        self.approval_cutoff = approval_score_cutoff
        self.referral_cutoff = referral_score_cutoff

    def generate_report(
        self,
        application_id: str,
        applicant_features: pd.Series,
        scorecard_points_breakdown: pd.Series,
        total_score: int,
        predicted_pd: float,
        top_k: int = 4
    ) -> AdverseActionReport:
        """Generate institutional adverse action report with ranked reasons.

        Args:
            application_id: Unique application identifier.
            applicant_features: Raw applicant attributes.
            scorecard_points_breakdown: Series of points awarded per feature.
            total_score: Total computed credit score (300 - 850).
            predicted_pd: Calibrated 12-month default probability.
            top_k: Number of regulatory reasons to produce (ECOA mandates 3-4).
        """
        # Determine underwriting decision
        if total_score >= self.approval_cutoff:
            decision = "APPROVED"
        elif total_score >= self.referral_cutoff:
            decision = "REFERRED_MANUAL_REVIEW"
        else:
            decision = "REJECTED"

        # Find features that contributed least points (most negative/risk drivers)
        # Lower scorecard points = higher risk contribution
        points_clean = scorecard_points_breakdown.copy()
        
        # Strip suffix '_points' or '_woe' to match reason catalog
        reasons_ranked = []
        contrib_dict = {}

        # Sort features ascending by points (lowest points = highest negative impact)
        sorted_items = sorted(points_clean.items(), key=lambda item: item[1])

        for feat_name, pts in sorted_items:
            base_col = feat_name.replace("_points", "").replace("_woe", "")
            contrib_dict[base_col] = round(float(pts), 2)
            desc = self.REASON_CODE_CATALOG.get(
                base_col,
                f"Adverse risk profile in factor: {base_col.replace('_', ' ').title()}"
            )
            if desc not in reasons_ranked:
                reasons_ranked.append(desc)

        top_reasons = reasons_ranked[:top_k] if decision != "APPROVED" else []

        return AdverseActionReport(
            application_id=application_id,
            credit_score=int(total_score),
            predicted_pd=round(float(predicted_pd), 5),
            decision=decision,
            top_adverse_reasons=top_reasons,
            feature_contributions=contrib_dict
        )
