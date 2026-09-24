"""Survival Analysis and Multi-Horizon Lifetime PD Term Structure Engine.

Under IFRS 9, Stage 2 and Stage 3 instruments require Lifetime Expected Credit Loss.
This module translates 1-year default probabilities and borrower covariates into
multi-period Survival S(t), Cumulative Default F(t), and Marginal Default PD(t) curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class LifetimePDCurve:
    """Multi-year credit default probability term structure."""
    loan_id: str
    projection_years: List[int]
    survival_probabilities: List[float]       # S(t) = P(T > t)
    cumulative_default_probabilities: List[float]  # F(t) = 1 - S(t)
    marginal_default_probabilities: List[float]    # PD(t) = S(t-1) - S(t)
    forward_conditional_pds: List[float]          # h(t) = PD(t) / S(t-1)


class SurvivalPDEngine:
    """Models multi-period hazard rates and lifetime default term structures for IFRS 9."""

    def __init__(self, baseline_aging_shape: float = 1.05, max_years: int = 5):
        """
        Args:
            baseline_aging_shape: Weibull/gamma shape parameter reflecting portfolio seasoning.
            max_years: Maximum lifetime projection horizon in years (default: 5 years).
        """
        self.aging_shape = baseline_aging_shape
        self.max_years = max_years

    def generate_lifetime_curve(
        self,
        pd_12m: float,
        loan_id: str = "LOAN-001",
        tenor_months: int = 60,
        macro_adjustment_factor: float = 1.0
    ) -> LifetimePDCurve:
        """Construct multi-year survival and marginal default curve for a single loan.

        Args:
            pd_12m: Calibrated 12-month probability of default (0 < pd_12m < 1).
            loan_id: Identifier for reference.
            tenor_months: Contractual loan duration in months.
            macro_adjustment_factor: Forward-looking multiplier (>1.0 for downturn, <1.0 for expansion).

        Returns:
            LifetimePDCurve dataclass.
        """
        pd_12m_adj = float(np.clip(pd_12m * macro_adjustment_factor, 0.0001, 0.999))
        
        # Base annual hazard intensity: S(1) = 1 - pd_12m_adj = exp(-lambda) => lambda = -ln(1 - pd)
        base_lambda = -np.log(1.0 - pd_12m_adj)

        max_horizon_years = min(self.max_years, int(np.ceil(tenor_months / 12.0)))
        years = list(range(1, max_horizon_years + 1))

        surv_probs: List[float] = []
        cum_pds: List[float] = []
        marg_pds: List[float] = []
        fwd_pds: List[float] = []

        prev_s = 1.0
        for t in years:
            # Weibull-type cumulative hazard: H(t) = base_lambda * (t ** aging_shape)
            cum_hazard = base_lambda * (t ** self.aging_shape)
            s_t = float(np.exp(-cum_hazard))
            # Bound survival probability
            s_t = max(min(s_t, prev_s), 0.0001)

            cum_pd = 1.0 - s_t
            marg_pd = prev_s - s_t
            fwd_pd = marg_pd / prev_s if prev_s > 0 else 0.0

            surv_probs.append(round(s_t, 5))
            cum_pds.append(round(cum_pd, 5))
            marg_pds.append(round(marg_pd, 5))
            fwd_pds.append(round(fwd_pd, 5))

            prev_s = s_t

        return LifetimePDCurve(
            loan_id=loan_id,
            projection_years=years,
            survival_probabilities=surv_probs,
            cumulative_default_probabilities=cum_pds,
            marginal_default_probabilities=marg_pds,
            forward_conditional_pds=fwd_pds
        )

    def generate_portfolio_lifetime_pds(
        self,
        df_loans: pd.DataFrame,
        pd_12m_series: pd.Series,
        macro_adjustment_factor: float = 1.0
    ) -> pd.DataFrame:
        """Batch-generate marginal default probabilities across all years for portfolio.

        Returns DataFrame with columns:
        [application_id, pd_12m, marg_pd_y1, marg_pd_y2, ..., lifetime_pd]
        """
        records = []
        for i in range(len(df_loans)):
            app_id = str(df_loans.iloc[i].get("application_id", f"LOAN-{i}"))
            tenor = int(df_loans.iloc[i].get("tenor_months", 60))
            pd_1 = float(pd_12m_series.iloc[i])

            curve = self.generate_lifetime_curve(
                pd_12m=pd_1,
                loan_id=app_id,
                tenor_months=tenor,
                macro_adjustment_factor=macro_adjustment_factor
            )

            rec: Dict[str, float] = {
                "application_id": app_id,
                "pd_12m": round(pd_1, 5),
                "lifetime_pd": round(curve.cumulative_default_probabilities[-1], 5)
            }
            for y_idx, yr in enumerate(curve.projection_years):
                rec[f"marg_pd_y{yr}"] = curve.marginal_default_probabilities[y_idx]

            # Fill remaining years if loan matured earlier
            for yr in range(len(curve.projection_years) + 1, self.max_years + 1):
                rec[f"marg_pd_y{yr}"] = 0.0

            records.append(rec)

        return pd.DataFrame(records)
