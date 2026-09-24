"""IFRS 9 Expected Credit Loss (ECL) Calculation & Multi-Scenario Staging Engine.

Implements the official IFRS 9 impairment standards:
- Three-Stage Classification (Performing, SICR, Credit-Impaired)
- Forward-Looking Point-in-Time (PIT) Vasicek Macroeconomic Conditioning
- Effective Interest Rate (EIR) Discounting over 12-month and Lifetime horizons
- Probability-Weighted Scenario Provisioning (Baseline, Adverse, Favorable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.models.survival_pd import SurvivalPDEngine, LifetimePDCurve


@dataclass
class MacroScenario:
    """Macroeconomic scenario parameters under IFRS 9 forward-looking mandate."""
    name: str
    weight: float
    gdp_growth: float
    unemployment_rate: float
    central_bank_rate: float
    vasicek_z: float  # Latent systemic factor: 0.0=base, -1.645=severe downturn


@dataclass
class LoanECLResult:
    """Individual loan IFRS 9 staging and provision calculation."""
    application_id: str
    stage: int  # 1, 2, or 3
    sicr_triggered: bool
    ead: float
    lgd: float
    pd_12m: float
    lifetime_pd: float
    weighted_ecl: float
    ecl_baseline: float
    ecl_adverse: float
    ecl_favorable: float
    coverage_ratio: float


@dataclass
class PortfolioECLSummary:
    """Executive portfolio-level IFRS 9 impairment summary."""
    total_exposure: float
    total_weighted_ecl: float
    total_coverage_ratio: float
    stage1_exposure: float
    stage1_ecl: float
    stage1_coverage_ratio: float
    stage1_count: int
    stage2_exposure: float
    stage2_ecl: float
    stage2_coverage_ratio: float
    stage2_count: int
    stage3_exposure: float
    stage3_ecl: float
    stage3_coverage_ratio: float
    stage3_count: int
    baseline_ecl_total: float
    adverse_ecl_total: float
    favorable_ecl_total: float
    adverse_ecl_delta_pct: float


class IFRS9ECLEngine:
    """Enterprise IFRS 9 Impairment and Multi-Scenario ECL Engine."""

    def __init__(
        self,
        effective_interest_rate: float = 0.065,
        sicr_pd_ratio_threshold: float = 2.0,
        asset_correlation: float = 0.15,
        max_projection_years: int = 5
    ):
        """
        Args:
            effective_interest_rate: Annual contractual effective interest rate (EIR) for discounting.
            sicr_pd_ratio_threshold: Relative lifetime PD increase threshold trigger for Stage 2 (SICR).
            asset_correlation: Vasicek single-risk-factor systemic asset correlation (rho).
            max_projection_years: Maximum lifetime horizon in years.
        """
        self.eir = effective_interest_rate
        self.sicr_ratio_threshold = sicr_pd_ratio_threshold
        self.rho = asset_correlation
        self.survival_engine = SurvivalPDEngine(baseline_aging_shape=1.05, max_years=max_projection_years)

        # Standard Forward-Looking Scenarios
        self.scenarios = [
            MacroScenario(
                name="Baseline",
                weight=0.50,
                gdp_growth=2.1,
                unemployment_rate=4.2,
                central_bank_rate=4.5,
                vasicek_z=0.0
            ),
            MacroScenario(
                name="Severe Downturn",
                weight=0.30,
                gdp_growth=-1.8,
                unemployment_rate=7.5,
                central_bank_rate=6.2,
                vasicek_z=-1.645
            ),
            MacroScenario(
                name="Macro Expansion",
                weight=0.20,
                gdp_growth=3.9,
                unemployment_rate=3.4,
                central_bank_rate=3.75,
                vasicek_z=1.282
            )
        ]

    def vasicek_pit_pd(self, pd_ttc: float, z: float) -> float:
        """Translate Through-the-Cycle (TTC) baseline PD to Point-in-Time (PIT) conditional PD.

        Formula:
            PD(Z) = Phi( (Phi^(-1)(PD_TTC) - sqrt(rho)*Z) / sqrt(1 - rho) )
        """
        pd_safe = float(np.clip(pd_ttc, 0.0001, 0.999))
        phi_inv = norm.ppf(pd_safe)
        numerator = phi_inv - np.sqrt(self.rho) * z
        denominator = np.sqrt(1.0 - self.rho)
        pit_pd = norm.cdf(numerator / denominator)
        return float(np.clip(pit_pd, 0.0001, 0.999))

    def classify_stage(
        self,
        days_past_due: int,
        current_pd_12m: float,
        origination_pd_12m: float,
        is_default_flag: int = 0
    ) -> Tuple[int, bool]:
        """Classify loan into IFRS 9 Stage 1, Stage 2, or Stage 3.

        Returns:
            Tuple of (stage [1, 2, 3], sicr_triggered [bool])
        """
        # Stage 3: Credit-Impaired / Default
        if days_past_due >= 90 or is_default_flag == 1:
            return 3, True

        # Stage 2: Significant Increase in Credit Risk (SICR)
        # Criteria: 30 <= DPD < 90 OR Relative PD increase >= threshold
        pd_ratio = current_pd_12m / max(origination_pd_12m, 0.001)
        if days_past_due >= 30 or pd_ratio >= self.sicr_ratio_threshold:
            return 2, True

        # Stage 1: Performing (Low credit risk, 12-month ECL)
        return 1, False

    def calculate_loan_ecl(
        self,
        loan_id: str,
        stage: int,
        ead: float,
        lgd: float,
        base_pd_12m: float,
        tenor_months: int = 60
    ) -> Dict[str, float]:
        """Compute discounted ECL for a loan across all macroeconomic scenarios."""
        scenario_ecls: Dict[str, float] = {}

        for scn in self.scenarios:
            pit_pd = self.vasicek_pit_pd(base_pd_12m, scn.vasicek_z)

            if stage == 1:
                # 12-Month ECL discounted at EIR
                ecl_12m = (pit_pd * lgd * ead) / (1.0 + self.eir)
                scenario_ecls[scn.name] = max(ecl_12m, 0.0)
            else:
                # Lifetime ECL (Stage 2 or 3) discounted over projection horizon
                curve = self.survival_engine.generate_lifetime_curve(
                    pd_12m=pit_pd,
                    loan_id=loan_id,
                    tenor_months=tenor_months
                )
                lifetime_ecl = 0.0
                for yr, marg_pd in zip(curve.projection_years, curve.marginal_default_probabilities):
                    discount_factor = (1.0 + self.eir) ** yr
                    # If Stage 3, default already occurred or imminent, loss conditioned on full exposure
                    effective_marg_pd = marg_pd if stage == 2 else (1.0 if yr == 1 else 0.0)
                    lifetime_ecl += (effective_marg_pd * lgd * ead) / discount_factor

                scenario_ecls[scn.name] = max(lifetime_ecl, 0.0)

        # Weighted average across scenarios
        weighted_ecl = sum(
            scenario_ecls[scn.name] * scn.weight for scn in self.scenarios
        )

        return {
            "weighted_ecl": round(weighted_ecl, 2),
            "baseline": round(scenario_ecls["Baseline"], 2),
            "adverse": round(scenario_ecls["Severe Downturn"], 2),
            "favorable": round(scenario_ecls["Macro Expansion"], 2)
        }

    def process_portfolio(
        self,
        df_loans: pd.DataFrame,
        calibrated_pds: np.ndarray,
        origination_pds: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, PortfolioECLSummary]:
        """Run batch IFRS 9 staging and ECL provisioning over full portfolio.

        Args:
            df_loans: Portfolio DataFrame containing EAD, LGD, DPD, Tenors.
            calibrated_pds: Calibrated 12-month PDs at observation date.
            origination_pds: 12-month PDs at origination. If None, uses baseline factor.

        Returns:
            Tuple of (detailed results DataFrame, PortfolioECLSummary).
        """
        n_loans = len(df_loans)
        if origination_pds is None:
            # Synthetic origination PD: 70% of current PD on average
            origination_pds = calibrated_pds * 0.70

        results = []
        for i in range(n_loans):
            row = df_loans.iloc[i]
            app_id = str(row.get("application_id", f"LOAN-{i}"))
            dpd = int(row.get("days_past_due", 0))
            is_def = int(row.get("default_12m", 0))
            ead = float(row.get("exposure_at_default", 10000.0))
            lgd = float(row.get("loss_given_default", 0.45))
            tenor = int(row.get("tenor_months", 60))

            cur_pd = float(calibrated_pds[i])
            orig_pd = float(origination_pds[i])

            stage, sicr = self.classify_stage(
                days_past_due=dpd,
                current_pd_12m=cur_pd,
                origination_pd_12m=orig_pd,
                is_default_flag=is_def
            )

            ecl_dict = self.calculate_loan_ecl(
                loan_id=app_id,
                stage=stage,
                ead=ead,
                lgd=lgd,
                base_pd_12m=cur_pd,
                tenor_months=tenor
            )

            # Generate lifetime PD for reporting
            curve = self.survival_engine.generate_lifetime_curve(cur_pd, app_id, tenor)
            lifetime_pd = curve.cumulative_default_probabilities[-1]

            cov_ratio = (ecl_dict["weighted_ecl"] / max(ead, 1.0))

            results.append({
                "application_id": app_id,
                "stage": stage,
                "sicr_triggered": sicr,
                "days_past_due": dpd,
                "exposure_at_default": round(ead, 2),
                "loss_given_default": round(lgd, 4),
                "pd_12m": round(cur_pd, 5),
                "lifetime_pd": round(lifetime_pd, 5),
                "weighted_ecl": ecl_dict["weighted_ecl"],
                "ecl_baseline": ecl_dict["baseline"],
                "ecl_adverse": ecl_dict["adverse"],
                "ecl_favorable": ecl_dict["favorable"],
                "coverage_ratio": round(cov_ratio, 4)
            })

        res_df = pd.DataFrame(results)

        # Portfolio Aggregations
        total_exp = float(res_df["exposure_at_default"].sum())
        total_ecl = float(res_df["weighted_ecl"].sum())
        total_cov = total_ecl / max(total_exp, 1.0)

        s1_df = res_df[res_df["stage"] == 1]
        s2_df = res_df[res_df["stage"] == 2]
        s3_df = res_df[res_df["stage"] == 3]

        s1_exp = float(s1_df["exposure_at_default"].sum())
        s1_ecl = float(s1_df["weighted_ecl"].sum())
        s2_exp = float(s2_df["exposure_at_default"].sum())
        s2_ecl = float(s2_df["weighted_ecl"].sum())
        s3_exp = float(s3_df["exposure_at_default"].sum())
        s3_ecl = float(s3_df["weighted_ecl"].sum())

        base_tot = float(res_df["ecl_baseline"].sum())
        adv_tot = float(res_df["ecl_adverse"].sum())
        fav_tot = float(res_df["ecl_favorable"].sum())
        delta_pct = ((adv_tot - base_tot) / max(base_tot, 1.0)) * 100.0

        summary = PortfolioECLSummary(
            total_exposure=round(total_exp, 2),
            total_weighted_ecl=round(total_ecl, 2),
            total_coverage_ratio=round(total_cov, 4),
            stage1_exposure=round(s1_exp, 2),
            stage1_ecl=round(s1_ecl, 2),
            stage1_coverage_ratio=round(s1_ecl / max(s1_exp, 1.0), 4),
            stage1_count=len(s1_df),
            stage2_exposure=round(s2_exp, 2),
            stage2_ecl=round(s2_ecl, 2),
            stage2_coverage_ratio=round(s2_ecl / max(s2_exp, 1.0), 4),
            stage2_count=len(s2_df),
            stage3_exposure=round(s3_exp, 2),
            stage3_ecl=round(s3_ecl, 2),
            stage3_coverage_ratio=round(s3_ecl / max(s3_exp, 1.0), 4),
            stage3_count=len(s3_df),
            baseline_ecl_total=round(base_tot, 2),
            adverse_ecl_total=round(adv_tot, 2),
            favorable_ecl_total=round(fav_tot, 2),
            adverse_ecl_delta_pct=round(delta_pct, 2)
        )

        return res_df, summary
