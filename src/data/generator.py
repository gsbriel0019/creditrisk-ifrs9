"""Credit portfolio simulation and synthetic data generator.

Implements realistic retail, SME, and mortgage credit portfolios with:
- Standard risk factors (DTI, Bureau Score, Delinquencies, LTV, Income)
- Macroeconomic conditioning (GDP, Unemployment, Interest Rates)
- Selection mechanism for accepted vs. rejected applications (Reject Inference)
- Time-to-default and multi-horizon performance flags for IFRS 9 and Survival Modeling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class LoanPortfolio:
    """Container for generated portfolio data."""
    all_applications: pd.DataFrame
    accepted_loans: pd.DataFrame
    rejected_applications: pd.DataFrame
    metadata: dict


class CreditPortfolioGenerator:
    """Generates synthetic loan portfolios for retail, auto, mortgage, and SME lines."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate(
        self,
        n_samples: int = 5000,
        reject_rate: float = 0.25,
        default_rate_target: float = 0.08,
        macro_z: float = 0.0,
    ) -> LoanPortfolio:
        """Generate comprehensive credit application & loan performance dataset.

        Args:
            n_samples: Total number of applications to simulate.
            reject_rate: Target fraction of applications that are rejected at underwriting.
            default_rate_target: Unconditional target 12-month default rate among accepted.
            macro_z: Macroeconomic stress latent factor (Vasicek Z: 0=base, -1.645=severe downturn).

        Returns:
            LoanPortfolio dataclass containing all, accepted, and rejected records.
        """
        rng = self.rng

        # Application IDs
        ids = [f"APP-{100000 + i}" for i in range(n_samples)]

        # Demographics & Financial Profiles
        age = np.clip(rng.normal(loc=42, scale=12, size=n_samples).astype(int), 21, 75)
        emp_years = np.clip(rng.exponential(scale=6, size=n_samples).astype(int), 0, age - 18)
        
        home_ownership = rng.choice(
            ["RENT", "MORTGAGE", "OWN"],
            size=n_samples,
            p=[0.40, 0.45, 0.15]
        )
        
        # Income with lognormal distribution
        log_income = rng.normal(loc=10.8, scale=0.6, size=n_samples)
        annual_income = np.round(np.exp(log_income), -2)

        # Loan Characteristics
        loan_types = rng.choice(
            ["retail_unsecured", "auto_loan", "mortgage", "sme_credit_line"],
            size=n_samples,
            p=[0.45, 0.25, 0.20, 0.10]
        )

        loan_amounts = []
        tenors = []
        interest_rates = []
        for l_type, inc in zip(loan_types, annual_income):
            if l_type == "retail_unsecured":
                amt = np.clip(rng.normal(loc=12000, scale=6000), 1000, 45000)
                ten = rng.choice([12, 24, 36, 48, 60], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                rate = np.clip(0.11 + rng.normal(0, 0.02), 0.06, 0.24)
            elif l_type == "auto_loan":
                amt = np.clip(rng.normal(loc=26000, scale=9000), 5000, 65000)
                ten = rng.choice([36, 48, 60, 72], p=[0.2, 0.4, 0.3, 0.1])
                rate = np.clip(0.075 + rng.normal(0, 0.015), 0.04, 0.16)
            elif l_type == "mortgage":
                amt = np.clip(inc * rng.uniform(2.5, 4.5), 60000, 550000)
                ten = rng.choice([120, 180, 240, 360], p=[0.1, 0.2, 0.4, 0.3])
                rate = np.clip(0.045 + rng.normal(0, 0.01), 0.025, 0.09)
            else:  # sme_credit_line
                amt = np.clip(rng.normal(loc=45000, scale=20000), 10000, 150000)
                ten = rng.choice([12, 24, 36], p=[0.4, 0.4, 0.2])
                rate = np.clip(0.095 + rng.normal(0, 0.025), 0.05, 0.20)
            loan_amounts.append(round(amt, -2))
            tenors.append(ten)
            interest_rates.append(round(rate, 4))

        loan_amounts = np.array(loan_amounts)
        tenors = np.array(tenors)
        interest_rates = np.array(interest_rates)

        # Debt to Income Ratio (DTI)
        dti = np.clip(
            (loan_amounts / np.maximum(tenors / 12, 1)) / annual_income + rng.normal(0.18, 0.08, n_samples),
            0.04, 0.78
        )

        # Bureau FICO/Score (300 to 850)
        latent_creditworthiness = (
            - 2.8 * dti
            + 0.5 * np.log(annual_income / 10000)
            + 0.02 * age
            + 0.03 * emp_years
            + rng.normal(0, 1.0, n_samples)
        )
        bureau_score = np.clip(
            (620 + 75 * latent_creditworthiness).astype(int),
            320, 845
        )

        # Delinquency history & Credit Utilization
        delinq_prob = 1.0 / (1.0 + np.exp(0.015 * (bureau_score - 580)))
        delinquencies_2yrs = rng.binomial(n=4, p=np.clip(delinq_prob, 0.01, 0.85))
        
        utilization = np.clip(
            0.95 - (bureau_score - 300) / 600 + rng.normal(0, 0.12, n_samples),
            0.02, 0.99
        )
        credit_lines = np.clip(rng.poisson(lam=5 + emp_years * 0.15), 1, 20)

        # Collateral & LTV
        collateral = np.zeros(n_samples)
        for i, l_type in enumerate(loan_types):
            if l_type == "mortgage":
                collateral[i] = loan_amounts[i] / rng.uniform(0.65, 0.95)
            elif l_type == "auto_loan":
                collateral[i] = loan_amounts[i] / rng.uniform(0.80, 1.10)
            elif l_type == "sme_credit_line":
                collateral[i] = loan_amounts[i] * rng.uniform(0.2, 0.8)
            else:
                collateral[i] = 0.0
        collateral_safe = np.where(collateral > 0, collateral, 1.0)
        ltv = np.where(collateral > 0, np.round(loan_amounts / collateral_safe, 3), 0.0)

        # Underwriting / Application Decision (Selection Mechanism for Reject Inference)
        underwrite_score = (
            0.006 * bureau_score
            - 2.4 * dti
            - 0.55 * delinquencies_2yrs
            + 0.15 * np.log(annual_income / 10000)
            - 0.8 * utilization
        )
        underwrite_cutoff = np.percentile(underwrite_score, reject_rate * 100)
        is_accepted = (underwrite_score >= underwrite_cutoff)

        # Performance & Default Generation via Structural Vasicek / Merton Latent Factor
        # Latent Asset Return X_i = sqrt(rho)*Z + sqrt(1-rho)*eps_i
        asset_correlation = 0.15
        idiosyncratic_eps = rng.normal(0, 1.0, n_samples)
        composite_z = macro_z  # Macroeconomic systemic shock
        latent_asset = np.sqrt(asset_correlation) * composite_z + np.sqrt(1 - asset_correlation) * idiosyncratic_eps

        # Default propensity logit calibrated on borrower fundamentals (5-8% benchmark default)
        logit_pd = (
            - 2.85
            - 0.012 * (bureau_score - 600)
            + 3.2 * (dti - 0.30)
            + 0.65 * delinquencies_2yrs
            + 1.6 * (utilization - 0.35)
            - 0.35 * np.log(annual_income / 35000)
        )
        fundamental_pd = 1.0 / (1.0 + np.exp(-logit_pd))

        # Shift default threshold by macroeconomic cycle
        default_threshold = norm.ppf(np.clip(fundamental_pd, 0.001, 0.99))
        
        # 12-month default occurrence: latent_asset < default_threshold
        # (Only observable if accepted in historical production data)
        default_flag_12m = (latent_asset < default_threshold).astype(int)

        # Days Past Due (DPD) at observation date
        # Performing: DPD 0-29; Underperforming (SICR): DPD 30-89; Default: DPD >= 90
        dpd = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if default_flag_12m[i] == 1:
                dpd[i] = int(rng.choice([90, 120, 150, 180]))
            else:
                prob_delinq = fundamental_pd[i]
                if prob_delinq > 0.15:
                    dpd[i] = int(rng.choice([0, 15, 30, 45, 60], p=[0.60, 0.15, 0.12, 0.08, 0.05]))
                elif prob_delinq > 0.05:
                    dpd[i] = int(rng.choice([0, 15, 30], p=[0.85, 0.10, 0.05]))
                else:
                    dpd[i] = 0

        # Multi-Horizon / Time-to-default (months) for Survival Analysis
        # Exponential/Weibull hazard conditioned on fundamentals
        hazard_rate = np.maximum(fundamental_pd / 12.0, 0.001)
        simulated_time_to_event = np.round(rng.exponential(scale=1.0 / hazard_rate), 1)
        observed_tenor = np.minimum(simulated_time_to_event, tenors)
        event_observed = (simulated_time_to_event <= tenors).astype(int)

        # Exposure at Default (EAD) & Loss Given Default (LGD)
        # For term loans EAD is balance; for credit lines EAD = Balance + CCF * Undrawn
        drawn_ratio = np.where(loan_types == "sme_credit_line", utilization, 1.0)
        current_balance = np.round(loan_amounts * drawn_ratio * rng.uniform(0.75, 1.0, n_samples), 2)
        ccf = 0.75
        ead = np.where(
            loan_types == "sme_credit_line",
            np.round(current_balance + ccf * (loan_amounts - current_balance), 2),
            current_balance
        )

        # LGD based on collateralization: secured (30-45%) vs unsecured (60-80%)
        lgd = np.zeros(n_samples)
        for i, l_type in enumerate(loan_types):
            if l_type == "retail_unsecured":
                lgd[i] = np.clip(rng.beta(a=7, b=3), 0.40, 0.95)
            elif l_type == "sme_credit_line":
                lgd[i] = np.clip(rng.beta(a=5, b=4), 0.30, 0.85)
            elif l_type == "auto_loan":
                lgd[i] = np.clip(rng.beta(a=4, b=6), 0.20, 0.65)
            else:  # mortgage
                # LTV-driven recovery haircut
                haircut = 0.20
                expected_recovery = max(0.0, collateral[i] * (1 - haircut))
                uncovered = max(0.0, ead[i] - expected_recovery)
                lgd[i] = np.clip(uncovered / max(ead[i], 1.0), 0.05, 0.60)

        # Assemble full master DataFrame
        df = pd.DataFrame({
            "application_id": ids,
            "loan_type": loan_types,
            "loan_amount": loan_amounts,
            "tenor_months": tenors,
            "interest_rate": interest_rates,
            "borrower_age": age,
            "employment_years": emp_years,
            "annual_income": annual_income,
            "home_ownership": home_ownership,
            "debt_to_income": np.round(dti, 4),
            "bureau_score": bureau_score,
            "delinquencies_2yrs": delinquencies_2yrs,
            "credit_lines_count": credit_lines,
            "revolving_utilization": np.round(utilization, 4),
            "collateral_value": np.round(collateral, 2),
            "loan_to_value": ltv,
            "underwrite_score": np.round(underwrite_score, 4),
            "is_accepted": is_accepted.astype(int),
            "days_past_due": dpd,
            "current_balance": current_balance,
            "exposure_at_default": ead,
            "loss_given_default": np.round(lgd, 4),
            "observed_time_months": observed_tenor,
            "event_observed": event_observed,
            "default_12m": default_flag_12m,
            "fundamental_pd": np.round(fundamental_pd, 5)
        })

        accepted_df = df[df["is_accepted"] == 1].copy().reset_index(drop=True)
        rejected_df = df[df["is_accepted"] == 0].copy().reset_index(drop=True)

        metadata = {
            "n_total": n_samples,
            "n_accepted": len(accepted_df),
            "n_rejected": len(rejected_df),
            "acceptance_rate": round(len(accepted_df) / n_samples, 4),
            "accepted_default_rate_12m": round(accepted_df["default_12m"].mean(), 4),
            "rejected_hypothetical_default_rate": round(rejected_df["default_12m"].mean(), 4),
            "macro_latent_z": macro_z
        }

        return LoanPortfolio(
            all_applications=df,
            accepted_loans=accepted_df,
            rejected_applications=rejected_df,
            metadata=metadata
        )
