"""FastAPI Microservice for Real-Time Credit Scoring and IFRS 9 ECL Provisioning.

Endpoints:
- POST /score: Real-time applicant credit scoring, PD prediction & Adverse Action notice
- POST /batch-ecl: Portfolio batch IFRS 9 staging and discounted ECL calculation
- GET /health: Healthcheck and system status
- GET /scorecard: Exportable scorecard points table
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.generator import CreditPortfolioGenerator
from src.features.scorecard import ScorecardTransformer
from src.models.pd_engine import ProbabilityOfDefaultEngine
from src.models.explainability import AdverseActionEngine
from src.simulation.ecl_engine import IFRS9ECLEngine


app = FastAPI(
    title="CreditRisk-IFRS9 Regulatory & Scoring API",
    description="Enterprise Credit Risk Modeling, Scorecard Rating & IFRS 9 ECL Impairment Engine",
    version="1.0.0"
)

# Global trained instances (cached at startup)
_state = {
    "scorecard": None,
    "pd_engine": None,
    "adverse_engine": None,
    "ecl_engine": None
}


def _get_or_init_models():
    if _state["scorecard"] is None:
        gen = CreditPortfolioGenerator(random_seed=42)
        portfolio = gen.generate(n_samples=2500)
        acc_df = portfolio.accepted_loans

        feature_cols = [
            "bureau_score",
            "debt_to_income",
            "delinquencies_2yrs",
            "revolving_utilization",
            "annual_income",
            "employment_years",
            "loan_to_value"
        ]

        # Fit scorecard
        sc = ScorecardTransformer(target_score=600, target_odds=50, pdo=20)
        sc.fit(acc_df[feature_cols], acc_df["default_12m"])
        X_woe = sc.transform_woe(acc_df[feature_cols])

        # Fit PD engine
        pd_eng = ProbabilityOfDefaultEngine(random_state=42)
        pd_eng.fit(X_woe, acc_df[feature_cols], acc_df["default_12m"])

        # Map weights
        intercept, coefs = pd_eng.get_logistic_coefficients()
        sc.set_model_weights(intercept, coefs)

        _state["scorecard"] = sc
        _state["pd_engine"] = pd_eng
        _state["adverse_engine"] = AdverseActionEngine(approval_score_cutoff=580, referral_score_cutoff=540)
        _state["ecl_engine"] = IFRS9ECLEngine()

    return _state["scorecard"], _state["pd_engine"], _state["adverse_engine"], _state["ecl_engine"]


class ApplicantInput(BaseModel):
    application_id: str = Field(default="APP-ONLINE-01", description="Application ID")
    loan_type: str = Field(default="retail_unsecured", description="retail_unsecured, auto_loan, mortgage, sme_credit_line")
    loan_amount: float = Field(default=15000.0, ge=500.0, description="Requested principal")
    tenor_months: int = Field(default=36, ge=6, le=360, description="Duration in months")
    annual_income: float = Field(default=55000.0, ge=5000.0, description="Borrower annual income")
    debt_to_income: float = Field(default=0.28, ge=0.01, le=1.0, description="DTI ratio")
    bureau_score: int = Field(default=680, ge=300, le=850, description="Bureau credit score")
    delinquencies_2yrs: int = Field(default=0, ge=0, le=20, description="Past 2yr delinquencies")
    credit_lines_count: int = Field(default=5, ge=1, le=50, description="Total active lines")
    revolving_utilization: float = Field(default=0.35, ge=0.0, le=1.5, description="Credit card utilization")
    loan_to_value: float = Field(default=0.0, ge=0.0, le=2.0, description="LTV ratio")


class ScoringResponse(BaseModel):
    application_id: str
    credit_score: int
    predicted_pd_12m: float
    decision: str
    top_adverse_reasons: List[str]
    points_breakdown: Dict[str, float]


class BatchECLRequest(BaseModel):
    loans: List[ApplicantInput]


class BatchECLResponse(BaseModel):
    total_exposure: float
    total_weighted_ecl: float
    coverage_ratio: float
    stage_breakdown: Dict[str, Dict[str, float]]


@app.on_event("startup")
def startup_event():
    _get_or_init_models()


@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "service": "CreditRisk-IFRS9 API",
        "version": "1.0.0",
        "governance": "IFRS 9 / BCBS Basel III / ECOA"
    }


@app.post("/score", response_model=ScoringResponse)
def score_applicant(applicant: ApplicantInput):
    """Real-time applicant scoring, PD inference and Adverse Action disclosure."""
    sc, pd_eng, adv_eng, _ = _get_or_init_models()

    df_single = pd.DataFrame([{
        "bureau_score": applicant.bureau_score,
        "debt_to_income": applicant.debt_to_income,
        "delinquencies_2yrs": applicant.delinquencies_2yrs,
        "revolving_utilization": applicant.revolving_utilization,
        "annual_income": applicant.annual_income,
        "employment_years": 5,
        "loan_to_value": applicant.loan_to_value
    }])

    # Transform WoE & Score
    X_woe = sc.transform_woe(df_single)
    pd_12m = float(pd_eng.predict_pd_logistic(X_woe, calibrated=True)[0])
    total_score, pts_df = sc.transform_score(df_single)

    score_val = int(total_score.iloc[0])
    pts_series = pts_df.iloc[0]

    report = adv_eng.generate_report(
        application_id=applicant.application_id,
        applicant_features=df_single.iloc[0],
        scorecard_points_breakdown=pts_series,
        total_score=score_val,
        predicted_pd=pd_12m
    )

    return ScoringResponse(
        application_id=applicant.application_id,
        credit_score=score_val,
        predicted_pd_12m=round(pd_12m, 5),
        decision=report.decision,
        top_adverse_reasons=report.top_adverse_reasons,
        points_breakdown={k: round(v, 1) for k, v in report.feature_contributions.items()}
    )


@app.get("/scorecard")
def get_scorecard_table():
    """Retrieve full scorecard points table with WoE, IV and bin thresholds."""
    sc, _, _, _ = _get_or_init_models()
    table = sc.get_scorecard_table()
    return table.to_dict(orient="records")
