"""Streamlit Studio for Enterprise Credit Risk, Scorecard Rating, and IFRS 9 ECL Impairment.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, calibration_curve

from src.data.generator import CreditPortfolioGenerator
from src.features.scorecard import ScorecardTransformer
from src.models.pd_engine import ProbabilityOfDefaultEngine
from src.models.explainability import AdverseActionEngine
from src.models.survival_pd import SurvivalPDEngine
from src.simulation.ecl_engine import IFRS9ECLEngine

st.set_page_config(
    page_title="CreditRisk-IFRS9 Studio",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 12px;
    }
    .badge-approved {
        background-color: #10B981;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-rejected {
        background-color: #EF4444;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-referred {
        background-color: #F59E0B;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_train_pipeline(n_samples: int = 3500, seed: int = 42):
    """Generate portfolio, fit scorecard, train PD models, and run IFRS 9 ECL."""
    gen = CreditPortfolioGenerator(random_seed=seed)
    portfolio = gen.generate(n_samples=n_samples)
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

    # Scorecard
    sc = ScorecardTransformer(target_score=600, target_odds=50, pdo=20, max_bins=5)
    sc.fit(acc_df[feature_cols], acc_df["default_12m"])
    X_woe = sc.transform_woe(acc_df[feature_cols])

    # PD Engine
    pd_eng = ProbabilityOfDefaultEngine(random_state=seed)
    pd_eng.fit(X_woe, acc_df[feature_cols], acc_df["default_12m"])

    # Link weights
    intercept, coefs = pd_eng.get_logistic_coefficients()
    sc.set_model_weights(intercept, coefs)

    # Predictions
    calibrated_pd = pd_eng.predict_pd_logistic(X_woe, calibrated=True)
    scores, pts_breakdown = sc.transform_score(acc_df[feature_cols])

    # ECL Simulation
    ecl_eng = IFRS9ECLEngine()
    ecl_df, ecl_summary = ecl_eng.process_portfolio(acc_df, calibrated_pd)

    # Evaluate
    metrics_log, metrics_lgb = pd_eng.evaluate(X_woe, acc_df[feature_cols], acc_df["default_12m"])

    return {
        "portfolio": portfolio,
        "acc_df": acc_df,
        "feature_cols": feature_cols,
        "sc": sc,
        "pd_eng": pd_eng,
        "ecl_eng": ecl_eng,
        "ecl_df": ecl_df,
        "ecl_summary": ecl_summary,
        "calibrated_pd": calibrated_pd,
        "scores": scores,
        "metrics_log": metrics_log,
        "metrics_lgb": metrics_lgb,
        "adv_eng": AdverseActionEngine()
    }


# Sidebar
st.sidebar.title("🏛️ CreditRisk-IFRS9")
st.sidebar.markdown("**Enterprise Credit Risk & Accounting Impairment**")
st.sidebar.divider()

portfolio_size = st.sidebar.slider("Portfolio Sample Size", 1000, 6000, 3000, step=500)
sim_seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)

with st.spinner("Initializing and training credit risk models..."):
    data = load_and_train_pipeline(n_samples=portfolio_size, seed=sim_seed)

st.title("🏛️ CreditRisk-IFRS9: Enterprise Credit Risk & Impairment Studio")
st.caption("Quantitative Risk Engineering, Scorecard Calibration, and IFRS 9 Expected Credit Loss (ECL)")

tabs = st.tabs([
    "📊 Portfolio & Staging",
    "🎯 Scorecard & WoE Studio",
    "📈 PD Benchmark & Calibration",
    "🏦 IFRS 9 Multi-Scenario Stress",
    "👤 Underwriting & Adverse Action"
])

# ----------------- TAB 1: PORTFOLIO & STAGING -----------------
with tabs[0]:
    st.subheader("IFRS 9 Impairment & Portfolio Capital Overview")
    summary = data["ecl_summary"]
    ecl_df = data["ecl_df"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Exposure at Default (EAD)", f"${summary.total_exposure:,.0f}")
    with col2:
        st.metric("Weighted IFRS 9 Provision (ECL)", f"${summary.total_weighted_ecl:,.0f}")
    with col3:
        st.metric("Aggregate Coverage Ratio", f"{summary.total_coverage_ratio * 100:.2f}%")
    with col4:
        st.metric("Severe Downturn Stress Impact", f"+{summary.adverse_ecl_delta_pct:.1f}%")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        stage_counts = pd.DataFrame({
            "Stage": ["Stage 1 (Performing)", "Stage 2 (SICR)", "Stage 3 (Default)"],
            "Exposure": [summary.stage1_exposure, summary.stage2_exposure, summary.stage3_exposure],
            "ECL": [summary.stage1_ecl, summary.stage2_ecl, summary.stage3_ecl]
        })
        fig_pie = px.pie(
            stage_counts,
            names="Stage",
            values="Exposure",
            title="Exposure Distribution by IFRS 9 Stage",
            color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444"],
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        fig_bar = px.bar(
            stage_counts,
            x="Stage",
            y="ECL",
            color="Stage",
            title="Credit Impairment Provisions (ECL) by Stage ($)",
            color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Audited Portfolio Staging Matrix")
    staging_display = pd.DataFrame([
        {
            "Classification": "Stage 1 (Performing - 12m ECL)",
            "Active Loans": summary.stage1_count,
            "Total EAD ($)": f"${summary.stage1_exposure:,.2f}",
            "Provision ECL ($)": f"${summary.stage1_ecl:,.2f}",
            "Coverage (%)": f"{summary.stage1_coverage_ratio * 100:.2f}%"
        },
        {
            "Classification": "Stage 2 (Underperforming - Lifetime ECL)",
            "Active Loans": summary.stage2_count,
            "Total EAD ($)": f"${summary.stage2_exposure:,.2f}",
            "Provision ECL ($)": f"${summary.stage2_ecl:,.2f}",
            "Coverage (%)": f"{summary.stage2_coverage_ratio * 100:.2f}%"
        },
        {
            "Classification": "Stage 3 (Credit-Impaired - Lifetime ECL)",
            "Active Loans": summary.stage3_count,
            "Total EAD ($)": f"${summary.stage3_exposure:,.2f}",
            "Provision ECL ($)": f"${summary.stage3_ecl:,.2f}",
            "Coverage (%)": f"{summary.stage3_coverage_ratio * 100:.2f}%"
        }
    ])
    st.table(staging_display)

# ----------------- TAB 2: SCORECARD & WOE STUDIO -----------------
with tabs[1]:
    st.subheader("Feature Screening: Information Value (IV) Ranking")
    sc: ScorecardTransformer = data["sc"]
    iv_df = pd.DataFrame([
        {
            "Feature": s.feature_name,
            "Information Value": s.information_value,
            "Predictive Power": s.predictive_power,
            "Bins": s.num_bins
        }
        for s in sc.iv_summary
    ])

    fig_iv = px.bar(
        iv_df,
        x="Information Value",
        y="Feature",
        orientation="h",
        color="Predictive Power",
        title="Information Value (IV) by Risk Feature",
        color_discrete_map={
            "Strong (0.30 - 0.50)": "#10B981",
            "Medium (0.10 - 0.30)": "#3B82F6",
            "Weak (0.02 - 0.10)": "#F59E0B",
            "Suspiciously High (>=0.50)": "#8B5CF6",
            "Unpredictive (<0.02)": "#9CA3AF"
        }
    )
    fig_iv.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_iv, use_container_width=True)

    st.subheader("Interactive Weight of Evidence (WoE) Deep-Dive")
    selected_feat = st.selectbox("Select Risk Feature to Inspect Bins", options=list(sc.bin_rules.keys()))
    if selected_feat:
        bins = sc.bin_rules[selected_feat]
        bin_df = pd.DataFrame([
            {
                "Bin Range": b.bin_label,
                "Total Count": b.total_count,
                "Bad Count": b.bad_count,
                "Bad Rate (%)": round(b.bad_rate * 100, 2),
                "WoE": b.woe,
                "Points": b.points
            }
            for b in bins
        ])

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_woe = px.line(
                bin_df,
                x="Bin Range",
                y="WoE",
                markers=True,
                title=f"Weight of Evidence (WoE) Monotonicity: {selected_feat}"
            )
            st.plotly_chart(fig_woe, use_container_width=True)
        with col_b2:
            fig_bad = px.bar(
                bin_df,
                x="Bin Range",
                y="Bad Rate (%)",
                title=f"Empirical Default Rate (%) across Bins: {selected_feat}",
                color="Bad Rate (%)",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_bad, use_container_width=True)

        st.dataframe(bin_df, use_container_width=True)

# ----------------- TAB 3: PD BENCHMARK & CALIBRATION -----------------
with tabs[2]:
    st.subheader("Model Discrimination & Calibration Benchmark")
    m_log = data["metrics_log"]
    m_lgb = data["metrics_lgb"]

    bench_df = pd.DataFrame([
        {
            "Model": m_log.model_name,
            "ROC-AUC": m_log.roc_auc,
            "Gini Coefficient": m_log.gini_coefficient,
            "KS Statistic": f"{m_log.ks_statistic * 100:.1f}%",
            "Brier Score": m_log.brier_score,
            "Calibration Error (ECE)": m_log.expected_calibration_error
        },
        {
            "Model": m_lgb.model_name,
            "ROC-AUC": m_lgb.roc_auc,
            "Gini Coefficient": m_lgb.gini_coefficient,
            "KS Statistic": f"{m_lgb.ks_statistic * 100:.1f}%",
            "Brier Score": m_lgb.brier_score,
            "Calibration Error (ECE)": m_lgb.expected_calibration_error
        }
    ])
    st.table(bench_df)

    y_test = data["acc_df"]["default_12m"].values
    prob_log = data["calibrated_pd"]

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fpr, tpr, _ = roc_curve(y_test, prob_log)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Logistic (AUC={m_log.roc_auc})', line=dict(color='#3B82F6', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(title="ROC Curve: Model Discrimination", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_m2:
        prob_true, prob_pred = calibration_curve(y_test, prob_log, n_bins=10)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Calibrated PD', line=dict(color='#10B981', width=2)))
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration', line=dict(dash='dash', color='gray')))
        fig_cal.update_layout(title="Reliability Diagram (Probability Calibration)", xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives")
        st.plotly_chart(fig_cal, use_container_width=True)

# ----------------- TAB 4: IFRS 9 FORWARD-LOOKING STRESS -----------------
with tabs[3]:
    st.subheader("Forward-Looking Macroeconomic Scenarios & Sensitivity Analysis")
    summary = data["ecl_summary"]

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Baseline Scenario (50% Weight)", f"${summary.baseline_ecl_total:,.0f}")
    with col_s2:
        st.metric("Severe Downturn (30% Weight)", f"${summary.adverse_ecl_total:,.0f}", f"+{summary.adverse_ecl_delta_pct:.1f}% Stress")
    with col_s3:
        st.metric("Macro Expansion (20% Weight)", f"${summary.favorable_ecl_total:,.0f}", f"{(summary.favorable_ecl_total - summary.baseline_ecl_total)/summary.baseline_ecl_total*100:.1f}%")

    scenario_comp = pd.DataFrame({
        "Scenario": ["Baseline (Weight 50%)", "Severe Downturn (Weight 30%)", "Macro Expansion (Weight 20%)"],
        "Provision Required ($)": [summary.baseline_ecl_total, summary.adverse_ecl_total, summary.favorable_ecl_total]
    })
    fig_scn = px.bar(
        scenario_comp,
        x="Scenario",
        y="Provision Required ($)",
        color="Scenario",
        title="Portfolio Credit Loss Provisions Under Forward-Looking Stress Scenarios",
        color_discrete_sequence=["#3B82F6", "#EF4444", "#10B981"]
    )
    st.plotly_chart(fig_scn, use_container_width=True)

# ----------------- TAB 5: UNDERWRITING SIMULATOR -----------------
with tabs[4]:
    st.subheader("Real-Time Underwriting & Adverse Action Simulator")
    st.write("Simulate a live applicant decision and generate ECOA/FCRA compliant Adverse Action disclosures.")

    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        sim_income = st.number_input("Annual Income ($)", 10000, 500000, 48000, step=5000)
        sim_dti = st.slider("Debt-to-Income (DTI)", 0.05, 0.85, 0.38, step=0.01)
        sim_bureau = st.slider("Credit Bureau Score (FICO)", 350, 850, 620, step=5)
    with col_u2:
        sim_delinq = st.selectbox("Delinquencies (Past 2 Years)", [0, 1, 2, 3, 4])
        sim_util = st.slider("Revolving Line Utilization", 0.05, 1.0, 0.55, step=0.05)
        sim_emp = st.slider("Years in Employment", 0, 30, 4)
    with col_u3:
        sim_amount = st.number_input("Requested Loan Amount ($)", 1000, 200000, 18000, step=1000)
        sim_tenor = st.selectbox("Loan Tenor (Months)", [12, 24, 36, 48, 60, 120])
        sim_ltv = st.slider("Loan to Value (LTV)", 0.0, 1.5, 0.0, step=0.05)

    if st.button("Evaluate Credit Application", type="primary"):
        test_df = pd.DataFrame([{
            "bureau_score": sim_bureau,
            "debt_to_income": sim_dti,
            "delinquencies_2yrs": sim_delinq,
            "revolving_utilization": sim_util,
            "annual_income": sim_income,
            "employment_years": sim_emp,
            "loan_to_value": sim_ltv
        }])

        sc: ScorecardTransformer = data["sc"]
        pd_eng: ProbabilityOfDefaultEngine = data["pd_eng"]
        adv_eng: AdverseActionEngine = data["adv_eng"]

        X_woe = sc.transform_woe(test_df)
        pred_pd = float(pd_eng.predict_pd_logistic(X_woe, calibrated=True)[0])
        score_val, pts_df = sc.transform_score(test_df)
        score_int = int(score_val.iloc[0])

        report = adv_eng.generate_report(
            application_id="APP-LIVE-SIM",
            applicant_features=test_df.iloc[0],
            scorecard_points_breakdown=pts_df.iloc[0],
            total_score=score_int,
            predicted_pd=pred_pd
        )

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Calculated Credit Score", f"{score_int} / 850")
            st.metric("Estimated 12-Month PD", f"{pred_pd * 100:.2f}%")

            if report.decision == "APPROVED":
                st.markdown('<div class="badge-approved">DECISION: APPROVED</div>', unsafe_allow_html=True)
            elif report.decision == "REFERRED_MANUAL_REVIEW":
                st.markdown('<div class="badge-referred">DECISION: REFERRED TO CREDIT COMMITTEE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-rejected">DECISION: ADVERSE ACTION (REJECTED)</div>', unsafe_allow_html=True)

        with res_col2:
            # Score Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_int,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Credit Score Gauge"},
                gauge={
                    'axis': {'range': [300, 850]},
                    'bar': {'color': "#3B82F6"},
                    'steps': [
                        {'range': [300, 540], 'color': "#EF4444"},
                        {'range': [540, 620], 'color': "#F59E0B"},
                        {'range': [620, 720], 'color': "#10B981"},
                        {'range': [720, 850], 'color': "#059669"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        if report.top_adverse_reasons:
            st.subheader("📋 Regulatory Adverse Action Statement (ECOA / FCRA)")
            st.info("The following principal factors contributed negatively to this credit decision:")
            for idx, r in enumerate(report.top_adverse_reasons, 1):
                st.markdown(f"**{idx}.** {r}")

        # Lifetime survival curve for applicant
        surv_eng = SurvivalPDEngine()
        curve = surv_eng.generate_lifetime_curve(pred_pd, tenor_months=sim_tenor)
        curve_df = pd.DataFrame({
            "Year": [f"Year {y}" for y in curve.projection_years],
            "Cumulative Default Prob (%)": [p * 100 for p in curve.cumulative_default_probabilities],
            "Marginal Default Prob (%)": [p * 100 for p in curve.marginal_default_probabilities]
        })
        fig_curve = px.line(
            curve_df,
            x="Year",
            y=["Cumulative Default Prob (%)", "Marginal Default Prob (%)"],
            markers=True,
            title="Multi-Year Lifetime Default Term Structure (IFRS 9 Horizon)"
        )
        st.plotly_chart(fig_curve, use_container_width=True)
