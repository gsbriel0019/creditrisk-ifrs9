"""Probability of Default (PD) Modeling and Probability Calibration Engine.

Benchmarks regulatory Logistic Regression against LightGBM Gradient Boosting,
calibrates probabilities (Isotonic / Platt Scaling), and computes discrimination
metrics: AUC, Gini, Kolmogorov-Smirnov (KS), and Brier Score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import lightgbm as lgb


@dataclass
class ModelBenchmarkMetrics:
    """Quantitative performance and calibration metrics for credit risk models."""
    model_name: str
    roc_auc: float
    gini_coefficient: float
    ks_statistic: float
    ks_optimal_threshold: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float


class ProbabilityOfDefaultEngine:
    """Dual PD engine: Classical Regulatory WoE Logistic Regression vs. Modern LightGBM."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.logistic_model: Optional[LogisticRegression] = None
        self.calibrated_logistic: Optional[CalibratedClassifierCV] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.calibrated_lgb: Optional[CalibratedClassifierCV] = None
        self.feature_names_woe: List[str] = []
        self.feature_names_raw: List[str] = []

    def fit(
        self,
        X_woe: pd.DataFrame,
        X_raw: pd.DataFrame,
        y: pd.Series,
        calibration_method: str = "isotonic"
    ) -> ProbabilityOfDefaultEngine:
        """Fit and calibrate both Logistic Regression and LightGBM models.

        Args:
            X_woe: Features in Weight of Evidence scale.
            X_raw: Raw numerical and encoded features.
            y: Binary default target (1=Default, 0=Non-Default).
            calibration_method: 'isotonic' or 'sigmoid' (Platt scaling).
        """
        self.feature_names_woe = list(X_woe.columns)
        self.feature_names_raw = list(X_raw.columns)
        y_arr = np.asarray(y)

        # 1. Classical Regulatory WoE Logistic Regression
        self.logistic_model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state
        )
        self.logistic_model.fit(X_woe, y_arr)

        self.calibrated_logistic = CalibratedClassifierCV(
            estimator=self.logistic_model,
            method=calibration_method,
            cv=3
        )
        self.calibrated_logistic.fit(X_woe, y_arr)

        # 2. Modern Gradient Boosting (LightGBM)
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        self.lgb_model.fit(X_raw, y_arr)

        self.calibrated_lgb = CalibratedClassifierCV(
            estimator=self.lgb_model,
            method=calibration_method,
            cv=3
        )
        self.calibrated_lgb.fit(X_raw, y_arr)

        return self

    def predict_pd_logistic(self, X_woe: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Predict 12-month Probability of Default using Logistic Regression."""
        model = self.calibrated_logistic if calibrated else self.logistic_model
        if model is None:
            raise ValueError("Logistic model is not fitted yet.")
        return model.predict_proba(X_woe)[:, 1]

    def predict_pd_lgb(self, X_raw: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Predict 12-month Probability of Default using LightGBM."""
        model = self.calibrated_lgb if calibrated else self.lgb_model
        if model is None:
            raise ValueError("LightGBM model is not fitted yet.")
        return model.predict_proba(X_raw)[:, 1]

    def evaluate(
        self,
        X_woe: pd.DataFrame,
        X_raw: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[ModelBenchmarkMetrics, ModelBenchmarkMetrics]:
        """Compute comprehensive metrics for both models on validation/test set."""
        y_arr = np.asarray(y)

        # Evaluate Logistic Regression
        pd_log = self.predict_pd_logistic(X_woe, calibrated=True)
        metrics_log = self._compute_metrics("Regulatory Logistic (WoE)", y_arr, pd_log)

        # Evaluate LightGBM
        pd_lgb = self.predict_pd_lgb(X_raw, calibrated=True)
        metrics_lgb = self._compute_metrics("Calibrated LightGBM", y_arr, pd_lgb)

        return metrics_log, metrics_lgb

    def _compute_metrics(
        self,
        name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> ModelBenchmarkMetrics:
        """Calculate statistical discrimination and calibration metrics."""
        auc = float(roc_auc_score(y_true, y_prob))
        gini = float(2.0 * auc - 1.0)
        brier = float(brier_score_loss(y_true, y_prob))
        ll = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6)))

        # Kolmogorov-Smirnov (KS) statistic
        ks_stat, ks_thresh = self._compute_ks(y_true, y_prob)

        # Expected Calibration Error (ECE)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ece = float(np.mean(np.abs(prob_true - prob_pred)))

        return ModelBenchmarkMetrics(
            model_name=name,
            roc_auc=round(auc, 4),
            gini_coefficient=round(gini, 4),
            ks_statistic=round(ks_stat, 4),
            ks_optimal_threshold=round(ks_thresh, 4),
            brier_score=round(brier, 5),
            log_loss=round(ll, 4),
            expected_calibration_error=round(ece, 4)
        )

    @staticmethod
    def _compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        """Calculate Kolmogorov-Smirnov separation metric and cutoff threshold."""
        df = pd.DataFrame({"y": y_true, "prob": y_prob}).sort_values(by="prob", ascending=False)
        total_bads = max(int(np.sum(y_true == 1)), 1)
        total_goods = max(int(np.sum(y_true == 0)), 1)

        df["cum_bads"] = (df["y"] == 1).cumsum() / total_bads
        df["cum_goods"] = (df["y"] == 0).cumsum() / total_goods
        df["ks"] = np.abs(df["cum_bads"] - df["cum_goods"])

        max_idx = df["ks"].idxmax()
        max_ks = float(df.loc[max_idx, "ks"])
        opt_thresh = float(df.loc[max_idx, "prob"])
        return max_ks, opt_thresh

    def get_logistic_coefficients(self) -> Tuple[float, Dict[str, float]]:
        """Extract intercept and coefficient weights from uncalibrated logistic model."""
        if self.logistic_model is None:
            raise ValueError("Logistic model is not fitted.")
        intercept = float(self.logistic_model.intercept_[0])
        coefs = {
            col: float(coef)
            for col, coef in zip(self.feature_names_woe, self.logistic_model.coef_[0])
        }
        return intercept, coefs
