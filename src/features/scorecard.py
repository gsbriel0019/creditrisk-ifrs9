"""Weight of Evidence (WoE) and Information Value (IV) Scorecard Engineering.

Provides regulatory-grade binning, WoE transformations, Information Value calculations,
and scorecard scaling (Points to Double Odds, Base Points & Offset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class ScorecardBin:
    """Individual bin metadata and statistics for a feature."""
    feature_name: str
    bin_label: str
    bin_min: float
    bin_max: float
    total_count: int
    good_count: int
    bad_count: int
    bad_rate: float
    woe: float
    iv: float
    points: float = 0.0


@dataclass
class ScorecardSummary:
    """Feature-level predictive power summary."""
    feature_name: str
    information_value: float
    predictive_power: str
    num_bins: int


class ScorecardTransformer:
    """Transforms raw features into Weight of Evidence (WoE) and scaled Scorecard Points.
    
    Standard Scorecard scaling formula:
        Factor = PDO / ln(2)
        Offset = TargetScore - Factor * ln(TargetOdds)
        Score = Offset + Factor * ln((1 - PD) / PD)
    """

    def __init__(
        self,
        target_score: float = 600.0,
        target_odds: float = 50.0,
        pdo: float = 20.0,
        max_bins: int = 5,
        min_iv: float = 0.02
    ):
        self.target_score = target_score
        self.target_odds = target_odds
        self.pdo = pdo
        self.max_bins = max_bins
        self.min_iv = min_iv

        self.factor = self.pdo / np.log(2.0)
        self.offset = self.target_score - (self.factor * np.log(self.target_odds))

        self.bin_rules: Dict[str, List[ScorecardBin]] = {}
        self.iv_summary: List[ScorecardSummary] = []
        self.selected_features: List[str] = []
        self.intercept: float = 0.0
        self.coefficients: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None
    ) -> ScorecardTransformer:
        """Fit optimal bins, calculate WoE and Information Value for each feature.

        Args:
            X: Input features DataFrame.
            y: Binary target series (1 for Default/Bad, 0 for Non-Default/Good).
            feature_cols: List of column names to consider. If None, uses all X columns.

        Returns:
            self
        """
        if feature_cols is None:
            feature_cols = [c for c in X.columns if c not in ["application_id", "is_accepted"]]

        y_clean = pd.Series(y).values
        n_total_bads = max(int(np.sum(y_clean == 1)), 1)
        n_total_goods = max(int(np.sum(y_clean == 0)), 1)

        self.bin_rules = {}
        self.iv_summary = []

        for col in feature_cols:
            series = X[col]
            bins = self._compute_feature_bins(
                series=series,
                y=y_clean,
                feature_name=col,
                n_total_goods=n_total_goods,
                n_total_bads=n_total_bads
            )
            if not bins:
                continue

            total_iv = sum(b.iv for b in bins)
            
            # Categorize predictive strength
            if total_iv < 0.02:
                power = "Unpredictive (<0.02)"
            elif total_iv < 0.10:
                power = "Weak (0.02 - 0.10)"
            elif total_iv < 0.30:
                power = "Medium (0.10 - 0.30)"
            elif total_iv < 0.50:
                power = "Strong (0.30 - 0.50)"
            else:
                power = "Suspiciously High (>=0.50)"

            self.bin_rules[col] = bins
            self.iv_summary.append(ScorecardSummary(
                feature_name=col,
                information_value=round(total_iv, 5),
                predictive_power=power,
                num_bins=len(bins)
            ))

        # Filter features meeting minimum IV threshold
        self.selected_features = [
            s.feature_name for s in self.iv_summary if s.information_value >= self.min_iv
        ]
        # Sort summary descending by IV
        self.iv_summary.sort(key=lambda s: s.information_value, reverse=True)
        return self

    def _compute_feature_bins(
        self,
        series: pd.Series,
        y: np.ndarray,
        feature_name: str,
        n_total_goods: int,
        n_total_bads: int
    ) -> List[ScorecardBin]:
        """Bin a continuous or categorical feature and compute WoE/IV statistics."""
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric and series.nunique() > self.max_bins:
            # Quantile-based coarse classing
            try:
                # Use qcut with duplicates='drop'
                _, bin_edges = pd.qcut(
                    series,
                    q=self.max_bins,
                    retbins=True,
                    duplicates="drop"
                )
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
            except Exception:
                bin_edges = np.linspace(series.min(), series.max(), self.max_bins + 1)
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf

            bins_list: List[ScorecardBin] = []
            for i in range(len(bin_edges) - 1):
                low = bin_edges[i]
                high = bin_edges[i + 1]
                mask = (series > low) & (series <= high) if i > 0 else (series >= low) & (series <= high)

                n_goods = int(np.sum((y == 0) & mask))
                n_bads = int(np.sum((y == 1) & mask))
                n_total = n_goods + n_bads

                # Regularized frequencies (avoid log(0) or division by zero)
                dist_good = max((n_goods + 0.5) / (n_total_goods + 1.0), 1e-6)
                dist_bad = max((n_bads + 0.5) / (n_total_bads + 1.0), 1e-6)
                
                woe = np.log(dist_good / dist_bad)
                iv = (dist_good - dist_bad) * woe
                bad_rate = n_bads / max(n_total, 1)

                label = f"({round(low, 2) if low != -np.inf else '-inf'}, {round(high, 2) if high != np.inf else '+inf'}]"
                bins_list.append(ScorecardBin(
                    feature_name=feature_name,
                    bin_label=label,
                    bin_min=float(low),
                    bin_max=float(high),
                    total_count=n_total,
                    good_count=n_goods,
                    bad_count=n_bads,
                    bad_rate=round(bad_rate, 4),
                    woe=round(float(woe), 4),
                    iv=round(float(iv), 5)
                ))
            return bins_list
        else:
            # Categorical or low cardinality discrete
            unique_vals = series.unique()
            bins_list = []
            for val in unique_vals:
                mask = (series == val)
                n_goods = int(np.sum((y == 0) & mask))
                n_bads = int(np.sum((y == 1) & mask))
                n_total = n_goods + n_bads

                dist_good = max((n_goods + 0.5) / (n_total_goods + 1.0), 1e-6)
                dist_bad = max((n_bads + 0.5) / (n_total_bads + 1.0), 1e-6)

                woe = np.log(dist_good / dist_bad)
                iv = (dist_good - dist_bad) * woe
                bad_rate = n_bads / max(n_total, 1)

                bins_list.append(ScorecardBin(
                    feature_name=feature_name,
                    bin_label=str(val),
                    bin_min=-np.inf,
                    bin_max=np.inf,
                    total_count=n_total,
                    good_count=n_goods,
                    bad_count=n_bads,
                    bad_rate=round(bad_rate, 4),
                    woe=round(float(woe), 4),
                    iv=round(float(iv), 5)
                ))
            return bins_list

    def transform_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw dataframe into Weight of Evidence (WoE) values."""
        X_woe = pd.DataFrame(index=X.index)

        for col in self.selected_features:
            if col not in self.bin_rules:
                continue

            bins = self.bin_rules[col]
            series = X[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)

            woe_vals = np.zeros(len(series))

            if is_numeric:
                for b in bins:
                    mask = (series > b.bin_min) & (series <= b.bin_max) if b.bin_min != -np.inf else (series <= b.bin_max)
                    woe_vals[mask] = b.woe
            else:
                val_to_woe = {b.bin_label: b.woe for b in bins}
                default_woe = np.mean([b.woe for b in bins])
                woe_vals = series.map(lambda v: val_to_woe.get(str(v), default_woe)).values

            X_woe[f"{col}_woe"] = woe_vals

        return X_woe

    def set_model_weights(self, intercept: float, coefficients: Dict[str, float]) -> None:
        """Assign trained logistic regression weights to calculate scaled point values."""
        self.intercept = intercept
        self.coefficients = coefficients
        p = max(len(coefficients), 1)

        for col, bins in self.bin_rules.items():
            coef_key = f"{col}_woe" if f"{col}_woe" in coefficients else col
            if coef_key not in coefficients:
                continue
            beta = coefficients[coef_key]
            for b in bins:
                # Standard credit scorecard points mapping:
                # Score = Offset + Factor * ln(Good/Bad)
                # ln(Good/Bad) = - (alpha + sum(beta * WoE))
                # Distributed points per feature bin:
                pts = (self.offset / p) - self.factor * ((self.intercept / p) + beta * b.woe)
                b.points = round(float(pts), 1)

    def transform_score(self, X: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Compute scorecard points breakdown and total credit score per applicant."""
        pts_df = pd.DataFrame(index=X.index)

        for col in self.selected_features:
            if col not in self.bin_rules:
                continue
            bins = self.bin_rules[col]
            series = X[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)

            feature_pts = np.zeros(len(series))

            if is_numeric:
                for b in bins:
                    mask = (series > b.bin_min) & (series <= b.bin_max) if b.bin_min != -np.inf else (series <= b.bin_max)
                    feature_pts[mask] = b.points
            else:
                val_to_pts = {b.bin_label: b.points for b in bins}
                default_pts = np.mean([b.points for b in bins])
                feature_pts = series.map(lambda v: val_to_pts.get(str(v), default_pts)).values

            pts_df[f"{col}_points"] = feature_pts

        total_score = np.round(pts_df.sum(axis=1)).astype(int)
        # Bounded between standard bureau limits (300 to 850)
        total_score = total_score.clip(300, 850)
        return total_score, pts_df

    def get_scorecard_table(self) -> pd.DataFrame:
        """Return exportable scorecard table with bins, WoE, IV, and assigned points."""
        records = []
        for col, bins in self.bin_rules.items():
            for b in bins:
                records.append({
                    "Feature": b.feature_name,
                    "Attribute / Bin": b.bin_label,
                    "Total Count": b.total_count,
                    "Bad Count": b.bad_count,
                    "Bad Rate (%)": round(b.bad_rate * 100, 2),
                    "WoE": b.woe,
                    "IV": b.iv,
                    "Assigned Points": b.points
                })
        return pd.DataFrame(records)
