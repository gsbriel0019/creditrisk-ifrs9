"""Reject Inference Engine for Credit Scoring.

Corrects for sample selection bias inherent in credit scoring, where repayment performance
is only observable for accepted applicants. Implements:
1. Hard Cutoff assignment
2. Score-band Parceling (Proportional Assignment with Bad-rate multiplier)
3. Fuzzy Augmentation (Soft Probability Weights)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class RejectInferenceEngine:
    """Manages rejection inference techniques to debias credit underwriting models."""

    def __init__(
        self,
        method: Literal["hard_cutoff", "parceling", "fuzzy_augmentation"] = "parceling",
        parceling_multiplier: float = 1.5,
        cutoff_percentile: float = 70.0,
        random_state: int = 42
    ):
        self.method = method
        self.parceling_multiplier = parceling_multiplier
        self.cutoff_percentile = cutoff_percentile
        self.random_state = random_state

    def infer_and_augment(
        self,
        accepted_X: pd.DataFrame,
        accepted_y: pd.Series,
        rejected_X: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Infer outcomes on rejected applicants and produce an augmented dataset.

        Args:
            accepted_X: Feature matrix of accepted borrowers.
            accepted_y: Known binary default labels of accepted borrowers.
            rejected_X: Feature matrix of rejected applicants.
            feature_cols: Column names used for inference.

        Returns:
            Tuple of:
                - Augmented feature DataFrame (Accepts + Rejects)
                - Augmented target Series
                - Sample weights array
        """
        # Step 1: Train baseline model on accepted applications
        base_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        base_model.fit(accepted_X[feature_cols], accepted_y)

        # Step 2: Predict default probabilities on rejected applications
        rejected_pred_pd = base_model.predict_proba(rejected_X[feature_cols])[:, 1]
        n_rejects = len(rejected_X)

        if self.method == "hard_cutoff":
            # Assign Bad if predicted PD exceeds threshold
            cutoff = np.percentile(rejected_pred_pd, self.cutoff_percentile)
            inferred_y = (rejected_pred_pd >= cutoff).astype(int)

            aug_X = pd.concat([accepted_X[feature_cols], rejected_X[feature_cols]], ignore_index=True)
            aug_y = pd.Series(np.concatenate([accepted_y.values, inferred_y]))
            weights = np.ones(len(aug_y), dtype=float)

        elif self.method == "parceling":
            # Parceling: Score into 5 bands, compute accept bad rate * multiplier
            rng = np.random.default_rng(self.random_state)
            
            # Predict accepted PD to build bands
            accepted_pred_pd = base_model.predict_proba(accepted_X[feature_cols])[:, 1]
            quantiles = np.linspace(0, 1, 6)
            bins = np.quantile(accepted_pred_pd, quantiles)
            bins[0] = -1.0
            bins[-1] = 2.0

            # Calculate empirical default rate in each band for accepts
            band_acc = np.digitize(accepted_pred_pd, bins)
            band_rej = np.digitize(rejected_pred_pd, bins)

            band_bad_rates = {}
            for b in np.unique(band_acc):
                mask = (band_acc == b)
                emp_rate = accepted_y.iloc[mask].mean() if np.sum(mask) > 0 else 0.1
                band_bad_rates[b] = min(emp_rate * self.parceling_multiplier, 0.95)

            inferred_y = np.zeros(n_rejects, dtype=int)
            for i, b in enumerate(band_rej):
                target_prob = band_bad_rates.get(b, 0.15)
                inferred_y[i] = int(rng.binomial(1, target_prob))

            aug_X = pd.concat([accepted_X[feature_cols], rejected_X[feature_cols]], ignore_index=True)
            aug_y = pd.Series(np.concatenate([accepted_y.values, inferred_y]))
            weights = np.ones(len(aug_y), dtype=float)

        else:  # fuzzy_augmentation
            # Duplicate rejected population with soft weights: PD for Bad, (1-PD) for Good
            rej_bad_X = rejected_X[feature_cols].copy()
            rej_bad_y = np.ones(n_rejects, dtype=int)
            weights_bad = rejected_pred_pd

            rej_good_X = rejected_X[feature_cols].copy()
            rej_good_y = np.zeros(n_rejects, dtype=int)
            weights_good = 1.0 - rejected_pred_pd

            aug_X = pd.concat(
                [accepted_X[feature_cols], rej_bad_X, rej_good_X],
                ignore_index=True
            )
            aug_y = pd.Series(np.concatenate([accepted_y.values, rej_bad_y, rej_good_y]))
            weights = np.concatenate([
                np.ones(len(accepted_y), dtype=float),
                weights_bad,
                weights_good
            ])

        return aug_X, aug_y, weights
