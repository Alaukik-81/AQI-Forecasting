"""
quantile_regression.py
======================
Graduate Depth Component (ES 678):
Uncertainty Quantification (UQ) via Multi-Quantile Regression for 24-Hour Ahead AQI.

For severe environmental episodes, point predictions are hazardous for municipal decision-makers.
This module fits models for P10, P50, and P90 using Pinball (Quantile) Loss, yielding
an 80% Prediction Interval: [P10, P90].

Metrics:
1. Prediction Interval Coverage Probability (PICP): Target >= 80% nominal coverage.
2. Mean Prediction Interval Width (MPIW): Evaluates sharpness of the uncertainty bound.
3. Pinball Loss / Quantile Score.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Compute empirical Pinball (Quantile) Loss for a given quantile q in (0, 1)."""
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1.0) * error)
    return float(np.mean(loss))


def evaluate_prediction_intervals(
    y_true: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    p50: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes PICP (Coverage) and MPIW (Sharpness) metrics.
    """
    mask = (~np.isnan(y_true)) & (~np.isnan(p10)) & (~np.isnan(p90))
    yt = y_true[mask]
    lo = p10[mask]
    hi = p90[mask]

    # Prediction Interval Coverage Probability (PICP)
    covered = (yt >= lo) & (yt <= hi)
    picp = float(np.mean(covered) * 100.0)

    # Mean Prediction Interval Width (MPIW)
    width = hi - lo
    mpiw = float(np.mean(width))

    res = {
        "Nominal_Confidence_%": 80.0,
        "PICP_Coverage_%": picp,
        "MPIW_Width": mpiw,
        "Min_Interval_Width": float(np.min(width)),
        "Max_Interval_Width": float(np.max(width)),
    }

    if p50 is not None:
        med = p50[mask]
        res["Median_MAE"] = float(np.mean(np.abs(yt - med)))
        res["Pinball_P10"] = pinball_loss(yt, lo, 0.10)
        res["Pinball_P50"] = pinball_loss(yt, med, 0.50)
        res["Pinball_P90"] = pinball_loss(yt, hi, 0.90)

    return res


class MultiQuantileTreeForecaster:
    """
    Fits three independent gradient boosted trees corresponding to quantiles:
    q = 0.10 (lower bound), q = 0.50 (median estimate), q = 0.90 (upper bound).
    Guarantees monotonic sorting to prevent quantile crossing.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.models: Dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MultiQuantileTreeForecaster":
        for q in self.quantiles:
            logger.info("Fitting LightGBM Quantile Regressor for q=%.2f...", q)
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            self.models[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X)

        # Monotonic sorting adjustment to strictly prevent quantile crossing (P10 <= P50 <= P90)
        if 0.10 in preds and 0.50 in preds and 0.90 in preds:
            stacked = np.column_stack([preds[0.10], preds[0.50], preds[0.90]])
            sorted_stacked = np.sort(stacked, axis=1)
            preds[0.10] = sorted_stacked[:, 0]
            preds[0.50] = sorted_stacked[:, 1]
            preds[0.90] = sorted_stacked[:, 2]

        return preds
