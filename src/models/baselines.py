"""
baselines.py
============
Naive and statistical reference baselines for 24-hour ahead AQI forecasting.

Models:
1. Pure Persistence Baseline: y_hat_{t+24} = y_t
   Assumption: The current atmospheric state will persist unchanged 24 hours into the future.
2. Seasonal Persistence (Daily Cycle): y_hat_{t+24} = y_{t} (or rolling 24h diurnal average)
3. Seasonal Weekly Persistence: y_hat_{t+24} = y_{t+24 - 168} = y_{t - 144}
   Assumption: Weekly traffic and emission rhythms dominate future pollution states.
"""

from __future__ import annotations

import logging
from typing import Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PurePersistenceModel:
    """
    Pure Persistence Baseline:
    Predicts that AQI 24 hours ahead is identical to current AQI at time t.
    y_hat_{t+24} = y_t
    """

    def __init__(self, horizon: int = 24):
        self.horizon = horizon

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "PurePersistenceModel":
        return self  # Non-parametric

    def predict(self, df_with_raw_target: pd.DataFrame, target_col: str = "AQI") -> np.ndarray:
        """
        Takes raw unshifted DataFrame and returns persistence forecasts.
        At index t, the prediction for t+horizon is df[target_col].iloc[t].
        """
        return df_with_raw_target[target_col].values


class SeasonalWeeklyPersistenceModel:
    """
    Seasonal Weekly Persistence Baseline:
    y_hat_{t+24} = y_{t - 144} (same hour exactly one week prior to the target timestamp).
    """

    def __init__(self, horizon: int = 24, weekly_period: int = 168):
        self.horizon = horizon
        self.weekly_lag = weekly_period - horizon  # 168 - 24 = 144 hours in past

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "SeasonalWeeklyPersistenceModel":
        return self

    def predict(self, df_with_raw_target: pd.DataFrame, target_col: str = "AQI") -> np.ndarray:
        series = df_with_raw_target[target_col]
        # Shift past observation forward: at index t, use value at t - 144
        forecasts = series.shift(self.weekly_lag).bfill().values
        return forecasts
