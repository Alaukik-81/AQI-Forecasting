"""
trees.py
========
Machine Learning tree-based baselines (LightGBM and XGBoost)
configured for direct 24-hour ahead regression.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


class LightGBMForecaster:
    """
    LightGBM Gradient Boosted Decision Trees for 24-hour ahead point prediction.
    Features include multi-lag pollutants, causal rolling statistics, and cyclical encodings.
    """

    def __init__(
        self,
        n_estimators: int = 350,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "regression",
            "verbosity": -1,
            "n_jobs": -1,
        }
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 25,
    ) -> "LightGBMForecaster":
        self.feature_names = list(X_train.columns)
        self.model = lgb.LGBMRegressor(**self.params)

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)] if eval_set else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        logger.info("LightGBM successfully trained (Best iteration: %s).", getattr(self.model, "best_iteration_", "N/A"))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBMForecaster must be fitted before predict.")
        return self.model.predict(X)

    def get_feature_importances(self, top_n: int = 15) -> pd.Series:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model is not fitted.")
        imp = pd.Series(self.model.feature_importances_, index=self.feature_names)
        return imp.sort_values(ascending=False).head(top_n)


class XGBoostForecaster:
    """
    XGBoost Gradient Boosted Trees for 24-hour ahead point prediction.
    """

    def __init__(
        self,
        n_estimators: int = 350,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 25,
    ) -> "XGBoostForecaster":
        self.feature_names = list(X_train.columns)
        self.model = xgb.XGBRegressor(**self.params, early_stopping_rounds=early_stopping_rounds if X_val is not None else None)

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info("XGBoost successfully trained (Best iteration: %s).", getattr(self.model, "best_iteration", "N/A"))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoostForecaster must be fitted before predict.")
        return self.model.predict(X)

    def get_feature_importances(self, top_n: int = 15) -> pd.Series:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model is not fitted.")
        imp = pd.Series(self.model.feature_importances_, index=self.feature_names)
        return imp.sort_values(ascending=False).head(top_n)
