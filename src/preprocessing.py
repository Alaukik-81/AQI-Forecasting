"""
preprocessing.py
================
Strict zero-leakage preprocessing, temporal feature engineering,
bounded imputation, and sliding-window sequential tensor formulation.

Key Design Principles:
1. NO LOOK-AHEAD BIAS: Learned parameters (imputation medians, scalers) are fitted
   STRICTLY on the training partition and applied out-of-sample to validation and test.
2. CAUSAL ROLLING FILTERS: All rolling statistics are strictly backward-looking.
3. BOUNDED IMPUTATION: Missing values are only forward-filled up to a fixed threshold (e.g. 6 hours).
   Residual gaps are filled using strictly training-set medians.
4. PHYSICAL VECTOR TRANSFORMS: Wind speed and direction are decomposed into orthogonal
   U (zonal) and V (meridional) vectors to eliminate angular discontinuity (0 deg == 360 deg).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Split Specification
# ---------------------------------------------------------------------------
@dataclass
class DatasetSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_dates: Tuple[pd.Timestamp, pd.Timestamp]
    val_dates: Tuple[pd.Timestamp, pd.Timestamp]
    test_dates: Tuple[pd.Timestamp, pd.Timestamp]


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DatasetSplits:
    """
    Split time-series data chronologically to preserve temporal order.
    NEVER shuffles or uses future timestamps in training partitions.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0."
    df_sorted = df.sort_index().copy()
    n = len(df_sorted)

    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end_idx].copy()
    val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
    test_df = df_sorted.iloc[val_end_idx:].copy()

    logger.info(
        "Chronological Split complete:\n"
        "  Train: %s to %s (%d hours, %.1f%%)\n"
        "  Val:   %s to %s (%d hours, %.1f%%)\n"
        "  Test:  %s to %s (%d hours, %.1f%%)",
        train_df.index.min(), train_df.index.max(), len(train_df), (len(train_df)/n)*100,
        val_df.index.min(), val_df.index.max(), len(val_df), (len(val_df)/n)*100,
        test_df.index.min(), test_df.index.max(), len(test_df), (len(test_df)/n)*100,
    )

    return DatasetSplits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_dates=(train_df.index.min(), train_df.index.max()),
        val_dates=(val_df.index.min(), val_df.index.max()),
        test_dates=(test_df.index.min(), test_df.index.max()),
    )


# ---------------------------------------------------------------------------
# Temporal Feature Engineering (Causal & Physical)
# ---------------------------------------------------------------------------
class EnvironmentalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs domain-specific meteorological and temporal features:
    1. Cyclical sin/cos encodings for hour-of-day, day-of-week, day-of-year.
    2. Wind vector decomposition: U (zonal) and V (meridional).
    3. Ventilation & boundary layer stagnation proxy.
    4. Causal backward-looking rolling statistics (mean, std, max) and lag features.
    """

    def __init__(
        self,
        pollutant_cols: Optional[List[str]] = None,
        met_cols: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        include_rolling: bool = True,
    ):
        self.pollutant_cols = pollutant_cols or ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        self.met_cols = met_cols or ["Temperature", "Relative_Humidity", "Wind_Speed", "Pressure"]
        self.lags = lags or [1, 2, 3, 6, 12, 24, 48, 72]
        self.rolling_windows = rolling_windows or [6, 12, 24, 72]
        self.include_rolling = include_rolling

    def fit(self, X: pd.DataFrame, y=None):
        return self  # Stateless transformation

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply causal feature engineering to DataFrame with DatetimeIndex."""
        res = df.copy()
        if "Timestamp" in res.columns and not isinstance(res.index, pd.DatetimeIndex):
            res["Timestamp"] = pd.to_datetime(res["Timestamp"])
            res = res.set_index("Timestamp")
        elif not isinstance(res.index, pd.DatetimeIndex):
            res.index = pd.to_datetime(res.index)

        # 1. Cyclical Calendar Encodings
        hours = res.index.hour
        days_of_week = res.index.dayofweek
        days_of_year = res.index.dayofyear

        res["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
        res["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
        res["sin_dow"] = np.sin(2 * np.pi * days_of_week / 7.0)
        res["cos_dow"] = np.cos(2 * np.pi * days_of_week / 7.0)
        res["sin_doy"] = np.sin(2 * np.pi * days_of_year / 365.25)
        res["cos_doy"] = np.cos(2 * np.pi * days_of_year / 365.25)

        # 2. Wind Vector Decomposition (U = East-West, V = North-South)
        if "Wind_Speed" in res.columns and "Wind_Direction" in res.columns:
            rad = np.deg2rad(res["Wind_Direction"])
            res["wind_u"] = -res["Wind_Speed"] * np.sin(rad)
            res["wind_v"] = -res["Wind_Speed"] * np.cos(rad)

        # 3. Atmospheric Dispersion & Ventilation Proxy
        if "Wind_Speed" in res.columns and "Temperature" in res.columns:
            # Proxy for atmospheric mixing: ventilation index proportional to wind speed * thermal buoyancy
            res["ventilation_index_proxy"] = res["Wind_Speed"] * np.maximum(1.0, res["Temperature"] + 15.0)

        # 4. Causal Lag Features for Primary Pollutants and Target AQI
        key_vars = ["AQI", "PM2.5", "PM10"]
        for var in key_vars:
            if var in res.columns:
                for lag in self.lags:
                    res[f"{var}_lag_{lag}h"] = res[var].shift(lag)

        # 5. Causal Rolling Statistics (Strictly closed='right', backward-looking only)
        if self.include_rolling:
            for var in ["AQI", "PM2.5"]:
                if var in res.columns:
                    for w in self.rolling_windows:
                        # Shift by 0 or 1? At time t, value at time t is known.
                        # Rolling over past w hours including t:
                        res[f"{var}_roll_mean_{w}h"] = res[var].rolling(window=w, min_periods=max(1, w // 2)).mean()
                        res[f"{var}_roll_std_{w}h"] = res[var].rolling(window=w, min_periods=max(1, w // 2)).std()
                        res[f"{var}_roll_max_{w}h"] = res[var].rolling(window=w, min_periods=max(1, w // 2)).max()
                        res[f"{var}_roll_min_{w}h"] = res[var].rolling(window=w, min_periods=max(1, w // 2)).min()

        return res


# ---------------------------------------------------------------------------
# Zero-Leakage Pipeline: Bounded Imputation & Distribution Scaling
# ---------------------------------------------------------------------------
class LeakageFreePreprocessor:
    """
    Encapsulates bounded temporal imputation, training-split parameter estimation,
    and feature/target normalization.

    Parameters
    ----------
    max_imputation_gap_hours : int
        Maximum consecutive missing hours to forward-fill.
    scaler_type : str
        'robust' (uses median and IQR, optimal for heavy-tailed pollution spikes)
        or 'standard' (zero mean, unit variance).
    target_col : str
        Name of the prediction target variable ('AQI').
    """

    def __init__(
        self,
        max_imputation_gap_hours: int = 6,
        scaler_type: str = "robust",
        target_col: str = "AQI",
    ):
        self.max_imputation_gap_hours = max_imputation_gap_hours
        self.scaler_type = scaler_type
        self.target_col = target_col

        self.feature_scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.target_scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.train_medians: Optional[pd.Series] = None
        self.feature_cols: Optional[List[str]] = None
        self.is_fitted: bool = False

    def _impute_bounded(self, df: pd.DataFrame, fill_with_medians: bool = False) -> pd.DataFrame:
        """
        1. Forward-fill short gaps up to `max_imputation_gap_hours`.
        2. If fill_with_medians=True and train_medians is available, impute residual NaNs
           using strictly the fitted training-set medians.
        """
        res = df.copy()
        # Strictly causal forward fill (only bounded past values propagate forward)
        res = res.ffill(limit=self.max_imputation_gap_hours)

        if fill_with_medians and self.train_medians is not None:
            # Impute any initial boundary NaNs or extended outages with training medians
            res = res.fillna(self.train_medians)
        return res

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> "LeakageFreePreprocessor":
        """
        Fit imputation statistics and feature/target scalers strictly on train_df.
        """
        self.feature_cols = [c for c in feature_cols if c in train_df.columns and c != self.target_col]

        # 1. Bounded forward-fill on training data
        clean_train = self._impute_bounded(train_df, fill_with_medians=False)

        # 2. Learn medians solely from the training partition
        self.train_medians = clean_train[self.feature_cols + [self.target_col]].median()

        # Fill any remaining training NaNs with learned training medians
        clean_train = clean_train.fillna(self.train_medians)

        # 3. Instantiate and fit feature scalers
        if self.scaler_type == "robust":
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()

        self.feature_scaler.fit(clean_train[self.feature_cols])
        self.target_scaler.fit(clean_train[[self.target_col]])

        self.is_fitted = True
        logger.info(
            "LeakageFreePreprocessor successfully fitted on %d training rows across %d features.",
            len(clean_train),
            len(self.feature_cols),
        )
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Imputes and scales features and target using solely pre-fitted training statistics.
        Returns (scaled_X_df, scaled_y_series).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training split before transforming.")

        # 1. Apply bounded causal imputation followed by training median fallback
        clean_df = self._impute_bounded(df, fill_with_medians=True)

        # 2. Scale features
        X_scaled_arr = self.feature_scaler.transform(clean_df[self.feature_cols])
        X_scaled = pd.DataFrame(X_scaled_arr, index=clean_df.index, columns=self.feature_cols)

        # 3. Scale target
        y_scaled_arr = self.target_scaler.transform(clean_df[[self.target_col]])
        y_scaled = pd.Series(y_scaled_arr.ravel(), index=clean_df.index, name=self.target_col)

        return X_scaled, y_scaled

    def inverse_transform_target(self, y_scaled: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Invert scaled predictions back to physical AQI units (0-500 scale)."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted.")
        arr = np.asarray(y_scaled).reshape(-1, 1)
        inv = self.target_scaler.inverse_transform(arr)
        return inv.ravel()


# ---------------------------------------------------------------------------
# Supervised Dataset Formulation: Direct 24-Hour Ahead Prediction
# ---------------------------------------------------------------------------
def create_direct_forecasting_dataset(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    horizon: int = 24,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates tabular (X_t, y_{t+h}) pairs for gradient boosted trees.

    At hour t, features X_t represent all past and current information up to hour t.
    The prediction target is y_{t+h} (e.g., h=24).

    Parameters
    ----------
    X_df : pd.DataFrame
        Engineered feature set up to hour t.
    y_series : pd.Series
        Target series at hour t.
    horizon : int
        Forecast horizon (default = 24 hours).

    Returns
    -------
    (X_aligned, y_aligned) : Tuple[pd.DataFrame, pd.Series]
        Properly shifted and time-aligned tabular dataset.
    """
    # Shift target backward by horizon hours so that row t contains y_{t+horizon}
    y_target = y_series.shift(-horizon)

    # Align and drop trailing unobserved targets
    valid_mask = (~X_df.isna().any(axis=1)) & (~y_target.isna())
    X_aligned = X_df.loc[valid_mask]
    y_aligned = y_target.loc[valid_mask]

    return X_aligned, y_aligned


# ---------------------------------------------------------------------------
# Sequential 3D Tensor Generation (LSTM / GRU / Transformer)
# ---------------------------------------------------------------------------
def create_sequential_tensors(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    lookback_window: int = 72,
    horizon: int = 24,
    multi_step: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 2D time-series matrices into 3D sliding window tensors:
    - X_seq: Shape (N, lookback_window, n_features)
    - y_seq: Shape (N,) if direct single-step at t+horizon, or (N, horizon) if multi_step sequence
    - time_indices: 1D array of sample forecast origin indices (t)

    Mathematical Formulation:
      For sample index i, corresponding to forecast origin t:
      X_seq[i] = [ X_{t - lookback_window + 1}, ..., X_t ]
      y_seq[i] = y_{t + horizon}   (if direct)
               = [ y_{t + 1}, ..., y_{t + horizon} ] (if multi_step)
    """
    n_samples = len(X_arr)
    max_idx = n_samples - horizon if multi_step or horizon > 0 else n_samples

    X_list = []
    y_list = []
    idx_list = []

    for t in range(lookback_window - 1, max_idx):
        window_x = X_arr[t - lookback_window + 1 : t + 1]

        if multi_step:
            window_y = y_arr[t + 1 : t + horizon + 1]
        else:
            window_y = y_arr[t + horizon]

        # Check that window does not contain any NaNs
        if not np.isnan(window_x).any() and not np.isnan(window_y).any():
            X_list.append(window_x)
            y_list.append(window_y)
            idx_list.append(t)

    X_seq = np.array(X_list, dtype=np.float32)
    y_seq = np.array(y_list, dtype=np.float32)
    indices = np.array(idx_list, dtype=np.int64)

    return X_seq, y_seq, indices


# ---------------------------------------------------------------------------
# Integrated Preprocessing Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_leakage_free_pipeline(
    df: pd.DataFrame,
    lookback_window: int = 72,
    horizon: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scaler_type: str = "robust",
) -> Dict[str, Union[pd.DataFrame, np.ndarray, LeakageFreePreprocessor, DatasetSplits]]:
    """
    Executes the complete leakage-free pipeline:
    1. Chronological splitting.
    2. Causal feature engineering with lag/rolling statistics across full sequence.
       (Note: rolling features use past values only; feature scalers are strictly fitted on train).
    3. Scaler fitting on train and out-of-sample transformation of val and test.
    4. Alignment into tabular tabular formats (for GBDTs) and 3D sequential tensors (for Deep Learning).
    """
    logger.info("Starting integrated Leakage-Free Pipeline (Horizon=%dh, Lookback=%dh)...", horizon, lookback_window)

    # 1. Feature Engineering (causal and stateless transforms)
    engineer = EnvironmentalFeatureEngineer()
    df_engineered = engineer.transform(df)

    # Select engineered feature columns (exclude raw non-features like Station_ID)
    exclude_cols = ["Station_ID", "Timestamp"]
    candidate_features = [c for c in df_engineered.columns if c not in exclude_cols and c != "AQI"]

    # 2. Chronological Split
    splits = temporal_train_val_test_split(
        df_engineered,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # 3. Fit Preprocessor strictly on training split
    preprocessor = LeakageFreePreprocessor(scaler_type=scaler_type, target_col="AQI")
    preprocessor.fit(splits.train_df, candidate_features)

    # 4. Transform each split independently
    train_X_scaled, train_y_scaled = preprocessor.transform(splits.train_df)
    val_X_scaled, val_y_scaled = preprocessor.transform(splits.val_df)
    test_X_scaled, test_y_scaled = preprocessor.transform(splits.test_df)

    # 5. Formulate Tabular Direct Forecasting Datasets for Trees
    X_train_tab, y_train_tab = create_direct_forecasting_dataset(train_X_scaled, train_y_scaled, horizon=horizon)
    X_val_tab, y_val_tab = create_direct_forecasting_dataset(val_X_scaled, val_y_scaled, horizon=horizon)
    X_test_tab, y_test_tab = create_direct_forecasting_dataset(test_X_scaled, test_y_scaled, horizon=horizon)

    # 6. Formulate Sequential 3D Tensors for Deep Learning
    X_train_seq, y_train_seq, _ = create_sequential_tensors(
        train_X_scaled.values, train_y_scaled.values, lookback_window=lookback_window, horizon=horizon
    )
    X_val_seq, y_val_seq, _ = create_sequential_tensors(
        val_X_scaled.values, val_y_scaled.values, lookback_window=lookback_window, horizon=horizon
    )
    X_test_seq, y_test_seq, _ = create_sequential_tensors(
        test_X_scaled.values, test_y_scaled.values, lookback_window=lookback_window, horizon=horizon
    )

    logger.info(
        "Preprocessing complete:\n"
        "  Tabular Shapes: Train %s, Val %s, Test %s\n"
        "  Sequential Tensors: Train %s, Val %s, Test %s",
        X_train_tab.shape, X_val_tab.shape, X_test_tab.shape,
        X_train_seq.shape, X_val_seq.shape, X_test_seq.shape,
    )

    return {
        "splits": splits,
        "preprocessor": preprocessor,
        "feature_cols": preprocessor.feature_cols,
        # Tabular datasets for GBDTs
        "X_train_tab": X_train_tab,
        "y_train_tab": y_train_tab,
        "X_val_tab": X_val_tab,
        "y_val_tab": y_val_tab,
        "X_test_tab": X_test_tab,
        "y_test_tab": y_test_tab,
        # Sequential Tensors for Deep Learning
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
    }


if __name__ == "__main__":
    from data_loader import load_and_validate_data

    raw_df = load_and_validate_data()
    pipeline_data = run_leakage_free_pipeline(raw_df)
    print("Preprocessing executed successfully.")
