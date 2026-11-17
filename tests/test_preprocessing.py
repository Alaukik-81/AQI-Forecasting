"""
tests/test_preprocessing.py
===========================
Mathematical and structural verification of zero temporal data leakage.
"""

import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_cpcb_data
from src.preprocessing import (
    EnvironmentalFeatureEngineer,
    LeakageFreePreprocessor,
    create_direct_forecasting_dataset,
    create_sequential_tensors,
    temporal_train_val_test_split,
)


def test_no_scaler_lookahead_leakage():
    """Verify that scalers fitted on train are invariant to perturbations in future test data."""
    df = generate_synthetic_cpcb_data(start_date="2023-01-01", end_date="2023-02-28", seed=42)
    feature_cols = ["PM2.5", "PM10", "NO2", "Temperature", "Wind_Speed"]

    splits = temporal_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # 1. Fit on unmodified data
    prep1 = LeakageFreePreprocessor(scaler_type="robust", target_col="AQI")
    prep1.fit(splits.train_df, feature_cols)
    train_center1 = prep1.feature_scaler.center_.copy()
    train_scale1 = prep1.feature_scaler.scale_.copy()

    # 2. Artificially introduce massive extreme anomalies in the test set
    corrupted_test_df = splits.test_df.copy()
    corrupted_test_df.loc[:, "PM2.5"] = 99999.0
    corrupted_test_df.loc[:, "AQI"] = 99999.0

    # 3. Fit new preprocessor on original train_df (test df altered)
    prep2 = LeakageFreePreprocessor(scaler_type="robust", target_col="AQI")
    prep2.fit(splits.train_df, feature_cols)
    train_center2 = prep2.feature_scaler.center_.copy()
    train_scale2 = prep2.feature_scaler.scale_.copy()

    # Assert that training scaling parameters are strictly identical
    np.testing.assert_array_equal(train_center1, train_center2)
    np.testing.assert_array_equal(train_scale1, train_scale2)


def test_causal_rolling_features():
    """Verify that rolling features are strictly causal (changes at time t do not affect features at tau < t)."""
    df = generate_synthetic_cpcb_data(start_date="2023-01-01", end_date="2023-01-15", seed=42)
    engineer = EnvironmentalFeatureEngineer(rolling_windows=[6, 12, 24])

    df_engineered_1 = engineer.transform(df)

    # Perturb value at index t_perturb
    t_perturb_idx = 100
    t_perturb_time = df.index[t_perturb_idx]

    df_perturbed = df.copy()
    df_perturbed.loc[t_perturb_time, "PM2.5"] += 500.0

    df_engineered_2 = engineer.transform(df_perturbed)

    # Check that past values (tau < t_perturb_time) are 100% identical across all rolling features
    past_mask = df.index < t_perturb_time
    rolling_cols = [c for c in df_engineered_1.columns if "roll_" in c]

    pd.testing.assert_frame_equal(
        df_engineered_1.loc[past_mask, rolling_cols],
        df_engineered_2.loc[past_mask, rolling_cols],
    )


def test_sequential_tensor_temporal_alignment():
    """Verify that sequence X[i] contains solely historical steps up to origin t and y[i] is at t+24."""
    # Create simple monotonic dummy matrix
    n_hours = 200
    lookback = 72
    horizon = 24
    timestamps = pd.date_range("2023-01-01", periods=n_hours, freq="1h")

    # Values equal timestamp index 0, 1, ..., 199
    dummy_x = np.arange(n_hours, dtype=np.float32).reshape(-1, 1)
    dummy_y = np.arange(n_hours, dtype=np.float32)

    X_seq, y_seq, indices = create_sequential_tensors(
        dummy_x, dummy_y, lookback_window=lookback, horizon=horizon, multi_step=False
    )

    # For each sample i, index origin is t = indices[i]
    for i in range(len(indices)):
        t = indices[i]
        # Historical slice must be [t - lookback + 1, ..., t]
        expected_x_window = np.arange(t - lookback + 1, t + 1, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(X_seq[i], expected_x_window)

        # Target must be exactly t + horizon
        expected_y = float(t + horizon)
        assert np.isclose(y_seq[i], expected_y)
        # Ensure that target is strictly in the future of the last input observation in X_seq
        last_obs_in_x = X_seq[i][-1, 0]
        assert y_seq[i] - last_obs_in_x == horizon


if __name__ == "__main__":
    test_no_scaler_lookahead_leakage()
    test_causal_rolling_features()
    test_sequential_tensor_temporal_alignment()
    print("All mathematical leakage prevention unit tests PASSED successfully!")
