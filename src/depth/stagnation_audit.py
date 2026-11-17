"""
stagnation_audit.py
===================
Graduate Depth Component (ES 678):
Micro-meteorological Stagnation & Inversion Failure Audit.

Quantifies model failure mechanisms under planetary boundary layer (PBL) compression:
1. Calm wind regimes (< 1.0 m/s) where horizontal advection ceases.
2. Low temperature and high relative humidity conditions characteristic of North Indian winter fog.
3. Quantifies how model error inflates non-linearly when physical dispersion shuts down.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def audit_stagnation_subgroups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wind_speed: np.ndarray,
    temperature: np.ndarray,
    relative_humidity: np.ndarray,
) -> pd.DataFrame:
    """
    Partitions evaluation records into physical meteorological regimes and computes
    stratified MAE, RMSE, and bias.
    """
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "error": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
            "wind_speed": wind_speed,
            "temperature": temperature,
            "humidity": relative_humidity,
        }
    ).dropna()

    regimes = {
        "All Hours": df,
        "Stagnant Air (Wind < 1.0 m/s)": df[df["wind_speed"] < 1.0],
        "Breezy Air (Wind >= 2.5 m/s)": df[df["wind_speed"] >= 2.5],
        "Cold Stagnation (Wind < 1.0 m/s & Temp < 15C)": df[(df["wind_speed"] < 1.0) & (df["temperature"] < 15.0)],
        "Winter Fog Proxy (Wind < 1.5 m/s & RH > 75%)": df[(df["wind_speed"] < 1.5) & (df["humidity"] > 75.0)],
        "High Ventilation (Wind >= 3.0 m/s & Temp > 25C)": df[(df["wind_speed"] >= 3.0) & (df["temperature"] > 25.0)],
    }

    records = []
    for regime_name, sub_df in regimes.items():
        n = len(sub_df)
        if n == 0:
            continue
        mae = sub_df["abs_error"].mean()
        rmse = np.sqrt((sub_df["error"] ** 2).mean())
        bias = sub_df["error"].mean()  # pred - true (negative = under-prediction)
        mean_true_aqi = sub_df["y_true"].mean()

        records.append(
            {
                "Meteorological Regime": regime_name,
                "Hours_Count": n,
                "Pct_Total_Time": (n / len(df)) * 100.0,
                "Mean_Actual_AQI": mean_true_aqi,
                "MAE": mae,
                "RMSE": rmse,
                "Mean_Bias": bias,
            }
        )

    res_df = pd.DataFrame(records).set_index("Meteorological Regime")
    return res_df
