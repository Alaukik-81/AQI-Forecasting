"""
metrics.py
==========
Atmospheric evaluation metrics and stratified extreme event slices for AQI forecasting.

Includes:
1. Standard regression metrics: RMSE, MAE, MAPE, R2, Pearson r, Willmott's Index of Agreement (d).
2. Stratified evaluation slices: CPCB categories (Moderate, Poor, Very Poor, Severe).
3. Dedicated Extreme Event Slice (Actual AQI > 300 and > 400).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def compute_standard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute foundational time-series regression metrics."""
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan, "Pearson_r": np.nan, "Willmott_d": np.nan}

    diff = yp - yt
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    # MAPE (avoid div by zero with small eps)
    non_zero = np.abs(yt) > 1e-3
    mape = float(np.mean(np.abs((yt[non_zero] - yp[non_zero]) / yt[non_zero])) * 100.0) if np.any(non_zero) else np.nan

    # R2
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    # Pearson r
    pr, _ = pearsonr(yt, yp) if len(yt) > 1 and np.std(yt) > 0 and np.std(yp) > 0 else (np.nan, np.nan)

    # Willmott's Index of Agreement (d): Standard atmospheric evaluation metric
    # Bounds: 0 (no agreement) to 1 (perfect agreement)
    y_mean = np.mean(yt)
    denom = np.sum((np.abs(yp - y_mean) + np.abs(yt - y_mean)) ** 2)
    d = float(1.0 - (ss_res / (denom + 1e-8)))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Pearson_r": float(pr),
        "Willmott_d": d,
        "N_samples": len(yt),
    }


def compute_stratified_slices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extreme_threshold: float = 300.0,
    severe_threshold: float = 400.0,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates model performance stratified across environmental severity tiers.
    Crucial for assessing public health warning reliability.
    """
    slices = {}

    # 1. Overall
    slices["Overall"] = compute_standard_metrics(y_true, y_pred)

    # 2. Extreme Slice (AQI > 300: Very Poor + Severe)
    extreme_mask = y_true >= extreme_threshold
    slices[f"Extreme (AQI > {int(extreme_threshold)})"] = compute_standard_metrics(y_true[extreme_mask], y_pred[extreme_mask])

    # 3. Severe Hazard Slice (AQI > 400: Emergency Action Plan)
    severe_mask = y_true >= severe_threshold
    slices[f"Hazardous (AQI > {int(severe_threshold)})"] = compute_standard_metrics(y_true[severe_mask], y_pred[severe_mask])

    # 4. Moderate Slice (100 < AQI <= 200)
    mod_mask = (y_true > 100.0) & (y_true <= 200.0)
    slices["Moderate (101-200)"] = compute_standard_metrics(y_true[mod_mask], y_pred[mod_mask])

    # 5. Poor Slice (200 < AQI <= 300)
    poor_mask = (y_true > 200.0) & (y_true <= 300.0)
    slices["Poor (201-300)"] = compute_standard_metrics(y_true[poor_mask], y_pred[poor_mask])

    return slices


def format_evaluation_summary_table(slice_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Format dictionary of slice metrics into a clean publication-ready DataFrame."""
    rows = []
    for slice_name, metrics in slice_results.items():
        row = {"Slice": slice_name}
        row.update(metrics)
        rows.append(row)
    df_out = pd.DataFrame(rows).set_index("Slice")
    return df_out
