"""
error_audit.py
==============
In-depth diagnostic error auditing for 24-hour ahead AQI forecasting.

Audits:
1. Phase Lag & Synchronization Error (Cross-Correlation peak alignment).
2. Peak Compression & Under-Prediction Ratio during extreme episodes.
3. Meteorological Residual Breakdown (calm winds & boundary layer stagnation).
4. Automated diagnostic plotting and risk assessment reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate

logger = logging.getLogger(__name__)


class AtmosphericErrorAuditor:
    """
    Diagnostic auditor analyzing failure modes of machine learning and
    deep learning models during extreme pollution episodes.
    """

    def __init__(self, timestamps: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        self.timestamps = timestamps[mask]
        self.y_true = np.asarray(y_true[mask], dtype=float)
        self.y_pred = np.asarray(y_pred[mask], dtype=float)
        self.residuals = self.y_true - self.y_pred
        self.model_name = model_name

    def audit_phase_lag(self, max_lag_hours: int = 36) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Determines whether the model suffers from phase lag (time-delay artifact).
        Computes cross-correlation between predictions and targets across lags [-max_lag, +max_lag].
        A maximum at negative lag indicates the model is lagging behind reality (acting like persistence).
        """
        yt_norm = (self.y_true - np.mean(self.y_true)) / (np.std(self.y_true) + 1e-8)
        yp_norm = (self.y_pred - np.mean(self.y_pred)) / (np.std(self.y_pred) + 1e-8)

        lags = np.arange(-max_lag_hours, max_lag_hours + 1)
        # Compute normalized cross correlation
        ccf = np.correlate(yp_norm, yt_norm, mode="full") / len(yt_norm)
        mid_idx = len(yt_norm) - 1
        ccf_window = ccf[mid_idx - max_lag_hours : mid_idx + max_lag_hours + 1]

        best_lag_idx = np.argmax(ccf_window)
        optimal_lag = int(lags[best_lag_idx])
        max_corr = float(ccf_window[best_lag_idx])
        zero_lag_corr = float(ccf_window[max_lag_hours])

        # Phase lag flaw indicator: optimal lag != 0
        is_lagging = optimal_lag < 0

        logger.info(
            "[%s] Phase Lag Audit: Optimal Lag = %d hrs (Corr=%.3f), Zero-Lag Corr = %.3f. Lagging? %s",
            self.model_name,
            optimal_lag,
            max_corr,
            zero_lag_corr,
            is_lagging,
        )

        return {
            "optimal_lag_hours": optimal_lag,
            "max_cross_correlation": max_corr,
            "zero_lag_correlation": zero_lag_corr,
            "is_lagging": is_lagging,
            "lags": lags,
            "ccf_values": ccf_window,
        }

    def audit_peak_underprediction(self, extreme_threshold: float = 300.0) -> Dict[str, float]:
        """
        Evaluates peak attenuation / regression-to-the-mean during extreme episodes.
        """
        extreme_mask = self.y_true >= extreme_threshold
        n_extreme = int(np.sum(extreme_mask))

        if n_extreme == 0:
            return {"n_extreme": 0, "peak_ratio": np.nan, "mean_underprediction": np.nan, "max_underprediction": np.nan}

        yt_ext = self.y_true[extreme_mask]
        yp_ext = self.y_pred[extreme_mask]

        true_peak = float(np.max(yt_ext))
        pred_peak = float(np.max(yp_ext))
        peak_ratio = pred_peak / true_peak if true_peak > 0 else 1.0

        underpred = yt_ext - yp_ext
        mean_underpred = float(np.mean(underpred))
        max_underpred = float(np.max(underpred))
        pct_underpredicted = float(np.mean(underpred > 0.0) * 100.0)

        return {
            "n_extreme_hours": n_extreme,
            "true_max_aqi": true_peak,
            "pred_max_aqi": pred_peak,
            "peak_ratio": peak_ratio,
            "mean_underprediction_error": mean_underpred,
            "max_underprediction_error": max_underpred,
            "pct_hours_underpredicted": pct_underpredicted,
        }

    def audit_meteorological_stagnation_residuals(
        self, wind_speed: np.ndarray, temp: np.ndarray
    ) -> Dict[str, float]:
        """
        Examines how forecast errors concentrate during atmospheric stagnation:
        - Stagnant air (wind speed < 1.0 m/s)
        - Boundary layer temperature drops
        """
        mask = (~np.isnan(self.residuals)) & (~np.isnan(wind_speed)) & (~np.isnan(temp))
        res = self.residuals[mask]
        ws = wind_speed[mask]
        t = temp[mask]

        stagnant_mask = ws < 1.0
        breezy_mask = ws >= 2.5

        res_stagnant = res[stagnant_mask] if np.any(stagnant_mask) else np.array([np.nan])
        res_breezy = res[breezy_mask] if np.any(breezy_mask) else np.array([np.nan])

        mae_stagnant = float(np.mean(np.abs(res_stagnant)))
        mae_breezy = float(np.mean(np.abs(res_breezy)))
        bias_stagnant = float(np.mean(res_stagnant))  # Positive bias = true > pred (underprediction)

        return {
            "mae_stagnant_air": mae_stagnant,
            "mae_breezy_air": mae_breezy,
            "bias_stagnant_air": bias_stagnant,
            "stagnation_error_inflation_ratio": mae_stagnant / (mae_breezy + 1e-6),
        }

    def plot_diagnostic_dashboard(self, save_path: Optional[Union[str, Path]] = None):
        """
        Generates a 3-panel publication-grade error diagnostic dashboard:
        Panel 1: Actual vs Predicted Time-Series showing peak capture and zoom-in.
        Panel 2: Cross-Correlation Function (CCF) indicating phase synchronization or lag.
        Panel 3: Residuals vs Actual AQI showing regression-to-the-mean underprediction at tail.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]})

        # Panel 1: Time-series zoom (first 250 hours)
        ax1 = axes[0]
        zoom_n = min(300, len(self.timestamps))
        ax1.plot(self.timestamps[:zoom_n], self.y_true[:zoom_n], label="Actual AQI (t+24)", color="#1f77b4", lw=2.0)
        ax1.plot(self.timestamps[:zoom_n], self.y_pred[:zoom_n], label=f"{self.model_name} Forecast", color="#d62728", lw=1.8, ls="--")
        ax1.axhline(300, color="orange", ls=":", label="CPCB Very Poor (300)")
        ax1.axhline(400, color="red", ls=":", label="CPCB Severe (400)")
        ax1.set_title(f"A. 24-Hour Ahead Forecast Horizon Time-Series [{self.model_name}]", fontsize=12, fontweight="bold")
        ax1.set_ylabel("AQI")
        ax1.legend(loc="upper right", frameon=True)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Phase Lag Cross-Correlation
        ax2 = axes[1]
        lag_res = self.audit_phase_lag(max_lag_hours=30)
        ax2.plot(lag_res["lags"], lag_res["ccf_values"], color="#2ca02c", lw=2.0, marker="o", ms=3)
        ax2.axvline(0, color="black", ls="--", label="Ideal Zero-Lag Alignment (k=0)")
        ax2.axvline(lag_res["optimal_lag_hours"], color="crimson", ls=":", label=f"Optimal Peak Lag ({lag_res['optimal_lag_hours']}h)")
        ax2.set_title("B. Phase Synchronization Audit (Normalized Cross-Correlation)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Time Shift k (Hours)")
        ax2.set_ylabel("Cross-Correlation")
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Residuals vs Actual AQI
        ax3 = axes[2]
        ax3.scatter(self.y_true, self.residuals, alpha=0.35, color="#9467bd", edgecolors="none", s=20)
        ax3.axhline(0, color="black", ls="--")
        ax3.axvline(300, color="orange", ls=":", label="Severe Threshold (300)")
        # Plot mean residual trend line
        bins = np.linspace(min(self.y_true), max(self.y_true), 12)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_means = [np.mean(self.residuals[(self.y_true >= bins[i]) & (self.y_true < bins[i+1])]) for i in range(len(bins)-1)]
        ax3.plot(bin_centers, bin_means, color="red", lw=2.5, marker="s", label="Conditional Residual Trend")
        ax3.set_title("C. Residual Bias vs Actual Pollution Severity (Tail Compression Audit)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Actual AQI")
        ax3.set_ylabel("Residual (Actual - Predicted)")
        ax3.legend(loc="upper left", frameon=True)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            logger.info("Saved diagnostic dashboard to %s", save_path)
        plt.close()
