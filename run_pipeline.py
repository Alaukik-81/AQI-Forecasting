"""
run_pipeline.py
===============
End-to-end execution pipeline for ES 678 research project:
"24-Hour Ahead Hourly Air Quality Index (AQI) Forecasting:
Comparing Sequential Deep Learning, Tree-Based Baselines, and Persistence under Extreme Pollution Events"

Executes:
1. Data Ingestion & Diagnostic Summary
2. Leakage-Free Preprocessing Pipeline
3. Model Training (Baselines, LightGBM, XGBoost, Attention-LSTM, Transformer)
4. Comprehensive & Stratified Extreme Slices Evaluation
5. Error Audits (Phase Lag, Peak Compression, Stagnation Breakdown)
6. Graduate Depth (Quantile Regression & Prediction Intervals)
7. Report Figures & LaTeX Tables Export
"""

import argparse
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import audit_dataset, load_and_validate_data
from src.depth.quantile_regression import MultiQuantileTreeForecaster, evaluate_prediction_intervals
from src.depth.stagnation_audit import audit_stagnation_subgroups
from src.evaluation.error_audit import AtmosphericErrorAuditor
from src.evaluation.metrics import compute_standard_metrics, compute_stratified_slices, format_evaluation_summary_table
from src.models.baselines import PurePersistenceModel, SeasonalWeeklyPersistenceModel
from src.models.deep_models import DeepLearningTrainer, RecurrentForecaster, TransformerForecaster
from src.models.trees import LightGBMForecaster, XGBoostForecaster
from src.preprocessing import run_leakage_free_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Pipeline")


def main():
    parser = argparse.ArgumentParser(description="ES 678 AQI 24h Forecasting Pipeline")
    parser.add_argument("--data-path", type=str, default="data/raw/station_hourly.csv", help="Path to raw CSV dataset")
    parser.add_argument("--real", action="store_true", help="Use empirical CPCB Delhi dataset (delhi_cpcb_real_hourly.csv)")
    args = parser.parse_args()

    data_file = "data/raw/delhi_cpcb_real_hourly.csv" if args.real else args.data_path

    print("=" * 80)
    print("ES 678: 24-HOUR AHEAD HOURLY AIR QUALITY FORECASTING PIPELINE")
    print(f"Target Dataset: {data_file}")
    print("=" * 80)

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Data Ingestion & Diagnostics
    # -----------------------------------------------------------------------
    logger.info("Step 1: Ingesting dataset from %s...", data_file)
    df = load_and_validate_data(data_file, synthetic_backup=True)
    audit = audit_dataset(df)

    print("\n--- DATASET PROFILE ---")
    print(f"Date Range: {audit['date_range'][0]} to {audit['date_range'][1]}")
    print(f"Total Hourly Timestamps: {audit['total_records']}")
    print(f"Extreme Pollution Hours (AQI > 300): {audit['severe_events_count']} ({audit['aqi_category_pct']['Very Poor (301-400)'] + audit['aqi_category_pct']['Severe (401-500)']:.1f}%)")

    # -----------------------------------------------------------------------
    # Step 2: Strict Leakage-Free Preprocessing
    # -----------------------------------------------------------------------
    logger.info("Step 2: Executing strict zero-leakage preprocessing...")
    data_dict = run_leakage_free_pipeline(
        df,
        lookback_window=72,
        horizon=24,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        scaler_type="robust",
    )

    preprocessor = data_dict["preprocessor"]
    splits = data_dict["splits"]

    # Tabular datasets for GBDTs
    X_train_tab = data_dict["X_train_tab"]
    y_train_tab = data_dict["y_train_tab"]
    X_val_tab = data_dict["X_val_tab"]
    y_val_tab = data_dict["y_val_tab"]
    X_test_tab = data_dict["X_test_tab"]
    y_test_tab = data_dict["y_test_tab"]

    # Sequential datasets for Deep Learning
    X_train_seq = data_dict["X_train_seq"]
    y_train_seq = data_dict["y_train_seq"]
    X_val_seq = data_dict["X_val_seq"]
    y_val_seq = data_dict["y_val_seq"]
    X_test_seq = data_dict["X_test_seq"]
    y_test_seq = data_dict["y_test_seq"]

    # Unscaled physical targets for final evaluation
    y_test_actual_scaled = y_test_tab.values
    y_test_actual = preprocessor.inverse_transform_target(y_test_actual_scaled)
    test_timestamps = y_test_tab.index

    # Store all model predictions
    predictions: dict[str, np.ndarray] = {}

    # -----------------------------------------------------------------------
    # Step 3: Model Tier 1 - Baselines
    # -----------------------------------------------------------------------
    logger.info("Step 3: Evaluating Tier 1 Reference Baselines...")
    # Persistence: predict current AQI as target at t+24
    # Note: X_test_tab features at row t contain AQI at t (lag 0 or unlagged)
    # Using clean test slice from splits
    pure_pers = PurePersistenceModel(horizon=24)
    # Target value at row t in test_df
    y_curr_test = splits.test_df.loc[y_test_tab.index, "AQI"].ffill().values
    predictions["Persistence (t)"] = y_curr_test

    weekly_pers = SeasonalWeeklyPersistenceModel(horizon=24)
    predictions["Seasonal Weekly"] = weekly_pers.predict(splits.test_df.loc[y_test_tab.index])

    # -----------------------------------------------------------------------
    # Step 4: Model Tier 2 - Machine Learning Trees
    # -----------------------------------------------------------------------
    logger.info("Step 4: Training Tier 2 Machine Learning Trees...")
    lgbm_model = LightGBMForecaster(n_estimators=300, learning_rate=0.03, max_depth=6)
    lgbm_model.fit(X_train_tab, y_train_tab, X_val_tab, y_val_tab)
    lgbm_preds_scaled = lgbm_model.predict(X_test_tab)
    predictions["LightGBM"] = preprocessor.inverse_transform_target(lgbm_preds_scaled)

    xgb_model = XGBoostForecaster(n_estimators=300, learning_rate=0.03, max_depth=6)
    xgb_model.fit(X_train_tab, y_train_tab, X_val_tab, y_val_tab)
    xgb_preds_scaled = xgb_model.predict(X_test_tab)
    predictions["XGBoost"] = preprocessor.inverse_transform_target(xgb_preds_scaled)

    # -----------------------------------------------------------------------
    # Step 5: Model Tier 3 - Sequential Deep Learning
    # -----------------------------------------------------------------------
    logger.info("Step 5: Training Tier 3 Sequential Deep Learning Models...")
    input_dim = X_train_seq.shape[2]

    # Model A: Recurrent Attention-LSTM
    logger.info("Training Attention-LSTM Forecaster...")
    lstm_net = RecurrentForecaster(input_dim=input_dim, hidden_dim=96, num_layers=2, dropout=0.2, cell_type="lstm")
    lstm_trainer = DeepLearningTrainer(lstm_net, learning_rate=0.002)
    lstm_trainer.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=12, batch_size=256, patience=4)
    lstm_preds_scaled = lstm_trainer.predict(X_test_seq, batch_size=256)
    lstm_preds = preprocessor.inverse_transform_target(lstm_preds_scaled)

    # Align sequence indices with test slice
    # Note: Sequential tensor starts at lookback-1. We match the last len(lstm_preds)
    seq_eval_len = min(len(lstm_preds), len(y_test_actual))
    y_test_actual_seq = y_test_actual[-seq_eval_len:]
    test_timestamps_seq = test_timestamps[-seq_eval_len:]
    predictions["Attention-LSTM"] = lstm_preds[-seq_eval_len:]

    # Model B: Temporal Transformer Encoder
    logger.info("Training Temporal Transformer Forecaster...")
    trans_net = TransformerForecaster(input_dim=input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2)
    trans_trainer = DeepLearningTrainer(trans_net, learning_rate=0.002)
    trans_trainer.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=12, batch_size=256, patience=4)
    trans_preds_scaled = trans_trainer.predict(X_test_seq, batch_size=256)
    predictions["Transformer"] = preprocessor.inverse_transform_target(trans_preds_scaled)[-seq_eval_len:]

    # -----------------------------------------------------------------------
    # Step 6: Comprehensive Multi-Metric & Sliced Evaluation
    # -----------------------------------------------------------------------
    logger.info("Step 6: Computing comprehensive metrics and extreme pollution slices...")
    # Harmonize lengths across models for unified evaluation comparison
    eval_n = seq_eval_len
    y_eval_true = y_test_actual[-eval_n:]
    eval_timestamps = test_timestamps[-eval_n:]

    all_model_slices = {}
    summary_records = []

    for model_name, preds in predictions.items():
        p_eval = preds[-eval_n:]
        slices = compute_stratified_slices(y_eval_true, p_eval, extreme_threshold=300.0, severe_threshold=400.0)
        all_model_slices[model_name] = slices

        overall = slices["Overall"]
        extreme = slices["Extreme (AQI > 300)"]
        summary_records.append(
            {
                "Model": model_name,
                "Overall_RMSE": overall["RMSE"],
                "Overall_MAE": overall["MAE"],
                "Overall_R2": overall["R2"],
                "Overall_Pearson_r": overall["Pearson_r"],
                "Overall_Willmott_d": overall["Willmott_d"],
                "Extreme_AQI>300_RMSE": extreme["RMSE"],
                "Extreme_AQI>300_MAE": extreme["MAE"],
                "Extreme_AQI>300_Pearson_r": extreme["Pearson_r"],
            }
        )

    summary_df = pd.DataFrame(summary_records).set_index("Model")
    print("\n" + "=" * 80)
    print("TABLE 1: 24-HOUR AHEAD AQI FORECAST BENCHMARK EVALUATION")
    print("=" * 80)
    print(summary_df.round(2).to_string())

    # Export LaTeX table
    latex_table = summary_df.round(2).to_latex(
        caption="Comparative 24-hour ahead AQI forecasting performance across overall test timeline and extreme pollution episodes ($AQI > 300$).",
        label="tab:main_benchmark",
    )
    with open("reports/benchmark_table.tex", "w") as f:
        f.write(latex_table)

    # -----------------------------------------------------------------------
    # Step 7: Atmospheric Error Auditing & Diagnostic Plotting
    # -----------------------------------------------------------------------
    logger.info("Step 7: Conducting error auditing (Phase Lag & Peak Underprediction)...")
    audit_records = []
    for model_name, preds in predictions.items():
        p_eval = preds[-eval_n:]
        auditor = AtmosphericErrorAuditor(eval_timestamps, y_eval_true, p_eval, model_name=model_name)
        phase_res = auditor.audit_phase_lag(max_lag_hours=30)
        peak_res = auditor.audit_peak_underprediction(extreme_threshold=300.0)

        audit_records.append(
            {
                "Model": model_name,
                "Optimal_Lag_(h)": phase_res["optimal_lag_hours"],
                "Zero_Lag_Corr": phase_res["zero_lag_correlation"],
                "Is_Lagging_Flaw": phase_res["is_lagging"],
                "True_Max_AQI": peak_res.get("true_max_aqi", np.nan),
                "Pred_Max_AQI": peak_res.get("pred_max_aqi", np.nan),
                "Peak_Ratio": peak_res.get("peak_ratio", np.nan),
                "Extreme_Mean_Underpred": peak_res.get("mean_underprediction_error", np.nan),
            }
        )

        # Plot dashboard for top deep model and top tree model
        if model_name in ["Attention-LSTM", "LightGBM"]:
            clean_name = model_name.replace(" ", "_").lower()
            auditor.plot_diagnostic_dashboard(figures_dir / f"error_dashboard_{clean_name}.png")

    audit_df = pd.DataFrame(audit_records).set_index("Model")
    print("\n" + "=" * 80)
    print("TABLE 2: ATMOSPHERIC ERROR AUDITING: PHASE LAG & PEAK ATTENUATION")
    print("=" * 80)
    print(audit_df.round(3).to_string())

    # -----------------------------------------------------------------------
    # Step 8: Graduate Depth Component - Uncertainty Quantification
    # -----------------------------------------------------------------------
    logger.info("Step 8: Executing Graduate Depth: Quantile Regression & Prediction Intervals...")
    mq_forecaster = MultiQuantileTreeForecaster(quantiles=[0.10, 0.50, 0.90], n_estimators=250)
    mq_forecaster.fit(X_train_tab, y_train_tab)
    q_preds_scaled = mq_forecaster.predict(X_test_tab)

    p10 = preprocessor.inverse_transform_target(q_preds_scaled[0.10])[-eval_n:]
    p50 = preprocessor.inverse_transform_target(q_preds_scaled[0.50])[-eval_n:]
    p90 = preprocessor.inverse_transform_target(q_preds_scaled[0.90])[-eval_n:]

    uq_metrics = evaluate_prediction_intervals(y_eval_true, p10, p90, p50)
    print("\n" + "=" * 80)
    print("TABLE 3: UNCERTAINTY QUANTIFICATION (80% NOMINAL PREDICTION INTERVAL)")
    print("=" * 80)
    for k, v in uq_metrics.items():
        print(f"  {k:28s}: {v:8.2f}")

    # Plot Prediction Intervals
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_hours = min(250, eval_n)
    ts_slice = eval_timestamps[-plot_hours:]
    ax.fill_between(ts_slice, p10[-plot_hours:], p90[-plot_hours:], color="skyblue", alpha=0.45, label="80% Prediction Interval [P10, P90]")
    ax.plot(ts_slice, p50[-plot_hours:], color="navy", lw=2.0, label="P50 Median Forecast")
    ax.plot(ts_slice, y_eval_true[-plot_hours:], color="black", lw=2.2, label="Actual Ground Truth AQI (t+24)")
    ax.axhline(300, color="orange", ls="--", label="CPCB Very Poor (300)")
    ax.set_title("Graduate Depth Component: Calibrated 24-Hour Ahead Prediction Intervals under Extreme Pollution", fontsize=12, fontweight="bold")
    ax.set_ylabel("AQI")
    ax.set_xlabel("Forecast Target Timestamp")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "quantile_prediction_intervals.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Step 9: Graduate Depth Component - Stagnation & Meteorological Failure Audit
    # -----------------------------------------------------------------------
    logger.info("Step 9: Auditing model failure mechanisms across micro-meteorological regimes...")
    # Extract test meteorological variables
    test_met = splits.test_df.loc[eval_timestamps]
    ws_test = test_met["Wind_Speed"].values
    temp_test = test_met["Temperature"].values
    rh_test = test_met["Relative_Humidity"].values

    top_model_preds = predictions["Attention-LSTM"]
    stagnation_df = audit_stagnation_subgroups(
        y_eval_true, top_model_preds, ws_test, temp_test, rh_test
    )
    print("\n" + "=" * 80)
    print("TABLE 4: METEOROLOGICAL STAGNATION FAILURE AUDIT (Attention-LSTM)")
    print("=" * 80)
    print(stagnation_df.round(2).to_string())

    print("\n" + "=" * 80)
    print("Pipeline completed successfully! All artifacts, tables, and figures saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
