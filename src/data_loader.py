"""
data_loader.py
==============
Data ingestion, schema validation, realistic benchmark generation,
and exploratory time-series diagnostics for hourly air quality data.

Compatible with CPCB (Central Pollution Control Board, India) & OpenAQ formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPCB Official AQI Sub-Index Calculation Breakpoints
# ---------------------------------------------------------------------------
# Reference: Central Pollution Control Board (CPCB), India (2014)
# "National Air Quality Index" Report, Ministry of Environment, Forest & Climate Change.
AQI_BREAKPOINTS = {
    "PM2.5": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "PM10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    "NO2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 600.0, 401, 500),
    ],
    "SO2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2500.0, 401, 500),
    ],
    "CO": [
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "O3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ],
}


def calculate_sub_index(conc: float, pollutant: str) -> float:
    """Calculate sub-index for a single pollutant using linear interpolation across breakpoints."""
    if np.isnan(conc) or conc < 0:
        return np.nan
    breakpoints = AQI_BREAKPOINTS.get(pollutant)
    if not breakpoints:
        return np.nan

    for b_lo, b_hi, i_lo, i_hi in breakpoints:
        if b_lo <= conc <= b_hi:
            return i_lo + ((i_hi - i_lo) / (b_hi - b_lo)) * (conc - b_lo)

    # If concentration exceeds highest breakpoint, extrapolate capped at 500
    if conc > breakpoints[-1][1]:
        return 500.0
    return 0.0


def compute_overall_aqi(row: pd.Series) -> float:
    """
    Compute CPCB overall AQI as the maximum of sub-indices.
    CPCB requires at least 3 pollutants with at least one particulate (PM2.5 or PM10).
    """
    sub_indices = []
    has_pm = False
    for poll in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
        if poll in row and not pd.isna(row[poll]):
            si = calculate_sub_index(row[poll], poll)
            if not np.isnan(si):
                sub_indices.append(si)
                if poll in ["PM2.5", "PM10"]:
                    has_pm = True

    if len(sub_indices) >= 3 and has_pm:
        return float(np.max(sub_indices))
    elif len(sub_indices) > 0:
        return float(np.max(sub_indices))  # Best available fallback
    return np.nan


# ---------------------------------------------------------------------------
# Physics-Informed Realistic Benchmark Dataset Generator
# ---------------------------------------------------------------------------
def generate_synthetic_cpcb_data(
    start_date: str = "2022-01-01 00:00:00",
    end_date: str = "2023-12-31 23:00:00",
    station_id: str = "DEL_AV_001",
    seed: int = 42,
    missing_rate: float = 0.035,
) -> pd.DataFrame:
    """
    Generates a realistic, multi-pollutant and meteorological hourly dataset
    reflecting Delhi's boundary-layer dynamics, winter temperature inversions,
    monsoon scavenging, and diurnal traffic patterns.

    Parameters
    ----------
    start_date : str
        Start timestamp.
    end_date : str
        End timestamp.
    station_id : str
        Station identifier.
    seed : int
        RNG seed for strict reproducibility.
    missing_rate : float
        Proportion of synthetic sensor outages/missing values.

    Returns
    -------
    pd.DataFrame
        Hourly indexed DataFrame containing pollutants, meteorology, and computed AQI.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="1h")
    n = len(timestamps)

    hours = timestamps.hour.values
    days_of_year = timestamps.dayofyear.values
    months = timestamps.month.values

    # 1. Meteorology Simulation
    # Annual temperature cycle (hottest in May/June ~40C, coldest in Jan ~8C)
    t_annual = 25.0 + 12.0 * np.sin(2 * np.pi * (days_of_year - 105) / 365.25)
    t_diurnal = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
    temperature = t_annual + t_diurnal + rng.normal(0, 1.5, n)

    # Relative Humidity: inverse to temperature, high during monsoon (July-Aug) and winter fog (Dec-Jan)
    rh_base = 60.0 - 15.0 * np.sin(2 * np.pi * (days_of_year - 105) / 365.25)
    rh_monsoon = 25.0 * np.exp(-0.5 * ((days_of_year - 215) / 30.0) ** 2)
    rh_winter = 20.0 * np.exp(-0.5 * ((days_of_year - 15) / 25.0) ** 2) + 20.0 * np.exp(-0.5 * ((days_of_year - 355) / 25.0) ** 2)
    rh_diurnal = -12.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
    relative_humidity = np.clip(rh_base + rh_monsoon + rh_winter + rh_diurnal + rng.normal(0, 3.0, n), 15.0, 99.0)

    # Wind speed: calmer in winter (stagnant inversion < 1.5 m/s), breezier in pre-monsoon
    ws_base = 2.5 + 1.2 * np.sin(2 * np.pi * (days_of_year - 120) / 365.25)
    # Winter stagnation drop
    winter_mask = (months >= 11) | (months <= 1)
    ws_base[winter_mask] *= 0.55
    ws_diurnal = 0.8 * np.sin(2 * np.pi * (hours - 11) / 24.0)
    wind_speed = np.clip(ws_base + ws_diurnal + rng.lognormal(0, 0.35, n) - 0.9, 0.2, 12.0)

    # Wind direction: prevailing NW in winter, SE during monsoon
    wind_direction = np.zeros(n)
    for i in range(n):
        if months[i] in [10, 11, 12, 1, 2]:
            wind_direction[i] = (305 + rng.normal(0, 35)) % 360
        elif months[i] in [6, 7, 8, 9]:
            wind_direction[i] = (115 + rng.normal(0, 45)) % 360
        else:
            wind_direction[i] = (260 + rng.normal(0, 50)) % 360

    # Atmospheric Pressure (hPa): higher in winter (~1018), lower in summer (~995)
    pressure = 1008.0 + 8.0 * np.cos(2 * np.pi * (days_of_year - 10) / 365.25) + rng.normal(0, 1.2, n)

    # Ventilation Index proxy: Wind_Speed * Boundary_Layer_Height
    # In winter, shallow boundary layer (<400m) + low wind = severe trap
    ventilation_factor = wind_speed * np.maximum(0.2, (temperature + 10.0) / 40.0)

    # 2. Pollutant Dynamics
    # PM2.5 base: massive winter peak due to stubble burning, low ventilation, domestic heating
    pm25_seasonal = 70.0 + 130.0 * np.exp(-0.5 * ((days_of_year - 325) / 40.0) ** 2) + 110.0 * np.exp(-0.5 * ((days_of_year - 15) / 35.0) ** 2)
    # Monsoon scavenging: July-August washout
    monsoon_scavenge = 1.0 - 0.65 * np.exp(-0.5 * ((days_of_year - 215) / 30.0) ** 2)
    pm25_seasonal *= monsoon_scavenge

    # Traffic diurnal signature (dual peak: 8-10 AM, 7-10 PM)
    traffic_peak1 = 35.0 * np.exp(-0.5 * ((hours - 9) / 1.8) ** 2)
    traffic_peak2 = 55.0 * np.exp(-0.5 * ((hours - 21) / 2.2) ** 2)
    pm25_diurnal = traffic_peak1 + traffic_peak2

    # Episode shocks: Diwali / Post-harvest stubble burning burst (early Nov)
    stubble_burst = 180.0 * np.exp(-0.5 * ((days_of_year - 310) / 5.0) ** 2)
    diwali_burst = 220.0 * np.exp(-0.5 * ((days_of_year - 318) / 2.0) ** 2)

    # Autoregressive AR(1) turbulence for temporal persistence
    ar_noise = np.zeros(n)
    for i in range(1, n):
        ar_noise[i] = 0.82 * ar_noise[i - 1] + rng.normal(0, 12.0)

    pm25 = (pm25_seasonal + pm25_diurnal + stubble_burst + diwali_burst + ar_noise) / (ventilation_factor ** 0.45)
    pm25 = np.clip(pm25, 5.0, 580.0)

    # PM10: correlated with PM2.5 + summer dust storms (May-June)
    summer_dust = 140.0 * np.exp(-0.5 * ((days_of_year - 145) / 20.0) ** 2)
    pm10 = np.clip(pm25 * 1.65 + summer_dust + rng.normal(0, 18.0, n), 12.0, 750.0)

    # NO2: dominated by vehicular emissions and power generation
    no2 = np.clip(25.0 + 35.0 * (traffic_peak1 + traffic_peak2) / 60.0 + 20.0 * winter_mask + rng.normal(0, 6.0, n), 5.0, 280.0)

    # SO2: industrial background + sporadic plumes
    so2 = np.clip(12.0 + rng.exponential(7.0, n) + 8.0 * winter_mask, 2.0, 95.0)

    # CO: closely tracks vehicular PM2.5 and low winter dispersion
    co = np.clip(0.4 + 0.007 * pm25 + 0.3 * (traffic_peak1 + traffic_peak2) / 60.0 + rng.normal(0, 0.1, n), 0.1, 8.5)

    # O3: Photochemical formation driven by solar radiation (peaks around 14:00, depressed in winter fog)
    o3_diurnal = 60.0 * np.maximum(0.0, np.sin(np.pi * (hours - 7) / 12.0))
    o3 = np.clip(15.0 + o3_diurnal * (1.0 - 0.4 * winter_mask) + rng.normal(0, 5.0, n), 2.0, 240.0)

    df = pd.DataFrame(
        {
            "Station_ID": station_id,
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "Temperature": temperature,
            "Relative_Humidity": relative_humidity,
            "Wind_Speed": wind_speed,
            "Wind_Direction": wind_direction,
            "Pressure": pressure,
        },
        index=timestamps
    )
    df.index.name = "Timestamp"

    # Compute official continuous AQI target
    df["AQI"] = df.apply(compute_overall_aqi, axis=1)

    # Inject realistic missing values (random dropouts + contiguous block outages)
    if missing_rate > 0:
        mask_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Relative_Humidity", "Wind_Speed"]
        for col in mask_cols:
            col_idx = df.columns.get_loc(col)
            # 1. Sporadic random dropouts
            random_drop = rng.random(n) < (missing_rate * 0.4)
            df.iloc[random_drop, col_idx] = np.nan

            # 2. Contiguous sensor outage blocks (e.g. 3-8 hours)
            num_blocks = int(n * missing_rate * 0.6 / 5.0)
            block_starts = rng.integers(0, n - 8, size=num_blocks)
            for bs in block_starts:
                blen = rng.integers(2, 7)
                df.iloc[bs : bs + blen, col_idx] = np.nan

        # Re-evaluate AQI where values went missing
        df["AQI"] = df.apply(compute_overall_aqi, axis=1)

    logger.info("Generated %d synthetic hourly records for station %s [%s to %s].", len(df), station_id, start_date, end_date)
    return df


# ---------------------------------------------------------------------------
# Data Ingestion & Schema Diagnostics Pipeline
# ---------------------------------------------------------------------------
def load_and_validate_data(
    filepath_or_synthetic: Optional[Union[str, Path]] = None,
    synthetic_backup: bool = True,
) -> pd.DataFrame:
    """
    Load station hourly dataset from CSV, validate schema and timestamps,
    or fall back to generating the benchmark dataset.

    Parameters
    ----------
    filepath_or_synthetic : str or Path, optional
        Path to CSV file. If None or not found, generates benchmark data.
    synthetic_backup : bool
        Whether to generate benchmark data if filepath does not exist.

    Returns
    -------
    pd.DataFrame
        Cleaned, hourly indexed DataFrame.
    """
    if filepath_or_synthetic is not None and Path(filepath_or_synthetic).exists():
        logger.info("Loading empirical data from: %s", filepath_or_synthetic)
        df = pd.read_csv(filepath_or_synthetic)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        elif "date" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["date"])
        elif "Datetime" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Datetime"])
        else:
            raise ValueError("Dataset must contain 'Timestamp', 'Datetime', or 'date' column.")
    elif synthetic_backup:
        logger.info("No existing dataset found at '%s'. Generating realistic benchmark data...", filepath_or_synthetic)
        df = generate_synthetic_cpcb_data()
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "station_hourly.csv", index=True)
        logger.info("Saved raw benchmark data to data/raw/station_hourly.csv")
    else:
        raise FileNotFoundError(f"File not found: {filepath_or_synthetic}")

    # Set DatetimeIndex and sort chronologically
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Timestamp" in df.columns:
            df = df.set_index("Timestamp")
        elif "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"]))
        elif "Datetime" in df.columns:
            df = df.set_index(pd.to_datetime(df["Datetime"]))
    df = df.sort_index()

    # Enforce strict 1-hour frequency grid
    df = df.asfreq("1h")

    # Validate essential columns
    expected_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Relative_Humidity", "Wind_Speed", "AQI"]
    for c in expected_cols:
        if c not in df.columns:
            logger.warning("Expected column '%s' missing from dataset.", c)

    return df


def audit_dataset(df: pd.DataFrame) -> Dict[str, Union[int, float, pd.DataFrame]]:
    """
    Computes rigorous missingness audit, descriptive statistics,
    and extreme event prevalence for the environmental dataset.
    """
    total_hours = len(df)
    missing_summary = df.isna().sum()
    missing_pct = (missing_summary / total_hours) * 100.0

    stats = df.describe().T
    stats["missing_count"] = missing_summary
    stats["missing_pct"] = missing_pct

    # Extreme event distribution (CPCB AQI categories)
    aqi_valid = df["AQI"].dropna()
    categories = {
        "Good (0-50)": (aqi_valid <= 50).sum(),
        "Satisfactory (51-100)": ((aqi_valid > 50) & (aqi_valid <= 100)).sum(),
        "Moderate (101-200)": ((aqi_valid > 100) & (aqi_valid <= 200)).sum(),
        "Poor (201-300)": ((aqi_valid > 200) & (aqi_valid <= 300)).sum(),
        "Very Poor (301-400)": ((aqi_valid > 300) & (aqi_valid <= 400)).sum(),
        "Severe (401-500)": (aqi_valid > 400).sum(),
    }
    cat_pct = {k: (v / len(aqi_valid)) * 100.0 for k, v in categories.items()}

    audit_res = {
        "total_records": total_hours,
        "date_range": (str(df.index.min()), str(df.index.max())),
        "missing_statistics": stats[["missing_count", "missing_pct", "mean", "std", "min", "50%", "max"]],
        "aqi_category_counts": categories,
        "aqi_category_pct": cat_pct,
        "severe_events_count": int(categories["Very Poor (301-400)"] + categories["Severe (401-500)"]),
    }
    return audit_res


if __name__ == "__main__":
    df = load_and_validate_data()
    audit = audit_dataset(df)
    print("=== Dataset Temporal Span ===")
    print(f"Start: {audit['date_range'][0]}, End: {audit['date_range'][1]}, Total Hours: {audit['total_records']}")
    print("\n=== Missing Data & Summary Statistics ===")
    print(audit["missing_statistics"])
    print("\n=== CPCB Category Breakdown ===")
    for k, v in audit["aqi_category_pct"].items():
        print(f"  {k:25s}: {v:6.2f}% ({audit['aqi_category_counts'][k]} hours)")
    print(f"\nTotal Extreme Pollution Hours (AQI > 300): {audit['severe_events_count']}")
