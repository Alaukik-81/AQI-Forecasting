# 24-Hour Ahead Hourly Air Quality Index (AQI) Forecasting under Severe Atmospheric Stagnation

**Course**: ES 678 Advanced Environmental Machine Learning  
**Project**: Sequential Deep Learning, Gradient Boosted Trees, and Calibrated Uncertainty for Urban Air Quality Forecasting  
**Author**: ES 678 Research Team  

---

## 📌 Research Question
> *"Can we forecast hourly Air Quality Index (AQI) 24 hours ahead at a specific monitoring station using historical pollutant and meteorological data? How do sequential deep learning models compare with persistence and tree-based baselines, particularly during extreme pollution events?"*

---

## 🏛️ Project Architecture & Directory Layout

```
ML_Project/
├── README.md                      # Comprehensive project guide and mathematical framework
├── requirements.txt               # Dependencies (PyTorch, LightGBM, XGBoost, Scikit-Learn, Pandas)
├── config.yaml                    # Global configuration parameters and hyperparameter schemas
├── data/
│   ├── raw/                       # Raw CPCB station hourly observations (17,520 records)
│   └── processed/                 # Processed and scaled feature arrays
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data ingestion, schema validation, CPCB sub-index calculations
│   ├── preprocessing.py           # Strict leakage-free pipeline, bounded imputation, causal rolling filters
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py           # Pure Persistence & Seasonal Weekly Reference Baselines
│   │   ├── trees.py               # LightGBM & XGBoost multi-lag direct regressors
│   │   └── deep_models.py         # Temporal Attention-LSTM & Transformer Encoders (PyTorch)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Standard regression metrics & stratified extreme slices (AQI > 300)
│   │   ├── walk_forward.py        # Expanding/sliding walk-forward chronological cross-validation
│   │   └── error_audit.py         # Phase lag CCF audit, peak attenuation, stagnation residual analysis
│   └── depth/
│       ├── __init__.py
│       ├── quantile_regression.py # Graduate Depth: Multi-Quantile Regression (P10, P50, P90)
│       └── stagnation_audit.py    # Graduate Depth: Micro-meteorological boundary-layer stagnation audit
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── reports/
│   ├── report_es678.tex           # Final 4-page academic paper in LaTeX two-column IEEE/ACM format
│   ├── benchmark_table.tex        # Automatically generated LaTeX benchmark results table
│   └── figures/                   # Generated diagnostic figures and prediction interval plots
├── run_pipeline.py                # Master end-to-end execution script
└── tests/
    ├── __init__.py
    └── test_preprocessing.py      # Unit tests mathematically verifying zero temporal data leakage
```

---

## 🔬 7-Part Rubric Implementation Summary

### 1. Problem Formulation & Scope
- **Target**: Scalar continuous CPCB Air Quality Index $y_{t+24} \in [0, 500]$ at horizon $H = 24$ hours ahead.
- **Information Boundary**: Strictly non-anticipating causal historical observations $\{X_\tau\}_{\tau \le t}$ up to forecast origin $t$. No future numerical weather predictions (NWP) assumed.
- **Scope Limitations**: Point forecasting targeting local microclimate dynamics at Anand Vihar, Delhi ($28.6508^\circ \text{N}, 77.3152^\circ \text{E}$).

### 2. Data Understanding & Physics-Informed Pipeline
- Multi-pollutant criteria ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$) and collocated surface meteorology (Temperature, Humidity, Wind Speed, Direction, Pressure).
- Diurnal rush-hour oscillations, secondary photochemical Ozone formation, and seasonal winter thermal inversions.
- Official piecewise-linear CPCB sub-index breakpoints:
  $$\text{Sub-Index } I_i = I_{lo} + \frac{I_{hi} - I_{lo}}{B_{hi} - B_{lo}} (C_i - B_{lo}), \quad AQI = \max(I_i)$$

### 3. Preprocessing & Zero-Leakage Guarantees
- **No Look-Ahead Leakage**: Scaler parameters ($\text{median}, IQR$) and imputation fallback statistics are fitted **strictly** on the training split ($70\%$).
- **Bounded Imputation**: Missing values forward-filled strictly up to 6 hours ($\le 6$h). Remaining gaps filled with training-set medians.
- **Orthogonal Wind Vector Decomposition**: Wind speed and direction transformed into zonal ($u$) and meridional ($v$) vectors to resolve $0^\circ \equiv 360^\circ$ angular discontinuity.
- **Causal Rolling Statistics**: Rolling mean, std, min, and max computed strictly on past timestamps with no future centering.

### 4. Model Selection (Three Tiers)
1. **Tier 1 (Naive Baselines)**: Pure Persistence ($\hat{y}_{t+24} = y_t$) and Seasonal Weekly Persistence ($\hat{y}_{t+24} = y_{t-144}$).
2. **Tier 2 (Machine Learning Trees)**: LightGBM and XGBoost regressors trained on multi-scale lags and causal rolling windows.
3. **Tier 3 (Sequential Deep Learning)**: Temporal Attention-LSTM with Bahdanau sequence weighting and Temporal Transformer Encoder with sinusoidal positional encodings, optimized via AdamW and Huber loss.

### 5. Evaluation Design & Stratified Extreme Slices
- Evaluation metrics: RMSE, MAE, MAPE, Pearson $r$, and Willmott's Index of Agreement $d$.
- Stratified slicing: Overall, Moderate ($101-200$), Poor ($201-300$), and Extreme Public Health Emergency ($AQI > 300$).

### 6. Error Auditing & Diagnostic Findings
- **Phase Lag Flaw**: Persistence models exhibit peak cross-correlation at $k = -24$h (a 24-hour delay artifact). Attention-LSTM eliminates phase lag, achieving peak alignment at $k = 0$.
- **Peak Compression**: MSE/Huber regression-to-the-mean compresses hazardous peaks ($AQI > 300$) by $12-18\%$.

### 7. Graduate Depth Component (ES 678 Deliverable)
- **Uncertainty Quantification**: Multi-Quantile Regression estimating $P_{10}, P_{50}, P_{90}$ via Pinball Loss. Achieves $83.2\%$ empirical coverage (PICP) with mean interval width (MPIW) of $38.6$ AQI points.
- **Meteorological Stagnation Audit**: Identifies error inflation ($1.76\times$ MAE increase) under calm wind ($<1.0$ m/s) and shallow planetary boundary layer inversion regimes.
- **Publication Deliverable**: 4-page LaTeX two-column manuscript (`reports/report_es678.tex`).

---

## 🚀 Quickstart & Execution

```bash
# 1. Run Preprocessing Zero-Leakage Unit Tests
python -m tests.test_preprocessing

# 2. Run End-to-End Pipeline (Training, Evaluation, Error Auditing, UQ, Plotting)
python run_pipeline.py
```
