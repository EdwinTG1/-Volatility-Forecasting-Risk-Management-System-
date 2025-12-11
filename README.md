# -Volatility-Forecasting-Risk-Management-System-
Compares GARCH and XGBoost volatility forecasts in a dynamic volatility targeting strategy using Apple stock. Implements daily rebalancing with rolling window estimation and evaluates performance through Value-at-Risk (VaR) calibration, forecast error (MAPE), and drawdown analysis

Volatility Forecasting & Risk Management System

Methodology
1. Data & Preprocessing
Dataset: Apple Inc. (AAPL) daily OHLCV data from Jan 2015 to Jan 2025.
Regime Filtering: Focused on the modern tech era to ensure relevance.
Splitting: Dynamic 75% training / 25% testing split (out-of-sample test on last ~634 days).

2. Feature Engineering To allow the machine learning model to outperform standard statistics, we engineered institutional-grade volatility proxies:
Parkinson Volatility: Uses High/Low prices to capture intraday trading ranges.
Garman-Klass Volatility: Incorporates Open prices to capture overnight gap risk.
Bias Correction: Strict time-series lagging (t-1) applied to all features to eliminate Look-Ahead Bias.

3. Modeling Approach
Benchmark: GARCH(1,1) with Generalised Error Distribution (GED).
XGBoost Regressor optimised via RandomizedSearchCV.
Ensemble: A weighted average model combining GARCH stability with XGBoost responsiveness.

4. Risk Engine
Calculates 95% Value at Risk (VaR) using Normal Distribution assumptions.
Calibrates models against the Basel III  system (target < 5% breach rate).
Computes Expected Shortfall (ES) to measure tail risk severity.

Key Features
Volatility Proxies: Replaces standard Close-to-Close volatility with Parkinson and Garman-Klass estimators.
Look-Ahead Bias Elimination: Rigorous feature lagging ensures realistic out-of-sample performance.
Hyperparameter Tuning: Automated RandomizedSearchCV for XGBoost optimization.
Risk Metrics: Comprehensive calculation of VaR, ES, and Traffic Light breach analysis.
Ensemble Forecasting: Combines statistical and ML approaches for superior error reduction.

Results
1. Forecast Accuracy
XGBoost significantly outperformed the traditional benchmark in daily volatility tracking accuracy.
GARCH(1,1): 16.17% Error
XGBoost: 8.27% Error
Improvement: 49% reduction in forecast error

2. Risk Calibration (VaR Breach Rate) Both models achieved  safety levels, landing in the regulatory zone (target: 5.00%).
Target: 32 Breaches (5.00%)
Result: Both models recorded 26 Breaches (4.10%)
Verdict: Near-perfect calibration. The ML model matched the safety of GARCH while providing daily accuracy.

3. Model Comparison (RMSE)
GARCH RMSE: 0.2287
XGBoost RMSE: 0.2086
Ensemble RMSE: 0.1829 (Best Performer)

4. Trading Strategy
A dynamic Volatility Targeting strategy was backtested against Buy & Hold.
Buy & Hold Return: 26.12%
Vol-Target Return: 19.64%
Analysis: While the strategy yielded lower absolute returns in this specific bull market period, it maintained a comparable risk profile with dynamic exposure management (Avg Weight: 0.92).

Requirements
Python 3.11+
pandas
numpy
matplotlib
seaborn
xgboost
arch (arch-python)
scikit-learn
scipy
plotly
tabulate


Usage
1. Run the Notebook Execute volatility_risk.ipynb. The notebook is self-contained and proceeds through the following steps:
Data Loading: Loads AAPL data and filters for the 2015-2025 regime.
GARCH Modelling: Fits statistical benchmarks and runs rolling forecasts.
XGBoost Training: Generates intraday features, lags data, tunes hyperparameters, and predicts.
Evaluation: Calculates RMSE, MAPE, and visualizes forecasts.
Risk Engine: Computes VaR/ES and generates breach plots.
Strategy: Backtests the Volatility Targeting logic.

2. Data Source
Ensure AAPL.csv (or AAPL1.csv) is placed in the correct directory path specified in the notebook.

Author
Edwin Donkor

Role: Quantitative Researcher / Data Scientist
Focus: Financial Machine Learning & Risk Management
