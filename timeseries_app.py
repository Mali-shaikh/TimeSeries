"""
Streamlit app that:
- Lets user choose ticker from AMZN, AAPL, GOOGL, TSLA
- Runs EDA, 7-day MA, decomposition, ADF on Volume
- Fits ARIMA(1,0,0) on Close and Volume (backtest + forecast)
- Displays results interactively and allows downloading CSVs/plots
"""

import csv
import warnings

from matplotlib import ticker
warnings.filterwarnings("ignore")

import datetime as dt
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Time Series Analysis & Forecasting", layout="wide")

# Sidebar controls
st.sidebar.header("Settings")
TICKER = st.sidebar.selectbox("Select ticker", options=["AMZN","AAPL", "GOOGL", "TSLA"], index=0)
YEARS = st.sidebar.slider("Years of history", min_value=1, max_value=10, value=3)
HOLDOUT_DAYS = st.sidebar.slider("Holdout (days) for backtest", min_value=10, max_value=90, value=30)
FUTURE_DAYS = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30)

st.title("Time Series Analysis & Forecasting (ARIMA on Close)")

@st.cache_data
def load_data(ticker: str, years: int):
    end = dt.date.today()
    start = dt.date(end.year - years, end.month, end.day)
    data = yf.download(ticker, start=start, end=end)
    data.index = pd.to_datetime(data.index)
    return data

def adf_test(series):
    res = adfuller(series.dropna(), autolag="AIC")
    return {"adf_stat": res[0], "p_value": res[1], "crit_vals": res[4]}

def plot_line(series, title=""):
    # If series is a DataFrame, convert it to a Series using first column
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series)

    # Handle ylabel safely
    ylabel = series.name if hasattr(series, "name") else "Value"
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.set_xlabel("Date")
    st.pyplot(fig)
    plt.close(fig)

def plot_hist(series, title=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series.dropna(), bins=50)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def decompose_and_plot(series, period=None):
    if period is None:
        period = 252 if len(series) > 252 else max(2, int(len(series) / 4))
    decomposition = seasonal_decompose(series.dropna(), model="additive", period=period, extrapolate_trend="freq")
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    st.pyplot(fig)
    plt.close(fig)
    return decomposition

def train_test_split(series, holdout_days):
    s = series.dropna()
    if len(s) <= holdout_days + 5:
        st.warning("Not enough data for the chosen holdout size.")
        return None, None
    return s.iloc[:-holdout_days], s.iloc[-holdout_days:]

def fit_arima_and_eval(train, test, order=(1,0,0), future_steps=30):
    model = ARIMA(train, order=order)
    fitted = model.fit()
    pred_test = fitted.forecast(steps=len(test))
    mae = mean_absolute_error(test, pred_test)
    # Refit on full series
    full_model = ARIMA(pd.concat([train, test]), order=order)
    full_fitted = full_model.fit()
    future_forecast = full_fitted.forecast(steps=future_steps)
    return fitted, pred_test, mae, future_forecast

# Load data
with st.spinner("Loading data..."):
    df = load_data(TICKER, YEARS)

if df.empty:
    st.error("No data returned. Check ticker or your internet connection.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.tail(10))

# EDA
st.subheader("Exploratory Data Analysis")
st.write("Summary statistics:")
st.write(df.describe())

st.write("Close price over time:")
plot_line(df["Close"], title=f"{TICKER} Close Price")

cols = st.columns(3)
numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
for i, col in enumerate(numeric_cols):
    with cols[i % 3]:
        st.write(f"Histogram: {col}")
        plot_hist(df[col], title=f"{TICKER} - {col}")

# Moving average
st.subheader("Moving Average")
ma7 = df["Close"].rolling(window=7).mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["Close"], label="Close")
ax.plot(df.index, ma7, label="7-day MA")
ax.set_title(f"{TICKER} Close and 7-day MA")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# Decomposition
st.subheader("Time Series Decomposition (Close)")
decomposition = decompose_and_plot(df["Close"])

# ADF on Close
st.subheader("Stationarity Check - ADF test (Close)")
if "Close" in df.columns:
    adf_res = adf_test(df["Close"])
    st.write(f"ADF Statistic: {adf_res['adf_stat']:.4f}")
    st.write(f"p-value: {adf_res['p_value']:.4f}")
    st.write("Critical values:")
    st.write(adf_res["crit_vals"])
    if adf_res["p_value"] < 0.05:
        st.success("Close appears stationary (reject null at 5%).")
    else:
        st.info("Close appears non-stationary (cannot reject null at 5%).")
else:
    st.warning("Close column not found; skipping ADF test.")

# ARIMA modeling on Close only
st.subheader("ARIMA(1,0,0) Modeling and Forecasting (Close)")
close_series = df["Close"].dropna()
train, test = train_test_split(close_series, HOLDOUT_DAYS)

if train is None:
    st.warning("Insufficient data to fit ARIMA with current holdout. Reduce holdout or increase years.")
else:
    with st.spinner("Fitting ARIMA(1,0,0) ..."):
        fitted, pred_test, mae, future_forecast = fit_arima_and_eval(train, test, order=(1, 0, 0), future_steps=FUTURE_DAYS)

    st.write("ARIMA(1,0,0) summary (training fit):")
    st.text(fitted.summary().as_text())

    st.write(f"MAE on holdout ({HOLDOUT_DAYS} days): {mae:.4f}")

    # Plot observed, predicted test and future forecast
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close_series.index, close_series, label="Observed")
    ax.axvline(x=test.index[0], color="gray", linestyle="--", label="Train/Test split")
    ax.plot(test.index, pred_test, label="Predicted (test)", linestyle="--")
    last_date = close_series.index[-1]
    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_forecast), freq="B")
    ax.plot(future_index, future_forecast, label=f"Forecast next {len(future_forecast)} (business days)", linestyle=":")
    ax.set_title(f"{TICKER} Observed vs Predicted (Close)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

summary = {
    "ticker": TICKER,
    "data_start": df.index.min().strftime("%Y-%m-%d"),
    "data_end": df.index.max().strftime("%Y-%m-%d"),
    "close_holdout_mae": (mae if "mae" in locals() else None),
    "close_adf_pvalue": (adf_res["p_value"] if "adf_res" in locals() else None),
}
summary_df = pd.DataFrame([summary])
csv = summary_df.to_csv(index=False).encode("utf-8")

st.download_button("Download summary CSV", csv, file_name=f"{TICKER}_summary.csv", mime="text/csv")

st.success("Analysis complete. You can re-run with different settings from the sidebar.")


st.markdown("---")
st.write("Notes:")
st.write("- ARIMA is fitted on Close only (as required).")
st.write("- Close is used for stationarity check (ADF).")
st.write("- Forecasts use business-day frequency for plotting; actual trading calendar can differ.")