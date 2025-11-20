"""
timeseries_analysis_forecasting.py

- Option A implementation: ARIMA(1,0,0) trained on Close only and forecasts next 30 business days of Close.
- Volume is used only for ADF stationarity test (due to forecasting on close).
- Designed to be run as a regular Python script (prints outputs and saves plots to `ts_outputs/`).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# ---------------------------
# USER PARAMETERS (change as needed)
# ---------------------------
TICKERS = ["AAPL","AMZN", "GOOGL", "TSLA"]  # will loop through these if you want, default set
SELECTED_TICKER = "GOOGL"             # change to "GOOGL" or "TSLA" if you like
YEARS = 3
HOLDOUT_DAYS = 30
FUTURE_DAYS = 30
OUTPUT_DIR = "ts_outputs"
# ---------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_data(ticker: str, years: int) -> pd.DataFrame:
    end = dt.date.today()
    start = dt.date(end.year - years, end.month, end.day)
    print(f"Downloading {ticker} data from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError("No data downloaded. Check ticker or internet connection.")
    df.index = pd.to_datetime(df.index)
    return df


def eda(df: pd.DataFrame, ticker: str):
    print("\n--- Exploratory Data Analysis ---")
    print("Data types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())

    # Line plot of Close
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], lw=1)
    plt.title(f"{ticker} Close Price")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_close_line.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # Histograms
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins=50)
        plt.title(f"{ticker} - Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"{ticker}_hist_{col.replace(' ', '_')}.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved: {out}")


def moving_average_plot(df: pd.DataFrame, ticker: str, window: int = 7):
    if "Close" not in df.columns:
        raise ValueError("Close column not found.")
    ma = df["Close"].rolling(window=window).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label="Close")
    plt.plot(df.index, ma, label=f"{window}-day MA")
    plt.title(f"{ticker} Close and {window}-day MA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_close_ma_{window}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def decompose_series(df: pd.DataFrame, ticker: str, column: str = "Close", period: int = None):
    series = df[column].dropna()
    if period is None:
        period = 252 if len(series) > 252 else max(2, int(len(series) / 4))
    print(f"Decomposing {column} with period = {period}")
    decomposition = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    out = os.path.join(OUTPUT_DIR, f"{ticker}_decompose_{column}.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")
    return decomposition


def adf_test(series: pd.Series, name: str = "series"):
    print(f"\nADF test on {name}")
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_value = result[0], result[1]
    crit_vals = result[4]
    print(f"ADF Statistic: {adf_stat:.4f}; p-value: {p_value:.4f}")
    for k, v in crit_vals.items():
        print(f"Critical Value ({k}): {v:.4f}")
    if p_value < 0.05:
        print("-> Stationary (reject unit root at 5%)")
    else:
        print("-> Non-stationary (cannot reject unit root at 5%)")
    return {"adf_stat": adf_stat, "p_value": p_value, "crit_vals": crit_vals}


def train_test_split_series(series: pd.Series, holdout_days: int):
    series_clean = series.dropna()
    if len(series_clean) <= holdout_days + 5:
        raise ValueError("Not enough data for requested holdout days.")
    train = series_clean.iloc[:-holdout_days]
    test = series_clean.iloc[-holdout_days:]
    return train, test


def fit_arima_and_forecast(train: pd.Series, test: pd.Series, order=(1, 0, 0), future_steps=30):
    model = ARIMA(train, order=order)
    fitted = model.fit()
    pred_test = fitted.forecast(steps=len(test))
    mae = mean_absolute_error(test, pred_test)
    # Refit on full series for future forecast
    full_model = ARIMA(pd.concat([train, test]), order=order)
    full_fitted = full_model.fit()
    future_forecast = full_fitted.forecast(steps=future_steps)
    return {"fitted": fitted, "pred_test": pred_test, "mae": mae, "future": future_forecast}


def plot_predictions(series: pd.Series,
                     train: pd.Series,
                     test: pd.Series,
                     pred_test: pd.Series,
                     future_forecast: pd.Series,
                     ticker: str,
                     filename_prefix: str):
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label="Observed", linewidth=1)
    plt.axvline(x=test.index[0], color="gray", linestyle="--", label="Train/Test split")
    plt.plot(test.index, pred_test, label="Predicted (test)", linestyle="--")
    last_date = series.index[-1]
    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_forecast), freq="B")
    plt.plot(future_index, future_forecast, label=f"Forecast next {len(future_forecast)}", linestyle=":")
    plt.title(f"{ticker} Observed vs Predicted (Close)")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{filename_prefix}_predictions.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def run_for_ticker(ticker: str):
    df = download_data(ticker, YEARS)
    eda(df, ticker)
    moving_average_plot(df, ticker, window=7)
    decompose_series(df, ticker, column="Close")

    # ADF on Close 
    if "Close" in df.columns:
        adf_res = adf_test(df["Close"], name=f"{ticker} Close")
    else:
        adf_res = None
        print("Close column not found; skipped ADF.")

    # ARIMA on Close 
    close_series = df["Close"].dropna()
    train_c, test_c = train_test_split_series(close_series, HOLDOUT_DAYS)
    arima_res = fit_arima_and_forecast(train_c, test_c, order=(1, 0, 0), future_steps=FUTURE_DAYS)
    print(f"\nARIMA(1,0,0) MAE on holdout ({HOLDOUT_DAYS} days): {arima_res['mae']:.4f}")
    plot_predictions(close_series, train_c, test_c, arima_res["pred_test"], arima_res["future"], ticker,
                     filename_prefix=ticker)

    # Save summary
    summary = {
        "ticker": ticker,
        "data_start": df.index.min().strftime("%Y-%m-%d"),
        "data_end": df.index.max().strftime("%Y-%m-%d"),
        "close_holdout_mae": arima_res["mae"],
        "volume_adf_pvalue": adf_res["p_value"] if adf_res is not None else np.nan,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(OUTPUT_DIR, f"{ticker}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary CSV: {summary_path}")


def main():
    print("Starting time series analysis (Option A: ARIMA on Close only).")
    run_for_ticker(SELECTED_TICKER)
    print("Done. Check the 'ts_outputs' folder for plots and summary.")


if __name__ == "__main__":

    main()
