# ──────────────────────────────────────────────────────────────────────────────
#  Mike Market-Vibe  ▸  SPY · QQQ · MSFT · NVDA · AAPL · AMD  –  Vital-Signs
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.title("📈 SPY · QQQ  –  Mike Intraday Vital Signs")

# ── 1. sidebar controls
st.sidebar.header("Settings")
start_date = st.sidebar.date_input(
    "Start date", datetime.today() - timedelta(days=1))
end_date   = st.sidebar.date_input(
    "End date"  , datetime.today())
run_btn    = st.sidebar.button("🚀  Run Analysis")

tickers = ["SPY", "QQQ", "MSFT", "NVDA", "AAPL", "AMD"]

# ── 2. helper to fetch + massage 5-min bars
@st.cache_data(show_spinner=False)
def load_intraday(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),   # yfinance end is exclusive
        interval="5m",
        progress=False,
        actions=False,
        auto_adjust=False
    )
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # timezone → New York, keep bare datetime
    time_col = "Datetime" if "Datetime" in df else "Date"
    df["Time"] = (df[time_col]
                  .dt.tz_localize("UTC")
                  .dt.tz_convert("America/New_York")
                  .dt.tz_localize(None))

    # NY regular-session filter
    df = df[
        (df["Time"].dt.time >= datetime.strptime("09:30", "%H:%M").time()) &
        (df["Time"].dt.time <= datetime.strptime("16:00", "%H:%M").time())
    ].copy()

    # ── core metrics
    df["%Change"]   = df["Close"].pct_change()*100
    df["SD_Change"] = df["%Change"].rolling(window=5).std()

    df["Tenkan"] = (
        df["High"].rolling(9).max() + df["Low"].rolling(9).min()
    )/2
    df["Kijun"] = (
        df["High"].rolling(26).max() + df["Low"].rolling(26).min()
    )/2

    base_close = df["Close"].iloc[0]
    df["F% Tenkan"] = ((df["Tenkan"]-base_close)/base_close)*10000
    df["F% Kijun"]  = ((df["Kijun"] -base_close)/base_close)*10000

    df = df[["Time", "Close", "%Change",
             "SD_Change", "F% Tenkan", "F% Kijun"]].copy()
    df.columns = pd.MultiIndex.from_product(
        [[ticker], df.columns[1:]], names=["Ticker", "Metric"]
    ).insert(0, ("Time", ""))  # first col = Time w/ empty metric

    return df

# ── 3. run
if run_btn:
    pieces = []
    for t in tickers:
        part = load_intraday(t, start_date, end_date)
        if not part.empty:
            pieces.append(part)

    if pieces:
        # merge on Time
        merged = pieces[0]
        for part in pieces[1:]:
            merged = merged.merge(part, on=("Time", ""), how="outer")

        merged = merged.sort_values("Time").reset_index(drop=True)

        # pretty display
        merged.columns = pd.MultiIndex.from_tuples(
            merged.columns, names=["Ticker", "Metric"])
        merged = merged.set_index("Time")

        st.dataframe(merged, use_container_width=True,
                     height=600)
    else:
        st.warning("No intraday data returned for the selected range.")
