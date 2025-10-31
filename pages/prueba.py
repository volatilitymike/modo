import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

# ───────────────────────────────────────────────
# Page configuration
# ───────────────────────────────────────────────
st.set_page_config(page_title="OHLC Session Viewer", layout="wide")
st.title("📊 Session Close & Yesterday OHLC")

# ───────────────────────────────────────────────
# Sidebar inputs
# ───────────────────────────────────────────────
ticker = st.sidebar.text_input("Ticker", "NVDA").upper()

# Select a trading session (default today)
session_date = st.sidebar.date_input("Session (date)", value=date.today())

# Intraday timeframe
bar_size = st.sidebar.selectbox("Intraday Interval", ["1m", "2m", "5m", "15m", "30m", "60m"], index=2)

run = st.sidebar.button("Run")

# ───────────────────────────────────────────────
# Helper – convert tz to New‑York and drop tz‑info
# ───────────────────────────────────────────────

def to_est(series: pd.Series) -> pd.Series:
    if series.dt.tz is None:
        return series.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    return series.dt.tz_convert("America/New_York").dt.tz_localize(None)

# ───────────────────────────────────────────────
# Main logic
# ───────────────────────────────────────────────
if run:
    try:
        # 1️⃣  Pull daily data to get yesterday's OHLC
        daily = yf.download(ticker, period="7d", interval="1d", progress=False, threads=False)
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = [c[0] if isinstance(c, tuple) else c for c in daily.columns]

        # yesterday relative to chosen session
        prev_row = daily[daily.index.date < session_date].iloc[-1]
        y_open, y_high, y_low, y_close = prev_row["Open"], prev_row["High"], prev_row["Low"], prev_row["Close"]

        st.sidebar.markdown(
            f"**Yesterday**  "+
            f"Open: {y_open:.2f}  "+
            f"High: {y_high:.2f}  "+
            f"Low: {y_low:.2f}  "+
            f"Close: {y_close:.2f}"
        )

        # 2️⃣  Fetch intraday for the session
        start_dt = pd.Timestamp(session_date)
        end_dt = start_dt + pd.Timedelta(days=1)

        intra = yf.download(
            ticker,
            start=start_dt.date(),
            end=end_dt.date(),
            interval=bar_size,
            progress=False,
            threads=False,
        )

        if intra.empty:
            st.warning("No intraday data returned for that date / interval.")
            st.stop()

        intra.reset_index(inplace=True)
        if isinstance(intra.columns, pd.MultiIndex):
            intra.columns = [c[0] if isinstance(c, tuple) else c for c in intra.columns]

        if "Datetime" in intra.columns:
            intra.rename(columns={"Datetime": "Date"}, inplace=True)
        intra["Date"] = pd.to_datetime(intra["Date"])
        intra["Date"] = to_est(intra["Date"])
        # Compute F% based on Close relative to y_close
        intra["F%"] = ((intra["Close"] - y_close) / y_close) * 1000

        # 3️⃣  Plot Close and yesterday OHLC lines
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=intra["Date"], y=intra["Close"], mode="lines+markers", name="Close"))

        line_specs = [
            (y_open, "Yesterday Open", "blue"),
            (y_high, "Yesterday High", "green"),
            (y_low, "Yesterday Low", "red"),
            (y_close, "Yesterday Close", "gray")
        ]
        for y_val, label, color in line_specs:
            # add constant‑y scatter so users can hover the price
            fig.add_trace(
                go.Scatter(
                    x=intra["Date"],
                    y=[y_val] * len(intra),
                    mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    name=label,
                    hovertemplate=f"{label}: $%{{y:.2f}}<extra></extra>",
                    showlegend=True,
                )
            )



        fig.update_layout(
            title=f"{ticker} – Close Prices with Yesterday's OHLC ({bar_size})",
            xaxis_title="Time (ET)",
            yaxis_title="Price ($)",
            height=650,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Optional preview table
        with st.expander("Data Preview"):
            st.dataframe(intra[["Date", "Open", "High", "Low", "Close"]].head())

    except Exception as e:
        st.error(f"Error: {e}")
