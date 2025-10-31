import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="📈 Pattern Strategy Picker", layout="wide")
st.title("📊 Pattern-Based Strategy Selector")

# --- Input: List of Tickers ---
def_tickers = ["AAPL", "SPY", "AMZN", "GOOGL", "TSLA", "PLTR", "AVGO", "MRVL", "HOOD", "META", "WFC", "C", "COIN", "UBER"]
ticker_list = st.text_area("Enter Tickers (comma-separated):", ", ".join(def_tickers))
tickers = [x.strip().upper() for x in ticker_list.split(",") if x.strip()]

# --- Helper: Detect Candlestick Patterns ---
def detect_pattern(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    preprev = df.iloc[-3]

    # Evening Star: Up candle, doji/small, strong red
    if (preprev['Close'] > preprev['Open'] and
        abs(prev['Close'] - prev['Open']) / (prev['High'] - prev['Low'] + 1e-6) < 0.3 and
        last['Close'] < last['Open'] and
        last['Close'] < prev['Close'] and
        last['Open'] > prev['Close']):
        return "Evening Star"

    # Bearish Engulfing
    if (prev['Close'] > prev['Open'] and
        last['Open'] > prev['Close'] and
        last['Close'] < prev['Open']):
        return "Bearish Engulfing"

    # Three Black Crows
    if (df.iloc[-3:]['Close'] < df.iloc[-3:]['Open']).all():
        return "Three Black Crows"

    # Gap Down Large Red Candle
    if (last['Open'] < prev['Low'] and (last['Open'] - last['Close']) > (prev['Close'] - prev['Open'])):
        return "Gap Down Red Candle"

    return "-"

# --- Helper: Recommend Strategy ---
def strategy_suggestion(pattern, ticker):
    delta_neutral = ["SPY", "TSLA", "C", "WFC", "AMZN"]
    if pattern in ["Evening Star", "Bearish Engulfing"] and ticker in delta_neutral:
        return "Delta-Neutral"
    elif pattern in ["Three Black Crows", "Gap Down Red Candle"]:
        return "Fere Put"
    elif pattern in ["Evening Star", "Bearish Engulfing"]:
        return "Fere Put Combo"
    else:
        return "--"

# --- Run Detection ---
results = []

if st.button("🧠 Analyze Patterns"):
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="7d", interval="1d")
            if len(df) >= 3:
                df = df.reset_index()
                pattern = detect_pattern(df)
                strategy = strategy_suggestion(pattern, ticker)
                results.append({"Ticker": ticker, "Pattern": pattern, "Suggested Strategy": strategy})
        except:
            results.append({"Ticker": ticker, "Pattern": "Error", "Suggested Strategy": "N/A"})

    result_df = pd.DataFrame(results)
    st.subheader("📋 Strategy Recommendations")
    st.dataframe(result_df, use_container_width=True)
else:
    st.info("Click 'Analyze Patterns' to scan your tickers.")
