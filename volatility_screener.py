import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Volatility Screener", layout="wide")

# Title
st.title("📊 Volatility Screener")

# Ticker and date input
# --- STOCK GROUP SELECTION ---
stock_groups = {
    "Tech": ["MSFT", "AMZN", "AAPL", "AMD", "MU", "GOOGL", "PLTR", "UBER", "SMCI", "PANW", "CRWD", "NVDA", "AVGO", "MRVL","ON"],
    "Cyclical": ["CART", "SBUX", "DKNG", "CMG", "URBN","TSLA", "CHWY", "NKE", "ETSY", "CROX", "W", "TGT", "GM", "GME", "AMZN", "HD", "CHWY"],
    "Finance": ["C", "WFC", "JPM", "HOOD", "V", "BAC", "PYPL", "COIN"],
    "Communication": ["NFLX", "GOOGL", "RBLX", "PINS", "NFLX", "DASH", "DIS", "META"],
    "ETF": ["SPY","KBE", "QQQ"],
    "Defensive": ["TGT", "COST"]
}

# Group Dropdown
selected_group = st.selectbox("📂 Choose Stock Group", options=list(stock_groups.keys()))
group_tickers = stock_groups[selected_group]

# Ticker dropdown from selected group
selected_group_ticker = st.selectbox("📈 Select Ticker from Group", options=group_tickers)

# Horizontal rule to separate custom search
st.markdown("---")

# Optional manual override (any stock)
custom_ticker_input = st.text_input("🔍 Or enter a custom stock symbol:")

# Final ticker logic: use custom if typed, otherwise use group selection
ticker = custom_ticker_input.strip().upper() if custom_ticker_input else selected_group_ticker
st.markdown(f"**✅ Active Ticker Selected:** `{ticker}`")

date_range = st.date_input("Select Date Range:",
                           value=(pd.to_datetime("2023-10-01"),
                                  pd.to_datetime("2024-03-25")))








if st.button("Load Data"):
    if ticker and len(date_range) == 2:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                st.warning("⚠️ No data returned. Check the ticker symbol or date range.")
                st.stop()

            df.reset_index(inplace=True)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not all(col in df.columns for col in ["Open", "Close"]):
                st.warning("⚠️ Missing 'Open' or 'Close' columns. This ticker may not have valid historical data.")
                st.stop()

            df = df[["Date", "Open", "Close"]].copy()
            df["Daily Return (%)"] = df["Close"].pct_change() * 100
            df["Daily Return"] = df["Close"].pct_change()
            df["Rolling StdDev (Daily Return)"] = df["Daily Return"].rolling(window=20).std()
            df["Log Return (C-C)"] = np.log(df["Close"] / df["Close"].shift(1))
            df["Rolling StdDev (C-C)"] = df["Log Return (C-C)"].rolling(window=20).std()
            df["StdDev of C-C Change"] = (df["Log Return (C-C)"] / df["Rolling StdDev (C-C)"]).round(2)
            df["Value of 1 StdDev Change (C-C)"] = (df["Rolling StdDev (C-C)"] * df["Close"]).round(2)
            df.dropna(subset=["Value of 1 StdDev Change (C-C)"], inplace=True)
            df["Annual Volatility (%)"] = df["Rolling StdDev (C-C)"] * np.sqrt(252) * 100



            # Trig-based analysis of Annual Volatility
            # Normalize AV to [0,1] by dividing by 100
            df["Theta AV"] = np.arccos(np.clip(df["Annual Volatility (%)"] / 100, -1, 1)) * 100
            df["Cotangent AV"] = 1 / np.tan(df["Theta AV"])

            # Optional: round for readability
            df["Theta AV"] = df["Theta AV"].round(2)
            df["Cotangent AV"] = df["Cotangent AV"].round(2)


            # Compute sine and cosine (hidden), secant and cosecant (displayed)
            theta_radians = df["Theta AV"] / 100  # Convert back to radians for trig functions

            df["Sine AV"] = np.sin(theta_radians)
            df["Cosine AV"] = np.cos(theta_radians)

            # Compute secant and cosecant
            df["Secant AV"] = 1 / df["Cosine AV"]
            df["Cosecant AV"] = 1 / df["Sine AV"]

            # Optional: round for readability
            df["Secant AV"] = df["Secant AV"].round(2)
            df["Cosecant AV"] = df["Cosecant AV"].round(2)

            # --- BBW: Bollinger Band Width ---
            # Use a 20-period rolling window (standard)
            rolling_mean = df["Close"].rolling(window=20).mean()
            rolling_std = df["Close"].rolling(window=20).std()

            # Upper and lower bands
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std

            # BBW = (Upper - Lower) / Middle
            df["BBW"] = (upper_band - lower_band) / rolling_mean
            df["BBW"] = df["BBW"] * 100  # optional: convert to percentage
            df["BBW"] = df["BBW"].round(2)


            df.dropna(subset=["Annual Volatility (%)"], inplace=True)
            df["Dollar Change"] = df["Close"] - df["Close"].shift(1)
            df["Intraday Price Change (O-C)"] = df["Close"] - df["Open"]

            df["Log Return (O-C)"] = np.log(df["Close"] / df["Open"])
            df["Rolling StdDev (O-C)"] = df["Log Return (O-C)"].rolling(window=20).std()
            df["StdDev of O-C Change"] = df["Log Return (O-C)"] / df["Rolling StdDev (O-C)"]
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["StdDev of O-C Change"], inplace=True)

            # RVI
            window = 14
            df["Rolling StdDev"] = df["Close"].rolling(window=window).std()
            df["Up Move"] = np.where(df["Close"] > df["Close"].shift(1), df["Rolling StdDev"], 0)
            df["EMA Up"] = df["Up Move"].ewm(span=window, adjust=False).mean()
            df["EMA Std"] = df["Rolling StdDev"].ewm(span=window, adjust=False).mean()
            df["RVI"] = 100 * (df["EMA Up"] / df["EMA Std"])

            # VAS
            df["Daily Return"] = df["Close"].pct_change()
            df["MAD"] = df["Daily Return"].rolling(window=21).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
            )
            df["Daily_MAD_Ratio"] = df["Daily Return"] / (df["MAD"] + 1e-10)
            df["VAS"] = df["Daily_MAD_Ratio"].rolling(window=21).sum()

            # ATR
            df["High"] = df["Close"].rolling(2).max()
            df["Low"] = df["Close"].rolling(2).min()
            df["Previous Close"] = df["Close"].shift(1)
            df["High-Low"] = df["High"] - df["Low"]
            df["High-PC"] = abs(df["High"] - df["Previous Close"])
            df["Low-PC"] = abs(df["Low"] - df["Previous Close"])
            df["True Range"] = df[["High-Low", "High-PC", "Low-PC"]].max(axis=1)
            df["ATR"] = df["True Range"].rolling(window=14).mean()
            df["ATR %"] = (df["ATR"] / df["Close"]) * 100
            df["ATR %"] = df["ATR %"].round(2)

            # ✅ Save full calculated DataFrame to session state
            st.session_state.df = df.copy()

            # Display table
            display_cols = [
                "Date", "Open", "Close", "Dollar Change", "Daily Return (%)", "Annual Volatility (%)","Theta AV", "Cotangent AV","Secant AV", "Cosecant AV",

                "StdDev of C-C Change", "Value of 1 StdDev Change (C-C)",
                "StdDev of O-C Change", "RVI", "VAS", "ATR %", "ATR","BBW"
            ]

            if not df[display_cols].empty:
                st.subheader("📈 Final Volatility Table")
                st.dataframe(df[display_cols].sort_values(by="Date", ascending=False), use_container_width=True)
            else:
                st.warning("⚠️ No data left after calculations. Try extending the date range.")




        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
    else:
        st.info("Please enter a valid ticker and date range.")

# ✅ Persistent charting block — always visible after data is loaded
if "df" in st.session_state:
    df = st.session_state.df
    st.subheader("📊 Annual Volatility Over Time")
    fig = px.line(
        df.sort_values(by="Date"),
        x="Date",
        y="Annual Volatility (%)",
        title="Annualized Volatility (%) Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)



    st.markdown("---")
    st.subheader("🧠 Indicator Response: Next-Day Price Change")

    # Let user choose which indicator to analyze
    indicator_options = ["Cotangent AV", "Theta AV", "Secant AV", "Cosecant AV", "VAS", "RVI", "ATR %", "BBW"]
    selected_indicator = st.selectbox("📐 Select Indicator to Analyze Next-Day Change:", options=indicator_options)

    # Threshold input
    threshold = st.number_input(f"🔍 Threshold for {selected_indicator} (e.g. >= this value)", value=7.0)
    # ✅ Add Next Close and Change to full df
    df["Next Close"] = df["Close"].shift(-1)
    df["Next-Day Change ($)"] = (df["Next Close"] - df["Close"]).round(2)

    # 🔁 Apply signed logic
    if threshold >= 0:
        trigger_rows = df[df[selected_indicator] >= threshold].copy()
    else:
        trigger_rows = df[df[selected_indicator] <= threshold].copy()


    # Select what to show
    result_cols = ["Date", selected_indicator, "Close", "Next Close", "Next-Day Change ($)"]
    st.dataframe(trigger_rows[result_cols].dropna(), use_container_width=True)

