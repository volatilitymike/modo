import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import numpy as np

# =================
# Page Config
# =================
st.set_page_config(page_title="Multi-Timeframe Dashboard", layout="wide")
st.title("Multi-Timeframe Analysis with F%")

# =================
# SIDEBAR
# =================
st.sidebar.header("Input Options")

default_tickers = ["SPY", "QQQ", "AAPL"]
tickers = st.sidebar.multiselect("Select Tickers", options=default_tickers, default=["SPY"])

start_date = st.sidebar.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

# ------------------------------------------------
# 1) fix_yf_dataframe
# ------------------------------------------------
def fix_yf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans yfinance DataFrame: resets index, renames columns,
       converts UTC to New York time, then formats time as 12-hour AM/PM."""
    if df.empty:
        return df

    # Flatten multi-index columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map("_".join)

    # Reset index so we have a 'Datetime' column
    df = df.reset_index()

    # Rename to "Datetime"
    if "Date" in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    elif "index" in df.columns:
        df.rename(columns={"index": "Datetime"}, inplace=True)

    # Convert to datetime in UTC, then shift to America/New_York
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert("America/New_York")

    # Finally, format it as HH:MM AM/PM
    df["Datetime"] = df["Datetime"].dt.strftime("%I:%M %p")

    # Ensure we have a "Close" column
    if "Close" not in df.columns:
        possible_closes = [c for c in df.columns if c.startswith("Close")]
        if possible_closes:
            df.rename(columns={possible_closes[0]: "Close"}, inplace=True)

    return df

# ------------------------------------------------
# 2) fetch_data
# ------------------------------------------------
@st.cache_data
def fetch_data(symbol, interval, start, end):
    """Fetch intraday or daily data from Yahoo Finance,
       and then apply fix_yf_dataframe to finalize the time format."""
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        return df

    # Just keep some columns we care about:
    df = fix_yf_dataframe(df)

    # Filter out empty or invalid rows if needed
    df = df[df["Close"].notna()]

    return df

# ------------------------------------------------
# 3) F% Calculation
# ------------------------------------------------
def calculate_f_percentage(intraday_df, prev_close_val):
    if prev_close_val is not None and not intraday_df.empty:
        intraday_df["F_numeric"] = ((intraday_df["Close"] - prev_close_val) / prev_close_val) * 10000
        intraday_df["F%"] = intraday_df["F_numeric"].round(0).astype(int).astype(str) + "%"
    else:
        intraday_df["F%"] = "N/A"
    return intraday_df


# ------------------------------------------------
# 4) detect_40ish_reversal
# ------------------------------------------------
def detect_40ish_reversal(intraday_df):
    """
    Flags reversals when F% is between 44% to 55% (up) or -55% to -44% (down),
    and the next row moves significantly in the opposite direction.
    """
    if "F_numeric" not in intraday_df.columns:
        intraday_df["40ish"] = ""
        return intraday_df

    intraday_df["40ish"] = ""
    for i in range(len(intraday_df) - 1):
        current_val = intraday_df.loc[i, "F_numeric"]
        next_val = intraday_df.loc[i + 1, "F_numeric"]
        if pd.isna(current_val) or pd.isna(next_val):
            continue

        # 44% - 55% (Reversal Down) & -55% to -44% (Reversal Up)
        if 44 <= current_val <= 55 and next_val < current_val:
            intraday_df.loc[i, "40ish"] = "40ish UP & Reversed Down"
        elif -55 <= current_val <= -44 and next_val > current_val:
            intraday_df.loc[i, "40ish"] = "❄️ 40ish DOWN & Reversed Up"
    return intraday_df

def calculate_f_theta(df, scale_factor=1):
    """Computes F% Theta with normalized F_numeric to avoid excessive scaling."""
    if "F_numeric" in df.columns and not df.empty:
        df["F_numeric"] = df["F_numeric"].astype(float)  # Ensure it's numeric

        # Normalize by dividing by a reasonable factor (e.g., 100)
        df["F% Theta"] = np.degrees(np.arctan(df["F_numeric"].diff() / 100)) * scale_factor
    else:
        df["F% Theta"] = 0
    return df


# ------------------------------------------------
# 6) create_chart
# ------------------------------------------------
def create_chart(df: pd.DataFrame, label: str, y_col: str = "F_numeric") -> go.Figure:
    """Creates a simple line chart for a given DataFrame column (default F_numeric)."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{label} - No Data")
        return fig

    fig = px.line(df, x="Datetime", y=y_col, title=f"{label} Timeframe - {y_col}")
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Time",
        yaxis_title=y_col
    )
    return fig

# ------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------
if st.sidebar.button("Run Analysis"):
    if not tickers:
        st.warning("No ticker selected. Please select at least one ticker.")
    else:
        tab_tickers = st.tabs(tickers)  # Create a tab for each ticker
        for i, symbol in enumerate(tickers):
            with tab_tickers[i]:
                st.write(f"## {symbol}")

                # 1) Get daily data for previous close
                daily_data = fetch_data(symbol, "1d", start_date, end_date)
                prev_close = daily_data["Close"].iloc[-1] if not daily_data.empty else None
                st.write(f"Previous day's close for {symbol} = {prev_close}")

                # 2) Fetch intraday data
                df_60m = fetch_data(symbol, "60m", start_date, end_date)
                df_15m = fetch_data(symbol, "15m", start_date, end_date)
                df_5m = fetch_data(symbol, "5m", start_date, end_date)

                # 3) Compute F%
                for df_intraday in [df_60m, df_15m, df_5m]:
                    df_intraday = calculate_f_percentage(df_intraday, prev_close)
                    df_intraday = detect_40ish_reversal(df_intraday)
                    df_intraday = calculate_f_theta(df_intraday)

                st.success("Intraday data + F% calculations complete!")

                # Show final data: 60m, 15m, 5m
                st.write("### 60-Minute Data")
                if df_60m.empty:
                    st.warning("No data for 60m timeframe.")
                else:
                    st.write(f"Rows: {df_60m.shape[0]}")
                    fig_60m = create_chart(df_60m, label="60m", y_col="F_numeric")
                    st.plotly_chart(fig_60m, use_container_width=True)
                    st.dataframe(df_60m)

                st.write("### 15-Minute Data")
                if df_15m.empty:
                    st.warning("No data for 15m timeframe.")
                else:
                    st.write(f"Rows: {df_15m.shape[0]}")
                    fig_15m = create_chart(df_15m, label="15m", y_col="F_numeric")
                    st.plotly_chart(fig_15m, use_container_width=True)
                    st.dataframe(df_15m)

                st.write("### 5-Minute Data")
                if df_5m.empty:
                    st.warning("No data for 5m timeframe.")
                else:
                    st.write(f"Rows: {df_5m.shape[0]}")
                    fig_5m = create_chart(df_5m, label="5m", y_col="F_numeric")
                    st.plotly_chart(fig_5m, use_container_width=True)
                    st.dataframe(df_5m)

else:
    st.write("Click **Run Analysis** to get started.")
