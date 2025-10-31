import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

def fetch_f_percent_data(ticker, start_date, end_date, timeframe, prev_close):
    """
    Fetches intraday data for a given timeframe, calculates F%, and returns the processed DataFrame.

    Args:
        ticker (str): Stock symbol.
        start_date (str): Start date for data retrieval.
        end_date (str): End date for data retrieval.
        timeframe (str): Timeframe (e.g., '1h', '15m', '5m').
        prev_close (float): Previous day's closing price.

    Returns:
        DataFrame: Processed data with F% calculations.
    """
    try:
        # Fetch intraday data
        intraday = yf.download(
            ticker, start=start_date, end=end_date, interval=timeframe, progress=False
        )

        if intraday.empty:
            return None

        # Reset index to work with timestamps
        intraday.reset_index(inplace=True)

        # Rename 'Datetime' column if it exists
        if "Datetime" in intraday.columns:
            intraday.rename(columns={"Datetime": "Date"}, inplace=True)

        # Convert to New York time
        intraday["Date"] = intraday["Date"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)

        # Extract time in 12-hour format
        intraday["Time"] = intraday["Date"].dt.strftime("%I:%M %p")

        # Keep only date in 'Date' column
        intraday["Date"] = intraday["Date"].dt.strftime("%Y-%m-%d")

        # Calculate F%
        if prev_close is not None:
            intraday["F_numeric"] = ((intraday["Close"] - prev_close) / prev_close) * 10000
        else:
            intraday["F_numeric"] = 0  # Fallback

        return intraday

    except Exception as e:
        st.error(f"Error fetching {timeframe} data for {ticker}: {e}")
        return None


# =================
# Page Config
# =================
st.set_page_config(
    page_title="Modern Day Trading Dashboard",
    layout="wide"
)

st.title("Day Trading (2025 Edition)")

# ======================================
# Sidebar - User Inputs & Advanced Options
# ======================================
st.sidebar.header("Input Options")

default_tickers = ["NVDA", "AVGO", "MU","AMD","PLTR","MRVL","uber","mu","crwd","AMZN","AAPL","googl","MSFT","META","tsla","sbux","nke","chwy","GM","cmg","c","wfc","hood","coin","bac","jpm","PYPL","tgt","wmt","elf"]
tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=default_tickers,
    default=["AVGO"]  # Start with one selected
)

# Date range inputs
start_date = st.sidebar.date_input("Start Date", value=date(2025, 2, 28))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Timeframe selection
timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=["2m", "5m", "15m", "30m", "60m", "1d"],
    index=1  # Default to 5m
)

# Gap threshold slider
gap_threshold = st.sidebar.slider(
    "Gap Threshold (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.1,
    help="Sets the % gap threshold for UP or DOWN alerts."
)

st.sidebar.markdown(
    "_Tip: Increase performance with caching, but be mindful of frequent data updates._"
)

# ======================================
# Helper function to detect "40ish" + reversal
# ======================================
def detect_40ish_reversal(intraday_df):
    """
    Flags reversals when F% is between 44% to 55% (up) or -55% to -44% (down),
    and the next row moves significantly in the opposite direction.
    """
    intraday_df["40ish"] = ""

    for i in range(len(intraday_df) - 1):
        current_val = intraday_df.loc[i, "F_numeric"]
        next_val = intraday_df.loc[i + 1, "F_numeric"]

        # 44% - 55% (Reversal Down) & -55% to -44% (Reversal Up)
        if 44 <= current_val <= 55 and next_val < current_val:
            intraday_df.loc[i, "40ish"] = "40ish UP & Reversed Down"
        elif -55 <= current_val <= -44 and next_val > current_val:
            intraday_df.loc[i, "40ish"] = "❄️ 40ish DOWN & Reversed Up"

    return intraday_df

# Momentum helper (for 2 and 7 periods)
def add_momentum(df, price_col="Close"):
    """
    Adds Momentum_2 and Momentum_7 columns:
      Momentum_2 = Close[t] - Close[t-2]
      Momentum_7 = Close[t] - Close[t-7]
    """
    df["Momentum_2"] = df[price_col].diff(periods=7)
    df["Momentum_7"] = df[price_col].diff(periods=14)
    return df


# ======================================
# Main Button to Run
# ======================================
if st.sidebar.button("Run Analysis"):
    main_tabs = st.tabs([f"Ticker: {t}" for t in tickers])

    for idx, t in enumerate(tickers):
        with main_tabs[idx]:
            try:
                # ================
                # 1) Fetch Previous Day's Data
                # ================
                daily_data = yf.download(
                    t,
                    end=start_date,
                    interval="1d",
                    progress=False
                )

                prev_close, prev_high, prev_low = None, None, None
                prev_close_str, prev_high_str, prev_low_str = "N/A", "N/A", "N/A"

                if not daily_data.empty:
                    if isinstance(daily_data.columns, pd.MultiIndex):
                        daily_data.columns = daily_data.columns.map(
                            lambda x: x[0] if isinstance(x, tuple) else x
                        )
                    prev_close = daily_data["Close"].iloc[-1]
                    prev_high = daily_data["High"].iloc[-1]
                    prev_low = daily_data["Low"].iloc[-1]

                    prev_close_str = f"{prev_close:.2f}"
                    prev_high_str = f"{prev_high:.2f}"
                    prev_low_str = f"{prev_low:.2f}"

                # ================
                # 2) Fetch Intraday Data
                # ================
                intraday = yf.download(
                    t,
                    start=start_date,
                    end=end_date,
                    interval=timeframe,
                    progress=False
                )

                if intraday.empty:
                    st.error(f"No intraday data for {t}.")
                    continue

                intraday.reset_index(inplace=True)
                if isinstance(intraday.columns, pd.MultiIndex):
                    intraday.columns = intraday.columns.map(
                        lambda x: x[0] if isinstance(x, tuple) else x
                    )

                if "Datetime" in intraday.columns:
                    intraday.rename(columns={"Datetime": "Date"}, inplace=True)

                # Convert to New York time
                if intraday["Date"].dtype == "datetime64[ns]":
                    intraday["Date"] = intraday["Date"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                else:
                    intraday["Date"] = intraday["Date"].dt.tz_convert("America/New_York")
                intraday["Date"] = intraday["Date"].dt.tz_localize(None)

                # Add a Time column (12-hour)
                intraday["Time"] = intraday["Date"].dt.strftime("%I:%M %p")
                # Keep only YYYY-MM-DD in Date column
                intraday["Date"] = intraday["Date"].dt.strftime("%Y-%m-%d")

                # Add a Range column
                intraday["Range"] = intraday["High"] - intraday["Low"]

                # ================
                # 3) Calculate Gap Alerts
                # ================
            # Ensure we have a previous close
                gap_alert = ""
                gap_type = None
                gap_threshold_decimal = gap_threshold / 100.0

                if prev_close is not None and not intraday.empty:
                    first_open = intraday["Open"].iloc[0]

                    # Ensure first_open is valid (not NaN)
                    if pd.isna(first_open):
                        first_open = prev_close  # Default to prev close if missing

                    # Calculate the gap percentage
                    gap_percentage = (first_open - prev_close) / prev_close

                    # **Corrected Logic**
                    if first_open > prev_high:  # Must open *above* previous high to count as gap up
                        if gap_percentage > gap_threshold_decimal:
                            gap_alert = "🚀 UP GAP ALERT"
                            gap_type = "UP"
                    elif first_open < prev_low:  # Must open *below* previous low to count as gap down
                        if gap_percentage < -gap_threshold_decimal:
                            gap_alert = "🔻 DOWN GAP ALERT"
                            gap_type = "DOWN"


                # ================
                # 4) High of Day / Low of Day
                # ================
                intraday["High of Day"] = ""
                for date_value, group_df in intraday.groupby("Date", as_index=False):
                    day_indices = group_df.index
                    current_high = -float("inf")
                    last_high_row = None

                    for i2 in day_indices:
                        row_high = intraday.loc[i2, "High"]
                        if row_high > current_high:
                            current_high = row_high
                            last_high_row = i2
                            intraday.loc[i2, "High of Day"] = f"{current_high:.2f}"
                        else:
                            offset = i2 - last_high_row
                            intraday.loc[i2, "High of Day"] = f"+{offset}"

                intraday["Low of Day"] = ""
                for date_value, group_df in intraday.groupby("Date", as_index=False):
                    day_indices = group_df.index
                    current_low = float("inf")
                    last_low_row = None

                    for i2 in day_indices:
                        row_low = intraday.loc[i2, "Low"]
                        if row_low < current_low:
                            current_low = row_low
                            last_low_row = i2
                            intraday.loc[i2, "Low of Day"] = f"{current_low:.2f}"
                        else:
                            offset = i2 - last_low_row
                            intraday.loc[i2, "Low of Day"] = f"+{offset}"

                # ================
                # 5) TD Open Column Example
                # ================
                def check_td_open(row):
                    # Simple example logic
                    if gap_type == "UP":
                        # If price reversed and touched previous day's high
                        if row["Low"] <= prev_high:
                            return "Sell SIGNAL (Reversed Down)"
                    elif gap_type == "DOWN":
                        # If price reversed and touched previous day's low
                        if row["High"] >= prev_low:
                            return "Buy SIGNAL (Reversed Up)"
                    return ""

                intraday["TD Open"] = intraday.apply(check_td_open, axis=1)

                # Get the first intraday open price
                first_open = intraday["Open"].iloc[0]

                def check_td_trap(row):
                    # Only consider TD Trap if the day opened within the previous day's range
                    if first_open > prev_low and first_open < prev_high:
                        # If price moves above previous high, it's a BUY trap signal
                        if row["High"] >= prev_high:
                            return "Buy SIGNAL (TD Trap)"
                        # If price falls below previous low, it's a SELL trap signal
                        elif row["Low"] <= prev_low:
                            return "Sell SIGNAL (TD Trap)"
                    return ""

                intraday["TD Trap"] = intraday.apply(check_td_trap, axis=1)









                prev_open = daily_data["Open"].iloc[-1]   # Yesterday's Open
                prev_close = daily_data["Close"].iloc[-1] # Yesterday's Close
                # Function to check TD CLoP conditions
                def check_td_clop(row):
                    """
                    Checks for TD CLoP signals using previous day's Open (prev_open) and Close (prev_close).
                    - Buy SIGNAL (TD CLoP): Current open < both prev_open & prev_close, then current high > both.
                    - Sell SIGNAL (TD CLoP): Current open > both prev_open & prev_close, then current low < both.
                    """
                    if row["Open"] < prev_open and row["Open"] < prev_close and row["High"] > prev_open and row["High"] > prev_close:
                        return "Buy SIGNAL (TD CLoP)"
                    elif row["Open"] > prev_open and row["Open"] > prev_close and row["Low"] < prev_open and row["Low"] < prev_close:
                        return "Sell SIGNAL (TD CLoP)"
                    return ""

                # Apply function properly
                intraday["TD CLoP"] = intraday.apply(check_td_clop, axis=1)
                def calculate_f_williams_r(df, period=14):
                    """
                    Computes Williams %R using Close Price.
                    """
                    if "Close" not in df.columns:
                        return pd.Series(index=df.index, dtype=float)  # Prevent crashing if column is missing

                    highest_high = df["Close"].rolling(period).max()
                    lowest_low = df["Close"].rolling(period).min()

                    williams_r = -100 * (highest_high - df["Close"]) / (highest_high - lowest_low)

                    return williams_r.fillna(0)  # Fill missing values


                # Now call the function outside the definition:
            # Compute F% numeric (ensure this is the final calculation)
                if prev_close is not None:
                    intraday["F_numeric"] = ((intraday["Close"] - prev_close) / prev_close) * 10000
                else:
                    intraday["F_numeric"] = 0  # fallback


                # Calculate Williams %R after finalizing F_numeric
                intraday["F%_Williams_R"] = calculate_f_williams_r(intraday, period=14)

                # ================
                # Gap-Firm Check (60-min Reversion)
                # ================
                intraday["Gap Firm"] = ""

                if gap_type in ["UP", "DOWN"]:
                    timeframe_map = {
                        "1m": 60,
                        "2m": 30,
                        "5m": 12,
                        "15m": 4,
                        "30m": 2,
                        "60m": 1
                    }
                    bars_needed = timeframe_map.get(timeframe, 12)  # default 12 if missing

                    grouped = intraday.groupby("Date", as_index=False)
                    for date_val, day_df in grouped:
                        open_rows = day_df[day_df["Time"] == "09:30 AM"].index
                        if len(open_rows) == 0:
                            continue

                        open_idx = open_rows[0]
                        day_indices = list(day_df.index)
                        if open_idx not in day_indices:
                            continue

                        open_pos = day_indices.index(open_idx)
                        end_pos = open_pos + bars_needed
                        sub_indices = day_indices[open_pos : end_pos + 1]

                        revert_found = False
                        for idx_i in sub_indices:
                            high_i = intraday.loc[idx_i, "High"]
                            low_i = intraday.loc[idx_i, "Low"]

                            if gap_type == "UP":
                                # e.g. reversion if price dips below prev_high
                                if low_i <= prev_high:
                                    revert_found = True
                                    break
                            elif gap_type == "DOWN":
                                # e.g. reversion if price climbs above prev_low
                                if high_i >= prev_low:
                                    revert_found = True
                                    break

                        # If no reversion happened in 60 mins window, label last bar
                        if not revert_found and sub_indices:
                            final_bar_idx = sub_indices[-1]
                            if gap_type == "UP":
                                intraday.loc[final_bar_idx, "Gap Firm"] = "gap-up firm"
                            else:
                                intraday.loc[final_bar_idx, "Gap Firm"] = "gap-down firm"

                # ================
                # 6) Day Type at 9:30 AM
                # ================
                def determine_trap_status(open_price, p_high, p_low):
                    if open_price is None or pd.isna(open_price):
                        return ""
                    if p_high is None or p_low is None:
                        return "Unknown"
                    if open_price > p_high:
                        return "OUTSIDE (Above Prev High)"
                    elif open_price < p_low:
                        return "OUTSIDE (Below Prev Low)"
                    else:
                        return "WITHIN Range"

                intraday["Day Type"] = ""
                mask_930 = intraday["Time"] == "09:30 AM"
                intraday.loc[mask_930, "Day Type"] = intraday[mask_930].apply(
                    lambda row: determine_trap_status(row["Open"], prev_high, prev_low),
                    axis=1
                )




                # Ensure we have at least 5 rows for calculation
                if len(intraday) >= 5:
                    # 1) Calculate the 5-period moving average of volume
                    intraday["Avg_Vol_5"] = intraday["Volume"].rolling(window=5).mean()

                    # 2) Calculate Relative Volume (RVOL)
                    intraday["RVOL_5"] = intraday["Volume"] / intraday["Avg_Vol_5"]

                    # 3) Drop Avg_Vol_5 column since we only need RVOL_5
                    intraday.drop(columns=["Avg_Vol_5"], inplace=True)
                else:
                    # If not enough data, set RVOL_5 to "N/A"
                    intraday["RVOL_5"] = "N/A"
                # ================
                # 7) Calculate F%
                # ================
                def calculate_f_percentage(intraday_df, prev_close_val):
                    if prev_close_val is not None and not intraday_df.empty:
                        intraday_df["F%"] = ((intraday_df["Close"] - prev_close_val) / prev_close_val) * 10000
                        # Round to nearest integer
                        intraday_df["F%"] = intraday_df["F%"].round(0).astype(int).astype(str) + "%"
                    else:
                        intraday_df["F%"] = "N/A"
                    return intraday_df

                intraday = calculate_f_percentage(intraday, prev_close)




                import numpy as np


                def calculate_f_theta(df, scale_factor=100):
                    """
                    Computes tan(theta) of F% to detect sharp movements.
                    Formula: tan(theta) = F% change (approximate slope)
                    Scales result by scale_factor (default 100).
                    """
                    if "F_numeric" in df.columns:
                        df["F% Theta"] = np.degrees(np.arctan(df["F_numeric"].diff())) * scale_factor
                    else:
                        df["F% Theta"] = 0  # Fallback if column is missing
                    return df

                # Apply function after calculating F_numeric
                intraday = calculate_f_theta(intraday, scale_factor=100)  # Adjust scale_factor if needed

                def detect_theta_spikes(df):
                    """
                    Identifies large spikes in F% Theta automatically using standard deviation.
                    - Uses 2.5x standard deviation as a dynamic threshold.
                    - Detects both positive and negative spikes.
                    """
                    if "F% Theta" not in df.columns:
                        return df  # Avoid crash if missing column

                    theta_std = df["F% Theta"].std()  # Compute stock-specific volatility
                    threshold = 0.8 * theta_std  # Set dynamic threshold

                    df["Theta_Change"] = df["F% Theta"].diff()  # Compute directional change
                    df["Theta_Spike"] = df["Theta_Change"].abs() > threshold  # Detect both up/down spikes

                    return df
                intraday = detect_theta_spikes(intraday)





                def calculate_f_velocity_and_speed(df):
                    """
                    Computes:
                    - **F% Velocity** = directional rate of F% change per bar.
                    - **F% Speed** = absolute rate of F% change per bar (ignores direction).
                    """
                    if "F_numeric" in df.columns:
                        df["F% Velocity"] = df["F_numeric"].diff()  # Includes direction (+/-)
                        df["F% Speed"] = df["F% Velocity"].abs()    # Only magnitude, no direction
                    else:
                        df["F% Velocity"] = 0  # Fallback
                        df["F% Speed"] = 0      # Fallback
                    return df

                # Apply function after calculating F_numeric
                intraday = calculate_f_velocity_and_speed(intraday)


                def calculate_f_theta_cot(df, scale_factor=100):
                    """
                    Computes tan(theta) and cot(theta) of F% to detect sharp movements.
                    - tan(theta) = slope of F% movement
                    - cot(theta) = inverse of tan(theta) (sensitive to small changes)
                    - Results are scaled by `scale_factor` for readability.
                    """
                    if "F_numeric" in df.columns:
                        df["F% Theta"] = np.tan(np.radians(df["F_numeric"].diff())) * scale_factor

                        # Avoid division by zero
                        df["F% Cotangent"] = np.where(df["F% Theta"] != 0, 1 / df["F% Theta"], 0)
                    else:
                        df["F% Theta"] = 0  # Fallback
                        df["F% Cotangent"] = 0  # Fallback
                    return df

                # Apply function after calculating F_numeric
                intraday = calculate_f_theta_cot(intraday, scale_factor=100)



                def calculate_f_sine_cosine(df):
                    """
                    Computes sine and cosine of F% Theta:
                    - sin(θ) indicates how steep the price change is.
                    - cos(θ) indicates how stable the price trend is.
                    """
                    if "F% Theta" in df.columns:
                        df["F% Sine"] = np.sin(np.radians(df["F% Theta"]))
                        df["F% Cosine"] = np.cos(np.radians(df["F% Theta"]))
                    else:
                        df["F% Sine"] = 0  # Fallback
                        df["F% Cosine"] = 0  # Fallback
                    return df

                # Apply the function after calculating F% Theta
                intraday = calculate_f_sine_cosine(intraday)



                def calculate_f_trig(df):
                    """
                    Computes sine, cosine, cosecant, and secant of F% to detect oscillatory trends.
                    - sin(F%) and cos(F%) capture cyclic behavior.
                    - csc(F%) and sec(F%) detect extreme changes.
                    """
                    if "F_numeric" in df.columns:
                        df["F% Sine"] = np.sin(np.radians(df["F_numeric"]))
                        df["F% Cosine"] = np.cos(np.radians(df["F_numeric"]))

                        # Avoid division by zero
                        df["F% Cosecant"] = np.where(df["F% Sine"] != 0, 1 / df["F% Sine"], np.nan)
                        df["F% Secant"] = np.where(df["F% Cosine"] != 0, 1 / df["F% Cosine"], np.nan)
                    else:
                        df["F% Sine"] = df["F% Cosine"] = df["F% Cosecant"] = df["F% Secant"] = np.nan

                    return df

                # Apply the function after computing F_numeric
                intraday = calculate_f_trig(intraday)


                def detect_velocity_spikes(df):
                    """
                    Identifies large spikes in F% Velocity automatically using standard deviation.
                    - Uses 2.5x standard deviation as a dynamic threshold.
                    - Detects both positive and negative spikes.
                    """
                    if "F% Velocity" not in df.columns:
                        return df  # Avoid crash if missing column

                    velocity_std = df["F% Velocity"].std()  # Compute stock-specific volatility
                    threshold = 2.5 * velocity_std  # Set dynamic threshold (adjust multiplier if needed)

                    df["Velocity_Change"] = df["F% Velocity"].diff()  # Compute directional change
                    df["Velocity_Spike"] = df["Velocity_Change"].abs() > threshold  # Detect both up/down spikes

                    return df

                # Apply function with user-defined threshold
                intraday = detect_velocity_spikes(intraday)


                import numpy as np

                def calculate_rsi(f_percent, period=14):
                    delta = f_percent.diff()

                    gain = np.where(delta > 0, delta, 0)
                    loss = np.where(delta < 0, -delta, 0)

                    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
                    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                    return rsi

                # After fetching intraday data and ensuring you have prev_close:
                if prev_close is not None:
                    intraday["F_numeric"] = ((intraday["Close"] - prev_close) / prev_close) * 10000
                else:
                    intraday["F_numeric"] = 0  # fallback

                # Now calculate RSI on numeric F%
                intraday["RSI_F%"] = calculate_rsi(intraday["F_numeric"])

                intraday["RSI_Signal"] = intraday["RSI_F%"].rolling(window=7, min_periods=1).mean()

                # Sample DataFrame
                # Ensure 'Time' is in datetime format and market starts at 9:30 AM
                intraday["Time"] = pd.to_datetime(intraday["Time"])

                # Define the morning session (first two hours)
                morning_mask = (intraday["Time"].dt.time >= pd.to_datetime("09:30").time()) & (intraday["Time"].dt.time <= pd.to_datetime("11:30").time())

                # Get highest and lowest price in the first two hours
                ctod_high = intraday.loc[morning_mask, "High"].max()
                ctod_low = intraday.loc[morning_mask, "Low"].min()

                # Add new columns for CTOD High and Low
                intraday["CTOD_High"] = ctod_high
                intraday["CTOD_Low"] = ctod_low

                # Generate Buy/Sell Alerts
                intraday["Buy_Alert"] = intraday["Close"] > intraday["CTOD_High"]
                intraday["Sell_Alert"] = intraday["Close"] < intraday["CTOD_Low"]

                # Convert boolean alerts to text
                intraday["Alert"] = intraday.apply(
                    lambda row: "Buy" if row["Buy_Alert"] else ("Sell" if row["Sell_Alert"] else ""), axis=1
                )

                # Drop boolean alert columns if not needed
                intraday.drop(columns=["Buy_Alert", "Sell_Alert"], inplace=True)








                # Add numeric version of F% for plotting
                if prev_close is not None:
                    intraday["F_numeric"] = ((intraday["Close"] - prev_close) / prev_close) * 10000
                else:
                    intraday["F_numeric"] = 0  # fallback

                # ================
                # 8) 40ish Column & Reversal Detection
                # ================
                intraday = detect_40ish_reversal(intraday)

                # Add 2-bar momentum (existing example), plus a 7-bar momentum
                intraday = add_momentum(intraday, price_col="Close")  # => Momentum_2, Momentum_7

                # =================================
                # Display Results
                # =================================
                st.subheader(
                    f"{t} (Prev Close: {prev_close_str}, High: {prev_high_str}, Low: {prev_low_str})"
                )
                if gap_alert:
                    st.warning(gap_alert)

                ticker_tabs = st.tabs(["Interactive F% & Momentum", "Intraday Data Table"])

                with ticker_tabs[0]:
                    # -- Create Subplots: Row1=F%, Row2=Momentum
                    fig = make_subplots(
                        rows=5,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[7,5, 6,4,3]  # Adjust row heights to your preference
                    )

                    # (A) F% over time as lines+markers
                    # ---------------------------------
                    max_abs_val = intraday["F_numeric"].abs().max()
                    scatter_f = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F_numeric"],
                        mode="lines+markers",
                        # We'll use a "color scale" trick by mapping F_numeric to color
                        # in a separate step if desired, or just set a single color here.
                        name="F% (scaled)"
                    )
                    fig.add_trace(scatter_f, row=1, col=1)

                    # (A.1) 40ish Reversal (star markers)
                    mask_40ish = intraday["40ish"] != ""
                    scatter_40ish = go.Scatter(
                        x=intraday.loc[mask_40ish, "Time"],
                        y=intraday.loc[mask_40ish, "F_numeric"],
                        mode="markers",
                        marker_symbol="star",
                        marker_size=12,
                        marker_color="gold",
                        name="40ish Reversal",
                        text=intraday.loc[mask_40ish, "40ish"],
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )
                    fig.add_trace(scatter_40ish, row=1, col=1)

                    # (A.2) Dashed horizontal line at 0
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        row=1, col=1,
                        annotation_text="0%",
                        annotation_position="top left"
                    )

                    # (B) Momentum subplot
                    # --------------------
                    # We'll plot both Momentum_2 and Momentum_7
                    scatter_mom2 = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["Momentum_2"],
                        mode="lines",
                        name="Momentum_2",
                        line=dict(color="salmon"),

                    )
                    fig.add_trace(scatter_mom2, row=2, col=1)

                    scatter_mom7 = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["Momentum_7"],
                        mode="lines",
                        name="Momentum_7",
                        line=dict(color="lightblue"),

                    )
                    fig.add_trace(scatter_mom7, row=2, col=1)


                    # Detect Momentum Crossovers
                    intraday["M2_Cross"] = ""

                    for i in range(1, len(intraday)):
                        prev_m2 = intraday.loc[i - 1, "Momentum_2"]
                        prev_m7 = intraday.loc[i - 1, "Momentum_7"]
                        curr_m2 = intraday.loc[i, "Momentum_2"]
                        curr_m7 = intraday.loc[i, "Momentum_7"]

                        # M2 Crosses Above M7 → Bullish
                        if prev_m2 < prev_m7 and curr_m2 > curr_m7:
                            intraday.loc[i, "M2_Cross"] = "Up"

                        # M2 Crosses Below M7 → Bearish
                        elif prev_m2 > prev_m7 and curr_m2 < curr_m7:
                            intraday.loc[i, "M2_Cross"] = "Down"

                    # Create masks for the crossover points
                    mask_m2_up = intraday["M2_Cross"] == "Up"
                    mask_m2_down = intraday["M2_Cross"] == "Down"

                    # M2 Up Crossover (Green "M2")
                    scatter_m2_cross_up = go.Scatter(
                        x=intraday.loc[mask_m2_up, "Time"],
                        y=intraday.loc[mask_m2_up, "Momentum_2"],
                        mode="text",
                        text="M2",
                        textposition="top center",
                        textfont=dict(size=14, color="green"),
                        name="M2 Up Crossover"
                    )

                    # M2 Down Crossover (Red "M2")
                    scatter_m2_cross_down = go.Scatter(
                        x=intraday.loc[mask_m2_down, "Time"],
                        y=intraday.loc[mask_m2_down, "Momentum_2"],
                        mode="text",
                        text="M2",
                        textposition="bottom center",
                        textfont=dict(size=14, color="red"),
                        name="M2 Down Crossover"
                    )

                    # Add to the Momentum Plot (Row 2)
                    fig.add_trace(scatter_m2_cross_up, row=2, col=1)  # Bullish M2 Cross
                    fig.add_trace(scatter_m2_cross_down, row=2, col=1)  # Bearish M2 Cross


                    # Create mask for high velocity change points
                    mask_velocity_spike = intraday["Velocity_Spike"]

                    # Maroon Arrows for F% Velocity Spikes
                    scatter_velocity_spikes = go.Scatter(
                        x=intraday.loc[mask_velocity_spike, "Time"],
                        y=intraday.loc[mask_velocity_spike, "F_numeric"],
                        mode="markers",
                        marker=dict(symbol="arrow-bar-up", size=14, color="#0ff"),
                        name="F% Velocity Spike",
                        text="Velocity Surge",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )



                    # Add to the F% plot (Row 1)
                    fig.add_trace(scatter_velocity_spikes, row=1, col=1)


                    theta_std = intraday["F% Theta"].std()
                    threshold = .8 * theta_std  # Adjust multiplier if needed

                    mask_theta_up = intraday["F% Theta"] > threshold  # Very high positive spikes
                    mask_theta_down = intraday["F% Theta"] < -threshold  # Very low negative spikes

                    scatter_theta_up = go.Scatter(
                        x=intraday.loc[mask_theta_up, "Time"],
                        y=intraday.loc[mask_theta_up, "F_numeric"],
                        mode="text",
                        text="*",
                        textposition="top center",
                        textfont=dict(size=40, color="gray"),
                        name="Extreme Theta UP",
                        showlegend=True,
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Theta Spike"
                    )

                    scatter_theta_down = go.Scatter(
                        x=intraday.loc[mask_theta_down, "Time"],
                        y=intraday.loc[mask_theta_down, "F_numeric"],
                        mode="text",
                        text="*",
                        textposition="bottom center",
                        textfont=dict(size=20, color="black"),
                        name="Extreme Theta DOWN",
                        showlegend=True,
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Theta Spike"

                    )

                    # Ensure they are added to the figure
                    fig.add_trace(scatter_theta_up, row=1, col=1)
                    fig.add_trace(scatter_theta_down, row=1, col=1)




                    # Initialize Momentum Cross column
                    intraday["Momentum_Cross"] = ""

                    # Detect Momentum Crosses Above/Below Zero
                    for i in range(1, len(intraday)):
                        prev_m7 = intraday.loc[i - 1, "Momentum_2"]
                        prev_m14 = intraday.loc[i - 1, "Momentum_7"]
                        curr_m7 = intraday.loc[i, "Momentum_2"]
                        curr_m14 = intraday.loc[i, "Momentum_7"]

                        # Bullish Cross (Above Zero)
                        if prev_m7 < 0 and curr_m7 > 0:
                            intraday.loc[i, "Momentum_Cross"] = "+"
                        elif prev_m14 < 0 and curr_m14 > 0:
                            intraday.loc[i, "Momentum_Cross"] = "+"

                        # Bearish Cross (Below Zero)
                        elif prev_m7 > 0 and curr_m7 < 0:
                            intraday.loc[i, "Momentum_Cross"] = "-"
                        elif prev_m14 > 0 and curr_m14 < 0:
                            intraday.loc[i, "Momentum_Cross"] = "-"

                    # Create Masks for the "+" and "-" signals
                    mask_momentum_plus = intraday["Momentum_Cross"] == "+"
                    mask_momentum_minus = intraday["Momentum_Cross"] == "-"

                    # Plot "+" for Bullish Crosses
                    scatter_momentum_plus = go.Scatter(
                        x=intraday.loc[mask_momentum_plus, "Time"],
                        y=intraday.loc[mask_momentum_plus, "Momentum_7"],  # Plotted on M7
                        mode="text",
                        text="+",
                        textposition="top center",
                        textfont=dict(size=20, color="green"),
                        name="M7/M14 Cross Above Zero"
                    )

                    # Plot "-" for Bearish Crosses
                    scatter_momentum_minus = go.Scatter(
                        x=intraday.loc[mask_momentum_minus, "Time"],
                        y=intraday.loc[mask_momentum_minus, "Momentum_7"],  # Plotted on M7
                        mode="text",
                        text="-",
                        textposition="bottom center",
                        textfont=dict(size=20, color="red"),
                        name="M7/M14 Cross Below Zero"
                    )

                    # Add to Momentum Subplot (Row 2)
                    fig.add_trace(scatter_momentum_plus, row=2, col=1)
                    fig.add_trace(scatter_momentum_minus, row=2, col=1)


                    # Detect RSI Crossovers
                    intraday["RSI_Cross"] = ""

                    for i in range(1, len(intraday)):
                        prev_rsi = intraday.loc[i - 1, "RSI_F%"]
                        prev_signal = intraday.loc[i - 1, "RSI_Signal"]
                        curr_rsi = intraday.loc[i, "RSI_F%"]
                        curr_signal = intraday.loc[i, "RSI_Signal"]

                        # RSI Crosses Above Signal Line → Bullish Crossover
                        if prev_rsi < prev_signal and curr_rsi > curr_signal:
                            intraday.loc[i, "RSI_Cross"] = "Up"

                        # RSI Crosses Below Signal Line → Bearish Crossover
                        elif prev_rsi > prev_signal and curr_rsi < curr_signal:
                            intraday.loc[i, "RSI_Cross"] = "Down"

                    # Create masks for the crossover points
                    mask_rsi_up = intraday["RSI_Cross"] == "Up"
                    mask_rsi_down = intraday["RSI_Cross"] == "Down"

                    scatter_rsi = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["RSI_F%"],
                        mode="lines",
                        line=dict(color="purple"),
                        name="RSI (F%)"
                    )

                    scatter_rsi_signal = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["RSI_Signal"],
                        mode="lines",
                        line=dict(color="yellow", dash="dot"),  # Dashed yellow line for signal
                        name="RSI Signal (7MA)"
                    )


                    # RSI Up Crossover (Green "C")
                    scatter_rsi_cross_up = go.Scatter(
                        x=intraday.loc[mask_rsi_up, "Time"],
                        y=intraday.loc[mask_rsi_up, "RSI_F%"],
                        mode="text",
                        text="C",
                        textposition="top center",
                        textfont=dict(size=14, color="green"),
                        name="RSI Up Crossover"
                    )

                    # RSI Down Crossover (Red "C")
                    scatter_rsi_cross_down = go.Scatter(
                        x=intraday.loc[mask_rsi_down, "Time"],
                        y=intraday.loc[mask_rsi_down, "RSI_F%"],
                        mode="text",
                        text="C",
                        textposition="bottom center",
                        textfont=dict(size=14, color="red"),
                        name="RSI Down Crossover"
                    )

                    fig.add_trace(scatter_rsi, row=3, col=1)
                    fig.add_trace(scatter_rsi_signal, row=3, col=1)  # Adding Signal Line
                    fig.add_trace(scatter_rsi_cross_up, row=3, col=1)  # Add Up Cross
                    fig.add_trace(scatter_rsi_cross_down, row=3, col=1)

                    # Add reference lines for RSI 70 and 30
                    fig.add_hline(
                        y=70,
                        line_dash="dash",
                        line_color="red",
                        row=3, col=1,
                        annotation_text="70 (Overbought)",
                        annotation_position="top right"
                    )
                    fig.add_hline(
                        y=30,
                        line_dash="dash",
                        line_color="blue",
                        row=3, col=1,
                        annotation_text="30 (Oversold)",
                        annotation_position="bottom right"
                    )


                    # (C) CTOD Buy/Sell Triggers (Red & Green Dots)
                    # ----------------------------------------------
                    mask_ctod_buy = intraday["Alert"] == "Buy"
                    mask_ctod_sell = intraday["Alert"] == "Sell"

                    # Buy Alert (Green Dot)
                    scatter_ctod_buy = go.Scatter(
                        x=intraday.loc[mask_ctod_buy, "Time"],
                        y=intraday.loc[mask_ctod_buy, "F_numeric"],
                        mode="markers",
                        marker=dict(symbol="circle", size=10, color="green"),
                        name="CTOD Buy Signal",
                        text="Buy Triggered",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )
                    fig.add_trace(scatter_ctod_buy, row=1, col=1)

                    # Sell Alert (Red Dot)
                    scatter_ctod_sell = go.Scatter(
                        x=intraday.loc[mask_ctod_sell, "Time"],
                        y=intraday.loc[mask_ctod_sell, "F_numeric"],
                        mode="markers",
                        marker=dict(symbol="circle", size=10, color="red"),
                        name="CTOD Sell Signal",
                        text="Sell Triggered",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )
                    fig.add_trace(scatter_ctod_sell, row=1, col=1)


                    # Possibly add a dashed horizontal line at 0 momentum as well
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        row=2, col=1,
                        annotation_text="0 (Momentum)",
                        annotation_position="top left"
                    )
            #   # Here we plot the F%_Williams_R values that were computed
            #         scatter_williams = go.Scatter(
            #             x=intraday["Time"],
            #             y=intraday["F%_Williams_R"],
            #             mode="lines",
            #             line=dict(color="orange"),
            #             name="Williams %R (F%)"
            #         )

            #         fig.add_trace(scatter_williams, row=4, col=1)



                    # scatter_velocity = go.Scatter(
                    #     x=intraday["Time"],
                    #     y=intraday["F% Velocity"],
                    #     mode="lines",
                    #     line=dict(color="darkred"),
                    #     name="F% Velocity"
                    # )

                    # fig.add_trace(scatter_velocity, row=5, col=1)




                  # (D) TD Trap Arrows - Only First Sell TD Trap
                    # ----------------------------------------------
                    td_trap_mask = intraday["TD Trap"].str.contains("Sell", na=False)  # Only Sell TD Traps

                    if not td_trap_mask.empty and td_trap_mask.any():
                        first_sell_trap_idx = td_trap_mask.idxmax()  # Get first occurrence index
                        first_sell_trap_time = intraday.loc[first_sell_trap_idx, "Time"]
                        first_sell_trap_value = intraday.loc[first_sell_trap_idx, "F_numeric"]

                        # Add annotation for first Sell TD Trap (Short = ST)
                        fig.add_annotation(
                            x=first_sell_trap_time,
                            y=first_sell_trap_value - 10,  # Offset to avoid overlap
                            text="ST",  # Short label instead of full text
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowcolor="yellow",
                            font=dict(size=12, color="red", family="Arial Black"),
                        )

                    # Update layout overall
                    fig.update_layout(
                        title=f"{t} – F% (Top),Momentum (Middle), RSI (Bottom), F% Velocity",
                        margin=dict(l=30, r=30, t=50, b=30),
                        height=1000,  # Increase overall figure height (default ~450-600)

                        showlegend=True
                    )

                    fig.update_xaxes(title_text="Time", row=2, col=1)
                    fig.update_yaxes(title_text="F% Scale", row=1, col=1)
                    fig.update_yaxes(title_text="Momentum", row=2, col=1)
                    fig.update_yaxes(title_text="RSI", row=3, col=1)


                    st.plotly_chart(fig, use_container_width=True)

                with ticker_tabs[1]:
                    # Show data table, including new columns
                    cols_to_show = [
                        "Date", "Time", "Open", "High", "Low", "Close",
                       "RVOL_5", "Range", "Day Type", "High of Day",
                        "Low of Day", "F%","F% Velocity","Velocity_Spike","F% Theta","F% Cotangent","F% Sine","F% Cosine" ,"F% Cosecant", "F% Secant","TD Open","TD Trap","TD CLoP", "40ish",
                       "RSI_F%","F%_Williams_R","Alert"
                    ]
                    st.dataframe(intraday[cols_to_show], use_container_width=True)

            except Exception as e:
                st.error(f"Error fetching data for {t}: {e}")
