import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date


# =================
# Page Config
# =================
st.set_page_config(
    page_title="Modern Day Trading Dashboard",
    layout="wide"
)

st.title("Options Wealth")

# ======================================
# Sidebar - User Inputs & Advanced Options
# ======================================
st.sidebar.header("Input Options")

default_tickers = ["SPY", "QQQ","low" ,"hd","NVDA", "AVGO","AMD","PLTR","MRVL","uber","mu","crwd","AMZN","AAPL","googl","MSFT","META","tsla","sbux","nke","chwy","DKNG","GM","cmg","c","wfc","hood","coin","bac","jpm","PYPL","tgt","wmt","elf"]
tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=default_tickers,
    default=["SPY"]  # Start with one selected
)

# Date range inputs
start_date = st.sidebar.date_input("Start Date", value=date(2025, 3, 6))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Timeframe selection
timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=["2m", "5m", "15m", "30m", "60m", "1d"],
    index=1  # Default to 5m
)

# # 🔥 Candlestick Chart Toggle (Place this here)
# show_candlestick = st.sidebar.checkbox("Show Candlestick Chart", value=False)



# Gap threshold slider
gap_threshold = st.sidebar.slider(
    "Gap Threshold (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.1,
    help="Sets the % gap threshold for UP or DOWN alerts."
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


                def adjust_marker_y_positions(data, column, base_offset=5):
                    """
                    Adjusts Y-axis positions dynamically to prevent symbol overlap.
                    - `column`: Column containing the markers (e.g., "TD REI Crossover", "VAS Transition").
                    - `base_offset`: Minimum gap between symbols.
                    """
                    y_positions = {}  # Dictionary to track adjustments

                    adjusted_y = []  # List to store adjusted Y-values
                    for i, time in enumerate(data["Time"]):
                        marker = data.at[data.index[i], column]

                        if pd.notna(marker) and marker != "":
                            # If multiple markers exist at the same time, increment the y-offset
                            if time in y_positions:
                                y_positions[time] -= base_offset  # Push down
                            else:
                                y_positions[time] = data.at[data.index[i], "F_numeric"]  # Start at F%

                            adjusted_y.append(y_positions[time])  # Assign adjusted position
                        else:
                            adjusted_y.append(data.at[data.index[i], "F_numeric"])  # Default to F%

                    return adjusted_y




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



                # Now call the function outside the definition:
            # Compute F% numeric (ensure this is the final calculation)
                if prev_close is not None:
                    intraday["F_numeric"] = ((intraday["Close"] - prev_close) / prev_close) * 10000
                else:
                    intraday["F_numeric"] = 0  # fallback

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
                    threshold = 2 * theta_std  # Set dynamic threshold

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


                def calculate_f_std_bands(df, window=20):
                    if "F_numeric" in df.columns:
                        df["F% MA"] = df["F_numeric"].rolling(window=window, min_periods=1).mean()
                        df["F% Std"] = df["F_numeric"].rolling(window=window, min_periods=1).std()
                        df["F% Upper"] = df["F% MA"] + (2 * df["F% Std"])
                        df["F% Lower"] = df["F% MA"] - (2 * df["F% Std"])
                    return df

                # Apply it to the dataset BEFORE calculating BBW
                intraday = calculate_f_std_bands(intraday, window=20)

                def calculate_f_bbw(df, scale_factor=10):
                            """
                            Computes Bollinger Band Width (BBW) for F% and scales it down.
                            BBW = (Upper Band - Lower Band) / |Middle Band| * 100
                            The result is then divided by `scale_factor` to adjust its magnitude.
                            """
                            if "F% Upper" in df.columns and "F% Lower" in df.columns and "F% MA" in df.columns:
                                df["F% BBW"] = (((df["F% Upper"] - df["F% Lower"]) / df["F% MA"].abs().replace(0, np.nan)) * 100) / scale_factor
                                df["F% BBW"].fillna(0, inplace=True)
                            return df

                        # Apply the function with scaling (e.g., divide by 100)
                intraday = calculate_f_bbw(intraday, scale_factor=10)

                def calculate_kijun_sen(df, period=26):
                    highest_high = df["High"].rolling(window=period, min_periods=1).max()
                    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()
                    df["Kijun_sen"] = (highest_high + lowest_low) / 2
                    return df

                intraday = calculate_kijun_sen(intraday, period=26)
                # Use the previous close (prev_close) from your daily data
                intraday["Kijun_F"] = ((intraday["Kijun_sen"] - prev_close) / prev_close) * 10000


                # Apply the function to your intraday data
                intraday = calculate_kijun_sen(intraday, period=26)

                def f_ichimoku_confirmation(row):
                    if row["Close"] > row["Kijun_sen"]:
                        # Price is above Kijun → bullish bias
                        if row["F_numeric"] > 0:
                            return "Confirmed Bullish"
                        else:
                            return "Bullish Price, but F% negative"
                    else:
                        # Price is below Kijun → bearish bias
                        if row["F_numeric"] < 0:
                            return "Confirmed Bearish"
                        else:
                            return "Bearish Price, but F% positive"

            # Apply this function row-wise
                intraday["F_Ichimoku_Confirmation"] = intraday.apply(f_ichimoku_confirmation, axis=1)



                def detect_cross(series, reference):
                    """
                    Returns a Series with:
                    - "up" if the series crosses above the reference (i.e. previous value below and current value at/above)
                    - "down" if it crosses below (previous value above and current value at/below)
                    - "" otherwise.
                    """
                    cross = []
                    for i in range(len(series)):
                        if i == 0:
                            cross.append("")
                        else:
                            if series.iloc[i-1] < reference.iloc[i-1] and series.iloc[i] >= reference.iloc[i]:
                                cross.append("up")
                            elif series.iloc[i-1] > reference.iloc[i-1] and series.iloc[i] <= reference.iloc[i]:
                                cross.append("down")
                            else:
                                cross.append("")
                    return pd.Series(cross, index=series.index)

                # Detect crosses of F_numeric over its middle band:
                intraday["Cross_Mid"] = detect_cross(intraday["F_numeric"], intraday["F% MA"])

                # Detect crosses of F_numeric over the Kijun_F line:
                intraday["Cross_Kijun"] = detect_cross(intraday["F_numeric"], intraday["Kijun_F"])


                def map_alert_mid(cross):
                    if cross == "up":
                        return "POMB"
                    elif cross == "down":
                        return "PUMB"
                    else:
                        return ""

                def map_alert_kijun(cross):
                    if cross == "up":
                        return "POK"
                    elif cross == "down":
                        return "PUK"
                    else:
                        return ""

                intraday["Alert_Mid"] = intraday["Cross_Mid"].apply(map_alert_mid)
                intraday["Alert_Kijun"] = intraday["Cross_Kijun"].apply(map_alert_kijun)




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
                intraday["CTOD Alert"] = intraday.apply(
                    lambda row: "Buy" if row["Buy_Alert"] else ("Sell" if row["Sell_Alert"] else ""), axis=1
                )


                # Drop boolean alert columns if not needed
                intraday.drop(columns=["Buy_Alert", "Sell_Alert"], inplace=True)


                # Ensure RSI Crossovers are calculated before Master Buy Signal
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



                    def detect_wealth_signals(df, expiration_bars=12):
                            """
                            Wealth Trading Signals with Color Coding:
                            - Wealth Buy/Sell I triggers when RVOL_5 > 1.8 (Volume Spike).
                            - Different colors based on spike intensity:
                            - 🔴 Red: Extreme Volume (RVOL_5 > 1.8)
                            - 🟡 Yellow: Strong Volume (RVOL_5 > 1.5)
                            - 🌸 Pink: Moderate Volume (RVOL_5 > 1.2)
                            - Buy II & Sell II depend on Kijun crossovers.
                            - Buy III & Sell III confirm trend reversals.
                            - Buy IV & Sell IV confirm additional volume spikes in trend direction.
                            """

                            df["Wealth Signal"] = ""
                            volume_spike_active = False
                            buy_ii_active = False  # Track if Buy II has happened
                            sell_ii_active = False  # Track if Sell II has happened
                            volume_spike_index = None  # Track when volume spike happened
                            above_kijun = False  # Track if price is already above Kijun after Buy II
                            below_kijun = False  # Track if price is already below Kijun after Sell II

                            for i in range(1, len(df)):
                                # ✅ **Check for Volume Spike (Triggers Wealth Buy I / Wealth Sell I)**
                                if df.loc[i, "RVOL_5"] > 1.2:  # Any RVOL spike above 1.2 triggers a signal
                                    if df.loc[i, "RVOL_5"] > 1.8:
                                        color = "red"  # Extreme Volume → Default (Red for Sell, Green for Buy)
                                    elif df.loc[i, "RVOL_5"] > 1.5:
                                        color = "yellow"  # Strong Volume → Change to Yellow
                                    else:
                                        color = "pink"  # Moderate Volume → Change to Pink



                                    # ✅ **Continue with Volume Spike Activation**
                                    volume_spike_active = True
                                    buy_ii_active = False  # Reset buy tracking
                                    sell_ii_active = False  # Reset sell tracking
                                    volume_spike_index = i  # Track when it happened

                                # ✅ **Check if the signal should expire**
                                if volume_spike_active and volume_spike_index is not None:
                                    if i - volume_spike_index > expiration_bars:
                                        volume_spike_active = False  # Expire the signal
                                        buy_ii_active = False
                                        sell_ii_active = False
                                        volume_spike_index = None  # Reset tracking
                                        above_kijun = False  # Reset tracking
                                        below_kijun = False  # Reset tracking

                                # ✅ **If volume spike is active, check for confirmation signals**
                                if volume_spike_active:
                                    prev_f, curr_f = df.loc[i - 1, "F_numeric"], df.loc[i, "F_numeric"]
                                    prev_kijun, curr_kijun = df.loc[i - 1, "Kijun_F"], df.loc[i, "Kijun_F"]

                                    kijun_cross_up = prev_f < prev_kijun and curr_f >= curr_kijun
                                    kijun_cross_down = prev_f > prev_kijun and curr_f <= curr_kijun

                                    # ✅ **Handle first Kijun cross (Buy II / Sell II)**
                                    if not buy_ii_active and not sell_ii_active:
                                        if kijun_cross_up:  # ✅ **Only Kijun UP Cross**
                                            df.loc[i, "Wealth Signal"] = "Wealth Buy II"
                                            buy_ii_active = True
                                            above_kijun = True
                                        elif kijun_cross_down:  # ✅ **Only Kijun DOWN Cross**
                                            df.loc[i, "Wealth Signal"] = "Wealth Sell II"
                                            sell_ii_active = True
                                            below_kijun = True

                                    # ✅ **Handle second Kijun cross (Buy III / Sell III)**
                                    elif buy_ii_active:
                                        if kijun_cross_down:  # Second confirmation **ONLY Kijun**
                                            df.loc[i, "Wealth Signal"] = "Wealth Sell III"
                                            volume_spike_active = False  # Reset everything
                                            buy_ii_active = False
                                            sell_ii_active = False
                                            above_kijun = False  # Reset
                                    elif sell_ii_active:
                                        if kijun_cross_up:  # Second confirmation **ONLY Kijun**
                                            df.loc[i, "Wealth Signal"] = "Wealth Buy III"
                                            volume_spike_active = False  # Reset everything
                                            buy_ii_active = False
                                            sell_ii_active = False
                                            below_kijun = False  # Reset

                                    # ✅ **NEW: Handle Wealth Buy IV (Strength Confirmation)**
                                    elif above_kijun and df.loc[i, "RVOL_5"] > 1.8:
                                        df.loc[i, "Wealth Signal"] = "Wealth Buy IV (Strength Continuation)"
                                        above_kijun = False  # Prevent further signals

                                    # ✅ **NEW: Handle Wealth Sell IV (Continuation Below Kijun)**
                                    elif below_kijun and df.loc[i, "RVOL_5"] > 1.8:
                                        df.loc[i, "Wealth Signal"] = "Wealth Sell IV (Downtrend Strength)"
                                        below_kijun = False  # Prevent further signals

                            return df








                intraday = detect_wealth_signals(intraday)




                def generate_market_snapshot(df, current_time, current_price, prev_close, symbol):
                    """
                    Generates a concise market snapshot:
                    - Time and current price
                    - Opening price & where it stands now
                    - F% change in raw dollars
                    - Price position relative to Kijun and Bollinger Mid
                    - Latest Buy/Sell Signal
                    """

                    # Convert time to 12-hour format (e.g., "03:55 PM")
                    current_time_str = pd.to_datetime(current_time).strftime("%I:%M %p")

                    # Get today's opening price
                    open_price = df["Open"].iloc[0]

                    # Calculate today's price changes
                    price_change = current_price - prev_close
                    f_percent_change = (price_change / prev_close) * 10000  # F%

                    # Identify price position relative to Kijun and Bollinger Middle
                    last_kijun = df["Kijun_sen"].iloc[-1]
                    last_mid_band = df["F% MA"].iloc[-1]

                    position_kijun = "above Kijun" if current_price > last_kijun else "below Kijun"
                    position_mid = "above Mid Band" if current_price > last_mid_band else "below Mid Band"

                    # Get the latest Buy/Sell signal
                    latest_signal = df.loc[df["Wealth Signal"] != "", ["Wealth Signal"]].tail(1)
                    signal_text = latest_signal["Wealth Signal"].values[0] if not latest_signal.empty else "No Signal"

                    # Construct the message
                    snapshot = (
                        f"📌 {current_time_str} – **{symbol}** is trading at **${current_price:.2f}**\n\n"
                        f"• Opened at **${open_price:.2f}** and is now sitting at **${current_price:.2f}**\n"
                        f"• F% Change: **{f_percent_change:.0f} F%** (${price_change:.2f})\n"
                        f"• Price is **{position_kijun}** & **{position_mid}**\n"
                        f"• **Latest Signal**: {signal_text}\n"
                    )

                    return snapshot





                if not intraday.empty:
                    current_time = intraday["Time"].iloc[-1]
                    current_price = intraday["Close"].iloc[-1]
                    st.markdown(generate_market_snapshot(intraday, current_time, current_price, prev_close, symbol=t))
                else:
                    st.warning(f"No intraday data available for {t}.")

                def detect_kijun_f_cross(df):
                    """
                    Detects when F% crosses above or below Kijun_F%.
                    - "Buy Kijun Cross" → F_numeric crosses above Kijun_F
                    - "Sell Kijun Cross" → F_numeric crosses below Kijun_F
                    """
                    df["Kijun_F_Cross"] = ""

                    for i in range(1, len(df)):
                        prev_f = df.loc[i - 1, "F_numeric"]
                        prev_kijun = df.loc[i - 1, "Kijun_F"]
                        curr_f = df.loc[i, "F_numeric"]
                        curr_kijun = df.loc[i, "Kijun_F"]

                        # Bullish Cross (Buy Signal)
                        if prev_f < prev_kijun and curr_f >= curr_kijun:
                            df.loc[i, "Kijun_F_Cross"] = "Buy Kijun Cross"

                        # Bearish Cross (Sell Signal)
                        elif prev_f > prev_kijun and curr_f <= curr_kijun:
                            df.loc[i, "Kijun_F_Cross"] = "Sell Kijun Cross"

                    return df

                # Apply function to detect Kijun F% crosses
                intraday = detect_kijun_f_cross(intraday)


                def detect_rsi_vs_spike(df):
                    """
                    Checks if RSI crossed before (leading) or after (confirmation) the volume spike.
                    - If RSI crosses UP before → RSI Leading: Bullish
                    - If RSI crosses DOWN before → RSI Leading: Bearish
                    - If RSI crosses UP after → RSI Confirmed: Bullish
                    - If RSI crosses DOWN after → RSI Confirmed: Bearish
                    - Only triggers if Wealth Buy II / III or Wealth Sell II / III was activated via **Kijun F% Cross**
                    """
                    df["RSI Prediction Raw"] = ""  # Store raw predictions
                    df["RSI Prediction Alert"] = ""  # Filtered for Wealth II/III only

                    lookback_window = 5   # Bars before the spike to check for a leading signal
                    lookahead_window = 5  # Bars after the spike to check for confirmation

                    for i in range(len(df)):
                        # Check if this row has a volume spike (Wealth Buy I / Sell I)
                        if df.loc[i, "RVOL_5"] > 1.8:
                            start_idx = max(0, i - lookback_window)
                            end_idx = min(len(df) - 1, i + lookahead_window)

                            # Look at RSI cross before and after the volume spike
                            rsi_before_spike = df.loc[start_idx:i-1, "RSI_Cross"]
                            rsi_after_spike = df.loc[i+1:end_idx, "RSI_Cross"]

                            # Before the spike → Leading indicator
                            if "Up" in rsi_before_spike.values:
                                df.loc[i, "RSI Prediction Raw"] = "RSI Leading: Bullish"
                            elif "Down" in rsi_before_spike.values:
                                df.loc[i, "RSI Prediction Raw"] = "RSI Leading: Bearish"

                            # After the spike → Confirmation indicator
                            elif "Up" in rsi_after_spike.values:
                                df.loc[i, "RSI Prediction Raw"] = "RSI Confirmed: Bullish"
                            elif "Down" in rsi_after_spike.values:
                                df.loc[i, "RSI Prediction Raw"] = "RSI Confirmed: Bearish"

                    # **Step 2: Filter Predictions to Only Show on Wealth II/III via Kijun F%**
                    df["RSI Prediction Alert"] = df.apply(
                        lambda row: row["RSI Prediction Raw"] if (
                            ("Wealth Buy II" in row["Wealth Signal"] or "Wealth Sell II" in row["Wealth Signal"] or
                            "Wealth Buy III" in row["Wealth Signal"] or "Wealth Sell III" in row["Wealth Signal"]) and
                            ("POK" in row["Alert_Kijun"] or "PUK" in row["Alert_Kijun"])  # Strictly Kijun Cross Only
                        ) else "",
                        axis=1
                    )

                    return df




                # Apply to intraday dataframe
                intraday = detect_rsi_vs_spike(intraday)


                import numpy as np
                import pandas as pd

                def calculate_f_dmi(df, period=14):
                    """
                    Computes +DI, -DI, and ADX for F% instead of price.
                    - Uses the correct True Range logic for ATR calculation.
                    - Ensures +DM and -DM use absolute differences correctly.
                    """
                    # Compute F% movement between bars
                    df["F_Diff"] = df["F_numeric"].diff()

                    # Compute True Range for F% (ATR Equivalent)
                    df["TR_F%"] = np.abs(df["F_numeric"].diff())

                    # Compute Directional Movement
                    df["+DM"] = np.where(df["F_Diff"] > 0, df["F_Diff"], 0)
                    df["-DM"] = np.where(df["F_Diff"] < 0, -df["F_Diff"], 0)

                    # Ensure no double-counting
                    df["+DM"] = np.where(df["+DM"] > df["-DM"], df["+DM"], 0)
                    df["-DM"] = np.where(df["-DM"] > df["+DM"], df["-DM"], 0)

                    # Smooth using Wilder's Moving Average (EMA Approximation)
                    df["+DM_Smoothed"] = df["+DM"].rolling(window=period, min_periods=1).mean()
                    df["-DM_Smoothed"] = df["-DM"].rolling(window=period, min_periods=1).mean()
                    df["ATR_F%"] = df["TR_F%"].rolling(window=period, min_periods=1).mean()

                    # Compute Directional Indicators (Avoid divide-by-zero)
                    df["+DI_F%"] = (df["+DM_Smoothed"] / df["ATR_F%"]) * 100
                    df["-DI_F%"] = (df["-DM_Smoothed"] / df["ATR_F%"]) * 100

                    # Handle potential NaN or infinite values
                    df["+DI_F%"] = df["+DI_F%"].replace([np.inf, -np.inf], np.nan).fillna(0)
                    df["-DI_F%"] = df["-DI_F%"].replace([np.inf, -np.inf], np.nan).fillna(0)

                    # Compute DX (Directional Movement Index)
                    df["DX_F%"] = np.abs((df["+DI_F%"] - df["-DI_F%"]) / (df["+DI_F%"] + df["-DI_F%"])) * 100
                    df["ADX_F%"] = df["DX_F%"].rolling(window=period, min_periods=1).mean()

                    return df

                # Apply the fixed DMI Calculation
                intraday = calculate_f_dmi(intraday, period=14)

                def detect_dmi_cross(df):
                    """
                    Detects crossovers between +DI_F% and -DI_F%.
                    - "Up" when +DI crosses above -DI (bullish).
                    - "Down" when +DI crosses below -DI (bearish).
                    """
                    df["DMI_Cross"] = ""

                    for i in range(1, len(df)):
                        prev_di_plus, prev_di_minus = df.loc[i - 1, "+DI_F%"], df.loc[i - 1, "-DI_F%"]
                        curr_di_plus, curr_di_minus = df.loc[i, "+DI_F%"], df.loc[i, "-DI_F%"]

                        # Bullish Crossover (👆)
                        if prev_di_plus < prev_di_minus and curr_di_plus >= curr_di_minus:
                            df.loc[i, "DMI_Cross"] = "👆"

                        # Bearish Crossover (👇)
                        elif prev_di_plus > prev_di_minus and curr_di_plus <= curr_di_minus:
                            df.loc[i, "DMI_Cross"] = "👇"

                    return df

                # Apply the detection function
                intraday = detect_dmi_cross(intraday)

                def calculate_f_tenkan(df, period=9):
                    """
                    Computes the F% version of Tenkan-sen (Conversion Line).
                    Formula: (Tenkan-sen - Prev Close) / Prev Close * 10000
                    """
                    highest_high = df["High"].rolling(window=period, min_periods=1).max()
                    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()
                    df["Tenkan_sen"] = (highest_high + lowest_low) / 2

                    if "Prev_Close" in df.columns:
                        df["F% Tenkan"] = ((df["Tenkan_sen"] - df["Prev_Close"]) / df["Prev_Close"]) * 10000
                    else:
                        df["F% Tenkan"] = 0  # Fallback in case Prev_Close is missing

                    return df

                # Apply to intraday dataset
                intraday = calculate_f_tenkan(intraday, period=9)

                def detect_f_tenkan_cross(df):
                    """
                    Detects F% Tenkan crosses over F% Kijun.
                    - Returns 'up' if F% Tenkan crosses above F% Kijun
                    - Returns 'down' if F% Tenkan crosses below F% Kijun
                    """
                    df["F% Tenkan Cross"] = ""

                    for i in range(1, len(df)):
                        prev_tenkan = df.loc[i - 1, "F% Tenkan"]
                        prev_kijun = df.loc[i - 1, "Kijun_F"]
                        curr_tenkan = df.loc[i, "F% Tenkan"]
                        curr_kijun = df.loc[i, "Kijun_F"]

                        if prev_tenkan < prev_kijun and curr_tenkan >= curr_kijun:
                            df.loc[i, "F% Tenkan Cross"] = "Up"
                        elif prev_tenkan > prev_kijun and curr_tenkan <= curr_kijun:
                            df.loc[i, "F% Tenkan Cross"] = "Down"

                    return df

                # Apply crossover detection
                intraday = detect_f_tenkan_cross(intraday)

                def track_ll_hh_streaks(df, min_streak=10):
                    """
                    Tracks consecutive occurrences of Low of Day (LL) and High of Day (HH).
                    - If LL or HH persists for at least `min_streak` rows, it gets labeled as "LL + X" or "HH + X".
                    """
                    df["LL_Streak"] = ""
                    df["HH_Streak"] = ""

                    # Track streaks
                    low_streak, high_streak = 0, 0

                    for i in range(len(df)):
                        if df.loc[i, "Low of Day"] != "":
                            low_streak += 1
                        else:
                            low_streak = 0

                        if df.loc[i, "High of Day"] != "":
                            high_streak += 1
                        else:
                            high_streak = 0

                        # Assign labels only if streaks exceed the minimum threshold
                        if low_streak >= min_streak:
                            df.loc[i, "LL_Streak"] = f"LL +{low_streak}"
                        if high_streak >= min_streak:
                            df.loc[i, "HH_Streak"] = f"HH +{high_streak}"

                    return df

                def calculate_td_sequential(data):
                        """
                        Calculates TD Sequential buy/sell setups while avoiding ambiguous
                        boolean errors by using NumPy arrays for comparisons.
                        """

                        # Initialize columns
                        data['Buy Setup'] = np.nan
                        data['Sell Setup'] = np.nan

                        # Convert Close prices to a NumPy array for guaranteed scalar access
                        close_vals = data['Close'].values

                        # Arrays to track consecutive buy/sell counts
                        buy_count = np.zeros(len(data), dtype=np.int32)
                        sell_count = np.zeros(len(data), dtype=np.int32)

                        # Iterate through the rows
                        for i in range(len(data)):
                            # We need at least 4 prior bars to do the comparison
                            if i < 4:
                                continue

                            # Compare scalars from the NumPy array (guaranteed single float)
                            is_buy = (close_vals[i] < close_vals[i - 4])
                            is_sell = (close_vals[i] > close_vals[i - 4])

                            # Update consecutive counts
                            if is_buy:
                                buy_count[i] = buy_count[i - 1] + 1  # increment
                                sell_count[i] = 0                   # reset sell
                            else:
                                buy_count[i] = 0

                            if is_sell:
                                sell_count[i] = sell_count[i - 1] + 1  # increment
                                buy_count[i] = 0                       # reset buy
                            else:
                                sell_count[i] = 0

                            # Assign setup labels if the count is nonzero or completed
                            if buy_count[i] == 9:
                                data.at[data.index[i], 'Buy Setup'] = 'Buy Setup Completed'
                                buy_count[i] = 0  # reset after completion
                            elif buy_count[i] > 0:
                                data.at[data.index[i], 'Buy Setup'] = f'Buy Setup {buy_count[i]}'

                            if sell_count[i] == 9:
                                data.at[data.index[i], 'Sell Setup'] = 'Sell Setup Completed'
                                sell_count[i] = 0  # reset after completion
                            elif sell_count[i] > 0:
                                data.at[data.index[i], 'Sell Setup'] = f'Sell Setup {sell_count[i]}'

                        return data
                intraday = calculate_td_sequential(intraday)



                def calculate_td_countdown(data):
                    """
                    Calculates TD Sequential Countdown after a Buy or Sell Setup completion.
                    """

                    # Initialize Countdown columns
                    data['Buy Countdown'] = np.nan
                    data['Sell Countdown'] = np.nan

                    # Convert Close prices to NumPy array for fast comparisons
                    close_vals = data['Close'].values

                    # Initialize countdown arrays
                    buy_countdown = np.zeros(len(data), dtype=np.int32)
                    sell_countdown = np.zeros(len(data), dtype=np.int32)

                    # Iterate through the dataset
                    for i in range(len(data)):
                        if i < 2:  # Need at least 2 prior bars for comparison
                            continue

                        # Start Buy Countdown after Buy Setup Completion
                        if data.at[data.index[i], 'Buy Setup'] == 'Buy Setup Completed':
                            buy_countdown[i] = 1  # Start countdown

                        # Increment Buy Countdown if conditions are met
                        if buy_countdown[i - 1] > 0 and close_vals[i] < close_vals[i - 2]:
                            buy_countdown[i] = buy_countdown[i - 1] + 1
                            data.at[data.index[i], 'Buy Countdown'] = f'Buy Countdown {buy_countdown[i]}'
                            if buy_countdown[i] == 13:
                                data.at[data.index[i], 'Buy Countdown'] = 'Buy Countdown Completed'

                        # Start Sell Countdown after Sell Setup Completion
                        if data.at[data.index[i], 'Sell Setup'] == 'Sell Setup Completed':
                            sell_countdown[i] = 1  # Start countdown

                        # Increment Sell Countdown if conditions are met
                        if sell_countdown[i - 1] > 0 and close_vals[i] > close_vals[i - 2]:
                            sell_countdown[i] = sell_countdown[i - 1] + 1
                            data.at[data.index[i], 'Sell Countdown'] = f'Sell Countdown {sell_countdown[i]}'
                            if sell_countdown[i] == 13:
                                data.at[data.index[i], 'Sell Countdown'] = 'Sell Countdown Completed'

                    return data

                intraday = calculate_td_countdown(intraday)

                def calculate_tenkan_sen(df, period=9):
                    """
                    Computes Tenkan-sen for F% based on the midpoint of high/low over a rolling period.
                    """
                    highest_high = df["High"].rolling(window=period, min_periods=1).max()
                    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()
                    df["Tenkan_sen"] = (highest_high + lowest_low) / 2

                    # Convert to F% scale
                    df["Tenkan_F"] = ((df["Tenkan_sen"] - prev_close) / prev_close) * 10000
                    return df

                # Apply to intraday data
                intraday = calculate_tenkan_sen(intraday, period=9)


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

                def calculate_chikou_span(df, period=26):
                    """
                    Computes the Chikou Span (Lagging Span) for Ichimoku.
                    Chikou Span is the closing price shifted back by `period` bars.
                    """
                    df["Chikou_Span"] = df["Close"].shift(-period)  # Shift forward
                    return df

                # Apply Chikou Span calculation
                intraday = calculate_chikou_span(intraday, period=26)

                def calculate_kumo(df, period_a=26, period_b=52, shift=26):
                    """
                    Computes Senkou Span A and Senkou Span B for Ichimoku Cloud (Kumo).
                    - Senkou Span A = (Tenkan-Sen + Kijun-Sen) / 2, shifted forward
                    - Senkou Span B = (Highest High + Lowest Low) / 2 over 52 periods, shifted forward
                    """
                    df["Senkou_Span_A"] = ((df["Tenkan_sen"] + df["Kijun_sen"]) / 2).shift(shift)

                    highest_high = df["High"].rolling(window=period_b, min_periods=1).max()
                    lowest_low = df["Low"].rolling(window=period_b, min_periods=1).min()
                    df["Senkou_Span_B"] = ((highest_high + lowest_low) / 2).shift(shift)

                    return df

                # Apply Kumo (Cloud) calculations
                intraday = calculate_kumo(intraday)

                def calculate_td_pressure(data):


                        # 1) Compute the price range per bar.
                        #    Where the range is zero, we'll get division by zero — so we handle that by assigning NaN.
                        price_range = data['High'] - data['Low']

                        # 2) Compute the "pressure ratio" for each bar.
                        #    ratio = ((Close - Open) / price_range) * Volume
                        #    If price_range == 0, replace with NaN to avoid inf or division by zero.
                        ratio = (data['Close'] - data['Open']) / price_range * data['Volume']
                        ratio[price_range == 0] = np.nan  # Mark division-by-zero cases as NaN

                        # 3) Compute absolute price difference per bar
                        abs_diff = (data['Close'] - data['Open']).abs()

                        # 4) Sum over a rolling 5-bar window using .rolling(5).
                        #    - rolling_ratio_sum: Sum of the 5-bar pressure ratios
                        #    - rolling_abs_diff_sum: Sum of the 5-bar absolute price differences
                        #    - min_periods=5 ensures we only output a valid sum starting at the 5th bar
                        rolling_ratio_sum = ratio.rolling(5, min_periods=5).sum()
                        rolling_abs_diff_sum = abs_diff.rolling(5, min_periods=5).sum()

                        # 5) Compute the normalized TD Pressure:
                        #    TD Pressure = (sum_of_5_bar_ratios / sum_of_5_bar_abs_diff) / 100000
                        #    If rolling_abs_diff_sum is 0, the result will be NaN (safe handling).
                        data['TD Pressure'] = (rolling_ratio_sum / rolling_abs_diff_sum) / 100000
                        data['TD Pressure'] = data['TD Pressure'].fillna(0)  # Replace NaNs with 0 or another suitable value
                        return data

                intraday = calculate_td_pressure(intraday)

                def calculate_td_rei(data):
                    """
                    Calculates the TD Range Expansion Index (TD REI).
                    TD REI measures the strength of price expansion relative to its range over the last 5 bars.
                    """

                    # Initialize TD REI column
                    data['TD REI'] = np.nan

                    # Convert High and Low prices to NumPy arrays for faster calculations
                    high_vals = data['High'].values
                    low_vals = data['Low'].values

                    # Iterate through the dataset, starting from the 5th row
                    for i in range(5, len(data)):
                        # Step 1: Calculate numerator (high_diff + low_diff)
                        high_diff = high_vals[i] - high_vals[i - 2]  # Current high - high two bars ago
                        low_diff = low_vals[i] - low_vals[i - 2]    # Current low - low two bars ago
                        numerator = high_diff + low_diff  # Sum of the differences

                        # Step 2: Calculate denominator (highest high - lowest low over the last 5 bars)
                        highest_high = np.max(high_vals[i - 4:i + 1])  # Highest high in the last 5 bars
                        lowest_low = np.min(low_vals[i - 4:i + 1])    # Lowest low in the last 5 bars
                        denominator = highest_high - lowest_low

                        # Step 3: Calculate TD REI, ensuring no division by zero
                        if denominator != 0:
                            td_rei_value = (numerator / denominator) * 100
                        else:
                            td_rei_value = 0  # Prevent division by zero

                        # **Fix for extreme values:** Ensure TD REI remains within [-100, 100]
                        td_rei_value = max(min(td_rei_value, 100), -100)

                        # Assign calculated TD REI to the DataFrame
                        data.at[data.index[i], 'TD REI'] = td_rei_value

                    return data
                intraday = calculate_td_rei(intraday)  # Compute TD REI


                import numpy as np

                def calculate_td_poq(data):
                    """
                    Computes TD POQ signals based on TD REI conditions and price action breakouts.
                    - Scenario 1 & 3: Buy Calls
                    - Scenario 2 & 4: Buy Puts
                    """
                    data["TD_POQ"] = np.nan  # Initialize column

                    for i in range(6, len(data)):  # Start at row 6 to account for prior bars
                        td_rei = data.at[data.index[i], "TD REI"]
                        close_1, close_2 = data.at[data.index[i - 1], "Close"], data.at[data.index[i - 2], "Close"]
                        open_i, high_i, low_i, close_i = data.at[data.index[i], "Open"], data.at[data.index[i], "High"], data.at[data.index[i], "Low"], data.at[data.index[i], "Close"]
                        high_1, low_1, high_2, low_2 = data.at[data.index[i - 1], "High"], data.at[data.index[i - 1], "Low"], data.at[data.index[i - 2], "High"], data.at[data.index[i - 2], "Low"]

                        # Scenario 1: Qualified TD POQ Upside Breakout — Buy Call
                        if (
                            not np.isnan(td_rei) and td_rei < -45 and  # TD REI in oversold condition
                            close_1 > close_2 and  # Previous close > close two bars ago
                            open_i <= high_1 and  # Current open <= previous high
                            high_i > high_1  # Current high > previous high
                        ):
                            data.at[data.index[i], "TD_POQ"] = "Scenario 1: Buy Call"

                        # Scenario 2: Qualified TD POQ Downside Breakout — Buy Put
                        elif (
                            not np.isnan(td_rei) and td_rei > 45 and  # TD REI in overbought condition
                            close_1 < close_2 and  # Previous close < close two bars ago
                            open_i >= low_1 and  # Current open >= previous low
                            low_i < low_1  # Current low < previous low
                        ):
                            data.at[data.index[i], "TD_POQ"] = "Scenario 2: Buy Put"

                        # Scenario 3: Alternative TD POQ Upside Breakout — Buy Call
                        elif (
                            not np.isnan(td_rei) and td_rei < -45 and  # TD REI in mild oversold condition
                            close_1 > close_2 and  # Previous close > close two bars ago
                            open_i > high_1 and open_i < high_2 and  # Current open > previous high but < high two bars ago
                            high_i > high_2 and  # Current high > high two bars ago
                            close_i > open_i  # Current close > current open
                        ):
                            data.at[data.index[i], "TD_POQ"] = "Scenario 3: Buy Call"

                        # Scenario 4: Alternative TD POQ Downside Breakout — Buy Put
                        elif (
                            not np.isnan(td_rei) and td_rei > 45 and  # TD REI in mild overbought condition
                            close_1 < close_2 and  # Previous close < close two bars ago
                            open_i < low_1 and open_i > low_2 and  # Current open < previous low but > low two bars ago
                            low_i < low_2 and  # Current low < low two bars ago
                            close_i < open_i  # Current close < current open
                        ):
                            data.at[data.index[i], "TD_POQ"] = "Scenario 4: Buy Put"

                    return data

                # Apply function to intraday DataFrame
                intraday = calculate_td_poq(intraday)



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

                def detect_td_rei_crossovers(data):
                    """
                    Identifies TD REI crossovers:
                    - 🧨 (Firecracker) when TD REI crosses from + to -
                    - 🔑 (Key) when TD REI crosses from - to +
                    """
                    data["TD REI Crossover"] = np.nan  # Initialize crossover column

                    for i in range(1, len(data)):  # Start from second row
                        prev_rei = data.at[data.index[i - 1], "TD REI"]
                        curr_rei = data.at[data.index[i], "TD REI"]

                        if pd.notna(prev_rei) and pd.notna(curr_rei):
                            # **From + to - (Bearish) → Firecracker 🧨**
                            if prev_rei > 0 and curr_rei < 0:
                                data.at[data.index[i], "TD REI Crossover"] = "🧨"

                            # **From - to + (Bullish) → Key 🔑**
                            elif prev_rei < 0 and curr_rei > 0:
                                data.at[data.index[i], "TD REI Crossover"] = "🔑"

                    return data
                intraday = detect_td_rei_crossovers(intraday)  # Detect TD REI crossovers


                def calculate_td_poq(data):
                    data['TD POQ'] = np.nan  # Use NaN for consistency

                    for i in range(5, len(data)):  # Start from the 6th row for sufficient prior data
                        if pd.notna(data['TD REI'].iloc[i]):  # Ensure TD REI is not NaN

                            # Buy POQ Logic: Qualified Upside Breakout
                            if (data['TD REI'].iloc[i] < -45 and
                                data['Close'].iloc[i - 1] > data['Close'].iloc[i - 2] and
                                data['Open'].iloc[i] <= data['High'].iloc[i - 1] and
                                data['High'].iloc[i] > data['High'].iloc[i - 1]):
                                data.loc[data.index[i], 'TD POQ'] = 'Buy POQ'

                            # Sell POQ Logic: Qualified Downside Breakout
                            elif (data['TD REI'].iloc[i] > 45 and
                                data['Close'].iloc[i - 1] < data['Close'].iloc[i - 2] and
                                data['Open'].iloc[i] >= data['Low'].iloc[i - 1] and
                                data['Low'].iloc[i] < data['Low'].iloc[i - 1]):
                                data.loc[data.index[i], 'TD POQ'] = 'Sell POQ'

                    return data
                intraday = calculate_td_poq(intraday)  # Detect TD REI crossovers



                def calculate_vas(data, signal_col="F_numeric", volatility_col="ATR", period=14):
                    """
                    Computes Volatility Adjusted Score (VAS) using the given signal and volatility measure.
                    Default: F% as signal, ATR as volatility.
                    """
                    if volatility_col == "ATR":
                        data["ATR"] = data["High"].rolling(window=period).max() - data["Low"].rolling(window=period).min()

                    elif volatility_col == "MAD":
                        data["MAD"] = data["Close"].rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)

                    elif volatility_col == "STD":
                        data["STD"] = data["Close"].rolling(window=period).std()

                    # Compute VAS using selected volatility measure
                    selected_vol = data[volatility_col].fillna(method="bfill")  # Avoid NaN errors
                    data["VAS"] = data[signal_col] / selected_vol
                    return data

                # Apply function to intraday data (defaulting to ATR)
                intraday = calculate_vas(intraday, signal_col="F_numeric", volatility_col="ATR", period=14)
                def calculate_stop_loss(df, prev_low, prev_high):
                    """
                    Calculates stop loss levels:
                    - **CALL Stop Loss**: 1/8 point **below** previous day's Low.
                    - **PUT Stop Loss**: 1/8 point **above** previous day's High.
                    """
                    df["Call Stop Loss"] = prev_low - 0.125  # ✅ Corrected for CALL stop loss
                    df["Put Stop Loss"] = prev_high + 0.125  # ✅ Corrected for PUT stop loss
                    return df

                # Apply the function to intraday data
                intraday = calculate_stop_loss(intraday, prev_low, prev_high)


                intraday["Opening Price Signal"] = intraday["Close"] - prev_open
                intraday["Net Price"] = intraday["Close"] - prev_close

                # Detect Net Price direction changes
                intraday["Net Price Direction"] = ""
                net_price_shifted = intraday["Net Price"].shift(1)

                # 🦅 Eagle: Net Price flips from negative to positive
                intraday.loc[(net_price_shifted < 0) & (intraday["Net Price"] >= 0), "Net Price Direction"] = "🦅"

                # 🦉 Owl: Net Price flips from positive to negative
                intraday.loc[(net_price_shifted > 0) & (intraday["Net Price"] <= 0), "Net Price Direction"] = "🦉"


                # Step 1: Calculate OBV
                def calculate_obv(df):
                    df["OBV"] = 0  # Initialize OBV column
                    df["OBV"] = np.where(df["Close"] > df["Close"].shift(1), df["Volume"],
                                        np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0)).cumsum()

                    # Normalize OBV to be in hundreds instead of thousands
                    df["OBV"] = df["OBV"] / 10000

                    return df

                # Step 2: Detect OBV Crossovers
                def detect_obv_crossovers(df):
                    df["OBV_Crossover"] = ""

                    for i in range(1, len(df)):
                        prev_obv = df.loc[i - 1, "OBV"]
                        curr_obv = df.loc[i, "OBV"]

                        if prev_obv < 0 and curr_obv >= 0:
                            df.loc[i, "OBV_Crossover"] = "🔈"  # Speaker (Bullish Volume Shift)
                        elif prev_obv > 0 and curr_obv <= 0:
                            df.loc[i, "OBV_Crossover"] = "🔇"  # Muted Speaker (Bearish Volume Weakness)

                    return df

                # Apply OBV & Crossover Detection
                intraday = calculate_obv(intraday)
                intraday = detect_obv_crossovers(intraday)


                # Detect Tenkan-Kijun Crosses
                intraday["Tenkan_Kijun_Cross"] = ""

                for i in range(1, len(intraday)):
                    prev_tenkan, prev_kijun = intraday.loc[i - 1, "Tenkan_F"], intraday.loc[i - 1, "Kijun_F"]
                    curr_tenkan, curr_kijun = intraday.loc[i, "Tenkan_F"], intraday.loc[i, "Kijun_F"]

                    # Bullish Cross (🌞)
                    if prev_tenkan < prev_kijun and curr_tenkan >= curr_kijun:
                        intraday.loc[i, "Tenkan_Kijun_Cross"] = "🌞"

                    # Bearish Cross (🌙)
                    elif prev_tenkan > prev_kijun and curr_tenkan <= curr_kijun:
                        intraday.loc[i, "Tenkan_Kijun_Cross"] = "🌙"




                def detect_vas_transitions(data):
                    """
                    Identifies transitions in VAS:
                    - 💎 (Gemstone) when VAS transitions from `-` to `+`
                    - 🎈 (Balloon) when VAS transitions from `+` to `-`
                    """
                    data["VAS Transition"] = None  # Initialize column

                    for i in range(1, len(data)):  # Start from second row
                        prev_vas = data.at[data.index[i - 1], "VAS"]
                        curr_vas = data.at[data.index[i], "VAS"]

                        if pd.notna(prev_vas) and pd.notna(curr_vas):
                            # **From - to + (Bullish) → Gemstone 💎**
                            if prev_vas < 0 and curr_vas > 0:
                                data.at[data.index[i], "VAS Transition"] = "💎"

                            # **From + to - (Bearish) → Balloon 🎈**
                            elif prev_vas > 0 and curr_vas < 0:
                                data.at[data.index[i], "VAS Transition"] = "🎈"

                    return data

                # Apply function to intraday DataFrame
                intraday = detect_vas_transitions(intraday)
                # Convert previous day levels to F% scale
                intraday["Yesterday Open F%"] = ((prev_open - prev_close) / prev_close) * 10000
                intraday["Yesterday High F%"] = ((prev_high - prev_close) / prev_close) * 10000
                intraday["Yesterday Low F%"] = ((prev_low - prev_close) / prev_close) * 10000
                intraday["Yesterday Close F%"] = ((prev_close - prev_close) / prev_close) * 10000  # Always 0


                # Function to detect OPS transitions
                def detect_ops_transitions(df):
                    df["OPS Transition"] = ""

                    for i in range(1, len(df)):  # Start from second row to compare with previous
                        prev_ops = df.loc[i - 1, "Opening Price Signal"]
                        curr_ops = df.loc[i, "Opening Price Signal"]

                        if prev_ops > 0 and curr_ops <= 0:  # Bearish transition
                            df.loc[i, "OPS Transition"] = "🐻"
                        elif prev_ops < 0 and curr_ops >= 0:  # Bullish transition
                            df.loc[i, "OPS Transition"] = "🐼"

                    return df

                # Apply OPS transition detection
                intraday = detect_ops_transitions(intraday)



                intraday["F% MA (8)"] = intraday["F_numeric"].rolling(window=8, min_periods=1).mean()

                # Ensure F_numeric exists in intraday
                if "F_numeric" in intraday.columns:
                    # Compute High & Low of Day in F% scale
                    intraday["F% High"] = intraday["F_numeric"].cummax()  # Rolling highest F%
                    intraday["F% Low"] = intraday["F_numeric"].cummin()   # Rolling lowest F%

                    # Calculate Bounce (Recovery from Lows)
                    intraday["Bounce"] = ((intraday["F_numeric"] - intraday["F% Low"]) / intraday["F% Low"].abs()) * 100

                    # Calculate Retrace (Pullback from Highs)
                    intraday["Retrace"] = ((intraday["F% High"] - intraday["F_numeric"]) / intraday["F% High"].abs()) * 100

                    # Clean up: Replace infinities and NaN values
                    intraday["Bounce"] = intraday["Bounce"].replace([np.inf, -np.inf], 0).fillna(0).round(2)
                    intraday["Retrace"] = intraday["Retrace"].replace([np.inf, -np.inf], 0).fillna(0).round(2)





                # Identify the first OPS value at 9:30 AM
                first_ops_row = intraday[intraday["Time"] == "09:30 AM"]
                if not first_ops_row.empty:
                    first_ops_value = first_ops_row["Opening Price Signal"].iloc[0]
                    first_ops_time = first_ops_row["Time"].iloc[0]

                    # Determine if OPS started positive or negative
                    ops_label = "OPS 🔼" if first_ops_value > 0 else "OPS 🔽"
                    ops_color = "green" if first_ops_value > 0 else "red"



                # Apply the function after computing F_numeric
                intraday = calculate_f_trig(intraday)
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

# 3) Now that intraday is fully processed,
                #    let's get the final row (which has all new columns).
                # =================================
                # AFTER all pipeline transformations
                # =================================
                # Fetch last 3 bars
                recent_rows = intraday.tail(3)







                if gap_alert:
                    st.warning(gap_alert)



                st.subheader(f"Live Market Snapshot - Last 5 Rows for {t}")
                snapshot_cols = ["Time", "Close", "RVOL_5","F%","40ish", "Kijun_F_Cross", "Alert_Kijun","Alert_Mid","CTOD Alert","Wealth Signal"]  # Adjust as needed
                snapshot_df = intraday[snapshot_cols].tail(1)

                st.dataframe(snapshot_df, use_container_width=True)








                ticker_tabs = st.tabs(["Interactive F% & Momentum", "Intraday Data Table"])

                with ticker_tabs[0]:
                    # -- Create Subplots: Row1=F%, Row2=Momentum
                    fig = make_subplots(
                        rows=1,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[30]  # Adjust row heights to your preference
                    )

                    # (A) F% over time as lines+markers
                    # ---------------------------------
                    max_abs_val = intraday["F_numeric"].abs().max()
                    scatter_f = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F_numeric"],
                        mode="lines+markers",
                        customdata=intraday["Close"],

                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Close: $%{customdata:.2f}<extra></extra>",

                        name="F% (scaled)",

                    )
                    fig.add_trace(scatter_f, row=1, col=1)

                    # (A.1) 40ish Reversal (star markers)
                    mask_40ish = intraday["40ish"] != ""
                    scatter_40ish = go.Scatter(
                        x=intraday.loc[mask_40ish, "Time"],
                        y=intraday.loc[mask_40ish, "F_numeric"] +22,
                        mode="markers",
                        marker_symbol="star",
                        marker_size=18,
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





                    # # Create mask for high velocity change points
                    # mask_velocity_spike = intraday["Velocity_Spike"]

                    # # Maroon Arrows for F% Velocity Spikes
                    # scatter_velocity_spikes = go.Scatter(
                    #     x=intraday.loc[mask_velocity_spike, "Time"],
                    #     y=intraday.loc[mask_velocity_spike, "F_numeric"],
                    #     mode="markers",
                    #     marker=dict(symbol="arrow-bar-up", size=14, color="#0ff"),
                    #     name="F% Velocity Spike",
                    #     text="Velocity Surge",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    # )



                    # # Add to the F% plot (Row 1)
                    # fig.add_trace(scatter_velocity_spikes, row=1, col=1)


                    # theta_std = intraday["F% Theta"].std()
                    # threshold = 1.8 * theta_std  # Adjust multiplier if needed

                    # mask_theta_up = intraday["F% Theta"] > threshold  # Very high positive spikes
                    # mask_theta_down = intraday["F% Theta"] < -threshold  # Very low negative spikes

                    # scatter_theta_up = go.Scatter(
                    #     x=intraday.loc[mask_theta_up, "Time"],
                    #     y=intraday.loc[mask_theta_up, "F_numeric"]+ 4,
                    #     mode="text",
                    #     text="*",
                    #     textposition="top center",
                    #     textfont=dict(size=30, color="green"),
                    #     name="Extreme Theta UP",
                    #     showlegend=True,
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Theta Spike"
                    # )

                    # scatter_theta_down = go.Scatter(
                    #     x=intraday.loc[mask_theta_down, "Time"],
                    #     y=intraday.loc[mask_theta_down, "F_numeric"] - 4,
                    #     mode="text",
                    #     text="*",
                    #     textposition="bottom center",
                    #     textfont=dict(size=30, color="red"),
                    #     name="Extreme Theta DOWN",
                    #     showlegend=True,
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Theta Spike"

                    # )

                    # # Ensure they are added to the figure
                    # fig.add_trace(scatter_theta_up, row=1, col=1)
                    # fig.add_trace(scatter_theta_down, row=1, col=1)

                    kijun_line = go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Kijun_F"],
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Kijun (F% scale)"
                )
                    fig.add_trace(kijun_line, row=1, col=1)

           # 🟢 Wealth Buy & Sell Signals (Plotted with RVOL Color Code)
                    # -----------------------------------------------------------
                    mask_buy_signal = intraday["Wealth Signal"].str.contains("Wealth Buy", na=False)
                    mask_sell_signal = intraday["Wealth Signal"].str.contains("Wealth Sell", na=False)

                    # Determine Wealth Signal colors based on RVOL levels
                    color_map = {
                        "red": intraday["RVOL_5"] > 1.8,
                        "yellow": (intraday["RVOL_5"] >= 1.5) & (intraday["RVOL_5"] < 1.8),
                        "pink": (intraday["RVOL_5"] >= 1.2) & (intraday["RVOL_5"] < 1.5),
                    }

                    def assign_color(row):
                        for color, condition in color_map.items():
                            if condition.loc[row.name]:  # row.name gives the index
                                return color
                        return "black"  # Default if no condition is met

                    # Apply colors to Buy and Sell signals
                    intraday["Wealth Color"] = intraday.apply(assign_color, axis=1)
                    # Wealth Buy Signal (Always Green)
                    scatter_buy_signal = go.Scatter(
                        x=intraday.loc[mask_buy_signal, "Time"],
                        y=intraday.loc[mask_buy_signal, "F_numeric"],
                        mode="markers",
                        marker=dict(symbol="x-thin-open", size=55, color="green"),  # Force Green
                        name="Wealth Buy Signal",
                        text=intraday.loc[mask_buy_signal, "Wealth Signal"],
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Wealth Sell Signal (Always Red)
                    scatter_sell_signal = go.Scatter(
                        x=intraday.loc[mask_sell_signal, "Time"],
                        y=intraday.loc[mask_sell_signal, "F_numeric"],
                        mode="markers",
                        marker=dict(symbol="x-thin-open", size=55, color="red"),  # Force Red
                        name="Wealth Sell Signal",
                        text=intraday.loc[mask_sell_signal, "Wealth Signal"],
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    fig.add_trace(scatter_buy_signal, row=1, col=1)
                    fig.add_trace(scatter_sell_signal, row=1, col=1)






                    # (C) CTOD Buy/Sell Triggers (Red & Green Dots)
                    # ----------------------------------------------
                    mask_ctod_buy = intraday["CTOD Alert"] == "Buy"
                    mask_ctod_sell = intraday["CTOD Alert"] == "Sell"

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
                        row=1, col=1,
                        annotation_text="0 (Momentum)",
                        annotation_position="top left"
                    )






                    # Add alert text for middle band crossings
                    alert_mid_trace = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F_numeric"],
                        mode="text",
                        text=intraday["Alert_Mid"],
                        textposition="top center",
                        textfont=dict(color="orange", size=9, family="Arial "),
                        name="Alert Mid"
                    )
                    fig.add_trace(alert_mid_trace, row=1, col=1)

                    # Add alert text for Kijun crossings
                    alert_kijun_trace = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F_numeric"],
                        mode="text",
                        text=intraday["Alert_Kijun"],
                        textposition="bottom center",
                        textfont=dict(color="orange", size=9, family="Arial "),
                        name="Alert Kijun"
                    )
                    fig.add_trace(alert_kijun_trace, row=1, col=1)


                                    # Mask for different RVOL thresholds
                    mask_rvol_extreme = intraday["RVOL_5"] > 1.8
                    mask_rvol_strong = (intraday["RVOL_5"] >= 1.5) & (intraday["RVOL_5"] < 1.8)
                    mask_rvol_moderate = (intraday["RVOL_5"] >= 1.2) & (intraday["RVOL_5"] < 1.5)

                    # Scatter plot for extreme volume spikes (red triangle)
                    scatter_rvol_extreme = go.Scatter(
                        x=intraday.loc[mask_rvol_extreme, "Time"],
                        y=intraday.loc[mask_rvol_extreme, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=18, color="red"),
                        name="RVOL > 1.8 (Extreme Surge)",
                        text="Extreme Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Scatter plot for strong volume spikes (orange triangle)
                    scatter_rvol_strong = go.Scatter(
                        x=intraday.loc[mask_rvol_strong, "Time"],
                        y=intraday.loc[mask_rvol_strong, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=16, color="orange"),
                        name="RVOL 1.5-1.79 (Strong Surge)",
                        text="Strong Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Scatter plot for moderate volume spikes (pink triangle)
                    scatter_rvol_moderate = go.Scatter(
                        x=intraday.loc[mask_rvol_moderate, "Time"],
                        y=intraday.loc[mask_rvol_moderate, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=14, color="pink"),
                        name="RVOL 1.2-1.49 (Moderate Surge)",
                        text="Moderate Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Add to the F% plot (Row 1)
                    fig.add_trace(scatter_rvol_extreme, row=1, col=1)
                    fig.add_trace(scatter_rvol_strong, row=1, col=1)
                    fig.add_trace(scatter_rvol_moderate, row=1, col=1)

                    # (B) Upper Band
                    upper_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% Upper"],
                        mode="lines",
                        line=dict(dash="dot", color="grey"),
                        name="Upper Band"
                    )

                    # (C) Lower Band
                    lower_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% Lower"],
                        mode="lines",
                        line=dict(dash="dot", color="grey"),
                        name="Lower Band"
                    )

                    # (D) Moving Average (Middle Band)
                    middle_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% MA"],
                        mode="lines",
                        line=dict(color="white", dash="dash"),  # Set dash style
                        name="Middle Band (14-MA)"
                    )

                    # Add all traces

                    fig.add_trace(upper_band, row=1, col=1)
                    fig.add_trace(lower_band, row=1, col=1)
                    fig.add_trace(middle_band, row=1, col=1)



                    # 🟢 Step 1: Create Masks for F% Tenkan Crosses
                    mask_tenkan_up = intraday["F% Tenkan Cross"] == "Up"
                    mask_tenkan_down = intraday["F% Tenkan Cross"] == "Down"

                    # 🟩 Tenkan Cross Up (Green "T")
                    scatter_tenkan_up = go.Scatter(
                        x=intraday.loc[mask_tenkan_up, "Time"],
                        y=intraday.loc[mask_tenkan_up, "F_numeric"] + 5,  # Offset to avoid overlap
                        mode="text",
                        text="T",
                        textposition="top center",
                        textfont=dict(size=10, color="green"),
                        name="F% Tenkan Up",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Cross Up"
                    )

                    # 🟥 Tenkan Cross Down (Red "T")
                    scatter_tenkan_down = go.Scatter(
                        x=intraday.loc[mask_tenkan_down, "Time"],
                        y=intraday.loc[mask_tenkan_down, "F_numeric"] - 5,  # Offset to avoid overlap
                        mode="text",
                        text="T",
                        textposition="bottom center",
                        textfont=dict(size=10, color="red"),
                        name="F% Tenkan Down",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Cross Down"
                    )

                    # 🟢 Step 2: Add to the F% Plot (Row 1)
                    fig.add_trace(scatter_tenkan_up, row=1, col=1)  # Bullish Tenkan Cross
                    fig.add_trace(scatter_tenkan_down, row=1, col=1)  # Bearish Tenkan Cross

                    tenkan_line = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["Tenkan_F"],
                        mode="lines",
                        line=dict(color="red", width=2, dash="dot"),
                        name="Tenkan (F%)"
                    )
                    fig.add_trace(tenkan_line, row=1, col=1)

                    # # 🟢 Step 1: Create Masks for RSI Crosses
                    # mask_rsi_up = intraday["RSI_Cross"] == "Up"
                    # mask_rsi_down = intraday["RSI_Cross"] == "Down"

                    # # 🟩 Bullish RSI Cross (Lime Square) - Plotted in F% Chart
                    # scatter_rsi_cross_up = go.Scatter(
                    #     x=intraday.loc[mask_rsi_up, "Time"],
                    #     y=intraday.loc[mask_rsi_up, "F_numeric"] + 4,  # Offset to avoid overlap
                    #     mode="markers",
                    #     marker=dict(symbol="square", size=12, color="lime"),
                    #     name="RSI Up Cross"
                    # )

                    # # 🟪 Bearish RSI Cross (Pink Square) - Plotted in F% Chart
                    # scatter_rsi_cross_down = go.Scatter(
                    #     x=intraday.loc[mask_rsi_down, "Time"],
                    #     y=intraday.loc[mask_rsi_down, "F_numeric"] - 4,  # Offset to avoid overlap
                    #     mode="markers",
                    #     marker=dict(symbol="square", size=12, color="red"),
                    #     name="RSI Down Cross"
                    # )

                    # # 🟢 Step 2: Add to F% Plot (Row 1)
                    # fig.add_trace(scatter_rsi_cross_up, row=1, col=1)  # Bullish RSI Cross
                    # fig.add_trace(scatter_rsi_cross_down, row=1, col=1)  # Bearish RSI Cross



                    # 🟢 Bullish Kijun Cross (Green "K")
                    mask_kijun_buy = intraday["Kijun_F_Cross"] == "Buy Kijun Cross"
                    scatter_kijun_buy = go.Scatter(
                        x=intraday.loc[mask_kijun_buy, "Time"],
                        y=intraday.loc[mask_kijun_buy, "F_numeric"] + 12,  # Offset slightly above
                        mode="text",
                        text="K",
                        textposition="top center",
                        textfont=dict(size=14, color="green"),
                        hovertemplate="Time: %{x}<br>Kijun Cross: Buy<extra></extra>",

                        name="Buy Kijun Cross"
                    )

                    # 🔴 Bearish Kijun Cross (Red "K")
                    mask_kijun_sell = intraday["Kijun_F_Cross"] == "Sell Kijun Cross"
                    scatter_kijun_sell = go.Scatter(
                        x=intraday.loc[mask_kijun_sell, "Time"],
                        y=intraday.loc[mask_kijun_sell, "F_numeric"] - 12,  # Offset slightly below
                        mode="text",
                        text="K",
                        textposition="bottom center",
                        textfont=dict(size=14, color="red"),
                        hovertemplate="Time: %{x}<br>Kijun Cross: Sell<extra></extra>",

                        name="Sell Kijun Cross"
                    )

                    # Add Kijun Cross markers to F% plot (Row 1)
                    fig.add_trace(scatter_kijun_buy, row=1, col=1)
                    fig.add_trace(scatter_kijun_sell, row=1, col=1)

                    # # 🟢 Cotangent Spikes (Skull 💀) - Catches both >3 and <-3
                    # mask_cotangent_spike = intraday["F% Cotangent"].abs() > 3


                    # scatter_cotangent_spike = go.Scatter(
                    #     x=intraday.loc[mask_cotangent_spike, "Time"],
                    #     y=intraday.loc[mask_cotangent_spike, "F_numeric"] - 9,  # Slightly offset for visibility
                    #     mode="text",
                    #     text="💀",
                    #     textposition="top center",
                    #     textfont=dict(size=18),  # Larger for emphasis
                    #     name="Cotangent Spike",
                    #     hovertext=intraday.loc[mask_cotangent_spike, "F% Cotangent"].round(2),  # Display rounded cotangent value
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Cotangent: %{hovertext}<extra></extra>"
                    # )

                    #  # Add to the F% plot (Row 1)
                    # fig.add_trace(scatter_cotangent_spike, row=1, col=1)

                    # # 🟠 Cosecant Spikes (Lightning ⚡) - Detects |F% Cosecant| > 20
                    # mask_cosecant_spike = intraday["F% Cosecant"].abs() > 20

                    # scatter_cosecant_spike = go.Scatter(
                    #     x=intraday.loc[mask_cosecant_spike, "Time"],
                    #     y=intraday.loc[mask_cosecant_spike, "F_numeric"] + 20,  # Offset for visibility
                    #     mode="text",
                    #     text="⚡",
                    #     textposition="top center",
                    #     textfont=dict(size=18, color="orange"),  # Larger and orange for emphasis
                    #     name="Cosecant Spike",
                    #     hovertext=intraday.loc[mask_cosecant_spike, "F% Cosecant"].round(2),  # Display rounded cosecant value
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Cosecant: %{hovertext}<extra></extra>"
                    # )

                    # # Add to the F% plot (Row 1)
                    # fig.add_trace(scatter_cosecant_spike, row=1, col=1)

                    # # 🔵 Secant Spikes (Tornado 🌪) - Detects |F% Secant| > 3
                    # mask_secant_spike = intraday["F% Secant"].abs() > 5

                    # scatter_secant_spike = go.Scatter(
                    #     x=intraday.loc[mask_secant_spike, "Time"],
                    #     y=intraday.loc[mask_secant_spike, "F_numeric"] + 20,  # Offset for visibility
                    #     mode="text",
                    #     text="🌪",
                    #     textposition="top center",
                    #     textfont=dict(size=18, color="blue"),  # Large and blue for emphasis
                    #     name="Secant Spike",
                    #     hovertext=intraday.loc[mask_secant_spike, "F% Secant"].round(2),  # Display rounded secant value
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Secant: %{hovertext}<extra></extra>"
                    # )

                    # # Add to the F% plot (Row 1)
                    # fig.add_trace(scatter_secant_spike, row=1, col=1)

                    # # 🟢 Mask for VAS transitions
                    # mask_vas_gemstone = intraday["VAS Transition"] == "💎"
                    # mask_vas_balloon = intraday["VAS Transition"] == "🎈"

                    # # 💎 Gemstone (Bullish VAS Transition)
                    # scatter_vas_gemstone = go.Scatter(
                    #     x=intraday.loc[mask_vas_gemstone, "Time"],
                    #     y=intraday.loc[mask_vas_gemstone, "F_numeric"] + 30,  # Offset for visibility
                    #     mode="text",
                    #     text="💎",
                    #     textposition="top center",
                    #     textfont=dict(size=18, color="cyan"),  # Gemstone is cyan
                    #     name="VAS Bullish Transition",
                    #     hovertext=intraday.loc[mask_vas_gemstone, "VAS"].round(2),
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>VAS: %{hovertext}<extra></extra>"
                    # )

                    # # 🎈 Balloon (Bearish VAS Transition)
                    # scatter_vas_balloon = go.Scatter(
                    #     x=intraday.loc[mask_vas_balloon, "Time"],
                    #     y=intraday.loc[mask_vas_balloon, "F_numeric"] - 30,  # Offset for visibility
                    #     mode="text",
                    #     text="🎈",
                    #     textposition="bottom center",
                    #     textfont=dict(size=18, color="red"),  # Balloon is red
                    #     name="VAS Bearish Transition",
                    #     hovertext=intraday.loc[mask_vas_balloon, "VAS"].round(2),
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>VAS: %{hovertext}<extra></extra>"
                    # )

                    # # Add VAS transition markers to F% plot
                    # fig.add_trace(scatter_vas_gemstone, row=1, col=1)
                    # fig.add_trace(scatter_vas_balloon, row=1, col=1)


                    # Detect actual changes in High of Day (HOD) and Low of Day (LOD)
                    hod_changes = intraday["High of Day"].astype(str).str.contains(r"^\d", na=False)  # Only numbers, not "+X"
                    lod_changes = intraday["Low of Day"].astype(str).str.contains(r"^\d", na=False)  # Only numbers, not "+X"

                    # Plot High of Day (🚡 Aerial Tramway)
                    scatter_hod = go.Scatter(
                        x=intraday.loc[hod_changes, "Time"],
                        y=intraday.loc[hod_changes, "F_numeric"] + 5,  # Offset for visibility
                        mode="text",
                        text="🚡",
                        textposition="top center",
                        textfont=dict(size=18),  # Make emoji large
                        name="High of Day Update (🚡)",
                        hovertemplate="Time: %{x}<br>New HOD: %{text}<extra></extra>"
                    )

                    # Plot Low of Day (⚓ Anchor)
                    scatter_lod = go.Scatter(
                        x=intraday.loc[lod_changes, "Time"],
                        y=intraday.loc[lod_changes, "F_numeric"] - 5,  # Offset for visibility
                        mode="text",
                        text="⚓",
                        textposition="bottom center",
                        textfont=dict(size=18),  # Make emoji large
                        name="Low of Day Update (⚓)",
                        hovertemplate="Time: %{x}<br>New LOD: %{text}<extra></extra>"
                    )

                    # Add emojis to the F% plot (Row 1)
                    fig.add_trace(scatter_hod, row=1, col=1)
                    fig.add_trace(scatter_lod, row=1, col=1)



                    #             # Shift TD Pressure to compare with the previous value
                    # prev_td_pressure = intraday["TD Pressure"].shift(1)

                    # # Create mask for **bullish** TD Pressure flips (previously negative, now positive)
                    # mask_td_bullish = (prev_td_pressure < 0) & (intraday["TD Pressure"] >= 0)

                    # # Create mask for **bearish** TD Pressure flips (previously positive, now negative)
                    # mask_td_bearish = (prev_td_pressure > 0) & (intraday["TD Pressure"] <= 0)


                    # # 🔥 Bullish TD Pressure Flip (Fire Emoji)
                    # scatter_td_bullish = go.Scatter(
                    #     x=intraday.loc[mask_td_bullish, "Time"],
                    #     y=intraday.loc[mask_td_bullish, "F_numeric"] + 5,  # Slightly above for visibility
                    #     mode="text",
                    #     text="🔥",
                    #     textposition="top center",
                    #     textfont=dict(size=18, color="red"),
                    #     name="TD Pressure Bullish Flip",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>TD Pressure: %{text}<extra></extra>"
                    # )

                    # # 💧 Bearish TD Pressure Flip (Water Emoji)
                    # scatter_td_bearish = go.Scatter(
                    #     x=intraday.loc[mask_td_bearish, "Time"],
                    #     y=intraday.loc[mask_td_bearish, "F_numeric"] - 5,  # Slightly below for visibility
                    #     mode="text",
                    #     text="💧",
                    #     textposition="bottom center",
                    #     textfont=dict(size=18, color="blue"),
                    #     name="TD Pressure Bearish Flip",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>TD Pressure: %{text}<extra></extra>"
                    # )

                    # # Add these traces to the F% plot (Row 1)
                    # fig.add_trace(scatter_td_bullish, row=1, col=1)
                    # fig.add_trace(scatter_td_bearish, row=1, col=1)

                    # Mask for Tenkan-Kijun Crosses
                    mask_tk_sun = intraday["Tenkan_Kijun_Cross"] == "🌞"
                    mask_tk_moon = intraday["Tenkan_Kijun_Cross"] == "🌙"

                    # 🌞 Bullish Tenkan-Kijun Cross (Sun Emoji)
                    scatter_tk_sun = go.Scatter(
                        x=intraday.loc[mask_tk_sun, "Time"],
                        y=intraday.loc[mask_tk_sun, "F_numeric"] + 2,  # Offset for visibility
                        mode="text",
                        text="🌞",
                        textposition="top center",
                        textfont=dict(size=25),
                        name="Tenkan-Kijun Bullish Cross",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Above Kijun<extra></extra>"
                    )

                    # 🌙 Bearish Tenkan-Kijun Cross (Moon Emoji)
                    scatter_tk_moon = go.Scatter(
                        x=intraday.loc[mask_tk_moon, "Time"],
                        y=intraday.loc[mask_tk_moon, "F_numeric"] - 2,  # Offset for visibility
                        mode="text",
                        text="🌙",
                        textposition="bottom center",
                        textfont=dict(size=25),
                        name="Tenkan-Kijun Bearish Cross",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Below Kijun<extra></extra>"
                    )

                    # Add to the F% Plot
                    fig.add_trace(scatter_tk_sun, row=1, col=1)
                    fig.add_trace(scatter_tk_moon, row=1, col=1)

                    # ✅ Yesterday's Open - Grey Dashed Line (F% Scale)
                    y_open_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday Open F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="grey", dash="dash"),
                        name="Yesterday Open (F%)"
                    )

                    # ✅ Yesterday's High - Blue Dashed Line (F% Scale)
                    y_high_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday High F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="green", dash="dash"),
                        name="Yesterday High (F%)"
                    )

                    # ✅ Yesterday's Low - Green Dashed Line (F% Scale)
                    y_low_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday Low F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="Yesterday Low (F%)"
                    )

                    # ✅ Yesterday's Close - Red Dashed Line (F% Scale) (Always at 0)
                    y_close_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[0] * len(intraday),
                        mode="lines",
                        line=dict(color="blue", dash="dash"),
                        name="Yesterday Close (F%)"
                    )

                    # 🎯 Add all lines to the F% plot
                    fig.add_trace(y_open_f_line, row=1, col=1)
                    fig.add_trace(y_high_f_line, row=1, col=1)
                    fig.add_trace(y_low_f_line, row=1, col=1)
                    fig.add_trace(y_close_f_line, row=1, col=1)


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
                        title=f"{t} – Pure Demark",
                        margin=dict(l=30, r=30, t=50, b=30),
                        height=1580,  # Increase overall figure height (default ~450-600)

                        showlegend=True
                    )

                    fig.update_xaxes(title_text="Time", row=1, col=1)
                    fig.update_yaxes(title_text="F% Scale", row=1, col=1)

                    # Add to F% plot
                    mask_ops_bear = intraday["OPS Transition"] == "🐻"
                    mask_ops_panda = intraday["OPS Transition"] == "🐼"

                    scatter_ops_bear = go.Scatter(
                        x=intraday.loc[mask_ops_bear, "Time"],
                        y=intraday.loc[mask_ops_bear, "F_numeric"] - 7,  # Offset to avoid overlap
                        mode="text",
                        text="🐻",
                        textposition="bottom center",
                        textfont=dict(size=22, color="red"),
                        name="OPS Bearish Flip",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>OPS Turned Bearish<extra></extra>"
                    )

                    scatter_ops_panda = go.Scatter(
                        x=intraday.loc[mask_ops_panda, "Time"],
                        y=intraday.loc[mask_ops_panda, "F_numeric"] + 7,  # Offset to avoid overlap
                        mode="text",
                        text="🐼",
                        textposition="top center",
                        textfont=dict(size=22, color="green"),
                        name="OPS Bullish Flip",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>OPS Turned Bullish<extra></extra>"
                    )

                    # Add to the F% plot
                    fig.add_trace(scatter_ops_bear, row=1, col=1)
                    fig.add_trace(scatter_ops_panda, row=1, col=1)

                    # Plot Eagle & Owl
                    mask_eagle = intraday["Net Price Direction"] == "🦅"
                    mask_owl = intraday["Net Price Direction"] == "🦉"

                    scatter_eagle = go.Scatter(
                        x=intraday.loc[mask_eagle, "Time"],
                        y=intraday.loc[mask_eagle, "F_numeric"] + 25,
                        mode="text",
                        text="🦅",
                        textposition="top center",
                        textfont=dict(size=32),
                        name="Net Price Flip: Eagle (Bullish)",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Net Price flipped to Positive<extra></extra>"
                    )

                    scatter_owl = go.Scatter(
                        x=intraday.loc[mask_owl, "Time"],
                        y=intraday.loc[mask_owl, "F_numeric"] - 25,
                        mode="text",
                        text="🦉",
                        textposition="bottom center",
                        textfont=dict(size=32),
                        name="Net Price Flip: Owl (Bearish)",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Net Price flipped to Negative<extra></extra>"
                    )

                    fig.add_trace(scatter_eagle, row=1, col=1)
                    fig.add_trace(scatter_owl, row=1, col=1)


                    # Add annotation for OPS at market open
                    fig.add_annotation(
                            x=first_ops_time,
                            y=first_ops_value,  # Adjust Y position based on value
                            text=ops_label,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowcolor=ops_color,
                            font=dict(size=12, color=ops_color, family="Arial Black"),
                    )

                    # Detect OPS direction changes for emoji markers
                    ops_shifted = intraday["Opening Price Signal"].shift(1)
                    mask_bear = (ops_shifted > 0) & (intraday["Opening Price Signal"] <= 0)  # Bearish flip
                    mask_panda = (ops_shifted < 0) & (intraday["Opening Price Signal"] >= 0)  # Bullish flip




                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("Show/Hide Data Table"):
                    # Show data table, including new columns
                    cols_to_show = [
                          "Time", "Close","Opening Price Signal","OPS Transition","Net Price","Net Price Direction","Bounce","Retrace",
                       "RVOL_5","F% BBW", "Day Type", "High of Day",
                        "Low of Day", "F%","TD Open","TD Trap","TD CLoP", "40ish","Tenkan_Kijun_Cross",
             "CTOD Alert","Alert_Kijun","Alert_Mid","Wealth Signal"
                    ]
                    st.dataframe(intraday[cols_to_show], use_container_width=True)

            except Exception as e:
                st.error(f"Error fetching data for {t}: {e}")





                # 📆 Daily Chart Overview (Fix)
                with st.expander("📆 Daily Chart Overview", expanded=True):
                    try:
                        daily_chart_data = yf.download(t, period="60d", interval="1d", progress=False)

                        if not daily_chart_data.empty:
                            daily_chart_data.reset_index(inplace=True)

                            if isinstance(daily_chart_data.columns, pd.MultiIndex):
                                daily_chart_data.columns = daily_chart_data.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

                            if "Date" not in daily_chart_data.columns:
                                daily_chart_data.rename(columns={"index": "Date"}, inplace=True)

                            daily_chart_data["Date"] = pd.to_datetime(daily_chart_data["Date"])

                            fig_daily = go.Figure()
                            fig_daily.add_trace(go.Candlestick(
                                x=daily_chart_data["Date"],
                                open=daily_chart_data["Open"],
                                high=daily_chart_data["High"],
                                low=daily_chart_data["Low"],
                                close=daily_chart_data["Close"],
                                name="Daily Candles"
                            ))

                            fig_daily.update_layout(
                                title=f"{t} – Daily Candlestick Chart (Past 60 Days)",
                                 height=2000,                # taller canvas
                                 width=1800,
                                xaxis_rangeslider_visible=False,
                                margin=dict(l=30, r=30, t=40, b=20)
                            )

                            st.plotly_chart(fig_daily, use_container_width=True)
                        else:
                            st.warning("No daily data available.")
                    except Exception as e:
                        st.error(f"Failed to load daily chart: {e}")




            except Exception as e:
                st.error(f"Error fetching data for {t}: {e}")