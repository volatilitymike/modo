import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np  # make sure this is at the top too



def calculate_td_demand_supply_lines_fpercent(df):
        """
        Calculate TD Demand and Supply Lines using ringed lows/highs in F_numeric space.
        """
        df['TD Demand Line F'] = np.nan
        df['TD Supply Line F'] = np.nan

        demand_points = []
        supply_points = []
        f_vals = df['F_numeric'].to_numpy()

        for i in range(1, len(df) - 1):
            # Ringed Low (Demand in F%)
            if f_vals[i] < f_vals[i - 1] and f_vals[i] < f_vals[i + 1]:
                demand_points.append(f_vals[i])
                if len(demand_points) >= 2:
                    df.at[df.index[i], 'TD Demand Line F'] = max(demand_points[-2:])
                else:
                    df.at[df.index[i], 'TD Demand Line F'] = demand_points[-1]

            # Ringed High (Supply in F%)
            if f_vals[i] > f_vals[i - 1] and f_vals[i] > f_vals[i + 1]:
                supply_points.append(f_vals[i])
                if len(supply_points) >= 2:
                    df.at[df.index[i], 'TD Supply Line F'] = min(supply_points[-2:])
                else:
                    df.at[df.index[i], 'TD Supply Line F'] = supply_points[-1]

        df['TD Demand Line F'] = df['TD Demand Line F'].ffill()
        df['TD Supply Line F'] = df['TD Supply Line F'].ffill()
        return df



#***************************************************************************************************
def calculate_f_tenkan_kijun(df):
    """
    Computes F%-based Tenkan, Kijun, SpanA, and SpanB (Ichimoku logic).
    Also returns F% versions of each for plotting.
    """
    df["Prev_Close"] = df["Close"].shift(1)

    # Tenkan
    tenkan_high = df["High"].rolling(window=9, min_periods=1).max()
    tenkan_low = df["Low"].rolling(window=9, min_periods=1).min()
    df["Tenkan_sen"] = (tenkan_high + tenkan_low) / 2
    df["F% Tenkan"] = ((df["Tenkan_sen"] - df["Prev_Close"]) / df["Prev_Close"]) * 10000

    # Kijun
    kijun_high = df["High"].rolling(window=26, min_periods=1).max()
    kijun_low = df["Low"].rolling(window=26, min_periods=1).min()
    df["Kijun_sen"] = (kijun_high + kijun_low) / 2
    df["F% Kijun"] = ((df["Kijun_sen"] - df["Prev_Close"]) / df["Prev_Close"]) * 10000

    # Span A
    df["SpanA"] = ((df["Tenkan_sen"] + df["Kijun_sen"]) / 2).shift(26)
    df["SpanA_F"] = ((df["SpanA"] - df["Prev_Close"]) / df["Prev_Close"]) * 10000

    # Span B
    spanB_high = df["High"].rolling(window=52, min_periods=1).max()
    spanB_low = df["Low"].rolling(window=52, min_periods=1).min()
    df["SpanB"] = ((spanB_high + spanB_low) / 2).shift(26)
    df["SpanB_F"] = ((df["SpanB"] - df["Prev_Close"]) / df["Prev_Close"]) * 10000

    return df
#***************************************************************************************************
def detect_bbw_tight(df, window=5, percentile_threshold=10):
    """
    Detects BBW Tight Compression using dynamic threshold based on the ticker’s own BBW distribution.
    Fires 🐝 when at least `window` of the last `window` BBW values are below the Xth percentile.
    """
    if "F% BBW" not in df.columns:
        return df

    # 1) Compute the dynamic threshold
    threshold = np.percentile(df["F% BBW"].dropna(), percentile_threshold)

    # 2) Mark each bar if its BBW is below that threshold
    df["BBW_Tight"] = df["F% BBW"] < threshold

    # 3) Build the emoji column
    df["BBW_Tight_Emoji"] = ""
    for i in range(window, len(df)):
        # Check how many of the last `window` BBW bars were "tight"
        if df["BBW_Tight"].iloc[i-window:i].sum() >= 3:
            df.at[df.index[i], "BBW_Tight_Emoji"] = "🐝"

    return df


#***************************************************************************************************
    # ------------------------------
    # Span A and Span B (Ichimoku)
    # ------------------------------
    df["SpanA"] = ((df["Tenkan_sen"] + df["Kijun_sen"]) / 2).shift(26)
    df["SpanB"] = (
        (df["High"].rolling(window=52, min_periods=1).max() +
        df["Low"].rolling(window=52, min_periods=1).min()) / 2
    ).shift(26)

    df["SpanA_F"] = ((df["SpanA"] - df["Prev_Close"]) / df["Prev_Close"]) * 10000
    df["SpanB_F"] = ((df["SpanB"] - df["Prev_Close"]) / df["Prev_Close"]) * 10000

#***************************************************************************************************


#***************************************************************************************************
def detect_atr_expansion(df, lookback=5):
    """
    Flags ATR expansion by comparing current ATR to ATR 'lookback' periods ago.
    """
    # make sure df["ATR"] already exists
    df["ATR_Lag"] = df["ATR"].shift(lookback)

    df["ATR_Exp_Alert"] = np.select(
        [
            df["ATR"] >= 2 * df["ATR_Lag"],  # ≥150% of lagged ATR
            df["ATR"] >= 1.3 * df["ATR_Lag"],  # ≥120%
        ],
        [
            "☄️",  # big burst
            "💥",  # medium burst
        ],
        default=""
    )
    return df





# --------------------
# Page Config
# --------------------
st.set_page_config(page_title="Daily Dashboard", layout="wide")
st.title("📅 Daily Chart Dashboard")

# --------------------
# Sidebar – Inputs
# --------------------
st.sidebar.header("Daily Mode – Ticker & Dates")

default_tickers = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Choose Ticker", default_tickers)

start_date = st.sidebar.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

# --------------------
# Fetch Daily Data
# --------------------
@st.cache_data(show_spinner=False)
def load_daily_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), interval="1d")
    df["Time"] = df.index.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)

try:
    df = load_daily_data(selected_ticker, start_date, end_date)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# --------------------
# Run Button
# --------------------
run_analysis = st.sidebar.button("🚀 Run Analysis")

if run_analysis:
    # --------------------
    # Quick F% Setup
    # --------------------
# Setup F%
    prev_close = df["Close"].shift(1)
    df["F_numeric"] = ((df["Close"] - prev_close) / prev_close) * 10000


    # — ATR Calculation (no external lib needed) —
    df["Prev_Close"] = df["Close"].shift(1)
    # True Range
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Prev_Close"]).abs(),
        (df["Low"]  - df["Prev_Close"]).abs()
    ], axis=1).max(axis=1)
    # 14‑period ATR
    df["ATR"] = df["TR"].rolling(window=14, min_periods=1).mean()

    # Detect ATR expansion
    df = detect_atr_expansion(df, lookback=5)




    # — STD Expansion Alerts 🐦‍🔥 —
    df["F%_STD"] = df["F_numeric"].rolling(window=9, min_periods=1).std()

    lookback_std     = 5
    df["STD_Anchor"] = df["F%_STD"].shift(lookback_std)
    df["STD_Ratio"]  = df["F%_STD"] / df["STD_Anchor"]

    def std_alert(ratio):
        if np.isnan(ratio):
            return ""
        if ratio >= 3:
            return "🐦‍🔥"   # Triple STD expansion
        if ratio >= 2:
            return "🐦‍🔥"   # Double STD expansion
        return ""

    # map over the scalar ratios
    df["STD_Alert"] = df["STD_Ratio"].apply(std_alert)

#*****************************************************************************************





    # Calculate Tenkan/Kijun
    df = calculate_f_tenkan_kijun(df)




    # ── Bollinger Bands & BBW Alerts ─────────────────────────────
    df["MiddleBand"] = df["Close"].rolling(20).mean()
    df["STD"]        = df["Close"].rolling(20).std()
    df["UpperBand"]  = df["MiddleBand"] + 2 * df["STD"]
    df["LowerBand"]  = df["MiddleBand"] - 2 * df["STD"]

    # F%‑scaled Band‑Width
    df["F% BBW"] = ((df["UpperBand"] - df["LowerBand"]) / df["MiddleBand"]) * 10_000

    lookback         = 5
    df["BBW_Anchor"] = df["F% BBW"].shift(lookback)
    df["BBW_Ratio"]  = df["F% BBW"] / df["BBW_Anchor"]

  # After you compute df["BBW_Ratio"], do:

    def bbw_alert(ratio):
        if np.isnan(ratio):
            return ""
        if ratio >= 3:
            return "🔥"   # Triple expansion
        if ratio >= 2:
            return "🔥"   # Double expansion
        return ""

    # Map it over the BBW_Ratio column directly:
    df["BBW Alert"] = df["BBW_Ratio"].apply(bbw_alert)


    df = detect_bbw_tight(df, window=5, percentile_threshold=10)


    # THEN calculate the emoji signals
    df["Prev_F"] = df["F_numeric"].shift(1)
    df["Prev_Kijun"] = df["F% Kijun"].shift(1)
    df["Kijun_F_Cross_Emoji"] = ""

    up_cross = (df["Prev_F"] <= df["Prev_Kijun"]) & (df["F_numeric"] > df["F% Kijun"])
    down_cross = (df["Prev_F"] >= df["Prev_Kijun"]) & (df["F_numeric"] < df["F% Kijun"])

    df.loc[up_cross, "Kijun_F_Cross_Emoji"] = "🕊️"
    df.loc[down_cross, "Kijun_F_Cross_Emoji"] = "🐦‍⬛"

    # Then run TD levels
    df = calculate_td_demand_supply_lines_fpercent(df)

    df.loc[down_cross, "Kijun_F_Cross_Emoji"] = "🐦‍⬛"

    df["Prev_Tenkan"] = df["F% Tenkan"].shift(1)
    df["Prev_Kijun"] = df["F% Kijun"].shift(1)
    df["Tenkan_Kijun_Cross"] = ""

    # Bullish Cross (Tenkan crosses above Kijun)
    bull_cross = (df["Prev_Tenkan"] <= df["Prev_Kijun"]) & (df["F% Tenkan"] > df["F% Kijun"])
    # Bearish Cross (Tenkan crosses below Kijun)
    bear_cross = (df["Prev_Tenkan"] >= df["Prev_Kijun"]) & (df["F% Tenkan"] < df["F% Kijun"])

    df.loc[bull_cross, "Tenkan_Kijun_Cross"] = "🌞"
    df.loc[bear_cross, "Tenkan_Kijun_Cross"] = "🌙"
    df["F% BBW"] = ((df["UpperBand"] - df["LowerBand"]) / df["MiddleBand"]) * 10000

    # --------------------
    # Plot F% Chart
    # --------------------
    fig = go.Figure()


    fig.update_layout(
        height=700,     # Taller
        width=1400,     # W
        title=f"F% Movement – {selected_ticker}",
        xaxis_title="Date",
        yaxis_title="F%",
        margin=dict(l=40, r=40, t=60, b=40)
    )



    base_price = df["Close"].iloc[0]
    df["Price_F%_scaled"] = ((df["Close"] - base_price) / base_price) * 10000



    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["Price_F%_scaled"],
        mode="lines+markers",
        name="F%-like Price Trend"
    ))


    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["TD Supply Line F"],
        mode="lines",
        name="TD Supply Line F",
        line=dict(color="gray", dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["TD Demand Line F"],
        mode="lines",
        name="TD Demand Line F",
        line=dict(color="lightpink", dash="dash")
    ))



    df["Tenkan_scaled"] = ((df["Tenkan_sen"] - base_price) / base_price) * 10000
    df["Kijun_scaled"] = ((df["Kijun_sen"]  - base_price) / base_price) * 10000

    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["Tenkan_scaled"],
        mode="lines",
        name="Tenkan (scaled)",
        line=dict(color="red", dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["Kijun_scaled"],
        mode="lines",
        name="Kijun (scaled)",
        line=dict(color="green", dash="dot")
    ))


#     # Masks
# # Masks
#     mask_kijun_up = df["Kijun_F_Cross_Emoji"] == "🕊️"
#     mask_kijun_down = df["Kijun_F_Cross_Emoji"] == "🐦‍⬛"

# # Traces (add to fig)


#     # 🕊️ Upward Cross Trace
#     up_cross_trace = go.Scatter(
#         x=df.loc[mask_kijun_up, "Time"],
#         y=df.loc[mask_kijun_up, "F_numeric"] + 144,
#         mode="text",
#         text=df.loc[mask_kijun_up, "Kijun_F_Cross_Emoji"],
#         textposition="top center",
#         textfont=dict(size=34),
#         name="Kijun Cross Up (🕊️)",
#         hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Upward Cross: %{text}<extra></extra>"
#     )

#     # 🐦‍⬛ Downward Cross Trace
#     down_cross_trace = go.Scatter(
#         x=df.loc[mask_kijun_down, "Time"],
#         y=df.loc[mask_kijun_down, "F_numeric"] - 144,
#         mode="text",
#         text=df.loc[mask_kijun_down, "Kijun_F_Cross_Emoji"],
#         textposition="bottom center",
#         textfont=dict(size=34),
#         name="Kijun Cross Down (🐦‍⬛)",
#         hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Downward Cross: %{text}<extra></extra>"
#     )

#     # Add both traces
#     fig.add_trace(up_cross_trace)
#     fig.add_trace(down_cross_trace)

#     # Tenkan-Kijun Cross Emojis
#     mask_tk_sun = df["Tenkan_Kijun_Cross"] == "🌞"
#     mask_tk_moon = df["Tenkan_Kijun_Cross"] == "🌙"

#     fig.add_trace(go.Scatter(
#         x=df.loc[mask_tk_sun, "Time"],
#         y=df.loc[mask_tk_sun, "F_numeric"] + 377,
#         mode="text",
#         text="🌞",
#         textposition="top center",
#         textfont=dict(size=55),
#         name="Tenkan-Kijun Bullish Cross",
#         hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Above Kijun<extra></extra>"
#     ))

#     fig.add_trace(go.Scatter(
#         x=df.loc[mask_tk_moon, "Time"],
#         y=df.loc[mask_tk_moon, "F_numeric"] - 377,
#         mode="text",
#         text="🌙",
#         textposition="bottom center",
#         textfont=dict(size=55),
#         name="Tenkan-Kijun Bearish Cross",
#         hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Below Kijun<extra></extra>"
#     ))



#     bbw_mask = df["BBW Alert"] == "🔥"
#     fig.add_trace(go.Scatter(
#         x=df.loc[bbw_mask, "Time"],
#         y=df.loc[bbw_mask, "F_numeric"] + 322,
#         mode="text",
#         text=df.loc[bbw_mask, "BBW Alert"],   # ← series, not a scalar
#         textposition="top center",
#         textfont=dict(size=34),
#         name="BBW Expansion 🔥",
#         hovertemplate="Time: %{x}<br>BBW Expansion<extra></extra>"
#     ))




#     # ── Just before plotting to Streamlit:
#     # add the BBW‑Tight 🐝 trace:
#     mask_bee = df["BBW_Tight_Emoji"] == "🐝"
#     fig.add_trace(go.Scatter(
#         x=df.loc[mask_bee, "Time"],
#         y=df.loc[mask_bee, "F_numeric"] + 200,
#         mode="text",
#         text=df.loc[mask_bee, "BBW_Tight_Emoji"],
#         textposition="top center",
#         textfont=dict(size=32),
#         name="BBW Tight 🐝",
#         hovertemplate="Time: %{x}<br>BBW Tight Cluster<extra></extra>"
#     ))

#     mask_std = df["STD_Alert"] == "🐦‍🔥"
#     fig.add_trace(go.Scatter(
#         x=df.loc[mask_std, "Time"],
#         y=df.loc[mask_std, "F_numeric"] + 350,    # adjust offset if needed
#         mode="text",
#         text=df.loc[mask_std, "STD_Alert"],
#         textposition="top center",
#         textfont=dict(size=34),
#         name="STD Expansion 🐦‍🔥",
#         hovertemplate="Time: %{x}<br>STD Ratio: %{customdata:.2f}<extra></extra>",
#         customdata=df.loc[mask_std, "STD_Ratio"]
#     ))


#     mask_atr = df["ATR_Exp_Alert"] != ""
#     fig.add_trace(go.Scatter(
#         x=df.loc[mask_atr, "Time"],
#         y=df.loc[mask_atr, "F_numeric"] + 200,      # tweak the vertical offset as needed
#         mode="text",
#         text=df.loc[mask_atr, "ATR_Exp_Alert"],
#         textposition="top center",
#         textfont=dict(size=32),
#         name="ATR Expansion ☄️",
#         hovertemplate="Time: %{x}<br>ATR: %{customdata:.2f}<extra></extra>",
#         customdata=df.loc[mask_atr, "ATR"]
#     ))

#     # — Plot ATR Expansion Alerts ☄️ —
#     mask_atr = df["ATR_Exp_Alert"] != ""
#     fig.add_trace(go.Scatter(
#         x=df.loc[mask_atr, "Time"],
#         y=df.loc[mask_atr, "F_numeric"] + 200,      # tweak vertical offset
#         mode="text",
#         text=df.loc[mask_atr, "ATR_Exp_Alert"],
#         textposition="top center",
#         textfont=dict(size=32),
#         name="ATR Expansion ☄️",
#         hovertemplate="Time: %{x}<br>ATR: %{customdata:.2f}<extra></extra>",
#         customdata=df.loc[mask_atr, "ATR"]
#     ))


    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Show Data Table
    # --------------------
    with st.expander("Show Raw Data"):
        st.dataframe(df.tail(20), use_container_width=True)
else:
    st.info("Select your options and click **Run Analysis** to begin.")
