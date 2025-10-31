import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from ta.trend import ADXIndicator
from ta.volume import MFIIndicator

import plotly.graph_objects as go
import pandas as pd
import numpy as np



st.title("Spot Analysis")

# User input for ticker symbol
# --- STOCK GROUP SELECTION ---
stock_groups = {
    "Tech": ["MSFT", "AMZN", "AAPL", "AMD", "MU", "GOOGL", "PLTR", "UBER", "SMCI", "PANW", "CRWD", "NVDA", "AVGO", "MRVL", "ON"],
    "Cyclical": ["CART", "SBUX", "DKNG", "CMG", "URBN", "TSLA", "CHWY", "NKE", "ETSY", "CROX", "W", "TGT", "GM", "GME", "AMZN", "HD", "CHWY"],
    "Finance": ["C", "WFC", "JPM", "HOOD", "V", "BAC", "PYPL", "COIN"],
    "Communication": ["NFLX", "GOOGL", "RBLX", "PINS", "DASH", "DIS", "META"],
    "ETF": ["SPY", "KBE", "QQQ"],
    "Defensive": ["TGT", "COST"]
}

# Group Dropdown
selected_group = st.selectbox("📂 Choose Stock Group", options=list(stock_groups.keys()))
group_tickers = stock_groups[selected_group]

# Ticker dropdown from selected group
selected_group_ticker = st.selectbox("📈 Select Ticker from Group", options=group_tickers)

# Optional manual override (any stock)
custom_ticker_input = st.text_input("🔍 Or enter a custom stock symbol:")

# Final ticker logic: use custom if typed, otherwise use group selection
ticker = custom_ticker_input.strip().upper() if custom_ticker_input else selected_group_ticker
st.markdown(f"**✅ Active Ticker Selected:** `{ticker}`")

# Date range selection: default to the past year
today = datetime.today()
one_year_ago = today - timedelta(days=365)
date_range = st.date_input("Select Date Range:", value=[one_year_ago, today])

# Ensure two dates are selected
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a start and end date.")
    st.stop()

# Define columns you want to hide (e.g., hide Volume column)
columns_to_hide = ['Volume']  # Adjust as needed

if st.button("Load Data"):
    # Download data for the specified ticker and date range
    df = yf.download(ticker, start=start_date, end=end_date)

    # Flatten the multi-level header if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the columns we need
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
   # Calculate indicators BEFORE sorting
    df['Relative Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    df['Dollar Change'] = df['Close'].diff()

    df['Percent Change'] = df['Close'].pct_change() * 100

    df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Daily return (close to close)
    df['Daily Return'] = df['Close'].pct_change()

    # Rolling standard deviation (20-day default)
    df['Rolling_SD'] = df['Daily Return'].rolling(window=20).std()

    # Return in Standard Deviation units
    df['Return_in_SD'] = df['Daily Return'] / df['Rolling_SD']


    # Log return: Open to Close
    df['Log_Change_OC'] = np.log(df['Close'] / df['Open'])

    # Rolling standard deviation of O–C log returns
    df['Rolling_SD_OC'] = df['Log_Change_OC'].rolling(window=20).std()

    # Return in SD units (O–C)
    df['Return_OC_in_SD'] = df['Log_Change_OC'] / df['Rolling_SD_OC']


    # Momentum indicators (will be shown only in the expander)
    df['ROC'] = df['Close'].pct_change(periods=7) * 100
    df['Momentum'] = df['Close'] - df['Close'].shift(7)
    # Stochastic Oscillator (%K and %D)
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['%K'] = ((df['Close'] - low14) / (high14 - low14)) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()
    # Bollinger Band % (BB%)
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    df['BB%'] = (df['Close'] - lower_band) / (upper_band - lower_band)
    df['BBW'] = (upper_band - lower_band) / sma20


    df['Upper Band'] = upper_band
    df['Lower Band'] = lower_band
    df['SMA 20'] = sma20

    # DMI: ADX, +DI, -DI
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['+DI'] = adx.adx_pos()
    df['-DI'] = adx.adx_neg()
    # --- Tom DeMark Calculations ---
    df['Change_%_c-c'] = ((df['Close'] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    df['Change_%_o-c'] = ((df['Close'] - df['Open']) / df['Open']) * 100

    # Dollar Change - Close to Close and Open to Close
    df['DC_CloseToClose'] = df['Close'].diff()  # Now matches R's lag-based change
    df['DC_OpenToClose'] = df['Close'] - df['Open']



# --- TD Sequential: Buy and Sell Setup Only ---

    buy_setup = []
    sell_setup = []
    buy_count = 0
    sell_count = 0

    for i in range(len(df)):
        # Buy Setup condition: Close < Close 4 bars ago
        if i >= 4 and df['Close'].iloc[i] < df['Close'].iloc[i - 4]:
            buy_count += 1
            if buy_count == 9:
                buy_setup.append("Buy Setup Completed")
                buy_count = 0
            else:
                buy_setup.append(f"Buy Setup {buy_count}")
        else:
            buy_count = 0
            buy_setup.append(None)

        # Sell Setup condition: Close > Close 4 bars ago
        if i >= 4 and df['Close'].iloc[i] > df['Close'].iloc[i - 4]:
            sell_count += 1
            if sell_count == 9:
                sell_setup.append("Sell Setup Completed")
                sell_count = 0
            else:
                sell_setup.append(f"Sell Setup {sell_count}")
        else:
            sell_count = 0
            sell_setup.append(None)

    df['Buy Setup'] = buy_setup
    df['Sell Setup'] = sell_setup


    # --- TD Sequential Countdown Logic ---

    buy_countdown = [None] * len(df)
    sell_countdown = [None] * len(df)

    active_buy_countdown = False
    active_sell_countdown = False
    buy_count = 0
    sell_count = 0

    for i in range(2, len(df)):
        # Activate Buy Countdown if Buy Setup Completed appears
        if df['Buy Setup'].iloc[i] == "Buy Setup Completed":
            active_buy_countdown = True
            buy_count = 0

        # Activate Sell Countdown if Sell Setup Completed appears
        if df['Sell Setup'].iloc[i] == "Sell Setup Completed":
            active_sell_countdown = True
            sell_count = 0

        # Stop Buy Countdown if new Sell Setup starts
        if df['Sell Setup'].iloc[i] is not None and "Sell Setup" in df['Sell Setup'].iloc[i]:
            active_buy_countdown = False

        # Stop Sell Countdown if new Buy Setup starts
        if df['Buy Setup'].iloc[i] is not None and "Buy Setup" in df['Buy Setup'].iloc[i]:
            active_sell_countdown = False

        # Buy Countdown condition
        if active_buy_countdown and df['Close'].iloc[i] <= df['Low'].iloc[i - 2]:
            buy_count += 1
            buy_countdown[i] = buy_count
            if buy_count == 13:
                active_buy_countdown = False

        # Sell Countdown condition
        if active_sell_countdown and df['Close'].iloc[i] >= df['High'].iloc[i - 2]:
            sell_count += 1
            sell_countdown[i] = sell_count
            if sell_count == 13:
                active_sell_countdown = False

    df['Buy Countdown'] = buy_countdown
    df['Sell Countdown'] = sell_countdown



    # --- TD Open Signal Logic ---

    td_open_signal = [None] * len(df)

    for i in range(1, len(df)):
        buy_setup_active = df['Buy Setup'].iloc[i] is not None and "Buy Setup" in df['Buy Setup'].iloc[i]
        sell_setup_active = df['Sell Setup'].iloc[i] is not None and "Sell Setup" in df['Sell Setup'].iloc[i]

        # Buy TD Open Signal
        if buy_setup_active and df['Open'].iloc[i] < df['Low'].iloc[i - 1] and df['Close'].iloc[i] > df['Low'].iloc[i - 1]:
            td_open_signal[i] = "Buy TD Open"

        # Sell TD Open Signal
        elif sell_setup_active and df['Open'].iloc[i] > df['High'].iloc[i - 1] and df['Close'].iloc[i] < df['High'].iloc[i - 1]:
            td_open_signal[i] = "Sell TD Open"

    df['TD Open'] = td_open_signal


    # --- TD Trap Logic ---

    td_trap = [None] * len(df)

    for i in range(1, len(df)):
        buy_setup_active = df['Buy Setup'].iloc[i] is not None and "Buy Setup" in df['Buy Setup'].iloc[i]
        sell_setup_active = df['Sell Setup'].iloc[i] is not None and "Sell Setup" in df['Sell Setup'].iloc[i]

        # Buy Trap
        if (
            buy_setup_active and
            df['Open'].iloc[i] >= df['Low'].iloc[i - 1] and df['Open'].iloc[i] <= df['High'].iloc[i - 1] and
            df['High'].iloc[i] > df['High'].iloc[i - 1] and
            df['Close'].iloc[i] < df['High'].iloc[i - 1]
        ):
            td_trap[i] = "Buy Trap"

        # Sell Trap
        elif (
            sell_setup_active and
            df['Open'].iloc[i] >= df['Low'].iloc[i - 1] and df['Open'].iloc[i] <= df['High'].iloc[i - 1] and
            df['Low'].iloc[i] < df['Low'].iloc[i - 1] and
            df['Close'].iloc[i] > df['Low'].iloc[i - 1]
        ):
            td_trap[i] = "Sell Trap"

    df['TD Trap'] = td_trap





    # --- TD CLoP Logic ---

    td_clop = [None] * len(df)

    for i in range(1, len(df)):
        buy_setup_active = df['Buy Setup'].iloc[i] is not None and "Buy Setup" in df['Buy Setup'].iloc[i]
        sell_setup_active = df['Sell Setup'].iloc[i] is not None and "Sell Setup" in df['Sell Setup'].iloc[i]

        # Buy CLoP
        if (
            buy_setup_active and
            df['Open'].iloc[i] < df['Close'].iloc[i - 1] and
            df['Open'].iloc[i] < df['Open'].iloc[i - 1] and
            df['High'].iloc[i] > df['Close'].iloc[i - 1] and
            df['High'].iloc[i] > df['Open'].iloc[i - 1]
        ):
            td_clop[i] = "Buy CLoP"

        # Sell CLoP
        elif (
            sell_setup_active and
            df['Open'].iloc[i] > df['Close'].iloc[i - 1] and
            df['Open'].iloc[i] > df['Open'].iloc[i - 1] and
            df['Low'].iloc[i] < df['Close'].iloc[i - 1] and
            df['Low'].iloc[i] < df['Open'].iloc[i - 1]
        ):
            td_clop[i] = "Sell CLoP"

    df['TD CLoP'] = td_clop



    # --- TD CLoPWIN Logic ---

    td_clopwin = [None] * len(df)

    for i in range(1, len(df) - 1):  # Stop at len-1 since signal applies to i+1
        buy_setup_active = df['Buy Setup'].iloc[i] is not None and "Buy Setup" in df['Buy Setup'].iloc[i]
        sell_setup_active = df['Sell Setup'].iloc[i] is not None and "Sell Setup" in df['Sell Setup'].iloc[i]

        prev_open = df['Open'].iloc[i - 1]
        prev_close = df['Close'].iloc[i - 1]
        curr_open = df['Open'].iloc[i]
        curr_close = df['Close'].iloc[i]

        low_bound = min(prev_open, prev_close)
        high_bound = max(prev_open, prev_close)

        # Buy CLoPWIN
        if (
            buy_setup_active and
            low_bound <= curr_open <= high_bound and
            low_bound <= curr_close <= high_bound and
            curr_close > prev_close
        ):
            td_clopwin[i + 1] = "Buy CLoPWIN"

        # Sell CLoPWIN
        if (
            sell_setup_active and
            low_bound <= curr_open <= high_bound and
            low_bound <= curr_close <= high_bound and
            curr_close < prev_close
        ):
            td_clopwin[i + 1] = "Sell CLoPWIN"

    df['TD CLoPWIN'] = td_clopwin



    # --- TDST: Show only once at creation, blank until changed ---

    tdst = [None] * len(df)
    current_tdst = None

    for i in range(9, len(df)):
        # Buy TDST
        if df['Buy Setup'].iloc[i] == "Buy Setup Completed":
            high_1 = df['High'].iloc[i - 8]
            high_2 = df['High'].iloc[i - 7]
            new_tdst = f"Buy TDST: {round(max(high_1, high_2), 2)}"

            if new_tdst != current_tdst:
                current_tdst = new_tdst
                tdst[i] = current_tdst

        # Sell TDST
        elif df['Sell Setup'].iloc[i] == "Sell Setup Completed":
            low_1 = df['Low'].iloc[i - 8]
            new_tdst = f"Sell TDST: {round(low_1, 2)}"

            if new_tdst != current_tdst:
                current_tdst = new_tdst
                tdst[i] = current_tdst

        # Otherwise: keep it blank (None)

    df['TDST'] = tdst


    # --- TD REI Calculation ---

    td_rei = [None] * len(df)

    for i in range(4, len(df)):
        try:
            high_diff = df['High'].iloc[i] - df['High'].iloc[i - 2]
            low_diff = df['Low'].iloc[i] - df['Low'].iloc[i - 2]
            numerator = high_diff + low_diff

            highest_high = df['High'].iloc[i - 4:i + 1].max()
            lowest_low = df['Low'].iloc[i - 4:i + 1].min()
            denominator = highest_high - lowest_low

            if denominator != 0:
                rei = (numerator / denominator) * 100
                rei = max(min(rei, 100), -100)  # Clamp between -100 and 100
                td_rei[i] = rei
        except:
            td_rei[i] = None

    df['TD REI'] = td_rei

    # --- TD REI Mild Oscillator Alert ---

    rei_alert = [None] * len(df)
    zone = None
    counter = 0

    for i in range(1, len(df)):
        rei = df['TD REI'].iloc[i]

        # Detect entering a zone
        if rei is not None:
            if rei > 40:
                if zone != "overbought":
                    zone = "overbought"
                    counter = 1
                else:
                    counter += 1

            elif rei < -40:
                if zone != "oversold":
                    zone = "oversold"
                    counter = 1
                else:
                    counter += 1

            else:  # Back to neutral
                if zone in ["overbought", "oversold"] and 1 <= counter <= 5:
                    rei_alert[i] = f"Mild {zone.title()} Pullback"
                zone = None
                counter = 0

    df["REI Mild Alert"] = rei_alert


    # --- Add REI Zone and Counter columns for POQ conditions ---
    rei_zone_debug = []
    rei_counter_debug = []
    zone = None
    counter = 0

    for i in range(len(df)):
        rei = df['TD REI'].iloc[i]

        if pd.notna(rei) and rei < -45:
            if zone != 'oversold':
                zone = 'oversold'
                counter = 1
            else:
                counter += 1
        elif pd.notna(rei) and rei > 45:
            if zone != 'overbought':
                zone = 'overbought'
                counter = 1
            else:
                counter += 1
        else:
            zone = None
            counter = 0

        rei_zone_debug.append(zone)
        rei_counter_debug.append(counter)

    df["REI Zone"] = rei_zone_debug
    df["REI Zone Count"] = rei_counter_debug




    # --- TD POQ Signal Detection (Scenarios 1-4) ---
    td_poq = [None] * len(df)

    for i in range(2, len(df)):
        rei = df['TD REI'].iloc[i]
        zone = df['REI Zone'].iloc[i]
        zone_count = df['REI Zone Count'].iloc[i]

        if zone not in ['oversold', 'overbought'] or zone_count > 5:
            continue

        prev_close = df['Close'].iloc[i - 1]
        curr_open = df['Open'].iloc[i]
        curr_close = df['Close'].iloc[i]

        if zone == 'oversold':
            # POQ Scenario 1
            if curr_open < prev_close and curr_close > df['High'].iloc[i - 1]:
                td_poq[i] = "Scenario 1: Buy Call"

            # POQ Scenario 3
            elif curr_open > df['Low'].iloc[i - 2] and curr_close > df['Low'].iloc[i - 2] and curr_close > prev_close:
                td_poq[i] = "Scenario 3: Buy Call"

        elif zone == 'overbought':
            # POQ Scenario 2
            if curr_open > prev_close and curr_close < df['Low'].iloc[i - 1]:
                td_poq[i] = "Scenario 2: Buy Put"

            # POQ Scenario 4
            elif curr_open < df['High'].iloc[i - 2] and curr_close < df['High'].iloc[i - 2] and curr_close < prev_close:
                td_poq[i] = "Scenario 4: Buy Put"

    df['TD_POQ'] = td_poq




    # --- TD Pressure Alert (Single-Trigger Logic) ---

    td_pressure_alert = [None] * len(df)
    last_alert = None

    for i in range(2, len(df)):
        prev = df['Close'].iloc[i - 1]
        curr = df['Close'].iloc[i]
        pre_prev = df['Close'].iloc[i - 2]

        # Detect Alert - to + (Buy Pressure)
        if curr > prev and pre_prev < prev:
            if last_alert != "Alert - to +":
                td_pressure_alert[i] = "Alert - to +"
                last_alert = "Alert - to +"

        # Detect Alert + to - (Sell Pressure)
        elif curr < prev and pre_prev > prev:
            if last_alert != "Alert + to -":
                td_pressure_alert[i] = "Alert + to -"
                last_alert = "Alert + to -"

    df["TD Pressure Alert"] = td_pressure_alert




    def calculate_td_lines(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replicates the 'calculate_td_lines' logic from R to identify
        TD Demand and TD Supply Lines based on ringed lows/highs.
        """
        # Make a copy so we don't mutate original df
        df = df.copy()

        # Ensure ascending date index (if needed)
        # If your df is already in ascending order, you can skip this.
        df.sort_index(ascending=True, inplace=True)

        # Identify ringed lows/highs:
        # TD_Point_Low = current Low is lower than the prior and the following bar
        df['TD_Point_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))

        # TD_Point_High = current High is higher than the prior and the following bar
        df['TD_Point_High'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))

        demand_points = []
        supply_points = []

        TD_Demand_Line = []
        TD_Supply_Line = []

        # Iterate row by row
        for i in range(len(df)):
            # Demand Line
            if df['TD_Point_Low'].iloc[i]:
                # We found a ringed low
                demand_points.append(df['Low'].iloc[i])
                if len(demand_points) >= 2:
                    # Use the most recent two ringed lows
                    TD_Demand_Line.append(max(demand_points[-2:]))
                else:
                    TD_Demand_Line.append(demand_points[-1])
            else:
                TD_Demand_Line.append(np.nan)

            # Supply Line
            if df['TD_Point_High'].iloc[i]:
                # We found a ringed high
                supply_points.append(df['High'].iloc[i])
                if len(supply_points) >= 2:
                    # Use the most recent two ringed highs
                    TD_Supply_Line.append(min(supply_points[-2:]))
                else:
                    TD_Supply_Line.append(supply_points[-1])
            else:
                TD_Supply_Line.append(np.nan)

        # Assign them back to the DataFrame
        df['TD_Demand_Line'] = TD_Demand_Line
        df['TD_Supply_Line'] = TD_Supply_Line

        # Forward fill so that each line extends until a new ringed low/high overrides it
        df['TD_Demand_Line'] = df['TD_Demand_Line'].ffill()
        df['TD_Supply_Line'] = df['TD_Supply_Line'].ffill()

        return df



    df = calculate_td_lines(df)

    df["Alert_Close_Above_Demand"] = (df['Close'] > df['TD_Demand_Line']) & (df['Close'].shift(1) <= df['TD_Demand_Line'].shift(1))
    df["Alert_Close_Below_Demand"] = (df['Close'] < df['TD_Demand_Line']) & (df['Close'].shift(1) >= df['TD_Demand_Line'].shift(1))
    df["Alert_Close_Above_Supply"] = (df['Close'] > df['TD_Supply_Line']) & (df['Close'].shift(1) <= df['TD_Supply_Line'].shift(1))
    df["Alert_Close_Below_Supply"] = (df['Close'] < df['TD_Supply_Line']) & (df['Close'].shift(1) >= df['TD_Supply_Line'].shift(1))


    def demand_supply_alerts(row):
        if row["Alert_Close_Above_Demand"]:
            return "↑ Close > Demand"
        elif row["Alert_Close_Below_Demand"]:
            return "↓ Close < Demand"
        elif row["Alert_Close_Above_Supply"]:
            return "↑ Close > Supply"
        elif row["Alert_Close_Below_Supply"]:
            return "↓ Close < Supply"
        return None

    df["TD_Alert"] = df.apply(demand_supply_alerts, axis=1)


    df['Relative Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    # On-Balance Volume (OBV)
    df['OBV'] = (
        np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    ).cumsum() / 100000  # Scale down OBV



    # Accumulation/Distribution Line (ADL)
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)  # handle divide-by-zero

    mf_volume = mf_multiplier * df['Volume']
    df['ADL'] = mf_volume.cumsum() / 100000  # Optional: scale down like OBV



    # Money Flow Index (MFI)
    mfi = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14)
    df['MFI'] = mfi.money_flow_index()







    # Sort latest first
    df = df.sort_index(ascending=False)



    st.subheader(f"Data for {ticker} from {start_date} to {end_date}")
    # Remove the specified columns before displaying
# Display only selected columns in the main table
    df_display = df.drop(columns=columns_to_hide + ['ROC', 'Momentum','%K', '%D','BB%', 'ADX', '+DI', '-DI','Return_OC_in_SD','Log_Change_OC','Upper Band','Rolling_SD_OC','Lower Band',"SMA 20",'DC_CloseToClose','DC_OpenToClose','Change_%_c-c','Change_%_o-c','Buy Setup', 'Sell Setup', 'Buy Countdown', 'Sell Countdown','TD Open','TD Trap','TD CLoP','TD CLoPWIN','TDST', 'TD REI','REI Mild Alert','TD_POQ', 'TD Pressure Alert','TD_Demand_Line','TD_Supply_Line',"TD_Alert","REI Zone","REI Zone Count","Alert_Close_Above_Demand","Alert_Close_Below_Demand","Alert_Close_Above_Supply","Alert_Close_Below_Supply",'TD_Point_High','TD_Point_Low',"MFI","ADL","OBV","BBW","Relative Volume",'Daily Return','Rolling_SD'], errors='ignore')

    st.dataframe(df_display)

 
    df['Return_OC_in_SD'] = df['Log_Change_OC'] / df['Rolling_SD_OC']


    with st.expander("📊 Daily Returns Measured in Standard Deviations (Bar Color by Sign)"):
        fig_return_sd_bar = go.Figure()

        # Loop through each row and assign color
        for i in range(len(df)):
            sd_val = df['Return_in_SD'].iloc[i]
            date = df.index[i]

            fig_return_sd_bar.add_trace(go.Bar(
                x=[date],
                y=[sd_val],
                marker_color='green' if sd_val > 0 else 'red',
                showlegend=False
            ))

        fig_return_sd_bar.update_layout(
            title="Return in Standard Deviation Units (Daily Bars)",
            xaxis_title="Date",
            yaxis_title="Return in SD",
            template="plotly_white",
            height=350,
            margin=dict(l=30, r=30, t=30, b=30)
        )

        st.plotly_chart(fig_return_sd_bar, use_container_width=True)

    with st.expander("📊 Intraday (O–C) Returns in SD Units (Colored Bars)"):
        fig_oc_sd_bar = go.Figure()

        for i in range(len(df)):
            sd_val = df['Return_OC_in_SD'].iloc[i]
            date = df.index[i]

            fig_oc_sd_bar.add_trace(go.Bar(
                x=[date],
                y=[sd_val],
                marker_color='green' if sd_val > 0 else 'red',
                showlegend=False
            ))

        fig_oc_sd_bar.update_layout(
            title="Intraday (O–C) Return in Standard Deviation Units",
            xaxis_title="Date",
            yaxis_title="Return in SD (O–C)",
            template="plotly_white",
            height=350,
            margin=dict(l=30, r=30, t=30, b=30)
        )

        st.plotly_chart(fig_oc_sd_bar, use_container_width=True)



    # ✅ Paste HERE:
    with st.expander("📈 Momentum Details"):

        st.dataframe(df[['ROC', 'Momentum','%K', '%D' ,'BB%','ADX', '+DI', '-DI','Upper Band','Lower Band',"SMA 20",'BBW']].dropna())





    with st.expander("📊 Momentum Plots"):

        st.markdown("**2. ADX Plot**")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=df.index,
            y=df['ADX'],
            mode='lines',
            name='ADX',
            line=dict(color='green'),
            hoverinfo='x+y'
        ))

        fig2.add_trace(go.Scatter(
            x=df.index,
            y=df['+DI'],
            mode='lines',
            name='+DI',
            line=dict(color='blue'),
            hoverinfo='x+y'
        ))

        fig2.add_trace(go.Scatter(
            x=df.index,
            y=df['-DI'],
            mode='lines',
            name='-DI',
            line=dict(color='red'),
            hoverinfo='x+y'
        ))

        fig2.update_layout(
            height=400,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(x=0.01, y=0.99),
            title="ADX, +DI, -DI Over Time",
            template="plotly_white"
        )

        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📉 Bollinger Bands Plot"):
        fig3 = go.Figure()

        # Close price
        fig3.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close',
            line=dict(color='blue'),
            hoverinfo='x+y'
        ))

        # SMA 20
        fig3.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA 20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='black', dash='dash'),
            hoverinfo='x+y'
        ))

        # Upper Band
        fig3.add_trace(go.Scatter(
            x=df.index,
            y=df['Upper Band'],
            mode='lines',
            name='Upper Band',
            line=dict(color='gray'),
            hoverinfo='x+y'
        ))

        # Lower Band
        fig3.add_trace(go.Scatter(
            x=df.index,
            y=df['Lower Band'],
            mode='lines',
            name='Lower Band',
            line=dict(color='gray'),
            fill='tonexty',  # fill between upper and lower
            hoverinfo='x+y'
        ))

        fig3.update_layout(
            height=400,
            margin=dict(l=30, r=30, t=30, b=30),
            title="Bollinger Bands (Close, SMA 20, Upper, Lower)",
            template="plotly_white"
        )

        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📏 BBW Plot"):
        fig_bbw = go.Figure()

        fig_bbw.add_trace(go.Scatter(
            x=df.index,
            y=df['BBW'],
            mode='lines',
            name='BBW',
            line=dict(color='darkorange')
        ))

        fig_bbw.update_layout(
            title="Bollinger Band Width (BBW)",
            template="plotly_white",
            height=300,
            margin=dict(l=30, r=30, t=30, b=30),
            yaxis_title="BBW"
        )

        st.plotly_chart(fig_bbw, use_container_width=True)


    with st.expander("📦 Volume Details"):
        volume_df = df[['Volume', 'Relative Volume', 'OBV','ADL','MFI']].dropna()
        st.dataframe(volume_df, use_container_width=True)


    with st.expander("📈 OBV Plot"):
        fig_obv = go.Figure()

        fig_obv.add_trace(go.Scatter(
            x=df.index,
            y=df['OBV'],
            mode='lines',
            name='OBV (scaled)',
            line=dict(color='purple')
        ))

        fig_obv.update_layout(
            title="On-Balance Volume (OBV)",
            template="plotly_white",
            height=350,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(x=0.01, y=0.99)
        )

        st.plotly_chart(fig_obv, use_container_width=True)


    with st.expander("📈 ADL Plot"):
        fig_adl = go.Figure()

        fig_adl.add_trace(go.Scatter(
            x=df.index,
            y=df['ADL'],
            mode='lines',
            name='ADL (scaled)',
            line=dict(color='teal')
        ))

        fig_adl.update_layout(
            title="Accumulation/Distribution Line (ADL)",
            template="plotly_white",
            height=350,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(x=0.01, y=0.99)
        )

        st.plotly_chart(fig_adl, use_container_width=True)


    with st.expander("📈 MFI Plot"):
        fig_mfi = go.Figure()

        fig_mfi.add_trace(go.Scatter(
            x=df.index,
            y=df['MFI'],
            mode='lines',
            name='MFI',
            line=dict(color='darkgreen')
        ))

        fig_mfi.update_layout(
            title="Money Flow Index (MFI)",
            template="plotly_white",
            height=350,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(x=0.01, y=0.99),
            yaxis=dict(range=[0, 100], title='MFI')
        )

        st.plotly_chart(fig_mfi, use_container_width=True)



    with st.expander("🧠 Tom DeMark Details"):


        if not df.empty:
            td_df = df[['Close','Change_%_c-c', 'Change_%_o-c', 'DC_CloseToClose', 'DC_OpenToClose','Buy Setup', 'Sell Setup', 'Buy Countdown', 'Sell Countdown','TD Open','TD Trap','TD CLoP','TD CLoPWIN','TDST', 'TD REI','REI Mild Alert','TD_POQ', 'TD Pressure Alert','TD_Demand_Line','TD_Supply_Line',"TD_Alert"]].dropna(how='all')

            # Format percentage columns
            td_df['Change_%_c-c'] = td_df['Change_%_c-c'].round(2).astype(str) + '%'
            td_df['Change_%_o-c'] = td_df['Change_%_o-c'].round(2).astype(str) + '%'

            # Round dollar change columns to 2 decimal places
            td_df['DC_CloseToClose'] = td_df['DC_CloseToClose'].round(2)
            td_df['DC_OpenToClose'] = td_df['DC_OpenToClose'].round(2)

            st.dataframe(td_df, use_container_width=True, height=700)
        else:
            st.write("No data available.")



    with st.expander("📉 TD Sequential Plot"):

        fig = go.Figure()

        # Close Price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Close',
            mode='lines',
            line=dict(color='black')
        ))

        # Buy Setup Points
        fig.add_trace(go.Scatter(
            x=df[df['Buy Setup'] == 'Buy Setup Completed'].index,
            y=df[df['Buy Setup'] == 'Buy Setup Completed']['Close'],
            mode='markers+text',
            name='Buy Setup Completed',
            marker=dict(color='green', size=8, symbol='triangle-up'),
            text=["Buy 9"]*len(df[df['Buy Setup'] == 'Buy Setup Completed']),
            textposition="top center"
        ))

        # Sell Setup Points
        fig.add_trace(go.Scatter(
            x=df[df['Sell Setup'] == 'Sell Setup Completed'].index,
            y=df[df['Sell Setup'] == 'Sell Setup Completed']['Close'],
            mode='markers+text',
            name='Sell Setup Completed',
            marker=dict(color='red', size=8, symbol='triangle-down'),
            text=["Sell 9"]*len(df[df['Sell Setup'] == 'Sell Setup Completed']),
            textposition="bottom center"
        ))

        # Buy Countdown (optional)
        fig.add_trace(go.Scatter(
            x=df[df['Buy Countdown'] == 13].index,
            y=df[df['Buy Countdown'] == 13]['Close'],
            mode='markers+text',
            name='Buy Countdown 13',
            marker=dict(color='lime', size=10, symbol='star'),
            text=["BC 13"]*len(df[df['Buy Countdown'] == 13]),
            textposition="top center"
        ))

        # Sell Countdown (optional)
        fig.add_trace(go.Scatter(
            x=df[df['Sell Countdown'] == 13].index,
            y=df[df['Sell Countdown'] == 13]['Close'],
            mode='markers+text',
            name='Sell Countdown 13',
            marker=dict(color='orange', size=10, symbol='star'),
            text=["SC 13"]*len(df[df['Sell Countdown'] == 13]),
            textposition="bottom center"
        ))


    with st.expander("🏹 TD Demand and Supply Lines Plot"):
        df_plot = df.sort_index(ascending=True).copy()

        fig_ds = go.Figure()

        # Plot the closing price in solid black
        fig_ds.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close',
            line=dict(color='black')
        ))

        # Plot the TD Demand Line in dashed blue
        fig_ds.add_trace(go.Scatter(
            x=df.index,
            y=df['TD_Demand_Line'],
            mode='lines',
            name='TD Demand Line',
            line=dict(color='green', dash='dash')
        ))

        # Plot the TD Supply Line in dashed red
        fig_ds.add_trace(go.Scatter(
            x=df.index,
            y=df['TD_Supply_Line'],
            mode='lines',
            name='TD Supply Line',
            line=dict(color='red', dash='dash')
        ))







        fig_ds.update_layout(
            title="TD Demand & Supply Lines",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_ds, use_container_width=True)




        st.plotly_chart(fig, use_container_width=True)

    df = df.sort_index(ascending=True).copy()

      # Ichimoku lines
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    # SSA and SSB
    df['SSA'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['SSB'] = ((high_52 + low_52) / 2).shift(26)

    # Kumo boundaries
    df['Kumo_Top'] = df[['SSA', 'SSB']].max(axis=1)
    df['Kumo_Bottom'] = df[['SSA', 'SSB']].min(axis=1)



    with st.expander("🕯️ Kijun Candlestick Chart"):
        df_candle = df.sort_index(ascending=True).copy()

        fig_kijun = go.Figure()

        # Candlesticks
        fig_kijun.add_trace(go.Candlestick(
            x=df_candle.index,
            open=df_candle['Open'],
            high=df_candle['High'],
            low=df_candle['Low'],
            close=df_candle['Close'],
            name='Candles'
        ))

        # Tenkan-sen (red)
        fig_kijun.add_trace(go.Scatter(
            x=df_candle.index,
            y=df_candle['Tenkan_sen'],
            mode='lines',
            name='Tenkan-sen (9)',
            line=dict(color='red', width=1)
        ))

        # Kijun-sen (blue)
        fig_kijun.add_trace(go.Scatter(
            x=df_candle.index,
            y=df_candle['Kijun_sen'],
            mode='lines',
            name='Kijun-sen (26)',
            line=dict(color='green', width=1)
        ))



        fig_kijun.update_layout(
            title="Ichimoku: Kijun & Tenkan",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_kijun, use_container_width=True)



