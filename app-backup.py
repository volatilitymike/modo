import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 20)

st.set_page_config(layout="wide")
st.title("Day Trading")

# Sidebar for user input
st.sidebar.header("Input Options")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")
timeframe = st.sidebar.selectbox(
    "Select Timeframe:",
    options=["2m", "5m", "15m", "30m", "60m", "1d"],
    index=2  # Default to "15m"
)

# ─────────────────────────────────────────────────────────
# Function: Calculate Standard Deviations of Change (Close-to-Close)
def calculate_sd_of_change_cc(data, window=13):
    data = data.copy()
    data['Log_Change_CC'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Rolling_StdDev_CC'] = data['Log_Change_CC'].rolling(window=window).std()
    data['SD_of_Change_CC'] = data['Log_Change_CC'] / data['Rolling_StdDev_CC']
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data[['Log_Change_CC','Rolling_StdDev_CC','SD_of_Change_CC']] = data[
        ['Log_Change_CC','Rolling_StdDev_CC','SD_of_Change_CC']
    ].fillna(0)
    return data


def calculate_kijun_sen(data, period=26):
            """
            Calculate the Kijun-sen (Base Line) for the given DataFrame.

            Parameters:
            data (pd.DataFrame): A DataFrame with 'High' and 'Low' columns.
            period (int): The period over which to calculate the Kijun-sen. Default is 26.

            Returns:
            pd.DataFrame: Updated DataFrame with a new column 'Kijun-sen'.
            """
            # Ensure required columns exist
            if 'High' not in data.columns or 'Low' not in data.columns:
                raise ValueError("The DataFrame must contain 'High' and 'Low' columns.")

            # Calculate Kijun-sen as the average of the highest high and lowest low over the period
            data['Kijun-sen'] = (data['High'].rolling(window=period).max() +
                                data['Low'].rolling(window=period).min()) / 2

            return data

def detect_kijun_cross(data):
            """
            Detects when the Closing Price crosses above or below the Kijun line.
            Adds a 'Kijun_Alert' column with the type of crossover.

            Parameters:
                data (pd.DataFrame): DataFrame containing 'Close' and 'Kijun'.

            Returns:
                pd.DataFrame: DataFrame with an added 'Kijun_Alert' column.
            """
            # Initialize the Kijun_Alert column with empty strings
            data['Kijun_Alert'] = ''

            # Check for crossing above Kijun
            data.loc[(data['Close'].shift(1) <= data['Kijun'].shift(1)) & (data['Close'] > data['Kijun']), 'Kijun_Alert'] = 'Cross Up Kijun'

            # Check for crossing below Kijun
            data.loc[(data['Close'].shift(1) >= data['Kijun'].shift(1)) & (data['Close'] < data['Kijun']), 'Kijun_Alert'] = 'Cross Down Kijun'

            return data



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





def add_td_pressure_alert(data):
            """
            Add an alert for TD Pressure changes:
            - "alert + to -" when TD Pressure changes from positive to negative.
            - "alert - to +" when TD Pressure changes from negative to positive.

            Parameters:
                data (pd.DataFrame): DataFrame containing the 'TD Pressure' column.

            Returns:
                pd.DataFrame: DataFrame with the 'TD Pressure Alert' column added.
            """
            # Ensure 'TD Pressure' column exists
            if 'TD Pressure' not in data.columns:
                raise ValueError("The DataFrame must contain a 'TD Pressure' column.")

            # Initialize the alert column with empty strings
            data['TD Pressure Alert'] = ''

            # Iterate through the DataFrame starting from the second row
            for i in range(1, len(data)):
                current_pressure = data['TD Pressure'].iloc[i]
                previous_pressure = data['TD Pressure'].iloc[i - 1]

                # Check for "alert + to -"
                if previous_pressure > 0 and current_pressure <= 0:
                    data.at[data.index[i], 'TD Pressure Alert'] = 'alert + to -'

                # Check for "alert - to +"
                elif previous_pressure < 0 and current_pressure >= 0:
                    data.at[data.index[i], 'TD Pressure Alert'] = 'alert - to +'

            return data

def calculate_td_rei(data):
            """
            Calculates TD REI as per the formula outlined in the reference image.

            Steps:
            1. Add the difference between the high of the current price bar and the high two bars earlier
            to the difference between the low of the current price bar and the low two bars earlier.
            2. Divide the result by the absolute price move over the 5-bar period (highest high - lowest low).
            3. Multiply by 100 to normalize the result.
            """

            # Initialize TD REI column
            data['TD REI'] = np.nan

            for i in range(4, len(data)):
                # Step 1: Calculate numerator (high_diff + low_diff)
                high_diff = data['High'].iloc[i] - data['High'].iloc[i - 2]
                low_diff = data['Low'].iloc[i] - data['Low'].iloc[i - 2]
                numerator = high_diff + low_diff

                # Step 2: Calculate denominator (absolute price move over the 5-bar period)
                highest_high = data['High'].iloc[i - 4:i + 1].max()
                lowest_low = data['Low'].iloc[i - 4:i + 1].min()
                denominator = highest_high - lowest_low

                # Avoid division by zero
                if denominator == 0:
                    td_rei_value = 0
                else:
                    # Step 3: Calculate TD REI
                    td_rei_value = (numerator / denominator) * 100

                # Assign the calculated value
                data.at[data.index[i], 'TD REI'] = td_rei_value

            return data



def refine_td_rei_qualifiers(data):
            """
            Refine TD REI qualifiers using TD POQ rules for buy and sell signals.
            """
            data['TD POQ Signal'] = ''

            for i in range(2, len(data) - 1):
                # Previous and current REI values
                td_rei = data['TD REI'].iloc[i]
                prev_rei = data['TD REI'].iloc[i - 1]

                # Conditions for Buy Signal
                if td_rei < -40 and data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                    if (
                        data['Open'].iloc[i + 1] <= max(data['High'].iloc[i - 2:i]) and
                        data['High'].iloc[i + 1] > max(data['High'].iloc[i - 2:i])
                    ):
                        data.at[data.index[i], 'TD POQ Signal'] = 'Buy Signal'

                # Conditions for Sell Signal
                if td_rei > 40 and data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                    if (
                        data['Open'].iloc[i + 1] >= min(data['Low'].iloc[i - 2:i]) and
                        data['Low'].iloc[i + 1] < min(data['Low'].iloc[i - 2:i])
                    ):
                        data.at[data.index[i], 'TD POQ Signal'] = 'Sell Signal'

            return data



def add_td_rei_alert(data):
            """
            Add an alert for TD REI changes:
            - "alert + to -" when TD REI changes from positive to negative.
            - "alert - to +" when TD REI changes from negative to positive.

            Parameters:
                data (pd.DataFrame): DataFrame containing the 'TD REI' column.

            Returns:
                pd.DataFrame: DataFrame with the 'TD REI Alert' column added.
            """
            # Ensure 'TD REI' column exists
            if 'TD REI' not in data.columns:
                raise ValueError("The DataFrame must contain a 'TD REI' column.")

            # Initialize the alert column with empty strings
            data['TD REI Alert'] = ''

            # Iterate through the DataFrame starting from the second row
            for i in range(1, len(data)):
                current_rei = data['TD REI'].iloc[i]
                previous_rei = data['TD REI'].iloc[i - 1]

                # Check for "alert + to -"
                if previous_rei > 0 and current_rei <= 0:
                    data.at[data.index[i], 'TD REI Alert'] = 'alert + to -'

                # Check for "alert - to +"
                elif previous_rei < 0 and current_rei >= 0:
                    data.at[data.index[i], 'TD REI Alert'] = 'alert - to +'

            return data
def add_td_rei_qualifiers(data):
            """
            Adds qualifiers for TD REI:
            - 'Weakness': TD REI crosses below +40 after being above +40 for fewer than six consecutive bars.
            - 'Strength': TD REI crosses above -40 after being below -40 for fewer than six consecutive bars.

            Parameters:
            - data (pd.DataFrame): Must contain a 'TD REI' column from a prior calculation step.

            Returns:
            - pd.DataFrame: DataFrame with an added 'TD REI Qualifier' column.
            """

            # Initialize the Qualifier column with empty strings
            data['TD REI Qualifier'] = ''

            # Create a shifted column of TD REI for prior values (note the underscore here)
            data['TD_REI_prev'] = data['TD REI'].shift(1)

            # Loop through each row (starting from index 1)
            for i in range(1, len(data)):
                prev_rei = data['TD_REI_prev'].iloc[i]
                current_rei = data['TD REI'].iloc[i]

                # -----------------------------
                # 1. Check for Weakness
                # -----------------------------
                # Condition: previously above 40, now at or below 40
                if (prev_rei > 40) and (current_rei <= 40):
                    count_above_40 = 0
                    # Count how many consecutive bars before this were > 40
                    for j in range(i - 1, -1, -1):
                        if data['TD REI'].iloc[j] > 40:
                            count_above_40 += 1
                        else:
                            break
                    # If the count is fewer than 6, label as Weakness
                    if count_above_40 < 6:
                        data.at[data.index[i], 'TD REI Qualifier'] = 'Weakness'

                # -----------------------------
                # 2. Check for Strength
                # -----------------------------
                # Condition: previously below -40, now at or above -40
                if (prev_rei < -40) and (current_rei >= -40):
                    count_below_40 = 0
                    # Count how many consecutive bars before this were < -40
                    for j in range(i - 1, -1, -1):
                        if data['TD REI'].iloc[j] < -40:
                            count_below_40 += 1
                        else:
                            break
                    # If the count is fewer than 6, label as Strength
                    if count_below_40 < 6:
                        data.at[data.index[i], 'TD REI Qualifier'] = 'Strength'

            # (Optional) Remove the helper column if you no longer need it
            # data.drop(columns=['TD_REI_prev'], inplace=True)

            return data



def refine_td_rei_qualifiers(data):
            """
            Refine TD REI qualifiers using TD POQ rules for buy and sell signals.
            """
            data['TD POQ Signal'] = ''

            for i in range(2, len(data) - 1):
                # Previous and current REI values
                td_rei = data['TD REI'].iloc[i]
                prev_rei = data['TD REI'].iloc[i - 1]

                # Conditions for Buy Signal
                if td_rei < -40 and data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                    if (
                        data['Open'].iloc[i + 1] <= max(data['High'].iloc[i - 2:i]) and
                        data['High'].iloc[i + 1] > max(data['High'].iloc[i - 2:i])
                    ):
                        data.at[data.index[i], 'TD POQ Signal'] = 'Buy Signal'

                # Conditions for Sell Signal
                if td_rei > 40 and data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                    if (
                        data['Open'].iloc[i + 1] >= min(data['Low'].iloc[i - 2:i]) and
                        data['Low'].iloc[i + 1] < min(data['Low'].iloc[i - 2:i])
                    ):
                        data.at[data.index[i], 'TD POQ Signal'] = 'Sell Signal'

            return data


def calculate_td_demand_supply_lines(data):
            """
            Calculate TD Demand and Supply Lines using ringed lows and ringed highs.
            Avoids ambiguous Series comparisons by using NumPy arrays.
            """

            # Initialize columns with NaN instead of None
            data['TD Demand Line'] = np.nan
            data['TD Supply Line'] = np.nan

            # Keep track of ringed lows/highs
            demand_points = []
            supply_points = []

            # Convert to NumPy arrays for scalar comparisons
            low_vals = data['Low'].to_numpy()
            high_vals = data['High'].to_numpy()

            # Loop over rows (except first and last to avoid out-of-bounds)
            for i in range(1, len(data) - 1):
                # Check for a ringed low (TD Point Low)
                # If low[i] is less than both the previous and next lows
                if low_vals[i] < low_vals[i - 1] and low_vals[i] < low_vals[i + 1]:
                    demand_points.append(low_vals[i])  # Store the scalar low
                    # Once we have at least 2 lows, use their max as the demand line
                    if len(demand_points) >= 2:
                        data.at[data.index[i], 'TD Demand Line'] = max(demand_points[-2:])
                    else:
                        data.at[data.index[i], 'TD Demand Line'] = demand_points[-1]

                # Check for a ringed high (TD Point High)
                # If high[i] is greater than both the previous and next highs
                if high_vals[i] > high_vals[i - 1] and high_vals[i] > high_vals[i + 1]:
                    supply_points.append(high_vals[i])  # Store the scalar high
                    # Once we have at least 2 highs, use their min as the supply line
                    if len(supply_points) >= 2:
                        data.at[data.index[i], 'TD Supply Line'] = min(supply_points[-2:])
                    else:
                        data.at[data.index[i], 'TD Supply Line'] = supply_points[-1]

            # Forward-fill lines to extend them until the next update
            data['TD Demand Line'] = data['TD Demand Line'].ffill()
            data['TD Supply Line'] = data['TD Supply Line'].ffill()

            return data

def detect_close_crosses(data):
            """
            Detects when the Closing Price crosses above or below the TD Supply Line (TSL) or TD Demand Line (TDL).
            Adds an 'Alert' column with the type of crossover.

            Parameters:
                data (pd.DataFrame): DataFrame containing 'Close', 'TD Supply Line', and 'TD Demand Line'.

            Returns:
                pd.DataFrame: DataFrame with an added 'Alert' column.
            """
            # Initialize the Alert column with empty strings
            data['Alert'] = ''

            # Check for crossing above TSL
            data.loc[(data['Close'].shift(1) <= data['TD Supply Line'].shift(1)) & (data['Close'] > data['TD Supply Line']), 'Alert'] = 'Cross Up TSL'

            # Check for crossing below TSL
            data.loc[(data['Close'].shift(1) >= data['TD Supply Line'].shift(1)) & (data['Close'] < data['TD Supply Line']), 'Alert'] = 'Cross Down TSL'

            # Check for crossing above TDL
            data.loc[(data['Close'].shift(1) <= data['TD Demand Line'].shift(1)) & (data['Close'] > data['TD Demand Line']), 'Alert'] = 'Cross Up TDL'

            # Check for crossing below TDL
            data.loc[(data['Close'].shift(1) >= data['TD Demand Line'].shift(1)) & (data['Close'] < data['TD Demand Line']), 'Alert'] = 'Cross Down TDL'

            return data

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


def calculate_setup_qualifier(data):
            """
            Determines whether a completed Buy Setup or Sell Setup is qualified or disqualified.
            - Buy Setup is "Qualified" if the last 3 lows are below the low of bar 6 in the setup.
            - Sell Setup is "Qualified" if the last 3 highs are above the high of bar 6 in the setup.

            Parameters:
            - data (pd.DataFrame): Must contain 'Buy Setup', 'Sell Setup', 'Low', 'High'.

            Returns:
            - pd.DataFrame with a new 'Setup Qualifier' column.
            """

            data['Setup Qualifier'] = None  # Initialize the column

            for i in range(8, len(data)):  # Start from the 9th bar (index 8 in zero-based indexing)
                # Check for Buy Setup Qualification
                if data['Buy Setup'].iloc[i] == "Buy Setup Completed":
                    low_6 = data['Low'].iloc[i - 3]  # Low of the 6th bar in the setup
                    if (
                        data['Low'].iloc[i - 2] < low_6 and
                        data['Low'].iloc[i - 1] < low_6 and
                        data['Low'].iloc[i] < low_6
                    ):
                        data.loc[i, 'Setup Qualifier'] = "Qualified"
                    else:
                        data.loc[i, 'Setup Qualifier'] = "Disqualified"

                # Check for Sell Setup Qualification
                elif data['Sell Setup'].iloc[i] == "Sell Setup Completed":
                    high_6 = data['High'].iloc[i - 3]  # High of the 6th bar in the setup
                    if (
                        data['High'].iloc[i - 2] > high_6 and
                        data['High'].iloc[i - 1] > high_6 and
                        data['High'].iloc[i] > high_6
                    ):
                        data.loc[i, 'Setup Qualifier'] = "Qualified"
                    else:
                        data.loc[i, 'Setup Qualifier'] = "Disqualified"

            return data  # Return after processing all rows

import pandas as pd
import numpy as np

def calculate_td_countdown(data):
    """
    Implements TD Countdown after TD Sequential Setup is completed.
    - Buy Countdown: Closes <= Low[2 bars ago]
    - Sell Countdown: Closes >= High[2 bars ago]
    - Stops at 13

    Parameters:
    - data (pd.DataFrame): Must contain 'Buy Setup', 'Sell Setup', 'Close', 'Low', 'High' columns.

    Returns:
    - pd.DataFrame with 'Buy Countdown' and 'Sell Countdown' columns.
    """

    # Ensure required columns exist
    required_cols = ['Buy Setup', 'Sell Setup', 'Close', 'Low', 'High']
    if not all(col in data.columns for col in required_cols):
        raise ValueError("DataFrame is missing required columns.")

    # Initialize Countdown columns
    data['Buy Countdown'] = 0
    data['Sell Countdown'] = 0

    # Find indices where Buy/Sell Setup is completed
    buy_setup_indices = data.index[data['Buy Setup'] == "Buy Setup Completed"].tolist()
    sell_setup_indices = data.index[data['Sell Setup'] == "Sell Setup Completed"].tolist()

    # Initialize countdown tracking
    buy_count = 0
    sell_count = 0
    buy_active = False
    sell_active = False

    for i in range(len(data)):
        # Start Buy Countdown
        if i in buy_setup_indices:
            buy_active = True
            sell_active = False  # Reset sell countdown
            buy_count = 0  # Start at 0 (to increment properly on the next valid bar)
            continue

        # Start Sell Countdown
        if i in sell_setup_indices:
            sell_active = True
            buy_active = False  # Reset buy countdown
            sell_count = 0  # Start at 0 (to increment properly on the next valid bar)
            continue

        # Increment Buy Countdown if active & Close <= Low[2 bars ago]
        if buy_active and i >= 2:
            if data['Close'].iloc[i] <= data['Low'].iloc[i - 2]:
                buy_count += 1
                data.loc[i, 'Buy Countdown'] = buy_count
                if buy_count == 13:  # Stop at 13
                    buy_active = False
                    buy_count = 0

        # Increment Sell Countdown if active & Close >= High[2 bars ago]
        if sell_active and i >= 2:
            if data['Close'].iloc[i] >= data['High'].iloc[i - 2]:
                sell_count += 1
                data.loc[i, 'Sell Countdown'] = sell_count
                if sell_count == 13:  # Stop at 13
                    sell_active = False
                    sell_count = 0

    return data


def calculate_td_combo_countdown(data):
            """
            Implements TD Combo Countdown after TD Sequential Setup is completed.
            - Buy Combo Countdown: Closes < Low[2 bars ago]
            - Sell Combo Countdown: Closes > High[2 bars ago]
            - Stops at 13 unless a new TD Setup appears.

            Parameters:
            - data (pd.DataFrame): Must contain 'Buy Setup', 'Sell Setup', 'Close', 'Low', 'High' columns.

            Returns:
            - pd.DataFrame with 'Buy Combo Countdown' and 'Sell Combo Countdown' columns.
            """

            # Initialize Countdown columns properly as integers
            data['Buy Combo Countdown'] = 0
            data['Sell Combo Countdown'] = 0

            # Flags to track active countdowns
            buy_countdown_active = False
            sell_countdown_active = False
            buy_count = 0
            sell_count = 0

            for i in range(len(data)):
                # Reset Buy Combo Countdown if a new Sell Setup appears
                if data['Sell Setup'].iloc[i] == "Sell Setup Completed":
                    buy_countdown_active = False
                    buy_count = 0

                # Reset Sell Combo Countdown if a new Buy Setup appears
                if data['Buy Setup'].iloc[i] == "Buy Setup Completed":
                    sell_countdown_active = False
                    sell_count = 0

                # Activate Buy Combo Countdown when Buy Setup is completed
                if data['Buy Setup'].iloc[i] == "Buy Setup Completed":
                    buy_countdown_active = True
                    sell_countdown_active = False  # Reset opposing countdown
                    buy_count = 1  # Start at 1 (Corrected)
                    continue

                # Activate Sell Combo Countdown when Sell Setup is completed
                if data['Sell Setup'].iloc[i] == "Sell Setup Completed":
                    sell_countdown_active = True
                    buy_countdown_active = False  # Reset opposing countdown
                    sell_count = 1  # Start at 1 (Corrected)
                    continue

                # Increment Buy Combo Countdown if active & Close < Low[2 bars ago]
                if buy_countdown_active and i >= 2:  # Ensure sufficient prior data
                    if data['Close'].iloc[i] < data['Low'].iloc[i - 2]:
                        data.loc[i, 'Buy Combo Countdown'] = buy_count
                        buy_count += 1
                        if buy_count > 13:  # Stop at 13
                            buy_countdown_active = False
                            buy_count = 0

                # Increment Sell Combo Countdown if active & Close > High[2 bars ago]
                if sell_countdown_active and i >= 2:
                    if data['Close'].iloc[i] > data['High'].iloc[i - 2]:
                        data.loc[i, 'Sell Combo Countdown'] = sell_count
                        sell_count += 1
                        if sell_count > 13:  # Stop at 13
                            sell_countdown_active = False
                            sell_count = 0

            return data

def calculate_bollinger_band_width(data, window=20, multiplier=2):
    """
    Calculate Bollinger Band Width (BBW) as a percentage.

    Parameters:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        window (int): Rolling window for SMA and standard deviation. Default is 20.
        multiplier (int): Multiplier for the standard deviation. Default is 2.

    Returns:
        pd.DataFrame: DataFrame with an added 'BBW' column.
    """
    # Calculate the rolling mean (Middle Band)
    middle_band = data['Close'].rolling(window=window).mean()

    # Calculate the rolling standard deviation
    rolling_std = data['Close'].rolling(window=window).std()

    # Calculate the Upper and Lower Bands
    upper_band = middle_band + (multiplier * rolling_std)
    lower_band = middle_band - (multiplier * rolling_std)

    # Calculate Bollinger Band Width (BBW) as a percentage
    data['BBW'] = ((upper_band - lower_band) / middle_band) * 100

    return data
# ─────────────────────────────────────────────────────────
# SINGLE-TICKER BLOCK
if st.sidebar.button("Fetch One Ticker"):
    try:
        intraday = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=timeframe,
            progress=False
        )

        if intraday.empty:
            st.error(f"No data fetched for ticker {ticker}. Please check inputs.")
            st.stop()

        # Flatten multi-index columns if any
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

        # Reset index to have a 'Date' column
        intraday.reset_index(inplace=True)
        intraday.rename(columns={"index": "Date"}, inplace=True)
        if "Date" not in intraday.columns:
            intraday.rename(columns={intraday.columns[0]: "Date"}, inplace=True)

        # Example transformations
        intraday["OC"] = intraday["Close"] - intraday["Open"]
        intraday["CC"] = intraday["Close"].diff()
        intraday["Range"] = intraday["High"] - intraday["Low"]
        intraday = calculate_sd_of_change_cc(intraday)
        intraday['Dollar Change'] = intraday["Close"].diff()
        intraday = detect_kijun_cross(intraday)
        intraday = calculate_td_sequential(intraday)
        intraday = add_td_pressure_alert(intraday)
        intraday = calculate_td_rei(intraday)
        intraday = refine_td_rei_qualifiers(intraday)
        intraday = add_td_rei_qualifiers(intraday)
        intraday = calculate_bollinger_band_width(intraday)

        # Show final result
        st.dataframe(intraday.tail(5), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ─────────────────────────────────────────────────────────
# MULTI-TICKER BLOCK
if st.sidebar.button("Fetch Multiple Tickers"):
    try:
        tickers = ["SPY", "QQQ", "NVDA","AMZN","AAPL","MSFT","AMD","AVGO","MU","GOOGL","MU","PLTR","MRVL","META","NFLX"]
        for t in tickers:
            st.subheader(f"{t}")

            intraday = yf.download(
                t,
                start=start_date,
                end=end_date,
                interval=timeframe,
                progress=False
            )

            if intraday.empty:
                st.error(f"No data fetched for {t}.")
                continue

            # Flatten multi-index columns if any
            if isinstance(intraday.columns, pd.MultiIndex):
                intraday.columns = intraday.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

            # Reset index to have a 'Date' column
            intraday.reset_index(inplace=True)
            intraday.rename(columns={"index": "Date"}, inplace=True)
            if "Date" not in intraday.columns:
                intraday.rename(columns={intraday.columns[0]: "Date"}, inplace=True)
            period = 52
            average_volume = intraday['Volume'].rolling(window=period, min_periods=1).mean()
            intraday['RV'] = intraday['Volume'] / average_volume
            # Reuse transformations
            intraday["OC"] = intraday["Close"] - intraday["Open"]
            intraday["CC"] = intraday["Close"].diff()
            intraday["Range"] = intraday["High"] - intraday["Low"]
            intraday = calculate_sd_of_change_cc(intraday)
            intraday["Dollar Change"] = intraday["Close"].diff()
            intraday['Pct Change'] = intraday['Close'].pct_change() * 100
            intraday['Kijun'] = (intraday['High'].rolling(window=26).max() + intraday['Low'].rolling(window=26).min()) / 2
            intraday = calculate_td_sequential(intraday)
            intraday = calculate_setup_qualifier(intraday)
            intraday = calculate_td_countdown(intraday)
            intraday = calculate_td_combo_countdown(intraday)

            intraday =  calculate_td_pressure(intraday)
            intraday = add_td_pressure_alert(intraday)
            intraday = calculate_td_rei(intraday)
            intraday = add_td_rei_alert(intraday)
            intraday = add_td_rei_qualifiers(intraday)
            intraday = refine_td_rei_qualifiers(intraday)
            intraday = calculate_td_demand_supply_lines(intraday)
            intraday = detect_close_crosses(intraday)

            # Display the last row or however many you prefer
            # Define the columns you want to hide
            columns_to_hide = ["OC","Log_Change_CC","Volume","Rolling_StdDev_CC", "CC", "Range", "Kijun",'Adj Close',"High", "Low","Open"]

            # Display the dataframe without the unwanted columns
            st.dataframe(intraday.drop(columns=columns_to_hide, errors="ignore"), use_container_width=True)


    except Exception as e:
        st.error(f"An error occurred while fetching multiple tickers: {e}")
