import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta  # If you haven’t installed it: pip install ta


# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 20)


st.set_page_config(layout="wide")  # Default layout


st.title("Day Trading")




# Sidebar for user input
st.sidebar.header("Input Options")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")
timeframe = st.sidebar.selectbox(
    "Select Timeframe:",
    options=["2m", "5m", "15m", "30m", "60m", "1d"],
    index=1  # Default to "15m"
)

# ─────────────────────────────────────────────────────────
# Function: Calculate Standard Deviations of Change (Close-to-Close)
def calculate_sd_of_change_cc(data, window=13):
    data = data.copy()
    data['Log_Change_CC'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Rolling_StdDev_CC'] = data['Log_Change_CC'].rolling(window=window).std()
    data['SD'] = data['Log_Change_CC'] / data['Rolling_StdDev_CC']
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data[['Log_Change_CC','Rolling_StdDev_CC','SD']] = data[
        ['Log_Change_CC','Rolling_StdDev_CC','SD']
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
            data['Lines Alert'] = ''

            # Check for crossing above TSL
            data.loc[(data['Close'].shift(1) <= data['TD Supply Line'].shift(1)) & (data['Close'] > data['TD Supply Line']), 'Lines Alert'] = 'Cross Up TSL'

            # Check for crossing below TSL
            data.loc[(data['Close'].shift(1) >= data['TD Supply Line'].shift(1)) & (data['Close'] < data['TD Supply Line']), 'Lines Alert'] = 'Cross Down TSL'

            # Check for crossing above TDL
            data.loc[(data['Close'].shift(1) <= data['TD Demand Line'].shift(1)) & (data['Close'] > data['TD Demand Line']), 'Lines Alert'] = 'Cross Up TDL'

            # Check for crossing below TDL
            data.loc[(data['Close'].shift(1) >= data['TD Demand Line'].shift(1)) & (data['Close'] < data['TD Demand Line']), 'Lines Alert'] = 'Cross Down TDL'

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

def calculate_td_demarker_ii(data):
    data['TD DeMarker II'] = np.nan  # Initialize the column with NaN

    for i in range(8, len(data)):  # Start at the 9th row since we need 8 prior periods
        # Numerator
        high_diff = max(data['High'].iloc[i] - data['Close'].iloc[i - 1], 0)
        true_high_diff = max(data['Close'].iloc[i - 1] - data['High'].iloc[i], 0)
        numerator = high_diff + true_high_diff

        # Denominator
        low_diff = max(data['Close'].iloc[i - 1] - data['Low'].iloc[i], 0)
        denominator = numerator + low_diff

        # Calculate TD DeMarker II (avoid division by zero)
        if denominator != 0:
            data.at[data.index[i], 'TD DeMarker II'] = numerator / denominator
        else:
            data.at[data.index[i], 'TD DeMarker II'] = np.nan

    return data






# ─────────────────────────────────────────────────────────
def calculate_theta_tenkan_kijun(data):
    """
    Calculates Theta (angle of slope) for Tenkan and Kijun using vectorized operations.
    """
    # Ensure Tenkan and Kijun columns are present
    if 'Tenkan' not in data or 'Kijun' not in data:
        raise ValueError("Tenkan and Kijun columns are required.")

    # Calculate slope (rise/run) for Tenkan
    data['Theta_Tenkan'] = np.degrees(np.arctan(data['Tenkan'].diff() / 1))  # Run is always 1 (time difference)

    # Calculate slope (rise/run) for Kijun
    data['Theta_Kijun'] = np.degrees(np.arctan(data['Kijun'].diff() / 1))  # Run is always 1

    # Handle first row (NaN due to diff) if needed
    data['Theta_Tenkan'].iloc[0] = np.nan
    data['Theta_Kijun'].iloc[0] = np.nan

    return data

def calculate_cotangent(theta_series):
    """
    Calculate the cotangent for a series of theta values (in degrees).

    Parameters:
        theta_series (pd.Series): A pandas Series containing theta values in degrees.

    Returns:
        pd.Series: A pandas Series containing cotangent values.
    """
    # Convert degrees to radians
    theta_radians = np.radians(theta_series)

    # Compute tangent values
    tangent_values = np.tan(theta_radians)

    # Compute cotangent values, handling division by zero
    cotangent_values = np.where(
        np.abs(tangent_values) < np.finfo(float).eps,  # Near-zero tangent values
        np.nan,  # Assign NaN to avoid division by zero
        1 / tangent_values
    )

    return pd.Series(cotangent_values, index=theta_series.index)
def calculate_sine(values, frequency, time_index):
    """
    Calculate sine wave values using given data, frequency, and time index.
    """
    return np.sin(2 * np.pi * frequency * time_index + values)

def calculate_cosecant(sine_values):
    """
    Calculate cosecant values from sine values.
    Handle cases where sine is zero to avoid division by zero.
    """
    return np.where(
        sine_values == 0,
        np.nan,  # Avoid division by zero by returning NaN
        1 / sine_values
    )
def calculate_cosine(values, frequency, time_index):
    """
    Calculate cosine wave values using given data, frequency, and time index.
    """
    return np.cos(2 * np.pi * frequency * time_index + values)

def calculate_secant(cosine_values):
    """
    Calculate secant values from cosine values.
    Handle cases where cosine is zero to avoid division by zero.

    Parameters:
        cosine_values (pd.Series): A pandas Series containing cosine values.

    Returns:
        pd.Series: A pandas Series containing secant values.
    """
    return np.where(
        cosine_values == 0,
        np.nan,  # Avoid division by zero by returning NaN
        1 / cosine_values
    )



    return pd.Series(cotangent_values, index=theta_series.index)

def calculate_directional_flatness(data):
    """
    Calculate directional flatness based on cotangent values for Tenkan and Kijun.
    Rows will remain blank unless the value is explicitly True.
    """
    # Initialize the column with empty strings
    data['Flat'] = ''

    # Apply the directional flatness logic
    data.loc[
        (data['Cotangent_Tenkan'].abs() > 100) | (data['Cotangent_Kijun'].abs() > 100),
        'Flat'
    ] = 'True'

    return data

def calculate_instability(cosecant_tenkan, cosecant_kijun):
    """
    Calculate instability based on cosecant values for Tenkan and Kijun.

    Parameters:
        cosecant_tenkan (pd.Series): Series of cosecant values for Tenkan.
        cosecant_kijun (pd.Series): Series of cosecant values for Kijun.

    Returns:
        pd.Series: A Series with True where instability is detected, and blank otherwise.
    """
    # Calculate instability values
    instability_values = np.where(
        (np.abs(cosecant_tenkan) > 10) | (np.abs(cosecant_kijun) > 10),
        True,
        ""
    )
    return pd.Series(instability_values, index=cosecant_tenkan.index)

def calculate_secant_insta(secant_tenkan, secant_kijun):
    """
    Calculate secant instability based on secant values for Tenkan and Kijun.

    Parameters:
        secant_tenkan (pd.Series): Series of secant values for Tenkan.
        secant_kijun (pd.Series): Series of secant values for Kijun.

    Returns:
        pd.Series: A Series with True where instability is detected, and blank otherwise.
    """
    # Calculate secant instability values
    secant_insta_values = np.where(
        (np.abs(secant_tenkan) > 10) | (np.abs(secant_kijun) > 10),
        True,
        ""
    )
    return pd.Series(secant_insta_values, index=secant_tenkan.index)
def calculate_relative_volume_alert(data, threshold=0.95):
    """
    Add a Relative Volume Alert column with blank rows unless the condition is True.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'RV' (Relative Volume) column.
        threshold (float): Threshold for triggering the alert. Default is 0.75.

    Returns:
        pd.DataFrame: DataFrame with the 'Relative Volume Alert' column added.
    """
    # Ensure the 'RV' column exists
    if 'RV' not in data.columns:
        raise ValueError("The DataFrame must contain an 'RV' column.")

    # Create the alert column with blank rows unless the condition is met
    data['RV Alert'] = data['RV'].apply(
        lambda x: True if x > threshold else ''
    )

    return data
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def add_dmi_alerts(data):
    """
    Adds a single 'DMI Alert' column with the appropriate Buy/Sell state.
    Removes intermediate indicator columns.
    """

    # Ensure required columns exist
    required_cols = ['DI+', 'DI-', 'ADX']
    if not all(col in data.columns for col in required_cols):
        raise ValueError("Missing required columns: DI+, DI-, ADX")

    # Compute indicators but don't store them
    buy_indicator_1 = (data['DI+'] > data['DI-']) & (data['DI+'].shift(1) <= data['DI-'].shift(1))
    sell_indicator_1 = (data['DI-'] > data['DI+']) & (data['DI-'].shift(1) <= data['DI+'].shift(1))

    buy_indicator_2 = buy_indicator_1 & (data['ADX'] > 20)
    sell_indicator_2 = sell_indicator_1 & (data['ADX'] > 20)

    buy_indicator_3 = buy_indicator_2 & ~((data['ADX'] > data['DI+']) & (data['ADX'] > data['DI-']))
    sell_indicator_3 = sell_indicator_2 & ~((data['ADX'] > data['DI+']) & (data['ADX'] > data['DI-']))

    # Create the single 'DMI Alert' column
    data['DMI Alert'] = np.select(
        [
            buy_indicator_3, buy_indicator_2, buy_indicator_1,
            sell_indicator_3, sell_indicator_2, sell_indicator_1
        ],
        [
            "Buy Indicator III Completed", "Buy Indicator II", "Buy Indicator I",
            "Sell Indicator III Completed", "Sell Indicator II", "Sell Indicator I"
        ],
        default=""  # Empty if no alert
    )

    return data


# ─────────────────────────────────────────────────────────
# MULTI-TICKER BLOCK
if st.sidebar.button("Fetch Multiple Tickers"):
    try:
        tickers = ["SPY", "QQQ", "NVDA","AMZN","AAPL","MSFT","AMD","AVGO","MU","GOOGL","PLTR","MRVL","META","NFLX"]
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
                    # Ensure Date column is properly set
            intraday.reset_index(inplace=True)  # Reset index to convert it into a column

            # Ensure 'Datetime' is explicitly converted to 'Date'
            if "Datetime" in intraday.columns:
                intraday.rename(columns={"Datetime": "Date"}, inplace=True)

            # Ensure the 'Date' column is present; otherwise, use the index
            if "Date" not in intraday.columns:
                intraday["Date"] = intraday.index  # Assign index as Date if missing

            # ✅ Convert directly to New York Time (since it's already timezone-aware)
            intraday['Date'] = intraday['Date'].dt.tz_convert('America/New_York')
# Print the first and last timestamps
            st.write("First timestamp:", intraday['Date'].iloc[0])
            st.write("Last timestamp:", intraday['Date'].iloc[-1])
            # ✅ Extract only the time in 12-hour format (HH:MM AM/PM)
            intraday['Time'] = intraday['Date'].dt.strftime('%I:%M %p')


            period = 52
            average_volume = intraday['Volume'].rolling(window=period, min_periods=1).mean()
            intraday['RV'] = intraday['Volume'] / average_volume
            intraday = calculate_relative_volume_alert(intraday)

            # Reuse transformations
            intraday["OC"] = intraday["Close"] - intraday["Open"]
            intraday["CC"] = intraday["Close"].diff()
            intraday["Range"] = intraday["High"] - intraday["Low"]
            intraday = calculate_sd_of_change_cc(intraday)
            intraday["$"] = intraday["Close"].diff()
            intraday['%'] = intraday['Close'].pct_change() * 100
            intraday['Kijun'] = (intraday['High'].rolling(window=26).max() + intraday['Low'].rolling(window=26).min()) / 2
            intraday['ADX'] = ta.trend.adx(intraday['High'], intraday['Low'], intraday['Close'], window=14)
            intraday['DI+'] = ta.trend.adx_pos(intraday['High'], intraday['Low'], intraday['Close'], window=14)
            intraday['DI-'] = ta.trend.adx_neg(intraday['High'], intraday['Low'], intraday['Close'], window=14)
            intraday = add_dmi_alerts(intraday)
            intraday  =   calculate_kijun_sen(intraday)
            intraday = detect_kijun_cross(intraday)
            intraday = calculate_td_sequential(intraday)
            intraday = calculate_setup_qualifier(intraday)
            intraday = calculate_td_countdown(intraday)
            intraday = calculate_td_combo_countdown(intraday)

            # 🟢 Ensure TD REI Calculation is Done and friends
            intraday = calculate_td_rei(intraday)
            intraday = calculate_td_rei(intraday)
            intraday = add_td_rei_alert(intraday)
            intraday = add_td_rei_qualifiers(intraday)
            intraday = refine_td_rei_qualifiers(intraday)

            intraday = calculate_td_demarker_ii(intraday)

            intraday =  calculate_td_pressure(intraday)
            intraday = add_td_pressure_alert(intraday)


            intraday = calculate_td_demand_supply_lines(intraday)
            intraday = detect_close_crosses(intraday)
            intraday = calculate_bollinger_band_width(intraday)
            intraday['Tenkan'] = (intraday['High'].rolling(window=9).max() + intraday['Low'].rolling(window=9).min()) / 2
            intraday['Kijun'] = (intraday['High'].rolling(window=26).max() + intraday['Low'].rolling(window=26).min()) / 2
            intraday = calculate_theta_tenkan_kijun(intraday)
            intraday = calculate_theta_tenkan_kijun(intraday)  # Function defined earlier
            intraday['Cotangent_Tenkan'] = calculate_cotangent(intraday['Theta_Tenkan'])
            intraday['Cotangent_Kijun'] = calculate_cotangent(intraday['Theta_Kijun'])
            intraday = calculate_directional_flatness(intraday)
        # Add Sine Tenkan and Sine Kijun
            frequency = 1 / 3  # One cycle every 8 periods
            time_index = np.arange(len(intraday))
            intraday['Sine_Tenkan'] = calculate_sine(intraday['Tenkan'], frequency, time_index)
            intraday['Sine_Kijun'] = calculate_sine(intraday['Kijun'], frequency, time_index)
            # Calculate Theta Tenkan and Theta Kijun
                # Calculate Cosecant for Tenkan
            intraday['Cosecant_Tenkan'] = calculate_cosecant(intraday['Sine_Tenkan'])
            intraday['Cosecant_Kijun'] = calculate_cosecant(intraday['Sine_Kijun'])

            intraday['Verti'] = calculate_instability(
            intraday['Cosecant_Tenkan'], intraday['Cosecant_Kijun']
)
            intraday['Cosine_Tenkan'] = calculate_cosine(intraday['Tenkan'], frequency, time_index)
            intraday['Cosine_Kijun'] = calculate_cosine(intraday['Kijun'], frequency, time_index)



            # Add Secant Tenkan and Secant Kijun
            intraday['Secant_Tenkan'] = calculate_secant(intraday['Cosine_Tenkan'])
            intraday['Secant_Kijun'] = calculate_secant(intraday['Cosine_Kijun'])
            intraday['Hori'] = calculate_secant_insta(
            intraday['Secant_Tenkan'], intraday['Secant_Kijun']
        )
            intraday  =   calculate_kijun_sen(intraday)


            intraday['Tenkan'] = (intraday['High'].rolling(window=9).max() + intraday['Low'].rolling(window=9).min()) / 2

            # Kijun-sen (Base Line)
            intraday['Kijun_ichimoku'] = (intraday['High'].rolling(window=26).max() + intraday['Low'].rolling(window=26).min()) / 2

            # Senkou Span A (Leading Span A)
            intraday['Senkou_Span_A'] = ((intraday['Tenkan'] + intraday['Kijun_ichimoku']) / 2).shift(26)

            # Senkou Span B (Leading Span B)
            intraday['Senkou_Span_B'] = ((intraday['High'].rolling(window=52).max() + intraday['Low'].rolling(window=52).min()) / 2).shift(26)

            # Chikou Span (Lagging Span)
            intraday['Chikou_Span'] = intraday['Close'].shift(-26)

            # -------------------------------
            # NEW: Calculate Bollinger Bands
            # -------------------------------
            intraday['Middle_Band'] = intraday['Close'].rolling(window=20).mean()
            intraday['Std_Dev'] = intraday['Close'].rolling(window=20).std()
            intraday['Upper_Band'] = intraday['Middle_Band'] + (intraday['Std_Dev'] * 2)
            intraday['Lower_Band'] = intraday['Middle_Band'] - (intraday['Std_Dev'] * 2)

            # Calculate ADX, DI+, DI-


            # 🟢 Define Function to Generate the TD REI Plot
            def plot_td_rei_chart(intraday):
                fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,  # Adjust rows to 7
            row_heights=[0.6, 0.3,0.6],  # Adjust row heights as needed
            vertical_spacing=0.06,
            subplot_titles=("Price Action","Volume","DMI")
                    )
                # 🔹 1. Add Candlestick Chart for Price Action
                fig.add_trace(
                    go.Candlestick(
                        x=intraday['Date'],
                        open=intraday['Open'],
                        high=intraday['High'],
                        low=intraday['Low'],
                        close=intraday['Close'],
                        name="Candlestick",

                    ),
                    row=1, col=1
                )

                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Tenkan'],
                            mode='lines',
                            line=dict(color='red', width=1),
                            name='Tenkan-sen'
                        ),
                        row=1, col=1
                    )


                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Kijun_ichimoku'],
                            mode='lines',
                            line=dict(color='green', width=1),
                            name='Kijun-sen'
                        ),
                        row=1, col=1
                    )
                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Chikou_Span'],
                            mode='lines',
                            line=dict(color='purple', width=1),
                            name='Chikou Span'
                        ),
                        row=1, col=1
                    )

                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Senkou_Span_A'],
                            mode='lines',
                            line=dict(color='orange', width=1),
                            name='Senkou Span A'
                        ),
                        row=1, col=1
                    )

                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Senkou_Span_B'],
                            mode='lines',
                            line=dict(color='blue', width=1),
                            name='Senkou Span B'
                        ),
                        row=1, col=1
                    )

                            # Create the Kumo Cloud (fill between Senkou Span A and B)
                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Senkou_Span_A'],
                            fill=None,
                            mode='lines',
                            line=dict(color='orange', width=0),
                            showlegend=False
                        ),
                    row=1, col=1

                    )

                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Senkou_Span_B'],
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='blue', width=0),
                            fillcolor='rgba(128, 128, 128, 0.2)',
                            name='Kumo Cloud'
                        ),
                        row=1, col=1
                    )

            # -------------------------------
                # Add Bollinger Bands
                # -------------------------------
                fig.add_trace(
                    go.Scatter(
                        x=intraday['Date'],
                        y=intraday['Upper_Band'],
                        mode='lines',
                        line=dict(color='grey', width=1, dash='dot'),
                        name='Upper Bollinger Band'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Middle_Band'],
                            mode='lines',
                            line=dict(color='#ccc', width=1, dash='dot'),
                            name='Middle Bollinger Band'
                        ),
                        row=1, col=1
                    )
                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['Lower_Band'],
                            mode='lines',
                            line=dict(color='grey', width=1, dash='dot'),
                            name='Lower Bollinger Band'
                        ),
                        row=1, col=1
                    )


                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=intraday.index,
                        y=intraday['Volume'],
                        name="Volume",
                        marker_color='blue'
                    ),
                    row=2, col=1
                )
            # # Add TD REI line
            #     fig.add_trace(
            #         go.Scatter(
            #             x=intraday['Date'],
            #             y=intraday['TD REI'],
            #             mode='lines',
            #             name="TD REI Smoothed",
            #             line=dict(color='purple')
            #         ),
            #         row=3, col=1
            #     )
            #     fig.add_trace(
            #         go.Scatter(
            #          x=intraday['Date'],
            #         y=intraday['Close'],
            #         mode='lines',
            #         name='Closing Price',
            #         line=dict(color='gray')  # Blue solid line for Closing Price
            #     ),
            #         row=4, col=1
            # )
                fig.add_trace(
                    go.Scatter(
                         x=intraday['Date'],
                        y=intraday['Kijun-sen'],
                        mode='lines',
                        name='Kijun-sen',
                        line=dict(color='orange', dash='dash')  # Orange dashed line for Kijun-sen
                    ),
                    row=2, col=1
                    )

                #             # 6. TD DeMarker II
                # fig.add_trace(
                #     go.Scatter(
                #          x=intraday['Date'],
                #         y=intraday['TD DeMarker II'],
                #         mode='lines',
                #         name='TD DeMarker II',
                #         line=dict(color='green', dash='dash')  # Green dashed line
                #     ),
                #     row=5, col=1
                # )
            #                 # 5. TD Pressure
            #     # fig.add_trace(
            #     #     go.Scatter(
            #     #          x=intraday['Date'],
            #     #         y=intraday['TD Pressure'],
            #     #         mode='lines',
            #     #         name='TD Pressure',
            #     #         line=dict(color='blue', dash='solid')  # Blue solid line
            #     #     ),
            #     #     row=6, col=1
            #     # )
            #     fig.add_trace(
            #         go.Scatter(
            #              x=intraday['Date'],
            #             y=intraday['TD Supply Line'],
            #             mode='lines',
            #             name='TD Supply Line',
            #             line=dict(color='salmon', dash='dot')  # Red dotted line for Supply
            #         ),
            #         row=7, col=1
            #     )


            #     fig.add_trace(
            #         go.Scatter(
            #             x=intraday['Date'],
            #             y=intraday['TD Demand Line'],
            #             mode='lines',
            #             name='TD Demand Line',
            #             line=dict(color='lightblue', dash='dot')  # Blue dotted line for Demand
            #         ),
            #         row=7, col=1
            #     )
            #     fig.add_trace(
            #         go.Scatter(
            #          x=intraday['Date'],
            #         y=intraday['Close'],
            #         mode='lines',
            #         name='Closing Price',
            #         line=dict(color='gray')  # Blue solid line for Closing Price
            #     ),
            #         row=7, col=1
            # )

            #     # 🔹 8. ADD **BBW** (Bollinger Band Width)
            #     fig.add_trace(
            #         go.Scatter(
            #             x=intraday['Date'],
            #             y=intraday['BBW'],
            #             mode='lines',
            #             name='Bollinger Band Width (BBW)',
            #             line=dict(color='magenta')  # Magenta to stand out
            #         ),
            #         row=8, col=1
            #     )

                fig.add_trace(
                    go.Scatter(
                        x=intraday['Date'],
                        y=intraday['ADX'],
                        mode='lines',
                        name='ADX',
                        line=dict(color='green')
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['DI+'],
                            mode='lines',
                            name='DI+ (Bullish)',
                            line=dict(color='blue')
                        ),
                        row=3, col=1
                    )

                fig.add_trace(
                        go.Scatter(
                            x=intraday['Date'],
                            y=intraday['DI-'],
                            mode='lines',
                            name='DI- (Bearish)',
                            line=dict(color='red')
                        ),
                        row=3, col=1
                    )

                # 🔹 Update Layout for Better Visualization
                fig.update_layout(
                    height=1500, width=1200,
                    title=f"Intraday Chart - {t}",
                    xaxis_title="Time",
                    yaxis_title='Price',
                    showlegend=True,
                    template='plotly_white',
                    xaxis=dict(
                    rangeslider=dict(visible=False)  # Disable range slider
    )

                )

                return fig

            # 🟢 Render Plotly Chart in Streamlit
            # st.write("### 📊 Charting")
            rei_chart = plot_td_rei_chart(intraday)
            st.plotly_chart(rei_chart, use_container_width=True)


            if "Date" not in intraday.columns:
                intraday.rename(columns={intraday.columns[0]: "Date"}, inplace=True)

            # Check if the 'Date' column is timezone-aware.
            if intraday['Date'].dt.tz is None:
                # First localize it to UTC, then convert to New York time.
                intraday['Date'] = intraday['Date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            else:
                # It's already timezone-aware.
                intraday['Date'] = intraday['Date'].dt.tz_convert('America/New_York')

            # ✅ Extract only the time in 12-hour format (HH:MM AM/PM)
            intraday['Time'] = intraday['Date'].dt.strftime('%I:%M %p')



#             period = 52
#             average_volume = intraday['Volume'].rolling(window=period, min_periods=1).mean()
#             intraday['RV'] = intraday['Volume'] / average_volume
#             intraday = calculate_relative_volume_alert(intraday)

#             # Reuse transformations
#             intraday["OC"] = intraday["Close"] - intraday["Open"]
#             intraday["CC"] = intraday["Close"].diff()
#             intraday["Range"] = intraday["High"] - intraday["Low"]
#             intraday = calculate_sd_of_change_cc(intraday)
#             # intraday["Dollar Change"] = intraday["Close"].diff()
#             # intraday['Pct Change'] = intraday['Close'].pct_change() * 100
#             intraday['Kijun'] = (intraday['High'].rolling(window=26).max() + intraday['Low'].rolling(window=26).min()) / 2
#             intraday  =   calculate_kijun_sen(intraday)
#             intraday = detect_kijun_cross(intraday)
#             intraday = calculate_td_sequential(intraday)
#             intraday = calculate_setup_qualifier(intraday)
#             intraday = calculate_td_countdown(intraday)
#             intraday = calculate_td_combo_countdown(intraday)
#             intraday = calculate_td_demarker_ii(intraday)

#             intraday =  calculate_td_pressure(intraday)
#             intraday = add_td_pressure_alert(intraday)
#             intraday = calculate_td_rei(intraday)
#             intraday = add_td_rei_alert(intraday)
#             intraday = add_td_rei_qualifiers(intraday)
#             intraday = refine_td_rei_qualifiers(intraday)

#             intraday = calculate_td_demand_supply_lines(intraday)
#             intraday = detect_close_crosses(intraday)
#             intraday = calculate_bollinger_band_width(intraday)
#             intraday['Tenkan'] = (intraday['High'].rolling(window=9).max() + intraday['Low'].rolling(window=9).min()) / 2
#             intraday['Kijun'] = (intraday['High'].rolling(window=26).max() + intraday['Low'].rolling(window=26).min()) / 2
#             intraday = calculate_theta_tenkan_kijun(intraday)
#             intraday = calculate_theta_tenkan_kijun(intraday)  # Function defined earlier
#             intraday['Cotangent_Tenkan'] = calculate_cotangent(intraday['Theta_Tenkan'])
#             intraday['Cotangent_Kijun'] = calculate_cotangent(intraday['Theta_Kijun'])
#             intraday = calculate_directional_flatness(intraday)
#         # Add Sine Tenkan and Sine Kijun
#             frequency = 1 / 3  # One cycle every 8 periods
#             time_index = np.arange(len(intraday))
#             intraday['Sine_Tenkan'] = calculate_sine(intraday['Tenkan'], frequency, time_index)
#             intraday['Sine_Kijun'] = calculate_sine(intraday['Kijun'], frequency, time_index)
#             # Calculate Theta Tenkan and Theta Kijun
#                 # Calculate Cosecant for Tenkan
#             intraday['Cosecant_Tenkan'] = calculate_cosecant(intraday['Sine_Tenkan'])
#             intraday['Cosecant_Kijun'] = calculate_cosecant(intraday['Sine_Kijun'])

#             intraday['Instability'] = calculate_instability(
#             intraday['Cosecant_Tenkan'], intraday['Cosecant_Kijun']
# )
#             intraday['Cosine_Tenkan'] = calculate_cosine(intraday['Tenkan'], frequency, time_index)
#             intraday['Cosine_Kijun'] = calculate_cosine(intraday['Kijun'], frequency, time_index)



#             # Add Secant Tenkan and Secant Kijun
#             intraday['Secant_Tenkan'] = calculate_secant(intraday['Cosine_Tenkan'])
#             intraday['Secant_Kijun'] = calculate_secant(intraday['Cosine_Kijun'])
#             intraday['Secant_Instability'] = calculate_secant_insta(
#             intraday['Secant_Tenkan'], intraday['Secant_Kijun']
#         )
#             intraday  =   calculate_kijun_sen(intraday)

            intraday['Time'] = intraday['Date'].dt.strftime('%I:%M %p')

                    # 🟢 Mini Table for Kijun Crosses
            crosses = intraday[intraday['Kijun_Alert'].isin(["Cross Up Kijun", "Cross Down Kijun"])]
            if not crosses.empty:
                st.write("*Kijun Crosses Detected*:")
                st.dataframe(crosses[['Time', 'Close', 'Kijun_Alert']].tail(2), use_container_width=True)
  # Mini Table for DMI Alerts History
            dmi_history = intraday[intraday['DMI Alert'] != ""]
            if not dmi_history.empty:
                st.write("*DMI Alerts History*:")
                st.dataframe(dmi_history[['Time', 'Close', 'DMI Alert']].tail(2), use_container_width=True)
            else:
                st.info("ℹ️ No DMI Alerts in the recent history.")
            # 🟢 Mini Table for TD POQ Signals
            poq_signals = intraday[intraday['TD POQ Signal'].isin(["Buy Signal", "Sell Signal"])].tail(2)
            if not poq_signals.empty:
                st.write("*TD POQ Signals Detected*:")
                st.dataframe(poq_signals[['Time', 'Close', 'TD POQ Signal']], use_container_width=True)

            # 🟢 Mini Table for TD REI & TD Pressure Alerts
            rei_pressure_alerts = intraday[intraday['TD REI Alert'].isin(["alert + to -", "alert - to +"]) |
                                        intraday['TD Pressure Alert'].isin(["alert + to -", "alert - to +"])]
            if not rei_pressure_alerts.empty:
                st.write("*TD REI & TD Pressure Alerts Detected*:")
                st.dataframe(rei_pressure_alerts[['Time', 'Close', 'TD REI Alert', 'TD Pressure Alert']].tail(2), use_container_width=True)

            # 🟢 Real-time Alerts for Latest Signals
            latest_alert = intraday['Kijun_Alert'].iloc[-1]
            if latest_alert == "Cross Up Kijun":
                st.success("✅ **ALERT: Cross Up Kijun detected on the latest bar!**")
            elif latest_alert == "Cross Down Kijun":
                st.warning("⚠️ **ALERT: Cross Down Kijun detected on the latest bar!**")

            latest_poq_signal = intraday['TD POQ Signal'].iloc[-1]
            if latest_poq_signal == "Buy Signal":
                st.success("✅ **ALERT: TD POQ Buy Signal detected on the latest bar!**")
            elif latest_poq_signal == "Sell Signal":
                st.warning("⚠️ **ALERT: TD POQ Sell Signal detected on the latest bar!**")

                        # 🟢 Mini Table for Lines Alert
            lines_alerts = intraday[intraday['Lines Alert'].isin(["Cross Up TSL", "Cross Down TSL", "Cross Up TDL", "Cross Down TDL"])]
            if not lines_alerts.empty:
                st.write("*Lines Alert Signals Detected*:")
                st.dataframe(lines_alerts[['Time', 'Close', 'Lines Alert']].tail(2), use_container_width=True)





            latest_rei_alert = intraday['TD REI Alert'].iloc[-1]
            latest_pressure_alert = intraday['TD Pressure Alert'].iloc[-1]

            if latest_rei_alert == "alert - to +" and latest_pressure_alert == "alert - to +":
                st.success("✅ **BULL ALERT: TD REI & TD Pressure both flipped from - to +!** 🚀")
            elif latest_rei_alert == "alert + to -" and latest_pressure_alert == "alert + to -":
                st.warning("⚠️ **BEAR ALERT: TD REI & TD Pressure both flipped from + to -!** 📉")



               # ----- Define your styling function (if not already defined) -----
            def style_rows(data):
                def color_row(row):
                    # Use the 'Dollar Change' column to determine row color.
                    # (Adjust the column name if needed.)
                    value = row['$']  # In your code, you seem to store the Dollar Change in column '$'

                    if value < 0:
                        return ['background-color: #DC143C'] * len(row)
                    elif 0 <= value <= 0.10:
                        return ['background-color: #778899'] * len(row)
                    elif 0.11 <= value <= 0.30:
                        return ['background-color:#3CB371'] * len(row)
                    elif 0.31 <= value <= 0.50:
                        return ['background-color: #32CD32'] * len(row)
                    elif value > 0.50:
                        return ['background-color: #48D1CC'] * len(row)
                    else:
                        return [''] * len(row)  # No coloring for other cases

                return data.style.apply(color_row, axis=1)




            # 🟢 🟢 🟢 MAIN DATA TABLE (LAST 13 ROWS) 🟢 🟢 🟢

            columns_to_hide = ["Date","OC","Volume","Log_Change_CC","Kijun-sen","Tenkan","Rolling_StdDev_CC", "CC", "Range", "Kijun",'Adj Close',"High", "Low","Open", 'Tenkan', 'Kijun', 'Kijun-sen', 'Sine_Tenkan', 'Sine_Kijun',
    'Theta_Tenkan', 'Theta_Kijun', 'Cotangent_Tenkan', 'Cotangent_Kijun',
    'Cosecant_Tenkan', 'Cosecant_Kijun', 'Cosine_Tenkan', 'Cosine_Kijun',
    'Secant_Tenkan', 'Secant_Kijun','TD_REI_prev','Adj Close','Log_Change_CC','Rolling_StdDev_CC','Sell Combo Countdown','Buy Combo Countdown','Buy Countdown','Sell Countdown','TD Supply Line', "TD Demand Line","Upper_Band","Lower_Band","Middle_Band","Std_Dev","Kijun_ichimoku","Senkou_Span_A","Senkou_Span_B","Chikou_Span"]


            # st.write("🔍 Checking full dataset before filtering:")
            # st.dataframe(intraday)

            # Create a display DataFrame (adjust column names as needed)
            display_df = intraday.drop(columns=columns_to_hide, errors="ignore")

            # Get the last 22 rows of the display DataFrame
            display_df_tail = display_df

            styled_display_df_tail = style_rows(display_df_tail)

            # ----- Display the styled DataFrame in Streamlit -----
            st.write("Intraday Table Analysis", styled_display_df_tail)
 # Check for Kijun Alerts on the latest row
            st.write("Total rows in dataset:", len(intraday))

            latest_alert = intraday['Kijun_Alert'].iloc[-1]

            if latest_alert == "Cross Up Kijun":
                st.success("Alert: Cross Up Kijun detected on the latest bar!")
                st.markdown(
                    """
                    <audio autoplay>
                        <source src="https://www.soundjay.com/buttons/sounds/button-16.mp3" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True
                )

            elif latest_alert == "Cross Down Kijun":
                st.warning("Alert: Cross Down Kijun detected on the latest bar!")
                st.markdown(
                    """
                    <audio autoplay>
                        <source src="https://www.soundjay.com/buttons/sounds/button-1.mp3" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True
                )
        # After all your calculations have been done:
        latest_row = intraday.iloc[-1]  # Get latest data point


        # Function to handle TD POQ signals
            # Check for TD POQ Signal on the latest row
        latest_poq_signal = intraday['TD POQ Signal'].iloc[-1]

        if latest_poq_signal == "Buy Signal":
            st.success("✅ ALERT: TD POQ Buy Signal detected on the latest bar!")
            st.markdown(
                """
                <audio autoplay>
                    <source src="https://www.soundjay.com/buttons/sounds/button-16.mp3" type="audio/mp3">
                </audio>
                """,
                unsafe_allow_html=True
            )

        elif latest_poq_signal == "Sell Signal":
            st.warning("⚠️ ALERT: TD POQ Sell Signal detected on the latest bar!")
            st.markdown(
                """
                <audio autoplay>
                    <source src="https://www.soundjay.com/buttons/sounds/button-2.mp3" type="audio/mp3">
                </audio>
                """,
                unsafe_allow_html=True
            )

            # 🟢 Check for TD REI & TD Pressure Alert on the latest row
            latest_rei_alert = intraday['TD REI Alert'].iloc[-1]
            latest_pressure_alert = intraday['TD Pressure Alert'].iloc[-1]

            if latest_rei_alert == "alert - to +" and latest_pressure_alert == "alert - to +":
                st.success("✅ **BULL ALERT: TD REI & TD Pressure both flipped from - to +!** 🚀")
                st.markdown(
                    """
                    <audio autoplay>
                        <source src="https://www.soundjay.com/buttons/sounds/button-16.mp3" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True
                )

            elif latest_rei_alert == "alert + to -" and latest_pressure_alert == "alert + to -":
                st.warning("⚠️ **BEAR ALERT: TD REI & TD Pressure both flipped from + to -!** 📉")
                st.markdown(
                    """
                    <audio autoplay>
                        <source src="https://www.soundjay.com/buttons/sounds/button-3.mp3" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True


                )




                        # Extract the latest DMI Alert value
                latest_dmi_alert = intraday['DMI Alert'].iloc[-1]

                # Check if an alert exists and then trigger it
                if latest_dmi_alert == "Buy Signal":
                    st.success("✅ **ALERT: DMI Buy Signal detected on the latest bar!** 🚀")
                    st.markdown(
                        """
                        <audio autoplay>
                            <source src="https://www.soundjay.com/buttons/sounds/button-16.mp3" type="audio/mp3">
                        </audio>
                        """,
                        unsafe_allow_html=True
                    )
                elif latest_dmi_alert == "Sell Signal":
                    st.warning("⚠️ **ALERT: DMI Sell Signal detected on the latest bar!** 📉")
                    st.markdown(
                        """
                        <audio autoplay>
                            <source src="https://www.soundjay.com/buttons/sounds/button-4.mp3" type="audio/mp3">
                        </audio>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.info("ℹ️ No DMI Alert detected on the latest bar.")


    except Exception as e:
        st.error(f"An error occurred while fetching multiple tickers: {e}")
