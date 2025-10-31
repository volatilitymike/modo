import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

# Set page config first thing
st.set_page_config(page_title="Options Wealth - Option Pricing", layout="wide")






st.title("💰 Option Pricing Module")

# --- Ticker Input ---
ticker = st.text_input("Enter Stock Symbol:", value="AAPL")

# --- STEP 1: Define All Needed Functions (Greeks + Delta-Neutral) ---

def estimate_otm_option_price(atm_price, spot_price, otm_strike, delta_pct=0.5):
    """
    Estimate OTM option price using delta-weighted adjustment from ATM price.
    """
    intrinsic_diff = abs(otm_strike - spot_price)
    adjustment = delta_pct * intrinsic_diff

    if otm_strike > spot_price:  # OTM Call
        return round(atm_price - adjustment, 2)
    else:  # OTM Put
        return round(atm_price + adjustment, 2)


def calculate_annual_volatility(df, window=20):
    """
    Calculate rolling annualized volatility (%), based on daily return std dev.
    df: DataFrame with 'Close' column
    window: Rolling window size (default 20 days)
    """
    df = df.copy()
    df["Daily Return"] = df["Close"].pct_change()
    df["Rolling StdDev (Daily Return)"] = df["Daily Return"].rolling(window=window).std()
    df["Annual Volatility (%)"] = df["Rolling StdDev (Daily Return)"] * np.sqrt(252) * 100
    df["Annual Volatility (%)"] = df["Annual Volatility (%)"].round(2)
    return df

def calculate_greeks(option_type, S, K, T, r, sigma):
    try:
        if T <= 0 or sigma <= 0:
            return None, None, None, None, None
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type.lower() == "call":
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
            charm = -norm.pdf(d1) * (2 * r * T - sigma**2) / (2 * sigma * sqrt(T)) - r * norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
            charm = -norm.pdf(d1) * (2 * r * T - sigma**2) / (2 * sigma * sqrt(T)) + r * norm.cdf(-d1)

        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T)

        return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta / 365, 4), round(charm / 365, 4)

    except:
        return None, None, None, None, None

# 1) Define a helper function to calculate *historical* annualized volatility
def get_annual_volatility(ticker):
    # Download 1 year of daily price data
    df = yf.download(ticker, period='1y', interval='1d')
    df["returns"] = df["Close"].pct_change()
    daily_std = df["returns"].std()
    # Annualize the daily standard deviation (~252 trading days/yr)
    annual_vol = daily_std * np.sqrt(252)
    return annual_vol

def estimate_atm_premium(spot, iv, T):
    """
    Estimate ATM option premium as a % of spot using 0.4 * IV * sqrt(T)
    """
    return round(spot * 0.4 * iv * np.sqrt(T), 2)


def get_iv_rank_and_percentile(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y", interval="1d")
    current_price = hist["Close"].iloc[-1]

    # Pick 1y of option chains to approximate daily IV
    iv_list = []

    for date in pd.date_range(end=datetime.today(), periods=252, freq='B'):
        try:
            options = stock.option_chain(date.strftime('%Y-%m-%d'))
            calls = options.calls
            atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
            iv = atm_call["impliedVolatility"].values[0]
            iv_list.append(iv)
        except:
            continue  # skip dates that fail

    if len(iv_list) >= 30:
        iv_series = pd.Series(iv_list)
        iv_rank = round((iv_series.iloc[-1] - iv_series.min()) / (iv_series.max() - iv_series.min()) * 100, 2)
        iv_percentile = round((iv_series < iv_series.iloc[-1]).sum() / len(iv_series) * 100, 2)
        return iv_rank, iv_percentile
    else:
        return None, None



def find_delta_neutral_matches(option_type, strike, merged_df, tolerance=0.05):
    """
    Given an option_type ('call' or 'put') and a strike from 'merged_df',
    find combos on the opposite side that offset the selected option's delta
    within 'tolerance'.
    """
    opposite = "Call" if option_type == "put" else "Put"  # Opposite side
    selected_delta_col = f"{option_type.capitalize()} Delta"
    hedge_delta_col = f"{opposite} Delta"
    price_col = f"{opposite} Price"

    try:
        # This is the chosen option row (call or put) by its strike
        selected_row = merged_df[merged_df["strike"] == strike].iloc[0]
        selected_delta = selected_row[selected_delta_col]
        st.write(f"📐 Selected {option_type.capitalize()} Delta: {selected_delta}")
        selected_premium = selected_row[f"{option_type.capitalize()} Price"]
        st.write(f"💵 Selected {option_type.capitalize()} Premium: ${selected_premium}")



                # Subset with the hedge side
        iv_col = f"{opposite} IV"
        hedge_df = merged_df[["strike", hedge_delta_col, price_col, iv_col]].copy()

        hedge_df.rename(columns={
            hedge_delta_col: "Hedge Delta",
            price_col: "Hedge Price",
            iv_col: "IV"

        }, inplace=True)

        combo_list = []
        # We'll try 1x to 4x to see if that gets us close to neutral
        for i in range(1, 5):
            hedge_df["Legs"] = i
            hedge_df["Total Delta"] = hedge_df["Hedge Delta"] * i
            hedge_df["Net Position Delta"] = selected_delta + hedge_df["Total Delta"]
            hedge_df["Delta Diff"] = hedge_df["Net Position Delta"].abs()
            hedge_df["Total Cost"] = hedge_df["Hedge Price"] * i
            hedge_df["Premium"] = hedge_df["Hedge Price"]  # single leg price
            hedge_df["Gamma"] = merged[f"{opposite} Gamma"]  # get gamma per leg
            hedge_df["Total Gamma"] = hedge_df["Gamma"] * hedge_df["Legs"]
            hedge_df["Gamma Alert"] = hedge_df["Gamma"].apply(lambda g: "⚠️ High Gamma" if g > 0.07 else "")
            hedge_df["Full Combo Cost"] = hedge_df["Total Cost"] + selected_row[f"{option_type.capitalize()} Price"]
            main_delta = selected_row[f"{option_type.capitalize()} Delta"]
            main_gamma = selected_row[f"{option_type.capitalize()} Gamma"]
            main_price = selected_row[f"{option_type.capitalize()} Price"]
            # Assuming daily percent move (S%) → e.g., 0.02 for a 2% move
            expected_move_pct = 0.02
            hedge_df["Delta-Hedged P&L"] = ((expected_move_pct ** 2) * hedge_df["Total Gamma"]) / 2


            a = 0.5 * main_gamma
            b = abs(main_delta)

            c = -hedge_df["Full Combo Cost"]  # full strangle cost

            discriminant = (b ** 2 + 4 * a * (-c)).clip(lower=0)
            move_to_cover = (-b + np.sqrt(discriminant)) / (2 * a)

            if option_type == "put":
                move_to_cover *= -1

            hedge_df["Move to Cover"] = move_to_cover









            a_h = 0.5 * hedge_df["Total Gamma"]
            b_h = abs(hedge_df["Total Delta"])
            c_h = -hedge_df["Full Combo Cost"]  # same full cost!

            discriminant_h = (b_h**2 + 4 * a_h * (-c_h)).clip(lower=0)
            hedge_move_to_cover = (-b_h + np.sqrt(discriminant_h)) / (2 * a_h)

            if option_type == "call":
                hedge_move_to_cover *= -1

            hedge_df["Hedge Move to Cover"] = hedge_move_to_cover




            hedge_move_to_cover = (-b_h + np.sqrt(discriminant_h)) / (2 * a_h)

            # Direction depends on hedge type (opposite of main leg)
            if option_type == "call":
                hedge_move_to_cover *= -1  # hedge is a put → needs stock drop
            else:
                hedge_move_to_cover *= 1   # hedge is a call → needs stock rise

            hedge_df["Hedge Move to Cover"] = hedge_move_to_cover


            # Inside find_delta_neutral_matches()
            hedge_df["Variance"] = hedge_df["IV"] ** 2
            hedge_df["Variance Edge"] = hedge_df["Gamma"] * hedge_df["Variance"]
            hedge_df["Total Variance Edge"] = hedge_df["Variance Edge"] * hedge_df["Legs"]




            close_to_neutral = hedge_df[hedge_df["Delta Diff"] <= tolerance].copy()
            combo_list.append(close_to_neutral)

        if len(combo_list) > 0:
            all_matches = pd.concat(combo_list).sort_values(by="Delta Diff")
            return all_matches
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"⚠️ Failed to calculate delta-neutral match: {e}")
        return pd.DataFrame()





# Initialize session_state keys if they don't exist
if "merged" not in st.session_state:
    st.session_state.merged = None
if "stock_price" not in st.session_state:
    st.session_state.stock_price = None
if "expirations" not in st.session_state:
    st.session_state.expirations = []
if "selected_exp" not in st.session_state:
    st.session_state.selected_exp = None


# --- STEP 2: Let User Load the Data ---

if st.button("Load Options Data"):
    # If user provided a ticker, fetch its expiration dates
    if ticker:
        stock = yf.Ticker(ticker)
        exps = stock.options
        if exps:
            # Store expiration dates in session state
            st.session_state.expirations = exps
            # By default, pick the first expiration
            st.session_state.selected_exp = exps[0]
        else:
            st.error("No expiration dates found for this ticker.")
    else:
        st.warning("Please enter a ticker symbol.")



# If we have expiration dates, let user pick from them
if st.session_state.expirations:
    st.session_state.selected_exp = st.selectbox(
        "Select Expiration Date",
        st.session_state.expirations,
        index=0
    )

# Once we have an expiration selected, we can load the chain
if st.session_state.selected_exp:
    # Try to fetch & process the chain
    try:
        stock = yf.Ticker(ticker)
        option_chain = stock.option_chain(st.session_state.selected_exp)
        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()

        # Get current stock price
        stock_price = stock.history(period="1d")["Close"].iloc[-1]
        st.session_state.stock_price = stock_price

        # Calculate Put/Call Ratio
        total_call_volume = calls["volume"].sum()
        total_put_volume = puts["volume"].sum()

        # Option classification
        calls["Type"] = "Call"
        puts["Type"] = "Put"
        calls["Classification"] = calls["strike"].apply(
            lambda x: "ATM" if abs(x - stock_price) < 0.5 else ("ITM" if x < stock_price else "OTM")
        )
        puts["Classification"] = puts["strike"].apply(
            lambda x: "ATM" if abs(x - stock_price) < 0.5 else ("ITM" if x > stock_price else "OTM")
        )

        # Merge calls and puts
        merged = pd.merge(
            calls[["strike", "lastPrice", "impliedVolatility", "volume"]].rename(columns={
                "lastPrice": "Call Price",
                "impliedVolatility": "Call IV",
                "volume": "Call Volume"
            }),
            puts[["strike", "lastPrice", "impliedVolatility", "volume"]].rename(columns={
                "lastPrice": "Put Price",
                "impliedVolatility": "Put IV",
                "volume": "Put Volume"
            }),
            on="strike"
        )
        merged["Straddle Cost (1SD)"] = merged["Call Price"] + merged["Put Price"]
        r = 0.05  # 5% risk-free rate

        # Convert expiration string (e.g., '2025-04-19') to datetime
        exp_date = datetime.strptime(st.session_state.selected_exp, "%Y-%m-%d")
        today = datetime.today()
        days_to_exp = (exp_date - today).days

        # If it's a valid future date, compute T in years
        T = max(days_to_exp, 1) / 365
        # Add Expected Move based on Call IV
        merged["Call Expected Move ($)"] = round(stock_price * merged["Call IV"] * np.sqrt(T * 252), 2)
        merged["Put Expected Move ($)"] = round(stock_price * merged["Put IV"] * np.sqrt(T * 252), 2)



        iv_rank, iv_percentile = get_iv_rank_and_percentile(ticker)
        merged["IV Rank (%)"] = iv_rank
        merged["IV Percentile (%)"] = iv_percentile


        # Calculate call/put greeks
        call_greeks = merged.apply(
            lambda row: calculate_greeks("call", stock_price, row["strike"], T, r, row["Call IV"]),
            axis=1
        )
        put_greeks = merged.apply(
            lambda row: calculate_greeks("put", stock_price, row["strike"], T, r, row["Put IV"]),
            axis=1
        )
        annual_vol = get_annual_volatility(ticker)
        merged["Annual Volatility (%)"] = round(annual_vol * 100, 2)
        # Add AV - IV difference for both Call and Put
        merged["Call AV - IV"] = round((annual_vol * 100) - (merged["Call IV"] * 100), 2)
        merged["Put AV - IV"] = round((annual_vol * 100) - (merged["Put IV"] * 100), 2)
        # Add IV / AV ratio (using decimal values, then rounded to 2 decimals)
        merged["Call IV / AV"] = round(merged["Call IV"] / annual_vol, 2)
        merged["Put IV / AV"] = round(merged["Put IV"] / annual_vol, 2)

        merged["Call Delta"], merged["Call Gamma"], merged["Call Vega"], merged["Call Theta"], merged["Call Charm"] = zip(*call_greeks)

        merged["Put Delta"], merged["Put Gamma"], merged["Put Vega"], merged["Put Theta"], merged["Put Charm"] = zip(*put_greeks)


 


        # Store 'merged' in session_state so it persists across reruns
        st.session_state.merged = merged

        # Show current price
        st.success(f"Current Price: ${stock_price:.2f}")

        # Show PCR
        if total_call_volume > 0:
            pcr = round(total_put_volume / total_call_volume, 2)
            st.info(f"Put-Call Ratio (PCR): {pcr}")
        else:
            st.warning("Not enough call volume to calculate PCR.")





    except Exception as e:
        st.error(f"Error loading options data: {e}")


# --- STEP 3: If we have 'merged' in session_state, show the Option Chain & Vol Plot & Delta-Neutral Helper ---

if st.session_state.merged is not None:
    merged = st.session_state.merged  # local reference
    stock_price = st.session_state.stock_price













    # --- Show Data Table with Greeks ---
    st.subheader("📋 Option Chain Summary with Greeks")
    st.dataframe(merged.sort_values(by="Straddle Cost (1SD)", ascending=False), use_container_width=True)




    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged["strike"],
        y=merged["Call IV"],
        mode='lines+markers',
        name='Call IV',
        customdata=np.stack((merged["Call Volume"], merged["Call IV"] * 100), axis=-1),

        hovertemplate=
        "IV: %{customdata[1]:.2f}%<br>" +
        "Volume: %{customdata[0]}<extra></extra>" +
        "Open Interest: %{customdata[1]}<extra></extra>"

    ))

    fig.add_trace(go.Scatter(
        x=merged["strike"],
        y=merged["Put IV"],
        mode='lines+markers',
        name='Put IV',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title="Volatility Smile",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


    # --- Delta-Neutral Builder ---
    st.subheader("🎯 Delta-Neutral Strategy Helper")

    try:
        available_strikes = merged["strike"].sort_values().tolist()
        option_types = ["call", "put"]

        col1, col2 = st.columns(2)
        with col1:
            selected_type = st.selectbox("Option Type", option_types, key="opt_type")
        with col2:
            selected_strike = st.selectbox("Select Strike", available_strikes, key="opt_strike")

        # Run delta-neutral calculation
        delta_matches = find_delta_neutral_matches(
            option_type=selected_type,
            strike=selected_strike,
            merged_df=merged
        )


        selected_row = merged[merged["strike"] == selected_strike].iloc[0]
        selected_delta = selected_row[f"{selected_type.capitalize()} Delta"]
        main_gamma = selected_row[f"{selected_type.capitalize()} Gamma"]



        selected_row = merged[merged["strike"] == selected_strike].iloc[0]
        selected_delta = selected_row[f"{selected_type.capitalize()} Delta"]
        main_gamma = selected_row[f"{selected_type.capitalize()} Gamma"]

        # ⬇️ Add this block here
        iv_used = selected_row[f"{selected_type.capitalize()} IV"]
        def estimate_atm_premium(spot, iv, T):
            return round(spot * 0.4 * iv * np.sqrt(T), 2)
        atm_est_premium = estimate_atm_premium(stock_price, iv_used, T)
        st.write(f"📐 Estimated ATM Premium: **${atm_est_premium}** using 0.4 × IV × √T")


        iv_used = selected_row[f"{selected_type.capitalize()} IV"]
        st.write(f"📊 Implied Volatility at Strike {selected_strike}: **{iv_used:.2%}**")

        daily_move = stock_price * iv_used * np.sqrt(1 / 252)
        daily_move = round(daily_move, 2)

        st.write(f"📉 Expected Daily Move: **±${daily_move}** based on IV = {iv_used:.2%}")

        vol_carry = iv_used - annual_vol
        st.write(f"📈 Volatility Carry: **{vol_carry:.2%}** (IV - Realized Volatility)")

        iv_var = iv_used ** 2
        real_var = annual_vol ** 2
        var_carry = iv_var - real_var
        st.write(f"📉 Variance Carry: **{var_carry:.4f}** (IV² - Realized²)")


    # Calculate main_leg_row here (correct indentation)
        main_leg_row = pd.DataFrame([{
            "strike": selected_strike,
            "Legs": 1,
            "Hedge Delta": selected_delta,
            "Total Delta": selected_delta,
            "Gamma": selected_row[f"{selected_type.capitalize()} Gamma"],
            "Total Gamma": selected_row[f"{selected_type.capitalize()} Gamma"],
            "Gamma Alert": "🎯 Main Leg",
            "Premium": selected_row[f"{selected_type.capitalize()} Price"],
            "Net Position Delta": selected_delta,
            "Total Cost": selected_row[f"{selected_type.capitalize()} Price"],
            "Full Combo Cost": selected_row[f"{selected_type.capitalize()} Price"],
            "Move to Cover": np.nan,
            "Hedge Move to Cover": np.nan
        }])

        if not delta_matches.empty:
            st.subheader("📊 Matching Hedges for Delta Neutral")
            final_table = pd.concat([main_leg_row, delta_matches], ignore_index=True)
            # Calculate Gamma Diff
            final_table["Gamma Diff"] = (final_table["Total Gamma"] - main_gamma).abs()

            # Tier System
            def classify_tier(row):
                if abs(row["Net Position Delta"]) <= 0.05 and row["Gamma Diff"] <= 0.01:
                    return "✅ Optimal"
                elif abs(row["Net Position Delta"]) <= 0.10 and row["Gamma Diff"] <= 0.02:
                    return "🟢 Good"
                else:
                    return "⚠️ Imbalanced"

            final_table["Tier Match"] = final_table.apply(classify_tier, axis=1)

            final_table["Optimal Match"] = final_table.apply(
                lambda row: "✅ Optimal" if abs(row["Net Position Delta"]) <= 0.05 and abs(row["Total Gamma"] - main_gamma) <= 0.01 else "",
                axis=1
            )

            st.dataframe(final_table[[
                "strike", "Legs", "Hedge Delta","Total Delta",  "Gamma", "Total Gamma","Gamma Alert","Variance", "Variance Edge", "Total Variance Edge","Premium",
                "Net Position Delta", "Total Cost","Full Combo Cost","Move to Cover","Hedge Move to Cover","Optimal Match","Tier Match",
            ]], use_container_width=True)
        else:
            st.info("No delta-neutral matches found within ±0.05 range.")


    except Exception as e:
        st.error(f"Error building delta-neutral matches: {e}")




# -----------------------------------
# Start Fere & Medium Zone Section
# -----------------------------------

st.subheader("📊 Almost Neutral Zones (Fere & Medium)")

try:
    opposite = "Call" if selected_type == "put" else "Put"
    selected_row = merged[merged["strike"] == selected_strike].iloc[0]

    selected_delta = selected_row[f"{selected_type.capitalize()} Delta"]

    main_gamma = selected_row[f"{selected_type.capitalize()} Gamma"]

    call_delta = selected_row["Call Delta"]  # Might be 0.32, for example
    put_delta  = selected_row["Put Delta"]   # Might be -0.18, for example



    # Create a hedge DataFrame
    hedge_df = merged[["strike", f"{opposite} Delta", f"{opposite} Price", f"{opposite} Gamma"]].copy()
        # Only use hedge legs with delta magnitude ≥ 0.10
    hedge_df = hedge_df[hedge_df[f"{opposite} Delta"].abs() >= 0.12]

    hedge_df.rename(columns={
        f"{opposite} Delta": "Hedge Delta",
        f"{opposite} Price": "Hedge Price",
        f"{opposite} Gamma": "Gamma"
    }, inplace=True)

    combo_zone_list = []

    # Try 1-4 legs on the hedge side
    for i in range(1, 5):
        temp = hedge_df.copy()
        temp["Legs"] = i
        temp["Total Delta"] = temp["Hedge Delta"] * i
        temp["Net Position Delta"] = selected_delta + temp["Total Delta"]
        temp = temp[temp["Net Position Delta"] >= 0.05]
        temp["Abs Delta"] = temp["Net Position Delta"].abs()

        # Label the delta zone
        def label_delta_zone(x):
            if x <= 0.05:
                return "🎯 Neutral"
            elif x <= 0.10:
                return "🟦 Fere (Low)"
            elif x <= 0.25:
                return "🟩 Fere (High)"
            elif x <= 0.50:
                return "🟨 Medium Neutral"
            else:
                return None

        temp["Delta Zone"] = temp["Abs Delta"].apply(label_delta_zone)
        temp = temp[temp["Delta Zone"].notnull()]

        temp["Total Cost"] = temp["Hedge Price"] * i
        temp["Full Combo Cost"] = temp["Total Cost"] + selected_row[f"{selected_type.capitalize()} Price"]

        # Calculate Total Theta (main leg + hedge leg)
        temp["Theta"] = merged[f"{opposite} Theta"]
        temp["Total Theta"] = temp["Theta"] * temp["Legs"] + selected_row[f"{selected_type.capitalize()} Theta"]

        # Adjusted cost with 1-day decay
        adjusted_cost = temp["Full Combo Cost"].iloc[0] + abs(temp["Total Theta"].iloc[0])





                # Drop zero-gamma rows
        temp = temp[temp["Gamma"] > 0.01]

        # Drop hedges that cost way too much
        max_extra_cost = 5
        temp = temp[temp["Full Combo Cost"] <= selected_row[f"{selected_type.capitalize()} Price"] + max_extra_cost]


        # Main Call Delta/Gamma (use actual column from main leg)
        main_delta = selected_row[f"{selected_type.capitalize()} Delta"]
        main_gamma = selected_row[f"{selected_type.capitalize()} Gamma"]
        main_price = selected_row[f"{selected_type.capitalize()} Price"]

        # Solve quadratic: (0.5 * gamma) * x^2 + delta * x - full_cost = 0
        a = 0.5 * main_gamma
        b = abs(main_delta)
        c = -temp["Full Combo Cost"]
        discriminant = b**2 - 4*a*c
        discriminant = discriminant.clip(lower=0)

        move_to_cover = (-b + np.sqrt(discriminant)) / (2*a)

        # Flip direction if the main leg is a put
        if selected_type == "put":
            move_to_cover *= -1

        temp["Move to Cover"] = move_to_cover



        temp["Total Delta"] = temp["Hedge Delta"] *  temp["Legs"]

        temp["Total Gamma"] = temp["Gamma"] * temp["Legs"]

        # Hedge-based move to cover (using hedge's delta and gamma only)
        a_h = 0.5 * temp["Total Gamma"]
        b_h = abs(temp["Total Delta"])
        c_h = -temp["Total Cost"]

        discriminant_h = b_h**2 - 4 * a_h * c_h
        discriminant_h = discriminant_h.clip(lower=0)

        hedge_move_to_cover = (-b_h + np.sqrt(discriminant_h)) / (2 * a_h)

        # Adjust direction: if hedge is a put (main is call), movement is down
        if selected_type == "call":
            hedge_move_to_cover *= -1  # put hedge → stock must drop
        else:
            hedge_move_to_cover *= 1   # call hedge → stock must rise

        temp["Hedge Move to Cover"] = hedge_move_to_cover



        temp["Premium"] = temp["Hedge Price"]
        temp["Total Gamma"] = temp["Gamma"] * temp["Legs"]
        temp["Gamma Alert"] = temp["Gamma"].apply(lambda g: "⚠️ High Gamma" if g > 0.07 else "")

        def classify_fere_tier(row):
            if abs(row["Net Position Delta"]) <= 0.10 and row["Gamma"] >= 0.03 and row["Total Gamma"] >= 0.06:
                return "✅ Optimal"
            elif abs(row["Net Position Delta"]) <= 0.20 and row["Gamma"] >= 0.03 and row["Total Gamma"] >= 0.05:
                return "🟢 Good"
            else:
                return "⚠️ Imbalanced"


        temp["Tier Match"] = temp.apply(classify_fere_tier, axis=1)



        combo_zone_list.append(temp)





    if combo_zone_list:
        all_zone_matches = pd.concat(combo_zone_list).sort_values(by="Abs Delta")



        st.dataframe(all_zone_matches[[
            "strike", "Legs", "Hedge Delta", "Total Delta", "Net Position Delta",
            "Delta Zone", "Gamma", "Total Gamma", "Gamma Alert",
            "Premium", "Total Cost", "Full Combo Cost","Move to Cover","Hedge Move to Cover","Tier Match",
        ]], use_container_width=True)
    else:
        st.info("No matches found for Fere or Medium Neutral zones.")


except Exception as e:
    st.error(f"Error calculating Theta-Aware Combos: {e}")











