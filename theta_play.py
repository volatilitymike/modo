import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from math import log, sqrt
from scipy.stats import norm

st.set_page_config(page_title="Options Wealth - Theta Plays", layout="wide")

st.title("⏳ Theta Plays: Option Chain Summary with Greeks")


# --- Ticker Input ---
ticker = st.text_input("Enter Stock Symbol:", value="AAPL")



# --- Functions ---
def calculate_greeks(option_type, S, K, T, r, sigma):
    try:
        if T <= 0 or sigma <= 0:
            return None, None, None, None, None
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type.lower() == "call":
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))

        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T)

        return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta / 365, 4)
    except:
        return None, None, None, None

# --- Load Chain ---
if ticker:
    stock = yf.Ticker(ticker)
    expirations = stock.options

    if expirations:
        selected_exp = st.selectbox("Select Expiration Date", expirations)

        try:
            chain = stock.option_chain(selected_exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            current_price = stock.history(period="1d")['Close'].iloc[-1]
            st.markdown(f"### 📌 Spot Price: ${current_price:.2f}")

            r = 0.05  # Risk-free rate
            T = max((datetime.strptime(selected_exp, "%Y-%m-%d") - datetime.today()).days, 1) / 365

            # Calculate Greeks for Calls
            calls[["Delta", "Gamma", "Vega", "Theta"]] = calls.apply(
                lambda row: pd.Series(
                    calculate_greeks("call", current_price, row['strike'], T, r, row['impliedVolatility'])
                ), axis=1
            )

            # Calculate Greeks for Puts
            puts[["Delta", "Gamma", "Vega", "Theta"]] = puts.apply(
                lambda row: pd.Series(
                    calculate_greeks("put", current_price, row['strike'], T, r, row['impliedVolatility'])
                ), axis=1
            )
            df = pd.merge(
                calls[['strike', 'impliedVolatility', 'Delta', 'Gamma', 'Vega', 'Theta']].rename(columns={
                    'impliedVolatility': 'Call IV',
                    'Delta': 'Call Delta',
                    'Gamma': 'Call Gamma',
                    'Vega': 'Call Vega',
                    'Theta': 'Call Theta'
                }),
                puts[['strike', 'impliedVolatility', 'Delta', 'Gamma', 'Vega', 'Theta']].rename(columns={
                    'impliedVolatility': 'Put IV',
                    'Delta': 'Put Delta',
                    'Gamma': 'Put Gamma',
                    'Vega': 'Put Vega',
                    'Theta': 'Put Theta'
                }),
                on='strike'
            )


            st.dataframe(df.sort_values(by='strike'), use_container_width=True)

                        # Strike selector
                    # Strike selector
            available_strikes = df['strike'].sort_values().tolist()
            selected_strike = st.selectbox("📍 Select a Strike to View Details", available_strikes)

            # Extract and show details
            selected_row = df[df['strike'] == selected_strike].iloc[0]

            st.markdown(f"""
            ### 📋 Selected Strike: **{selected_strike}**
            - 🟢 Call IV: **{selected_row['Call IV']:.2%}**
            - 🔴 Put IV: **{selected_row['Put IV']:.2%}**
            - 📐 Call Delta: **{selected_row['Call Delta']}**
            - 📐 Put Delta: **{selected_row['Put Delta']}**
            - ♊️ Call Gamma: **{selected_row['Call Gamma']}**
            - ♊️ Put Gamma: **{selected_row['Put Gamma']}**
            - ⏳ Call Theta: **{selected_row['Call Theta']}**
            - ⏳ Put Theta: **{selected_row['Put Theta']}**
            - 📈 Call Vega: **{selected_row['Call Vega']}**
            - 📈 Put Vega: **{selected_row['Put Vega']}**
            """)



        except Exception as e:
            st.error(f"Error loading option chain: {e}")





        st.markdown("---")
        st.subheader("⏳ Theta Decay Survival Estimate")

        # User inputs
        selected_strike = st.selectbox("Select a Strike for Survival Estimate:", df['strike'].unique())
        days_to_test = st.slider("Survival Horizon (Days):", min_value=1, max_value=30, value=5)

        # Extract the row for that strike
        row = df[df['strike'] == selected_strike].iloc[0]

        # Call calculations
        call_price = (calls[calls['strike'] == selected_strike]['lastPrice'].values[0]
                    if not calls[calls['strike'] == selected_strike].empty else None)
        call_theta = row["Call Theta"]
        call_decay = round(abs(call_theta) * days_to_test, 2)
        call_survival = round(call_price - call_decay, 2) if call_price is not None else None


        # Put calculations
        put_price = (puts[puts['strike'] == selected_strike]['lastPrice'].values[0]
                    if not puts[puts['strike'] == selected_strike].empty else None)
        put_theta = row["Put Theta"]
        put_decay = round(abs(put_theta) * days_to_test, 2)
        put_survival = round(put_price - put_decay, 2) if put_price is not None else None

        # Display result
        st.markdown(f"### 📉 Theta Decay for Strike **{selected_strike}** over **{days_to_test} day(s)**")
        st.markdown(f"""
        #### 🧮 Theta Decay Summary for Strike **{selected_strike}**
        - **Call Theta**: {call_theta} per day → {call_theta} × {days_to_test} days = **${call_decay}**
        - **Call Premium Now**: ${call_price} → **Expected Value After {days_to_test} Days** = **${call_survival}**

        - **Put Theta**: {put_theta} per day → {put_theta} × {days_to_test} days = **${put_decay}**
        - **Put Premium Now**: ${put_price} → **Expected Value After {days_to_test} Days** = **${put_survival}**
        """)


    else:
        st.warning("No expiration dates found for this ticker.")
