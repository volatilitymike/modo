import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="📈 Term Structure Viewer", layout="wide")

st.title("⏳ Term Structure of Implied Volatility a.k.a SIGMA")

# --- Ticker Input ---
ticker = st.text_input("Enter Ticker (e.g., AAPL, NVDA):", value="AAPL").upper()

@st.cache_data(ttl=3600)
def get_term_structure(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options

    today = datetime.today()
    data = []

    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        days_to_exp = (exp_date - today).days
        if days_to_exp <= 90:  # Limit to 3 months
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls
                current_price = stock.history(period="1d")['Close'].iloc[-1]
                atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
                iv = atm_call['impliedVolatility'].values[0]
                data.append({
                    "Expiration": exp,
                    "Days to Expiry": days_to_exp,
                    "Implied Volatility (%)": round(iv * 100, 2)
                })
            except:
                continue

    df = pd.DataFrame(data)
    df = df.sort_values(by="Days to Expiry")
    df['Structure Alert'] = df["Implied Volatility (%)"].diff().apply(
        lambda x: "🔻 Backwardation" if x < 0 else ("🔺 Contango" if x > 0 else "")
    )
    df.iloc[0, df.columns.get_loc('Structure Alert')] = ""  # No alert for first row
    return df

def get_skew_data(ticker, expiration):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiration)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    current_price = stock.history(period="1d")['Close'].iloc[-1]

    call_data = calls[['strike', 'impliedVolatility']].rename(columns={'impliedVolatility': 'Call IV'})
    put_data = puts[['strike', 'impliedVolatility']].rename(columns={'impliedVolatility': 'Put IV'})

    merged = pd.merge(call_data, put_data, on='strike')
    merged = merged[(merged['strike'] >= current_price * 0.8) & (merged['strike'] <= current_price * 1.2)]
    merged = merged.sort_values(by='strike')
    return merged

if ticker:
    st.markdown(f"### 📅 Term Structure for **{ticker}**")
    df_term = get_term_structure(ticker)

    if not df_term.empty:
        st.dataframe(df_term, use_container_width=True)



        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_term["Days to Expiry"],
            y=df_term["Implied Volatility (%)"],
            mode='lines+markers',
            name='IV Term Structure'
        ))

        fig.update_layout(
            title=f"Term Structure for {ticker}",
            xaxis_title="Days to Expiry",
            yaxis_title="Implied Volatility (%)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


        st.markdown("---")
        st.subheader("📉 Call/Put Skew (Select Expiration)")

        selected_exp = st.selectbox("Select Expiration for Skew Chart:", df_term['Expiration'].tolist())

        df_skew = get_skew_data(ticker, selected_exp)
        if not df_skew.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                       x=df_skew['strike'], y=df_skew['Call IV'], mode='lines+markers', name='Call IV',
                line=dict(dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=df_skew['strike'], y=df_skew['Put IV'], mode='lines+markers', name='Put IV'
            ))
            fig.update_layout(
                title=f"Volatility Skew for {ticker} - Exp {selected_exp}",
                xaxis_title="Strike",
                yaxis_title="Implied Volatility",
                hovermode="x unified",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not load skew data for the selected expiration.")



           # --- Strike Skew Table (90%–110%) ---
    st.subheader("📏 Strike Skew Table (90%-ATM-110%)")
    current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    strike_90 = df_skew.iloc[(df_skew['strike'] - current_price * 0.90).abs().argsort()[:1]]['strike'].values[0]
    strike_atm = df_skew.iloc[(df_skew['strike'] - current_price).abs().argsort()[:1]]['strike'].values[0]
    strike_110 = df_skew.iloc[(df_skew['strike'] - current_price * 1.10).abs().argsort()[:1]]['strike'].values[0]





    skew_row = df_skew[df_skew['strike'].isin([strike_90, strike_atm, strike_110])].copy()


    skew_row = df_skew[df_skew['strike'].isin([strike_90, strike_atm, strike_110])].copy()
    skew_row = skew_row.set_index('strike').reindex([strike_90, strike_atm, strike_110])

    if not skew_row.isnull().values.any():
        iv_90 = skew_row.loc[strike_90, ['Call IV', 'Put IV']].mean()
        iv_110 = skew_row.loc[strike_110, ['Call IV', 'Put IV']].mean()
        skew_value = round(iv_90 - iv_110, 2)

        skew_table = pd.DataFrame({
            "Strike": [strike_90, strike_atm, strike_110],
            "Avg IV": [round(iv_90, 2),
                    round(skew_row.loc[strike_atm, ['Call IV', 'Put IV']].mean(), 2),
                    round(iv_110, 2)],
            "Skew Alert": [
                "🔴 Steep Skew" if skew_value > 5 else
                "🟠 Moderate Skew" if skew_value > 2 else
                "🟢 Flat", "", ""
            ]
        })

        st.dataframe(skew_table, use_container_width=True)
    else:
        st.warning("Could not calculate 90–110% skew: missing strike data.")

else:
    st.warning("Could not load skew data for the selected expiration.")