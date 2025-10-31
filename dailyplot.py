
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime


t = "AAPL"  # Default ticker if none is passed

def main():
    st.title("📆 Daily Candlestick Chart")
    try:
        with st.expander("📆 Daily Chart Overview", expanded=True):


            # 🎯 Ticker selection
            tickers = ["SPY", "QQQ","low" ,"hd","NVDA", "AVGO","AMD","PLTR","MRVL","uber","mu","crwd","AMZN","AAPL","googl","MSFT","META","tsla","sbux","nke","chwy","DKNG","GM","cmg","c","wfc","hood","coin","bac","jpm","PYPL","tgt","wmt","elf"]
            t = st.selectbox("Select a Ticker", tickers)

            # 📅 Date range input
            start_date = st.date_input("Start Date", value=datetime(2024, 12, 1))
            end_date = st.date_input("End Date", value=datetime.today())

            # 🛑 Check range logic
            if start_date >= end_date:
                st.error("Start date must be before end date.")
                st.stop()


            daily_chart_data = yf.download(t, start=start_date, end=end_date, interval="1d", progress=False)

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

                    xaxis_rangeslider_visible=False,
                    margin=dict(l=30, r=30, t=40, b=20)
                )

                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.warning("No daily data available.")
    except Exception as e:
        st.error(f"Failed to load daily chart: {e}")

if __name__ == "__main__":
    main()
