import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Modern Day Trading Dashboard",
    layout="wide"
)
# Sidebar for group + date selection
groups = {
    'ETFS': ['spy','qqq'],
    'banks diversified': ['wfc', 'c', 'jpm', 'bac'],
    'credit services': ['v', 'ma', 'axp', 'pypl'],
    'Semiconductor':['nvda','avgo','amd','mu','mrvl','qcom','txn','intc'],
    'Software Infracstructure':['msft','orcl','pltr','crwd','panw','gddy','afrm','tost','aapl'],
    'Communication':['nflx','dis','googl','roku','spot','dash','meta'],
    'Cyclical':['amzn','tsla','sbux','nke','abnb'],
}
selected_group = st.sidebar.selectbox("Select a sector group", list(groups.keys()))
tickers = groups[selected_group]

start_date = st.sidebar.date_input("Start date", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("2025-04-15"))

run_button = st.sidebar.button("▶️ Run Analysis")

if run_button:
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            st.warning(f"⚠️ Failed to load {ticker}: {e}")

    if not data:
        st.error("❌ No data available for selected group.")
        st.stop()

    # OHLC from individual tickers
    ohlc = pd.DataFrame(index=list(data.values())[0].index)
    ohlc['Open'] = np.mean([df['Open'] for df in data.values()], axis=0)
    ohlc['High'] = np.mean([df['High'] for df in data.values()], axis=0)
    ohlc['Low'] = np.mean([df['Low'] for df in data.values()], axis=0)
    ohlc['Close'] = np.mean([df['Close'] for df in data.values()], axis=0)
    ohlc['Volume'] = np.sum([df['Volume'] for df in data.values()], axis=0)

    # Ichimoku
    high9 = ohlc['High'].rolling(9).max()
    low9 = ohlc['Low'].rolling(9).min()
    tenkan = (high9 + low9) / 2

    high26 = ohlc['High'].rolling(26).max()
    low26 = ohlc['Low'].rolling(26).min()
    kijun = (high26 + low26) / 2

    senkou_a = ((tenkan + kijun) / 2).shift(26)
    high52 = ohlc['High'].rolling(52).max()
    low52 = ohlc['Low'].rolling(52).min()
    senkou_b = ((high52 + low52) / 2).shift(26)
    chikou = ohlc['Close'].shift(-26)

    # Bollinger Bands
    middle = ohlc['Close'].rolling(20).mean()
    std = ohlc['Close'].rolling(20).std()
    upper = middle + 2 * std
    lower = middle - 2 * std

    # --- BBW Tightness 🐝 Detection ---
    bbw = ((upper - lower) / middle) * 100
    bbw_percentile = np.percentile(bbw.dropna(), 10)
    tight_mask = bbw < bbw_percentile

    # Rolling window of 5 bars, check for 3+ tight values
    tight_cluster = tight_mask.rolling(5).sum() >= 3




    # --- BBW Ratio Expansion 🔥 Detection (dashboard style) ---
    lookback = 5
    bbw_anchor = bbw.shift(lookback)
    bbw_ratio = bbw / bbw_anchor

    # You can store these in the DataFrame for debugging or plotting
    ohlc["BBW"] = bbw
    ohlc["BBW_Ratio"] = bbw_ratio

    # Apply the 🔥 logic
    bbw_alerts = []
    for ratio in bbw_ratio:
        if pd.isna(ratio):
            bbw_alerts.append("")
        elif ratio >= 3:
            bbw_alerts.append("🔥")  # Triple expansion
        elif ratio >= 2:
            bbw_alerts.append("🔥")  # Double expansion
        else:
            bbw_alerts.append("")

    ohlc["BBW_Alert"] = bbw_alerts



    ohlc["F_numeric"] = (ohlc["Close"] - ohlc["Close"].mean()) / ohlc["Close"].std()
    def calculate_td_demand_supply_lines_fpercent(df):
        df['TD Demand Line F'] = np.nan
        df['TD Supply Line F'] = np.nan
        demand, supply = [], []
        f_vals = df['F_numeric'].to_numpy()

        for i in range(1, len(df) - 1):
            if f_vals[i] < f_vals[i - 1] and f_vals[i] < f_vals[i + 1]:
                demand.append(f_vals[i])
                df.at[df.index[i], 'TD Demand Line F'] = max(demand[-2:]) if len(demand) >= 2 else demand[-1]
            if f_vals[i] > f_vals[i - 1] and f_vals[i] > f_vals[i + 1]:
                supply.append(f_vals[i])
                df.at[df.index[i], 'TD Supply Line F'] = min(supply[-2:]) if len(supply) >= 2 else supply[-1]

        df['TD Demand Line F'] = df['TD Demand Line F'].ffill()
        df['TD Supply Line F'] = df['TD Supply Line F'].ffill()
        return df

    ohlc = calculate_td_demand_supply_lines_fpercent(ohlc)
    # Map F% values back to price space for plotting
    mean_price = ohlc["Close"].mean()
    std_price = ohlc["Close"].std()

    ohlc["TD Demand Price"] = (ohlc["TD Demand Line F"] * std_price) + mean_price
    ohlc["TD Supply Price"] = (ohlc["TD Supply Line F"] * std_price) + mean_price

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=ohlc.index, open=ohlc['Open'], high=ohlc['High'],
                                  low=ohlc['Low'], close=ohlc['Close'], name="Price"))
    fig.add_trace(go.Scatter(x=tenkan.index, y=tenkan, line=dict(color='red'), name='Tenkan-sen'))
    fig.add_trace(go.Scatter(x=kijun.index, y=kijun, line=dict(color='green'), name='Kijun-sen'))
    fig.add_trace(go.Scatter(x=chikou.index, y=chikou, line=dict(color='purple'), name='Chikou'))
    fig.add_trace(go.Scatter(x=senkou_a.index, y=senkou_a, line=dict(color='orange'), name='Senkou A'))
    fig.add_trace(go.Scatter(x=senkou_b.index, y=senkou_b, line=dict(color='blue'), name='Senkou B'))
    fig.add_trace(go.Scatter(x=senkou_b.index, y=senkou_b, fill='tonexty',
                             fillcolor='rgba(128,128,128,0.2)', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=upper.index, y=upper, line=dict(color='grey', dash='dot'), name='Upper BB'))
    fig.add_trace(go.Scatter(x=middle.index, y=middle, line=dict(color='white', dash='dot'), name='Middle BB'))
    fig.add_trace(go.Scatter(x=lower.index, y=lower, line=dict(color='grey', dash='dot'), name='Lower BB'))

    fig.add_trace(go.Scatter(
        x=ohlc.index[tight_cluster],
        y=ohlc['Close'][tight_cluster],
        mode='text',
        textfont=dict(size=21),
        text=['🐝'] * tight_cluster.sum(),
        textposition='top center',
        showlegend=False,
        name='BBW Tight'

    ))

    fig.add_trace(go.Scatter(
            x=ohlc.index[ohlc["BBW_Alert"] == "🔥"],
            y=ohlc["Close"][ohlc["BBW_Alert"] == "🔥"],
            mode='text',
            text=ohlc["BBW_Alert"][ohlc["BBW_Alert"] == "🔥"],
            textposition='bottom center',
            name="BBW Expansion",
            showlegend=False
        ))


    fig.add_trace(go.Scatter(
        x=ohlc.index, y=ohlc["TD Demand Price"],
        line=dict(color="green", width=1, dash="dash"), name="TD Demand Line"
    ))

    fig.add_trace(go.Scatter(
        x=ohlc.index, y=ohlc["TD Supply Price"],
        line=dict(color="red", width=1, dash="dash"), name="TD Supply Line"
    ))

    fig.update_layout(title=f"Ichimoku + Bollinger: {selected_group}",
                      xaxis_title="Date", yaxis_title="Price",
                      template="plotly_white", hovermode="x unified",width=2000, height=1000)

    st.plotly_chart(fig, use_container_width=True)
    st.bar_chart(ohlc['Volume'])
