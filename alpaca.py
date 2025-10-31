


from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime

# 🔑 Replace with your actual Alpaca API keys
API_KEY = "AKXO78BQSEVMYK915DKS"
API_SECRET = "7ih6FVD9Y03zfXeGDY0RhY1UQ10jQxPHs96ySvkp"
# ✅ Create the client
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# 🗓️ Define the request parameters
request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Minute,  # or TimeFrame.Day, TimeFrame.Hour, etc.
    start=datetime.datetime(2024, 4, 25),
    end=datetime.datetime(2024, 4, 26),
)

# 📊 Get historical bars
bars = client.get_stock_bars(request_params)

# 🖨️ Print the result
df = bars.df  # This is a pandas DataFrame
print(df)
