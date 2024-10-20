#!C:\Users\Ian\AppData\Local\Programs\Python\Python312\python.exe

import sys
print(sys.executable)
print(sys.path)

import pandas as pd
import pandas_ta as ta
import yfinance as yf

try:
    # Fetch data using yfinance
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1mo")  # Fetch 1 month of data

    # Add some technical indicators using pandas_ta
    df.ta.sma(length=20, append=True)
    df.ta.rsi(length=14, append=True)

    print(df.tail())
except ImportError:
    print("Please make sure yfinance is installed: pip install yfinance")
except Exception as e:
    print(f"An error occurred: {e}")



# Fetch historical data for EUR/USD
data = yf.download('EURUSD=X', start='2023-10-01', end='2024-07-10', interval='1d')

# Check the first 5 rows of data
print(data.head())