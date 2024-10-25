#!C:\Users\Ian\AppData\Local\Programs\Python\Python312\python.exe

import sys
print(sys.executable)
print(sys.path)

import pandas as pd
import pandas_ta as ta
import yfinance as yf

try:
    # Fetch data using yfinance
    ticker = yf.Ticker("BTC")
    df = ticker.history(period="1mo")  # Fetch 1 month of data

    # Add some technical indicators using pandas_ta
    df.ta.sma(length=20, append=True)
    df.ta.rsi(length=14, append=True)

    print(df.tail())
except ImportError:
    print("Please make sure yfinance is installed: pip install yfinance")
except Exception as e:
    print(f"An error occurred: {e}")



# Fetch historical data for Bitcoin (BTC)
data = yf.download('BTC-USD', period="1m", interval='15m')
print(data.head())

# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set the display width to a larger value
pd.set_option('display.max_colwidth', None)  # Allow columns to expand fully

# Print all data
print(data)
