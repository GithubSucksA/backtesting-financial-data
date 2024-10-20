# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
from typing import List, Tuple

# Cell 1: Data Fetching
def fetch_data() -> pd.DataFrame:
    url = "https://api.kraken.com/0/public/OHLC?pair=AAVEUSD&interval=1440"
    
    payload = {}
    headers = {
      'Accept': 'application/json'
    }
    
    response = requests.request("GET", url, headers=headers, data=payload)
    data = response.json()
    
    # Function to convert timestamp to formatted string
    def format_timestamp(timestamp):
        return datetime.datetime.fromtimestamp(timestamp)
    
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data['result']['AAVEUSD'], 
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = df['timestamp'].apply(format_timestamp)
    df.set_index('timestamp', inplace=True)
    
    # Convert price and volume columns to appropriate numeric types
    numeric_columns = ['open', 'high', 'low', 'close', 'vwap', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    
    return df

# Fetch the data
df = fetch_data()

# Display the first few rows
print("First few rows of the dataset:")
display(df.head())

# Display the last few rows
print("\nLast few rows of the dataset:")
display(df.tail())

# Cell 2: Data Analysis and Signal Generation
def calculate_signals(df: pd.DataFrame) -> pd.Series:
    # Implement your trading strategy here
    # This is a simple example using a 20-day moving average crossover
    df['MA20'] = df['close'].rolling(window=20).mean()
    return pd.Series(np.where(df['close'] > df['MA20'], 1, 0), index=df.index)

# Calculate signals
signals = calculate_signals(df)

# Display the DataFrame with signals
print("DataFrame with signals:")
display(df.join(signals.rename('signal')))

# Cell 3: Backtesting
def backtest(df: pd.DataFrame, signals: pd.Series) -> Tuple[List[float], List[float]]:
    position = 0
    balance = 10000  # Starting with $10,000
    balances = [balance]
    returns = [0]

    for i in range(1, len(df)):
        if signals.iloc[i] == 1 and position == 0:
            # Buy
            position = balance / df['close'].iloc[i]
            balance = 0
        elif signals.iloc[i] == 0 and position > 0:
            # Sell
            balance = position * df['close'].iloc[i]
            position = 0
        
        current_value = balance + position * df['close'].iloc[i]
        balances.append(current_value)
        returns.append((current_value - balances[i-1]) / balances[i-1])

    return balances, returns

# Run backtest
balances, returns = backtest(df, signals)

# Calculate performance metrics
total_return = (balances[-1] - balances[0]) / balances[0]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assuming 252 trading days in a year

print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Cell 4: Plotting Results
def plot_results(df: pd.DataFrame, balances: List[float], signals: pd.Series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(df.index, df['close'], label='Close Price')
    ax1.plot(df.index, df['MA20'], label='20-day MA')
    buy_signals = df.index[signals == 1]
    sell_signals = df.index[signals == 0]
    ax1.scatter(buy_signals, df.loc[buy_signals, 'close'], marker='^', color='g', label='Buy')
    ax1.scatter(sell_signals, df.loc[sell_signals, 'close'], marker='v', color='r', label='Sell')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.set_title('AAVEUSD Price and Signals')

    ax2.plot(df.index, balances, label='Portfolio Value')
    ax2.set_ylabel('Portfolio Value')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.set_title('Portfolio Value Over Time')

    plt.tight_layout()
    plt.show()

# Plot results
plot_results(df, balances, signals)