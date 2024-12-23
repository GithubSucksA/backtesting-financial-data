# ... (keep the existing imports)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
from typing import List, Tuple
try:
    import pandas_ta as ta
except ImportError:
    print("Error: pandas_ta is not installed. Please install it using 'pip install pandas_ta'")
    print("If you've already installed it, make sure you're using the correct Python environment.")
    import sys
    sys.exit(1)

# ... (keep the existing fetch_data function)
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
    
    # Convert price and volume columns to appropriate numeric types, because it is a json string
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
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Add multiple technical indicators using pandas_ta
    df['MA20'] = ta.sma(df['close'], length=20)
    df['MA50'] = ta.sma(df['close'], length=50)
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df

def calculate_signals(df: pd.DataFrame) -> pd.Series:
    # Implement a more sophisticated signal generation strategy
    buy_signal = (
        (df['close'] > df['MA20']) &
        (df['MA20'] > df['MA50']) &
        (df['RSI'] < 70) &
        (df['MACD'] > df['MACD_signal'])
    )
    sell_signal = (
        (df['close'] < df['MA20']) |
        (df['RSI'] > 70) |
        (df['MACD'] < df['MACD_signal'])
    )
    return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)

# Calculate indicators and signals
df = calculate_indicators(df)
signals = calculate_signals(df)

# ... (keep the existing display code)
# Display the DataFrame with signals
print("DataFrame with signals:")
# display is a function within jupyter notebook, that allows us to display the dataframe
display(df.join(signals.rename('signal')))

# Cell 3: Backtesting
def backtest(df: pd.DataFrame, signals: pd.Series, initial_balance: float = 10000, risk_per_trade: float = 0.02) -> Tuple[List[float], List[float]]:
    position = 0
    balance = initial_balance
    balances = [balance]
    returns = [0]
    transaction_cost = 0.001  # 0.1% transaction cost

    for i in range(1, len(df)):
        if signals.iloc[i] == 1 and position == 0:
            # Risk management: Use ATR for position sizing
            risk_amount = balance * risk_per_trade
            atr = df['ATR'].iloc[i]
            position_size = risk_amount / atr
            cost = position_size * df['close'].iloc[i] * (1 + transaction_cost)
            if cost <= balance:
                position = position_size
                balance -= cost
        elif signals.iloc[i] == -1 and position > 0:
            # Sell
            balance += position * df['close'].iloc[i] * (1 - transaction_cost)
            position = 0
        
        current_value = balance + position * df['close'].iloc[i]
        balances.append(current_value)
        returns.append((current_value - balances[i-1]) / balances[i-1])

    return balances, returns

# Run backtest
balances, returns = backtest(df, signals)

# Calculate performance metrics
total_return = (balances[-1] - balances[0]) / balances[0]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
max_drawdown = np.max(np.maximum.accumulate(balances) - balances) / np.max(balances)
win_rate = np.sum(np.array(returns) > 0) / len(returns)

print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Win Rate: {win_rate:.2%}")

# ... (keep the existing plot_results function and plotting code)
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