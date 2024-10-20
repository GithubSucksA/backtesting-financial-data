import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
from typing import List, Tuple
import pandas_ta as ta

class TradingBacktester:
    def __init__(self, symbol: str, interval: str = '1440', initial_balance: float = 10000, risk_per_trade: float = 0.02):
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.df = self.fetch_data()
        self.df = self.calculate_indicators(self.df)
        self.signals = self.calculate_signals(self.df)

    def fetch_data(self) -> pd.DataFrame:
        url = f"https://api.kraken.com/0/public/OHLC?pair={self.symbol}&interval={self.interval}"
    
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
        df = pd.DataFrame(data['result'][self.symbol], 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = df['timestamp'].apply(format_timestamp)
        df.set_index('timestamp', inplace=True)
        
        # Convert price and volume columns to appropriate numeric types, because it is a json string
        numeric_columns = ['open', 'high', 'low', 'close', 'vwap', 'volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO:Add multiple technical indicators using pandas_ta
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        # Add Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2, mamode='ema', offset=0)
        aberration = ta.aberration(df['high'], df['low'], df['close'], length=20, mamode='ema', offset=0)
        
        # Print the column names to see what's available
        print("Keltner Channel columns:", kc.columns)
        # Print the column names for Aberration
        print("Aberration columns:", aberration.columns)

        # Use the first three columns (assuming they are Lower, Middle, and Upper)
        df['KC_LOWER'] = kc.iloc[:, 0]
        df['KC_MIDDLE'] = kc.iloc[:, 1]
        df['KC_UPPER'] = kc.iloc[:, 2]

        df['ABERRATION_ZG'] = aberration.iloc[:, 0]
        df['ABERRATION_SG'] = aberration.iloc[:, 1]
        df['ABERRATION_XG'] = aberration.iloc[:, 2]
        df['ABERRATION_ATR'] = aberration.iloc[:, 3]
        return df

    def calculate_signals(self, df: pd.DataFrame) -> pd.Series:
        # TODO: Implement a more sophisticated signal generation strategy
        buy_signal = (
            (df['close'] > df['MA20']) &
            (df['MA20'] > df['MA50']) &
            (df['RSI'] < 70) &
            (df['MACD'] > df['MACD_signal']) &
            (df['close'] > df['KC_LOWER']) &  # Price above lower Keltner Channel
            (df['close'] < df['KC_UPPER']) &  # Price below upper Keltner Channel
            (df['close'] > df['ABERRATION_SG'])  # Price above Aberration support
        )
        sell_signal = (
            (df['close'] < df['MA20']) |
            (df['RSI'] > 70) |
            (df['MACD'] < df['MACD_signal']) |
            (df['close'] > df['KC_UPPER']) |  # Price above upper Keltner Channel
            (df['close'] < df['ABERRATION_SG'])  # Price below Aberration support
        )
        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)

    def backtest(self) -> Tuple[List[float], List[float]]:
        position = 0
        balance = self.initial_balance
        balances = [balance]
        returns = [0]
        transaction_cost = 0.001  # 0.1% transaction cost

        for i in range(1, len(self.df)):
            if self.signals.iloc[i] == 1 and position == 0:
                # Risk management: Use ATR for position sizing
                risk_amount = balance * self.risk_per_trade
                atr = self.df['ATR'].iloc[i]
                position_size = risk_amount / atr
                cost = position_size * self.df['close'].iloc[i] * (1 + transaction_cost)
                if cost <= balance:
                    position = position_size
                    balance -= cost
            elif self.signals.iloc[i] == -1 and position > 0:
                # Sell
                balance += position * self.df['close'].iloc[i] * (1 - transaction_cost)
                position = 0
            
            current_value = balance + position * self.df['close'].iloc[i]
            balances.append(current_value)
            returns.append((current_value - balances[i-1]) / balances[i-1])

        return balances, returns

    def plot_results(self, balances: List[float]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(self.df.index, self.df['close'], label='Close Price')
        ax1.plot(self.df.index, self.df['MA20'], label='20-day MA')
        buy_signals = self.df.index[self.signals == 1]
        sell_signals = self.df.index[self.signals == -1]
        ax1.scatter(buy_signals, self.df.loc[buy_signals, 'close'], marker='^', color='g', label='Buy')
        ax1.scatter(sell_signals, self.df.loc[sell_signals, 'close'], marker='v', color='r', label='Sell')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.set_title(f'{self.symbol} Price and Signals')

        ax2.plot(self.df.index, balances, label='Portfolio Value')
        ax2.set_ylabel('Portfolio Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.set_title('Portfolio Value Over Time')

        plt.tight_layout()
        safe_symbol = self.symbol.replace('/', '_')
        plt.savefig(f'{safe_symbol}_backtest_results.png')
        plt.close(fig)  # Close the figure to free up memory

    def display_data(self):
        try:
            print("\nFirst few rows of the dataset:")
            display(self.df.head())
            
            print("\nLast few rows of the dataset:")
            display(self.df.tail())
            
            print("\nDataFrame with signals:")
            display(self.df.join(self.signals.rename('signal')))
        except NameError:
            print("\nFirst few rows of the dataset:")
            print(self.df.head().to_string())
            
            print("\nLast few rows of the dataset:")
            print(self.df.tail().to_string())
            
            print("\nDataFrame with signals:")
            print(self.df.join(self.signals.rename('signal')).to_string())

    def run_backtest(self):
        self.display_data()
        balances, returns = self.backtest()
        self.plot_results(balances)
        self.print_performance_metrics(balances, returns)
        safe_symbol = self.symbol.replace('/', '_')
        print(f"Plot saved as {safe_symbol}_backtest_results.png")

    def print_performance_metrics(self, balances: List[float], returns: List[float]):
        total_return = (balances[-1] - balances[0]) / balances[0]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_drawdown = np.max(np.maximum.accumulate(balances) - balances) / np.max(balances)
        win_rate = np.sum(np.array(returns) > 0) / len(returns)

        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Win Rate: {win_rate:.2%}")

# Usage
if __name__ == "__main__":
    symbols = ["AAVEUSD"]
    for symbol in symbols:
        print(f"\nRunning backtest for {symbol}")
        backtester = TradingBacktester(symbol, 15)
        backtester.run_backtest()