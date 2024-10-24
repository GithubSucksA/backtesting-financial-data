import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
from typing import List, Tuple
import pandas_ta as ta
import time
import os

class TradingBacktester:
    def __init__(self, symbol: str, interval: str = '1440', initial_balance: float = 10000, risk_per_trade: float = 0.02):
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.df = self.fetch_data()
        self.check_data_quality()
        self.df = self.calculate_indicators(self.df)
        self.signals = self.calculate_signals(self.df)
        self.trades_executed = 0

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
        #print("Keltner Channel columns:", kc.columns)
        # Print the column names for Aberration
        #print("Aberration columns:", aberration.columns)

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
            trades_executed = 0
            
            print(f"Initial balance: {balance}")
            print(f"Risk per trade: {self.risk_per_trade}")

            for i in range(1, len(self.df)):
                current_price = self.df['close'].iloc[i]
                signal = self.signals.iloc[i]
                
                if signal == 1 and position == 0:
                    # Risk management: Use ATR for position sizing
                    risk_amount = balance * self.risk_per_trade
                    atr = max(self.df['ATR'].iloc[i], 0.001)  # Implement minimum ATR
                    position_size = risk_amount / (atr * 10)
                    cost = position_size * current_price * (1 + transaction_cost)
                    
                    print(f"Potential buy signal at {self.df.index[i]}")
                    print(f"Current balance: {balance:.2f}, ATR: {atr:.4f}")
                    print(f"Calculated position size: {position_size:.4f}, Cost: {cost:.2f}")
                    
                    if cost <= balance:
                        position = position_size
                        balance -= cost
                        trades_executed += 1
                        print(f"Buy executed. New position: {position:.4f}, New balance: {balance:.2f}")
                    else:
                        print("Insufficient balance for buy")
                
                elif signal == -1 and position > 0:
                    # Sell
                    sell_value = position * current_price * (1 - transaction_cost)
                    balance += sell_value
                    print(f"Sell signal at {self.df.index[i]}")
                    print(f"Selling position: {position:.4f} at {current_price:.2f}")
                    print(f"Sell value: {sell_value:.2f}, New balance: {balance:.2f}")
                    position = 0
                    trades_executed += 1
                
                current_value = balance + position * current_price
                balances.append(current_value)
                returns.append((current_value - balances[i-1]) / balances[i-1])

            print(f"Total trades executed: {trades_executed}")
            print(f"Final balance: {balances[-1]:.2f}")
            self.trades_executed = trades_executed
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
        os.makedirs('plots', exist_ok=True)
        filename = f'{safe_symbol}_{self.interval}_backtest_results.png'
        filepath = os.path.join('plots', filename)
        plt.savefig(filepath)
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
        #self.display_data()
        balances, returns = self.backtest()
        self.plot_results(balances)
        #safe_symbol = self.symbol.replace('/', '_')
        #filename = f'{safe_symbol}_{self.interval}_backtest_results.png'
        return self.print_performance_metrics(balances, returns)
        #print(f"Plot saved as plots/{filename}")

    def print_performance_metrics(self, balances: List[float], returns: List[float]) -> dict:
        total_return = (balances[-1] - balances[0]) / balances[0] if balances[0] != 0 else 0
        
        std_returns = np.std(returns)
        if std_returns != 0 and not np.isnan(std_returns):
            sharpe_ratio = np.mean(returns) / std_returns * np.sqrt(365)
        else:
            sharpe_ratio = 0
        
        max_balance = np.max(balances)
        max_drawdown = np.max(np.maximum.accumulate(balances) - balances) / max_balance if max_balance != 0 else 0
        win_rate = np.sum(np.array(returns) > 0) / len(returns) if len(returns) > 0 else 0

        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trades_executed": self.trades_executed  # Add this line
        }

        #print(f"Total Return: {total_return:.2%}")
        #print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        #print(f"Max Drawdown: {max_drawdown:.2%}")
        #print(f"Win Rate: {win_rate:.2%}")

    def check_data_quality(self):
        print(f"Total rows in dataframe: {len(self.df)}")
        print(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
        print(f"Any NaN values: {self.df.isna().any().any()}")
        print(f"Sample of close prices: {self.df['close'].head()}")

# Usage
if __name__ == "__main__":
    start_time = time.time()
    print(f"START TIME {start_time}")
    symbols = ["AAVEUSD", "BTC/USD", "ETH/USD", "ADAUSDT", "ALGOUSDT", "ATOMUSDT", "AVAXUSDT", "DOTUSDT", "CRVUSD", "EGLDUSD", "ENJUSD", "EWTUSD", "FETUSD", "FILUSD", "FLOKIUSD", "FLOWUSD", "GALAUSD", "GMXUSD", "ICPUSD", "INJUSD", "LINKUSDT", "LTCUSDT", "MANAUSDT", "LRCUSD", "MATICUSDT", "MINAUSD", "MKRUSD", "NEARUSD", "OCEANUSD", "PENDLEUSD", "PEPEUSD", "QNTUSD", "PYTHUSD", "RENDERUSD", "SANDUSD", "SHIBUSD", "SOLUSDT", "TAOUSD", "TRXUSD", "UNIUSD", "WIFUSD", "XDGUSD"] 
    intervals = [15, 30, 60, 240, 1440, 10080, 21600]

    all_results = []

    for interval in intervals:
        interval_results = []
        for symbol in symbols:
            try:
                backtester = TradingBacktester(symbol, str(interval))
                result = backtester.run_backtest()
                interval_results.append(result)
                if result['total_return'] == 0 and result['trades_executed'] == 0:
                    print(f"Warning: No trades executed for {symbol} with interval {interval}")
                    backtester.check_data_quality()
            except Exception as e:
                print(f"Error running backtest for {symbol} with interval {interval}: {e}")
        
        # Sort interval results by total return
        sorted_results = sorted(interval_results, key=lambda x: x['total_return'], reverse=True)
        
        # Display top 5 performers for this interval
        print(f"\nTop 5 performers for interval {interval}:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i}. {result['symbol']}: Total Return: {result['total_return']:.2%}, Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        
        # Calculate and display average metrics for this interval
        avg_return = np.mean([r['total_return'] for r in interval_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in interval_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in interval_results])
        avg_winrate = np.mean([r['win_rate'] for r in interval_results])
        
        print(f"\nAverage metrics for interval {interval}:")
        print(f"Avg Return: {avg_return:.2%}, Avg Sharpe: {avg_sharpe:.2f}, Avg Drawdown: {avg_drawdown:.2%}, Avg Win Rate: {avg_winrate:.2%}")
        
        all_results.extend(interval_results)
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('backtest_results.csv', index=False)
    print("\nDetailed results saved to 'backtest_results.csv'")
    
    end_time = time.time()
    print(f"\nEND TIME {end_time}")
    total_duration = end_time - start_time
    print(f"TOTAL TIME {total_duration:.2f} seconds")