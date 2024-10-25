import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import requests
import datetime
from typing import List, Tuple
#import time
import os
#import mplfinance as mpf  # Make sure to install mplfinance
from abc import ABC, abstractmethod

class BaseTradingBacktester(ABC):
    def __init__(self, symbol: str, interval: str = '1440', initial_balance: float = 10000, risk_per_trade: float = 0.05):
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.df = self.fetch_data()
        self.check_data_quality()
        self.df = self.calculate_indicators(self.df)
        self.signals = self.calculate_signals(self.df)
        self.trades_executed = 0
        self.trading_volume = 0
        self.winning_trades = 0
        self.losing_trades = 0

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

    @abstractmethod
    def calculate_indicators(self, df):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def calculate_signals(self, df):
        raise NotImplementedError("Subclasses should implement this method.")

    def backtest(self) -> Tuple[List[float], List[float], float, float, float, float]:
        position = 0
        balance = self.initial_balance
        balances = [balance]
        returns = [0]
        transaction_cost = 0.0033  # 0.1% transaction cost
        trades_executed = 0
        self.trading_volume = 0
        self.winning_trades = 0
        self.losing_trades = 0
        last_buy_price = 0
        fees_paid = []
        
        leverage = 2  # Example leverage factor
        stop_loss_percentage = 0.02  # 2% stop loss

        for i in range(1, len(self.df)):
            current_price = self.df['close'].iloc[i]
            signal = self.signals.iloc[i]
            
            # Check for stop-loss condition
            if position > 0 and current_price < last_buy_price * (1 - stop_loss_percentage) and signal != 1:
            #if position > 0 and current_price < last_buy_price * (1 - stop_loss_percentage):

                if not np.isnan(self.df['RSI'].iloc[i]):  # Ensure RSI is valid
                # Trigger stop-loss
                    sell_value = position * current_price * (1 - transaction_cost)
                    balance += sell_value
                    self.trading_volume += sell_value
                    self.losing_trades += 1
                    position = 0
                    trades_executed += 1
                    fees_paid.append(sell_value * transaction_cost)
                    last_buy_price = 0  # Reset last buy price after stop-loss
                    #print(f"\033[91mSTOP-LOSS TRIGGERED AT: {current_price:.2f}, BALANCE: {balance:.2f}, TIME: {self.get_trade_time(i)}\033[0m\n")

            if signal == 1 and position == 0:
                # Risk management: Use ATR for position sizing
                cost = self.risk_per_trade * balance * leverage  # Include leverage in cost
                position_size = (cost * (1 - transaction_cost)) / current_price  # Adjust position size calculation
                
                if cost <= balance:
                    position = position_size
                    balance -= cost
                    last_buy_price = current_price
                    fees_paid.append(cost * transaction_cost)
                    trade_time = self.get_trade_time(i)
                    #print(f"\033[92mBUY: {self.symbol},  BUY AT: {current_price:.2f}, BALANCE: {balance:.2f}, FEE: {cost * transaction_cost}, TIME: {trade_time}\033[0m\n")
                else:
                    print("Insufficient balance for buy")
            
            elif signal == -1 and position > 0:
                # Sell
                sell_value = position * current_price * (1 - transaction_cost)
                balance += sell_value
                self.trading_volume += sell_value
                trade_time = self.get_trade_time(i)
                #print(f"\033[91mSELL AT {current_price:.2f}, BALANCE: {balance:.2f}, FEE: {cost * transaction_cost}, TIME: {trade_time}\033[0m\n")

                # Determine if it's a winning or losing trade
                buy_value = position * last_buy_price  # Calculate the total buy value
                if sell_value > buy_value:  # Compare sell value with the buy value
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                position = 0
                trades_executed += 1
                fees_paid.append(sell_value * transaction_cost)
            
            current_value = balance + position * current_price
            balances.append(current_value)
            returns.append((current_value - balances[i-1]) / balances[i-1])

        # Calculate total fees paid
        total_fees_paid = sum(fees_paid)

        self.trades_executed = trades_executed
        return balances, returns, position, last_buy_price, balances[-1] - self.initial_balance, total_fees_paid, self.trading_volume

    def plot_results(self, balances: List[float]):
        # Limit the number of data points to plot
        max_data_points = 500  # Adjust this number as needed
        df_to_plot = self.df.tail(max_data_points)  # Get the last 'max_data_points' rows

        # Debugging output to check the DataFrame
        print("DataFrame to plot:")
        print(df_to_plot)

        # Check if the DataFrame is empty or missing necessary columns
        if df_to_plot.empty or 'MA20' not in df_to_plot or 'MA50' not in df_to_plot or 'AO' not in df_to_plot:
            print("Error: DataFrame is empty or missing required columns for plotting.")
            return  # Exit the function if there's an issue

        # Create a new figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Debugging output to check the type of axes
        print(f"Type of ax1: {type(ax1)}, Type of ax2: {type(ax2)}")

        # First plot: Candlestick chart with buy/sell signals
        mpf.plot(df_to_plot, type='candle', ax=ax1, volume=False, style='charles', title=f'{self.symbol} Price and Signals', ylabel='Price', addplot=[
            mpf.make_addplot(df_to_plot['MA20'], color='blue', title='20-day MA'),
            mpf.make_addplot(df_to_plot['MA50'], color='orange', title='50-day MA'),
            mpf.make_addplot(df_to_plot['AO'], panel=1, color='orange', title='AO'),
        ])

        # Add buy/sell signals
        buy_signals = df_to_plot.index[self.signals.tail(max_data_points) == 1]
        sell_signals = df_to_plot.index[self.signals.tail(max_data_points) == -1]
        ax1.scatter(buy_signals, df_to_plot.loc[buy_signals, 'close'], marker='^', color='g', label='Buy', s=100)
        ax1.scatter(sell_signals, df_to_plot.loc[sell_signals, 'close'], marker='v', color='r', label='Sell', s=100)

        # Second plot: Volume bars
        ax2.bar(df_to_plot.index, df_to_plot['volume'], color='lightblue', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.set_title('Volume Over Time')

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
        self.display_data()
        balances, returns, final_position, last_buy_price, net_profit_loss, total_fees_paid, trading_volume = self.backtest()
        #self.plot_results(balances)
        #safe_symbol = self.symbol.replace('/', '_')
        #filename = f'{safe_symbol}_{self.interval}_backtest_results.png'
        return self.print_performance_metrics(balances, returns, final_position, last_buy_price, net_profit_loss, total_fees_paid, trading_volume)
        #print(f"Plot saved as plots/{filename}")

    def print_performance_metrics(self, balances: List[float], returns: List[float], final_position: float, last_buy_price: float, net_profit_loss: float, total_fees_paid: float, trading_volume: float) -> dict:
        total_return = (balances[-1] - balances[0]) / balances[0] if balances[0] != 0 else 0
        
        std_returns = np.std(returns)
        if std_returns != 0 and not np.isnan(std_returns):
            sharpe_ratio = np.mean(returns) / std_returns * np.sqrt(365)
        else:
            sharpe_ratio = 0
        
        max_balance = np.max(balances)
        max_drawdown = np.max(np.maximum.accumulate(balances) - balances) / max_balance if max_balance != 0 else 0
        win_rate = np.sum(np.array(returns) > 0) / len(returns) if len(returns) > 0 else 0

        win_rate = self.winning_trades / self.trades_executed if self.trades_executed > 0 else 0

        current_price = self.df['close'].iloc[-1]
        open_position_value = final_position * current_price if final_position > 0 else 0
        open_position_return = ((current_price - last_buy_price) / last_buy_price) if final_position > 0 else 0

        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trades_executed": self.trades_executed,
            "trading_volume": trading_volume,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "open_position": final_position,
            "open_position_value": open_position_value,
            "open_position_return": open_position_return,
            "net_profit_loss": net_profit_loss,  # New field
            "total_fees_paid": total_fees_paid   # New field
        }

        #print(f"Total Return: {total_return:.2%}")
        #print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        #print(f"Max Drawdown: {max_drawdown:.2%}")
        #print(f"Win Rate: {win_rate:.2%}")

    def check_data_quality(self):
        pass
        #if len(self.df) != 720:
            #print(f"Total rows in dataframe: {len(self.df)}")

        #if self.df.isna().any().any():
            #print("Warning: NaN values found in the dataset")
            # Optionally, you can print which columns contain NaN values:
            #nan_columns = self.df.columns[self.df.isna().any()].tolist()
            #print(f"Columns with NaN values: {nan_columns}")

    def get_date_range(self):
        return f"Date range: {self.df.index[0]} to {self.df.index[-1]}"

    def get_trade_time(self, index: int) -> str:
        """Return the exact date and time of a trade based on the index."""
        return str(self.df.index[index])

