import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
from typing import List, Tuple
import pandas_ta as ta
import time
import os
import subprocess
import sys

class TradingBacktester:

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

    def print_to_output_handler(self, message: str):
        with open('trade_messages.txt', 'a') as f:
            f.write(message + '\n')

    def print_to_main_terminal(self, message: str):
        print(message, flush=True)


    def fetch_data(self) -> pd.DataFrame:
        url = f"https://api.kraken.com/0/public/OHLC?pair={self.symbol}&interval={self.interval}"
        payload = {}
        headers = {'Accept': 'application/json'}
        
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
        
        # Convert price and volume columns to appropriate numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'vwap', 'volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)

        # Check for NaN values after indicator calculation
        if df[['MA20', 'MA50', 'RSI']].isna().any().any():
            print(f"NaN values found in indicators for {self.symbol}. Data shape: {df.shape}")
            df.dropna(inplace=True)  # Drop rows with NaN values if necessary

        return df

    def calculate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Ensure that the DataFrame has enough data
        if df.shape[0] < 50:  # Adjust this threshold as needed
            print(f"Not enough data to calculate signals for {self.symbol}. Data shape: {df.shape}")
            return pd.Series(index=df.index)  # Return an empty series with the same index

        buy_signal = (df['close'] > df['MA20'])
        sell_signal = (df['close'] < df['MA20'])
        signals = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)

        # Check the length of signals
        print(f"Signals calculated for {self.symbol}. Signals length: {len(signals)}, Data length: {len(df)}")
        return signals

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
            if i >= len(self.signals):  # Check if index is within bounds
                print(f"Index {i} out of bounds for signals. Signals length: {len(self.signals)}")
                break
            
            current_price = self.df['close'].iloc[i]
            signal = self.signals.iloc[i]
            
            # Check for stop-loss condition
            if position > 0 and current_price < last_buy_price * (1 - stop_loss_percentage):
                sell_value = position * current_price * (1 - transaction_cost)
                balance += sell_value
                self.trading_volume += sell_value
                self.losing_trades += 1
                position = 0
                trades_executed += 1
                fees_paid.append(sell_value * transaction_cost)
                last_buy_price = 0  # Reset last buy price after stop-loss
                self.print_to_output_handler(f"STOP-LOSS TRIGGERED AT: {current_price:.2f}, BALANCE: {balance:.2f}, TIME: {self.get_trade_time(i)}")
                continue

            if signal == 1 and position == 0:
                cost = self.risk_per_trade * balance * leverage  # Include leverage in cost
                position_size = (cost * (1 - transaction_cost)) / current_price
                
                if cost <= balance:
                    position = position_size
                    balance -= cost
                    trades_executed += 1
                    self.trading_volume += cost
                    last_buy_price = current_price
                    fees_paid.append(cost * transaction_cost)
                    self.print_to_output_handler(f"BUY AT: {current_price:.2f}, BALANCE: {balance:.2f}, FEE: {cost * transaction_cost}, TIME: {self.get_trade_time(i)}")
                else:
                    print("Insufficient balance for buy")
            
            elif signal == -1 and position > 0:
                sell_value = position * current_price * (1 - transaction_cost)
                balance += sell_value
                self.trading_volume += sell_value
                trade_time = self.get_trade_time(i)
                self.print_to_output_handler(f"SELL AT {current_price:.2f}, BALANCE: {balance:.2f}, FEE: {cost * transaction_cost}, TIME: {trade_time}")

                buy_value = position * last_buy_price
                if sell_value > buy_value:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                position = 0
                trades_executed += 1
                fees_paid.append(sell_value * transaction_cost)
            
            current_value = balance + position * current_price
            balances.append(current_value)
            returns.append((current_value - balances[i-1]) / balances[i-1])

        total_fees_paid = sum(fees_paid)
        self.trades_executed = trades_executed
        return balances, returns, position, last_buy_price, balances[-1] - self.initial_balance, total_fees_paid, self.trading_volume

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
        # Proceed with the backtest
        self.print_to_main_terminal(f"Running backtest for {self.symbol} with interval {self.interval}")
        balances, returns, final_position, last_buy_price, net_profit_loss, total_fees_paid, trading_volume = self.backtest()
        self.plot_results(balances)
        return self.print_performance_metrics(balances, returns, final_position, last_buy_price, net_profit_loss, total_fees_paid, trading_volume)

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
            "net_profit_loss": net_profit_loss,
            "total_fees_paid": total_fees_paid
        }

    def check_data_quality(self):
        pass

    def get_date_range(self):
        return f"Date range: {self.df.index[0]} to {self.df.index[-1]}"

    def get_trade_time(self, index: int) -> str:
        """Return the exact date and time of a trade based on the index."""
        return str(self.df.index[index])

def start_output_handler():
    if sys.platform == "win32":
        process = subprocess.Popen(['start', 'cmd', '/k', 'python', 'output_handler.py'], shell=True)
    else:
        process = subprocess.Popen(['gnome-terminal', '--', 'python3', 'output_handler.py'])
    time.sleep(1)  # Give the output handler time to start
    return process

def stop_output_handler(process):
    if process:
        process.terminate()


# Usage
if __name__ == "__main__":
    output_handler_process = start_output_handler()

    symbols = ["AAVEUSD", "BTC/USD", "ETH/USD", "ADAUSDT", "ALGOUSDT", "ATOMUSDT", "AVAXUSDT", "DOTUSDT", "CRVUSD", "EGLDUSD", "ENJUSD", "EWTUSD", "FETUSD", "FILUSD", "FLOKIUSD", "FLOWUSD", "GALAUSD", "GMXUSD", "ICPUSD", "INJUSD", "LINKUSDT", "LTCUSDT", "MANAUSDT", "LRCUSD", "MATICUSDT", "MINAUSD", "MKRUSD", "NEARUSD", "OCEANUSD", "PENDLEUSD", "PEPEUSD", "QNTUSD", "PYTHUSD", "RENDERUSD", "SANDUSD", "SHIBUSD", "SOLUSDT", "TAOUSD", "TRXUSD", "UNIUSD", "WIFUSD", "XDGUSD"] 
    intervals = [15, 30, 60, 240, 1440, 10080]

    all_results = []

    for interval in intervals:
        print(f"\n{interval} MIN")
        interval_results = []
        date_range_printed = False
        interval_total_trades = 0
        interval_winning_trades = 0
        interval_trading_volume = 0

        for symbol in symbols:
            try:
                backtester = TradingBacktester(symbol, str(interval))

                if not date_range_printed:
                    print(backtester.get_date_range())
                    date_range_printed = True

                result = backtester.run_backtest()
                interval_results.append(result)

                interval_total_trades += result['trades_executed']
                interval_winning_trades += result['winning_trades']
                interval_trading_volume += result['trading_volume']
            
                if result['total_return'] == 0 and result['trades_executed'] == 0:
                    print(f"Warning: No trades executed for {symbol} with interval {interval}")
                    backtester.check_data_quality()
            except Exception as e:
                print(e)
                print(f"Error running backtest for {symbol} with interval {interval}: {e}")
        
        # Sort interval results by total return
        sorted_results = sorted(interval_results, key=lambda x: x['total_return'], reverse=True)
        
        # Display top performers for this interval
        print(f"\nTop {len(sorted_results)} performers for interval {interval}:")
        print(f"{'Rank':<5} {'Symbol':<10} {'Total Return':<20} {'Win Rate':<15} {'Trades Executed':<15} {'Total Fees Paid':<20} {'NET Profit/Loss':<20} {'Trading Volume':<20} {'Sharpe Ratio':<15}")
        print("=" * 160)  # Separator line
        for i, result in enumerate(sorted_results[:-1], 1):
            print(f"{i:<5} {result['symbol']:<10} {result['total_return']:<20.2%} {result['win_rate']:<15.2%} {result['trades_executed']:<15} {result['total_fees_paid']:<20.2f} {result['net_profit_loss']:<20.2f} {result['trading_volume']:<20.2f} {result['sharpe_ratio']:<15.2f}")
        
        # Calculate and display average metrics for this interval
        avg_return = np.mean([r['total_return'] for r in interval_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in interval_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in interval_results])
        avg_winrate = np.mean([r['win_rate'] for r in interval_results])
        
        print(f"\nAverage metrics:")
        print(f"Avg Return: {avg_return:.2%}, Avg Sharpe: {avg_sharpe:.2f}, Avg Drawdown: {avg_drawdown:.2%}, Avg Win Rate: {avg_winrate:.2%}")
        print(f"Total trades: {interval_total_trades}, Total winning trades: {interval_winning_trades}")
        print(f"Overall win rate: {interval_winning_trades / interval_total_trades:.2%}" if interval_total_trades > 0 else "No trades executed")
        print(f"Total trading volume: {interval_trading_volume:.2f}")
        print('\n\n')
        all_results.extend(interval_results)
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('backtest_results.csv', index=False)
    print("\nDetailed results saved to 'backtest_results.csv'")

    # Ensure we stop the output handler when we're done
    stop_output_handler(output_handler_process)

