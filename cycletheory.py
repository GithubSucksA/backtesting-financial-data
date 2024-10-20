import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
from typing import List, Tuple
import pandas_ta as ta
import time
import os
from scipy.fft import fft

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

    def calculate_cycle_indicator(self, df: pd.DataFrame, column: str = 'close', window: int = 10) -> pd.Series:
        """
        Calculate the cycle indicator using Fourier Transform.
        """
        def get_dominant_cycle(data):
            # Filter out NaN values
            data = data.dropna()
            if len(data) < 2:  # Ensure there are enough data points
                return np.nan
            
            print(f"Data for FFT: {data}")  # Debugging: print the data being passed to FFT
            fft_result = fft(data)
            frequencies = np.fft.fftfreq(len(data))
            positive_freq_idx = np.where(frequencies > 0)[0]
            magnitudes = np.abs(fft_result[positive_freq_idx])
            dominant_idx = np.argmax(magnitudes)
            dominant_cycle = 1 / frequencies[positive_freq_idx[dominant_idx]]
            return int(dominant_cycle)

        # Apply rolling window and calculate cycle indicator
        cycle_indicator = df[column].rolling(window=window).apply(get_dominant_cycle, raw=False)
        print(f"Cycle Indicator: {cycle_indicator}")  # Debugging: print the resulting cycle indicator
        return cycle_indicator

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate existing indicators
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df = df.join(macd)
        bollinger = ta.bbands(df['close'], length=20, std=2)
        df = df.join(bollinger)
        
        # Check for NaN values in the 'close' column
        if df['close'].isna().any():
            print("Warning: NaN values found in 'close' column before calculating cycle indicator.")
        
        # Calculate cycle indicator
        df['Cycle'] = self.calculate_cycle_indicator(df)
        
        return df

    def calculate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Generate signals based on multiple indicators, including the cycle indicator
        buy_signal = (
            (df['close'] > df['MA20']) &
            (df['RSI'] < 70) &
            (df['MACD_12_26_9'] > df['MACDs_12_26_9']) &
            (df['close'] > df['BBL_20_2.0']) &
            (df['Cycle'].shift(1) < df['Cycle'])  # Buy when cycle length is increasing
        )
        sell_signal = (
            (df['close'] < df['MA20']) |
            (df['RSI'] > 70) |
            (df['MACD_12_26_9'] < df['MACDs_12_26_9']) |
            (df['close'] < df['BBL_20_2.0']) |
            (df['Cycle'].shift(1) > df['Cycle'])  # Sell when cycle length is decreasing
        )
        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)

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

        print(f"Starting backtest for {self.symbol} with initial balance: {balance}")

        for i in range(1, len(self.df)):
            current_price = self.df['close'].iloc[i]
            signal = self.signals.iloc[i]

            print(f"Iteration {i}: Current Price: {current_price}, Signal: {signal}, Position: {position}, Balance: {balance}")

            if signal == 1 and position == 0:
                # Risk management: Use ATR for position sizing
                cost = self.risk_per_trade * balance
                position_size = (cost * (1 - transaction_cost)) / current_price

                if cost <= balance:
                    position = position_size
                    balance -= cost
                    trades_executed += 1
                    self.trading_volume += cost
                    last_buy_price = current_price
                    fees_paid.append(cost * transaction_cost)
                else:
                    print("Insufficient balance for buy")

            elif signal == -1 and position > 0:
                # Sell
                sell_value = position * current_price * (1 - transaction_cost)
                balance += sell_value
                self.trading_volume += sell_value

                # Determine if it's a winning or losing trade
                if sell_value > cost:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                position = 0
                trades_executed += 1
                fees_paid.append(sell_value * transaction_cost)

            current_value = balance + position * current_price
            balances.append(current_value)
            returns.append((current_value - balances[i-1]) / balances[i-1] if balances[i-1] != 0 else 0)

        # Calculate total fees paid
        total_fees_paid = sum(fees_paid)

        self.trades_executed = trades_executed
        print(f"Backtest completed for {self.symbol}. Total trades executed: {trades_executed}, Final balance: {balance}")
        return balances, returns, position, last_buy_price, balances[-1] - self.initial_balance, total_fees_paid, self.trading_volume

    def plot_results(self, balances: List[float]):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        ax1.plot(self.df.index, self.df['close'], label='Close Price')
        ax1.plot(self.df.index, self.df['MA20'], label='20-day MA')
        ax1.plot(self.df.index, self.df['BBU_20_2.0'], label='Upper BB', linestyle='--')
        ax1.plot(self.df.index, self.df['BBL_20_2.0'], label='Lower BB', linestyle='--')
        buy_signals = self.df.index[self.signals == 1]
        sell_signals = self.df.index[self.signals == -1]
        ax1.scatter(buy_signals, self.df.loc[buy_signals, 'close'], marker='^', color='g', label='Buy')
        ax1.scatter(sell_signals, self.df.loc[sell_signals, 'close'], marker='v', color='r', label='Sell')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.set_title(f'{self.symbol} Price and Signals')

        ax2.plot(self.df.index, balances, label='Portfolio Value')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend()
        ax2.set_title('Portfolio Value Over Time')

        ax3.plot(self.df.index, self.df['Cycle'], label='Cycle Length')
        ax3.set_ylabel('Cycle Length')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.set_title('Cycle Indicator')

        plt.tight_layout()
        safe_symbol = self.symbol.replace('/', '_')
        os.makedirs('plots', exist_ok=True)
        filename = f'{safe_symbol}_{self.interval}_backtest_results.png'
        filepath = os.path.join('plots', filename)
        plt.savefig(filepath)
        plt.close(fig)

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
        balances, returns, final_position, last_buy_price, net_profit_loss, total_fees_paid, trading_volume = self.backtest()
        self.plot_results(balances)
        #safe_symbol = self.symbol.replace('/', '_')
        #filename = f'{safe_symbol}_{self.interval}_backtest_results.png'
        return self.print_performance_metrics(balances, returns, final_position, last_buy_price, net_profit_loss, total_fees_paid, trading_volume)
        #print(f"Plot saved as plots/{filename}")

    def print_performance_metrics(self, balances: List[float], returns: List[float], final_position: float, last_buy_price: float, net_profit_loss: float, total_fees_paid: float, trading_volume: float) -> dict:
        total_return = (balances[-1] - balances[0]) / balances[0] if balances[0] != 0 else 0

        print(f"Returns length: {len(returns)}")
        print(f"Winning trades: {self.winning_trades}, Total trades executed: {self.trades_executed}")
        
        # Check if returns is not empty before calculating std and mean
        if len(returns) > 0:
            std_returns = np.std(returns)
            sharpe_ratio = np.mean(returns) / std_returns * np.sqrt(365) if std_returns != 0 else 0
        else:
            std_returns = 0
            sharpe_ratio = 0
        
        max_balance = np.max(balances)
        max_drawdown = np.max(np.maximum.accumulate(balances) - balances) / max_balance if max_balance != 0 else 0
        win_rate = (self.winning_trades / self.trades_executed) if self.trades_executed > 0 else 0  # Avoid division by zero

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


# Usage
if __name__ == "__main__":
    #start_time = time.time()
    #print(f"START TIME {start_time}")
    symbols = ["AAVEUSD", "BTC/USD", "ETH/USD", "ADAUSDT", "ALGOUSDT", "ATOMUSDT", "AVAXUSDT", "DOTUSDT", "CRVUSD", "EGLDUSD", "ENJUSD", "EWTUSD", "FETUSD", "FILUSD", "FLOKIUSD", "FLOWUSD", "GALAUSD", "GMXUSD", "ICPUSD", "INJUSD", "LINKUSDT", "LTCUSDT", "MANAUSDT", "LRCUSD", "MATICUSDT", "MINAUSD", "MKRUSD", "NEARUSD", "OCEANUSD", "PENDLEUSD", "PEPEUSD", "QNTUSD", "PYTHUSD", "RENDERUSD", "SANDUSD", "SHIBUSD", "SOLUSDT", "TAOUSD", "TRXUSD", "UNIUSD", "WIFUSD", "XDGUSD"] 
    #intervals = [15, 30, 60, 240, 1440, 10080, 21600]
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
                    #print('\n\n')
                    date_range_printed = True

                result = backtester.run_backtest()
                interval_results.append(result)

                interval_total_trades += result['trades_executed']
                interval_winning_trades += result['winning_trades']
                interval_trading_volume += result['trading_volume']
            
                if result['total_return'] == 0 and result['trades_executed'] == 0:
                    print(f"Warning: No trades executed for {symbol} with interval {interval}")
                    backtester.check_data_quality()
                        # Print open position information
                if result['open_position'] > 0:
                    pass
                    #print(f"Open position for {symbol}: {result['open_position']:.6f}")
                    #print(f"Open position value: ${result['open_position_value']:.2f}")
                    #print(f"Open position return: {result['open_position_return']:.2%}")
                #print('\n')
            except Exception as e:
                print(f"Error running backtest for {symbol} with interval {interval}: {e}")
        
        # Sort interval results by total return
        sorted_results = sorted(interval_results, key=lambda x: x['total_return'], reverse=True)
        
        # Display top 30 performers for this interval
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
    
    #end_time = time.time()
    #print(f"\nEND TIME {end_time}")
    #total_duration = end_time - start_time
    #print(f"TOTAL TIME {total_duration:.2f} seconds")

