from base_trading_backtester import BaseTradingBacktester
import pandas as pd
import pandas_ta as ta
import numpy as np

class AoBacktester(BaseTradingBacktester):

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate only SMA, RSI, and AO
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['AO'] = ta.ao(df['high'], df['low'])  # Add AO calculation
        return df

    def calculate_signals(self, df: pd.DataFrame, window: int = 5, buy_signal_config=None, sell_signal_config=None) -> pd.Series:
        # Calculate the moving average of AO
        df['AO_MA'] = df['AO'].rolling(window=10).mean()

        # Default buy and sell signal logic
        buy_signal = (df['AO'] < 0) & (df['AO'] > df['AO_MA'])  # Default Buy Signal
        sell_signal = (df['AO'] > 0) & (df['AO'] < df['AO_MA'])  # Default Sell Signal

        # Apply custom buy signal logic if provided
        if buy_signal_config:
            if buy_signal_config == 'signal1':
                buy_signal = (df['AO'] < -0.5)  # Arbitrary condition for testing
            elif buy_signal_config == 'signal2':
                buy_signal = (df['AO'] > 0.5)  # Arbitrary condition for testing
            elif buy_signal_config == 'signal3':
                buy_signal = (df['AO'] < 0) & (df['RSI'] < 30)  # Example of another buy signal

        # Apply custom sell signal logic if provided
        if sell_signal_config:
            if sell_signal_config == 'signal1':
                sell_signal = (df['AO'] > 0) & (df['AO'] < df['AO_MA'])
            elif sell_signal_config == 'signal2':
                sell_signal = (df['AO'] > 0) & (df['AO'].diff() < 0)  # Example of a different sell signal
            elif sell_signal_config == 'signal3':
                sell_signal = (df['AO'] > 0) & (df['RSI'] > 70)  # Example of another sell signal

        # Debugging output
        print(f"Buy Signal ({buy_signal_config}): {buy_signal.sum()} trades")
        print(f"Sell Signal ({sell_signal_config}): {sell_signal.sum()} trades")

        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)