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

    def calculate_signals(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        # Calculate the moving average of AO
        df['AO_MA'] = df['AO'].rolling(window=10).mean()
        #ao_rate_of_change = df['AO'].diff()

        # Generate signals based on AO and its moving average
        # Buy Signal: It adds the condition that AO was below its moving average in the previous period (shift(1)), but now crosses above it. This avoids false signals in flat markets.
        #buy_signal = (
        #    (df['AO'] < 0) & (df['AO'].shift(1) < df['AO_MA'].shift(1)) & (df['AO'] > df['AO_MA'])
        #)
        buy_signal = (
            (df['AO'] < 0) & (df['AO'] > df['AO_MA'])  # Buy when AO is negative and crosses above its moving average
            #(df['AO'] < 0) & (ao_rate_of_change > 0)  # Buy when AO is negative and its rate of change is positive
        )
        #sell_signal = (
        #    (df['AO'] > 0) & (df['AO'].shift(1) > df['AO_MA'].shift(1)) & (df['AO'] < df['AO_MA'])
        #)
        sell_signal = (
            (df['AO'] > 0) & (df['AO'] < df['AO_MA'])  # Sell when AO is positive and crosses below its moving average
            #(df['AO'] > 0) & (ao_rate_of_change < 0)  # Sell when AO is positive and its rate of change is negative
        )
        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)