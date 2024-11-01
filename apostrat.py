from base_trading_backtester import BaseTradingBacktester
import pandas as pd
import pandas_ta as ta
import numpy as np

class ApoBacktester(BaseTradingBacktester):

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate only SMA, RSI, and AO
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['AO'] = ta.ao(df['high'], df['low'])  # Add AO calculation
        # Calculate APO using Exponential Moving Averages
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()

        # Absolute Price Oscillator (APO) = EMA12 - EMA26
        df['APO'] = df['EMA12'] - df['EMA26']

        # Define a moving average for APO to smooth it, say 9-period APO moving average
        df['APO_MA'] = df['APO'].rolling(window=9).mean()
        return df

    def calculate_signals(self, df: pd.DataFrame, window: int = 5, buy_signal_config=None, sell_signal_config=None) -> pd.Series:
        # Buy when APO crosses above its moving average, indicating bullish momentum
        buy_signal = (
            (df['APO'].shift(1) < df['APO_MA'].shift(1)) & (df['APO'] > df['APO_MA'])
        )

        # Sell when APO crosses below its moving average, indicating bearish momentum
        sell_signal = (
            (df['APO'].shift(1) > df['APO_MA'].shift(1)) & (df['APO'] < df['APO_MA'])
        )
        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)