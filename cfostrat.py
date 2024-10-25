from base_trading_backtester import BaseTradingBacktester
import pandas as pd
import pandas_ta as ta
import numpy as np

class CfoBacktester(BaseTradingBacktester):

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate only SMA, RSI, AO, CCI, and CFO
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['AO'] = ta.ao(df['high'], df['low'])  # Add AO calculation
        
        # Calculate CCI
        df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=14)  # You can adjust the length as needed
        
        # Calculate Chande Forecast Oscillator
        df['CFO'] = ta.cfo(df['close'], length=14)  # You can adjust the length as needed

        return df

    def calculate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Buy when CFO is below 0 but starting to skew upwards
        buy_signal = (df['CFO'] < 0) & (df['CFO'] > df['CFO'].shift(1))

        # Sell when CFO crosses above 0 (indicating potential bearish momentum)
        sell_signal = (df['CFO'] > 0) & (df['CFO'] < df['CFO'].shift(1))

        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)