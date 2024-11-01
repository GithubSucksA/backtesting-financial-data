from base_trading_backtester import BaseTradingBacktester
import pandas as pd
import pandas_ta as ta
import numpy as np

class CciBacktester(BaseTradingBacktester):

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate only SMA, RSI, AO, and CCI
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['AO'] = ta.ao(df['high'], df['low'])  # Add AO calculation
        
        # Calculate CCI
        df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=14)  # You can adjust the length as needed

        return df

    def calculate_signals(self, df: pd.DataFrame, window: int = 5, buy_signal_config=None, sell_signal_config=None) -> pd.Series:
        # Buy when CCI crosses above -100 (indicating potential bullish momentum)
        buy_signal = df['CCI'] < -100

        # Sell when CCI crosses below 100 (indicating potential bearish momentum)
        sell_signal = df['CCI'] > 100

        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)