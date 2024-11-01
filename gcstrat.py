from base_trading_backtester import BaseTradingBacktester
import pandas as pd
import pandas_ta as ta
import numpy as np

class GcBacktester(BaseTradingBacktester):

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate only SMA, RSI, AO, CCI, CFO, and CoG
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['AO'] = ta.ao(df['high'], df['low'])  # Add AO calculation
        
        # Calculate CCI
        df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=14)  # You can adjust the length as needed
        
        # Calculate Chande Forecast Oscillator
        df['CFO'] = ta.cfo(df['close'])  # You can adjust the length as needed
        
        # Calculate Center of Gravity
        df['CoG'] = ta.cg(df['close'])  # You can adjust the length as needed

        return df

    def calculate_signals(self, df: pd.DataFrame, window: int = 5, buy_signal_config=None, sell_signal_config=None) -> pd.Series:
        # Buy when CFO is below 0 but starting to skew upwards and CoG is below the price
        buy_signal = (df['CoG'] < -0.03)  # Use & instead of and

        # Sell when CFO crosses above 0 (indicating potential bearish momentum) and CoG is above the price
        sell_signal = (df['CoG'] > 0.03)  # Use & instead of and

        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)