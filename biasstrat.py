from base_trading_backtester import BaseTradingBacktester
import pandas as pd
import pandas_ta as ta
import numpy as np

class BiasBacktester(BaseTradingBacktester):

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate only SMA, RSI, AO, and BIAS
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['AO'] = ta.ao(df['high'], df['low'])  # Add AO calculation
        
        # Calculate BIAS
        df['BIAS'] = ta.bias(df['close'], length=26).round(2)  # You can adjust the length as needed

        return df

    def calculate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Buy when BIAS crosses above 0 (indicating bullish momentum)
        buy_signal = df['BIAS'] > 0

        # Sell when BIAS crosses below 0 (indicating bearish momentum)
        sell_signal = df['BIAS'] < 0

        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
