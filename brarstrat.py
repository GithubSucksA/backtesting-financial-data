from base_trading_backtester import BaseTradingBacktester
import pandas as pd
import pandas_ta as ta
import numpy as np

class BrarBacktester(BaseTradingBacktester):

    def calculate_crypto_brar(self, df: pd.DataFrame, period: int = 26) -> pd.DataFrame:
        """
        Calculate crypto-adapted BRAR with additional debugging
        """
        # Calculate rolling averages
        df['rolling_4h'] = df['close'].rolling(window=4).mean()
        df['rolling_24h'] = df['close'].rolling(window=24).mean()
        
        # Calculate VWAP
        df['vwap'] = df['vwap']  # Using the provided VWAP from your data
        
        # Calculate modified AR components
        buying_power = df['high'] - df['rolling_24h']
        buying_power_sum = buying_power.rolling(window=period).sum()
        selling_power = df['rolling_24h'] - df['low']
        selling_power_sum = selling_power.rolling(window=period).sum()
        
        # Avoid division by zero
        df['AR'] = np.where(
            selling_power_sum != 0,
            (buying_power_sum / selling_power_sum * 100).round(2),
            0
        )
        
        # Calculate modified BR components
        buying_power_br = df['high'] - df['vwap'].shift(1)
        buying_power_br_sum = buying_power_br.rolling(window=period).sum()
        selling_power_br = df['vwap'].shift(1) - df['low']
        selling_power_br_sum = selling_power_br.rolling(window=period).sum()
        
        # Avoid division by zero
        df['BR'] = np.where(
            selling_power_br_sum != 0,
            (buying_power_br_sum / selling_power_br_sum * 100).round(2),
            0
        )
    
        # Debug output
        #print("\nCrypto BRAR Calculation Debug:")
        #print(f"Number of valid AR values: {df['AR'].notna().sum()}")
        #print(f"Number of valid BR values: {df['BR'].notna().sum()}")
        
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['AO'] = ta.ao(df['high'], df['low'])
        df['BIAS'] = ta.bias(df['close'], length=26).round(2)
        df['BOP'] = ta.bop(df['open'], df['high'], df['low'], df['close'])
        
        # Calculate crypto-adapted BRAR
        df = self.calculate_crypto_brar(df)
        
        return df

    def calculate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trading signals with more lenient conditions and debug output
        """
        # More lenient signal conditions
        buy_signal = (
            (df['BR'] > 90) &  # Reduced from 100
            (df['AR'] > 70) &  # Reduced from 80
            (df['BR'].shift(1) <= 90)  # Reduced threshold
        )

        sell_signal = (
            (df['BR'] < 70) &  # Increased from 80
            (df['AR'] < 60) &  # Increased from 70
            (df['BR'].shift(1) >= 70)
        )

        # Add debug output
        #print("\nDebug Information:")
        #print(f"BR Range: {df['BR'].min():.2f} to {df['BR'].max():.2f}")
        #print(f"AR Range: {df['AR'].min():.2f} to {df['AR'].max():.2f}")
        #print(f"Number of potential buy signals: {buy_signal.sum()}")
        #print(f"Number of potential sell signals: {sell_signal.sum()}")
        
        # Sample of values where BR and AR are both present
        #print("\nSample of indicator values:")
        #sample_df = df[['BR', 'AR']].tail(10)
        #print(sample_df)

        return pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)