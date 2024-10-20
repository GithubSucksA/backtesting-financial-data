MyStrategy = ta.Strategy(
    name="DCSMA10",
    ta=[
        {"kind": "ohlc4"},
        {"kind": "sma", "length": 10},
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "length": 12, "fast": 26, "slow": 9},
        {"kind": "bbands", "length": 20},
        {"kind": "kc", "length": 20},
        {"kind": "stc", "length": 20},
        {"kind": "stoch", "length": 20},
        {"kind": "stochf", "length": 20},
        {"kind": "ema", "close": "OHLC4", "length": 10, "suffix": "OHLC4"},
    ]
)

df.ta.strategy(MyStrategy)
