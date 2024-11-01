# Define a mapping for symbol tickers
symbol_mapping = {
    'kraken': {
        'btcusdt': 'BTC/USD',
        'ethusdt': 'ETH/USD',
    },
    'htx': {
        'BTC/USD': 'btcusdt',
        'ETH/USD': 'ethusdt',
    }
}

# Define a mapping for user-friendly interval inputs
interval_mapping = {
    'kraken': {
        '15': 15,
        '15min': 15,
        '30': 30,
        '30min': 30,
        '60': 60,
        '60min': 60,
        '240': 240,
        '4hour': 240,
        '1440': 1440,
        '1day': 1440,
    },
    'htx': {
        '15': '15min',
        '15min': '15min',
        '30': '30min',
        '30min': '30min',
        '60': '60min',
        '60min': '60min',
        '240': '4hour',
        '4hour': '4hour',
        '1440': '1day',
        '1day': '1day',
    }
}