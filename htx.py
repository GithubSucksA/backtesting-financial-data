import requests

url = 'https://api.huobi.pro/market/tickers'

response = requests.get(url)
data = response.json()

# Update to extract and sort the trading pair names alphabetically
trading_pairs = sorted([item['symbol'] for item in data['data']])

print('Trading Pairs:', end='\n\n')
print(trading_pairs)

# Optionally, print the number of trading pairs
print(f"\nTotal number of trading pairs: {len(trading_pairs)}")
for pair in trading_pairs:
    print(f"{pair}")

from operator import itemgetter

'''
def get_trading_pairs():
    url = "https://api.kraken.com/0/public/AssetPairs"
    response = requests.get(url)
    data = response.json()
    return list(data['result'].keys())

def get_ticker_info(pairs):
    url = "https://api.kraken.com/0/public/Ticker"
    params = {'pair': ','.join(pairs)}
    response = requests.get(url, params=params)
    return response.json()['result']

def main():
    trading_pairs = get_trading_pairs()
    ticker_info = get_ticker_info(trading_pairs)

    # Create a list of tuples (pair, volume) and sort by volume
    pair_volumes = [(pair, float(info['v'][1])) for pair, info in ticker_info.items()]
    sorted_pairs = sorted(pair_volumes, key=itemgetter(1), reverse=True)

    print('Trading Pairs Ranked by 24h Volume:', end='\n\n')
    for pair, volume in sorted_pairs:
        print(f"{pair}: {volume:.2f}")

    print(f"\nTotal number of trading pairs: {len(sorted_pairs)}")

if __name__ == "__main__":
    main()
'''