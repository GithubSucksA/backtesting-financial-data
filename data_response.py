import requests
import json
import datetime

# PRICE DATA GENERAL
url = "https://api.kraken.com/0/public/Ticker?pair=AAVEUSD"

payload = {}
headers = {
  'Accept': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)
full_response = requests.get(url)
data = response.json()
print('DATA')
print(data)

# RESPONSE DATA
print('')
print('RESPONSE.TEXT', end='\n\n')
print(response.text)
print('\n\n')
print('FULL RESPONSE METHODS', end='\n\n')
print(dir(response))
print('\n\n')
print('JSON DATA', end='\n\n')
print(json.dumps(data, indent=4))
print('\n\n')
print('FULL RESPONSE', end='\n\n')
print(vars(full_response))

# PRICE DATA
print('TODAYS OPENING PRICE')
print(data['result']['AAVEUSD']['o'], end='\n\n')
print('TODAYS LOW PRICE')
print(data['result']['AAVEUSD']['l'][0], end='\n\n')
print('TODAYS HIGH PRICE')
print(data['result']['AAVEUSD']['h'][0], end='\n\n')
print('CURRENT BUY PRICE: ' + data['result']['AAVEUSD']['b'][0])
print('AMOUNT OF COINS AVAILABLE TO BUY AT THIS PRICE: ' + data['result']['AAVEUSD']['b'][2], end='\n\n')
print('CURRENT SELL PRICE: ' + data['result']['AAVEUSD']['a'][0])
print('AMOUNT OF COINS AVAILABLE TO SELL AT THIS PRICE: ' + data['result']['AAVEUSD']['a'][2], end='\n\n')

print('----------------------------------')
print('\n\n')
print('----------------------------------')




# OHLC DATA // BACKTESTING
url = "https://api.kraken.com/0/public/OHLC?pair=AAVEUSD&interval=1440"

payload = {}
headers = {
  'Accept': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)
data = response.json()

# Function to convert timestamp to formatted string
def format_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

# Update the timestamps in the data
for i in range(len(data['result']['AAVEUSD'])):
    timestamp = data['result']['AAVEUSD'][i][0]
    formatted_date = format_timestamp(timestamp)
    data['result']['AAVEUSD'][i][0] = formatted_date

# Print each item in the list
for item in data['result']['AAVEUSD']:
    print(item)

print('JSON DATA', end='\n\n')
print(json.dumps(data, indent=4))
