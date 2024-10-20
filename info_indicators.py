import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    print("Error: pandas_ta is not installed. Please install it using 'pip install pandas_ta'")
    print("If you've already installed it, make sure you're using the correct Python environment.")
    import sys
    sys.exit(1)

# Help about this, 'ta', extension
df = pd.DataFrame()
print('INFORMATION ABOUT pandas_ta & ALL SIGNALS', end='\n\n\n')
help(df.ta)
print('\n\n\n')
print('\n\n\n')


# List of all indicators
print('LIST OF ALL INDICATORS', end='\n\n\n')
df.ta.indicators()
print('\n\n\n')
print('\n\n\n')

# Help about an indicator such as bbands
print('HELP ABOUT AN INDICATOR SUCH AS BBANDS')
help(ta.bbands)
print('\n\n\n')
print('HELP ABOUT AN INDICATOR SUCH AS ABERRATION')
help(ta.aberration)
print('\n\n\n')
print('HELP ABOUT AN INDICATOR SUCH AS KC')
help(ta.kc)

import pandas_ta as ta
print('BELOW HERE IS THE UTILS MODULE')
print(dir(ta.utils))  # List all available functions in the utils module
print(ta.__file__) # shows the file path of the pandas_ta library
