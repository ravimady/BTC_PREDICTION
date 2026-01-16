import sys
import os

# Set the base directory where the custom module is located
base_dir = '/workspaces/BTC_PREDICTION'  # Adjust this path as needed

# Add the parent directory of the current working directory to sys.path
sys.path.append(os.path.join(base_dir, 'dev'))


from custom.data_pull import (fetch_btc_history,
                              stock_extract,
                              dollar_value_extract,
                              gold_price_extract)

btc_history = fetch_btc_history()
stock_extract = stock_extract()
dollar_value_extract = dollar_value_extract()
gold_prices_euro = gold_price_extract()
import pandas as pd
# Convert the list to a DataFrame
btc_df = pd.DataFrame(btc_history, columns=['timestamp', 'price'])

# Convert timestamp to a readable date
btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')

os.chdir(os.path.join(base_dir, 'dev'))

# Save to CSV
btc_df.to_csv("./artifacts/input.csv", index=False)
stock_extract.to_csv("./artifacts/stock_index_data.csv")
dollar_value_extract.to_csv("./artifacts/dollar_fluctuations.csv")
gold_prices_euro.to_csv("./artifacts/gold_prices_euro.csv")
#cd /workspaces/BTC_PREDICTION/dev
