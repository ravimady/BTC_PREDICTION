import sys
import os
import yfinance as yf

# Set the base directory where the custom module is located
base_dir = '/workspaces/BTC_PREDICTION'  # Adjust this path as needed

# Add the parent directory of the current working directory to sys.path
sys.path.append(os.path.join(base_dir, 'dev'))

# Import the fetch_btc_history function
from custom.data_pull import fetch_btc_history

# Change to the validation directory
os.chdir(os.path.join(base_dir, 'dev', 'validation'))

# Fetch BTC history
btc_history = fetch_btc_history()

import pandas as pd

# Convert the list to a DataFrame
btc_df = pd.DataFrame(btc_history, columns=['timestamp', 'price'])

# Convert timestamp to a readable date
btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')


os.chdir(os.path.join(base_dir, 'dev'))

# Save to CSV
btc_df.to_csv("./artifacts/btc_validation.csv", index=False)

# Optionally change back to the original directory if needed
os.chdir(os.path.join(base_dir, 'dev'))  # Change back to the dev directory if necessary