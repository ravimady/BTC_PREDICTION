import pandas as pd

def process_timestamp(df):

    df['timestamp'] = df['timestamp'].str.split('+').str[0]
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('H')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['timestamp'] = df['timestamp'].str.strip()
    
    return df


def stock_preprocess(df):

    df = df.drop(index=0) 
    df = df.drop(columns=df.columns[0])

    return df



def daily_prices(df):

    # Convert the 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set the 'timestamp' as the index
    df.set_index('timestamp', inplace=True)
    df = df.astype(float).astype(int)

    # Group by date and aggregate btc_price 
    df = df.resample('D').mean() 

    # Resetting the index to have 'timestamp' as a column again
    df.reset_index(inplace=True)

    return df


def gold_preprocess(df):

    df = df.drop(index=0).reset_index(drop=True) 
    df = df.drop(columns=df.columns[0])
    # Specify columns to remove
    columns_to_remove = [ 'High','Low','Open','Volume']

    # Remove specified columns
    df = df.drop(columns=columns_to_remove)

    return df


import pandas as pd

def additional_features(merged_df_btc_index):


    # Create DataFrame from the input data
    df = pd.DataFrame(merged_df_btc_index)

    # Set 'timestamp' as index
    df.set_index('timestamp', inplace=True)

    # Create lagged features
    for i in range(1, 6):  # Create 5 lagged features
        df[f'price_lag{i}'] = df['btc_price'].shift(i)

    # Create rolling features
    df['rolling_mean_3'] = df['btc_price'].rolling(window=3).mean()


    # Drop NaN values created by lagging and rolling
    df.dropna(inplace=True)

    # Add time-based features
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['month'] = df.index.month
    df['day'] = df.index.day
    # day to day percent growth
    df['pct_growth'] = df['btc_price'].pct_change() * 100  # Calculate percentage change
    # df['pct_stock_index_close'] = df['stock_index_close'].pct_change() * 100  # Calculate percentage change
    # df['pct_stock_price_high'] = df['stock_price_high'].pct_change() * 100  # Calculate percentage change
    # df['pct_stock_price_low'] = df['stock_price_low'].pct_change() * 100  # Calculate percentage change
    # df['pct_stock_index_open'] = df['stock_index_open'].pct_change() * 100  # Calculate percentage change
    # df['pct_stock_price_high'] = df['pct_stock_price_high'].pct_change() * 100  # Calculate percentage change
    # df['pct_close_gold_price'] = df['close_gold_price'].pct_change() * 100  # Calculate percentage change
    df['pct_price_lag1'] = df['price_lag1'].pct_change() * 100  # Calculate percentage change
    df['pct_price_lag2'] = df['price_lag2'].pct_change() * 100  # Calculate percentage change
    df['pct_price_lag3'] = df['price_lag3'].pct_change() * 100  # Calculate percentage change
    df['pct_price_lag4'] = df['price_lag4'].pct_change() * 100  # Calculate percentage change
    df['pct_price_lag5'] = df['price_lag5'].pct_change() * 100  # Calculate percentage change

    df['pct_rolling_mean_3'] = df['rolling_mean_3'].pct_change() * 100  # Calculate percentage change
    # columns_to_remove = [ 'stock_index_close', 'stock_price_high', 'stock_price_low',
    #    'stock_index_open', 'Volume', 'close_gold_price', 'price_lag1',
    #    'price_lag2', 'price_lag3', 'price_lag4', 'price_lag5',
    #    'rolling_mean_3']
    columns_to_remove = [ 
         'price_lag1',
       'price_lag2', 'price_lag3', 'price_lag4', 'price_lag5',
       'rolling_mean_3']

    # Remove specified columns
    df = df.drop(columns=columns_to_remove)


    df['pct_growth'].fillna(0, inplace=True)  # Fill NaN values with 0

    return df