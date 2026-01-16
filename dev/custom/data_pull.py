import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pytz

def fetch_btc_history():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    parameters = {
        'vs_currency': 'eur',
        'days': '300'  # Adjust the number of days as needed
    }
    response = requests.get(url, params=parameters)
    data = response.json()
    return data['prices']



def stock_extract():
    # Define the stock index symbol (e.g., Euro Stoxx 50)
    index_symbol = '^STOXX50E'  # Euro Stoxx 50 Index

    # Define the time period
    end_date = datetime.now(pytz.timezone('Europe/Amsterdam'))
    start_date = end_date - timedelta(days=300)

    # Fetch historical data with hourly interval
    stock_data = yf.download(index_symbol, start=start_date, end=end_date, interval='1h', auto_adjust=True)

    # Check if the index is timezone aware and convert if necessary
    if stock_data.index.tz is None:
        stock_data.index = stock_data.index.tz_localize('UTC')

    # Convert the index to the specified timezone
    stock_data.index = stock_data.index.tz_convert('Europe/Amsterdam')

    # Reset index to have 'Date' as a column
    stock_data.reset_index(inplace=True)

    return stock_data


import yfinance as yf
from datetime import datetime, timedelta
import pytz

def gold_price_extract():
    # Define the gold symbol (e.g., Gold Futures)
    gold_symbol = 'SI=F'  # Gold Futures

    # Define the time period
    end_date = datetime.now(pytz.timezone('Europe/Amsterdam'))
    start_date = end_date - timedelta(days=300)

    # Fetch historical data with hourly interval
    gold_data = yf.download(gold_symbol, start=start_date, end=end_date, interval='1h', auto_adjust=True)

    # Check if the index is timezone aware and convert if necessary
    if gold_data.index.tz is None:
        gold_data.index = gold_data.index.tz_localize('UTC')

    # Convert the index to the specified timezone
    gold_data.index = gold_data.index.tz_convert('Europe/Amsterdam')

    # Reset index to have 'Date' as a column
    gold_data.reset_index(inplace=True)

    return gold_data






def dollar_value_extract():
    # Define the Dollar Index symbol
    index_symbol = 'DX-Y.NYB'  # U.S. Dollar Index

    # Define the time period
    end_date = datetime.now(pytz.timezone('Europe/Amsterdam'))
    start_date = end_date - timedelta(days=300)

    # Fetch historical data with hourly interval
    dollar_data = yf.download(index_symbol, start=start_date, end=end_date, interval='1h', auto_adjust=True)

    # Check if the index is timezone aware and convert if necessary
    if dollar_data.index.tz is None:
        dollar_data.index = dollar_data.index.tz_localize('UTC')

    # Convert the index to the specified timezone
    dollar_data.index = dollar_data.index.tz_convert('Europe/Amsterdam')

    # Reset index to have 'Date' as a column
    dollar_data.reset_index(inplace=True)

    return dollar_data


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
def adj_r2_score(predictors, targets, predictions):
    r2 = r2_score(targets, predictions)
    n = predictors.shape[0]
    k = predictors.shape[1]
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


# Function to compute MAPE
def mape_score(targets, predictions):
    return np.mean(np.abs(targets - predictions) / targets) * 100


def model_performance_regression(model, predictors, target):
    """
    Function to compute different metrics to check regression model performance

    model: regressor
    predictors: independent variables
    target: dependent variable
    """

    pred = model.predict(predictors)                  # Predict using the independent variables
    r2 = r2_score(target, pred)                       # To compute R-squared
    adjr2 = adj_r2_score(predictors, target, pred)    # To compute adjusted R-squared
    rmse = np.sqrt(mean_squared_error(target, pred))  # To compute RMSE
    mae = mean_absolute_error(target, pred)           # To compute MAE
    mape = mape_score(target, pred)                   # To compute MAPE

    # Creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "RMSE": rmse,
            "MAE": mae,
            "R-squared": r2,
            "Adj. R-squared": adjr2,
            "MAPE": mape,
        },
        index=[0],
    )

    return df_perf