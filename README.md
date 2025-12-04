# BTC EURO Forecasting Project


<img src="https://github.com/user-attachments/assets/f692ec88-6d26-40ca-8880-54c7e2ead062" alt="btc_pic" width="1400"> 


## **Objective:**
This project is a personal project to develop a Machine Learning process in continuously understanding the factors that influence the Bitcoin prices and be able to accurately predict the movement in Bitcoin and later even other coins. 


## **Approach:**
To start with, I had identified a list of features that I thought may be most influential in its movements like gold price, stock indices and dollar value. For this project, I had preprocessed about 300 days of historcal data (flexible to change as it was done through an API call). Since it is a time series data it makes sense to also derive some features such as day, day of the week (mon-fri),day of the month etc. Further, I had also created other features like price_lags, meaning how does yesterdays price influence days price.


## **Feature Engineering:**
The features vary significantly hence there was a need to standardise or normalise the scale. For this project, after trail and error I concluded that day to day percent growth was the best way to normalise the data.

## **Modelling:**
Models evaluated :
1. ARIMA models for EDA.
2. Random Forest Regressor.
3. Catboost Regressor.

## **Results:**

<img width="1400" height="513" alt="btc_results" src="https://github.com/user-attachments/assets/ea56c355-1021-4142-9dda-3cf96439da7a" />

## **Feature Importance:**

<img width="1400" height="800" alt="shap_values" src="https://github.com/user-attachments/assets/5f7be09b-4558-4a5e-b828-fc497a94f072" />
