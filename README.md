# BTC EURO Forecasting Project


<img src="https://github.com/user-attachments/assets/f692ec88-6d26-40ca-8880-54c7e2ead062" alt="btc_pic" width="1400"> 


## **Objective:**
This project is a personal project to develop a Machine Learning process that understands the factors that influence the Bitcoin prices and be able to accurately predict the movement in Bitcoin and later even other coins. 


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

## **Hypothesis testing**

Test for Heteroskadasticity

<img width="393" height="154" alt="image" src="https://github.com/user-attachments/assets/90d7dc56-c3cc-4d42-80c4-934b84adbec2" />

## Evaluation :


## **Lagrange Multiplier Statistic:**

This statistic tests the null hypothesis that the residuals from the regression model are homoscedastic (i.e., have constant variance). 

In your case, the Lagrange Multiplier statistic is approximately 12.10. Though not conclusive, this high number could indicate that absence of Heteroskedasticity.

## **p-value**

High p-value of 0.2 is also a strong indicator that the above experiment can reject the null hypothesis. 

<img width="1400" height="644" alt="image" src="https://github.com/user-attachments/assets/fcc82bac-e74b-410c-ab9f-2497ec41bf49" />


The above graphs are a sign that the model built is in good order and does not exhibit heteroskadasticity.
1. We see that first graph is random and does not follow any type of pattern.
2. The graph is normally distributed.
3. The Q-Q plot showing the plots closely following the red line is also a good sign.


