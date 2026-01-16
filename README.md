# BTC PREDICTION

## Objective:

Make an accurate weekly prediction on Bitcoin Euro. Crypto market is going to be the next big thing in the next 5-10 years. This personal project is started up to understand the factors that impact the crypto market. Bitcoin is taken as an indicator of the crypto market.

## Folder Structure :


- dev
  - artifacts
  - preprocessing.py
    - custom
        - data_pull.py
  - training(_v2).ipynb
    - custom
        - training_functions.py
        - modelling_functions.py

## Preprocessing 

This script does 3 things for the last 300 days
1. Fetch BTC history in Euro
2. Pulls the S&P 500 index from YahooFinance.
3. pulls the dollar value .
4. pulls gold price in euro

## Training


a. Loading data from input files.

b. preprocessing steps -- refer the (training_functions.py)

c. Modelling.

## Training version 2


a. Loading data from input files.

b. preprocessing steps -- refer the (training_functions.py) --  here, the stock prices, indices and gold prices were removed.

c. Modelling.


