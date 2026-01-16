
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import pandas as pd
import seaborn as sns

# Evaluate the best model
def model_performance_regression(model, predictors, target):
    pred = model.predict(predictors)
    r2 = r2_score(target, pred)
    rmse = np.sqrt(mean_squared_error(target, pred))
    mae = mean_absolute_error(target, pred)
    mape = np.mean(np.abs(target - pred) / target) * 100

    # Creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "RMSE": rmse,
            "MAE": mae,
            "R-squared": r2,
            # "MAPE": mape,
        },
        index=[0],
    )
    return df_perf, pred



def prepare_data(df, columns_to_remove=['btc_price'], validation_hours=10, train_split=0.8):
    # Remove specified columns
    df_cleaned = df.drop(columns=columns_to_remove)

    X = df_cleaned.drop(columns=['pct_growth'])
    y = df_cleaned['pct_growth']

    # Define the cutoff time for the validation set
    cutoff_time = df_cleaned.index.max() - pd.Timedelta(hours=validation_hours)

    # Create the validation set for the last specified hours
    validation_data = df_cleaned[df_cleaned.index > cutoff_time]

    # Create DataFrames for training and testing from the remaining data
    remaining_data = df_cleaned[df_cleaned.index <= cutoff_time]
    train_size = int(len(remaining_data) * train_split)  # Use specified percentage of the remaining data for training
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Combine X and y for a complete view (optional)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Prepare the data (X for predictors and y for target)
    X_train = train_data.drop(columns='pct_growth') 
    y_train = train_data['pct_growth']              

    X_test = test_data.drop(columns='pct_growth')    
    y_test = test_data['pct_growth']
    
    return X_train, y_train, X_test, y_test, validation_data,test_data,X,y



def train_and_evaluate_catboost(test_data,X_train, y_train, X_test, y_test, categorical_features, 
                                 iterations=1000, l2_leaf_reg=4.0, depth=4, 
                                 learning_rate=0.02, bagging_temperature=0.5, 
                                 subsample=0.8, random_strength=1.0, 
                                 eval_metric='RMSE', early_stopping_rounds=50, 
                                 verbose=50, random_seed=42):
    # Initialize CatBoost Regressor with specified hyperparameters
    catboost_model = CatBoostRegressor(
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        depth=depth,
        learning_rate=learning_rate,
        bagging_temperature=bagging_temperature,
        subsample=subsample,
        random_strength=random_strength,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        random_seed=random_seed
    )

    # Fit the model on the training data
    catboost_model.fit(X_train, y_train, cat_features=categorical_features)

    # Evaluate the model on the test set
    model_results, predictions = model_performance_regression(catboost_model, X_test, y_test)
    print(model_results)

    # Visualizing Actual vs Predicted values as time series
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, y_test, color='red', label='Actual BTC Prices', linewidth=2)  # Actual values
    plt.plot(test_data.index, predictions, color='blue', label='Predicted BTC Prices', linewidth=2)  # Predicted values
    plt.xlabel('Time')
    plt.ylabel('BTC Prices')
    plt.title('Actual vs Predicted BTC Prices Over Time')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()

    return catboost_model, predictions




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df):
    """
    Plots a correlation matrix for the given DataFrame, coloring cells
    above 0.95 in red.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to plot the correlation matrix.
    """
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create a mask for values above 0.95
    mask = correlation_matrix > 0.95

    # Create a custom color map
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Create a heatmap with the correlation matrix
    ax = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, mask=mask,
                     square=True, cbar_kws={"shrink": .8}, linewidths=.5, linecolor='gray',
                     annot_kws={"size": 10, "color": "black"})  # Set text color to black

    # Overlay the red cells for high correlations
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            if mask.iloc[i, j]:
                # Instead of using add_patch, we can directly use text to ensure it overlays correctly
                ax.text(j + 0.5, i + 0.5, f'{correlation_matrix.iat[i, j]:.2f}', 
                        color='white', ha='center', va='center', fontsize=10, weight='bold',
                        bbox=dict(facecolor='red', alpha=0.5, boxstyle='round,pad=0.3'))

    # Set the title
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X_train_num, categorical_features):

    # Drop rows with missing or infinite values
    X_train_num_cleaned = X_train_num.dropna()
    X_train_num_cleaned = X_train_num_cleaned[~np.isinf(X_train_num_cleaned).any(axis=1)]

    # Calculate VIF on the cleaned DataFrame
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_train_num_cleaned.columns
    vif_data['VIF'] = [variance_inflation_factor(X_train_num_cleaned.values, i) for i in range(X_train_num_cleaned.shape[1])]

    return vif_data

# Example usage:
# vif_results = calculate_vif(X_train, categorical_features)