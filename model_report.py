# Import Libraries

import pandas as pd
import numpy as np

import pandas_datareader.data as web

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb


###### Download Data ######

# Download Stock Data

start = "2010-01-01"
end = "2025-01-01"

df = web.DataReader("^SPX", "stooq", start = start, end = end).sort_index().reset_index()

# Downlaod Interest Rate and GDP

effr = web.DataReader('FEDFUNDS', 'fred', start = start, end = end)
gdp = web.DataReader('GDP', 'fred', start = start, end = end)


# Merge Datasets

df = df.merge(effr, left_on = 'Date', right_index = True, how = 'outer')
df = df.merge(gdp, left_on = 'Date', right_index = True, how = 'outer')


# Handle missing values and rename columns

df['FEDFUNDS'] = df['FEDFUNDS'].ffill()
df['GDP'] = df['GDP'].ffill()

df = df.dropna()

df = df.rename(columns = {'FEDFUNDS': 'Interest Rate'})


###### Engeneer Features ######

# Moving Average Indicators

df['MA_20'] = df['Close'].rolling(window = 20).mean()
df['MA_50'] = df['Close'].rolling(window = 50).mean()

# Percentage Change/Return

df['Close_pct_change'] = df['Close'].pct_change()
df['Volume_pct_change'] = df['Volume'].pct_change()

# RSI

window = 14

delta = df['Close'].diff()

gain = delta.clip(lower = 0)
loss = -delta.clip(upper = 0)

avg_gain = gain.rolling(window = window).mean()
avg_loss = loss.rolling(window = window).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))



####### Plot the Data #######

# Correlation Heatmap
# We want to understand better the relationships between the features
# To decide which features to use for our models

plt.figure(figsize = (10, 8))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Based on the correlation heatmap
# We decided to drop the following features:
# High, Low, MA_20, MA_50

df = df.drop(columns = ['High', 'Low', 'MA_20', 'MA_50'])
print(df.head())

df = df.dropna()


###### Prepare Data for Modeling ######

# Shift the features to create a lagged dataset

for col in df.columns:
    if col not in ['Date']:
        df[f"{col}_lag"] = df[col].shift(1)

df['target'] = df['Close'].shift(-1)

df = df.drop(columns = ['Open', 'Close', 'Volume', 'Interest Rate', 'GDP', 'Close_pct_change', 'Volume_pct_change', 'RSI'])
df = df.dropna()
df = df.sort_values(by = 'Date').reset_index(drop = True)
print(df.head())


# Split the data into training and testing sets

features = ['Open_lag', 'Volume_lag', 'Interest Rate_lag', 'GDP_lag', 'Close_pct_change_lag', 'Volume_pct_change_lag', 'RSI_lag']
target = 'target'

X = df[features]
y = df[target]

split_index = int(len(X) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


###### Run Models and Evaluate ######

### Models at iteration 1 ###

linearRegression = LinearRegression()
randomForest = RandomForestRegressor(random_state = 42)
xGBoost = xgb.XGBRegressor(objective = 'reg:squarederror', random_state = 42)
lightGBM = lgb.LGBMRegressor(random_state = 42)



# Linear Regression #1 (Default Parameters)

lr_model = 'Linear Regression'
linearRegression.fit(X_train, y_train)
linearPredictions = linearRegression.predict(X_test)

lr_mse = mean_squared_error(y_test, linearPredictions)
lr_r2 = r2_score(y_test, linearPredictions)
lr_residuals = y_test - linearPredictions

lr_results = {
    'model': lr_model,
    'predictions': linearPredictions,
    'mse': lr_mse,
    'r2': lr_r2,
    'residuals': lr_residuals,
    'true': y_test
}


# Random Forest #1 (Default Parameters)

rf_model = 'Random Forest'
randomForest.fit(X_train, y_train)
randomForestPredictions = randomForest.predict(X_test)

rf_mse = mean_squared_error(y_test, randomForestPredictions)
rf_r2 = r2_score(y_test, randomForestPredictions)
rf_residuals = y_test - randomForestPredictions

rf_results = {
    'model': rf_model,
    'predictions': randomForestPredictions,
    'mse': rf_mse,
    'r2': rf_r2,
    'residuals': rf_residuals,
    'true': y_test
}


# XGBoost #1 (Default Parameters)

xg_model = 'XGBoost'
xGBoost.fit(X_train, y_train)
xGBoostPredictions = xGBoost.predict(X_test)

xGB_mse = mean_squared_error(y_test, xGBoostPredictions)
xGB_r2 = r2_score(y_test, xGBoostPredictions)
xGB_residuals = y_test - xGBoostPredictions

xGB_results = {
    'model': xg_model,
    'predictions': xGBoostPredictions,
    'mse': xGB_mse,
    'r2': xGB_r2,
    'residuals': xGB_residuals,
    'true': y_test
}


# LightGBM #1 (Default Parameters)

light_model = 'LightGBM'
lightGBM.fit(X_train, y_train)
lightGBMPredictions = lightGBM.predict(X_test)

lightGBM_mse = mean_squared_error(y_test, lightGBMPredictions)
lightGBM_r2 = r2_score(y_test, lightGBMPredictions)
lightGBM_residuals = y_test - lightGBMPredictions

lightGBM_results = {
    'model': light_model,
    'predictions': lightGBMPredictions,
    'mse': lightGBM_mse,
    'r2': lightGBM_r2,
    'residuals': lightGBM_residuals,
    'true': y_test
}


# First Iteration Results

print(f"{lr_results['model']}:\n  MSE: {lr_results['mse']:.2f}\n  R²: {lr_results['r2']:.4f}\n")
print(f"{rf_results['model']}:\n  MSE: {rf_results['mse']:.2f}\n  R²: {rf_results['r2']:.4f}\n")
print(f"{xGB_results['model']}:\n  MSE: {xGB_results['mse']:.2f}\n  R²: {xGB_results['r2']:.4f}\n")
print(f"{lightGBM_results['model']}:\n  MSE: {lightGBM_results['mse']:.2f}\n  R²: {lightGBM_results['r2']:.4f}\n")


# Plotting the predictions vs true values for each model

fig, axes = plt.subplots(2, 2, figsize = (10, 8))

#Linear Regression
axes[0,0].plot(lr_results['true'].values, label = 'Actual', linewidth = 2)
axes[0,0].plot(lr_results['predictions'], label = 'Predicted', linestyle = '--')
axes[0,0].set_title('Linear Regression')
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel('Close Price')
axes[0,0].legend()

# Random Forest
axes[0,1].plot(rf_results['true'].values, label = 'Actual', linewidth = 2)
axes[0,1].plot(rf_results['predictions'], label = 'Predicted', linestyle = '--')
axes[0,1].set_title('Random Forest')
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel('Close Price')
axes[0,1].legend()

# XGBoost
axes[1,0].plot(xGB_results['true'].values, label = 'Actual', linewidth = 2)
axes[1,0].plot(xGB_results['predictions'], label = 'Predicted', linestyle = '--')
axes[1,0].set_title('XGBoost')
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Close Price')

# LightGBM
axes[1,1].plot(lightGBM_results['true'].values, label = 'Actual', linewidth = 2)
axes[1,1].plot(lightGBM_results['predictions'], label = 'Predicted', linestyle = '--')
axes[1,1].set_title('LightGBM')
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel('Close Price')

plt.tight_layout()
plt.show()


### Models at iteration 2 ###

linearRegression = LinearRegression()
randomForest = RandomForestRegressor(random_state = 42, n_estimators = 200, max_depth = 10)
xGBoost = xgb.XGBRegressor(objective = 'reg:squarederror', random_state = 42, n_estimators = 200, max_depth = 4, learning_rate = 0.1)
lightGBM = lgb.LGBMRegressor(random_state = 42, n_estimators = 200, max_depth = 5, learning_rate = 0.1)


# Linear Regression #2

lr_model = 'Linear Regression'
linearRegression.fit(X_train, y_train)
linearPredictions = linearRegression.predict(X_test)

lr_mse = mean_squared_error(y_test, linearPredictions)
lr_r2 = r2_score(y_test, linearPredictions)
lr_residuals = y_test - linearPredictions

lr_results = {
    'model': lr_model,
    'predictions': linearPredictions,
    'mse': lr_mse,
    'r2': lr_r2,
    'residuals': lr_residuals,
    'true': y_test
}


# Random Forest #2

rf_model = 'Random Forest'
randomForest.fit(X_train, y_train)
randomForestPredictions = randomForest.predict(X_test)

rf_mse = mean_squared_error(y_test, randomForestPredictions)
rf_r2 = r2_score(y_test, randomForestPredictions)
rf_residuals = y_test - randomForestPredictions

rf_results = {
    'model': rf_model,
    'predictions': randomForestPredictions,
    'mse': rf_mse,
    'r2': rf_r2,
    'residuals': rf_residuals,
    'true': y_test
}


# XGBoost #2

xg_model = 'XGBoost'
xGBoost.fit(X_train, y_train)
xGBoostPredictions = xGBoost.predict(X_test)

xGB_mse = mean_squared_error(y_test, xGBoostPredictions)
xGB_r2 = r2_score(y_test, xGBoostPredictions)
xGB_residuals = y_test - xGBoostPredictions

xGB_results = {
    'model': xg_model,
    'predictions': xGBoostPredictions,
    'mse': xGB_mse,
    'r2': xGB_r2,
    'residuals': xGB_residuals,
    'true': y_test
}


# LightGBM #2

light_model = 'LightGBM'
lightGBM.fit(X_train, y_train)
lightGBMPredictions = lightGBM.predict(X_test)

lightGBM_mse = mean_squared_error(y_test, lightGBMPredictions)
lightGBM_r2 = r2_score(y_test, lightGBMPredictions)
lightGBM_residuals = y_test - lightGBMPredictions

lightGBM_results = {
    'model': light_model,
    'predictions': lightGBMPredictions,
    'mse': lightGBM_mse,
    'r2': lightGBM_r2,
    'residuals': lightGBM_residuals,
    'true': y_test
}


# Second Iteration Results

print(f"{lr_results['model']}:\n  MSE: {lr_results['mse']:.2f}\n  R²: {lr_results['r2']:.4f}\n")
print(f"{rf_results['model']}:\n  MSE: {rf_results['mse']:.2f}\n  R²: {rf_results['r2']:.4f}\n")
print(f"{xGB_results['model']}:\n  MSE: {xGB_results['mse']:.2f}\n  R²: {xGB_results['r2']:.4f}\n")
print(f"{lightGBM_results['model']}:\n  MSE: {lightGBM_results['mse']:.2f}\n  R²: {lightGBM_results['r2']:.4f}\n")


# Plotting the predictions vs true values for each model in the second iteration

fig, axes = plt.subplots(2, 2, figsize = (10, 8))

#Linear Regression
axes[0,0].plot(lr_results['true'].values, label = 'Actual', linewidth = 2)
axes[0,0].plot(lr_results['predictions'], label = 'Predicted', linestyle = '--')
axes[0,0].set_title('Linear Regression')
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel('Close Price')
axes[0,0].legend()

# Random Forest
axes[0,1].plot(rf_results['true'].values, label = 'Actual', linewidth = 2)
axes[0,1].plot(rf_results['predictions'], label = 'Predicted', linestyle = '--')
axes[0,1].set_title('Random Forest')
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel('Close Price')
axes[0,1].legend()

# XGBoost
axes[1,0].plot(xGB_results['true'].values, label = 'Actual', linewidth = 2)
axes[1,0].plot(xGB_results['predictions'], label = 'Predicted', linestyle = '--')
axes[1,0].set_title('XGBoost')
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Close Price')

# LightGBM
axes[1,1].plot(lightGBM_results['true'].values, label = 'Actual', linewidth = 2)
axes[1,1].plot(lightGBM_results['predictions'], label = 'Predicted', linestyle = '--')
axes[1,1].set_title('LightGBM')
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel('Close Price')

plt.tight_layout()
plt.show()


### Models at iteration 3 ###

linearRegression = LinearRegression()
randomForest = RandomForestRegressor(random_state = 42, n_estimators = 300, max_depth = 20, min_samples_split = 5)
xGBoost = xgb.XGBRegressor(objective = 'reg:squarederror', random_state = 42, n_estimators = 300, max_depth = 6, learning_rate = 0.05, subsample = 0.8)
lightGBM = lgb.LGBMRegressor(random_state = 42, n_estimators = 300, max_depth = 8, learning_rate = 0.05, num_leaves = 31, subsample = 0.8)


# Linear Regression #3

lr_model = 'Linear Regression'
linearRegression.fit(X_train, y_train)
linearPredictions = linearRegression.predict(X_test)

lr_mse = mean_squared_error(y_test, linearPredictions)
lr_r2 = r2_score(y_test, linearPredictions)
lr_residuals = y_test - linearPredictions

lr_results = {
    'model': lr_model,
    'predictions': linearPredictions,
    'mse': lr_mse,
    'r2': lr_r2,
    'residuals': lr_residuals,
    'true': y_test
}


# Random Forest #3

rf_model = 'Random Forest'
randomForest.fit(X_train, y_train)
randomForestPredictions = randomForest.predict(X_test)

rf_mse = mean_squared_error(y_test, randomForestPredictions)
rf_r2 = r2_score(y_test, randomForestPredictions)
rf_residuals = y_test - randomForestPredictions

rf_results = {
    'model': rf_model,
    'predictions': randomForestPredictions,
    'mse': rf_mse,
    'r2': rf_r2,
    'residuals': rf_residuals,
    'true': y_test
}


# XGBoost #3

xg_model = 'XGBoost'
xGBoost.fit(X_train, y_train)
xGBoostPredictions = xGBoost.predict(X_test)

xGB_mse = mean_squared_error(y_test, xGBoostPredictions)
xGB_r2 = r2_score(y_test, xGBoostPredictions)
xGB_residuals = y_test - xGBoostPredictions

xGB_results = {
    'model': xg_model,
    'predictions': xGBoostPredictions,
    'mse': xGB_mse,
    'r2': xGB_r2,
    'residuals': xGB_residuals,
    'true': y_test
}


# LightGBM #3

light_model = 'LightGBM'
lightGBM.fit(X_train, y_train)
lightGBMPredictions = lightGBM.predict(X_test)

lightGBM_mse = mean_squared_error(y_test, lightGBMPredictions)
lightGBM_r2 = r2_score(y_test, lightGBMPredictions)
lightGBM_residuals = y_test - lightGBMPredictions

lightGBM_results = {
    'model': light_model,
    'predictions': lightGBMPredictions,
    'mse': lightGBM_mse,
    'r2': lightGBM_r2,
    'residuals': lightGBM_residuals,
    'true': y_test
}


# Third Iteration Results

print(f"{lr_results['model']}:\n  MSE: {lr_results['mse']:.2f}\n  R²: {lr_results['r2']:.4f}\n")
print(f"{rf_results['model']}:\n  MSE: {rf_results['mse']:.2f}\n  R²: {rf_results['r2']:.4f}\n")
print(f"{xGB_results['model']}:\n  MSE: {xGB_results['mse']:.2f}\n  R²: {xGB_results['r2']:.4f}\n")
print(f"{lightGBM_results['model']}:\n  MSE: {lightGBM_results['mse']:.2f}\n  R²: {lightGBM_results['r2']:.4f}\n")


# Plotting the predictions vs true values for each model in the second iteration

fig, axes = plt.subplots(2, 2, figsize = (10, 8))

#Linear Regression
axes[0,0].plot(lr_results['true'].values, label = 'Actual', linewidth = 2)
axes[0,0].plot(lr_results['predictions'], label = 'Predicted', linestyle = '--')
axes[0,0].set_title('Linear Regression')
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel('Close Price')
axes[0,0].legend()

# Random Forest
axes[0,1].plot(rf_results['true'].values, label = 'Actual', linewidth = 2)
axes[0,1].plot(rf_results['predictions'], label = 'Predicted', linestyle = '--')
axes[0,1].set_title('Random Forest')
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel('Close Price')
axes[0,1].legend()

# XGBoost
axes[1,0].plot(xGB_results['true'].values, label = 'Actual', linewidth = 2)
axes[1,0].plot(xGB_results['predictions'], label = 'Predicted', linestyle = '--')
axes[1,0].set_title('XGBoost')
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Close Price')

# LightGBM
axes[1,1].plot(lightGBM_results['true'].values, label = 'Actual', linewidth = 2)
axes[1,1].plot(lightGBM_results['predictions'], label = 'Predicted', linestyle = '--')
axes[1,1].set_title('LightGBM')
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel('Close Price')

plt.tight_layout()
plt.show()



### Models at iteration 4 ###

linearRegression = LinearRegression()
randomForest = RandomForestRegressor(random_state = 42, n_estimators = 100, max_depth = None, max_features = 'sqrt')
xGBoost = xgb.XGBRegressor(objective = 'reg:squarederror', random_state = 42, n_estimators = 100, max_depth = 3, learning_rate = 0.2, colsample_bytree = 0.7)
lightGBM = lgb.LGBMRegressor(random_state = 42, n_estimators = 150, max_depth = 10, learning_rate = 0.07, num_leaves = 50, subsample = 0.9, colsample_bytree = 0.7, min_child_samples = 20)


# Linear Regression #4

lr_model = 'Linear Regression'
linearRegression.fit(X_train, y_train)
linearPredictions = linearRegression.predict(X_test)

lr_mse = mean_squared_error(y_test, linearPredictions)
lr_r2 = r2_score(y_test, linearPredictions)
lr_residuals = y_test - linearPredictions

lr_results = {
    'model': lr_model,
    'predictions': linearPredictions,
    'mse': lr_mse,
    'r2': lr_r2,
    'residuals': lr_residuals,
    'true': y_test
}


# Random Forest #4

rf_model = 'Random Forest'
randomForest.fit(X_train, y_train)
randomForestPredictions = randomForest.predict(X_test)

rf_mse = mean_squared_error(y_test, randomForestPredictions)
rf_r2 = r2_score(y_test, randomForestPredictions)
rf_residuals = y_test - randomForestPredictions

rf_results = {
    'model': rf_model,
    'predictions': randomForestPredictions,
    'mse': rf_mse,
    'r2': rf_r2,
    'residuals': rf_residuals,
    'true': y_test
}


# XGBoost #4

xg_model = 'XGBoost'
xGBoost.fit(X_train, y_train)
xGBoostPredictions = xGBoost.predict(X_test)

xGB_mse = mean_squared_error(y_test, xGBoostPredictions)
xGB_r2 = r2_score(y_test, xGBoostPredictions)
xGB_residuals = y_test - xGBoostPredictions

xGB_results = {
    'model': xg_model,
    'predictions': xGBoostPredictions,
    'mse': xGB_mse,
    'r2': xGB_r2,
    'residuals': xGB_residuals,
    'true': y_test
}


# LightGBM #4

light_model = 'LightGBM'
lightGBM.fit(X_train, y_train)
lightGBMPredictions = lightGBM.predict(X_test)

lightGBM_mse = mean_squared_error(y_test, lightGBMPredictions)
lightGBM_r2 = r2_score(y_test, lightGBMPredictions)
lightGBM_residuals = y_test - lightGBMPredictions

lightGBM_results = {
    'model': light_model,
    'predictions': lightGBMPredictions,
    'mse': lightGBM_mse,
    'r2': lightGBM_r2,
    'residuals': lightGBM_residuals,
    'true': y_test
}


# Fourth Iteration Results

print(f"{lr_results['model']}:\n  MSE: {lr_results['mse']:.2f}\n  R²: {lr_results['r2']:.4f}\n")
print(f"{rf_results['model']}:\n  MSE: {rf_results['mse']:.2f}\n  R²: {rf_results['r2']:.4f}\n")
print(f"{xGB_results['model']}:\n  MSE: {xGB_results['mse']:.2f}\n  R²: {xGB_results['r2']:.4f}\n")
print(f"{lightGBM_results['model']}:\n  MSE: {lightGBM_results['mse']:.2f}\n  R²: {lightGBM_results['r2']:.4f}\n")


# Plotting the predictions vs true values for each model in the second iteration

fig, axes = plt.subplots(2, 2, figsize = (10, 8))

#Linear Regression
axes[0,0].plot(lr_results['true'].values, label = 'Actual', linewidth = 2)
axes[0,0].plot(lr_results['predictions'], label = 'Predicted', linestyle = '--')
axes[0,0].set_title('Linear Regression')
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel('Close Price')
axes[0,0].legend()

# Random Forest
axes[0,1].plot(rf_results['true'].values, label = 'Actual', linewidth = 2)
axes[0,1].plot(rf_results['predictions'], label = 'Predicted', linestyle = '--')
axes[0,1].set_title('Random Forest')
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel('Close Price')
axes[0,1].legend()

# XGBoost
axes[1,0].plot(xGB_results['true'].values, label = 'Actual', linewidth = 2)
axes[1,0].plot(xGB_results['predictions'], label = 'Predicted', linestyle = '--')
axes[1,0].set_title('XGBoost')
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Close Price')

# LightGBM
axes[1,1].plot(lightGBM_results['true'].values, label = 'Actual', linewidth = 2)
axes[1,1].plot(lightGBM_results['predictions'], label = 'Predicted', linestyle = '--')
axes[1,1].set_title('LightGBM')
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel('Close Price')

plt.tight_layout()
plt.show()