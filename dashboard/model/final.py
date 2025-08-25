# Import Libraries

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression

from datetime import datetime
import pandas_datareader as web

import io
import base64

# function to download stock data

def download_stock_data(symbol, start, end):

    df = web.DataReader(symbol, "stooq", start = start, end = end)

    return df


# function to download Interest Rate and GDP

def download_IR_GDP(start, end):

    # Interest Rate
    effr = web.DataReader('FEDFUNDS', 'fred', start = start, end = end)

    # GDP
    gdp =  web.DataReader('GDP', 'fred', start = start, end = end)

    return effr, gdp

# function to merge datasets

def merge_datasets(df, effr, gdp):
    df = df.merge(effr, left_on = 'Date', right_index = True, how = 'outer')
    df = df.merge(gdp, left_on = 'Date', right_index = True, how = 'outer')

    df['FEDFUNDS'] = df['FEDFUNDS'].ffill()
    df['GDP'] = df['GDP'].ffill()

    df = df.rename(columns = {'FEDFUNDS' : 'Interest_Rate'})

    return df

# function to clean dataset (drop columns, drop NA's)

def clean_dataset(df):
    df = df.drop(columns = ['High', 'Low'])
    df = df.dropna()

    return df

# function to calculate technical indicators

def calculate_technical_indicators(df):
    df['MA_50'] = df['Close'].rolling(window = 50).mean()

    df['Close_pct_change'] = df['Close'].pct_change()
    df['Volume_pct_change'] = df['Volume'].pct_change()

    df['RSI'] = df['Close'].rolling(window = 14).apply(lambda x: 100 - (100 / (1 + (x.diff() / x.diff().abs()).mean())))

    return df

# function to shift dataset (shift columns)

def shift_dataset(df, days):
    
    for col in df.columns:
        if col not in ['Date']:
            df[f"{col}_lag"] = df[col].shift(days)
            #df = df.drop(columns = [col], inplace = True)

    df['Target'] = df['Close'].shift(-days)

    df = df.drop(columns = ['Open', 'Close', 'Volume', 'Interest_Rate', 'GDP', 'Close_pct_change', 'Volume_pct_change', 'RSI'])
    df = df.dropna()
    df = df.sort_values(by = 'Date').reset_index(drop = True)

    return df

# function to split dataset into training and testing sets

def split_dataset(df):
    X = df.drop(columns = ['Target', 'Date'])
    y = df['Target']

    split_index = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

# function to train and predict model

def train_predict_model(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return predictions

# function to plot a graph

def generate_plot(X_test, y_test, predictions):
    fig, ax = plt.subplots(figsize = (10, 5))
    plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
    plt.plot(y_test.index, predictions, label='Predicted', linestyle='--')
    
    ax.set_ylabel('Price')
    ax.set_title('Stock Price Prediction')
    ax.legend()

    ax.xaxis.set_visible(False)
    ax.set_xlabel("")

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return image_base64

# Main function

def main(symbol, days):

    end = datetime.now()
    start = datetime(end.year - 30, end.month, end.day)

    # Download stock data
    df = download_stock_data(symbol, start, end)

    # Download Interest Rate and GDP
    effr, gdp = download_IR_GDP(start, end)

    # Merge datasets
    df = merge_datasets(df, effr, gdp)

    # Clean dataset
    df = clean_dataset(df)

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Shift dataset
    df = shift_dataset(df, days)

    # Drop NA's once more
    #df = df.dropna()

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Train and predict model
    predictions = train_predict_model(X_train, y_train, X_test, y_test)

    # Plot graph

    plot = generate_plot(X_test, y_test, predictions)

    # Print predictions
    return predictions, plot