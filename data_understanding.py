import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import matplotlib.dates as mdates

import pandas_datareader.data as web

# Download historical stock data

start = "2010-01-01"
end = "2025-01-01"

df = web.DataReader("^SPX", "stooq", start = start, end = end).sort_index().reset_index()

print(df.head(), "\n")
print("Data Types: \n", df.dtypes, "\n")


# Download historical data for Interest Rate and GDP

effr = web.DataReader('FEDFUNDS', 'fred', start = start, end = end)
gdp = web.DataReader('GDP', 'fred', start = start, end = end)

print(effr.head(), "\n", gdp.head(), "\n")


# Merge datasets
# We outer merge because we will want to forward fill missing values for Interest Rate and GDP after merge
# Forward filling because our data of Interest Rate and GDP starts from 1/1/2010
# and since its quarterly data, we might have missing values at the latest dates.

df = df.merge(effr, left_on = 'Date', right_index = True, how = 'outer')
df = df.merge(gdp, left_on = 'Date', right_index = True, how = 'outer')

print(df.head(), "\n")


# We will forward fill missing values for Interest Rate and GDP first
# This is because we want to keep the stock data intact and fill in the missing values for Interest Rate and GDP after merge
# Then we remove any rows with NaN values Because we want to only keep rows where Market data is available
# We forward fill because Interest rates and GDP are typically reported quarterly, so we want to fill in the gaps

df['FEDFUNDS'] = df['FEDFUNDS'].ffill()
df['GDP'] = df['GDP'].ffill()

df = df.dropna()

print(df.isnull().sum(), "\n")


# Rename columns for clarity

df = df.rename(columns = {'FEDFUNDS': 'Interest Rate'})

print(df.head(), "\n")


##### Plot the data to understand better #####

# Volume of S&P 500.
# We want to visualize the volume of trades for S&P 500 over time.
# This will help us understand the trading activity and liquidity of the index.
# We want to see if there are any trends or patterns in the volume of trades, or if there are any spikes or drops in volume that 
# correspond to significant market events

plt.figure(figsize = (14, 7))
plt.plot(df['Date'], df['Volume'], color = 'blue')
plt.title('Sales Volume for S&P 500')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid()
plt.show()


# Closing Price of S&P 500
# We want to visualize the closing price of S&P 500 over time.
# We want to see if S&P 500 is trending upwards or downwards overtime.

plt.figure(figsize = (14, 7))
plt.plot(df['Date'], df['Close'], color = 'blue')
plt.title('Closing Price of S&P 500')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.grid()
plt.show()


# Daily Return Histogram
# This can help us understand how volatile the stock is and how often it experiences large price changes.

df['daily_return'] = df['Close'].pct_change()
plt.figure(figsize=(14, 7))
plt.hist(df['daily_return'].dropna(), bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# Closing Price vs Interest Rate
# We want to see if there is any correlation between the closing price of S&P 500 and the interest rate.
# How does the interest rate affect the stock market?
# Does it respond to changes in interest rates?

fig, ax1 = plt.subplots(figsize = (14, 7))

# First axis
ax1.plot(df['Date'], df['Close'], color = 'blue', label = 'Closing Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Closing Price', color = 'blue')
ax1.tick_params(axis = 'y', labelcolor = 'blue')

# Second axis
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Interest Rate'], color = 'green', label = 'Interest Rate')
ax2.set_ylabel('Interest Rate', color = 'green')
ax2.tick_params(axis = 'y', labelcolor = 'green')

ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()

plt.title('Closing Price vs Interest Rate')
plt.show()


# Closing Price vs GDP
# We want to see if there is any correlation between the closing price of S&P 500 and GDP.
# How does the GDP affect the stock market?
# Does it respond to changes in GDP?

fig, ax1 = plt.subplots(figsize = (14, 7))

# First axis
ax1.plot(df['Date'], df['Close'], color = 'blue', label = 'Closing Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Closing Price', color = 'blue')
ax1.tick_params(axis = 'y', labelcolor = 'blue')

# Second axis
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['GDP'], color = 'green', label = 'GDP')
ax2.set_ylabel('GDP', color = 'green')
ax2.tick_params(axis = 'y', labelcolor = 'green')

ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()

plt.title('Closing Price vs GDP')
plt.show()