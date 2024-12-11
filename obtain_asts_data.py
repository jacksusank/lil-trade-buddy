import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Fetch historical data for $ASTS
ticker = "ASTS"
data = yf.download(ticker, start="2020-01-01", end="2024-12-31", interval="1d")

# Display the first few rows
print(data.head())


# Drop unused columns (keeping only Adjusted Close and Volume as an example)
# data = data[['Adj Close', 'Volume']]

# Add a simple moving average (SMA) as a feature
data['SMA_10 ASTS'] = data['Adj Close'].rolling(window=10).mean()
data['SMA_50 ASTS'] = data['Adj Close'].rolling(window=50).mean()


# Drop rows with NaN values (due to rolling calculations)
data = data.dropna()

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Convert to a NumPy array
data_array = np.array(data_scaled)

# Assuming `data_array` is your preprocessed NumPy array
np.save('./data/asts_stock_data.npy', data_array)


# Display the first few rows after preprocessing
print(data.head())
