import streamlit as st 
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
# import tensorflow 

stock_symbol= st.text_input("Enter Stock Ticker 1:")
end_date = datetime.now()
stock_symbol = 'AAPL'

# Fetching historical stock data using yfinance
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Fetching data for the last year, adjust as needed

stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Selecting 'Close' prices for prediction
data = stock_data['Close'].values.reshape(-1, 1)

# Normalizing the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Creating sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 30  # Adjust the sequence length as needed
X, y = create_sequences(scaled_data, sequence_length)

# Splitting the data into training and testing sets
split_index = int(0.8 * len(data))
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Reshaping data for LSTM (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compiling and fitting the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predicting the stock price after 30 days
last_30_days = scaled_data[-sequence_length:]  # Using the last 30 days' data for prediction
last_30_days = last_30_days.reshape(1, sequence_length, 1)
predicted_price_after_30_days = model.predict(last_30_days)
predicted_price_after_30_days = scaler.inverse_transform(predicted_price_after_30_days)
# print(f"Predicted stock price after 30 days: {predicted_price_after_30_days[0][0]}")

predicted_value = predicted_price_after_30_days[0][0]
# print(predicted_value)
st.title(f"Stock price predicted: {predicted_value}")