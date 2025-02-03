import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="ML Enhanced Trading Strategy", layout="wide")
st.title("Trading Strategy with ML Forecasting")

st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date", datetime.today())
forecast_days = st.sidebar.number_input("Forecast Days", min_value=1, value=30)

# Fetch data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    if pd.api.types.is_datetime64tz_dtype(data['Date']):
        data['Date'] = data['Date'].dt.tz_localize(None)
    return data

data = get_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data fetched. Check the ticker or date range.")
else:
    st.subheader(f"Historical Data for {ticker}")
    st.write(data.tail())

    # Prepare data for LSTM: we'll use only 'Close' prices
    df = data[['Date', 'Close']]
    df.set_index('Date', inplace=True)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create sequences for training
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_length = 60  # Look back 60 days
    X, y = create_sequences(scaled_data, seq_length)

    # Reshape data for LSTM: (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build a simple LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    st.write("Training the LSTM model (this may take a while)...")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)  # For demo purposes, we use 10 epochs

    # Forecast future prices
    last_sequence = scaled_data[-seq_length:]
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(forecast_days):
        pred = model.predict(current_sequence.reshape(1, seq_length, 1))[0,0]
        predictions.append(pred)
        current_sequence = np.append(current_sequence[1:], [[pred]], axis=0)

    # Convert predictions back to original scale
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    last_date = df.index[-1]
    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days)

    # Plot historical and predicted prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Close Price'))
    fig.add_trace(go.Scatter(x=forecast_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Price', line=dict(dash='dash', color='red')))
    fig.update_layout(title=f"{ticker} Price Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Explanation:**
        - The LSTM model is trained on the historical closing prices.
        - It then predicts the next {} days of prices based on patterns it has learned.
        - This forecast can be used alongside traditional backtesting to refine your investment strategy.
        """.format(forecast_days)
    )
