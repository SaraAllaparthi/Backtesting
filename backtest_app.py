import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# App configuration
st.set_page_config(page_title="Trading Strategy with ML Forecasting", layout="wide")
st.title("Trading Strategy with ML Forecasting")

# App Description
st.markdown(
    """
    **What does this app do?**

    This app does two main things:
    
    1. **Learns from the Past:**  
       It fetches historical stock prices (for example, for AAPL) and uses a machine learning model (an LSTM) to learn the patterns in those prices.
    
    2. **Predicts the Future:**  
       After training, the app forecasts future stock prices for a set number of days. It then shows you a graph that compares the past prices with the model’s predictions, giving you an idea of where the stock might be headed.
    
    In short, the app helps you see how a stock has behaved historically and uses that information to predict its future trend.
    """
)

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date", datetime.today())
forecast_days = st.sidebar.number_input("Forecast Days", min_value=1, value=30)
run_forecast = st.sidebar.button("Run Forecast")

# Only run when the user clicks "Run Forecast"
if run_forecast:
    @st.cache_data(ttl=3600)
    def get_data(ticker, start, end):
        """Fetch historical data for a given ticker and date range."""
        data = yf.download(ticker, start=start, end=end)
        data.reset_index(inplace=True)
        if pd.api.types.is_datetime64tz_dtype(data['Date']):
            data['Date'] = data['Date'].dt.tz_localize(None)
        return data

    data_load_state = st.text("Fetching historical data...")
    data = get_data(ticker, start_date, end_date)
    data_load_state.text("")

    if data.empty:
        st.error("No data fetched. Please check the ticker and date range.")
    else:
        st.subheader(f"Historical Data for {ticker}")
        st.write(data.tail())

        # Prepare data for LSTM using only the 'Close' prices.
        df = data[['Date', 'Close']].copy()
        df.set_index('Date', inplace=True)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        # Create sequences for training
        def create_sequences(data, seq_length=60):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i - seq_length:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        seq_length = 60  # Look-back period (days)
        X, y = create_sequences(scaled_data, seq_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and compile the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        st.write("Training the LSTM model... (this may take a few minutes)")
        # Increase epochs to help the model learn the trend
        history = model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        st.write("Training complete.")

        # Optionally, display the training loss curve
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=list(range(len(history.history['loss']))),
            y=history.history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        loss_fig.update_layout(
            title="Training Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white"
        )
        st.plotly_chart(loss_fig, use_container_width=True)

        # Forecast future prices using the trained LSTM model.
        last_sequence = scaled_data[-seq_length:]
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(forecast_days):
            pred = model.predict(current_sequence.reshape(1, seq_length, 1))[0, 0]
            predictions.append(pred)
            current_sequence = np.append(current_sequence[1:], [[pred]], axis=0)

        # Convert predictions back to original scale.
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        last_date = df.index[-1]
        forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days)

        # Plot historical and predicted prices.
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name='Historical Close Price',
            line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Price',
            line=dict(dash='dash', color='red', width=2)))
        fig.update_layout(
            title=f"{ticker} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"""
            **Explanation:**
            - The LSTM model is trained on historical closing prices.
            - It then predicts the next {forecast_days} days of prices based on the patterns it has learned.
            - The forecast is shown as a dashed red line compared to the blue line of historical prices.
            """
        )
