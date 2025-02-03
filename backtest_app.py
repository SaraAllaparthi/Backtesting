import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure the app
st.set_page_config(page_title="Predictive Analytics with Time Series", layout="wide")
st.title("Predictive Analytics with Time Series")

st.markdown(
    """
    **What does this app do?**

    - **Learns from Historical Data:**  
      It downloads two years of historical stock prices for a given ticker.
    
    - **Predicts Future Prices:**  
      It builds a time series model (using Prophet) to forecast future prices.
    
    - **Trend Analysis:**  
      It compares the model’s predicted prices with the actual share prices over the test period.
    
    This helps you understand the stock’s trend and see how well the model could predict its behavior.
    """
)

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")

# Fixed two-year period for historical data
period_years = 2
end_date = datetime.today()
start_date = end_date - timedelta(days=period_years * 365)
st.sidebar.markdown(f"**Data Period:** {start_date.date()} to {end_date.date()}")

@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    """Fetch historical data for a given ticker and date range."""
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# Fetch data
data = get_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data available for the selected ticker.")
else:
    st.subheader(f"Historical Data for {ticker}")
    st.write(data.tail())

    # Prepare data for Prophet: rename columns as required ("ds" and "y")
    df = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Ensure the 'y' column is numeric and drop any rows with non-numeric values
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y'])

    # Split data into training and testing portions (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Create and fit the Prophet model on training data
    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
    model.fit(train_df)

    # Forecast over the entire period (so that we get predictions for the test period)
    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast = model.predict(future)

    # Get the forecasted values corresponding to the test period
    forecast_test = forecast[['ds', 'yhat']].iloc[split_idx:].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    comparison = pd.concat([test_df, forecast_test['yhat']], axis=1)

    # Plot actual vs predicted prices for the test period
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison['ds'],
        y=comparison['y'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=comparison['ds'],
        y=comparison['yhat'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', dash='dash', width=2)
    ))
    fig.update_layout(
        title=f"{ticker} Actual vs Predicted Prices (Test Period)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Trend Analysis")
    st.markdown(
        """
        - **Actual Price Trend:** The blue line shows the real stock prices over the past two years.
        - **Predicted Price Trend:** The red dashed line shows the forecasted prices for the test period.
        - By comparing these trends, you can see how well the model predicted the stock’s behavior.
        """
    )
