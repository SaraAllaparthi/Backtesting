import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure the app
st.set_page_config(page_title="Predictive Analytics & Investment Forecasting", layout="wide")
st.title("Predictive Analytics & Investment Forecasting")

st.markdown(
    """
    ### What This App Does
    - **Historical Analysis:** Downloads two years of historical stock prices for a given ticker.
    - **Forecasting:** Uses the Prophet model to forecast future prices based on historical trends.
    - **Trend Comparison:** Compares actual prices (blue line) with predicted prices (red dashed line) for the test period.
    
    This helps investors understand past trends and see how well the model forecasts future price movements.
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
    
    # Debug: Display available columns
    st.write("Available columns:", data.columns.tolist())
    
    # Determine which column to use for prices
    if 'Close' in data.columns:
        price_col = 'Close'
    elif 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    else:
        st.error("No 'Close' or 'Adj Close' column found in the data.")
        st.stop()
    
    # Prepare data for Prophet: rename columns as required ("ds" and "y")
    df = data[['Date', price_col]].rename(columns={'Date': 'ds', price_col: 'y'})
    
    # Debug: Check before conversion
    st.write("Data before numeric conversion:", df.head())
    
    # Convert the 'y' column to numeric and drop rows with missing values
    try:
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
    except Exception as e:
        st.error(f"Error converting 'y' to numeric: {e}")
    df = df.dropna(subset=['y'])
    
    # Debug: Verify columns after cleaning
    st.write("Columns after cleaning:", df.columns.tolist())
    st.write("Sample data after cleaning:", df.head())

    # Split data into training (80%) and testing (20%) portions
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Build and fit the Prophet model on training data
    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
    try:
        model.fit(train_df)
    except Exception as e:
        st.error(f"Error training Prophet model: {e}")
        st.stop()

    # Forecast for the entire period (so we get predictions for the test period)
    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast = model.predict(future)

    # Extract forecasted values corresponding to the test period
    forecast_test = forecast[['ds', 'yhat']].iloc[split_idx:].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    comparison = pd.concat([test_df, forecast_test['yhat']], axis=1)

    # Debug: Check the comparison DataFrame
    st.write("Comparison sample data:", comparison.head())

    # Plot actual vs predicted prices for the test period using Plotly
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
        - **Predicted Price Trend:** The red dashed line shows the forecasted prices during the test period.
        - Comparing these trends helps you evaluate how well the model predicts future price movements.
        """
    )
