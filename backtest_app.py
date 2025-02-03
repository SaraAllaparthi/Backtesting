import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -------------------------
# Dashboard Configuration
# -------------------------
st.set_page_config(page_title="Predictive Analytics & Investment Forecasting", layout="wide")
st.markdown(
    """
    <h1 style="text-align: center; color: #4F8BF9;">Predictive Analytics & Investment Forecasting</h1>
    <h3 style="text-align: center;">Forecast Future Stock Prices with AI-driven Time Series Analysis</h3>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.markdown("## Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
period_years = st.sidebar.slider("Data Period (Years)", min_value=1, max_value=5, value=2)
forecast_days = st.sidebar.number_input("Forecast Days", min_value=1, value=30)

end_date = datetime.today()
start_date = end_date - timedelta(days=period_years * 365)
st.sidebar.markdown(f"**Data Period:** {start_date.date()} to {end_date.date()}")

# -------------------------
# Data Ingestion
# -------------------------
@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

data = get_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data available for the selected ticker.")
    st.stop()
    
# Display a preview of the data
st.subheader(f"Historical Data for {ticker}")
st.write(data.tail())
st.write("**Available columns:**", data.columns.tolist())

# -------------------------
# Data Preparation
# -------------------------
# Determine which column to use for prices
if 'Close' in data.columns:
    price_col = 'Close'
elif 'Adj Close' in data.columns:
    price_col = 'Adj Close'
else:
    st.error("No 'Close' or 'Adj Close' column found in the data.")
    st.stop()

# Prepare data for Prophet: rename columns ("ds" for date, "y" for price)
df = data[['Date', price_col]].rename(columns={'Date': 'ds', price_col: 'y'})

# Use a custom conversion function to ensure we get float values
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

df['y'] = df['y'].apply(safe_float)
df = df.dropna(subset=['y'])

st.markdown("### Data Prepared for Forecasting")
st.write(df.head())

# -------------------------
# Model Training & Forecasting
# -------------------------
# Split data into training (80%) and testing (20%)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# Build and fit the Prophet model on training data
model = Prophet(daily_seasonality=False, yearly_seasonality=True)
model.fit(train_df)

# Forecast for the entire period (this ensures predictions for the test period)
future = model.make_future_dataframe(periods=len(test_df), freq='D')
forecast = model.predict(future)

# Extract forecasted values for the test period
forecast_test = forecast[['ds', 'yhat']].iloc[split_idx:].reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
comparison = pd.concat([test_df, forecast_test['yhat']], axis=1)

# -------------------------
# Visualization
# -------------------------
st.markdown("### Actual vs Predicted Prices (Test Period)")
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
    - **Actual Price Trend:** The blue line shows the real historical stock prices.
    - **Predicted Price Trend:** The red dashed line shows the forecasted prices during the test period.
    - Compare these trends to assess how well the model predicts future price movements.
    """
)
