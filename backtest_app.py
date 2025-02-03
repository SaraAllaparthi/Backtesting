import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Backtesting Trading Strategies", layout="wide")
st.title("Backtesting Trading Strategies: Moving Average Crossover")

st.write(
    """
    This app demonstrates a simple moving average crossover strategy. 
    The strategy goes long when the short-term moving average crosses above the long-term moving average 
    and exits (or goes short) when the short-term moving average crosses below the long-term moving average.
    
    Adjust the parameters in the sidebar to see how the strategy performs.
    """
)

# Sidebar for user inputs
st.sidebar.header("Strategy Parameters")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date", datetime.today())
short_window = st.sidebar.number_input("Short Moving Average Window (days)", min_value=1, value=50, step=1)
long_window = st.sidebar.number_input("Long Moving Average Window (days)", min_value=1, value=200, step=1)

@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    """Fetch historical data for a given ticker and date range."""
    data = yf.download(ticker, start=start, end=end)
    # Reset index to convert the date index into a column
    data = data.reset_index()
    # Remove timezone information from the Date column, if any
    if pd.api.types.is_datetime64tz_dtype(data['Date']):
        data['Date'] = data['Date'].dt.tz_localize(None)
    return data

# Fetch historical data
data_load_state = st.text("Fetching data...")
data = get_data(ticker, start_date, end_date)
data_load_state.text("")

if data.empty:
    st.error("No data fetched. Check the ticker or date range.")
else:
    st.subheader(f"Historical Data for {ticker}")
    st.write(data.tail())

    # Calculate moving averages
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

    # Generate signals: 1 when short MA > long MA, else 0
    data['Signal'] = 0
    data.loc[short_window:, 'Signal'] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    # Calculate positions as the difference in signals (1 indicates a buy, -1 indicates a sell)
    data['Position'] = data['Signal'].diff()

    # Calculate daily returns and strategy returns
    data['Return'] = data['Close'].pct_change()
    # Assume we take the position from the previous day (shifted signal)
    data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)

    # Calculate cumulative returns
    data['Cumulative_Market'] = (1 + data['Return']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

    # Compute overall performance metrics
    total_return_market = data['Cumulative_Market'].iloc[-1] - 1
    total_return_strategy = data['Cumulative_Strategy'].iloc[-1] - 1

    st.write(f"**Market Total Return:** {total_return_market:.2%}")
    st.write(f"**Strategy Total Return:** {total_return_strategy:.2%}")

    # Plot the price, moving averages, and buy/sell signals
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Short_MA'], mode='lines', name=f'Short MA ({short_window} days)', line=dict(color='orange')))
    fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Long_MA'], mode='lines', name=f'Long MA ({long_window} days)', line=dict(color='green')))

    # Mark buy signals (when Position == 1) and sell signals (when Position == -1)
    buy_signals = data[data['Position'] == 1]
    sell_signals = data[data['Position'] == -1]

    fig_price.add_trace(go.Scatter(
        x=buy_signals['Date'], 
        y=buy_signals['Close'], 
        mode='markers', 
        marker_symbol='triangle-up', 
        marker_color='green', 
        marker_size=10, 
        name='Buy Signal'
    ))
    fig_price.add_trace(go.Scatter(
        x=sell_signals['Date'], 
        y=sell_signals['Close'], 
        mode='markers', 
        marker_symbol='triangle-down', 
        marker_color='red', 
        marker_size=10, 
        name='Sell Signal'
    ))

    fig_price.update_layout(title=f"{ticker} Price and Moving Averages", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_price, use_container_width=True)

    # Plot cumulative returns of the market vs. strategy
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Scatter(x=data['Date'], y=data['Cumulative_Market'], mode='lines', name='Market Return', line=dict(color='blue')))
    fig_returns.add_trace(go.Scatter(x=data['Date'], y=data['Cumulative_Strategy'], mode='lines', name='Strategy Return', line=dict(color='red')))
    fig_returns.update_layout(title="Cumulative Returns: Market vs Strategy", xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig_returns, use_container_width=True)
