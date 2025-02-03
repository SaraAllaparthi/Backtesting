import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# App configuration
st.set_page_config(page_title="Backtesting & Future Strategy Simulator", layout="wide")
st.title("Backtesting & Future Investment Strategy Simulator")

st.write(
    """
    This app demonstrates a simple moving average crossover strategy.
    
    **Part 1: Historical Backtest**  
    The app tests the strategy on past data: it calculates short- and long-term moving averages, generates buy/sell signals, 
    and shows the cumulative returns compared to a buy-and-hold approach.

    **Part 2: Future Projections**  
    It also runs a Monte Carlo simulation using historical return statistics to generate possible future price paths. 
    This can help you visualize potential future performance of the stock and the strategy.
    """
)

# Sidebar for user inputs
st.sidebar.header("Strategy & Simulation Parameters")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Historical Start Date", datetime.today() - timedelta(days=5*365))
end_date = st.sidebar.date_input("Historical End Date", datetime.today())
short_window = st.sidebar.number_input("Short MA Window (days)", min_value=1, value=50, step=1)
long_window = st.sidebar.number_input("Long MA Window (days)", min_value=1, value=200, step=1)
simulation_days = st.sidebar.number_input("Simulation Days", min_value=30, value=180, step=10)
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1, value=10, step=1)

@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    """Fetch historical data for a given ticker and date range."""
    data = yf.download(ticker, start=start, end=end)
    data = data.reset_index()  # Convert the date index to a column
    # Remove timezone information if present
    if pd.api.types.is_datetime64tz_dtype(data['Date']):
        data['Date'] = data['Date'].dt.tz_localize(None)
    return data

# Fetch historical data
data_load_state = st.text("Fetching historical data...")
data = get_data(ticker, start_date, end_date)
data_load_state.text("")

if data.empty:
    st.error("No historical data fetched. Check the ticker or date range.")
else:
    st.subheader(f"Historical Data for {ticker}")
    st.write(data.tail())

    # --- PART 1: Historical Backtest ---
    st.markdown("### Part 1: Historical Backtest")
    # Calculate moving averages
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

    # Generate signals: Signal = 1 when Short_MA > Long_MA, else 0
    data['Signal'] = 0
    data.loc[short_window:, 'Signal'] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    data['Position'] = data['Signal'].diff()  # 1 indicates a buy signal, -1 a sell signal

    # Calculate daily returns and strategy returns
    data['Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)
    data['Cumulative_Market'] = (1 + data['Return']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

    # Display performance metrics
    total_return_market = data['Cumulative_Market'].iloc[-1] - 1
    total_return_strategy = data['Cumulative_Strategy'].iloc[-1] - 1
    st.write(f"**Market Total Return:** {total_return_market:.2%}")
    st.write(f"**Strategy Total Return:** {total_return_strategy:.2%}")

    # Plot price with moving averages and signals
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Short_MA'], mode='lines', name=f'Short MA ({short_window} days)', line=dict(color='orange')))
    fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Long_MA'], mode='lines', name=f'Long MA ({long_window} days)', line=dict(color='green')))

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
    fig_price.update_layout(title=f"{ticker} Price & Moving Averages", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_price, use_container_width=True)

    # Plot cumulative returns
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Scatter(x=data['Date'], y=data['Cumulative_Market'], mode='lines', name='Market Return', line=dict(color='blue')))
    fig_returns.add_trace(go.Scatter(x=data['Date'], y=data['Cumulative_Strategy'], mode='lines', name='Strategy Return', line=dict(color='red')))
    fig_returns.update_layout(title="Cumulative Returns: Market vs. Strategy", xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig_returns, use_container_width=True)

    # --- PART 2: Future Projections via Monte Carlo Simulation ---
    st.markdown("### Part 2: Future Investment Strategy Projection")
    st.write(
        "Using historical return statistics, the following simulation shows several possible future price paths. "
        "This can help you visualize potential future outcomes and consider how the strategy might perform."
    )

    # Calculate daily return statistics from historical data
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    mu = data['Log_Return'].mean()
    sigma = data['Log_Return'].std()
    last_price = data['Close'].iloc[-1]

    st.write(f"Calculated daily drift (mu): {mu:.5f}, volatility (sigma): {sigma:.5f}")

    # Simulate future price paths using Geometric Brownian Motion
    dt = 1  # 1 day time step
    simulation_results = np.zeros((simulation_days, num_simulations))

    for sim in range(num_simulations):
        prices = [last_price]
        for i in range(1, simulation_days):
            random_shock = np.random.normal(loc=0, scale=1)
            # Geometric Brownian Motion formula
            price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shock)
            prices.append(price)
        simulation_results[:, sim] = prices

    # Create a date range for simulation results
    last_date = data['Date'].iloc[-1]
    simulation_dates = pd.date_range(last_date + timedelta(days=1), periods=simulation_days)

    # Plot the simulated future price paths
    fig_sim = go.Figure()
    for i in range(num_simulations):
        fig_sim.add_trace(go.Scatter(
            x=simulation_dates,
            y=simulation_results[:, i],
            mode='lines',
            name=f"Simulation {i+1}",
            line=dict(width=1),
            opacity=0.7
        ))
    fig_sim.update_layout(title=f"Monte Carlo Simulation: {num_simulations} Future Price Paths for {ticker}",
                          xaxis_title="Date", yaxis_title="Simulated Price")
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown(
        """
        **Note:**  
        - The Monte Carlo simulation is based on historical average returns and volatility and represents one way to visualize potential future outcomes.  
        - These projections are for illustrative purposes only and should not be taken as a guarantee of future performance.
        """
    )
