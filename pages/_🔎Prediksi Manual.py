import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# List of cryptocurrencies
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]

# Function to download the data
def download_data(crypto, start_date, end_date):
    data = yf.download(crypto, start=start_date, end=end_date)
    return data

# Calculate evaluation metrics
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = math.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, rmse, mape

# Streamlit app
def main():
    st.title("Manual Cryptocurrency Price Prediction")
    st.write("This app allows manual parameter tuning for Triple Exponential Smoothing and ARIMA models.")

    # Sidebar Input Data
    st.sidebar.header("Data Download")
    stock_symbol = st.sidebar.selectbox("Select Cryptocurrency:", cryptos)
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
    model_choice = st.sidebar.radio("Select Model:", ["Triple Exponential Smoothing (TES)", "ARIMA"])

    # Update title based on selections
    st.title(f"{model_choice} Prediction for {stock_symbol}")

    # Download stock price data
    data = download_data(stock_symbol, start_date, end_date)

    # Process Prices
    close_prices = data['Close']
    open_prices = data['Open']
    high_prices = data['High']
    low_prices = data['Low']

    # Split data
    train_size = int(len(close_prices) * 0.8)
    train_close = close_prices[:train_size]
    test_close = close_prices[train_size:]
    train_open = open_prices[:train_size]
    test_open = open_prices[train_size:]
    train_high = high_prices[:train_size]
    test_high = high_prices[train_size:]
    train_low = low_prices[:train_size]
    test_low = low_prices[train_size:]

    # Parameter selection
    st.header("Manual Parameter Selection")
    feature_choice = st.selectbox("Select Feature to Forecast:", ["Close", "Open", "High", "Low"])

    if model_choice == "Triple Exponential Smoothing (TES)":
        alpha = st.slider("Alpha (smoothing level)", 0.01, 1.0, 0.5, 0.01)
        beta = st.slider("Beta (smoothing slope)", 0.01, 1.0, 0.5, 0.01)
        gamma = st.slider("Gamma (smoothing seasonal)", 0.01, 1.0, 0.5, 0.01)
        seasonal_periods = st.slider("Seasonal Periods", 1, 365, 12)
    else:
        p = st.slider("p (AR order)", 0, 5, 1)
        d = st.slider("d (Difference order)", 0, 2, 1)
        q = st.slider("q (MA order)", 0, 5, 1)
        order = (p, d, q)

    if st.button("Apply Manual Parameters"):
        if model_choice == "Triple Exponential Smoothing (TES)":
            if feature_choice == "Close":
                model_tes = ExponentialSmoothing(train_close, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(
                    smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
                forecast = model_tes.forecast(steps=len(test_close))
                actual = test_close
            elif feature_choice == "Open":
                model_tes = ExponentialSmoothing(train_open, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(
                    smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
                forecast = model_tes.forecast(steps=len(test_open))
                actual = test_open
            elif feature_choice == "High":
                model_tes = ExponentialSmoothing(train_high, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(
                    smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
                forecast = model_tes.forecast(steps=len(test_high))
                actual = test_high
            elif feature_choice == "Low":
                model_tes = ExponentialSmoothing(train_low, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(
                    smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
                forecast = model_tes.forecast(steps=len(test_low))
                actual = test_low
        else:
            if feature_choice == "Close":
                model_arima = ARIMA(train_close, order=order).fit()
                forecast = model_arima.forecast(steps=len(test_close))
                actual = test_close
            elif feature_choice == "Open":
                model_arima = ARIMA(train_open, order=order).fit()
                forecast = model_arima.forecast(steps=len(test_open))
                actual = test_open
            elif feature_choice == "High":
                model_arima = ARIMA(train_high, order=order).fit()
                forecast = model_arima.forecast(steps=len(test_high))
                actual = test_high
            elif feature_choice == "Low":
                model_arima = ARIMA(train_low, order=order).fit()
                forecast = model_arima.forecast(steps=len(test_low))
                actual = test_low

        # Calculate metrics
        mae, rmse, mape = calculate_metrics(actual, forecast)

        # Display metrics
        st.subheader("Metrics")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.2f}%")

        # Plotting the results
        st.subheader(f"{feature_choice} Prices")
        fig = go.Figure()
        if feature_choice == "Close":
            fig.add_trace(go.Scatter(x=close_prices.index, y=close_prices, mode='lines', name='Actual (Train)', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=test_close.index, y=test_close, mode='lines', name='Actual (Test)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_close.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
        elif feature_choice == "Open":
            fig.add_trace(go.Scatter(x=open_prices.index, y=open_prices, mode='lines', name='Actual (Train)', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=test_open.index, y=test_open, mode='lines', name='Actual (Test)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_open.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
        elif feature_choice == "High":
            fig.add_trace(go.Scatter(x=high_prices.index, y=high_prices, mode='lines', name='Actual (Train)', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=test_high.index, y=test_high, mode='lines', name='Actual (Test)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_high.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
        elif feature_choice == "Low":
            fig.add_trace(go.Scatter(x=low_prices.index, y=low_prices, mode='lines', name='Actual (Train)', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=test_low.index, y=test_low, mode='lines', name='Actual (Test)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_low.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
        
        st.plotly_chart(fig)

        # Display forecast table with actual prices
        st.write("Forecast Table:")
        forecast_df = pd.DataFrame({"Actual Price": actual, "Forecast": forecast}).set_index(actual.index)
        # Format the index to remove time component
        forecast_df.index = forecast_df.index.strftime('%Y-%m-%d')
        st.write(forecast_df)

if __name__ == "__main__":
    main()
