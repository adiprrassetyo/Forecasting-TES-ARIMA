import streamlit as st
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from warnings import simplefilter
import plotly.graph_objects as go

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

# Load models
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
models_tes = {}
models_arima = {}

for crypto in cryptos:
    models_tes[crypto] = joblib.load(f"best_tes_model_close_{crypto}.pkl")
    models_arima[crypto] = joblib.load(f"best_arima_model_close_{crypto}.pkl")

# Streamlit app
def main():
    st.title("Top 5 Cryptocurrency Price Prediction")
    st.write("This app predicts cryptocurrency prices using Triple Exponential Smoothing and ARIMA models. The models are trained on historical data for faster predictions.")

    # Sidebar Input Data
    st.sidebar.header("Data Download")
    stock_symbol = st.sidebar.selectbox("Select Cryptocurrency:", cryptos)

    # Download stock price data
    data = yf.download(stock_symbol, start="2021-01-01", end="2024-01-01")

    # Process Close Prices
    close_prices = data['Close']
    open_prices = data['Open']
    high_prices = data['High']
    low_prices = data['Low']

    # Forecast Inputs
    st.header("Forecast Parameters")
    forecast_start_date = st.date_input("Forecast Start Date", pd.to_datetime("2024-01-01"))
    forecast_end_date = st.date_input("Forecast End Date", pd.to_datetime("2024-12-31"))
    forecast_period = st.slider("Forecast Period (days)", 1, 365, 30)
    
    # Ensure forecast_period does not exceed the range
    if (forecast_end_date - forecast_start_date).days < forecast_period:
        forecast_period = (forecast_end_date - forecast_start_date).days

    forecast_steps = forecast_period

    # Forecasting
    model_tes, alpha, beta, gamma = models_tes[stock_symbol]
    model_arima, order = models_arima[stock_symbol]

    # Future dates
    future_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date)
    
    # Adjust forecast_steps to match the length of future_dates if necessary
    forecast_steps = min(forecast_steps, len(future_dates))

    # Future Forecasts
    forecast_future_tes_close = ExponentialSmoothing(close_prices, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=forecast_steps)
    forecast_future_arima_close = ARIMA(close_prices, order=order).fit().forecast(steps=forecast_steps)
    forecast_future_tes_open = ExponentialSmoothing(open_prices, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=forecast_steps)
    forecast_future_arima_open = ARIMA(open_prices, order=order).fit().forecast(steps=forecast_steps)
    forecast_future_tes_high = ExponentialSmoothing(high_prices, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=forecast_steps)
    forecast_future_arima_high = ARIMA(high_prices, order=order).fit().forecast(steps=forecast_steps)
    forecast_future_tes_low = ExponentialSmoothing(low_prices, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=forecast_steps)
    forecast_future_arima_low = ARIMA(low_prices, order=order).fit().forecast(steps=forecast_steps)

    # Ensure the lengths match for future dates and forecasts
    future_dates = future_dates[:forecast_steps]

    # Visualization and Results
    tab1, tab2, tab3, tab4 = st.tabs(["Close Prices", "Open Prices", "High Prices", "Low Prices"])

    with tab1:
        st.subheader(f"Future Predictions for Close Prices ({stock_symbol})")
        visualize_future_predictions(future_dates, forecast_future_tes_close, forecast_future_arima_close, 'Close', close_prices, close_prices.index)
        display_future_table(future_dates, forecast_future_tes_close, forecast_future_arima_close, 'Close')
        
    with tab2:
        st.subheader(f"Future Predictions for Open Prices ({stock_symbol})")
        visualize_future_predictions(future_dates, forecast_future_tes_open, forecast_future_arima_open, 'Open', open_prices, open_prices.index)
        display_future_table(future_dates, forecast_future_tes_open, forecast_future_arima_open, 'Open')
        
    with tab3:
        st.subheader(f"Future Predictions for High Prices ({stock_symbol})")
        visualize_future_predictions(future_dates, forecast_future_tes_high, forecast_future_arima_high, 'High', high_prices, high_prices.index)
        display_future_table(future_dates, forecast_future_tes_high, forecast_future_arima_high, 'High')

    with tab4:
        st.subheader(f"Future Predictions for Low Prices ({stock_symbol})")
        visualize_future_predictions(future_dates, forecast_future_tes_low, forecast_future_arima_low, 'Low', low_prices, low_prices.index)
        display_future_table(future_dates, forecast_future_tes_low, forecast_future_arima_low, 'Low')

def visualize_future_predictions(dates, y_pred_tes, y_pred_arima, price_type, real_data=None, real_dates=None):
    fig = go.Figure()

    # Plot actual data (historical data before the forecast)
    if real_data is not None and real_dates is not None:
        fig.add_trace(go.Scatter(x=real_dates, y=real_data, mode='lines', name=f"Actual {price_type} Prices", line=dict(color='blue')))

    # Add TES future predictions
    fig.add_trace(go.Scatter(x=dates,
                             y=y_pred_tes,
                             mode='lines',
                             name="TES Future Predictions",
                             line=dict(color='red')))

    # Add ARIMA future predictions
    fig.add_trace(go.Scatter(x=dates,
                             y=y_pred_arima,
                             mode='lines',
                             name="ARIMA Future Predictions",
                             line=dict(color='green')))

    fig.update_layout(title=f"Future {price_type} Price Prediction for TES & ARIMA",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)

def display_future_table(dates, y_pred_tes, y_pred_arima, price_type):
    # Display table
    st.write(f"Table Future Predictions for {price_type} Prices")
    # Create DataFrame for future predictions
    df_future = pd.DataFrame({
        'Date': dates.date,
        'TES Prediction': y_pred_tes,
        'ARIMA Prediction': y_pred_arima
    })
    # Display the combined data table
    st.table(df_future.reset_index(drop=True))

if __name__ == "__main__":
    main()
