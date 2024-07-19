import streamlit as st
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from warnings import simplefilter

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
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

    # Download stock price data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Process Close Prices
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

    # Forecasting
    model_tes, alpha, beta, gamma = models_tes[stock_symbol]
    model_arima, order = models_arima[stock_symbol]
    forecast_tes_close = model_tes.forecast(steps=len(test_close))
    forecast_arima_close = model_arima.forecast(steps=len(test_close))
    forecast_tes_open = ExponentialSmoothing(train_open, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=len(test_open))
    forecast_arima_open = ARIMA(train_open, order=order).fit().forecast(steps=len(test_open))
    forecast_tes_high = ExponentialSmoothing(train_high, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=len(test_high))
    forecast_arima_high = ARIMA(train_high, order=order).fit().forecast(steps=len(test_high))
    forecast_tes_low = ExponentialSmoothing(train_low, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=len(test_low))
    forecast_arima_low = ARIMA(train_low, order=order).fit().forecast(steps=len(test_low))

    # Model Evaluation
    rmse_tes_close = np.sqrt(mean_squared_error(test_close, forecast_tes_close))
    mape_tes_close = mean_absolute_percentage_error(test_close, forecast_tes_close)
    rmse_arima_close = np.sqrt(mean_squared_error(test_close, forecast_arima_close))
    mape_arima_close = mean_absolute_percentage_error(test_close, forecast_arima_close)

    rmse_tes_open = np.sqrt(mean_squared_error(test_open, forecast_tes_open))
    mape_tes_open = mean_absolute_percentage_error(test_open, forecast_tes_open)
    rmse_arima_open = np.sqrt(mean_squared_error(test_open, forecast_arima_open))
    mape_arima_open = mean_absolute_percentage_error(test_open, forecast_arima_open)

    rmse_tes_high = np.sqrt(mean_squared_error(test_high, forecast_tes_high))
    mape_tes_high = mean_absolute_percentage_error(test_high, forecast_tes_high)
    rmse_arima_high = np.sqrt(mean_squared_error(test_high, forecast_arima_high))
    mape_arima_high = mean_absolute_percentage_error(test_high, forecast_arima_high)

    rmse_tes_low = np.sqrt(mean_squared_error(test_low, forecast_tes_low))
    mape_tes_low = mean_absolute_percentage_error(test_low, forecast_tes_low)
    rmse_arima_low = np.sqrt(mean_squared_error(test_low, forecast_arima_low))
    mape_arima_low = mean_absolute_percentage_error(test_low, forecast_arima_low)

    tab1, tab2, tab3, tab4 = st.tabs(["Close Prices", "Open Prices", "High Prices", "Low Prices"])

    with tab1:
        st.header(f"Results Close Price {stock_symbol} for TES and ARIMA Models")
        st.write("Triple Exponential Smoothing - RMSE:", round(rmse_tes_close, 5))
        st.write("Triple Exponential Smoothing - MAPE:", round(mape_tes_close * 100, 5), "%")
        st.write("ARIMA - RMSE:", round(rmse_arima_close, 5))
        st.write("ARIMA - MAPE:", round(mape_arima_close * 100, 5), "%")
        visualize_predictions(data, train_size, test_close, forecast_tes_close, forecast_arima_close, 'Close')

    with tab2:
        st.header(f"Results Open Price {stock_symbol} for TES and ARIMA Models")
        st.write("Triple Exponential Smoothing - RMSE:", round(rmse_tes_open, 5))
        st.write("Triple Exponential Smoothing - MAPE:", round(mape_tes_open * 100, 5), "%")
        st.write("ARIMA - RMSE:", round(rmse_arima_open, 5))
        st.write("ARIMA - MAPE:", round(mape_arima_open * 100, 5), "%")
        visualize_predictions(data, train_size, test_open, forecast_tes_open, forecast_arima_open, 'Open')

    with tab3:
        st.header(f"Results High Price {stock_symbol} for TES and ARIMA Models")
        st.write("Triple Exponential Smoothing - RMSE:", round(rmse_tes_high, 5))
        st.write("Triple Exponential Smoothing - MAPE:", round(mape_tes_high * 100, 5), "%")
        st.write("ARIMA - RMSE:", round(rmse_arima_high, 5))
        st.write("ARIMA - MAPE:", round(mape_arima_high * 100, 5), "%")
        visualize_predictions(data, train_size, test_high, forecast_tes_high, forecast_arima_high, 'High')

    with tab4:
        st.header(f"Results Low Price {stock_symbol} for TES and ARIMA Models")
        st.write("Triple Exponential Smoothing - RMSE:", round(rmse_tes_low, 5))
        st.write("Triple Exponential Smoothing - MAPE:", round(mape_tes_low * 100, 5), "%")
        st.write("ARIMA - RMSE:", round(rmse_arima_low, 5))
        st.write("ARIMA - MAPE:", round(mape_arima_low * 100, 5), "%")
        visualize_predictions(data, train_size, test_low, forecast_tes_low, forecast_arima_low, 'Low')

def visualize_predictions(data, train_size, y_test, y_pred_tes, y_pred_arima, price_type):
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(x=data.index[:train_size],
                             y=data[price_type][:train_size],
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    # Add actual stock prices
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_test,
                             mode='lines',
                             name="Actual Prices",
                             line=dict(color='blue')))

    # Add TES predictions
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_tes,
                             mode='lines',
                             name="TES Predictions",
                             line=dict(color='red')))

    # Add ARIMA predictions
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_arima,
                             mode='lines',
                             name="ARIMA Predictions",
                             line=dict(color='green')))

    fig.update_layout(title=f"{price_type} Price Prediction for TES & ARIMA",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
