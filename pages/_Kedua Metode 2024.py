import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from warnings import simplefilter

# Suppress warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

# Streamlit app
def main():
    st.title("Top 5 Cryptocurrency Price Prediction")
    st.write("This app predicts cryptocurrency prices using Triple Exponential Smoothing (TES) and Auto ARIMA models.")

    # Sidebar Input Data
    cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
    st.sidebar.header("Data Download")
    stock_symbol = st.sidebar.selectbox("Select Cryptocurrency:", cryptos)
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-11-30"))

    # Download stock price data
    st.write("Fetching data from Yahoo Finance...")
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found. Please adjust the date range.")
        return

    # Process data
    close_prices = data['Close']
    open_prices = data['Open']
    high_prices = data['High']
    low_prices = data['Low']

    # Split data
    train_size = int(len(close_prices) * 0.8)
    train_close, test_close = close_prices[:train_size], close_prices[train_size:]
    train_open, test_open = open_prices[:train_size], open_prices[train_size:]
    train_high, test_high = high_prices[:train_size], high_prices[train_size:]
    train_low, test_low = low_prices[:train_size], low_prices[train_size:]

    # Train TES Models
    st.write("Training Triple Exponential Smoothing models...")
    tes_model_close = ExponentialSmoothing(train_close, trend='add', seasonal='add', seasonal_periods=12).fit()
    tes_model_open = ExponentialSmoothing(train_open, trend='add', seasonal='add', seasonal_periods=12).fit()
    tes_model_high = ExponentialSmoothing(train_high, trend='add', seasonal='add', seasonal_periods=12).fit()
    tes_model_low = ExponentialSmoothing(train_low, trend='add', seasonal='add', seasonal_periods=12).fit()

    # Train Auto ARIMA Models
    st.write("Training Auto ARIMA models...")
    arima_model_close = auto_arima(train_close, start_p=0, max_p=5, start_q=0, max_q=5, d=None, seasonal=False, stepwise=True, trace=False)
    arima_model_open = auto_arima(train_open, start_p=0, max_p=5, start_q=0, max_q=5, d=None, seasonal=False, stepwise=True, trace=False)
    arima_model_high = auto_arima(train_high, start_p=0, max_p=5, start_q=0, max_q=5, d=None, seasonal=False, stepwise=True, trace=False)
    arima_model_low = auto_arima(train_low, start_p=0, max_p=5, start_q=0, max_q=5, d=None, seasonal=False, stepwise=True, trace=False)

    # Forecasting
    forecast_tes_close = tes_model_close.forecast(steps=len(test_close))
    forecast_arima_close = arima_model_close.predict(n_periods=len(test_close))
    forecast_tes_open = tes_model_open.forecast(steps=len(test_open))
    forecast_arima_open = arima_model_open.predict(n_periods=len(test_open))
    forecast_tes_high = tes_model_high.forecast(steps=len(test_high))
    forecast_arima_high = arima_model_high.predict(n_periods=len(test_high))
    forecast_tes_low = tes_model_low.forecast(steps=len(test_low))
    forecast_arima_low = arima_model_low.predict(n_periods=len(test_low))

    # Model Evaluation
    results = {
        "Close": (test_close, forecast_tes_close, forecast_arima_close),
        "Open": (test_open, forecast_tes_open, forecast_arima_open),
        "High": (test_high, forecast_tes_high, forecast_arima_high),
        "Low": (test_low, forecast_tes_low, forecast_arima_low),
    }

    tabs = st.tabs(["Close Prices", "Open Prices", "High Prices", "Low Prices"])

    for (price_type, (test, tes_forecast, arima_forecast)), tab in zip(results.items(), tabs):
        with tab:
            rmse_tes = np.sqrt(mean_squared_error(test, tes_forecast))
            mape_tes = mean_absolute_percentage_error(test, tes_forecast)
            rmse_arima = np.sqrt(mean_squared_error(test, arima_forecast))
            mape_arima = mean_absolute_percentage_error(test, arima_forecast)

            st.header(f"Results for {price_type} Price ({stock_symbol})")
            st.write("**Triple Exponential Smoothing (TES)**")
            st.write(f"- RMSE: {round(rmse_tes, 5)}")
            st.write(f"- MAPE: {round(mape_tes * 100, 5)}%")
            st.write("**Auto ARIMA**")
            st.write(f"- RMSE: {round(rmse_arima, 5)}")
            st.write(f"- MAPE: {round(mape_arima * 100, 5)}%")

            visualize_predictions(data, train_size, test, tes_forecast, arima_forecast, price_type)

def visualize_predictions(data, train_size, y_test, y_pred_tes, y_pred_arima, price_type):
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(x=data.index[:train_size],
                             y=data[price_type][:train_size],
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    # Add actual prices
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
    
def visualize_predictions(data, train_size, y_test, y_pred_tes, y_pred_arima, price_type):
    # Visualisasi Grafik
    fig = go.Figure()

    # Data training
    fig.add_trace(go.Scatter(x=data.index[:train_size],
                             y=data[price_type][:train_size],
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    # Harga aktual
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_test,
                             mode='lines',
                             name="Actual Prices",
                             line=dict(color='blue')))

    # Prediksi TES
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_tes,
                             mode='lines',
                             name="TES Predictions",
                             line=dict(color='red')))

    # Prediksi ARIMA
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

    # Membuat DataFrame untuk tabel
    table_data = {
    'Date': data.index[train_size:].date,  # Pastikan data index sudah sesuai
    'Actual Price': y_test.ravel(),  # Ubah ke 1 dimensi
    'TES Prediction': y_pred_tes.ravel(),  # Ubah ke 1 dimensi
    'ARIMA Prediction': y_pred_arima.ravel(),  # Ubah ke 1 dimensi
    'TES Difference': (y_test - y_pred_tes).ravel(),  # Selisih dalam 1 dimensi
    'ARIMA Difference': (y_test - y_pred_arima).ravel()  # Selisih dalam 1 dimensi
    }

    table_df = pd.DataFrame(table_data)


    # Menampilkan tabel
    st.subheader(f"{price_type} Price Table")
    st.write("Berikut adalah tabel harga aktual, prediksi, dan perbedaannya:")
    st.dataframe(table_df.reset_index(drop=True))

    # Rata-rata dari setiap kolom
    average_actual = y_test.mean()
    average_tes = y_pred_tes.mean()
    average_arima = y_pred_arima.mean()
    average_tes_diff = (y_test - y_pred_tes).mean()
    average_arima_diff = (y_test - y_pred_arima).mean()

    st.write("**Rata-rata:**")
    st.write(f"- Actual Price: {average_actual:.2f}")
    st.write(f"- TES Prediction: {average_tes:.2f}")
    st.write(f"- ARIMA Prediction: {average_arima:.2f}")
    st.write(f"- TES Difference: {average_tes_diff:.2f}")
    st.write(f"- ARIMA Difference: {average_arima_diff:.2f}")


if __name__ == "__main__":
    main()
