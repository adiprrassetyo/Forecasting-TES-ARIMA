import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import plotly.graph_objects as go
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

# Streamlit app
def main():
    st.title("Cryptocurrency Price Prediction")
    st.write("Upload your own cryptocurrency dataset (CSV) for future price prediction using automated Triple Exponential Smoothing (TES) and Auto ARIMA models.")

    # File Upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Load dataset
        data = pd.read_csv(uploaded_file, parse_dates=True, index_col="Date")
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Process Prices
        close_prices = data['Close']
        open_prices = data['Open']
        high_prices = data['High']
        low_prices = data['Low']

        # Use all available historical data
        historical_dates = data.index
        historical_close_prices = close_prices
        historical_open_prices = open_prices
        historical_high_prices = high_prices
        historical_low_prices = low_prices

        # Forecast Inputs
        st.header("Forecast Parameters")
        forecast_start_date = st.date_input("Forecast Start Date", pd.to_datetime("2024-01-01"))
        forecast_end_date = st.date_input("Forecast End Date", pd.to_datetime("2024-12-31"))
        forecast_period = st.slider("Forecast Period (days)", 1, 365, 30)

        # Ensure forecast period does not exceed the range
        forecast_steps = min((forecast_end_date - forecast_start_date).days, forecast_period)

        # Future dates for forecasting
        future_dates = pd.date_range(start=forecast_start_date, periods=forecast_steps)

        # Tombol untuk mulai prediksi
        if st.button("Mulai Prediksi"):
            st.write("Menjalankan model TES dan ARIMA...")

            # Perform predictions for all 4 tabs first
            st.write("Menghitung prediksi untuk semua tab...")
            forecast_results = {}  # Dictionary to store all results

            forecast_results['Close'] = predict_prices(close_prices, future_dates, forecast_steps, 'Close')
            forecast_results['Open'] = predict_prices(open_prices, future_dates, forecast_steps, 'Open')
            forecast_results['High'] = predict_prices(high_prices, future_dates, forecast_steps, 'High')
            forecast_results['Low'] = predict_prices(low_prices, future_dates, forecast_steps, 'Low')

            # Setelah semua prediksi selesai, tampilkan hasil di masing-masing tab
            tab1, tab2, tab3, tab4 = st.tabs(["Close", "Open", "High", "Low"])

            with tab1:
                st.subheader("Prediksi Harga Close")
                display_results(future_dates, forecast_results['Close'], 'Close', historical_close_prices, historical_dates)

            with tab2:
                st.subheader("Prediksi Harga Open")
                display_results(future_dates, forecast_results['Open'], 'Open', historical_open_prices, historical_dates)

            with tab3:
                st.subheader("Prediksi Harga High")
                display_results(future_dates, forecast_results['High'], 'High', historical_high_prices, historical_dates)

            with tab4:
                st.subheader("Prediksi Harga Low")
                display_results(future_dates, forecast_results['Low'], 'Low', historical_low_prices, historical_dates)

def predict_prices(prices, future_dates, forecast_steps, price_type):
    # TES Model (Holt-Winters Exponential Smoothing)
    model_tes = ExponentialSmoothing(prices, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast_future_tes = model_tes.forecast(steps=forecast_steps)

    # Auto ARIMA model
    model_arima = auto_arima(prices, 
                        start_p=0, max_p=5,  # Batasi p antara 0-5
                        start_q=0, max_q=5,  # Batasi q antara 0-5
                        d=None,              # Biarkan Auto ARIMA menentukan d
                        seasonal=False,      # Tidak menggunakan komponen seasonal
                        stepwise=True,       # Untuk mempercepat pencarian dengan pendekatan stepwise
                        trace=False)         # Menghilangkan log output
    forecast_future_arima = model_arima.predict(n_periods=forecast_steps)

    return {
        'TES Prediction': forecast_future_tes,
        'ARIMA Prediction': forecast_future_arima
    }

def display_results(dates, forecast_result, price_type, real_data=None, real_dates=None):
    # Visualizations and Results
    visualize_future_predictions(dates, forecast_result['TES Prediction'], forecast_result['ARIMA Prediction'], price_type, real_data, real_dates)
    display_future_table(dates, forecast_result['TES Prediction'], forecast_result['ARIMA Prediction'], price_type)

def visualize_future_predictions(dates, y_pred_tes, y_pred_arima, price_type, real_data=None, real_dates=None):
    fig = go.Figure()

    # Plot real data (historical data before the forecast)
    if real_data is not None and real_dates is not None:
        fig.add_trace(go.Scatter(x=real_dates, y=real_data, mode='lines', name=f"Real {price_type} Prices", line=dict(color='blue')))
    
    # Plot TES predictions
    fig.add_trace(go.Scatter(x=dates, y=y_pred_tes, mode='lines', name=f"TES Future Predictions - {price_type}", line=dict(color='red')))
    
    # Plot ARIMA predictions
    fig.add_trace(go.Scatter(x=dates, y=y_pred_arima, mode='lines', name=f"ARIMA Future Predictions - {price_type}", line=dict(color='green')))

    fig.update_layout(title=f"Prediksi Harga Masa Depan {price_type} untuk TES & ARIMA",
                      xaxis_title="Tanggal", 
                      yaxis_title=f"Harga {price_type} (USD)", 
                      template='plotly_dark')

    st.plotly_chart(fig)

def display_future_table(dates, y_pred_tes, y_pred_arima, price_type):
    st.write(f"Tabel Prediksi Masa Depan untuk Harga {price_type}")
    df_future = pd.DataFrame({
        'Date': dates, 
        'TES Prediction': y_pred_tes, 
        'ARIMA Prediction': y_pred_arima})
    st.table(df_future.reset_index(drop=True))

if __name__ == "__main__":
    main()
