import streamlit as st
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    model_choice = st.sidebar.radio("Select Model:", ["Triple Exponential Smoothing (TES)", "ARIMA"])

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
    if model_choice == "Triple Exponential Smoothing (TES)":
        model_tes, alpha, beta, gamma = models_tes[stock_symbol]
        forecast_close = model_tes.forecast(steps=len(test_close))
        forecast_open = ExponentialSmoothing(train_open, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=len(test_open))
        forecast_high = ExponentialSmoothing(train_high, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=len(test_high))
        forecast_low = ExponentialSmoothing(train_low, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=len(test_low))
    else:
        model_arima, order = models_arima[stock_symbol]
        forecast_close = model_arima.forecast(steps=len(test_close))
        forecast_open = ARIMA(train_open, order=order).fit().forecast(steps=len(test_open))
        forecast_high = ARIMA(train_high, order=order).fit().forecast(steps=len(test_high))
        forecast_low = ARIMA(train_low, order=order).fit().forecast(steps=len(test_low))

    # Model Evaluation
    rmse_close = np.sqrt(mean_squared_error(test_close, forecast_close))
    mape_close = mean_absolute_percentage_error(test_close, forecast_close)

    rmse_open = np.sqrt(mean_squared_error(test_open, forecast_open))
    mape_open = mean_absolute_percentage_error(test_open, forecast_open)

    rmse_high = np.sqrt(mean_squared_error(test_high, forecast_high))
    mape_high = mean_absolute_percentage_error(test_high, forecast_high)

    rmse_low = np.sqrt(mean_squared_error(test_low, forecast_low))
    mape_low = mean_absolute_percentage_error(test_low, forecast_low)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Close Prices", "Open Prices", "High Prices", "Low Prices", "Actual Prices", "All Predicted Prices"])

    with tab1:
        st.header(f"Results Close Price {stock_symbol} for {model_choice} Model")
        st.write(f"{model_choice} - RMSE:", round(rmse_close, 5))
        st.write(f"{model_choice} - MAPE:", round(mape_close * 100, 5), "%")
        visualize_predictions(data, train_size, test_close, forecast_close, 'Close')
        display_forecast_table(f"Table {model_choice} Model - Close Predicted Prices ", data.index[train_size:], test_close, forecast_close, key='close')

    with tab2:
        st.header(f"Results Open Price {stock_symbol} for {model_choice} Model")
        st.write(f"{model_choice} - RMSE:", round(rmse_open, 5))
        st.write(f"{model_choice} - MAPE:", round(mape_open * 100, 5), "%")
        visualize_predictions(data, train_size, test_open, forecast_open, 'Open')
        display_forecast_table(f"Table {model_choice} Model - Open Predicted Prices ", data.index[train_size:], test_open, forecast_open, key='open')

    with tab3:
        st.header(f"Results High Price {stock_symbol} for {model_choice} Model")
        st.write(f"{model_choice} - RMSE:", round(rmse_high, 5))
        st.write(f"{model_choice} - MAPE:", round(mape_high * 100, 5), "%")
        visualize_predictions(data, train_size, test_high, forecast_high, 'High')
        display_forecast_table(f"Table {model_choice} Model - High Predicted Prices ", data.index[train_size:], test_high, forecast_high, key='high')

    with tab4:
        st.header(f"Results Low Price {stock_symbol} for {model_choice} Model")
        st.write(f"{model_choice} - RMSE:", round(rmse_low, 5))
        st.write(f"{model_choice} - MAPE:", round(mape_low * 100, 5), "%")
        visualize_predictions(data, train_size, test_low, forecast_low, 'Low')
        display_forecast_table(f"Table {model_choice} Model - Low Predicted Prices ", data.index[train_size:], test_low, forecast_low, key='low')
        
    with tab5:
        st.header("Actual Price History")
        st.write("Actual stock prices history for the selected cryptocurrency.")

        fig_all_prices = go.Figure()

        fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Opening Price', line=dict(color='red')))
        fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', line=dict(color='green')))
        fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price', line=dict(color='yellow')))
        fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price', line=dict(color='blue')))

        fig_all_prices.update_layout(
            title='Actual Stock Price History',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Stock Price'),
            legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        )

        st.plotly_chart(fig_all_prices)
        # Plot subplots for each individual price
        fig_subplots = make_subplots(rows=2, cols=2, subplot_titles=('Opening Price', 'Closing Price', 'Low Price', 'High Price'))

        fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Opening Price', line=dict(color='red')), row=1, col=1)
        fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', line=dict(color='green')), row=1, col=2)
        fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price', line=dict(color='yellow')), row=2, col=1)
        fig_subplots.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price', line=dict(color='blue')), row=2, col=2)

        fig_subplots.update_layout(title='Stock Price Subplots', showlegend=False)

        st.plotly_chart(fig_subplots)

        # Display combined actual data table with time information
        st.header("Table All Price")

        # Combine data and time information into one dataframe with column names
        combined_data_all_actual = pd.DataFrame({
            'Date': data.index.date,
            'Open': data['Open'],
            'Close': data['Close'],
            'High': data['High'],
            'Low': data['Low']
        })

        # Display the combined data table
        st.write("Data range:", start_date, "to", end_date)
        st.table(combined_data_all_actual.reset_index(drop=True))
        
    with tab6:
        st.header("All Predicted Prices")
        st.write("Predicted stock prices for the selected cryptocurrency.")

        # Plot all predicted prices
        fig_all_predicted = go.Figure()

        fig_all_predicted.add_trace(go.Scatter(x=data.index[train_size:], y=forecast_open, mode='lines', name='Predicted Opening Price', line=dict(color='red')))
        fig_all_predicted.add_trace(go.Scatter(x=data.index[train_size:], y=forecast_close, mode='lines', name='Predicted Closing Price', line=dict(color='green')))
        fig_all_predicted.add_trace(go.Scatter(x=data.index[train_size:], y=forecast_low, mode='lines', name='Predicted Low Price', line=dict(color='yellow')))
        fig_all_predicted.add_trace(go.Scatter(x=data.index[train_size:], y=forecast_high, mode='lines', name='Predicted High Price', line=dict(color='blue')))

        fig_all_predicted.update_layout(
            title='All Predicted Prices',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Stock Price'),
            legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        )

        st.plotly_chart(fig_all_predicted)

        # Display combined predicted data table
        st.header("Table All Predicted Prices")

        # Combine predicted data into one dataframe with column names
        combined_data_all_predicted = pd.DataFrame({
            'Date': data.index[train_size:],
            'Predicted_Open': forecast_open,
            'Predicted_Close': forecast_close,
            'Predicted_High': forecast_high,
            'Predicted_Low': forecast_low
        })

        # Display the combined data table
        st.table(combined_data_all_predicted.reset_index(drop=True))

def visualize_predictions(data, train_size, y_test, y_pred, price_type):
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

    # Add predictions
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred,
                             mode='lines',
                             name="Predicted Prices",
                             line=dict(color='red')))

    fig.update_layout(title=f"{price_type} Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)

def display_forecast_table(title, dates, actual, predicted, key):
    st.write(f"### {title}")
    
    price_difference = actual - predicted
    percentage_difference = (price_difference / actual) * 100

    combined_data = pd.DataFrame({
        'Tanggal': dates.date,
        'Actual_Prices': actual,
        'Predicted_Prices': predicted,
        'Price_Difference': price_difference.abs(),
        'Percentage_Difference': percentage_difference.abs().map("{:.2f}%".format)
    })


    # Display the combined data table
    st.write("Data range:", dates.min(), "to", dates.max())
    
    # Display the combined data table
    st.table(combined_data.reset_index(drop=True))

    average_actual_prices = combined_data['Actual_Prices'].mean()
    average_price_difference = combined_data['Price_Difference'].mean()
    average_percentage_difference = combined_data['Percentage_Difference'].str.rstrip('%').astype('float').mean()

    # Display the averages
    st.write("Average Actual Prices:", average_actual_prices)
    st.write("Average Price Difference:", average_price_difference)
    st.write("Average Percentage Difference: {:.2f}%".format(average_percentage_difference))

    # Add a download button for the CSV file
    csv = combined_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='forecast_data.csv',
        mime='text/csv',
        key=key
    )

if __name__ == "__main__":
    main()
