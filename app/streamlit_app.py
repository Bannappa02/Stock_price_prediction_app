import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import joblib
import yfinance as yf

from tensorflow.keras.models import load_model
from data_fetcher import fetch_stock_data
from preprocessing import preprocess_data
from forecast_future import forecast_future
from company_list import indian_stocks, us_stocks

# Page setup
st.set_page_config(page_title="Stock Forecast App", layout="centered")
st.title(" Real-Time Stock Price Predictor")

# Sidebar options
market = st.sidebar.radio("Select Market", ["🇮🇳 Indian Stocks", "🇺🇸 US Stocks"])
company_list = indian_stocks if market == "🇮🇳 Indian Stocks" else us_stocks
company_name = st.sidebar.selectbox("Choose Company", list(company_list.keys()))
ticker = company_list[company_name]

future_days = st.sidebar.slider("Days to Forecast", 1, 90, 30)
investment_amount = st.sidebar.number_input("If I invest (₹)", min_value=100.0, value=10000.0)

try:
    st.subheader(f" Stock Selected: {company_name} ({ticker})")
    df = fetch_stock_data(ticker)
    df.dropna(inplace=True)  # Ensure no NaNs

    #  Convert to float for formatting
    current_price = float(df['Close'].iloc[-1])
    st.metric(" Current Price", f"₹{current_price:.2f}")
      
    #   preprocess the data for model input
    X, y, scaler = preprocess_data(df)
    model = load_model("models/lstm_model.keras")

    predicted = model.predict(X)
    predicted = scaler.inverse_transform(predicted)

    # Forecast future prices
    future_prices = forecast_future(
        model_path="models/lstm_model.keras",
        scaler_path="models/scaler.save",
        recent_data=df['Close'].values,
        days_to_predict=future_days
    )

    # Prepare future date range
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(future_days)]

    # temporariry debug print
    st.write("Raw predicted future prices:", future_prices.flatten().tolist())


    # Plotting
    actual = df['Close'].values[-len(predicted):]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Date'][-len(predicted):], actual, label="Actual", color='blue')
    ax.plot(df['Date'][-len(predicted):], predicted, label="Model Fit", color='red')
    ax.plot(future_dates, future_prices, label="Forecast", color='green')
    ax.set_title(f"{company_name} Price Forecast for {future_days} Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    st.pyplot(fig)

    predicted_price = float(future_prices[-1][0])
    st.success(f"Predicted price after {future_days} days: ₹{predicted_price:.2f}")

    expected_return = (predicted_price / current_price) * investment_amount
    profit = expected_return - investment_amount
    st.info(f" Investment Outcome:")
    st.markdown(f"→ Final Value: **₹{expected_return:.2f}**")
    st.markdown(f"→ Profit/Loss: **₹{profit:.2f}**")

    # Forecast table
    forecast_table = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price (₹)": future_prices.flatten()
    })
    st.dataframe(forecast_table)

except Exception as e:
    st.error(f" Error: {e}")
