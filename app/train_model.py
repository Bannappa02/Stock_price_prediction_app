from data_fetcher import fetch_stock_data
from preprocessing import preprocess_data
from model import build_lstm_model
import joblib
import os
import yfinance as yf
from datetime import datetime
from tensorflow.keras.models import load_model

# Choose ticker symbol
ticker = "TCS.NS"  # You can replace this with any other supported stock symbol

# Download historical stock data from Yahoo Finance
df = yf.download(ticker, start="2015-01-01", end=datetime.today().strftime('%Y-%m-%d'))

# Preprocess the data
X, y, scaler = preprocess_data(df)

# Build and train the LSTM model
model = build_lstm_model((X.shape[1], X.shape[2]))
model.fit(X, y, epochs=10, batch_size=16, verbose=1)

# Save model and scaler
# Save model and scaler inside app/models/
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")
joblib.dump(scaler, "models/scaler.save")

print(" Model trained and saved using Yahoo Finance data.")
