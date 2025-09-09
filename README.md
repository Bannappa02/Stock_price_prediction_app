# Stock Price Prediction App

A Python project that predicts future stock prices using real-time data from Yahoo Finance. Built with **LSTM (Deep Learning)** and a **Streamlit** frontend.

---

## Features

- Real-time stock selection (Indian & US stocks)  
- Predict future stock prices for N days  
- Show current price, forecast, and investment outcome  
- Interactive graphs with actual, model-fit, and forecast prices  

---

## Installation

1. Clone the repository:

``bash
git clone https://github.com/Bannappa02/Stock_price_prediction_app.git
cd Stock_price_prediction_app/app ''

2.Create a virtual environment and activate it:
  python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


3.Installation of dcependencies:
pip install -r requirments.txt

HOW TO RUN
1.Train the model:
python train_model.py

2.Run the streamlit app:
streamlit run streamlit_app.py


Notes
models/ folder contains the trained LSTM model and scaler.
You can add your own stock ticker symbols to company_list.py if needed.


 


