import yfinance as yf

def fetch_stock_data(ticker='TCS.NS', start='2015-01-01', end='2024-12-31'):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data
