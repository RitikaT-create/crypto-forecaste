import yfinance as yf

def load_crypto(symbol="BTC-USD", start="2020-01-01", end="2025-01-01"):
    df = yf.download(symbol, start=start, end=end)
    df.reset_index(inplace=True)
    return df