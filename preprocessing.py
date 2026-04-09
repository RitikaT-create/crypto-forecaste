import pandas as pd

def clean_data(df):
    df = df[['Date', 'Close']]
    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)
    return df

def add_features(df):
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    return df