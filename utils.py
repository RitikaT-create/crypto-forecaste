import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 📊 Plot actual vs predicted
def plot_predictions(actual, predicted, title="Prediction vs Actual"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=actual,
        mode='lines',
        name='Actual'
    ))

    fig.add_trace(go.Scatter(
        y=predicted,
        mode='lines',
        name='Predicted'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )

    return fig


# 📉 Calculate evaluation metrics
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae
    }


# 🔄 Train-test split for time series
def train_test_split(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    train = df[:split]
    test = df[split:]
    return train, test


# 📈 Normalize data
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)

    scaled = (data - min_val) / (max_val - min_val)

    return scaled, min_val, max_val


# 🔁 Inverse normalization
def inverse_normalize(scaled_data, min_val, max_val):
    return scaled_data * (max_val - min_val) + min_val


# 📊 Add RSI Indicator
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI'] = rsi
    return df


# 📉 Add Bollinger Bands
def bollinger_bands(df, window=20):
    df['BB_Middle'] = df['Close'].rolling(window).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window).std()
    return df


# 📊 Generate Buy/Sell Signals
def generate_signals(df):
    df['Signal'] = 0

    # Buy when price < lower band
    df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = 1

    # Sell when price > upper band
    df.loc[df['Close'] > df['BB_Upper'], 'Signal'] = -1

    return df