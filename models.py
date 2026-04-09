from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ARIMA
def arima_model(df):
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

# Prophet
def prophet_model(df):
    df_prophet = df.reset_index()
    df_prophet.columns = ['ds', 'y']

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast

# LSTM
def lstm_model(df):
    data = df['Close'].values.reshape(-1,1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32)

    prediction = model.predict(X[-30:])
    prediction = scaler.inverse_transform(prediction)

    return prediction