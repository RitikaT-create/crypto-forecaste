import streamlit as st
import plotly.graph_objects as go

from data_loader import load_crypto
from preprocessing import clean_data, add_features
from models import arima_model, prophet_model, lstm_model

st.set_page_config(page_title="Crypto Forecasting", layout="wide")

st.title("📊 Cryptocurrency Time Series Forecasting")

symbol = st.selectbox("Select Crypto", ["BTC-USD", "ETH-USD", "BNB-USD"])

df = load_crypto(symbol)
df = clean_data(df)
df = add_features(df)

# 📈 Plot price
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['MA_10'], name="MA 10"))
fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], name="MA 50"))

st.plotly_chart(fig, use_container_width=True)

# 📊 Forecast options
model_option = st.selectbox("Choose Model", ["ARIMA", "Prophet", "LSTM"])

if st.button("Predict"):

    if model_option == "ARIMA":
        forecast = arima_model(df)
        st.line_chart(forecast)

    elif model_option == "Prophet":
        forecast = prophet_model(df)
        st.line_chart(forecast[['ds','yhat']].set_index('ds'))

    elif model_option == "LSTM":
        forecast = lstm_model(df)
        st.line_chart(forecast)

st.success("Prediction Complete ✅")