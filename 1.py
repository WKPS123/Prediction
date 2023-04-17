import streamlit as st
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf # https://pypi.org/project/yfinance/
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import plotly.express as px

import shap
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

### SideBar
st.sidebar.subheader('Prediction vs Acutal Result')
option = st.sidebar.selectbox('Select one symbol', ('TSLA', 'RACE','POAHY','MBGAF'))
today = datetime.date.today()
before = today - datetime.timedelta(days=1000)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')
    
data = yf.download(option, start=start_date, end=end_date)
data = data.reset_index()


# Title
if option == 'TSLA':
    st.write('<p style="font-size:50px;font-weight: bold;">Tesla</p>', unsafe_allow_html=True)
elif option == 'RACE':
    st.write('<p style="font-size:50px;font-weight: bold;">Ferrari </p>', unsafe_allow_html=True)
elif option == 'POAHY':
    st.write('<p style="font-size:50px;font-weight: bold;">Porsche</p>', unsafe_allow_html=True)
elif option == 'MBGAF':
    st.write('<p style="font-size:50px;font-weight: bold;">Mercedes-Benz</p>', unsafe_allow_html=True)

st.subheader('Original Data(Latest 10 Days)')
st.write(data.tail(10))

fig = px.line(data, x='Date', y='Close', title='Stock Price for {}'.format(option))
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Price')
st.plotly_chart(fig)


### Predict Data
data1 = data[['Date','Close']]
data1 = data1.rename(columns={"Date": "ds", "Close": "y"})

n_days = st.slider('Days of prediction:', 1, 365)
period = n_days * 1

m = Prophet()
m.fit(data1)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.write(f'<p style="font-size:26px;"> Forecast Plot for next {n_days} days', unsafe_allow_html=True)
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components (Year, Week)")
fig2 = m.plot_components(forecast)
st.write(fig2)

data = data.filter(['Close'])

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values)

# Define input and output sequences
x = []
y = []
for i in range(len(scaled_data) - n_days):
    x.append(scaled_data[i:i+n_days])
    y.append(scaled_data[i+n_days])
x = np.array(x)
y = np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(x) * split_ratio)
x_train, y_train = x[:split_index], y[:split_index]
x_test, y_test = x[split_index:], y[split_index:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_days, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train LSTM model
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

# Predict next 7 days
last_sequence = scaled_data[-n_days:]
next_sequence = []
for i in range(n_days):
    predicted_value = model.predict(np.reshape(last_sequence, (1, n_days, 1)))
    next_sequence.append(predicted_value[0][0])
    last_sequence = np.append(last_sequence[1:], predicted_value.reshape(1,1), axis=0)

# Inverse transform the predicted values to the original scale
predicted_data = scaler.inverse_transform(np.array(next_sequence).reshape(-1, 1))

# Display predicted values in Streamlit
st.subheader('LSTM Prediction')
st.write(f'Predicted data for next {n_days} days:')
st.line_chart(predicted_data)

TSLA_data = yf.Ticker("TSLA").history(period="max")
RACE_data = yf.Ticker("RACE").history(period="max")
POAHY_data = yf.Ticker("POAHY").history(period="max")
MBGAF_data = yf.Ticker("MBGAF").history(period="max")

TSLA_X = TSLA_data.drop(["Close"], axis=1)
TSLA_y = TSLA_data["Close"]
RACE_X = RACE_data.drop(["Close"], axis=1)
RACE_y = RACE_data["Close"]
POAHY_X = POAHY_data.drop(["Close"], axis=1)
POAHY_y = POAHY_data["Close"]
MBGAF_X = MBGAF_data.drop(["Close"], axis=1)
MBGAF_y = MBGAF_data["Close"]

TSLA_model = LinearRegression().fit(TSLA_X, TSLA_y)
RACE_model = LinearRegression().fit(RACE_X, RACE_y)
POAHY_model = LinearRegression().fit(POAHY_X, POAHY_y)
MBGAF_model = LinearRegression().fit(MBGAF_X, MBGAF_y)

TSLA_explainer = shap.LinearExplainer(TSLA_model, TSLA_X)
TSLA_shap_values = TSLA_explainer.shap_values(TSLA_X)
RACE_explainer = shap.LinearExplainer(RACE_model, RACE_X)
RACE_shap_values = RACE_explainer.shap_values(RACE_X)
POAHY_explainer = shap.LinearExplainer(POAHY_model, POAHY_X)
POAHY_shap_values = POAHY_explainer.shap_values(POAHY_X)
MBGAF_explainer = shap.LinearExplainer(MBGAF_model, MBGAF_X)
MBGAF_shap_values = MBGAF_explainer.shap_values(MBGAF_X)

st.title("Stock Price Interpretation")

# Create a selectbox to choose the stock
stock = option

# Select the SHAP values and feature matrix for the chosen stock
if stock == "TSLA":
    shap_values = TSLA_shap_values
    X = TSLA_X
elif stock == "RACE":
    shap_values = RACE_shap_values
    X = RACE_X
elif stock == "POAHY":
    shap_values = POAHY_shap_values
    X = POAHY_X
elif stock == "MBGAF":
    shap_values = MBGAF_shap_values
    X = MBGAF_X

# Display the summary plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)


