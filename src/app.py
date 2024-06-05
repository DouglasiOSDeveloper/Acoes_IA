import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import streamlit as st
from datetime import datetime
import time 

st.title("Previsão de Ações")
st.write("Insira a ação e selecione as variáveis exógenas.")

# Entrada do usuário
ticker = st.text_input("Ação", value="PETR4.SA")
variables = st.multiselect("Variáveis Exógenas", options=["PIB", "Cambio"], default=["PIB", "Cambio"])

def add_exogenous_variables(data, variables):
    for var in variables:
        if var == 'PIB':
            data['PIB'] = np.random.normal(loc=10000, scale=100, size=len(data))
        elif var == 'Cambio':
            data['Cambio'] = np.random.normal(loc=5, scale=0.1, size=len(data))
    return data

def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_and_predict(ticker, variables, look_back):
    today = pd.Timestamp.now()
    start_date = today - pd.DateOffset(months=3)
    
    # Log: Data de início e término
    print(f"Coletando dados de {start_date} até {today}")
    
    data = yf.download(ticker, start=start_date, end=today)['Close']
    
    # Usar apenas um subconjunto dos dados
    data = data.head(60)
    
    # Log: Verificando os dados baixados
    print("Dados coletados:")
    print(data.head())
    
    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = add_exogenous_variables(data, variables)

    # Log: Verificando os dados após adicionar variáveis exógenas
    print("Dados após adicionar variáveis exógenas:")
    print(data.head())

    # Verificar dados para NaN ou infinitos
    print("Verificando dados para NaN ou infinitos:")
    print(data.isna().sum())
    print(np.isinf(data).sum())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = create_dataset(scaled_data, look_back)
    
    # Log: Verificando as dimensões dos conjuntos de dados
    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))

    # Usando regressão linear simples para previsão
    model = LinearRegression()
    model.fit(X, y)

    # Log: Treinamento concluído
    print("Treinamento concluído.")

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((predictions.shape[0], data.shape[1] - 1)))))

    last_samples = scaled_data[-look_back:]
    last_samples = np.reshape(last_samples, (1, -1))

    num_days = 7  # Prevendo para os próximos 7 dias
    future_predictions = []
    current_input = last_samples

    for _ in range(num_days):
        next_prediction = model.predict(current_input)
        future_predictions.append(next_prediction[0])
        next_input = np.concatenate((current_input[:, data.shape[1]:], next_prediction.reshape(1, -1)), axis=1)
        current_input = np.concatenate((next_input, np.zeros((1, 2))), axis=1)  # Garantindo que o número de características seja correto

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(np.column_stack((future_predictions, np.zeros((future_predictions.shape[0], 2)))))

    future_dates = pd.date_range(data.index[-1], periods=num_days + 1, freq='D')[1:]

    # Log: Previsões futuras
    print("Previsões futuras:")
    print(future_predictions)

    return data, predictions, future_dates, future_predictions, look_back

look_back = 30

if st.button("Prever"):
    data, predictions, future_dates, future_predictions, look_back = train_and_predict(ticker, variables, look_back)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[look_back + 1:], data['Close'][look_back + 1:], label='Original PETR4 Data')
    plt.plot(data.index[look_back + 1:], predictions[:, 0], label='Predicted PETR4 Price', linestyle='-')
    plt.plot(future_dates, future_predictions[:, 0], label='Forecasted PETR4 Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)