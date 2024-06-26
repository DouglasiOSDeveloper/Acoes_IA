import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from statsmodels.tsa.seasonal import seasonal_decompose

# Forçar o TensorFlow a usar a CPU
tf.config.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(10, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(ticker, variables, look_back):
    today = pd.Timestamp.now()
    start_date = today - pd.DateOffset(months=6)
    data = yf.download(ticker, start=start_date, end=today)['Close']
    data = data.dropna()

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = add_exogenous_variables(data, variables)
    data = data.dropna()
    data = data[~data.isin([np.inf, -np.inf]).any(axis=1)]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = create_dataset(scaled_data, look_back)

    print(f"Data shape: {data.shape}")
    print(f"Scaled data shape: {scaled_data.shape}")
    print(f"X shape before reshape: {X.shape}")
    print(f"y shape: {y.shape}")

    if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] == 0:
        raise ValueError("O conjunto de dados criado está vazio ou com formato incorreto. Verifique os dados de entrada e o parâmetro look_back.")

    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    model = build_model((X.shape[1], X.shape[2]))

    early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1)
    checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='loss', mode='min')

    print("Iniciando treinamento do modelo...")
    start_time = time.time()
    history = model.fit(X, y, epochs=5, batch_size=32, verbose=2, callbacks=[early_stopping, checkpoint])
    end_time = time.time()
    print(f"Tempo de treinamento: {end_time - start_time} segundos")

    model.summary()

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((predictions.shape[0], data.shape[1] - 1)))))

    last_samples = scaled_data[-look_back:]
    last_samples = np.reshape(last_samples, (1, last_samples.shape[0], last_samples.shape[1]))

    num_days = 7  # Prevendo para os próximos 7 dias
    future_predictions = []

    for _ in range(num_days):
        next_prediction = model.predict(last_samples)
        future_predictions.append(next_prediction[0])
        next_input = np.concatenate((last_samples[:, 1:, :], next_prediction.reshape(1, 1, -1)), axis=1)
        last_samples = next_input

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(np.column_stack((future_predictions, np.zeros((future_predictions.shape[0], data.shape[1] - 1)))))

    future_dates = pd.date_range(data.index[-1], periods=num_days + 1, freq='D')[1:]

    return data, predictions, future_dates, future_predictions, look_back

if __name__ == "__main__":
    look_back = 10
    ticker = 'PETR4.SA'
    selected_variables = ['PIB', 'Cambio']
    data, predictions, future_dates, future_predictions, look_back = train_and_predict(ticker, selected_variables, look_back)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[look_back + 1:], data['Close'][look_back + 1:], label='Original PETR4 Data')
    plt.plot(data.index[look_back + 1:], predictions[:, 0], label='Predicted PETR4 Price', linestyle='-')
    plt.plot(future_dates, future_predictions[:, 0], label='Forecasted PETR4 Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
