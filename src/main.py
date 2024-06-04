import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from matplotlib.dates import DateFormatter

# Função para adicionar variáveis exógenas
def add_exogenous_variables(data, variables):
    for var in variables:
        if var == 'PIB':
            # Exemplo de como adicionar uma série temporal fictícia de PIB
            data['PIB'] = np.random.normal(loc=10000, scale=100, size=len(data))
        elif var == 'Cambio':
            # Exemplo de como adicionar uma série temporal fictícia de Câmbio
            data['Cambio'] = np.random.normal(loc=5, scale=0.1, size=len(data))
    return data

# Configurações iniciais
today = pd.Timestamp.now()
tickers = ['PETR4.SA', 'CL=F']
data = yf.download(tickers, start=today - pd.DateOffset(years=5), end=today)['Close']
data = data.resample('M').last().dropna()  # Usando o último preço de cada mês

# Supondo que o usuário escolhe PIB e Câmbio
selected_variables = ['PIB', 'Cambio']
data = add_exogenous_variables(data, selected_variables)

# Decomposição de séries temporais
series = data['PETR4.SA']
result = seasonal_decompose(series, model='additive', period=12)
decomposed = result.plot()
plt.show()

# Normalização dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Função para criar o conjunto de dados
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])  # Previsão do preço da PETR4
    return np.array(X), np.array(Y)

look_back = 3
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))  # Ajustando a forma para incluir todas as features

# Validação cruzada em séries temporais
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
train_scores, test_scores = [], []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_score = mean_squared_error(y_train, train_pred)
    test_score = mean_squared_error(y_test, test_pred)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"Train Score for fold {len(train_scores)}: {train_score}")
    print(f"Test Score for fold {len(test_scores)}: {test_score}")
print(f"Average Train Score: {np.mean(train_scores)}")
print(f"Average Test Score: {np.mean(test_scores)}")

model = Sequential([
    InputLayer(input_shape=(look_back, X.shape[2])),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=1, verbose=1)

predictions = model.predict(X)
predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((predictions.shape[0], data.shape[1]-1)))))

last_samples = scaled_data[-look_back:]
last_samples = np.expand_dims(last_samples, axis=0)

num_months = 3
future_predictions = []
current_input = last_samples

for _ in range(num_months):
    next_prediction = model.predict(current_input)
    future_predictions.append(next_prediction[0, 0])
    next_prediction = np.expand_dims(next_prediction, axis=0)
    next_prediction = np.repeat(next_prediction, data.shape[1], axis=2)
    current_input = np.append(current_input[:, 1:, :], next_prediction, axis=1)

future_predictions = np.array(future_predictions)
future_predictions = scaler.inverse_transform(np.column_stack((future_predictions, np.zeros_like(future_predictions))))

future_dates = pd.date_range(data.index[-1], periods=num_months + 1, freq='M')[1:]
plt.figure(figsize=(12, 6))
plt.plot(data.index[look_back+1:], data['PETR4.SA'][look_back+1:], label='Original PETR4 Data')
plt.plot(data.index[look_back+1:], predictions[:, 0], label='Predicted PETR4 Price', linestyle='-')
plt.plot(future_dates, future_predictions[:, 0], label='Forecasted PETR4 Price', color='red')
plt.plot(data.index, data['CL=F'], label='Oil Price', color='green')
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.title('PETR4 Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()