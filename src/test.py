import tensorflow as tf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# Verificar se o TensorFlow está configurado corretamente
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Criar dados simples
X = np.random.rand(1000, 10)
y = np.random.rand(1000, 1)

# Definir o modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X, y, epochs=5, batch_size=32, verbose=2)

print("Treinamento concluído com sucesso.")
