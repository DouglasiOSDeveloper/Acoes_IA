import streamlit as st
import matplotlib.pyplot as plt
from main import train_and_predict

# Título da aplicação
st.title("Previsão de Ações")
st.write("Insira a ação e selecione as variáveis exógenas.")

# Entrada do usuário
ticker = st.text_input("Ação", value="PETR4.SA")
variables = st.multiselect("Variáveis Exógenas", options=["PIB", "Cambio"], default=["PIB", "Cambio"])

look_back = 10

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

