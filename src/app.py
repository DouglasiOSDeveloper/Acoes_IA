import streamlit as st

st.title("Previsão de Ações")
st.write("Insira a ação e selecione as variáveis exógenas.")

# Entrada do usuário
ticker = st.text_input("Ação", value="PETR4.SA")
variables = st.multiselect("Variáveis Exógenas", options=["PIB", "Cambio"], default=["PIB", "Cambio"])

# Botão para iniciar a previsão
if st.button("Prever"):
    st.write(f"Previsão para {ticker} com as variáveis {variables}")