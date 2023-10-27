import streamlit as st
import func as f

f.config()

st.title("Prediksi Harga Beli Mata Uang Asing Terhadap Rupiah")
col1, col2 = st.columns([1, 4])

with col1:
    forex_type = st.selectbox("Pilih Mata Uang Asing", ["USD", "EUR", "SGD"])
    duration = st.selectbox("Pilih Durasi", ["1 hari", "3 hari", "5 hari", "10 hari", "20 hari"])
    action = st.button("Prediksi")
    
with col2:
    if (action):
        data = f.show_data(forex_type=forex_type)
        model = f.load_model(forex_type=forex_type)
        f.predict_model(dataset=data, model=model, future_steps=duration)