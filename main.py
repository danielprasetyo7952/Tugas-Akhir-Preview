import streamlit as st
import function as f

st.title("Prediksi Harga Beli Mata Uang Asing Terhadap Rupiah")  

col1, col2 = st.columns(2)
with col1:
    forex_type = st.selectbox("Pilih Mata Uang Asing", ["USD", "EUR", "SGD"])
    duration = st.selectbox("Pilih Durasi", ["1 Hari", "3 Hari", "5 Hari", "1 Minggu", "2 Minggu", "1 Bulan"])

with col2:
    data = f.load_data(forex_type)
    st.line_chart(data['Close'])