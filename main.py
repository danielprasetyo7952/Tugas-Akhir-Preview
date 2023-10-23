import streamlit as st
import altair as alt
import func as f

f.config()

st.title("Prediksi Harga Beli Mata Uang Asing Terhadap Rupiah")
col1, col2 = st.columns([1, 2])

def show_data(source):
    chart = (
        alt.Chart(source)
        .mark_line()
        .encode(
            alt.X("Date"),
            alt.Y("Close",
                  scale=alt.Scale(domain=[min(source['Close']), max(source['Close'])])
            )
        )
        .interactive()
    )
    
    col2.subheader(forex_type)
    col2.altair_chart(chart, use_container_width=True)
    col2.write(f"Prediksi harga beli {forex_type} terhadap Rupiah untuk {duration} ke depan")

with col1:
    forex_type = st.selectbox("Pilih Mata Uang Asing", ["USD", "EUR", "SGD"])
    duration = st.selectbox("Pilih Durasi", ["1 Hari", "3 Hari", "5 Hari", "10 Hari", "20 Hari"])
    action = st.button("Prediksi")
    
with col2:
    data = f.load_data(forex_type)
    if (action):
        show_data(data)
    