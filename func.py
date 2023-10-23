import pandas as pd
import streamlit as st
import tensorflow as tf

def config():
    st.set_page_config(page_title="Forex Forecasting",
                       layout="wide",
                       page_icon=":currency_exchange:")
    
def load_data(sheet_name):
    url = f'https://docs.google.com/spreadsheets/d/1JDNv_mArl-GPIpxuWS5GxgVEwvjXocS1MrXGc6TYs8M/gviz/tq?tqx=out:csv&sheet={sheet_name}/IDR'
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
    return df

# def load_model():
#     model = tf.load_model()
#     return model