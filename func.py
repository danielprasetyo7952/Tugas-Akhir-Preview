import pandas as pd
import streamlit as st

def config():
    st.set_page_config(page_title="Forex Forecasting",
                       layout="wide",
                       page_icon=":moneybag:")
    
def load_data(sheet_name):
    url = f'https://docs.google.com/spreadsheets/d/1JDNv_mArl-GPIpxuWS5GxgVEwvjXocS1MrXGc6TYs8M/gviz/tq?tqx=out:csv&sheet={sheet_name}/IDR'
    df = pd.read_csv(url)
    return df