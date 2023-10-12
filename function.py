import pandas as pd

def load_data(sheet_name):
    url = f'https://docs.google.com/spreadsheets/d/1JDNv_mArl-GPIpxuWS5GxgVEwvjXocS1MrXGc6TYs8M/gviz/tq?tqx=out:csv&sheet={sheet_name}/IDR'
    df = pd.read_csv(url)
    return df