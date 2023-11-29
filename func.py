import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler

def config():
    """
    This function sets the page configuration for a Streamlit app.

    Args:
        None.

    Returns:
        None. The function sets the page configuration directly.
    """
    
    # Set the page configuration
    st.set_page_config(page_title="Forex Forecasting",
                       layout="wide",
                       page_icon=":currency_exchange:")

def show_data(forex_type: str) -> pd.DataFrame:
    """
    This function fetches and visualizes foreign exchange data.

    Args:
        forex_type (str): The type of foreign exchange data to fetch and visualize.

    Returns:
        None. The function outputs a line chart of the foreign exchange data.

    Example:
        >>> show_data('USD')
        This will fetch and visualize USD to IDR foreign exchange data.
    """
    
    # Define the URL for fetching the data
    url = f'https://docs.google.com/spreadsheets/d/1JDNv_mArl-GPIpxuWS5GxgVEwvjXocS1MrXGc6TYs8M/gviz/tq?tqx=out:csv&sheet={forex_type}/IDR'
    # Fetch the data
    data = pd.read_csv(url)
    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S')
    
    # Define the chart
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            alt.X("Date"),
            alt.Y("Close",
                  scale=alt.Scale(domain=[min(data['Close']) - 100, max(data['Close']) + 100])
            )
        )
        .interactive()
    )
    
    # Display the chart
    st.subheader(forex_type)
    st.altair_chart(chart, use_container_width=True)
    return data

def to_sequences(data, seq_len):
        d = []
        for index in range(len(data) - seq_len):
            d.append(data[index: index + seq_len])
        return np.array(d)

def sliding_window(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]
    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]
    return X_train, y_train, X_test, y_test

def preprocessing(data):
    scaler = MinMaxScaler()
    close_price = data.Close.values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)
    X_train, y_train, X_test, y_test = sliding_window(scaled_close, seq_len=21, train_split=0.8)
    return scaler, X_test

def load_model(forex_type):
    model_path = f"model\{forex_type}_Model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

def predict_model(dataset, model, future_steps):
    scaler, X_test = preprocessing(data=dataset)
    
    future_steps = int(future_steps.split(' ')[0])
    last_sequence = X_test[-1]  # Get the last sequence from the testing data
    predictions = []
    
    for _ in range(future_steps):
        prediction = model.predict(np.array([last_sequence]))  # Predict the next time step
        predictions.append(prediction)
        last_sequence = np.concatenate((last_sequence[1:], prediction), axis=0)

    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    last_sequence = scaler.inverse_transform(last_sequence)
    
    predictions = pd.DataFrame(predictions, columns=['Close'])
    last_sequence = pd.DataFrame(last_sequence, columns=['Close'])
    predictions.index += len(last_sequence)
    
    # data = pd.DataFrame({
    #     'Date': range(future_steps),
    #     'Last Sequence': last_sequence['Close'],
    #     'Predictions': predictions['Close'],
    # })

    # base = alt.Chart(data).encode(
    #     alt.X('Date'),
    # )

    # line_chart = base.mark_line().encode(
    #     y='Last Sequence',
    #     color=alt.value('blue'),
    #     opacity=alt.value(0.7),
    #     strokeDash=alt.value([5, 5])
    # )

    # line_chart += base.mark_line().encode(
    #     y='Predictions',
    #     color=alt.value('green')
    # )

    # st.altair_chart(line_chart)

    # Assuming you have the following variables defined:
    # last_sequence, predictions, and actual
    
    # st.write("See Plot for Future Predictions")
    # Create a Matplotlib figure and axis
    # fig, ax = plt.subplots()

    # Plot your data
    # ax.plot(last_sequence['Close'], label="Last Sequence")
    # ax.plot(predictions['Close'], label="Predictions")
    # ax.plot(actual['Close'][:future_steps], label="Actual Close")

    # Customize the plot
    # ax.set_title("Predicted Future of {} days".format(future_steps))
    # ax.set_xlabel("Days")
    # ax.set_ylabel("Price")
    # ax.legend()

    # Display the Matplotlib plot in Streamlit
    # st.pyplot(fig)
    
    st.write(f"Harga Mata Uang {future_steps} hari ke depan")
    date = datetime.datetime(2023, 6, 1)
    for value in predictions['Close']:
        st.markdown("- " + date.strftime("%d %B %Y") + " : " + str(value))
        date += datetime.timedelta(days=1)