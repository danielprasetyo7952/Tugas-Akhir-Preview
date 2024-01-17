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
    """Converts a sequence of data into multiple smaller sequences of a given length.

    This function takes a sequence of data and a sequence length as input. It then
    iterates over the data, creating a new sequence from each set of 'seq_len'
    consecutive elements in the original data. The resulting sequences are returned
    as a numpy array.

    Args:
        data (sequence): The original sequence of data.
        seq_len (int): The length of the sequences to be created.

    Returns:
        np.array: A numpy array containing the newly created sequences.

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6]
        >>> seq_len = 3
        >>> to_sequences(data, seq_len)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])
    """
    d = []
    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])
    return np.array(d)

def sliding_window(data_raw, seq_len, train_split):
    """Splits a sequence of data into testing sets using a sliding window approach.

    This function takes a raw sequence of data, a sequence length, and a training split ratio as input.
    It first converts the raw data into multiple smaller sequences of the given length using the 'to_sequences' function.
    It then splits these sequences into testing sets based on the 'train_split' ratio.
    The last element of each sequence is considered as the target (y), and the rest of the elements are considered as the input (X).

    Args:
        data_raw (sequence): The original sequence of raw data.
        seq_len (int): The length of the sequences to be created.
        train_split (float): The ratio of data to be used for training. Must be between 0 and 1.

    Returns:
        np.array: The input sequences for the testing set.

    Examples:
        >>> data_raw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> seq_len = 5
        >>> train_split = 0.8
        >>> sliding_window(data_raw, seq_len, train_split)
        array([[[0.55555556],
                [0.66666667],
                [0.77777778],
                [0.88888889]]])
    """
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    # X_train = data[:num_train, :-1, :]
    # y_train = data[:num_train, -1, :]
    X_test = data[num_train:, :-1, :]
    # y_test = data[num_train:, -1, :]
    return X_test

def preprocessing(data):
    """Preprocesses the given data for machine learning.

    This function takes a DataFrame 'data' as input, which should contain a 'Close' column.
    It scales the 'Close' column values using MinMaxScaler, and then splits the scaled data into training and testing sets using the 'sliding_window' function.

    Args:
        data (pandas.DataFrame): The original data. Must contain a 'Close' column.

    Returns:
        tuple: A tuple containing four elements:
            - scaler (sklearn.preprocessing.MinMaxScaler): The scaler used to scale the 'Close' column.
            - X_test (np.array): The input sequences for the testing set.

    Examples:
        >>> data = pd.DataFrame({'Close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        >>> preprocessing(data)
        (MinMaxScaler(), array([[[0.55555556],
                                [0.66666667],
                                [0.77777778],
                                [0.88888889]]]))
    """
    scaler = MinMaxScaler()
    close_price = data.Close.values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)
    X_test = sliding_window(scaled_close, seq_len=21, train_split=0.8)
    return scaler, X_test

def load_model(forex_type):
    """Loads a TensorFlow model from a specified path.

    This function takes a 'forex_type' as input, constructs a file path from it, and then loads a TensorFlow model from that path using the 'tf.keras.models.load_model' function.

    Args:
        forex_type (str): The type of forex for which the model was trained.

    Returns:
        tensorflow.python.keras.engine.training.Model: The loaded TensorFlow model.

    Examples:
        >>> forex_type = "EURUSD"
        >>> model = load_model(forex_type)
        >>> print(type(model))
        <class 'tensorflow.python.keras.engine.training.Model'>

    Raises:
        OSError: If the model file does not exist.
    """
    model_path = f"./model/{forex_type}_Model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

def predict_model(dataset, model, future_steps):
    """Predicts future time steps of a time series using a pre-trained model.

    This function takes a dataset, a pre-trained model, and a number of future steps as input.
    It first preprocesses the dataset using the 'preprocessing' function.
    It then uses the model to predict the next 'future_steps' time steps of the time series.
    The predictions are inverse transformed to bring them back to the original scale, and then returned as a DataFrame.

    Args:
        dataset (pandas.DataFrame): The original time series data.
        model (tensorflow.python.keras.engine.training.Model): The pre-trained model to be used for prediction.
        future_steps (str): The number of future time steps to predict.

    Returns:
        None: This function does not return a value. It writes the predictions to the Streamlit app.

    Examples:
        >>> dataset = pd.DataFrame({'Close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        >>> model = load_model("EURUSD")
        >>> future_steps = "5 days"
        >>> predict_model(dataset, model, future_steps)
        Harga Mata Uang 5 hari ke depan
        - 01 June 2023 : Rp 5.5
        - 02 June 2023 : Rp 6.0
        - 03 June 2023 : Rp 6.5
        - 04 June 2023 : Rp 7.0
        - 05 June 2023 : Rp 7.5
    """
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
    
    st.subheader(f"Harga Mata Uang {future_steps} hari ke depan")
    date = datetime.datetime(2023, 6, 1)
    
    col1, col2 = st.columns([2, 2])
    first_half = predictions['Close'][:int(future_steps/2)]
    second_half = predictions['Close'][int(future_steps/2):]
    
    with col1:
        for value in first_half:
            st.markdown("- " + date.strftime("%d %B %Y") + " : **Rp {:0,.3f}".format(value) + "**")
            date += datetime.timedelta(days=1)
    with col2:
        for value in second_half:
            st.markdown("- " + date.strftime("%d %B %Y") + " : **Rp {:0,.3f}".format(value) + "**")
            date += datetime.timedelta(days=1)
        
    
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