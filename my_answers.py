import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model

def window_transform_series(series: 'array', window_size: 'int'):
    """Purpose: cut the time series into x and y blocks
    """
    
    X, y = [], []
    
    for i in range(len(series)-window_size):
        temp = series[i:(i+window_size)]
        temp = np.array(temp)
        X.append(temp)
        y.append(series[i+window_size])
    
    return np.array(X), np.array(y)

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):

    LSTM_model = Sequential()
    LSTM_model.add(LSTM(5, input_shape=(window_size, 1)))
    LSTM_model.add(Dense(1))
    return LSTM_model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    strange_characters = ['à', 'â']
    strange_characters_2 = ['è', 'é']

    allowed_characters = {'a','b','c', 'd', 'e', 'f', 'g', 'h',
                         'i', 'j', 'k', 'l', 'm', 'n', 'o',
                         'p', 'q', 'r', 's',
                         't', 'u', 'v', 'w', 'x', 'y',
                         'z',' ', '!', ',', '.', ':', ';', '?'}

    for j in strange_characters:
        text = text.replace(str(j), 'a')

    for j in strange_characters_2:
        text = text.replace(j, 'e')

    # remove columns
    to_remove = set(text) - allowed_characters

    for j in to_remove:
        text = text.replace(j, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text)-window_size, step_size):
        inputs.append(text[i:(i+window_size)])
        outputs.append(text[i+window_size]) 

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(200, input_shape=(window_size, num_chars)))
    LSTM_model.add(Dense(num_chars, activation='softmax'))
    return LSTM_model
