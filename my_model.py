import os

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, InputLayer, TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

def create_model_1(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(16, return_sequences=True)),
        Bidirectional(LSTM(16, return_sequences=False)),
        Dense(25, activation='selu'),
        Dense(n_output, activation="softmax")
    ])
    return model

def create_model_1_1(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(25, activation='selu'),
        Dense(n_output, activation="softmax")
    ])
    return model

def create_model_1_2(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(n_output, activation="softmax")
    ])
    return model

def create_model_1_3(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Bidirectional(LSTM(16, return_sequences=False)),
        Dense(25, activation='selu'),
        Dense(n_output, activation="softmax")
    ])
    return model

def create_model_2(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same'),
        LSTM(32, return_sequences=False),   
        Dense(n_output, activation="softmax")
    ])
    return model

def create_model_2_1(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same'),
        MaxPooling1D(pool_size=4),
        LSTM(32, return_sequences=False),
        Dense(n_output, activation="softmax")
    ])
    return model

if __name__ == '__main__':
    input_shape = (854, 50)
    # input_shape = (1, 854, 50)
    n_output = 2
    model = create_model_2_1(input_shape, n_output)
    model.summary()
    # plot_model(
    #     model, 
    #     to_file='tmp.png', 
    #     show_shapes=True,
    #     show_layer_names=True
    # )
