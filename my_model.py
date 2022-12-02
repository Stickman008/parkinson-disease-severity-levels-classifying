import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, InputLayer, TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.layers import concatenate

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

def create_model_1_0(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        LSTM(32, return_sequences=True),
        LSTM(16, return_sequences=True),
        LSTM(16, return_sequences=False),
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
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(25, activation='selu'),
        Dropout(0.25),
        Dense(n_output, activation="softmax")
    ])
    return model

def create_model_1_4(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(25, activation='selu'),
        Dense(n_output, activation="softmax")
    ])
    return model

def create_model_1_5(input_shape, n_output):
    model = Sequential([
        InputLayer(input_shape),
        Bidirectional(LSTM(50, return_sequences=True)),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dense(25, activation= 'selu'),
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
        Conv1D(filters=64,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same'),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(n_output, activation="softmax")
    ])
    return model

def create_model_2_2(input_shape, n_output):
    input_layer = keras.layers.Input(input_shape)

    conv1 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)

    conv2 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)

    conv3 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)

    gap = layers.GlobalAveragePooling1D()(conv3)

    output_layer = layers.Dense(n_output, activation="softmax")(gap)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name="mnist_model")
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def create_model_3(
    input_shape,
    n_output,
    head_size=128,
    num_heads=2,
    ff_dim=2,
    num_transformer_blocks=2,
    mlp_units=[128],
    dropout=0.25,
    mlp_dropout=0.4
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_output, activation="softmax")(x)
    return keras.Model(inputs, outputs)

if __name__ == '__main__':
    input_shape = (854, 50)
    # input_shape = (1, 854, 50)
    n_output = 2
    model = create_model_3(input_shape, n_output)
    model.summary()
    # plot_model(
    #     model, 
    #     to_file='tmp.png', 
    #     show_shapes=True,
    #     show_layer_names=True
    # )
