"""LSTM model for water level and sediment prediction."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional


def build_lstm_model(input_shape: Tuple[int, int],
                     hidden_units: int = 64,
                     dropout_rate: float = 0.2,
                     output_units: int = 2,
                     learning_rate: float = 0.001) -> tf.keras.Model:
    """Build LSTM model for time series prediction.
    
    Args:
        input_shape: (sequence_length, n_features)
        hidden_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        output_units: Number of output targets (water_level, sediment)
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(hidden_units, activation='tanh', return_sequences=True, 
             input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(hidden_units // 2, activation='tanh'),
        Dropout(dropout_rate),
        Dense(output_units)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def create_sequences(data: np.ndarray, 
                     sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series prediction.
    
    Args:
        data: Array of shape (n_samples, n_features)
        sequence_length: Length of input sequences
    
    Returns:
        X: Array of shape (n_sequences, sequence_length, n_features)
        y: Array of shape (n_sequences, n_targets)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)
