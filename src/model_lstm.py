"""
LSTM Model Module for Stock Price Prediction

Implements Long Short-Term Memory (LSTM) neural network
for time series prediction.
"""

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from typing import Optional, Tuple
import os


def create_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: list = [50, 50],
    dropout_rate: float = 0.2,
    dense_units: int = 50,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Create an LSTM model for stock price prediction.

    Args:
        input_shape: Shape of input data (sequence_length, num_features)
        lstm_units: List of units for each LSTM layer
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in the dense layer
        learning_rate: Learning rate for the Adam optimizer

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # FIX: Single loop handles all layers correctly.
    # All layers except the last use return_sequences=True so that each
    # LSTM layer receives the full sequence from the one before it.
    # The last layer uses return_sequences=False to output a single vector.
    for idx, units in enumerate(lstm_units):
        return_seq = (idx < len(lstm_units) - 1)
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    # Dense output layers
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))

    # FIX: Actually use the learning_rate parameter instead of ignoring it
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )

    return model


def train_lstm_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    model_save_path: Optional[str] = None,
    patience: int = 10,
    verbose: int = 1
) -> dict:
    """
    Train the LSTM model.

    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_save_path: Path to save the best model
        patience: Early stopping patience
        verbose: Verbosity level

    Returns:
        Training history
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=verbose
        )
    ]

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=verbose
            )
        )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False
    )

    return history


def load_lstm_model(model_path: str) -> Sequential:
    """
    Load a trained LSTM model from file.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return load_model(model_path)


# def predict_lstm(
#     model: Sequential,
#     X: np.ndarray,
#     scaler: Optional = None
# ) -> np.ndarray:
#     """
#     Make predictions using the LSTM model.

#     Args:
#         model: Trained LSTM model
#         X: Input data
#         scaler: Scaler for inverse transform (optional)

#     Returns:
#         Predictions array
#     """
#     predictions = model.predict(X, verbose=0)

#     if scaler is not None:
#         predictions = scaler.inverse_transform(predictions)

#     return predictions


def evaluate_lstm_model(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: Optional = None
) -> dict:
    """
    Evaluate the LSTM model on test data.

    Args:
        model: Trained LSTM model
        X_test: Test features
        y_test: Test targets
        target_scaler: Scaler for inverse transform (optional)

    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    predictions = model.predict(X_test, verbose=0)

    if target_scaler is not None:
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_original = target_scaler.inverse_transform(predictions)
    else:
        y_test_original = y_test.reshape(-1, 1)
        predictions_original = predictions

    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions_original,
        'actual': y_test_original
    }


if __name__ == "__main__":
    print("LSTM Model Module")
    print("=" * 40)
    input_shape = (60, 5)  # 60 timesteps, 5 features
    model = create_lstm_model(input_shape)
    model.summary()