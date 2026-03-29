"""
Data Preprocessing Module for Stock Price Prediction

Handles data loading, cleaning, normalization, and sequence creation
for LSTM and GRU models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import os


def load_stock_data(filepath: str) -> pd.DataFrame:
    """
    Load stock data from a CSV file.

    Args:
        filepath: Path to the CSV file containing stock data

    Returns:
        DataFrame sorted by Date
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean stock data by handling missing values.

    Args:
        df: Raw stock data DataFrame

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    df = df.dropna(how='all')   # drop rows that are entirely empty
    df = df.ffill()              # fill gaps using the previous known value
    df = df.bfill()              # fill any remaining gaps at the start
    return df


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    target_column_index: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice the data into overlapping windows for time-series training.

    Each window of `sequence_length` rows becomes one input sample (X),
    and the value in the very next row at `target_column_index` becomes
    the label (y) for that sample.

    Args:
        data: 2-D NumPy array, shape (n_rows, n_features)
        sequence_length: Number of time steps per input window
        target_column_index: Column index to predict (default: 3 = Close)

    Returns:
        X: shape (n_samples, sequence_length, n_features)
        y: shape (n_samples,)
    """
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        sequences.append(data[i : i + sequence_length])
        targets.append(data[i + sequence_length][target_column_index])

    return np.array(sequences), np.array(targets)


def prepare_data(
    df: pd.DataFrame,
    feature_columns: List[str] = None,
    target_column: str = 'Close',
    sequence_length: int = 60,
    test_ratio: float = 0.2,
    normalize: bool = True,
) -> dict:
    """
    Full pipeline: extract features → create sequences → split → normalize.

    IMPORTANT: Scalers are fit ONLY on the training portion and then applied
    to the test portion. This prevents data leakage from the future into
    the training process.

    Args:
        df: Cleaned stock DataFrame
        feature_columns: Columns to use as input features
        target_column: Column to predict
        sequence_length: Number of past time steps per sample
        test_ratio: Fraction of data to hold out for testing
        normalize: Whether to apply MinMax scaling

    Returns:
        Dictionary with train/test arrays, scalers, and metadata
    """
    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    target_idx = feature_columns.index(target_column)  # e.g. 3 for 'Close'

    feature_data = df[feature_columns].to_numpy()
    target_data = df[[target_column]].to_numpy()

    # Build (X, y) pairs from sliding windows
    X, y = create_sequences(feature_data, sequence_length, target_idx)
    y = y.reshape(-1, 1)

    # Chronological split — no shuffling for time series
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )

    feature_scaler = None
    target_scaler = None

    if normalize:
        n_features = X_train_raw.shape[-1]

        # Fit on training data only, then transform both splits
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = feature_scaler.fit_transform(
            X_train_raw.reshape(-1, n_features)  
        ).reshape(X_train_raw.shape)
        X_test = feature_scaler.transform(
            X_test_raw.reshape(-1, n_features)
        ).reshape(X_test_raw.shape)
        '''
        The problem: MinMaxScaler only accepts 2D arrays
        MinMaxScaler expects input of shape (n_samples, n_features) — exactly 2 dimensions. But X_train_raw has 3 dimensions:
        X_train_raw.shape → (n_samples, sequence_length, n_features)
                            e.g. (800, 60, 5)
        So if you pass it directly to fit_transform, it will throw an error.
        '''

        target_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = target_scaler.fit_transform(y_train_raw).ravel()
        y_test = target_scaler.transform(y_test_raw).ravel()
    else:
        X_train, X_test = X_train_raw, X_test_raw
        y_train, y_test = y_train_raw.ravel(), y_test_raw.ravel()

    # Slice the original (un-scaled) target values that align with the test set.
    # The first valid target sits at index (sequence_length) in target_data,
    # and the test targets start after all training samples.
    test_start = len(X_train_raw) + sequence_length
    original_test_data = target_data[test_start : test_start + len(y_test_raw)]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'original_test_data': original_test_data,
        'sequence_length': sequence_length,
        'feature_columns': feature_columns,
        'target_column': target_column,
        'test_ratio': test_ratio,
    }


if __name__ == "__main__":
    print("Works")