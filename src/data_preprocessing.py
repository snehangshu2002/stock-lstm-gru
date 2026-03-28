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
    Load stock data from CSV file.
    
    Args:
        filepath: Path to the CSV file containing stock data
        
    Returns:
        DataFrame with stock data
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
    
    # Drop rows with all NaN values
    df = df.dropna(how='all')
    
    # Forward fill missing values (✅ Updated syntax)
    df = df.ffill()
    
    # Backward fill any remaining missing values (✅ Updated syntax)
    df = df.bfill()
    
    return df


def normalize_data(
    data: np.ndarray, 
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize data using MinMax scaling.
    
    Args:
        data: Array of data to normalize (can be 1D or 2D)
        feature_range: Tuple of (min, max) for scaled data
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data, scaler


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    target_column_index: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.

    Args:
        data: NumPy array of data (all columns)
        sequence_length: Number of time steps for each sequence
        target_column_index: Index of the column to predict (default: 3 for 'Close')

    Returns:
        Tuple of (sequences, targets)
    """
    sequences = []
    target_values = []

    for i in range(len(data) - sequence_length):
        
        sequences.append(data[i:i + sequence_length])
        target_values.append(data[i + sequence_length][target_column_index])

    return np.array(sequences), np.array(target_values)


def prepare_data(
    df: pd.DataFrame,
    feature_columns: List[str] = None,
    target_column: str = 'Close',
    sequence_length: int = 60,
    test_ratio: float = 0.2,
    normalize: bool = True,
) -> dict:
    """
    Prepare data for model training.
    
    IMPORTANT: Normalization is performed AFTER train/test split to prevent data leakage.
    The scaler is fit only on training data, then applied to both train and test sets.
    """
    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    feature_data = df[feature_columns].to_numpy()
    target_data = df[[target_column]].to_numpy()
    target_idx = feature_columns.index(target_column)

    X, y = create_sequences(feature_data, sequence_length, target_idx)
    y = y.reshape(-1, 1)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=test_ratio,
        shuffle=False
    )

    feature_scaler = None
    target_scaler = None

    if normalize:
        n_features = X_train_raw.shape[-1]

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = feature_scaler.fit_transform(
            X_train_raw.reshape(-1, n_features)
        ).reshape(X_train_raw.shape)
        X_test = feature_scaler.transform(
            X_test_raw.reshape(-1, n_features)
        ).reshape(X_test_raw.shape)

        target_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = target_scaler.fit_transform(y_train_raw).ravel()
        y_test = target_scaler.transform(y_test_raw).ravel()
    else:
        X_train, X_test = X_train_raw, X_test_raw
        y_train, y_test = y_train_raw.ravel(), y_test_raw.ravel()

    original_test_data = target_data[
        len(X_train_raw) + sequence_length: len(X_train_raw) + sequence_length + len(y_test)
    ]

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

def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    output_path: str
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Path to save the CSV file
        
    Returns:
        DataFrame with stock data
    """
    try:
        import yfinance as yf
        
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.reset_index()
        df.to_csv(output_path, index=False)
        
        return df
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")


if __name__ == "__main__":
   print("Works")
