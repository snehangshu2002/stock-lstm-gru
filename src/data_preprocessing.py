"""
Data Preprocessing Module for Stock Price Prediction

Handles data loading, cleaning, normalization, and sequence creation
for LSTM and GRU models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

    feature_data = df[feature_columns].values
    target_data = df[[target_column]].values  # Keep as 2D (n, 1)

    # Get target column index for sequence creation
    target_idx = feature_columns.index(target_column)

    # Create sequences from RAW data FIRST (before normalization)
    # targets here are the next-step values for target_column
    sequences, targets = create_sequences(
        feature_data,
        sequence_length,
        target_idx
    )

    # Keep targets as 2D for scaler compatibility
    targets = targets.reshape(-1, 1)

    # Split data FIRST (before normalization) - CRITICAL for time series!
    split_index = int(len(sequences) * (1 - test_ratio))
    
    X_train_raw = sequences[:split_index]
    X_test_raw = sequences[split_index:]
    y_train_raw = targets[:split_index]
    y_test_raw = targets[split_index:]

    feature_scaler = None
    target_scaler = None

    if normalize:
        # Get dimensions
        n_train_samples, n_timesteps, n_features = X_train_raw.shape
        n_test_samples = X_test_raw.shape[0]
        
        # Fit scaler ONLY on training data
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_reshaped = X_train_raw.reshape(-1, n_features)
        feature_scaler.fit(X_train_reshaped)
        
        # Transform both train and test using the SAME scaler (fit on train only!)
        X_train = feature_scaler.transform(X_train_reshaped).reshape(n_train_samples, n_timesteps, n_features)
        X_test = feature_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(n_test_samples, n_timesteps, n_features)
        
        # Scale targets separately (fit on train only)
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(y_train_raw)
        y_train = target_scaler.transform(y_train_raw).flatten()
        y_test = target_scaler.transform(y_test_raw).flatten()
    else:
        X_train = X_train_raw
        X_test = X_test_raw
        y_train = y_train_raw.flatten()
        y_test = y_test_raw.flatten()

    # Calculate test start index for original data
    test_start_idx = split_index
    original_test_data = target_data[test_start_idx + sequence_length:test_start_idx + sequence_length + len(y_test)]

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
