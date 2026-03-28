"""
Data Preprocessing Module for Stock Price Prediction

Handles data loading, cleaning, normalization, and sequence creation
for LSTM and GRU models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
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
    
    # Forward fill missing values
    df = df.fillna(method='ffill')
    
    # Backward fill any remaining missing values
    df = df.fillna(method='bfill')
    
    return df


def normalize_data(data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize data using MinMax scaling.
    
    Args:
        data: Array of data to normalize
        feature_range: Tuple of (min, max) for scaled data
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data, scaler


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.

    Args:
        features: Array of feature data (multiple columns)
        targets: Array of target data (single column)
        sequence_length: Number of time steps for each sequence

    Returns:
        Tuple of (sequences, targets)
    """
    sequences = []
    target_values = []

    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i + sequence_length])
        target_values.append(targets[i + sequence_length])

    return np.array(sequences), np.array(target_values)


def prepare_data(
    df: pd.DataFrame,
    feature_columns: list = None,
    target_column: str = 'Close',
    sequence_length: int = 60,
    train_ratio: float = 0.8,
    normalize: bool = True
) -> dict:
    """
    Prepare data for model training.

    Args:
        df: Stock data DataFrame
        feature_columns: List of columns to use as features (default: ['Open', 'High', 'Low', 'Close', 'Volume'])
        target_column: Column to predict (default: 'Close')
        sequence_length: Number of time steps for sequences
        train_ratio: Ratio of data for training
        normalize: Whether to normalize the data

    Returns:
        Dictionary containing training and testing data
    """
    # Set default feature columns if not provided
    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Extract feature and target columns
    feature_data = df[feature_columns].values
    target_data = df[[target_column]].values

    # Normalize data
    feature_scaler = None
    target_scaler = None
    if normalize:
        normalized_features, feature_scaler = normalize_data(feature_data)
        normalized_targets, target_scaler = normalize_data(target_data)
    else:
        normalized_features = feature_data
        normalized_targets = target_data

    # Create sequences
    sequences, targets = create_sequences(normalized_features, normalized_targets, sequence_length)

    # Split data
    split_index = int(len(sequences) * train_ratio)

    X_train = sequences[:split_index]
    y_train = targets[:split_index]
    X_test = sequences[split_index:]
    y_test = targets[split_index:]

    # Store original test data for inverse transform
    original_test_data = target_data[split_index + sequence_length:]

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
        'target_column': target_column
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
    # Example usage
    print("Data Preprocessing Module")
    print("=" * 40)

    # Test with sample data (multiple features)
    n_samples = 1000
    n_features = 5  # Open, High, Low, Close, Volume
    sample_features = np.random.randn(n_samples, n_features)
    sample_targets = np.random.randn(n_samples, 1)

    # Normalize data
    normalized_features, feature_scaler = normalize_data(sample_features)
    normalized_targets, target_scaler = normalize_data(sample_targets)

    # Create sequences
    sequences, targets = create_sequences(
        normalized_features,
        normalized_targets,
        sequence_length=60
    )

    print(f"Original features shape: {sample_features.shape}")
    print(f"Original targets shape: {sample_targets.shape}")
    print(f"Normalized features shape: {normalized_features.shape}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"\nSequence shape breakdown: (samples, timesteps, features)")
    print(f"  - Samples: {sequences.shape[0]}")
    print(f"  - Timesteps: {sequences.shape[1]}")
    print(f"  - Features: {sequences.shape[2]}")
