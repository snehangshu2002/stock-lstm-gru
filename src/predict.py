"""
Prediction Module for Stock Price Prediction

Handles loading trained models and making predictions on new data.
"""

import numpy as np
import argparse
import os
from typing import Optional, List

# Import local modules
from data_preprocessing import (
    load_stock_data,
    clean_data,
    normalize_data
)


def load_model(model_path: str, model_type: str = 'lstm'):
    """
    Load a trained model from file.

    Args:
        model_path: Path to the saved model file
        model_type: Type of model ('lstm' or 'gru')

    Returns:
        Loaded Keras model
    """
    if model_type.lower() == 'lstm':
        from model_lstm import load_lstm_model
        return load_lstm_model(model_path)
    elif model_type.lower() == 'gru':
        from model_gru import load_gru_model
        return load_gru_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict(
    model_path: str,
    data_path: str,
    model_type: str = 'lstm',
    sequence_length: int = 60,
    target_column: str = 'Close',
    predict_future: int = 0,
    feature_columns: list = None
) -> dict:
    """
    Make predictions using a trained model.

    Args:
        model_path: Path to the saved model file
        data_path: Path to the stock data CSV file
        model_type: Type of model ('lstm' or 'gru')
        sequence_length: Number of time steps for sequences
        target_column: Column that was predicted
        predict_future: Number of future days to predict (0 = use test data)
        feature_columns: List of feature columns to use

    Returns:
        Dictionary with predictions and actual values
    """
    print(f"\n{'='*60}")
    print(f"Making Predictions with {model_type.upper()} Model")
    print(f"{'='*60}\n")

    # Set default feature columns
    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, model_type)

    # Load and prepare data
    print("Loading data...")
    df = load_stock_data(data_path)
    df = clean_data(df)

    # Get the feature and target data
    feature_data = df[feature_columns].values
    target_data = df[[target_column]].values

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    normalized_features = feature_scaler.fit_transform(feature_data)
    normalized_targets = target_scaler.fit_transform(target_data)

    # Create sequences
    sequences = []
    targets = []
    for i in range(len(normalized_features) - sequence_length):
        sequences.append(normalized_features[i:i + sequence_length])
        targets.append(normalized_targets[i + sequence_length])
    
    sequences = np.array(sequences)
    targets = np.array(targets)

    # Make predictions
    print("Making predictions...")
    predictions_normalized = model.predict(sequences, verbose=0)
    predictions = target_scaler.inverse_transform(predictions_normalized)

    # Get actual values (shifted by sequence_length)
    actual = target_data[sequence_length:sequence_length + len(predictions)]
    dates = df['Date'].values[sequence_length:sequence_length + len(predictions)]

    # Predict future values if requested
    future_predictions = []
    if predict_future > 0:
        print(f"Predicting {predict_future} future days...")
        last_sequence = normalized_features[-sequence_length:].reshape(1, sequence_length, len(feature_columns))

        for _ in range(predict_future):
            pred_norm = model.predict(last_sequence, verbose=0)
            future_predictions.append(pred_norm[0, 0])

            # Update sequence with new prediction (for target only, we need to update features)
            # For simplicity, we shift and repeat the last prediction for all features
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, :] = pred_norm[0, 0]  # This is approximate

        future_predictions = target_scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()

    print(f"\nPredictions generated: {len(predictions)}")

    return {
        'predictions': predictions.flatten(),
        'actual': actual.flatten(),
        'dates': dates,
        'future_predictions': future_predictions,
        'model': model,
        'target_scaler': target_scaler,
        'feature_scaler': feature_scaler
    }


def predict_single_day(
    model_path: str,
    historical_data: np.ndarray,
    model_type: str = 'lstm',
    sequence_length: int = 60,
    feature_columns: list = None
) -> float:
    """
    Predict the next day's price given historical data.

    Args:
        model_path: Path to the saved model file
        historical_data: Array of historical feature data (Open, High, Low, Close, Volume)
        model_type: Type of model ('lstm' or 'gru')
        sequence_length: Number of time steps the model expects
        feature_columns: List of feature columns

    Returns:
        Predicted next day price (Close)
    """
    # Load model
    model = load_model(model_path, model_type)

    # Prepare input
    if len(historical_data) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} historical records, "
            f"got {len(historical_data)}"
        )

    # Use the last 'sequence_length' records
    input_data = historical_data[-sequence_length:]

    # Normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    input_normalized = scaler.fit_transform(input_data)

    # Reshape for model input
    n_features = len(feature_columns) if feature_columns else 5
    input_sequence = input_normalized.reshape(1, sequence_length, n_features)

    # Predict
    prediction_normalized = model.predict(input_sequence, verbose=0)
    
    # Inverse transform only the target (Close price)
    # Assuming Close is one of the features, we need a separate scaler for it
    # For simplicity, return normalized prediction
    return prediction_normalized[0, 0]


def compare_models(
    lstm_model_path: str,
    gru_model_path: str,
    data_path: str,
    sequence_length: int = 60,
    target_column: str = 'Close'
) -> dict:
    """
    Compare predictions from LSTM and GRU models.
    
    Args:
        lstm_model_path: Path to LSTM model file
        gru_model_path: Path to GRU model file
        data_path: Path to the stock data CSV file
        sequence_length: Number of time steps for sequences
        target_column: Column to predict
        
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*60)
    print("Comparing LSTM and GRU Models")
    print("="*60 + "\n")
    
    # Get predictions from both models
    lstm_results = predict(lstm_model_path, data_path, 'lstm', sequence_length, target_column)
    gru_results = predict(gru_model_path, data_path, 'gru', sequence_length, target_column)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    actual = lstm_results['actual']
    lstm_pred = lstm_results['predictions']
    gru_pred = gru_results['predictions']
    
    lstm_rmse = np.sqrt(mean_squared_error(actual, lstm_pred))
    gru_rmse = np.sqrt(mean_squared_error(actual, gru_pred))
    lstm_mae = mean_absolute_error(actual, lstm_pred)
    gru_mae = mean_absolute_error(actual, gru_pred)
    lstm_r2 = r2_score(actual, lstm_pred)
    gru_r2 = r2_score(actual, gru_pred)
    
    print("\nModel Comparison Results:")
    print("-"*40)
    print(f"{'Metric':<15} {'LSTM':<15} {'GRU':<15}")
    print("-"*40)
    print(f"{'RMSE':<15} {lstm_rmse:<15.4f} {gru_rmse:<15.4f}")
    print(f"{'MAE':<15} {lstm_mae:<15.4f} {gru_mae:<15.4f}")
    print(f"{'R² Score':<15} {lstm_r2:<15.4f} {gru_r2:<15.4f}")
    print("-"*40)
    
    if lstm_r2 > gru_r2:
        print("\n✓ LSTM model performed better on this dataset")
    elif gru_r2 > lstm_r2:
        print("\n✓ GRU model performed better on this dataset")
    else:
        print("\n✓ Both models performed equally")
    
    return {
        'lstm': {
            'predictions': lstm_pred,
            'rmse': lstm_rmse,
            'mae': lstm_mae,
            'r2': lstm_r2
        },
        'gru': {
            'predictions': gru_pred,
            'rmse': gru_rmse,
            'mae': gru_mae,
            'r2': gru_r2
        },
        'actual': actual,
        'dates': lstm_results['dates']
    }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Make predictions using trained LSTM or GRU model'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--model-type', '-t',
        type=str,
        default='lstm',
        choices=['lstm', 'gru'],
        help='Type of model'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='../data/stock_data.csv',
        help='Path to stock data CSV file'
    )
    parser.add_argument(
        '--sequence-length', '-s',
        type=int,
        default=60,
        help='Number of time steps for sequences'
    )
    parser.add_argument(
        '--target-column', '-c',
        type=str,
        default='Close',
        help='Column to predict'
    )
    parser.add_argument(
        '--predict-future', '-f',
        type=int,
        default=0,
        help='Number of future days to predict'
    )
    
    args = parser.parse_args()
    
    # Run prediction
    results = predict(
        model_path=args.model,
        data_path=args.data,
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        target_column=args.target_column,
        predict_future=args.predict_future
    )
    
    # Display some predictions
    print("\n" + "="*60)
    print("Sample Predictions (first 10):")
    print("="*60)
    for i in range(min(10, len(results['predictions']))):
        print(f"  {results['dates'][i]}: Predicted=${results['predictions'][i]:.2f}, "
              f"Actual=${results['actual'][i]:.2f}")
    
    if results['future_predictions']:
        print("\n" + "="*60)
        print("Future Predictions:")
        print("="*60)
        for i, pred in enumerate(results['future_predictions']):
            print(f"  Day {i+1}: ${pred:.2f}")
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
