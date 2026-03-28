"""
Prediction Module for Stock Price Prediction

Handles loading trained models and making predictions on new data.
"""

import numpy as np
import argparse
import os
import pickle

# Import local modules
from data_preprocessing import (
    load_stock_data,
    clean_data,
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
    feature_columns: list = None,
    scaler_path: str = None
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
        scaler_path: Path to saved scaler (optional, for proper inverse transform)

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

    # Load saved scalers when available; fallback keeps backward compatibility.
    from sklearn.preprocessing import MinMaxScaler
    feature_scaler = None
    target_scaler = None

    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler_bundle = pickle.load(f)
        feature_scaler = scaler_bundle.get('feature_scaler')
        target_scaler = scaler_bundle.get('target_scaler')
        print(f"Loaded scalers from: {scaler_path}")

    if feature_scaler is None or target_scaler is None:
        print("Scaler file not found or incomplete. Falling back to fit scalers on provided data.")
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        feature_scaler.fit(feature_data)
        target_scaler.fit(target_data)

    normalized_features = feature_scaler.transform(feature_data)
    normalized_targets = target_scaler.transform(target_data)

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

            # Simple autoregressive fallback:
            # shift sequence and use predicted target as proxy for all features.
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, :] = pred_norm[0, 0]

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
    parser.add_argument(
        '--scaler-path',
        type=str,
        default=None,
        help='Path to saved scalers.pkl from training (recommended)'
    )
    
    args = parser.parse_args()
    
    # Run prediction
    results = predict(
        model_path=args.model,
        data_path=args.data,
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        target_column=args.target_column,
        predict_future=args.predict_future,
        scaler_path=args.scaler_path
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
