"""
Prediction Module for Stock Price Prediction

Handles loading trained models and making predictions on new data.
"""

import numpy as np
import argparse
import os
import pickle

from data_preprocessing import load_stock_data, clean_data


def load_model(model_path: str, model_type: str = 'lstm'):
    """
    Load a trained model from file.

    Args:
        model_path: Path to the saved model file
        model_type: 'lstm' or 'gru'

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
        model_type: 'lstm' or 'gru'
        sequence_length: Number of past time steps per sample
        target_column: Column that was predicted during training
        predict_future: Number of future days to predict (0 = test-set predictions only)
        feature_columns: Input feature column names
        scaler_path: Path to scalers.pkl saved during training (recommended)

    Returns:
        Dictionary with predictions, actual values, dates, and future predictions
    """
    print(f"\n{'='*60}")
    print(f"Making Predictions with {model_type.upper()} Model")
    print(f"{'='*60}\n")

    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, model_type)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading data...")
    df = load_stock_data(data_path)
    df = clean_data(df)

    feature_data = df[feature_columns].values
    target_data  = df[[target_column]].values

    # ── Load scalers ─────────────────────────────────────────────────
    from sklearn.preprocessing import MinMaxScaler
    feature_scaler = None
    target_scaler  = None

    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            bundle = pickle.load(f)
        feature_scaler = bundle.get('feature_scaler')
        target_scaler  = bundle.get('target_scaler')
        print(f"Loaded scalers from: {scaler_path}")

    if feature_scaler is None or target_scaler is None:
        # Fallback: fit on all available data.
        # Note: this is less accurate than using the training-time scalers.
        print("Scaler file not found. Fitting scalers on provided data (fallback).")
        feature_scaler = MinMaxScaler().fit(feature_data)
        target_scaler  = MinMaxScaler().fit(target_data)

    # ── Build sequences ───────────────────────────────────────────────
    norm_features = feature_scaler.transform(feature_data)
    norm_targets  = target_scaler.transform(target_data)

    sequences, targets = [], []
    for i in range(len(norm_features) - sequence_length):
        sequences.append(norm_features[i : i + sequence_length])
        targets.append(norm_targets[i + sequence_length])

    sequences = np.array(sequences)
    targets   = np.array(targets)

    # ── Predict on existing data ──────────────────────────────────────
    print("Making predictions...")
    predictions_norm = model.predict(sequences, verbose=0)
    predictions = target_scaler.inverse_transform(predictions_norm)

    actual = target_data[sequence_length : sequence_length + len(predictions)]
    dates  = df['Date'].values[sequence_length : sequence_length + len(predictions)]

    # ── Predict future days ───────────────────────────────────────────
    future_predictions = []
    if predict_future > 0:
        print(f"Predicting {predict_future} future days...")
        # Take the last window from the normalised data as the starting point
        last_window = norm_features[-sequence_length:].reshape(1, sequence_length, len(feature_columns))

        for _ in range(predict_future):
            pred_norm = model.predict(last_window, verbose=0)
            future_predictions.append(pred_norm[0, 0])

            # Slide the window forward by one step.
            # We use the predicted Close value as a proxy for ALL features in
            # the new time step. This is a rough approximation — accuracy
            # degrades quickly over multiple future steps.
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1, :] = pred_norm[0, 0]

        future_predictions = target_scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()

    print(f"\nPredictions generated: {len(predictions)}")

    return {
        'predictions':        predictions.flatten(),
        'actual':             actual.flatten(),
        'dates':              dates,
        'future_predictions': future_predictions,
        'model':              model,
        'target_scaler':      target_scaler,
        'feature_scaler':     feature_scaler,
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Make predictions using a trained LSTM or GRU model'
    )
    parser.add_argument('--model',           '-m', type=str, required=True)
    parser.add_argument('--model-type',      '-t', type=str, default='lstm', choices=['lstm', 'gru'])
    parser.add_argument('--data',            '-d', type=str, default='../data/stock_data.csv')
    parser.add_argument('--sequence-length', '-s', type=int, default=60)
    parser.add_argument('--target-column',   '-c', type=str, default='Close')
    parser.add_argument('--predict-future',  '-f', type=int, default=0)
    parser.add_argument('--scaler-path',           type=str, default=None,
                        help='Path to scalers.pkl saved during training (recommended)')

    args = parser.parse_args()

    results = predict(
        model_path=args.model,
        data_path=args.data,
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        target_column=args.target_column,
        predict_future=args.predict_future,
        scaler_path=args.scaler_path
    )

    print("\n" + "="*60)
    print("Sample Predictions (first 10):")
    print("="*60)
    for i in range(min(10, len(results['predictions']))):
        print(f"  {results['dates'][i]}: "
              f"Predicted=${results['predictions'][i]:.2f}  "
              f"Actual=${results['actual'][i]:.2f}")

    if len(results['future_predictions']) > 0:
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