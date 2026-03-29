"""
Prediction Module for Stock Price Prediction
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import argparse
import pickle

from data_preprocessing import load_stock_data, clean_data


def load_model(model_path: str, model_type: str = 'lstm'):
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

    print(f"\n{'='*60}")
    print(f"Making Predictions with {model_type.upper()} Model")
    print(f"{'='*60}\n")

    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, model_type)

    # Load data
    print("Loading data...")
    df = load_stock_data(data_path)
    df = clean_data(df)

    feature_data = df[feature_columns].values
    target_data  = df[[target_column]].values

    # Load scalers
    from sklearn.preprocessing import MinMaxScaler

    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            bundle = pickle.load(f)
        feature_scaler = bundle.get('feature_scaler')
        target_scaler  = bundle.get('target_scaler')
        print(f"Loaded scalers from: {scaler_path}")
    else:
        print("Scaler file not found. Fitting scalers on provided data (fallback).")
        feature_scaler = MinMaxScaler().fit(feature_data)
        target_scaler  = MinMaxScaler().fit(target_data)

    # Scale features
    norm_features = feature_scaler.transform(feature_data)

    # Build sequences
    sequences = []
    for i in range(len(norm_features) - sequence_length):
        sequences.append(norm_features[i : i + sequence_length])
    sequences = np.array(sequences)

    # Predict on existing data
    print("Making predictions...")
    predictions_norm = model.predict(sequences, verbose=0)
    predictions = target_scaler.inverse_transform(predictions_norm)

    actual = target_data[sequence_length : sequence_length + len(predictions)]
    dates  = df['Date'].values[sequence_length : sequence_length + len(predictions)]

    # Predict future days
    future_predictions = []

    if predict_future > 0:
        print(f"Predicting {predict_future} future days...")

        close_idx   = feature_columns.index(target_column)
        last_window = norm_features[-sequence_length:].copy()  # shape (60, 5)

        for _ in range(predict_future):

            # reshape for model input → (1, 60, 5)
            input_for_model = last_window.reshape(1, sequence_length, len(feature_columns))

            # predict next day (output is in target_scaler scale)
            pred_norm  = model.predict(input_for_model, verbose=0)

            # convert to real dollar price and save
            real_price = target_scaler.inverse_transform(pred_norm)[0, 0]
            future_predictions.append(real_price)

            # convert last row back to real values
            last_row_real = feature_scaler.inverse_transform(last_window[-1].reshape(1, -1))

            # update only the Close column with the new predicted price
            last_row_real[0, close_idx] = real_price

            # normalize the updated row back using feature_scaler
            last_row_normalized = feature_scaler.transform(last_row_real)

            # slide window: remove oldest day, add new predicted day at the end
            last_window = np.vstack([last_window[1:], last_row_normalized])

    print(f"Predictions generated: {len(predictions)}")

    return {
        'predictions':        predictions.flatten(),
        'actual':             actual.flatten(),
        'dates':              dates,
        'future_predictions': future_predictions,
    }


def main():
    parser = argparse.ArgumentParser(description='Predict stock prices using LSTM or GRU')
    parser.add_argument('--model',           '-m', type=str, required=True)
    parser.add_argument('--model-type',      '-t', type=str, default='lstm', choices=['lstm', 'gru'])
    parser.add_argument('--data',            '-d', type=str, default='../data/stock_data.csv')
    parser.add_argument('--sequence-length', '-s', type=int, default=60)
    parser.add_argument('--target-column',   '-c', type=str, default='Close')
    parser.add_argument('--predict-future',  '-f', type=int, default=0)
    parser.add_argument('--scaler-path',           type=str, default=None)
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