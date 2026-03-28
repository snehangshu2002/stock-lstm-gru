"""
Training Module for Stock Price Prediction

Handles training of both LSTM and GRU models with configurable parameters.
"""

import numpy as np
import argparse
import os
import json
import pickle
from typing import Optional

# Import local modules
from data_preprocessing import (
    load_stock_data,
    clean_data,
    prepare_data
)
from model_lstm import create_lstm_model, train_lstm_model
from model_gru import create_gru_model, train_gru_model


def train(
    data_path: str,
    model_type: str = 'lstm',
    model_save_path: str = '../models/',
    sequence_length: int = 60,
    train_ratio: float = 0.8,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    target_column: str = 'Close',
    verbose: int = 1
) -> dict:
    """
    Train a stock price prediction model.
    
    Args:
        data_path: Path to the stock data CSV file
        model_type: Type of model ('lstm' or 'gru')
        model_save_path: Directory to save the trained model
        sequence_length: Number of time steps for sequences
        train_ratio: Ratio of data for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        patience: Early stopping patience
        target_column: Column to predict
        verbose: Verbosity level
        
    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model for Stock Price Prediction")
    print(f"{'='*60}\n")
    
    # Load and prepare data
    print("Loading data...")
    df = load_stock_data(data_path)
    print(f"Loaded {len(df)} rows of data")
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Preparing data for training...")
    data = prepare_data(
        df,
        target_column=target_column,
        sequence_length=sequence_length,
        test_ratio=1 - train_ratio
    )

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    target_scaler = data['target_scaler']

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Create model
    input_shape = (sequence_length, len(data['feature_columns']))
    print(f"\nCreating {model_type.upper()} model...")
    
    if model_type.lower() == 'lstm':
        model = create_lstm_model(input_shape)
        train_func = train_lstm_model
    elif model_type.lower() == 'gru':
        model = create_gru_model(input_shape)
        train_func = train_gru_model
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm' or 'gru'.")
    
    model.summary()
    
    # Create model save path
    os.makedirs(model_save_path, exist_ok=True)
    model_filename = os.path.join(model_save_path, f'{model_type.lower()}_model.h5')
    
    # Train model
    print(f"\nTraining model for {epochs} epochs...")
    history = train_func(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path=model_filename,
        patience=patience,
        verbose=verbose
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    if model_type.lower() == 'lstm':
        from model_lstm import evaluate_lstm_model
        eval_results = evaluate_lstm_model(model, X_test, y_test, target_scaler)
    else:
        from model_gru import evaluate_gru_model
        eval_results = evaluate_gru_model(model, X_test, y_test, target_scaler)
    
    print(f"\nTest Results:")
    print(f"  RMSE: {eval_results['rmse']:.4f}")
    print(f"  MAE:  {eval_results['mae']:.4f}")
    print(f"  R²:   {eval_results['r2']:.4f}")
    
    # Save training history
    history_path = os.path.join(model_save_path, f'{model_type.lower()}_training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']]
        }, f, indent=2)

    # Save scalers for consistent preprocessing at prediction time
    scalers_path = os.path.join(model_save_path, 'scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump({
            'feature_scaler': data['feature_scaler'],
            'target_scaler': target_scaler,
            'feature_columns': data['feature_columns'],
            'target_column': target_column,
            'sequence_length': sequence_length
        }, f)
    
    print(f"\nModel saved to: {model_filename}")
    print(f"Training history saved to: {history_path}")
    print(f"Scalers saved to: {scalers_path}")

    return {
        'model': model,
        'history': history,
        'eval_results': eval_results,
        'scaler': target_scaler,
        'target_scaler': target_scaler,
        'feature_scaler': data['feature_scaler'],
        'model_path': model_filename,
        'history_path': history_path,
        'scalers_path': scalers_path
    }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Train LSTM or GRU model for stock price prediction'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='../data/stock_data.csv',
        help='Path to stock data CSV file'
    )
    parser.add_argument(
        '--model-type', '-t',
        type=str,
        default='lstm',
        choices=['lstm', 'gru'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--model-save-path', '-m',
        type=str,
        default='../models/',
        help='Directory to save the trained model'
    )
    parser.add_argument(
        '--sequence-length', '-s',
        type=int,
        default=60,
        help='Number of time steps for sequences'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--patience', '-p',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--target-column', '-c',
        type=str,
        default='Close',
        help='Column to predict'
    )
    
    args = parser.parse_args()
    
    # Run training
    results = train(
        data_path=args.data,
        model_type=args.model_type,
        model_save_path=args.model_save_path,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        target_column=args.target_column
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
