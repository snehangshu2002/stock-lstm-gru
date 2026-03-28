# Stock Price Prediction with LSTM & GRU

A deep learning project for stock price prediction using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) neural networks.

## 📁 Project Structure

```
Stock_Price_Prediction/
│
├── data/
│   ├── Adani_port_stock.csv      # Adani port historical stock data
│   └── APSE Historical Data.csv  # APSE historical stock data
├── models/                       # Directory for trained models
├── src/
│   ├── data_preprocessing.py     # Data loading and preprocessing utilities
│   ├── model_lstm.py             # LSTM model architecture and training
│   ├── model_gru.py              # GRU model architecture and training
│   ├── train.py                  # Training script
│   └── predict.py                # Prediction script
├── notebooks/
│   └── analysis.ipynb            # Jupyter notebook for analysis
├── pyproject.toml                # Project configuration (uv)
└── README.md                     # This file
```

## 🚀 Features

- **LSTM Model**: Long Short-Term Memory network for sequence prediction
- **GRU Model**: Gated Recurrent Unit network (faster alternative to LSTM)
- **Data Preprocessing**: Automatic data cleaning, normalization, and sequence creation
- **Model Comparison**: Side-by-side comparison of LSTM and GRU performance
- **Future Prediction**: Predict future stock prices beyond the test set
- **Visualization**: Comprehensive plotting for analysis and results

## 📋 Requirements

- Python 3.12+
- TensorFlow 2.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn, plotly

## 🔧 Installation

```bash
# Navigate to project directory
cd "C:\Users\SNEHANGSHU\Desktop\Projects\Stock_Price_Prediction"

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment (if not auto-activated)
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

## 📊 Data Preparation

### Option 1: Download from Yahoo Finance

The project includes a utility to download stock data via `yfinance`:

```python
# In Python
import yfinance as yf

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
data.to_csv("data/stock_data.csv")
```

### Option 2: Use Your Own Data

Place your CSV files in the `data/` directory with the following columns:
- `Date` (YYYY-MM-DD format)
- `Open`
- `High`
- `Low`
- `Close`
- `Volume` (optional)
- `Adj Close` (optional)

Current datasets included:
- `Adani_port_stock.csv`
- `APSE Historical Data.csv`

## 🎯 Usage

### Training a Model

**Train LSTM Model:**
```bash
cd src
python train.py --data ../data/Adani_port_stock.csv --model-type lstm --epochs 50
```

**Train GRU Model:**
```bash
cd src
python train.py --data ../data/Adani_port_stock.csv --model-type gru --epochs 50
```

**Training Options:**
```
--data              Path to stock data CSV
--model-type        Model type: lstm or gru
--model-save-path   Directory to save model (default: ../models/)
--sequence-length   Time steps for sequences (default: 60)
--epochs            Number of training epochs (default: 50)
--batch-size        Training batch size (default: 32)
--patience          Early stopping patience (default: 10)
--target-column     Column to predict (default: Close)
```

### Making Predictions

**Predict with Trained Model:**
```bash
cd src
python predict.py --model ../models/lstm_model.h5 --model-type lstm --data ../data/Adani_port_stock.csv
```

**Predict Future Days:**
```bash
python predict.py --model ../models/lstm_model.h5 --model-type lstm --predict-future 7
```

**Compare LSTM and GRU:**
```python
from predict import compare_models

results = compare_models(
    lstm_model_path='../models/lstm_model.h5',
    gru_model_path='../models/gru_model.h5',
    data_path='../data/Adani_port_stock.csv'
)
```

### Using the Jupyter Notebook

Open `notebooks/analysis.ipynb` in Jupyter:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook provides a complete walkthrough of:
- Data exploration
- Preprocessing
- Model building
- Training
- Evaluation and comparison

## 📈 Model Architecture

### LSTM Model
```
Input (sequence_length, 1)
↓
LSTM (50 units, return_sequences=True)
↓
Dropout (0.2)
↓
LSTM (50 units)
↓
Dropout (0.2)
↓
Dense (50 units, relu)
↓
Dense (1 unit)
```

### GRU Model
```
Input (sequence_length, 1)
↓
GRU (50 units, return_sequences=True)
↓
Dropout (0.2)
↓
GRU (50 units)
↓
Dropout (0.2)
↓
Dense (50 units, relu)
↓
Dense (1 unit)
```

## 📊 Evaluation Metrics

- **RMSE** (Root Mean Square Error): Measures average prediction error
- **MAE** (Mean Absolute Error): Average absolute difference
- **R² Score**: Coefficient of determination (higher is better)

## 🧪 Programmatic Usage

```python
import sys
sys.path.append('src')

from data_preprocessing import load_stock_data, prepare_data, clean_data
from model_lstm import create_lstm_model, train_lstm_model, evaluate_lstm_model

# Load data
df = load_stock_data('data/Adani_port_stock.csv')
df = clean_data(df)

# Prepare data
data = prepare_data(df, sequence_length=60, train_ratio=0.8)

# Create and train model
model = create_lstm_model(input_shape=(60, 1))
history = train_lstm_model(
    model,
    data['X_train'], data['y_train'],
    data['X_test'], data['y_test'],
    epochs=50
)

# Evaluate
results = evaluate_lstm_model(model, data['X_test'], data['y_test'], data['scaler'])
print(f"RMSE: {results['rmse']:.4f}")
print(f"R²: {results['r2']:.4f}")
```

## 📝 Tips for Better Results

1. **More Data**: Use at least 2-3 years of historical data
2. **Sequence Length**: Try different values (30, 60, 90 days)
3. **Hyperparameters**: Experiment with layers, units, and dropout
4. **Features**: Add technical indicators (RSI, MACD, Moving Averages)
5. **Ensemble**: Combine predictions from both LSTM and GRU

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Do not use this model for actual trading decisions without thorough testing and professional advice.

## 📄 License

This project is open source and available for educational use.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📚 References

- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [GRU Paper](https://arxiv.org/abs/1412.3555)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Guide](https://www.tensorflow.org/)
