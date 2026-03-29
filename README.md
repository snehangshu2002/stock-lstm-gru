# Stock price prediction with LSTM and GRU

This project trains recurrent neural networks on historical stock data and compares LSTM vs GRU performance for next-step price prediction.

Current framing: this is a many-to-one RNN setup (a sequence of past timesteps predicts one next value).
Planned extension: add a many-to-many RNN setup in a future iteration of this project.

## Project structure

```text
Stock_Price_Prediction/
├── data/                         # Input CSV files
├── models/                       # Trained models and scalers
├── notebooks/
│   └── analysis_simple.ipynb     # Main notebook (follow this first)
├── src/
│   ├── data_preprocessing.py     # Cleaning, sequence prep, scaling
│   ├── model_lstm.py             # LSTM model helpers
│   ├── model_gru.py              # GRU model helpers
│   ├── train.py                  # Training entrypoint
│   └── predict.py                # Inference entrypoint
├── pyproject.toml
└── README.md
```

## Environment

- Python 3.12+
- Dependencies managed with `uv`

Install dependencies:

```bash
uv sync
```

## Data format

Put CSV files in `data/` with these columns:

- `Date` (`YYYY-MM-DD`)
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Included datasets:

- `data/Adani_port_stock.csv`


## Train

Train LSTM:

```bash
uv run src/train.py --data data/Adani_port_stock.csv --model-type lstm --model-save-path models --epochs 50
```

Train GRU:

```bash
uv run src/train.py --data data/Adani_port_stock.csv --model-type gru --model-save-path models --epochs 50
```

Common options:

```text
--data              Path to stock data CSV
--model-type        lstm or gru
--model-save-path   Directory for model artifacts
--sequence-length   Time steps per sample (default: 60)
--epochs            Training epochs (default: 50)
--batch-size        Batch size (default: 32)
--patience          Early stopping patience (default: 10)
--target-column     Column to predict (default: Close)
```

Training output:

- `models/lstm_model.h5` or `models/gru_model.h5`
- `models/scalers.pkl`

## Predict

Run prediction with a trained model:

```bash
uv run src/predict.py --model models/lstm_model.h5 --model-type lstm --data data/Adani_port_stock.csv --scaler-path models/scalers.pkl
```

Predict future steps:

```bash
uv run src/predict.py --model models/lstm_model.h5 --model-type lstm --data data/Adani_port_stock.csv --scaler-path models/scalers.pkl --predict-future 7
```

For unseen data, pass a CSV with the same schema via `--data`.

## Notebooks

Main workflow: follow `notebooks/analysis_simple.ipynb` first.

Run the notebook walkthrough:

```bash
jupyter notebook notebooks/analysis_simple.ipynb
```

## Model summary

In the current notebook version, both models use stacked recurrent layers without dropout.

- LSTM stack: `LSTM(100) -> LSTM(100) -> LSTM(50) -> Dense(1)`  
  Trainable params: `153,051`
- GRU stack: `GRU(100) -> GRU(100) -> GRU(50) -> Dense(1)`  
  Trainable params: `115,551`

`stacked` means more than one recurrent layer of the same type is used in sequence.

## Notebook run snapshot (`notebooks/analysis_simple.ipynb`)

This section reflects the latest executed outputs currently saved in the notebook.

Notebook settings used in that run:

- Sequence length: `100`
- Train/test split: `80/20` with `shuffle=False`
- Batch size: `64`
- Max epochs: `100`
- Early stopping patience: `15`

Evaluation snapshot from the notebook:

- LSTM (normalized): `RMSE 0.023672`, `MAE 0.017531`, `R2 0.838048`
- LSTM (price scale): `RMSE 26.684104`, `MAE 19.762132`, `R2 0.838048`
- GRU (normalized): `RMSE 0.021273`, `MAE 0.015023`, `R2 0.869207`

7-business-day forecast snapshot in the notebook:

- LSTM forecast (2026-03-30 to 2026-04-07): roughly `1369.97` to `1372.69`
- GRU forecast (2026-03-30 to 2026-04-07): roughly `1377.03` to `1375.13`
