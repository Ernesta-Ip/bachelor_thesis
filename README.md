# Weather Forecasting with Multi-Model Comparison

This project trains, evaluates, and uses several time-series forecasting models for **local short-term weather forecasting**. It predicts **temperature (`temp`)** and **precipitation (`prcp`)** from hourly Meteostat data using a shared preprocessing pipeline and a common evaluation setup.

The repository has **two entry points**:

- `main.py` — training/loading checkpoints, evaluation, comparison, CSV export, and visualizations
- `app.py` — inference script for generating a 6-hour forecast from the latest history for a station

## Models included

1. **Baseline (Persistence)**
2. **N-HiTS**
3. **DLinear** (custom multi-target implementation)
4. **LSTM** 
5. **Temporal Fusion Transformer (TFT)**

## What the project does

### `main.py`
`main.py` is the full experiment pipeline. It:

- loads and preprocesses hourly weather data from Meteostat
- creates train / validation / test `TimeSeriesDataSet` objects
- trains each model if no checkpoint is found
- otherwise loads the latest checkpoint automatically
- evaluates all models on validation and test sets
- computes **MAE** and **RMSE** for:
  - temperature
  - precipitation
  - forecast horizons **+3h** and **+6h**
- prints a comparison table
- exports results to CSV
- generates comparison plots

### `app.py`
`app.py` is the inference entry point. It:

- loads the latest checkpoints for all trained models
- prepares a forecast input window for one station
- runs all models on the same recent history
- prints a side-by-side 6-hour forecast table
- optionally fetches fresh Meteostat observations for quick manual comparison

This script is useful when you want to **use already trained models** without rerunning the full experiment pipeline.

## Project structure

```text
.
├── app.py
├── main.py
├── custom_dlinear.py
├── preparation_preambule.py
├── utilities.py
├── visualization_all_models.py
├── model_checkpoints/
├── README.md
└── requirements.txt
```

## Core files

### `main.py`
Main research pipeline for training, evaluating, and comparing models.

### `app.py`
Forecasting/inference script that loads saved checkpoints and produces a next-6-hour forecast for a given station.

### `custom_dlinear.py`
Contains the custom `MultiTargetDLinear` PyTorch Lightning implementation.

### `preparation_preambule.py`
Shared data preparation module. It:

- downloads data from Meteostat
- converts timestamps
- fills selected missing values
- creates `time_idx`
- creates training, validation, and testing datasets
- provides shared storage dictionaries for results and predictions

### `utilities.py`
Shared utility functions and configuration:

- device / accelerator selection
- checkpoint discovery
- recursive tensor-to-device transfer
- Meteostat data loading
- result printing

### `visualization_all_models.py`
Standalone plotting script for generating model comparison figures if prediction/result objects are already available in memory.

## Data source

The project uses **hourly Meteostat data** for the nearest station to the configured geographic point.

Current configuration in `utilities.py`:

- location: `Point(48.1622, 11.6406, 520)`
- date range: `2019-01-01` to `2025-12-31`

The data is fetched with:

- `meteostat.stations.nearby(...)`
- `meteostat.hourly(...).fetch()`

## Targets

The forecasting targets are:

- `temp` — temperature in **°C**
- `prcp` — precipitation in **mm**

## Important note on units

Predictions and ground truth are handled in **physical units**.

As noted in `main.py`:

- `x['encoder_target']` and `y[0]` are already in raw physical units
- `pytorch-forecasting` applies normalization internally during forward passes
- model outputs are treated as physical values in this project
- **manual denormalization is not needed**

## Dataset setup

Defined in `preparation_preambule.py`:

- encoder length: **24 hours**
- prediction length: **6 hours**
- group id: `station`
- targets: `['temp', 'prcp']`

Dataset split is based on `time_idx`:

- training: first 70%
- validation: next 20%
- test: final 10%

Normalizers:

- `GroupNormalizer(groups=["station"])` for one target
- `EncoderNormalizer()` for the other target
- combined through `MultiNormalizer(...)`

## Requirements

Install the main Python dependencies used by the codebase:

```bash
pip install torch lightning pytorch-forecasting pandas numpy matplotlib meteostat
```

Depending on your environment, you may also need compatible versions of:

- `pytorch-lightning` / `lightning`
- `scikit-learn`
- `scipy`

## Hardware / device selection

`utilities.py` selects the device automatically:

- `mps` if available
- otherwise `cuda` if available
- otherwise `cpu`

## How checkpoint loading works

All checkpoints are stored under:

```text
./model_checkpoints
```

For each model, the helper below searches for the most recent checkpoint:

```python
find_latest_checkpoint(model_name)
```

If a checkpoint exists:

- the model is loaded
- training is skipped

If no checkpoint exists:

- the model is trained
- the best checkpoint is saved via `ModelCheckpoint`

## Entry point 1: Run the full experiment

Use `main.py` when you want to train or evaluate the whole model suite.

```bash
python main.py
```

### What `main.py` produces

Console output:

- dataset split summary
- model-by-model validation/test metrics
- final comparison table
- best model summary

Saved files:

- `model_comparison_results.csv`
- `all_models_sample_predictions.png`
- `all_models_scatter_plots.png`
- `all_models_error_analysis.png`
- `all_models_performance_comparison.png`
- `raw_data_overview.png`

Checkpoint directory:

- `model_checkpoints/*.ckpt`

## Entry point 2: Run forecasting/inference

Use `app.py` when you already have trained checkpoints and want a real forecast.

```bash
python app.py
```

### What `app.py` does

At startup it loads latest checkpoints for:

- `nhits`
- `dlinear`
- `lstm`
- `tft`

Then it runs:

```python
forecast_tonight(station_id, hours_of_history=None)
```

### `forecast_tonight()` behavior

Inputs:

- `station_id`: station identifier from `df['station']`
- `hours_of_history`: optional fresh history DataFrame

If `hours_of_history` is omitted:

- the function uses the last context window from the already loaded historical dataset

If `hours_of_history` is provided:

- it replaces the most recent encoder values with fresh `temp` and `prcp` observations
- then forecasts the next 6 hours using the trained models

Return value:

```python
(results, forecast_times)
```

Where `results` is a dictionary like:

```python
{
    "Baseline": {"temp": ..., "prcp": ...},
    "N-HiTS":   {"temp": ..., "prcp": ...},
    "DLinear":  {"temp": ..., "prcp": ...},
    "LSTM":     {"temp": ..., "prcp": ...},
    "TFT":      {"temp": ..., "prcp": ...},
}
```

## Example workflow

### 1. Train / evaluate all models

```bash
python main.py
```

### 2. Generate a 6-hour forecast with saved checkpoints

```bash
python app.py
```

## Model details

### Baseline
Simple persistence forecast:

- repeats the last observed temperature and precipitation values across the forecast horizon

### N-HiTS
Created with `NHiTS.from_dataset(...)` and trained with `MultiLoss([QuantileLoss(), QuantileLoss()])`.

### DLinear
Custom multi-target implementation in `custom_dlinear.py`.

Features:

- moving-average series decomposition
- trend + seasonal linear projection
- multi-target support
- quantile-loss compatibility
- PyTorch Lightning training loop

### LSTM
Implemented with `RecurrentNetwork.from_dataset(...)`.

Configuration used in `main.py`:

- `cell_type="LSTM"`
- `hidden_size=64`
- `rnn_layers=2`
- `dropout=0.1`
- `loss=MultiLoss([MAE(), MAE()])`

### TFT
Implemented with `TemporalFusionTransformer.from_dataset(...)`.

Configuration used in `main.py` includes:

- `hidden_size=32`
- `attention_head_size=1`
- `dropout=0.1`
- `hidden_continuous_size=16`
- quantile loss for both targets

## Evaluation

Metrics are computed separately for:

- temperature
- precipitation

At horizons:

- **+3 hours**
- **+6 hours**

Metrics:

- **MAE**
- **RMSE**

The helper used is:

```python
def compute_metrics(pred, actual):
    diff = pred - actual
    return {
        'mae': float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
    }
```

## Visualizations generated by `main.py`

### `all_models_sample_predictions.png`
Compares sample 6-hour trajectories across all models against actual values.

### `all_models_scatter_plots.png`
Actual vs predicted scatter plots for temperature and precipitation.

### `all_models_error_analysis.png`
Error histograms and boxplots by forecast horizon.

### `all_models_performance_comparison.png`
Bar charts comparing MAE and RMSE across models at +3h and +6h.

### `raw_data_overview.png`
Scatter overview of all raw sensor variables over time.

## Notes and assumptions

- The code assumes Meteostat returns the required columns, especially `time`, `temp`, and `prcp`.
- `app.py` expects trained checkpoints to exist before running.
- `visualization_all_models.py` is not a standalone end-to-end script unless `all_predictions`, `all_results`, `df`, and `validation` are already available in memory.
- `forecast_tonight()` assumes a station id that exists in `df['station']`.

## Typical output artifacts

After a successful run of `main.py`, the project directory usually contains:

```text
model_checkpoints/
model_comparison_results.csv
all_models_sample_predictions.png
all_models_scatter_plots.png
all_models_error_analysis.png
all_models_performance_comparison.png
raw_data_overview.png
```

## Minimal usage summary

Run the full pipeline:

```bash
python main.py
```

Run inference with existing checkpoints:

```bash
python app.py
```

## Future improvements

Possible extensions for this repository:

- expose `station_id`, point coordinates, and date ranges via CLI arguments
- save inference forecasts from `app.py` to CSV
- add logging instead of print-based progress reporting
- add tests for `custom_dlinear.py`
- package inference into a small API or web app
