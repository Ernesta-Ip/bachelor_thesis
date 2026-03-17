"""
Weather Forecasting - Complete Multi-Model Comparison
======================================================

This script trains/loads and evaluates 5 models for weather forecasting:
1. Baseline (Persistence)
2. N-HiTS
3. DLinear (multi-target)
4. LSTM
5. Temporal Fusion Transformer (TFT)

NOTE on units:
    pytorch-forecasting returns x['encoder_target'] and y[0] in raw physical units
    (°C and mm). Normalization is applied internally by the model during the forward
    pass; model outputs are also returned in physical units. 
"""

# ============================================================
# DATA LOADING
# ============================================================
import lightning.pytorch as pl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import QuantileLoss, MAE, MultiLoss
from pytorch_forecasting.models import NHiTS, TemporalFusionTransformer
from pytorch_forecasting.models.rnn import RecurrentNetwork

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

import utilities
from custom_dlinear import MultiTargetDLinear
from preparation_preambule import df, training, validation, testing, all_results, all_predictions

# ============================================================
# MODEL 1: BASELINE (PERSISTENCE)
# ============================================================

print("\n" + "="*70)
print("MODEL 1: BASELINE (PERSISTENCE)")
print("="*70)

def compute_metrics(pred, actual):
    """Compute MAE and RMSE. Inputs are numpy arrays."""
    diff = pred - actual
    return {
        'mae':  float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
    }

def plot_raw_data_overview(df):
    """
    Scatter overview of all raw sensor columns in the dataset, one subplot per variable.
    Dots represent actual sampled hourly values without implying continuity between them.
    """
    plot_cols = [c for c in df.columns if c not in ('time', 'time_idx', 'station')]
    n = len(plot_cols)

    fig, axes = plt.subplots(n, 1, figsize=(18, 2.5 * n), sharex=True)

    for ax, col in zip(axes, plot_cols):
        for _, group in df.groupby('station'):
            ax.scatter(group['time'], group[col], s=0.5, alpha=0.4)
        ax.set_ylabel(col, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        f'Raw Data Overview — Station {df["station"].iloc[0]}',
        fontsize=13, fontweight='bold'
    )
    axes[-1].set_xlabel('Time', fontsize=11)

    plt.tight_layout()
    plt.savefig('raw_data_overview.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: raw_data_overview.png")

def evaluate_baseline(dataset, dataset_name="Dataset"):
    """
    Persistence forecast: predict that each target stays constant at its
    last observed encoder value for all prediction steps.

    x['encoder_target'] and y[0] are in physical units (°C / mm).
    """
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    pred_temp_list, pred_prcp_list = [], []
    act_temp_list,  act_prcp_list  = [], []

    for batch in dataloader:
        x, y = batch

        if isinstance(x['encoder_target'], list):
            last_temp = x['encoder_target'][0][:, -1]
            last_prcp = x['encoder_target'][1][:, -1]
        else:
            last_temp = x['encoder_target'][:, -1, 0]
            last_prcp = x['encoder_target'][:, -1, 1]

        pred_temp = last_temp.unsqueeze(1).repeat(1, dataset.max_prediction_length)
        pred_prcp = last_prcp.unsqueeze(1).repeat(1, dataset.max_prediction_length)

        if isinstance(y[0], list):
            actual_temp, actual_prcp = y[0][0], y[0][1]
        else:
            actual_temp = y[0][..., 0]
            actual_prcp = y[0][..., 1]

        pred_temp_list.append(pred_temp.cpu())
        pred_prcp_list.append(pred_prcp.cpu())
        act_temp_list.append(actual_temp.cpu())
        act_prcp_list.append(actual_prcp.cpu())

    pred_temp = torch.cat(pred_temp_list, dim=0).numpy()
    pred_prcp = torch.cat(pred_prcp_list, dim=0).numpy()
    act_temp  = torch.cat(act_temp_list,  dim=0).numpy()
    act_prcp  = torch.cat(act_prcp_list,  dim=0).numpy()

    results = {}
    for horizon in [3, 6]:
        h = horizon - 1
        m_temp = compute_metrics(pred_temp[:, h], act_temp[:, h])
        m_prcp = compute_metrics(pred_prcp[:, h], act_prcp[:, h])
        results[f'horizon_{horizon}'] = {
            'temp_mae': m_temp['mae'],  'temp_rmse': m_temp['rmse'],
            'prcp_mae': m_prcp['mae'],  'prcp_rmse': m_prcp['rmse'],
        }

    utilities.print_results("Baseline", dataset_name, results)

    return (torch.from_numpy(pred_temp), torch.from_numpy(pred_prcp),
            torch.from_numpy(act_temp),  torch.from_numpy(act_prcp),
            results)


(baseline_val_pred_temp, baseline_val_pred_prcp,
 baseline_val_act_temp,  baseline_val_act_prcp,
 baseline_val_results) = evaluate_baseline(validation, "Validation")

(baseline_test_pred_temp, baseline_test_pred_prcp,
 baseline_test_act_temp,  baseline_test_act_prcp,
 baseline_test_results) = evaluate_baseline(testing, "Test")

all_results['val']['Baseline']  = baseline_val_results
all_results['test']['Baseline'] = baseline_test_results
all_predictions['val']['Baseline'] = {
    'temp_pred': baseline_val_pred_temp, 'temp_actual': baseline_val_act_temp,
    'prcp_pred': baseline_val_pred_prcp, 'prcp_actual': baseline_val_act_prcp,
}


# ============================================================
# MODEL 2: N-HITS
# ============================================================

print("\n" + "="*70)
print("MODEL 2: N-HITS")
print("="*70)

checkpoint_path = utilities.find_latest_checkpoint("nhits")

if checkpoint_path:
    print("Loading N-HiTS from checkpoint...")
    best_nhits_model = NHiTS.load_from_checkpoint(checkpoint_path)
    best_nhits_model = best_nhits_model.to(utilities.device)
    best_nhits_model.eval()
else:
    print("Training N-HiTS...")
    nhits_model = NHiTS.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=64,
        loss=MultiLoss([QuantileLoss(), QuantileLoss()]),
        optimizer="adam",
    )
    train_dataloader = training.to_dataloader(train=True,  batch_size=64, num_workers=0)
    val_dataloader   = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    checkpoint_callback = ModelCheckpoint(
        dirpath=utilities.CHECKPOINT_DIR, monitor="val_loss", mode="min",
        filename="nhits-{epoch:02d}-{val_loss:.4f}"
    )
    trainer = pl.Trainer(
        max_epochs=50, accelerator=utilities.accelerator,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                   checkpoint_callback]
    )
    trainer.fit(nhits_model, train_dataloader, val_dataloader)
    best_nhits_model = NHiTS.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_nhits_model = best_nhits_model.to(utilities.device)
    best_nhits_model.eval()


def evaluate_nhits(model, dataset, dataset_name="Dataset"):
    """
    Evaluate N-HiTS. 
    """
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    pred_temp_list, pred_prcp_list = [], []
    act_temp_list,  act_prcp_list  = [], []

    loss_fn    = QuantileLoss()
    median_idx = loss_fn.quantiles.index(0.5)

    with torch.no_grad():
        for batch in dataloader:
            x, y  = batch
            x_dev = utilities.move_to_device(x, utilities.device)
            out   = model(x_dev)
            pred  = out.prediction if hasattr(out, 'prediction') else out

            if isinstance(pred, (list, tuple)):
                pred_temp = pred[0][..., median_idx] if pred[0].ndim == 3 else pred[0]
                pred_prcp = pred[1][..., median_idx] if pred[1].ndim == 3 else pred[1]
            else:
                pred_temp = pred[..., 0, median_idx] if pred.ndim == 4 else pred[..., 0]
                pred_prcp = pred[..., 1, median_idx] if pred.ndim == 4 else pred[..., 1]

            if isinstance(y[0], list):
                actual_temp, actual_prcp = y[0][0], y[0][1]
            else:
                actual_temp, actual_prcp = y[0][..., 0], y[0][..., 1]

            pred_temp_list.append(pred_temp.cpu())
            pred_prcp_list.append(pred_prcp.cpu())
            act_temp_list.append(actual_temp.cpu())
            act_prcp_list.append(actual_prcp.cpu())

    pred_temp = torch.cat(pred_temp_list, dim=0).numpy()
    pred_prcp = torch.cat(pred_prcp_list, dim=0).numpy()
    act_temp  = torch.cat(act_temp_list,  dim=0).numpy()
    act_prcp  = torch.cat(act_prcp_list,  dim=0).numpy()

    results = {}
    for horizon in [3, 6]:
        h = horizon - 1
        m_temp = compute_metrics(pred_temp[:, h], act_temp[:, h])
        m_prcp = compute_metrics(pred_prcp[:, h], act_prcp[:, h])
        results[f'horizon_{horizon}'] = {
            'temp_mae': m_temp['mae'],  'temp_rmse': m_temp['rmse'],
            'prcp_mae': m_prcp['mae'],  'prcp_rmse': m_prcp['rmse'],
        }

    utilities.print_results("N-HiTS", dataset_name, results)

    return (torch.from_numpy(pred_temp), torch.from_numpy(pred_prcp),
            torch.from_numpy(act_temp),  torch.from_numpy(act_prcp),
            results)


(nhits_val_pred_temp, nhits_val_pred_prcp,
 nhits_val_act_temp,  nhits_val_act_prcp,
 nhits_val_results) = evaluate_nhits(best_nhits_model, validation, "Validation")

(nhits_test_pred_temp, nhits_test_pred_prcp,
 nhits_test_act_temp,  nhits_test_act_prcp,
 nhits_test_results) = evaluate_nhits(best_nhits_model, testing, "Test")

all_results['val']['N-HiTS']  = nhits_val_results
all_results['test']['N-HiTS'] = nhits_test_results
all_predictions['val']['N-HiTS'] = {
    'temp_pred': nhits_val_pred_temp, 'temp_actual': nhits_val_act_temp,
    'prcp_pred': nhits_val_pred_prcp, 'prcp_actual': nhits_val_act_prcp,
}


# ============================================================
# MODEL 3: DLINEAR (MULTI-TARGET)
# ============================================================

print("\n" + "="*70)
print("MODEL 3: DLINEAR (MULTI-TARGET)")
print("="*70)

checkpoint_path = utilities.find_latest_checkpoint("dlinear")

if checkpoint_path:
    print("Loading DLinear from checkpoint...")
    best_dlinear_model = MultiTargetDLinear.load_from_checkpoint(
        checkpoint_path, loss=QuantileLoss())
    best_dlinear_model = best_dlinear_model.to(utilities.device)
    best_dlinear_model.eval()
else:
    print("Training DLinear...")
    dlinear_model = MultiTargetDLinear(
        context_length=training.max_encoder_length,
        prediction_length=training.max_prediction_length,
        n_targets=len(training.target_names),
        moving_avg=25, individual=False,
        loss=QuantileLoss(), learning_rate=0.001,
    )
    train_dataloader = training.to_dataloader(train=True,  batch_size=64, num_workers=0)
    val_dataloader   = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    checkpoint_callback = ModelCheckpoint(
        dirpath=utilities.CHECKPOINT_DIR, monitor="val_loss", mode="min",
        filename="dlinear-{epoch:02d}-{val_loss:.4f}"
    )
    trainer = pl.Trainer(
        max_epochs=50, accelerator=utilities.accelerator,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                   checkpoint_callback]
    )
    trainer.fit(dlinear_model, train_dataloader, val_dataloader)
    best_dlinear_model = MultiTargetDLinear.load_from_checkpoint(
        checkpoint_callback.best_model_path, loss=QuantileLoss())
    best_dlinear_model = best_dlinear_model.to(utilities.device)
    best_dlinear_model.eval()


def evaluate_dlinear(model, dataset, dataset_name="Dataset"):
    """
    Evaluate DLinear. 
    """
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    pred_temp_list, pred_prcp_list = [], []
    act_temp_list,  act_prcp_list  = [], []

    loss_fn    = QuantileLoss()
    median_idx = loss_fn.quantiles.index(0.5)

    with torch.no_grad():
        for batch in dataloader:
            x, y  = batch
            x_dev = utilities.move_to_device(x, utilities.device)
            pred  = model(x_dev)['prediction']

            pred_temp = pred[0][..., median_idx] if pred[0].ndim == 3 else pred[0]
            pred_prcp = pred[1][..., median_idx] if pred[1].ndim == 3 else pred[1]

            if isinstance(y[0], list):
                actual_temp, actual_prcp = y[0][0], y[0][1]
            else:
                actual_temp, actual_prcp = y[0][..., 0], y[0][..., 1]

            pred_temp_list.append(pred_temp.cpu())
            pred_prcp_list.append(pred_prcp.cpu())
            act_temp_list.append(actual_temp.cpu())
            act_prcp_list.append(actual_prcp.cpu())

    pred_temp = torch.cat(pred_temp_list, dim=0).numpy()
    pred_prcp = torch.cat(pred_prcp_list, dim=0).numpy()
    act_temp  = torch.cat(act_temp_list,  dim=0).numpy()
    act_prcp  = torch.cat(act_prcp_list,  dim=0).numpy()

    results = {}
    for horizon in [3, 6]:
        h = horizon - 1
        m_temp = compute_metrics(pred_temp[:, h], act_temp[:, h])
        m_prcp = compute_metrics(pred_prcp[:, h], act_prcp[:, h])
        results[f'horizon_{horizon}'] = {
            'temp_mae': m_temp['mae'],  'temp_rmse': m_temp['rmse'],
            'prcp_mae': m_prcp['mae'],  'prcp_rmse': m_prcp['rmse'],
        }

    utilities.print_results("DLinear", dataset_name, results)

    return (torch.from_numpy(pred_temp), torch.from_numpy(pred_prcp),
            torch.from_numpy(act_temp),  torch.from_numpy(act_prcp),
            results)


(dlinear_val_pred_temp, dlinear_val_pred_prcp,
 dlinear_val_act_temp,  dlinear_val_act_prcp,
 dlinear_val_results) = evaluate_dlinear(best_dlinear_model, validation, "Validation")

(dlinear_test_pred_temp, dlinear_test_pred_prcp,
 dlinear_test_act_temp,  dlinear_test_act_prcp,
 dlinear_test_results) = evaluate_dlinear(best_dlinear_model, testing, "Test")

all_results['val']['DLinear']  = dlinear_val_results
all_results['test']['DLinear'] = dlinear_test_results
all_predictions['val']['DLinear'] = {
    'temp_pred': dlinear_val_pred_temp, 'temp_actual': dlinear_val_act_temp,
    'prcp_pred': dlinear_val_pred_prcp, 'prcp_actual': dlinear_val_act_prcp,
}


# ============================================================
# MODEL 4: LSTM 
# ============================================================

print("\n" + "="*70)
print("MODEL 4: LSTM")
print("="*70)

checkpoint_path = utilities.find_latest_checkpoint("lstm")

if checkpoint_path:
    print("Loading LSTM from checkpoint...")
    best_lstm_model = RecurrentNetwork.load_from_checkpoint(checkpoint_path)
    best_lstm_model = best_lstm_model.to(utilities.device)
    best_lstm_model.eval()
else:
    print("Training LSTM...")
    lstm_model = RecurrentNetwork.from_dataset(
        training,
        cell_type="LSTM", hidden_size=64, rnn_layers=2, dropout=0.1,
        loss=MultiLoss([MAE(), MAE()]),
        learning_rate=0.001, optimizer="adam",
    )
    train_dataloader = training.to_dataloader(train=True,  batch_size=64, num_workers=0)
    val_dataloader   = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    checkpoint_callback = ModelCheckpoint(
        dirpath=utilities.CHECKPOINT_DIR, monitor="val_loss", mode="min",
        filename="lstm-{epoch:02d}-{val_loss:.4f}"
    )
    trainer = pl.Trainer(
        max_epochs=50, accelerator=utilities.accelerator,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                   checkpoint_callback]
    )
    trainer.fit(lstm_model, train_dataloader, val_dataloader)
    best_lstm_model = RecurrentNetwork.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_lstm_model = best_lstm_model.to(utilities.device)
    best_lstm_model.eval()


def evaluate_lstm(model, dataset, dataset_name="Dataset"):
    """
    Evaluate LSTM. 
    """
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    pred_temp_list, pred_prcp_list = [], []
    act_temp_list,  act_prcp_list  = [], []

    with torch.no_grad():
        for batch in dataloader:
            x, y  = batch
            x_dev = utilities.move_to_device(x, utilities.device)
            out   = model(x_dev)
            pred  = out.prediction if hasattr(out, 'prediction') else out

            if isinstance(pred, (list, tuple)):
                pred_temp = pred[0].squeeze(-1) if pred[0].ndim == 3 else pred[0]
                pred_prcp = pred[1].squeeze(-1) if pred[1].ndim == 3 else pred[1]
            else:
                pred_temp = pred[..., 0]
                pred_prcp = pred[..., 1]

            if isinstance(y[0], list):
                actual_temp, actual_prcp = y[0][0], y[0][1]
            else:
                actual_temp, actual_prcp = y[0][..., 0], y[0][..., 1]

            pred_temp_list.append(pred_temp.cpu())
            pred_prcp_list.append(pred_prcp.cpu())
            act_temp_list.append(actual_temp.cpu())
            act_prcp_list.append(actual_prcp.cpu())

    pred_temp = torch.cat(pred_temp_list, dim=0).numpy()
    pred_prcp = torch.cat(pred_prcp_list, dim=0).numpy()
    act_temp  = torch.cat(act_temp_list,  dim=0).numpy()
    act_prcp  = torch.cat(act_prcp_list,  dim=0).numpy()

    results = {}
    for horizon in [3, 6]:
        h = horizon - 1
        m_temp = compute_metrics(pred_temp[:, h], act_temp[:, h])
        m_prcp = compute_metrics(pred_prcp[:, h], act_prcp[:, h])
        results[f'horizon_{horizon}'] = {
            'temp_mae': m_temp['mae'],  'temp_rmse': m_temp['rmse'],
            'prcp_mae': m_prcp['mae'],  'prcp_rmse': m_prcp['rmse'],
        }

    utilities.print_results("LSTM", dataset_name, results)

    return (torch.from_numpy(pred_temp), torch.from_numpy(pred_prcp),
            torch.from_numpy(act_temp),  torch.from_numpy(act_prcp),
            results)


(lstm_val_pred_temp, lstm_val_pred_prcp,
 lstm_val_act_temp,  lstm_val_act_prcp,
 lstm_val_results) = evaluate_lstm(best_lstm_model, validation, "Validation")

(lstm_test_pred_temp, lstm_test_pred_prcp,
 lstm_test_act_temp,  lstm_test_act_prcp,
 lstm_test_results) = evaluate_lstm(best_lstm_model, testing, "Test")

all_results['val']['LSTM']  = lstm_val_results
all_results['test']['LSTM'] = lstm_test_results
all_predictions['val']['LSTM'] = {
    'temp_pred': lstm_val_pred_temp, 'temp_actual': lstm_val_act_temp,
    'prcp_pred': lstm_val_pred_prcp, 'prcp_actual': lstm_val_act_prcp,
}


# ============================================================
# MODEL 5: TEMPORAL FUSION TRANSFORMER (TFT)
# ============================================================

print("\n" + "="*70)
print("MODEL 5: TEMPORAL FUSION TRANSFORMER")
print("="*70)

checkpoint_path = utilities.find_latest_checkpoint("tft")

if checkpoint_path:
    print("Loading TFT from checkpoint...")
    best_tft_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    best_tft_model = best_tft_model.to(utilities.device)
    best_tft_model.eval()
else:
    print("Training TFT...")
    tft_model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001, hidden_size=32, attention_head_size=1,
        dropout=0.1, hidden_continuous_size=16,
        loss=MultiLoss([QuantileLoss(), QuantileLoss()]),
        optimizer="adam",
    )
    train_dataloader = training.to_dataloader(train=True,  batch_size=64, num_workers=0)
    val_dataloader   = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    checkpoint_callback = ModelCheckpoint(
        dirpath=utilities.CHECKPOINT_DIR, monitor="val_loss", mode="min",
        filename="tft-{epoch:02d}-{val_loss:.4f}"
    )
    trainer = pl.Trainer(
        max_epochs=30, accelerator=utilities.accelerator,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                   checkpoint_callback]
    )
    trainer.fit(tft_model, train_dataloader, val_dataloader)
    best_tft_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_tft_model = best_tft_model.to(utilities.device)
    best_tft_model.eval()


def evaluate_tft(model, dataset, dataset_name="Dataset"):
    """
    Evaluate TFT. 
    """
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    pred_temp_list, pred_prcp_list = [], []
    act_temp_list,  act_prcp_list  = [], []

    loss_fn    = QuantileLoss()
    median_idx = loss_fn.quantiles.index(0.5)

    with torch.no_grad():
        for batch in dataloader:
            x, y  = batch
            x_dev = utilities.move_to_device(x, utilities.device)
            out   = model(x_dev)
            pred  = out.prediction if hasattr(out, 'prediction') else out

            if isinstance(pred, (list, tuple)):
                pred_temp = pred[0][..., median_idx] if pred[0].ndim == 3 else pred[0]
                pred_prcp = pred[1][..., median_idx] if pred[1].ndim == 3 else pred[1]
            else:
                pred_temp = pred[..., 0]
                pred_prcp = pred[..., 1]

            if isinstance(y[0], list):
                actual_temp, actual_prcp = y[0][0], y[0][1]
            else:
                actual_temp, actual_prcp = y[0][..., 0], y[0][..., 1]

            pred_temp_list.append(pred_temp.cpu())
            pred_prcp_list.append(pred_prcp.cpu())
            act_temp_list.append(actual_temp.cpu())
            act_prcp_list.append(actual_prcp.cpu())

    pred_temp = torch.cat(pred_temp_list, dim=0).numpy()
    pred_prcp = torch.cat(pred_prcp_list, dim=0).numpy()
    act_temp  = torch.cat(act_temp_list,  dim=0).numpy()
    act_prcp  = torch.cat(act_prcp_list,  dim=0).numpy()

    results = {}
    for horizon in [3, 6]:
        h = horizon - 1
        m_temp = compute_metrics(pred_temp[:, h], act_temp[:, h])
        m_prcp = compute_metrics(pred_prcp[:, h], act_prcp[:, h])
        results[f'horizon_{horizon}'] = {
            'temp_mae': m_temp['mae'],  'temp_rmse': m_temp['rmse'],
            'prcp_mae': m_prcp['mae'],  'prcp_rmse': m_prcp['rmse'],
        }

    utilities.print_results("TFT", dataset_name, results)

    return (torch.from_numpy(pred_temp), torch.from_numpy(pred_prcp),
            torch.from_numpy(act_temp),  torch.from_numpy(act_prcp),
            results)


(tft_val_pred_temp, tft_val_pred_prcp,
 tft_val_act_temp,  tft_val_act_prcp,
 tft_val_results) = evaluate_tft(best_tft_model, validation, "Validation")

(tft_test_pred_temp, tft_test_pred_prcp,
 tft_test_act_temp,  tft_test_act_prcp,
 tft_test_results) = evaluate_tft(best_tft_model, testing, "Test")

all_results['val']['TFT']  = tft_val_results
all_results['test']['TFT'] = tft_test_results
all_predictions['val']['TFT'] = {
    'temp_pred': tft_val_pred_temp, 'temp_actual': tft_val_act_temp,
    'prcp_pred': tft_val_pred_prcp, 'prcp_actual': tft_val_act_prcp,
}


# ============================================================
# COMPARISON TABLES
# ============================================================

MODEL_ORDER = ['Baseline', 'N-HiTS', 'DLinear', 'LSTM', 'TFT']

print("\n\n" + "="*90)
print("FINAL COMPARISON - ALL MODELS - PHYSICAL UNITS (VALIDATION SET)")
print("="*90)

for horizon in [3, 6]:
    print(f"\n{'Model':<20} | Temp MAE (°C) | Temp RMSE (°C) | Prcp MAE (mm) | Prcp RMSE (mm)")
    print(f"{'Horizon: +' + str(horizon) + 'h':<20} |" + "-"*67)
    for model_name in MODEL_ORDER:
        if model_name in all_results['val']:
            r = all_results['val'][model_name][f'horizon_{horizon}']
            print(f"{model_name:<20} | {r['temp_mae']:>13.4f} | {r['temp_rmse']:>14.4f} | "
                  f"{r['prcp_mae']:>13.4f} | {r['prcp_rmse']:>14.4f}")

print("="*90)


# ============================================================
# EXPORT RESULTS TO CSV
# ============================================================

print("\n" + "="*70)
print("EXPORTING RESULTS TO CSV")
print("="*70)

rows = []
for model_name in MODEL_ORDER:
    if model_name not in all_results['val']:
        continue
    for horizon in [3, 6]:
        r = all_results['val'][model_name][f'horizon_{horizon}']
        rows.append({
            'Model':        model_name,
            'Horizon':      horizon,
            'Temp_MAE_C':   r['temp_mae'],
            'Temp_RMSE_C':  r['temp_rmse'],
            'Prcp_MAE_mm':  r['prcp_mae'],
            'Prcp_RMSE_mm': r['prcp_rmse'],
        })

results_df = pd.DataFrame(rows)
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results exported to: model_comparison_results.csv")
print(results_df.to_string(index=False))


# ============================================================
# SUMMARY
# ============================================================

print("\n\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nModels evaluated:")
for model_name in MODEL_ORDER:
    if model_name in all_results['val']:
        print(f"  ✓ {model_name}")

print(f"\nCheckpoints: {utilities.CHECKPOINT_DIR}")

print("\nBest model (Validation, +6h, Temp MAE):")
best_model = min(all_results['val'].items(),
                 key=lambda x: x[1]['horizon_6']['temp_mae'])
print(f"  {best_model[0]}: {best_model[1]['horizon_6']['temp_mae']:.4f} °C")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)


# ============================================================
# VISUALIZATIONS
# ============================================================

plt.style.use('seaborn-v0_8-darkgrid')
FIGURE_DPI   = 150
COLORS = {
    'Baseline': '#808080',
    'N-HiTS':   '#1f77b4',
    'DLinear':  '#ff7f0e',
    'LSTM':     '#2ca02c',
    'TFT':      '#d62728',
}
SAMPLE_INDICES = [100, 500, 1000]

time_idx_to_datetime = df.set_index('time_idx')['time'].to_dict()


def get_sample_timestamps(dataset, sample_indices):
    dataloader = dataset.to_dataloader(train=False, batch_size=len(dataset), num_workers=0)
    for batch in dataloader:
        x, y = batch
        if 'decoder_time_idx' in x:
            time_idx = x['decoder_time_idx']
        else:
            enc_idx  = x.get('encoder_time_idx', x.get('time_idx'))
            time_idx = (enc_idx[:, -1:] +
                        torch.arange(1, dataset.max_prediction_length + 1).unsqueeze(0)
                        if enc_idx is not None else None)
        break
    if time_idx is None:
        return [f"Sample {i+1}" for i in sample_indices]
    timestamps = []
    for idx in sample_indices:
        if idx < len(time_idx):
            t = time_idx_to_datetime.get(int(time_idx[idx, 0].item()))
            timestamps.append(t.strftime('%Y-%m-%d %H:%M') if t else f"Sample {idx+1}")
        else:
            timestamps.append(f"Sample {idx+1}")
    return timestamps


def plot_sample_predictions_all_models(all_predictions, validation, sample_indices=SAMPLE_INDICES):

    timestamps  = get_sample_timestamps(validation, sample_indices)
    first_model = list(all_predictions.keys())[0]
    hours       = np.arange(1, validation.max_prediction_length + 1)

    fig = plt.figure(figsize=(20, 4*len(sample_indices)))
    gs  = GridSpec(len(sample_indices), 2, figure=fig, hspace=0.3, wspace=0.3)

    for si, (idx, ts) in enumerate(zip(sample_indices, timestamps)):
        for col, (key, ylabel, title) in enumerate([
            ('temp', 'Temperature (°C)',   'Temperature'),
            ('prcp', 'Precipitation (mm)', 'Precipitation'),
        ]):
            ax = fig.add_subplot(gs[si, col])
            actual = all_predictions[first_model][f'{key}_actual'][idx].numpy()
            ax.plot(hours, actual, 'o-', color='black', lw=2, ms=6, label='Actual', zorder=10)
            for name, preds in all_predictions.items():
                pred = preds[f'{key}_pred'][idx].numpy()
                ax.plot(hours, pred, 's--', color=COLORS[name], lw=1.5, ms=4,
                        label=name, alpha=0.8)
            ax.set_xlabel('Hours ahead', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{title} - {ts}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(hours)

    plt.suptitle('Sample Predictions - All Models', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('all_models_sample_predictions.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_models_sample_predictions.png")


def plot_scatter_all_models(all_predictions):
    n_models = len(all_predictions)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    for col, (name, preds) in enumerate(all_predictions.items()):
        for row, (key, label, unit) in enumerate([
            ('temp', 'Temperature', '°C'),
            ('prcp', 'Precipitation', 'mm'),
        ]):
            ax = axes[row, col]
            actual = preds[f'{key}_actual'].numpy().flatten()
            pred   = preds[f'{key}_pred'].numpy().flatten()
            vmin, vmax = actual.min(), actual.max()
            ax.scatter(actual, pred, alpha=0.3, s=10, color=COLORS[name])
            ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2, label='Perfect')
            ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
            ax.set_xlabel(f'Actual {label} ({unit})', fontsize=11)
            ax.set_ylabel(f'Predicted {label} ({unit})', fontsize=11)
            ax.set_title(f'{name}\n{label}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    plt.suptitle('Actual vs Predicted - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_models_scatter_plots.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_models_scatter_plots.png")


def plot_error_distributions_all_models(all_predictions):
    n_models   = len(all_predictions)
    fig, axes  = plt.subplots(2, 2, figsize=(16, 12))
    legend_els = [Patch(facecolor=COLORS[n], alpha=0.6, label=n)
                  for n in all_predictions.keys()]

    for row, (key, label, unit) in enumerate([
        ('temp', 'Temperature', '°C'),
        ('prcp', 'Precipitation', 'mm'),
    ]):
        # Histogram
        ax = axes[row, 0]
        for name, preds in all_predictions.items():
            errors = (preds[f'{key}_pred'] - preds[f'{key}_actual']).numpy().flatten()
            ax.hist(errors, bins=50, alpha=0.5, label=name, color=COLORS[name], edgecolor='black')
        ax.axvline(0, color='red', ls='--', lw=2)
        ax.set_xlabel(f'{label} Error ({unit})', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{label} Error Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Boxplot by horizon
        ax = axes[row, 1]
        pred_len = list(all_predictions.values())[0][f'{key}_pred'].shape[1]
        for mi, (name, preds) in enumerate(all_predictions.items()):
            for hi in range(pred_len):
                errors = (preds[f'{key}_pred'][:, hi] - preds[f'{key}_actual'][:, hi]).numpy()
                pos = hi * (n_models + 1) + mi
                ax.boxplot([errors], positions=[pos], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor=COLORS[name], alpha=0.6),
                           medianprops=dict(color='black', lw=2))
        ax.axhline(0, color='red', ls='--', lw=2, alpha=0.5)
        ax.set_ylabel(f'{label} Error ({unit})', fontsize=11)
        ax.set_title(f'{label} Error by Forecast Horizon', fontsize=12, fontweight='bold')
        ax.set_xticks([i*(n_models+1)+n_models//2 for i in range(pred_len)])
        ax.set_xticklabels([f'+{i+1}h' for i in range(pred_len)])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(handles=legend_els, fontsize=9)

    plt.suptitle('Error Analysis - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_models_error_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_models_error_analysis.png")


def plot_performance_comparison(all_results):
    models     = [m for m in MODEL_ORDER if m in all_results]
    x, width   = np.arange(len(models)), 0.35
    fig, axes  = plt.subplots(2, 2, figsize=(14, 10))
    configs = [
        (0, 0, 'temp_mae',  'Temperature MAE',   '°C', 'lightblue',   'darkblue'),
        (0, 1, 'temp_rmse', 'Temperature RMSE',  '°C', 'lightcoral',  'darkred'),
        (1, 0, 'prcp_mae',  'Precipitation MAE', 'mm', 'lightgreen',  'darkgreen'),
        (1, 1, 'prcp_rmse', 'Precipitation RMSE','mm', 'lightyellow', 'orange'),
    ]
    for r, c, metric, title, unit, c3, c6 in configs:
        ax = axes[r, c]
        v3 = [all_results[m]['horizon_3'][metric] for m in models]
        v6 = [all_results[m]['horizon_6'][metric] for m in models]
        b1 = ax.bar(x - width/2, v3, width, label='+3h', color=c3, edgecolor='black')
        b2 = ax.bar(x + width/2, v6, width, label='+6h', color=c6, edgecolor='black')
        ax.set_ylabel(f'{title} ({unit})', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    plt.suptitle('Model Performance Comparison (Validation Set)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_models_performance_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_models_performance_comparison.png")


print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

print("\n1. Sample prediction plots...")
plot_sample_predictions_all_models(all_predictions['val'], validation, SAMPLE_INDICES)

print("\n2. Scatter plots...")
plot_scatter_all_models(all_predictions['val'])

print("\n3. Error distributions...")
plot_error_distributions_all_models(all_predictions['val'])

print("\n4. Performance comparison...")
plot_performance_comparison(all_results['val'])

print("\n5. Raw data overview...")
plot_raw_data_overview(df)

print("\n" + "="*70)
print("ALL DONE!")
print("="*70)
print("\nGenerated files:")
print("  - model_comparison_results.csv")
print("  - all_models_sample_predictions.png")
print("  - all_models_scatter_plots.png")
print("  - all_models_error_analysis.png")
print("  - all_models_performance_comparison.png")
print("  - raw_data_overview.png")
