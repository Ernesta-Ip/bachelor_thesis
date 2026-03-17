import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, QuantileLoss
from pytorch_forecasting.models import NHiTS, TemporalFusionTransformer
from pytorch_forecasting.models.rnn import RecurrentNetwork

from custom_dlinear import MultiTargetDLinear
import utilities
from preparation_preambule import df, training
from datetime import datetime, timedelta
import meteostat as ms


checkpoint_path = utilities.find_latest_checkpoint("nhits")
best_nhits_model = NHiTS.load_from_checkpoint(checkpoint_path).to(utilities.device).eval()
checkpoint_path = utilities.find_latest_checkpoint("dlinear")
best_dlinear_model = MultiTargetDLinear.load_from_checkpoint(checkpoint_path, loss=QuantileLoss()).to(utilities.device).eval()
checkpoint_path = utilities.find_latest_checkpoint("lstm")
best_lstm_model = RecurrentNetwork.load_from_checkpoint(checkpoint_path).to(utilities.device).eval()
checkpoint_path = utilities.find_latest_checkpoint("tft")
best_tft_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path).to(utilities.device).eval()

def forecast_tonight(station_id, hours_of_history=None):
    """
    Run a forecast for a specific station using the last 24h of data.

    Args:
        station_id: station name/ID as it appears in df['station']
        hours_of_history: optional DataFrame of 24 rows — if None, uses
                          the last 24 rows from the already-loaded df

    Returns:
        dict with 'temp' and 'prcp' arrays of length 6 (next 6 hours)
    """
    # ── 1. Get the last 24 hours of data for this station 
    CONTEXT = training.max_encoder_length + training.max_prediction_length  # 30

    if hours_of_history is None:
        hist = (df[df['station'] == station_id]
                .sort_values('time_idx')
                .tail(CONTEXT)
                .copy())
        last_obs_time = hist['time'].iloc[-1]
    else:
        fresh_sorted = hours_of_history.sort_values('time')
        last_obs_time = fresh_sorted['time'].iloc[-1]  

        fresh_temp = fresh_sorted['temp'].values[-training.max_encoder_length:]
        fresh_prcp = fresh_sorted['prcp'].values[-training.max_encoder_length:]

        hist = (df[df['station'] == station_id]
                .sort_values('time_idx')
                .tail(CONTEXT)
                .copy())
        hist.iloc[-training.max_encoder_length:,
        hist.columns.get_loc('temp')] = fresh_temp
        hist.iloc[-training.max_encoder_length:,
        hist.columns.get_loc('prcp')] = fresh_prcp

    # ── 2. Build a TimeSeriesDataSet from the training dataset 
    # predict=True tells it: use all rows as encoder, forecast forward
    forecast_dataset = TimeSeriesDataSet.from_dataset(
        training,  # inherits normalizer, encoder length, etc.
        hist,
        predict=True,  
        stop_randomization=True,
    )

    dataloader = forecast_dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    # ── 3. Run inference with each model 
    results = {}
    loss_fn = QuantileLoss()
    median_idx = loss_fn.quantiles.index(0.5)

    # --- Baseline (persistence: repeat last known value) ---
    last_temp = float(hist['temp'].iloc[-1])
    last_prcp = float(hist['prcp'].iloc[-1])
    results['Baseline'] = {
        'temp': np.full(6, last_temp),
        'prcp': np.full(6, last_prcp),
    }

    # --- Learned models ---
    model_map = {
        'N-HiTS': (best_nhits_model, 'quantile'),
        'DLinear': (best_dlinear_model, 'quantile_custom'),  # custom model
        'LSTM': (best_lstm_model, 'point'),
        'TFT': (best_tft_model, 'quantile'),
    }

    for name, (model, kind) in model_map.items():
        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                x_dev = utilities.move_to_device(x, utilities.device)

                if kind == 'quantile_custom':
                    # DLinear returns {'prediction': [temp_tensor, prcp_tensor]}
                    pred = model(x_dev)['prediction']
                    temp_pred = pred[0][0, :, median_idx].cpu().numpy()
                    prcp_pred = pred[1][0, :, median_idx].cpu().numpy()
                else:
                    out = model(x_dev)
                    pred = out.prediction if hasattr(out, 'prediction') else out

                    if isinstance(pred, (list, tuple)):
                        p_temp = pred[0][0]  # shape [6] or [6, n_quantiles]
                        p_prcp = pred[1][0]
                    else:
                        p_temp = pred[0, :, 0] if pred.ndim == 3 else pred[0, :, 0]
                        p_prcp = pred[0, :, 1] if pred.ndim == 3 else pred[0, :, 1]

                    if p_temp.ndim == 2:
                        if p_temp.shape[-1] == 1:
                            # Point forecast (MAE loss) 
                            temp_pred = p_temp[:, 0].cpu().numpy()
                            prcp_pred = p_prcp[:, 0].cpu().numpy()
                        else:
                            # Quantile output — extract median
                            temp_pred = p_temp[:, median_idx].cpu().numpy()
                            prcp_pred = p_prcp[:, median_idx].cpu().numpy()
                    else:
                        temp_pred = p_temp.cpu().numpy()
                        prcp_pred = p_prcp.cpu().numpy()

        results[name] = {'temp': temp_pred, 'prcp': prcp_pred}

    # ── 4. Print a readable forecast table 
    forecast_times = [last_obs_time + pd.Timedelta(hours=h + 1) for h in range(6)]
    print(f"\nForecast for station '{station_id}' from {last_obs_time}")
    print(f"{'Time':<20}", end="")
    for name in results:
        print(f"  {name:>10} °C  {name:>10} mm", end="")
    print()
    print("-" * (20 + len(results) * 26))

    for h, t in enumerate(forecast_times):
        print(f"{str(t):<20}", end="")
        for name in results:
            tc = results[name]['temp'][h]
            pc = results[name]['prcp'][h]
            print(f"  {tc:>12.1f}  {pc:>12.2f}", end="")
        print()

    return results, forecast_times

if __name__ == '__main__':

    ms.config.block_large_requests = False
    ms.config.include_model_data = True
    # Specify location and time range
    POINT = ms.Point(48.1622, 11.6406, 520)  # exact location in Munich
    START = datetime(2026, 2, 1,0,0,0)
    END = datetime(2026, 2, 4,5,0,0)

    # Get nearby weather stations
    stations = ms.stations.nearby(POINT, limit=1)

    # Get daily data & perform interpolation
    raw = ms.hourly(stations, START, END, ).fetch()
    fresh = raw.reset_index()
    print(f"Columns: {fresh.columns.tolist()}")
    print(f"Rows: {len(fresh)}")
    print(fresh[['time', 'temp', 'prcp']])
    fresh = fresh.sort_values('time')   # oldest first
    preds, times = forecast_tonight('10865', hours_of_history=fresh)

    raw2 = ms.hourly(stations, END + timedelta(hours=1), END + timedelta(hours=6), ).fetch()
    check_prediction = raw2.reset_index()
    print(check_prediction[['time', 'temp', 'prcp']])