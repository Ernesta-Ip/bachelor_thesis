import torch
import os
from pathlib import Path
import glob
import meteostat as ms
from datetime import date

# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_DIR = Path("./model_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.backends.mps.is_available():
    accelerator = "mps"
elif torch.cuda.is_available():
    accelerator = "gpu"
else:
    accelerator = "cpu"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_latest_checkpoint(model_name, checkpoint_dir=CHECKPOINT_DIR):
    """Find the latest checkpoint for a given model."""
    pattern = str(checkpoint_dir / f"{model_name}-*.ckpt")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        print(f"  Found checkpoint: {latest}")
        return latest
    return None

def print_results(model_name, dataset_name, results):
    """Print per-horizon metrics in physical units."""
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - {dataset_name.upper()}")
    print(f"{'='*60}")
    for horizon in [3, 6]:
        r = results[f'horizon_{horizon}']
        print(f"\nHorizon +{horizon}h:")
        print(f"  Temp: MAE={r['temp_mae']:.4f} °C,  RMSE={r['temp_rmse']:.4f} °C")
        print(f"  Prcp: MAE={r['prcp_mae']:.4f} mm,  RMSE={r['prcp_rmse']:.4f} mm")
    print(f"{'='*60}")


def move_to_device(obj, device):
    """Recursively move tensors (including nested lists/tuples) to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj

def get_data():
    ms.config.block_large_requests = False
    ms.config.include_model_data = True
    # Specify location and time range
    POINT = ms.Point(48.1622, 11.6406, 520)  
    START = date(2019, 1, 1)
    END = date(2025, 12, 31)

    # Get nearby weather stations
    stations = ms.stations.nearby(POINT, limit=1)

    # Get daily data & perform interpolation
    return ms.hourly(stations, START, END, ).fetch().reset_index()
