# ============================================================
# DATA LOADING
# ============================================================
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, MultiNormalizer, GroupNormalizer, EncoderNormalizer
import utilities

print("\n" + "="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)

df = utilities.get_data()

df['time'] = pd.to_datetime(df['time'])

values = {"snwd": 0, "tsun": 0, "cldc": 9}
df.fillna(value=values, inplace=True)
df = df.sort_values(['station', 'time'])
df['time_idx'] = df.groupby('station').cumcount()

max_time_idx      = df.groupby('station')['time_idx'].max()
training_cutoff   = int(max_time_idx.iloc[0] * 0.7)
validation_cutoff = int(max_time_idx.iloc[0] * 0.9)

print(f"\nDataset split:")
print(f"  Training:   {(df['time_idx'] <= training_cutoff).sum()} samples")
print(f"  Validation: {((df['time_idx'] > training_cutoff) & (df['time_idx'] <= validation_cutoff)).sum()} samples")
print(f"  Test:       {(df['time_idx'] > validation_cutoff).sum()} samples")

# Ground-truth persistence MAE (pandas) — used as sanity check
df_check = df.sort_values(['station', 'time_idx'])
print("\nGround-truth persistence MAE (pandas, full dataset):")
for horizon in [1, 3, 6]:
    df_check[f'temp_plus{horizon}'] = df_check.groupby('station')['temp'].shift(-horizon)
    mae = (df_check['temp'] - df_check[f'temp_plus{horizon}']).abs().mean()
    print(f"  +{horizon}h: {mae:.4f} °C")

# ============================================================
# CREATE DATASETS
# ============================================================

print("\n" + "="*70)
print("CREATING TIMESERIES DATASETS")
print("="*70)

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=["temp", "prcp"],
    group_ids=["station"],
    max_encoder_length=24,
    max_prediction_length=6,
    time_varying_unknown_reals=["temp", "prcp"],
    target_normalizer=MultiNormalizer([
        GroupNormalizer(groups=["station"]),
        EncoderNormalizer()
    ]),
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    df[(df.time_idx > training_cutoff) & (df.time_idx <= validation_cutoff)],
    predict=False,
    stop_randomization=True
)

testing = TimeSeriesDataSet.from_dataset(
    training,
    df[df.time_idx > validation_cutoff],
    predict=False,
    stop_randomization=True
)

print(f"Train={len(training)}, Val={len(validation)}, Test={len(testing)}")


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

# All metrics are in physical units (°C / mm)
all_results  = {'val': {}, 'test': {}}
all_predictions = {'val': {}}