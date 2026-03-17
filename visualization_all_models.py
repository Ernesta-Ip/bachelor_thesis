"""
Comprehensive Visualizations for Weather Forecasting Models
=============================================================

This script creates visualizations for all models:
1. Sample prediction plots (same samples across all models)
2. Scatter plots (actual vs predicted)
3. Error distribution plots
4. Model comparison plots

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
FIGURE_DPI = 150
COLORS = {
    'Baseline': '#808080',  # Gray
    'N-HiTS': '#1f77b4',  # Blue
    'DLinear': '#ff7f0e',  # Orange
    'LSTM': '#2ca02c',  # Green
    'TFT': '#d62728'  # Red
}

# Select same samples for all models 
SAMPLE_INDICES = [100, 500, 1000]  

# Time mapping for x-axis labels 
time_idx_to_datetime = df.set_index('time_idx')['time'].to_dict()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_sample_timestamps(dataset, sample_indices):
    """Get timestamps for sample indices"""
    dataloader = dataset.to_dataloader(train=False, batch_size=len(dataset), num_workers=0)

    for batch in dataloader:
        x, y = batch
        if 'decoder_time_idx' in x:
            time_idx = x['decoder_time_idx']
        else:
            # Estimate based on encoder
            encoder_time_idx = x.get('encoder_time_idx', x.get('time_idx'))
            if encoder_time_idx is not None:
                last_idx = encoder_time_idx[:, -1:]
                time_idx = last_idx + torch.arange(1, dataset.max_prediction_length + 1).unsqueeze(0)
            else:
                time_idx = None
        break

    if time_idx is not None:
        timestamps = []
        for idx in sample_indices:
            if idx < len(time_idx):
                start_time_idx = int(time_idx[idx, 0].item())
                start_time = time_idx_to_datetime.get(start_time_idx)
                if start_time:
                    timestamps.append(start_time.strftime('%Y-%m-%d %H:%M'))
                else:
                    timestamps.append(f"Sample {idx + 1}")
            else:
                timestamps.append(f"Sample {idx + 1}")
        return timestamps
    else:
        return [f"Sample {i + 1}" for i in sample_indices]


# ============================================================
# 1. SAMPLE PREDICTIONS - ALL MODELS
# ============================================================

def plot_sample_predictions_all_models(all_predictions, validation, sample_indices=SAMPLE_INDICES):
    """
    Plot sample predictions for all models side-by-side

    Creates a grid: rows = samples, columns = models
    Shows both temperature and precipitation
    """

    n_samples = len(sample_indices)
    n_models = len(all_predictions)

    # Get timestamps
    timestamps = get_sample_timestamps(validation, sample_indices)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 4 * n_samples))
    gs = GridSpec(n_samples, 2, figure=fig, hspace=0.3, wspace=0.3)

    for sample_idx, (actual_idx, timestamp) in enumerate(zip(sample_indices, timestamps)):

        # Temperature plot
        ax_temp = fig.add_subplot(gs[sample_idx, 0])

        # Plot actual values (same for all models)
        model_name = list(all_predictions.keys())[0]
        actual_temp = all_predictions[model_name]['temp_actual'][actual_idx].cpu().numpy()
        hours = np.arange(1, len(actual_temp) + 1)
        ax_temp.plot(hours, actual_temp, 'o-', color='black', linewidth=2,
                     markersize=6, label='Actual', zorder=10)

        # Plot predictions from all models
        for model_name, preds in all_predictions.items():
            pred_temp = preds['temp_pred'][actual_idx].cpu().numpy()
            ax_temp.plot(hours, pred_temp, 's--', color=COLORS[model_name],
                         linewidth=1.5, markersize=4, label=model_name, alpha=0.8)

        ax_temp.set_xlabel('Hours ahead', fontsize=11)
        ax_temp.set_ylabel('Temperature (normalized)', fontsize=11)
        ax_temp.set_title(f'Temperature - {timestamp}', fontsize=12, fontweight='bold')
        ax_temp.legend(fontsize=9, loc='best')
        ax_temp.grid(True, alpha=0.3)
        ax_temp.set_xticks(hours)

        # Precipitation plot
        ax_prcp = fig.add_subplot(gs[sample_idx, 1])

        # Plot actual values
        actual_prcp = all_predictions[model_name]['prcp_actual'][actual_idx].cpu().numpy()
        ax_prcp.plot(hours, actual_prcp, 'o-', color='black', linewidth=2,
                     markersize=6, label='Actual', zorder=10)

        # Plot predictions from all models
        for model_name, preds in all_predictions.items():
            pred_prcp = preds['prcp_pred'][actual_idx].cpu().numpy()
            ax_prcp.plot(hours, pred_prcp, 's--', color=COLORS[model_name],
                         linewidth=1.5, markersize=4, label=model_name, alpha=0.8)

        ax_prcp.set_xlabel('Hours ahead', fontsize=11)
        ax_prcp.set_ylabel('Precipitation (normalized)', fontsize=11)
        ax_prcp.set_title(f'Precipitation - {timestamp}', fontsize=12, fontweight='bold')
        ax_prcp.legend(fontsize=9, loc='best')
        ax_prcp.grid(True, alpha=0.3)
        ax_prcp.set_xticks(hours)

    plt.suptitle('Sample Predictions - All Models Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('all_models_sample_predictions.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_models_sample_predictions.png")


# ============================================================
# 2. SCATTER PLOTS - ACTUAL VS PREDICTED
# ============================================================

def plot_scatter_all_models(all_predictions):
    """
    Create scatter plots (actual vs predicted) for all models

    One figure with 2 rows (temp, prcp) and N columns (models)
    """

    n_models = len(all_predictions)

    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

    # Get global min/max for consistent axis ranges
    all_temp_actual = []
    all_temp_pred = []
    all_prcp_actual = []
    all_prcp_pred = []

    for preds in all_predictions.values():
        all_temp_actual.append(preds['temp_actual'].cpu().numpy().flatten())
        all_temp_pred.append(preds['temp_pred'].cpu().numpy().flatten())
        all_prcp_actual.append(preds['prcp_actual'].cpu().numpy().flatten())
        all_prcp_pred.append(preds['prcp_pred'].cpu().numpy().flatten())

    temp_min = min([a.min() for a in all_temp_actual + all_temp_pred])
    temp_max = max([a.max() for a in all_temp_actual + all_temp_pred])
    prcp_min = min([a.min() for a in all_prcp_actual + all_prcp_pred])
    prcp_max = max([a.max() for a in all_prcp_actual + all_prcp_pred])

    for col, (model_name, preds) in enumerate(all_predictions.items()):
        # Temperature scatter
        ax_temp = axes[0, col]
        temp_actual_flat = preds['temp_actual'].cpu().numpy().flatten()
        temp_pred_flat = preds['temp_pred'].cpu().numpy().flatten()

        ax_temp.scatter(temp_actual_flat, temp_pred_flat, alpha=0.3, s=10, color=COLORS[model_name])
        ax_temp.plot([temp_min, temp_max], [temp_min, temp_max], 'r--', linewidth=2, label='Perfect')
        ax_temp.set_xlim(temp_min, temp_max)
        ax_temp.set_ylim(temp_min, temp_max)
        ax_temp.set_xlabel('Actual Temperature (norm)', fontsize=11)
        ax_temp.set_ylabel('Predicted Temperature (norm)', fontsize=11)
        ax_temp.set_title(f'{model_name}\nTemperature', fontsize=12, fontweight='bold')
        ax_temp.legend(fontsize=9)
        ax_temp.grid(True, alpha=0.3)
        ax_temp.set_aspect('equal')

        # Precipitation scatter
        ax_prcp = axes[1, col]
        prcp_actual_flat = preds['prcp_actual'].cpu().numpy().flatten()
        prcp_pred_flat = preds['prcp_pred'].cpu().numpy().flatten()

        ax_prcp.scatter(prcp_actual_flat, prcp_pred_flat, alpha=0.3, s=10, color=COLORS[model_name])
        ax_prcp.plot([prcp_min, prcp_max], [prcp_min, prcp_max], 'r--', linewidth=2, label='Perfect')
        ax_prcp.set_xlim(prcp_min, prcp_max)
        ax_prcp.set_ylim(prcp_min, prcp_max)
        ax_prcp.set_xlabel('Actual Precipitation (norm)', fontsize=11)
        ax_prcp.set_ylabel('Predicted Precipitation (norm)', fontsize=11)
        ax_prcp.set_title(f'{model_name}\nPrecipitation', fontsize=12, fontweight='bold')
        ax_prcp.legend(fontsize=9)
        ax_prcp.grid(True, alpha=0.3)
        ax_prcp.set_aspect('equal')

    plt.suptitle('Actual vs Predicted - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_models_scatter_plots.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_models_scatter_plots.png")


# ============================================================
# 3. ERROR DISTRIBUTIONS
# ============================================================

def plot_error_distributions_all_models(all_predictions):
    """
    Plot error distributions for all models

    Histograms and boxplots by horizon
    """

    n_models = len(all_predictions)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate errors for all models
    errors_temp = {}
    errors_prcp = {}

    for model_name, preds in all_predictions.items():
        errors_temp[model_name] = (preds['temp_pred'] - preds['temp_actual']).cpu().numpy().flatten()
        errors_prcp[model_name] = (preds['prcp_pred'] - preds['prcp_actual']).cpu().numpy().flatten()

    # Get global min/max for consistent x-axis
    temp_error_min = min([e.min() for e in errors_temp.values()])
    temp_error_max = max([e.max() for e in errors_temp.values()])
    prcp_error_min = min([e.min() for e in errors_prcp.values()])
    prcp_error_max = max([e.max() for e in errors_prcp.values()])

    # Temperature error histogram
    ax = axes[0, 0]
    for model_name, errors in errors_temp.items():
        ax.hist(errors, bins=50, alpha=0.5, label=model_name, color=COLORS[model_name], edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Temperature Prediction Error (norm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Temperature Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(temp_error_min, temp_error_max)

    # Temperature error by horizon
    ax = axes[0, 1]
    positions = []
    for model_idx, (model_name, preds) in enumerate(all_predictions.items()):
        for horizon_idx in range(preds['temp_pred'].shape[1]):
            errors_h = (preds['temp_pred'][:, horizon_idx] - preds['temp_actual'][:, horizon_idx]).cpu().numpy()
            pos = horizon_idx * (n_models + 1) + model_idx
            bp = ax.boxplot([errors_h], positions=[pos], widths=0.6,
                            patch_artist=True,
                            boxprops=dict(facecolor=COLORS[model_name], alpha=0.6),
                            medianprops=dict(color='black', linewidth=2))
            if horizon_idx == 0:
                positions.append(pos)

    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_ylabel('Temperature Error (norm)', fontsize=11)
    ax.set_title('Temperature Error by Forecast Horizon', fontsize=12, fontweight='bold')
    ax.set_xticks([i * (n_models + 1) + n_models // 2 for i in range(preds['temp_pred'].shape[1])])
    ax.set_xticklabels([f'+{i + 1}h' for i in range(preds['temp_pred'].shape[1])])
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[name], alpha=0.6, label=name)
                       for name in all_predictions.keys()]
    ax.legend(handles=legend_elements, fontsize=9, loc='best')

    # Precipitation error histogram
    ax = axes[1, 0]
    for model_name, errors in errors_prcp.items():
        ax.hist(errors, bins=50, alpha=0.5, label=model_name, color=COLORS[model_name], edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Precipitation Prediction Error (norm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Precipitation Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(prcp_error_min, prcp_error_max)

    # Precipitation error by horizon
    ax = axes[1, 1]
    for model_idx, (model_name, preds) in enumerate(all_predictions.items()):
        for horizon_idx in range(preds['prcp_pred'].shape[1]):
            errors_h = (preds['prcp_pred'][:, horizon_idx] - preds['prcp_actual'][:, horizon_idx]).cpu().numpy()
            pos = horizon_idx * (n_models + 1) + model_idx
            bp = ax.boxplot([errors_h], positions=[pos], widths=0.6,
                            patch_artist=True,
                            boxprops=dict(facecolor=COLORS[model_name], alpha=0.6),
                            medianprops=dict(color='black', linewidth=2))

    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_ylabel('Precipitation Error (norm)', fontsize=11)
    ax.set_title('Precipitation Error by Forecast Horizon', fontsize=12, fontweight='bold')
    ax.set_xticks([i * (n_models + 1) + n_models // 2 for i in range(preds['prcp_pred'].shape[1])])
    ax.set_xticklabels([f'+{i + 1}h' for i in range(preds['prcp_pred'].shape[1])])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(handles=legend_elements, fontsize=9, loc='best')

    plt.suptitle('Error Analysis - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_models_error_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_models_error_analysis.png")


# ============================================================
# 4. PERFORMANCE COMPARISON BAR CHARTS
# ============================================================

def plot_performance_comparison(all_results):
    """
    Create bar charts comparing model performance

    Shows MAE and RMSE for both horizons and both targets
    """

    models = list(all_results.keys())
    horizons = [3, 6]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(models))
    width = 0.35

    # Temperature MAE
    ax = axes[0, 0]
    mae_3h = [all_results[m]['horizon_3']['temp_mae'] for m in models]
    mae_6h = [all_results[m]['horizon_6']['temp_mae'] for m in models]
    bars1 = ax.bar(x - width / 2, mae_3h, width, label='+3h', color='lightblue', edgecolor='black')
    bars2 = ax.bar(x + width / 2, mae_6h, width, label='+6h', color='darkblue', edgecolor='black')
    ax.set_ylabel('MAE (normalized)', fontsize=11)
    ax.set_title('Temperature - Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Temperature RMSE
    ax = axes[0, 1]
    rmse_3h = [all_results[m]['horizon_3']['temp_rmse'] for m in models]
    rmse_6h = [all_results[m]['horizon_6']['temp_rmse'] for m in models]
    bars1 = ax.bar(x - width / 2, rmse_3h, width, label='+3h', color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width / 2, rmse_6h, width, label='+6h', color='darkred', edgecolor='black')
    ax.set_ylabel('RMSE (normalized)', fontsize=11)
    ax.set_title('Temperature - Root Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Precipitation MAE
    ax = axes[1, 0]
    mae_3h = [all_results[m]['horizon_3']['prcp_mae'] for m in models]
    mae_6h = [all_results[m]['horizon_6']['prcp_mae'] for m in models]
    bars1 = ax.bar(x - width / 2, mae_3h, width, label='+3h', color='lightgreen', edgecolor='black')
    bars2 = ax.bar(x + width / 2, mae_6h, width, label='+6h', color='darkgreen', edgecolor='black')
    ax.set_ylabel('MAE (normalized)', fontsize=11)
    ax.set_title('Precipitation - Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Precipitation RMSE
    ax = axes[1, 1]
    rmse_3h = [all_results[m]['horizon_3']['prcp_rmse'] for m in models]
    rmse_6h = [all_results[m]['horizon_6']['prcp_rmse'] for m in models]
    bars1 = ax.bar(x - width / 2, rmse_3h, width, label='+3h', color='lightyellow', edgecolor='black')
    bars2 = ax.bar(x + width / 2, rmse_6h, width, label='+6h', color='orange', edgecolor='black')
    ax.set_ylabel('RMSE (normalized)', fontsize=11)
    ax.set_title('Precipitation - Root Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Model Performance Comparison (Validation Set)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_models_performance_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_models_performance_comparison.png")


# ============================================================
# MAIN: GENERATE ALL VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 70)

# 1. Sample predictions
print("\n1. Creating sample prediction plots...")
plot_sample_predictions_all_models(all_predictions['val'], validation, SAMPLE_INDICES)

# 2. Scatter plots
print("\n2. Creating scatter plots...")
plot_scatter_all_models(all_predictions['val'])

# 3. Error distributions
print("\n3. Creating error distribution plots...")
plot_error_distributions_all_models(all_predictions['val'])

# 4. Performance comparison
print("\n4. Creating performance comparison charts...")
plot_performance_comparison(all_results['val'])

print("\n" + "=" * 70)
print("ALL VISUALIZATIONS COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  - all_models_sample_predictions.png")
print("  - all_models_scatter_plots.png")
print("  - all_models_error_analysis.png")
print("  - all_models_performance_comparison.png")