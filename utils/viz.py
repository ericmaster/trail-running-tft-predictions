import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Any

def extract_hparams(hparams_file):
    """Extract simple key-value parameters from hparams.yaml without loading complex objects"""
    params = {}
    
    with open(hparams_file, 'r') as f:
        content = f.read()
        
    # Extract simple key-value parameters
    for line in content.split('\n'):
        line = line.strip()
        # Skip empty lines, comments, and complex objects
        if not line or line.startswith('#') or '!!python' in line or line.startswith('-'):
            continue
        
        # Match simple key: value patterns
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$', line)
        if match:
            key, value = match.groups()
            
            # Skip complex nested structures
            if value.strip() in ['{}', '[]', 'null'] or value.startswith('&') or value.startswith('*'):
                continue
            
            # Parse different value types
            value = value.strip()
            if value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            elif value.lower() == 'null':
                params[key] = None
            elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                params[key] = float(value) if '.' in value else int(value)
            else:
                # Keep as string, remove quotes if present
                params[key] = value.strip('"\'')
    
    # Extract nested parameters using regex
    # Look for max_prediction_length
    pred_length_match = re.search(r'max_prediction_length:\s*(\d+)', content)
    if pred_length_match:
        params['max_prediction_length'] = int(pred_length_match.group(1))
    
    # Look for min_encoder_length
    min_enc_match = re.search(r'min_encoder_length:\s*(\d+)', content)
    if min_enc_match:
        params['min_encoder_length'] = int(min_enc_match.group(1))
    
    # Look for randomize_length
    random_match = re.search(r'randomize_length:\s*(true|false|null)', content)
    if random_match:
        params['randomize_length'] = random_match.group(1)
    
    # Look for predict_mode
    predict_match = re.search(r'predict_mode:\s*(true|false)', content)
    if predict_match:
        params['predict_mode'] = predict_match.group(1) == 'true'
    
    # Look for target variables
    target_match = re.search(r'target:\s*&\w+\s*\n((?:\s*-\s*\w+\s*\n)+)', content)
    if target_match:
        targets = re.findall(r'-\s*(\w+)', target_match.group(1))
        params['target_variables'] = targets
    
    return params

def plot_metrics(
    metrics_csv_path=None,
    trainer=None,
    plot_metrics=[["train_loss", "valid_loss"], ["train_acc", "valid_acc"]],
    plot_titles=["Loss", "Accuracy"],
    save_svg_path=None,
):
    """Plot training and validation metrics from a CSV file or a PyTorch Lightning trainer."""
    if metrics_csv_path is None:
        if trainer is None:
            raise ValueError("Either metrics_csv_path or trainer must be provided.")
        else:
            metrics_csv_path = f"{trainer.logger.log_dir}/metrics.csv"
    metrics = pd.read_csv(metrics_csv_path)

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    for i, metrics in enumerate(plot_metrics):
        ax = df_metrics[metrics].plot(
            grid=True,
            legend=True,
            xlabel="Epoch",
            ylabel=plot_titles[i].replace('_', ' ').title(),
            title=f"{metrics[0].replace('_', ' ').title()} vs {metrics[1].replace('_', ' ').title()}",
        )
        if save_svg_path:
            # Save each plot to SVG, appending index if multiple plots
            base, ext = save_svg_path.rsplit('.', 1) if '.' in save_svg_path else (save_svg_path, 'svg')
            svg_file = f"{base}.{ext}"
            fig = ax.get_figure()
            fig.savefig(svg_file, format="svg")
            plt.close(fig)
    if not save_svg_path:
        plt.show()

def visualize_predictions(raw_predictions, batch_id=0, target_idx=0, target_name="Duration", 
                         show_accumulated=False, data_module=None):
    """
    Visualize predictions vs actual values for a specific batch and target variable.
    
    Args:
        raw_predictions: Raw predictions from model.predict()
        batch_id: Batch index to visualize (default: 0)
        target_idx: Target variable index (0=duration_diff, 1=heartRate, 2=temperature, 3=cadence)
        target_name: Name of the target variable for labeling
        show_accumulated: If True, show accumulated values (useful for duration_diff -> total duration)
        data_module: Optional TFTDataModule to extract session_id from
        
    Example:
        visualize_predictions(
            raw_predictions, 
            batch_id=0, 
            target_idx=0, 
            target_name="Duration Diff",
            show_accumulated=True,
            data_module=data_module
        )
    """
    # Try to extract session_id from the batch
    session_id = None
    session_id_encoded = None
    
    try:
        # Get the encoded session ID from groups
        if 'groups' in raw_predictions.x and len(raw_predictions.x['groups']) > 0:
            session_id_encoded = int(raw_predictions.x['groups'][0][batch_id].item())
        
        # Try to decode it using data_module
        if session_id_encoded and data_module is not None:
            # Get the dataframe with session info
            df = None
            if hasattr(data_module, 'full_data') and data_module.full_data is not None:
                df = data_module.full_data
            elif hasattr(data_module, 'train_data') and data_module.train_data is not None:
                df = data_module.train_data
            
            if df is not None and 'session_id' in df.columns and 'session_id_encoded' in df.columns:
                # Find the matching session_id
                matching = df[df['session_id_encoded'] == session_id_encoded]['session_id'].unique()
                if len(matching) > 0:
                    session_id = matching[0]
                    
    except Exception as e:
        pass  # Silently continue without session_id

    # Extract predictions - handle the nested list structure
    if isinstance(raw_predictions.output[0], list):
        # If output[0] is a list, get the specified batch sample, all time steps, specified target
        predictions = raw_predictions.output[0][target_idx][batch_id, :, 0].detach().cpu().numpy()
    else:
        # If it's a tensor directly, get specified batch sample, all time steps, specified target
        predictions = raw_predictions.output[0][batch_id, :, target_idx].detach().cpu().numpy()

    # Extract actuals - select specified batch sample and target
    actuals = raw_predictions.x['decoder_target'][target_idx][batch_id, :].detach().cpu().numpy()

    # Optionally compute accumulated values
    if show_accumulated:
        predictions_display = np.cumsum(predictions)
        actuals_display = np.cumsum(actuals)
        ylabel = f'Accumulated {target_name}'
    else:
        predictions_display = predictions
        actuals_display = actuals
        ylabel = f'{target_name}'

    # Create visualization plot
    plt.figure(figsize=(14, 6))
    time_steps = range(len(predictions_display))

    plt.plot(time_steps, actuals_display, label=f'Actual {target_name}', color='blue', linewidth=2)
    plt.plot(time_steps, predictions_display, label=f'Predicted {target_name}', color='red', linewidth=2, linestyle='--')

    plt.xlabel('Time Steps (5m intervals)')
    plt.ylabel(ylabel)
    
    # Build title with session info
    title = f'{target_name} Predictions vs Actual (Batch {batch_id})'
    if show_accumulated:
        title = f'Accumulated {title}'
    if session_id is not None:
        title += f'\n{session_id}'
    
    plt.title(title, fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate and display error metrics (always on non-accumulated values)
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    # Calculate percentage error if values are non-zero
    non_zero_mask = actuals != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((predictions[non_zero_mask] - actuals[non_zero_mask]) / actuals[non_zero_mask])) * 100
    else:
        mape = None
    
    print(f"\n{'='*60}")
    print(f"Error Metrics for {target_name} (Batch {batch_id})")
    print(f"{'='*60}")
    if session_id is not None:
        print(f"Session: {session_id}")
    elif session_id_encoded is not None:
        print(f"Session ID (encoded): {session_id_encoded}")
    print(f"\nMSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    if mape is not None:
        print(f"MAPE: {mape:.2f}%")
    
    if show_accumulated:
        # Show accumulated totals
        print(f"\nAccumulated Totals:")
        print(f"Actual Total: {actuals_display[-1]:.2f}")
        print(f"Predicted Total: {predictions_display[-1]:.2f}")
        print(f"Absolute Difference: {abs(predictions_display[-1] - actuals_display[-1]):.2f}")
        if actuals_display[-1] != 0:
            print(f"Percentage Difference: {abs(predictions_display[-1] - actuals_display[-1]) / abs(actuals_display[-1]) * 100:.2f}%")
    
    return predictions, actuals


def plot_chunk_predictions(all_predictions, all_actuals, chunk_errors, session_data, 
                          max_chunks=4, save_path='./assets/cold_start_chunks.png',
                          show_elevation=True):
    """
    Visualize per-chunk duration predictions vs actuals with optional elevation profile.
    
    Args:
        all_predictions: Dict with target names as keys and list of predictions as values
        all_actuals: Dict with target names as keys and list of actual values as values
        chunk_errors: List of chunk info dicts with 'chunk', 'start_idx', 'end_idx', 'mae'
        session_data: DataFrame with session data (must have 'distance' and 'altitude' columns)
        max_chunks: Maximum number of chunks to display (default: 4)
        save_path: Path to save the figure (default: './assets/cold_start_chunks.png')
        show_elevation: If True, show elevation profile alongside predictions (default: True)
        
    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects
    """
    if len(chunk_errors) == 0:
        print("No chunk errors to visualize.")
        return None, None
    
    print("\n" + "="*80)
    print("Per-Chunk Duration Predictions" + (" with Elevation Profile" if show_elevation else ""))
    print("="*80)

    num_plots = min(max_chunks, len(chunk_errors))
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5*num_plots))
    if num_plots == 1:
        axes = [axes]

    for i, chunk_info in enumerate(chunk_errors[:max_chunks]):
        ax = axes[i]
        start = chunk_info['start_idx']
        end = chunk_info['end_idx']
        
        # Get predictions and actuals for this chunk range
        chunk_start_in_list = sum(c['end_idx'] - c['start_idx'] for c in chunk_errors[:i])
        chunk_len = end - start
        
        pred_chunk = all_predictions['duration_diff'][chunk_start_in_list:chunk_start_in_list + chunk_len]
        actual_chunk = all_actuals['duration_diff'][chunk_start_in_list:chunk_start_in_list + chunk_len]
        
        steps = range(start, start + len(pred_chunk))
        distance_km = [session_data.iloc[s]['distance']/1000 if s < len(session_data) else s*5/1000 for s in steps]
        
        # Plot duration predictions on primary y-axis
        ax.plot(distance_km[:len(actual_chunk)], actual_chunk, 'b-', label='Actual Duration Diff', linewidth=2)
        ax.plot(distance_km[:len(pred_chunk)], pred_chunk, 'r--', label='Predicted Duration Diff', linewidth=2)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Duration Diff (s)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title(f'Chunk {chunk_info["chunk"]}: Steps {start}-{end} (MAE: {chunk_info["mae"]:.4f})')
        ax.grid(True, alpha=0.3)
        
        # Add elevation profile on secondary y-axis
        if show_elevation and 'altitude' in session_data.columns:
            ax2 = ax.twinx()
            
            # Get elevation data for this chunk
            elevation = [session_data.iloc[s]['altitude'] if s < len(session_data) else np.nan for s in steps]
            
            # Calculate y-limits with padding to show elevation variation clearly
            elev_min = min(elevation)
            elev_max = max(elevation)
            elev_range = elev_max - elev_min
            padding = max(elev_range * 0.1, 10)  # At least 10m padding, or 10% of range
            
            # Plot elevation as filled area (fill from bottom of visible range)
            ax2.fill_between(distance_km, elevation, elev_min - padding,
                           alpha=0.2, color='green', label='Elevation')
            ax2.plot(distance_km, elevation, 'g-', linewidth=1, alpha=0.7)
            ax2.set_ylabel('Elevation (m)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Set y-limits to focus on the actual elevation range
            ax2.set_ylim(elev_min - padding, elev_max + padding)
            
            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes


def plot_accumulated_duration_error(all_predictions, all_actuals, session_data, chunk_boundaries,
                                   session_id=None, save_path='./assets/cold_start_accumulated_error.png'):
    """
    Visualize accumulated duration and error over distance.
    
    Args:
        all_predictions: Dict with target names as keys and list of predictions as values
        all_actuals: Dict with target names as keys and list of actual values as values
        session_data: DataFrame with session data (must have 'distance' column)
        chunk_boundaries: List of chunk boundary indices
        session_id: Optional session ID for title
        save_path: Path to save the figure (default: './assets/cold_start_accumulated_error.png')
        
    Returns:
        dict: Summary statistics including accumulated durations and errors
    """
    print("\n" + "="*80)
    print("Accumulated Duration Error")
    print("="*80)

    # Calculate accumulated durations
    pred_duration_accumulated = np.cumsum(all_predictions['duration_diff'])
    actual_duration_accumulated = np.cumsum(all_actuals['duration_diff'])

    # Convert to minutes for readability
    pred_duration_min = pred_duration_accumulated / 60
    actual_duration_min = actual_duration_accumulated / 60
    error_min = pred_duration_min - actual_duration_min

    # Get distance values
    steps_range = range(len(pred_duration_accumulated))
    distance_km = [session_data.iloc[s]['distance']/1000 if s < len(session_data) else s*5/1000 for s in steps_range]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Accumulated Duration
    ax1 = axes[0]
    ax1.plot(distance_km, actual_duration_min, 'b-', label='Actual Duration', linewidth=2)
    ax1.plot(distance_km, pred_duration_min, 'r--', label='Predicted Duration', linewidth=2)

    # Add chunk boundaries
    for boundary in chunk_boundaries[1:]:  # Skip first boundary at 0
        if boundary < len(distance_km):
            ax1.axvline(x=distance_km[boundary], color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Accumulated Duration (minutes)')
    title = 'Cold-Start Sequential Prediction: Accumulated Duration'
    if session_id:
        title += f'\nSession: {session_id}'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error over distance
    ax2 = axes[1]
    ax2.plot(distance_km, error_min, 'g-', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(distance_km, error_min, 0, alpha=0.3, color='orange')

    # Add chunk boundaries
    for boundary in chunk_boundaries[1:]:
        if boundary < len(distance_km):
            ax2.axvline(x=distance_km[boundary], color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Error (minutes)')
    ax2.set_title('Accumulated Duration Error (Predicted - Actual)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Return summary statistics
    return {
        'pred_duration_accumulated': pred_duration_accumulated,
        'actual_duration_accumulated': actual_duration_accumulated,
        'pred_duration_min': pred_duration_min,
        'actual_duration_min': actual_duration_min,
        'error_min': error_min,
        'distance_km': distance_km
    }


def plot_v1_v2_comparison(
    pred_arr_v1, actual_arr_v1, pred_arr_v2, actual_arr_v2,
    session_id=None, save_path='./assets/v1_v2_comparison.png'
):
    """
    Plot V1 vs V2 model comparison for cold-start inference.
    
    Args:
        pred_arr_v1: V1 predictions array (duration_diff)
        actual_arr_v1: V1 actuals array (duration_diff)
        pred_arr_v2: V2 predictions array (duration_diff)
        actual_arr_v2: V2 actuals array (duration_diff)
        session_id: Optional session ID for title
        save_path: Path to save the figure
        
    Returns:
        dict: Summary statistics for both models
    """
    # Calculate errors
    errors_v1 = pred_arr_v1 - actual_arr_v1
    errors_v2 = pred_arr_v2 - actual_arr_v2
    
    # Calculate statistics
    v1_mae = np.mean(np.abs(errors_v1))
    v1_bias = np.mean(errors_v1)
    v1_rmse = np.sqrt(np.mean(errors_v1**2))
    
    v2_mae = np.mean(np.abs(errors_v2))
    v2_bias = np.mean(errors_v2)
    v2_rmse = np.sqrt(np.mean(errors_v2**2))
    
    # Print comparison table
    print(f"\n{'Metric':<20} {'V1 (SMAPE)':<15} {'V2 (Asym.SMAPE)':<15} {'Change':<15}")
    print("-"*65)
    print(f"{'MAE (s)':<20} {v1_mae:<15.3f} {v2_mae:<15.3f} {v2_mae-v1_mae:+.3f}")
    print(f"{'Bias (s)':<20} {v1_bias:<15.3f} {v2_bias:<15.3f} {v2_bias-v1_bias:+.3f}")
    print(f"{'RMSE (s)':<20} {v1_rmse:<15.3f} {v2_rmse:<15.3f} {v2_rmse-v1_rmse:+.3f}")
    
    # Accumulated duration comparison
    pred_accumulated_v1 = np.cumsum(pred_arr_v1)
    actual_accumulated_v1 = np.cumsum(actual_arr_v1)
    pred_accumulated_v2 = np.cumsum(pred_arr_v2)
    actual_accumulated_v2 = np.cumsum(actual_arr_v2)
    
    # Ensure same length for comparison
    min_len = min(len(pred_arr_v1), len(pred_arr_v2))
    
    print(f"\n{'Accumulated Duration (at step ' + str(min_len) + '):':<40}")
    print(f"  Actual: {actual_accumulated_v1[min_len-1]/60:.1f} min")
    print(f"  V1 Predicted: {pred_accumulated_v1[min_len-1]/60:.1f} min (error: {(pred_accumulated_v1[min_len-1]-actual_accumulated_v1[min_len-1])/60:+.1f} min)")
    print(f"  V2 Predicted: {pred_accumulated_v2[min_len-1]/60:.1f} min (error: {(pred_accumulated_v2[min_len-1]-actual_accumulated_v2[min_len-1])/60:+.1f} min)")
    
    # Final accumulated error percentage
    v1_final_error_pct = (pred_accumulated_v1[min_len-1] - actual_accumulated_v1[min_len-1]) / actual_accumulated_v1[min_len-1] * 100
    v2_final_error_pct = (pred_accumulated_v2[min_len-1] - actual_accumulated_v2[min_len-1]) / actual_accumulated_v2[min_len-1] * 100
    
    print(f"\n{'Final Accumulated Error %:':<40}")
    print(f"  V1: {v1_final_error_pct:+.1f}%")
    print(f"  V2: {v2_final_error_pct:+.1f}%")
    
    # Create 2x2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accumulated Duration Comparison
    ax1 = axes[0, 0]
    distance_km_arr = [i * 5 / 1000 for i in range(min_len)]
    ax1.plot(distance_km_arr, actual_accumulated_v1[:min_len]/60, 'k-', linewidth=2, label='Actual')
    ax1.plot(distance_km_arr, pred_accumulated_v1[:min_len]/60, 'b--', linewidth=1.5, alpha=0.8, label=f'V1 (SMAPE) - Error: {v1_final_error_pct:+.1f}%')
    ax1.plot(distance_km_arr, pred_accumulated_v2[:min_len]/60, 'r--', linewidth=1.5, alpha=0.8, label=f'V2 (Asym.SMAPE) - Error: {v2_final_error_pct:+.1f}%')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Accumulated Duration (min)')
    ax1.set_title('Accumulated Duration: V1 vs V2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Error Over Distance
    ax2 = axes[0, 1]
    errors_v1_acc = (pred_accumulated_v1[:min_len] - actual_accumulated_v1[:min_len]) / 60
    errors_v2_acc = (pred_accumulated_v2[:min_len] - actual_accumulated_v2[:min_len]) / 60
    ax2.plot(distance_km_arr, errors_v1_acc, 'b-', linewidth=1.5, alpha=0.8, label='V1 Error')
    ax2.plot(distance_km_arr, errors_v2_acc, 'r-', linewidth=1.5, alpha=0.8, label='V2 Error')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Accumulated Error (min)')
    ax2.set_title('Accumulated Error Over Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Per-Step Bias Distribution
    ax3 = axes[1, 0]
    ax3.hist(errors_v1[:min_len], bins=50, alpha=0.5, label=f'V1 (bias={v1_bias:.3f}s)', color='blue')
    ax3.hist(errors_v2[:min_len], bins=50, alpha=0.5, label=f'V2 (bias={v2_bias:.3f}s)', color='red')
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax3.axvline(x=v1_bias, color='blue', linestyle='-', alpha=0.8, linewidth=2)
    ax3.axvline(x=v2_bias, color='red', linestyle='-', alpha=0.8, linewidth=2)
    ax3.set_xlabel('Per-Step Error (s)')
    ax3.set_ylabel('Count')
    ax3.set_title('Per-Step Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Rolling Bias (50-step window)
    ax4 = axes[1, 1]
    window = 50
    rolling_bias_v1 = pd.Series(errors_v1[:min_len]).rolling(window=window).mean()
    rolling_bias_v2 = pd.Series(errors_v2[:min_len]).rolling(window=window).mean()
    ax4.plot(distance_km_arr, rolling_bias_v1, 'b-', linewidth=1.5, alpha=0.8, label='V1 Rolling Bias')
    ax4.plot(distance_km_arr, rolling_bias_v2, 'r-', linewidth=1.5, alpha=0.8, label='V2 Rolling Bias')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Distance (km)')
    ax4.set_ylabel(f'Rolling Bias (s, {window}-step window)')
    ax4.set_title('Rolling Bias Over Distance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    title = 'Cold-Start Inference: V1 (SMAPE) vs V2 (Asymmetric SMAPE)'
    if session_id:
        title += f'\nSession: {session_id}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if abs(v2_bias) < abs(v1_bias):
        duration_improvement = (abs(v1_bias) - abs(v2_bias)) / abs(v1_bias) * 100
        print(f"✅ V2 model REDUCED bias by {duration_improvement:.1f}%")
        print(f"   V1 bias: {v1_bias:+.3f}s → V2 bias: {v2_bias:+.3f}s")
    else:
        print(f"❌ V2 model did NOT reduce bias")
        print(f"   V1 bias: {v1_bias:+.3f}s → V2 bias: {v2_bias:+.3f}s")
    
    if abs(v2_final_error_pct) < abs(v1_final_error_pct):
        print(f"✅ V2 model REDUCED accumulated error")
        print(f"   V1: {v1_final_error_pct:+.1f}% → V2: {v2_final_error_pct:+.1f}%")
    else:
        print(f"❌ V2 model did NOT reduce accumulated error")
        print(f"   V1: {v1_final_error_pct:+.1f}% → V2: {v2_final_error_pct:+.1f}%")
    
    return {
        'v1_mae': v1_mae, 'v1_bias': v1_bias, 'v1_rmse': v1_rmse,
        'v2_mae': v2_mae, 'v2_bias': v2_bias, 'v2_rmse': v2_rmse,
        'v1_final_error_pct': v1_final_error_pct,
        'v2_final_error_pct': v2_final_error_pct,
        'min_len': min_len
    }


def plot_prediction_diagnostic(
    v1_pred, v1_actual, v2_pred, v2_actual,
    n_show=200, save_path='./assets/prediction_diagnostic.png'
):
    """
    Diagnostic plot to compare V1 vs V2 prediction variance and distribution.
    
    Args:
        v1_pred: V1 predictions array
        v1_actual: V1 actuals array  
        v2_pred: V2 predictions array
        v2_actual: V2 actuals array
        n_show: Number of steps to show in time series (default: 200)
        save_path: Path to save the figure
    """
    min_len = min(len(v1_pred), len(v2_pred))
    
    # Print statistics
    print("="*80)
    print("DIRECT COMPARISON OF V1 vs V2 PREDICTION ARRAYS")
    print("="*80)
    
    print(f"\nArray lengths:")
    print(f"  V1 predictions: {len(v1_pred)}")
    print(f"  V1 actuals:     {len(v1_actual)}")
    print(f"  V2 predictions: {len(v2_pred)}")
    print(f"  V2 actuals:     {len(v2_actual)}")
    print(f"\nUsing first {min_len} steps for comparison")
    
    print(f"\n{'Statistic':<25} {'V1 Pred':<12} {'V2 Pred':<12} {'Actual':<12}")
    print("-"*60)
    print(f"{'Mean':<25} {np.mean(v1_pred[:min_len]):<12.4f} {np.mean(v2_pred[:min_len]):<12.4f} {np.mean(v1_actual[:min_len]):<12.4f}")
    print(f"{'Std':<25} {np.std(v1_pred[:min_len]):<12.4f} {np.std(v2_pred[:min_len]):<12.4f} {np.std(v1_actual[:min_len]):<12.4f}")
    print(f"{'Min':<25} {np.min(v1_pred[:min_len]):<12.4f} {np.min(v2_pred[:min_len]):<12.4f} {np.min(v1_actual[:min_len]):<12.4f}")
    print(f"{'Max':<25} {np.max(v1_pred[:min_len]):<12.4f} {np.max(v2_pred[:min_len]):<12.4f} {np.max(v1_actual[:min_len]):<12.4f}")
    print(f"{'Range':<25} {np.ptp(v1_pred[:min_len]):<12.4f} {np.ptp(v2_pred[:min_len]):<12.4f} {np.ptp(v1_actual[:min_len]):<12.4f}")
    
    # Check for constant predictions
    unique_v1 = len(np.unique(np.round(v1_pred[:min_len], 2)))
    unique_v2 = len(np.unique(np.round(v2_pred[:min_len], 2)))
    print(f"\nUnique V1 predictions (rounded to 0.01): {unique_v1}")
    print(f"Unique V2 predictions (rounded to 0.01): {unique_v2}")
    
    # Correlation with actual
    v1_corr = np.corrcoef(v1_pred[:min_len], v1_actual[:min_len])[0,1]
    v2_corr = np.corrcoef(v2_pred[:min_len], v2_actual[:min_len])[0,1]
    print(f"\nCorrelation with actual:")
    print(f"  V1: {v1_corr:.4f}")
    print(f"  V2: {v2_corr:.4f}")
    
    # Visual check - 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: Time series of first n_show steps
    ax1 = axes[0]
    n = min(n_show, min_len)
    ax1.plot(v1_actual[:n], 'k-', alpha=0.7, label='Actual')
    ax1.plot(v1_pred[:n], 'b-', alpha=0.7, label='V1')
    ax1.plot(v2_pred[:n], 'r-', alpha=0.7, label='V2')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('duration_diff (s)')
    ax1.set_title(f'First {n} steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Scatter plot - Pred vs Actual
    ax2 = axes[1]
    ax2.scatter(v1_actual[:min_len], v1_pred[:min_len], alpha=0.3, s=10, label='V1')
    ax2.scatter(v2_actual[:min_len], v2_pred[:min_len], alpha=0.3, s=10, label='V2')
    max_val = max(np.max(v1_actual[:min_len]), np.max(v1_pred[:min_len]), np.max(v2_pred[:min_len]))
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Pred vs Actual (corr V1={v1_corr:.3f}, V2={v2_corr:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Distribution comparison
    ax3 = axes[2]
    ax3.hist(v1_pred[:min_len], bins=50, alpha=0.5, label=f'V1 (std={np.std(v1_pred[:min_len]):.2f})', density=True)
    ax3.hist(v2_pred[:min_len], bins=50, alpha=0.5, label=f'V2 (std={np.std(v2_pred[:min_len]):.2f})', density=True)
    ax3.hist(v1_actual[:min_len], bins=50, alpha=0.3, label=f'Actual (std={np.std(v1_actual[:min_len]):.2f})', density=True, color='green')
    ax3.set_xlabel('duration_diff (s)')
    ax3.set_ylabel('Density')
    ax3.set_title('Prediction Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'unique_v1': unique_v1,
        'unique_v2': unique_v2,
        'v1_corr': v1_corr,
        'v2_corr': v2_corr,
        'min_len': min_len
    }


def plot_session_overview(
    sample_data: pd.DataFrame,
    sample_file: str,
    columns: List[str] = None,
    smoothing_window: int = 100,
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
):
    """
    Plot normalized session variables over distance for a training/test session overview.
    
    Args:
        sample_data: DataFrame with session data (must have 'distance' column)
        sample_file: Filename for the plot title
        columns: List of columns to plot. Default: duration, heartRate, speed, altitude, 
                 temperature, cadence, elevation_gain, elevation_loss
        smoothing_window: Rolling average window for smoothing. Default: 100
        figsize: Figure size tuple. Default: (14, 7)
        save_path: Optional path to save the figure
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    if columns is None:
        columns = ['duration', 'heartRate', 'speed', 'altitude', 'temperature', 
                   'cadence', 'elevation_gain', 'elevation_loss']
    
    # Filter to columns that exist in the data
    columns = [col for col in columns if col in sample_data.columns]
    
    # Normalize sample data (scale between 0 and 1 for better visualization)
    normalized_cols = {}
    for col in columns:
        col_min = sample_data[col].min()
        col_max = sample_data[col].max()
        if col_max - col_min > 0:
            normalized_cols[col] = (sample_data[col] - col_min) / (col_max - col_min)
        else:
            normalized_cols[col] = sample_data[col]
    
    # Smooth noisy data using rolling average
    noisy_cols = ['speed', 'heartRate', 'cadence']
    for col in noisy_cols:
        if col in normalized_cols:
            normalized_cols[col] = normalized_cols[col].rolling(
                window=smoothing_window, min_periods=10
            ).mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    distance_km = sample_data['distance'] / 1000
    
    for col in normalized_cols:
        ax.plot(distance_km, normalized_cols[col], label=col.capitalize())
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Sample Training Session ({sample_file})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, ax


def plot_bias_analysis(
    predictions: np.ndarray,
    actuals: np.ndarray,
    session_data: pd.DataFrame,
    chunk_errors: List[Dict],
    train_data: pd.DataFrame,
    synthetic_encoder: Dict,
    calculate_weighted_first_sample_fn,
    save_path: str = './assets/bias_analysis.png'
) -> Dict[str, Any]:
    """
    Comprehensive bias analysis visualization for cold-start inference.
    
    Args:
        predictions: Array of duration_diff predictions
        actuals: Array of duration_diff actual values
        session_data: DataFrame with session data
        chunk_errors: List of chunk info dicts with 'chunk', 'start_idx', 'end_idx', 'mae'
        train_data: DataFrame with training data
        synthetic_encoder: Dict with synthetic encoder values
        calculate_weighted_first_sample_fn: Function to calculate weighted first sample
        save_path: Path to save the figure
        
    Returns:
        dict: Statistics including errors, biases by terrain/progress
    """
    pred_arr = np.array(predictions)
    actual_arr = np.array(actuals)
    errors = pred_arr - actual_arr
    
    stats = {
        'mean_pred': pred_arr.mean(),
        'mean_actual': actual_arr.mean(),
        'mean_error': errors.mean(),
        'median_error': np.median(errors),
        'std_error': errors.std()
    }
    
    # Bias percentage
    bias_pct = (pred_arr.mean() - actual_arr.mean()) / actual_arr.mean() * 100
    stats['bias_pct'] = bias_pct
    
    print("\nOVERALL STATISTICS")
    print(f"Mean Predicted duration_diff: {pred_arr.mean():.4f} seconds/5m")
    print(f"Mean Actual duration_diff:    {actual_arr.mean():.4f} seconds/5m")
    print(f"Mean Error (Pred - Actual):   {errors.mean():.4f} seconds/5m")
    print(f"Median Error:                 {np.median(errors):.4f} seconds/5m")
    print(f"Std of Errors:                {errors.std():.4f}")
    print(f"\nBIAS: {bias_pct:.2f}% (negative = under-prediction)")
    
    # Per-chunk analysis
    print("\nPER-CHUNK BIAS ANALYSIS")
    print("-"*60)
    chunk_biases = []
    for chunk in chunk_errors:
        chunk_start = chunk['start_idx']
        chunk_end = chunk['end_idx']
        chunk_len = chunk_end - chunk_start
        
        chunk_start_in_list = sum(
            c['end_idx'] - c['start_idx'] 
            for c in chunk_errors if c['chunk'] < chunk['chunk']
        )
        chunk_preds = pred_arr[chunk_start_in_list:chunk_start_in_list + chunk_len]
        chunk_actuals = actual_arr[chunk_start_in_list:chunk_start_in_list + chunk_len]
        
        chunk_bias = chunk_preds.mean() - chunk_actuals.mean()
        chunk_bias_pct = chunk_bias / chunk_actuals.mean() * 100 if chunk_actuals.mean() != 0 else 0
        
        chunk_biases.append({
            'chunk': chunk['chunk'],
            'bias': chunk_bias,
            'bias_pct': chunk_bias_pct
        })
        
        print(f"Chunk {chunk['chunk']:2d}: Bias = {chunk_bias:+.4f}s ({chunk_bias_pct:+.1f}%) | "
              f"Pred mean: {chunk_preds.mean():.3f} | Actual mean: {chunk_actuals.mean():.3f}")
    
    stats['chunk_biases'] = chunk_biases
    
    # Analyze by terrain (elevation)
    print("\nBIAS BY TERRAIN (Elevation Change)")
    print("-"*60)
    
    elev_diffs = session_data['elevation_diff'].values[:len(pred_arr)]
    climbing_mask = elev_diffs > 0.5
    flat_mask = (elev_diffs >= -0.5) & (elev_diffs <= 0.5)
    descending_mask = elev_diffs < -0.5
    
    terrain_biases = {}
    for name, mask in [("Climbing", climbing_mask), ("Flat", flat_mask), ("Descending", descending_mask)]:
        if mask.sum() > 0:
            bias = (pred_arr[mask] - actual_arr[mask]).mean()
            bias_pct_terrain = bias / actual_arr[mask].mean() * 100 if actual_arr[mask].mean() != 0 else 0
            terrain_biases[name] = {'bias': bias, 'bias_pct': bias_pct_terrain, 'count': mask.sum()}
            print(f"{name:12s}: Bias = {bias:+.4f}s ({bias_pct_terrain:+.1f}%) | N={mask.sum()} steps | "
                  f"Actual mean: {actual_arr[mask].mean():.3f}s")
    
    stats['terrain_biases'] = terrain_biases
    
    # Analyze by distance (fatigue effect)
    print("\nBIAS BY RACE PROGRESS (Fatigue)")
    print("-"*60)
    quartiles = [0, 0.25, 0.5, 0.75, 1.0]
    progress_biases = []
    for i in range(len(quartiles)-1):
        start_pct = quartiles[i]
        end_pct = quartiles[i+1]
        start_idx = int(len(pred_arr) * start_pct)
        end_idx = int(len(pred_arr) * end_pct)
        
        q_preds = pred_arr[start_idx:end_idx]
        q_actuals = actual_arr[start_idx:end_idx]
        bias = (q_preds - q_actuals).mean()
        bias_pct_progress = bias / q_actuals.mean() * 100 if q_actuals.mean() != 0 else 0
        
        dist_start = session_data.iloc[start_idx]['distance']/1000 if start_idx < len(session_data) else 0
        dist_end = session_data.iloc[min(end_idx-1, len(session_data)-1)]['distance']/1000
        
        progress_biases.append({
            'quartile': f"{int(start_pct*100)}-{int(end_pct*100)}%",
            'bias': bias,
            'bias_pct': bias_pct_progress,
            'dist_range': (dist_start, dist_end)
        })
        
        print(f"{int(start_pct*100):2d}-{int(end_pct*100):3d}% ({dist_start:.1f}-{dist_end:.1f}km): "
              f"Bias = {bias:+.4f}s ({bias_pct_progress:+.1f}%) | Actual mean: {q_actuals.mean():.3f}s")
    
    stats['progress_biases'] = progress_biases
    
    # Synthetic encoder analysis
    print("\nSYNTHETIC ENCODER ANALYSIS")
    print("-"*60)
    print("The cold-start encoder uses weighted averages from training sessions.")
    print("If training sessions were generally faster, encoder values may be too optimistic.\n")
    
    print("Synthetic encoder vs actual first step values:")
    first_step = session_data.iloc[0]
    for var in ['heartRate', 'speed', 'cadence', 'duration_diff']:
        if var in synthetic_encoder and var in first_step:
            synth_val = synthetic_encoder[var]
            actual_val = first_step[var]
            diff_pct = (synth_val - actual_val) / actual_val * 100 if actual_val != 0 else 0
            print(f"  {var:20s}: Synthetic={synth_val:.2f}, Actual={actual_val:.2f} ({diff_pct:+.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error distribution
    ax1 = axes[0, 0]
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=errors.mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Mean Error: {errors.mean():.3f}')
    ax1.set_xlabel('Prediction Error (seconds/5m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error vs actual value (bias correlation with pace)
    ax2 = axes[0, 1]
    ax2.scatter(actual_arr, errors, alpha=0.3, s=5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    z = np.polyfit(actual_arr, errors, 1)
    p = np.poly1d(z)
    ax2.plot(sorted(actual_arr), p(sorted(actual_arr)), 'g-', linewidth=2, 
             label=f'Trend (slope={z[0]:.3f})')
    ax2.set_xlabel('Actual Duration Diff (s)')
    ax2.set_ylabel('Error (Pred - Actual)')
    ax2.set_title('Error vs Actual Value (Bias Analysis)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative error over distance
    ax3 = axes[1, 0]
    cumulative_error = np.cumsum(errors)
    distance_km_arr = [
        session_data.iloc[i]['distance']/1000 if i < len(session_data) else i*5/1000 
        for i in range(len(errors))
    ]
    ax3.plot(distance_km_arr, cumulative_error/60, 'b-', linewidth=1.5)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax3.fill_between(distance_km_arr, cumulative_error/60, 0, alpha=0.3)
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Cumulative Error (minutes)')
    ax3.set_title('Cumulative Error Over Distance')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling bias (moving average of error)
    ax4 = axes[1, 1]
    window = 100
    rolling_error = np.convolve(errors, np.ones(window)/window, mode='valid')
    rolling_dist = distance_km_arr[window//2:-window//2+1]
    ax4.plot(rolling_dist, rolling_error, 'b-', linewidth=1.5)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.fill_between(rolling_dist, rolling_error, 0, alpha=0.3)
    ax4.set_xlabel('Distance (km)')
    ax4.set_ylabel(f'Rolling Mean Error ({window}-step window)')
    ax4.set_title('Rolling Bias Over Distance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return stats


def plot_train_test_distribution(
    train_data: pd.DataFrame,
    test_session_data: pd.DataFrame,
    session_id: str,
    test_session_ids: List[str],
    save_path: str = './assets/train_test_distribution_comparison.png'
) -> Dict[str, Any]:
    """
    Compare training and test session distributions for duration_diff and speed.
    
    Args:
        train_data: DataFrame with all training data
        test_session_data: DataFrame with single test session data
        session_id: Current test session ID
        test_session_ids: List of all test session IDs
        save_path: Path to save the figure
        
    Returns:
        dict: Comparison statistics
    """
    train_duration_diff = train_data['duration_diff'].values
    test_duration_diff = test_session_data['duration_diff'].values
    
    print("\nDURATION_DIFF DISTRIBUTION COMPARISON")
    print("-"*60)
    print(f"{'Metric':<20} {'Training':<15} {'Test Session':<15} {'Difference':<15}")
    print("-"*60)
    print(f"{'Mean':<20} {train_duration_diff.mean():<15.4f} {test_duration_diff.mean():<15.4f} "
          f"{test_duration_diff.mean() - train_duration_diff.mean():<+15.4f}")
    print(f"{'Median':<20} {np.median(train_duration_diff):<15.4f} {np.median(test_duration_diff):<15.4f} "
          f"{np.median(test_duration_diff) - np.median(train_duration_diff):<+15.4f}")
    print(f"{'Std':<20} {train_duration_diff.std():<15.4f} {test_duration_diff.std():<15.4f} "
          f"{test_duration_diff.std() - train_duration_diff.std():<+15.4f}")
    print(f"{'25th percentile':<20} {np.percentile(train_duration_diff, 25):<15.4f} "
          f"{np.percentile(test_duration_diff, 25):<15.4f}")
    print(f"{'75th percentile':<20} {np.percentile(train_duration_diff, 75):<15.4f} "
          f"{np.percentile(test_duration_diff, 75):<15.4f}")
    print(f"{'95th percentile':<20} {np.percentile(train_duration_diff, 95):<15.4f} "
          f"{np.percentile(test_duration_diff, 95):<15.4f}")
    
    # Calculate slowdown factor
    slowdown_factor = test_duration_diff.mean() / train_duration_diff.mean()
    print(f"\nTest session is {slowdown_factor:.2f}x slower than training average")
    print(f"   This explains ~{(1 - 1/slowdown_factor)*100:.1f}% of the under-prediction bias")
    
    # Speed comparison
    train_speed = train_data['speed'].values
    test_speed = test_session_data['speed'].values
    print(f"\nSPEED COMPARISON")
    print(f"{'Training avg speed:':<25} {train_speed.mean():.3f} m/s ({train_speed.mean()*3.6:.2f} km/h)")
    print(f"{'Test session avg speed:':<25} {test_speed.mean():.3f} m/s ({test_speed.mean()*3.6:.2f} km/h)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Duration diff distributions
    ax1 = axes[0]
    ax1.hist(train_duration_diff, bins=100, alpha=0.5, 
             label=f'Training (mean={train_duration_diff.mean():.2f})', density=True)
    ax1.hist(test_duration_diff, bins=50, alpha=0.5, 
             label=f'Test Session (mean={test_duration_diff.mean():.2f})', density=True)
    ax1.axvline(train_duration_diff.mean(), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(test_duration_diff.mean(), color='orange', linestyle='--', linewidth=2)
    ax1.set_xlabel('Duration Diff (seconds/5m)')
    ax1.set_ylabel('Density')
    ax1.set_title('Duration Diff Distribution: Training vs Test Session')
    ax1.legend()
    ax1.set_xlim(0, 15)
    ax1.grid(True, alpha=0.3)
    
    # Speed distributions
    ax2 = axes[1]
    ax2.hist(train_speed, bins=100, alpha=0.5, 
             label=f'Training (mean={train_speed.mean():.2f})', density=True)
    ax2.hist(test_speed, bins=50, alpha=0.5, 
             label=f'Test Session (mean={test_speed.mean():.2f})', density=True)
    ax2.axvline(train_speed.mean(), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(test_speed.mean(), color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('Speed (m/s)')
    ax2.set_ylabel('Density')
    ax2.set_title('Speed Distribution: Training vs Test Session')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'slowdown_factor': slowdown_factor,
        'train_mean_duration': train_duration_diff.mean(),
        'test_mean_duration': test_duration_diff.mean(),
        'train_mean_speed': train_speed.mean(),
        'test_mean_speed': test_speed.mean()
    }


def plot_custom_encoder_comparison(
    orig_predictions: Dict[str, List],
    custom_predictions: Dict[str, List],
    actuals: Dict[str, List],
    session_data: pd.DataFrame,
    orig_encoder: Dict,
    custom_encoder: Dict,
    save_path: str = './assets/encoder_comparison.png'
) -> Dict[str, Any]:
    """
    Visualize comparison between original and custom encoder cold-start inference.
    
    Args:
        orig_predictions: Dict with 'duration_diff' key containing original predictions
        custom_predictions: Dict with 'duration_diff' key containing custom encoder predictions
        actuals: Dict with 'duration_diff' key containing actual values
        session_data: DataFrame with session data
        orig_encoder: Dict with original synthetic encoder values
        custom_encoder: Dict with custom synthetic encoder values
        save_path: Path to save the figure
        
    Returns:
        dict: Comparison statistics
    """
    pred_orig = np.array(orig_predictions['duration_diff'])
    pred_custom = np.array(custom_predictions['duration_diff'])
    actual = np.array(actuals['duration_diff'])
    
    orig_error = pred_orig - actual
    custom_error = pred_custom - actual
    
    orig_accumulated = np.cumsum(pred_orig)
    custom_accumulated = np.cumsum(pred_custom)
    actual_accumulated = np.cumsum(actual)
    
    print("\n" + "="*80)
    print("COMPARISON: Original vs Custom Initial Values")
    print("="*80)
    print(f"Original encoder: HR={orig_encoder.get('heartRate', 'N/A'):.1f}, "
          f"Speed={orig_encoder.get('speed', 'N/A'):.2f}")
    print(f"Custom encoder: HR={custom_encoder.get('heartRate', 'N/A'):.1f}, "
          f"Speed={custom_encoder.get('speed', 'N/A'):.2f}")
    
    print(f"\n{'Metric':<35} {'Original':<20} {'Custom':<20} {'Improvement':<15}")
    print("-"*90)
    print(f"{'Mean Prediction (s/5m)':<35} {pred_orig.mean():<20.4f} {pred_custom.mean():<20.4f} "
          f"{pred_custom.mean() - pred_orig.mean():<+15.4f}")
    print(f"{'Mean Error (s/5m)':<35} {orig_error.mean():<20.4f} {custom_error.mean():<20.4f} "
          f"{abs(custom_error.mean()) - abs(orig_error.mean()):<+15.4f}")
    print(f"{'MAE (s/5m)':<35} {np.abs(orig_error).mean():<20.4f} {np.abs(custom_error).mean():<20.4f} "
          f"{np.abs(custom_error).mean() - np.abs(orig_error).mean():<+15.4f}")
    print(f"{'Bias %':<35} {(pred_orig.mean() - actual.mean())/actual.mean()*100:<20.2f}% "
          f"{(pred_custom.mean() - actual.mean())/actual.mean()*100:<20.2f}%")
    
    print(f"\n{'Final Accumulated Duration (min)':<35}")
    print(f"{'  Actual':<35} {actual_accumulated[-1]/60:<20.2f}")
    print(f"{'  Original Prediction':<35} {orig_accumulated[-1]/60:<20.2f} "
          f"{'(Error: ' + f'{(orig_accumulated[-1] - actual_accumulated[-1])/60:.2f} min)':<20}")
    print(f"{'  Custom Prediction':<35} {custom_accumulated[-1]/60:<20.2f} "
          f"{'(Error: ' + f'{(custom_accumulated[-1] - actual_accumulated[-1])/60:.2f} min)':<20}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    distance_km_arr = [
        session_data.iloc[i]['distance']/1000 if i < len(session_data) else i*5/1000 
        for i in range(len(actual))
    ]
    
    # Plot 1: Accumulated Duration Comparison
    ax1 = axes[0, 0]
    ax1.plot(distance_km_arr, actual_accumulated/60, 'b-', label='Actual', linewidth=2)
    ax1.plot(distance_km_arr, orig_accumulated/60, 'r--', 
             label=f'Original (HR={orig_encoder.get("heartRate", 0):.0f})', linewidth=2, alpha=0.7)
    ax1.plot(distance_km_arr, custom_accumulated/60, 'g-', 
             label=f'Custom (HR={custom_encoder.get("heartRate", 0):.0f})', linewidth=2)
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Accumulated Duration (min)')
    ax1.set_title('Accumulated Duration: Original vs Custom Encoder')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Error Comparison
    ax2 = axes[0, 1]
    ax2.plot(distance_km_arr, np.cumsum(orig_error)/60, 'r-', label='Original Error', linewidth=2)
    ax2.plot(distance_km_arr, np.cumsum(custom_error)/60, 'g-', label='Custom Error', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Cumulative Error (min)')
    ax2.set_title('Cumulative Error: Original vs Custom')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling Error Comparison
    ax3 = axes[1, 0]
    window = 100
    if len(orig_error) > window:
        rolling_orig = np.convolve(orig_error, np.ones(window)/window, mode='valid')
        rolling_custom = np.convolve(custom_error, np.ones(window)/window, mode='valid')
        rolling_dist = distance_km_arr[window//2:-window//2+1]
        ax3.plot(rolling_dist, rolling_orig, 'r-', label='Original', linewidth=1.5)
        ax3.plot(rolling_dist, rolling_custom, 'g-', label='Custom', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel(f'Rolling Mean Error ({window}-step window)')
    ax3.set_title('Rolling Bias: Original vs Custom')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error Distribution Comparison
    ax4 = axes[1, 1]
    ax4.hist(orig_error, bins=50, alpha=0.5, 
             label=f'Original (mean={orig_error.mean():.3f})', color='red')
    ax4.hist(custom_error, bins=50, alpha=0.5, 
             label=f'Custom (mean={custom_error.mean():.3f})', color='green')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.axvline(x=orig_error.mean(), color='red', linestyle='-', linewidth=2)
    ax4.axvline(x=custom_error.mean(), color='green', linestyle='-', linewidth=2)
    ax4.set_xlabel('Prediction Error (s/5m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution: Original vs Custom')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary
    improvement = (abs(orig_error.mean()) - abs(custom_error.mean())) / abs(orig_error.mean()) * 100
    duration_improvement = abs(orig_accumulated[-1] - actual_accumulated[-1]) - \
                          abs(custom_accumulated[-1] - actual_accumulated[-1])
    
    print("\n" + "="*80)
    print("SUMMARY: Impact of Initial Heart Rate and Speed on Predictions")
    print("="*80)
    print(f"""
By using more realistic initial values:
  - Mean error improved by {improvement:.1f}%
  - Final duration error reduced by {duration_improvement/60:.1f} minutes
  - Bias shifted from {(pred_orig.mean() - actual.mean())/actual.mean()*100:.1f}% to """
          f"""{(pred_custom.mean() - actual.mean())/actual.mean()*100:.1f}%
""")
    
    return {
        'orig_mean_error': orig_error.mean(),
        'custom_mean_error': custom_error.mean(),
        'improvement_pct': improvement,
        'duration_improvement_min': duration_improvement/60
    }


def plot_cold_start_error_distribution(
    v1_results: List[Dict],
    v2_results: List[Dict],
    save_path: str = './assets/cold_start_error_distribution.png'
) -> Dict[str, Any]:
    """
    Visualize error distribution across all test sessions for V1 vs V2 models.
    
    Args:
        v1_results: List of evaluation result dicts for V1 model
        v2_results: List of evaluation result dicts for V2 model
        save_path: Path to save the figure
        
    Returns:
        dict: Summary statistics and comparison
    """
    df_v1 = pd.DataFrame(v1_results)
    df_v2 = pd.DataFrame(v2_results)
    
    if len(df_v1) == 0 or len(df_v2) == 0:
        print("\nNo results collected!")
        return {}
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nV1 Model (SMAPE) - {len(df_v1)} sessions:")
    print(f"  Mean Final Error: {df_v1['final_error_pct'].mean():+.1f}%")
    print(f"  Std Final Error:  {df_v1['final_error_pct'].std():.1f}%")
    print(f"  Min Final Error:  {df_v1['final_error_pct'].min():+.1f}%")
    print(f"  Max Final Error:  {df_v1['final_error_pct'].max():+.1f}%")
    print(f"  Mean Bias:        {df_v1['bias'].mean():+.3f}s")
    print(f"  Mean MAE:         {df_v1['mae'].mean():.3f}s")
    
    print(f"\nV2 Model (Asymmetric SMAPE) - {len(df_v2)} sessions:")
    print(f"  Mean Final Error: {df_v2['final_error_pct'].mean():+.1f}%")
    print(f"  Std Final Error:  {df_v2['final_error_pct'].std():.1f}%")
    print(f"  Min Final Error:  {df_v2['final_error_pct'].min():+.1f}%")
    print(f"  Max Final Error:  {df_v2['final_error_pct'].max():+.1f}%")
    print(f"  Mean Bias:        {df_v2['bias'].mean():+.3f}s")
    print(f"  Mean MAE:         {df_v2['mae'].mean():.3f}s")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of Final Error Percentage
    ax1 = axes[0, 0]
    ax1.hist(df_v1['final_error_pct'], bins=15, alpha=0.6, 
             label=f'V1 (μ={df_v1["final_error_pct"].mean():+.1f}%)', color='blue')
    ax1.hist(df_v2['final_error_pct'], bins=15, alpha=0.6, 
             label=f'V2 (μ={df_v2["final_error_pct"].mean():+.1f}%)', color='red')
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=df_v1['final_error_pct'].mean(), color='blue', linestyle='-', alpha=0.8, linewidth=2)
    ax1.axvline(x=df_v2['final_error_pct'].mean(), color='red', linestyle='-', alpha=0.8, linewidth=2)
    ax1.set_xlabel('Final Accumulated Error (%)')
    ax1.set_ylabel('Number of Sessions')
    ax1.set_title('Distribution of Cold-Start Error Across Test Sessions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    box_data = [df_v1['final_error_pct'].values, df_v2['final_error_pct'].values]
    bp = ax2.boxplot(box_data, labels=['V1 (SMAPE)', 'V2 (Asym.SMAPE)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Final Accumulated Error (%)')
    ax2.set_title('Error Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-session error comparison
    ax3 = axes[1, 0]
    df_merged = df_v1[['session_id', 'final_error_pct']].merge(
        df_v2[['session_id', 'final_error_pct']], 
        on='session_id', 
        suffixes=('_v1', '_v2')
    )
    df_merged = df_merged.sort_values('final_error_pct_v1')
    x_pos = range(len(df_merged))
    ax3.bar([x - 0.2 for x in x_pos], df_merged['final_error_pct_v1'], 
            width=0.4, label='V1', color='blue', alpha=0.7)
    ax3.bar([x + 0.2 for x in x_pos], df_merged['final_error_pct_v2'], 
            width=0.4, label='V2', color='red', alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Session (sorted by V1 error)')
    ax3.set_ylabel('Final Accumulated Error (%)')
    ax3.set_title('Per-Session Error Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error vs Session Length
    ax4 = axes[1, 1]
    ax4.scatter(df_v1['session_length'] * 5 / 1000, df_v1['final_error_pct'], 
                alpha=0.6, label='V1', color='blue', s=50)
    ax4.scatter(df_v2['session_length'] * 5 / 1000, df_v2['final_error_pct'], 
                alpha=0.6, label='V2', color='red', s=50)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Session Distance (km)')
    ax4.set_ylabel('Final Accumulated Error (%)')
    ax4.set_title('Error vs Session Distance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Cold-Start Error Analysis Across All Test Sessions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate improvement summary
    v1_mean = df_v1['final_error_pct'].mean()
    v2_mean = df_v2['final_error_pct'].mean()
    
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    
    if abs(v2_mean) < abs(v1_mean):
        improvement = (abs(v1_mean) - abs(v2_mean)) / abs(v1_mean) * 100
        print(f"V2 reduced mean error magnitude by {improvement:.1f}%")
        print(f"   V1: {v1_mean:+.1f}% → V2: {v2_mean:+.1f}%")
    else:
        print(f"V2 did not reduce mean error magnitude")
        print(f"   V1: {v1_mean:+.1f}% → V2: {v2_mean:+.1f}%")
    
    # Under-prediction counts
    v1_under = (df_v1['final_error_pct'] < 0).sum()
    v2_under = (df_v2['final_error_pct'] < 0).sum()
    print(f"\nUnder-prediction sessions: V1={v1_under}/{len(df_v1)}, V2={v2_under}/{len(df_v2)}")
    
    return {
        'v1_mean_error': v1_mean,
        'v2_mean_error': v2_mean,
        'v1_std': df_v1['final_error_pct'].std(),
        'v2_std': df_v2['final_error_pct'].std(),
        'v1_under_pct': v1_under / len(df_v1),
        'v2_under_pct': v2_under / len(df_v2),
        'df_v1': df_v1,
        'df_v2': df_v2
    }


def print_per_session_comparison(
    v1_results: List[Dict],
    v2_results: List[Dict]
) -> pd.DataFrame:
    """
    Print detailed per-session comparison between V1 and V2 models.
    
    Args:
        v1_results: List of evaluation result dicts for V1 model
        v2_results: List of evaluation result dicts for V2 model
        
    Returns:
        DataFrame with merged comparison results
    """
    df_v1 = pd.DataFrame(v1_results)
    df_v2 = pd.DataFrame(v2_results)
    
    df_merged = df_v1[['session_id', 'final_error_pct', 'session_length', 'actual_duration_min']].merge(
        df_v2[['session_id', 'final_error_pct']], 
        on='session_id', 
        suffixes=('_v1', '_v2')
    )
    df_merged['v2_better'] = abs(df_merged['final_error_pct_v2']) < abs(df_merged['final_error_pct_v1'])
    df_merged['improvement'] = abs(df_merged['final_error_pct_v1']) - abs(df_merged['final_error_pct_v2'])
    
    print("\n" + "="*80)
    print("PER-SESSION DETAILED COMPARISON")
    print("="*80)
    
    print(f"\n{'Idx':<4} {'Session ID (last 40 chars)':<42} {'V1 Error':<12} {'V2 Error':<12} "
          f"{'Winner':<8} {'Δ':<10}")
    print("-" * 90)
    
    for idx, row in df_merged.iterrows():
        session_short = row['session_id'][-40:]
        winner = "V2 ✓" if row['v2_better'] else "V1 ✓"
        delta = f"{row['improvement']:+.1f}pp"
        print(f"{idx:<4} {session_short:<42} {row['final_error_pct_v1']:+7.1f}%     "
              f"{row['final_error_pct_v2']:+7.1f}%     {winner:<8} {delta}")
    
    v2_wins = df_merged['v2_better'].sum()
    v1_wins = len(df_merged) - v2_wins
    print("-" * 90)
    print(f"Sessions where V2 is better: {v2_wins}/{len(df_merged)}")
    print(f"Sessions where V1 is better: {v1_wins}/{len(df_merged)}")
    
    return df_merged
