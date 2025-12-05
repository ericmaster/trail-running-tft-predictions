import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
