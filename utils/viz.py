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
