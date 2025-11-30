import numpy as np
import pandas as pd

def print_cold_start_summary(all_predictions, chunk_errors, stats, session_id=None):
    """
    Print summary statistics for cold-start inference.
    
    Args:
        all_predictions: Dict with target names as keys and list of predictions as values
        chunk_errors: List of chunk info dicts with 'chunk', 'start_idx', 'end_idx', 'mae'
        stats: Dict from plot_accumulated_duration_error with accumulated durations/errors
        session_id: Optional session ID
        
    Returns:
        dict: Summary statistics
    """
    print("\n" + "="*80)
    print("FINAL SUMMARY STATISTICS")
    print("="*80)

    # Overall metrics
    pred_total = stats['pred_duration_accumulated'][-1]
    actual_total = stats['actual_duration_accumulated'][-1]
    total_error = pred_total - actual_total
    total_error_pct = (total_error / actual_total) * 100 if actual_total != 0 else 0

    if session_id:
        print(f"\nSession: {session_id}")
    print(f"Total Distance: {stats['distance_km'][-1]:.2f} km")
    print(f"Total Steps Predicted: {len(all_predictions['duration_diff'])}")
    print(f"Number of Chunks: {len(chunk_errors)}")

    print(f"\n--- Duration Prediction ---")
    print(f"Actual Total Duration: {actual_total/60:.2f} minutes ({actual_total/3600:.2f} hours)")
    print(f"Predicted Total Duration: {pred_total/60:.2f} minutes ({pred_total/3600:.2f} hours)")
    print(f"Absolute Error: {abs(total_error)/60:.2f} minutes")
    print(f"Percentage Error: {abs(total_error_pct):.2f}%")

    # Per-chunk metrics
    print(f"\n--- Per-Chunk MAE (duration_diff) ---")
    for chunk_info in chunk_errors:
        print(f"  Chunk {chunk_info['chunk']}: MAE = {chunk_info['mae']:.4f}")

    # Error evolution
    print(f"\n--- Error at Key Points ---")
    checkpoints = [0.25, 0.5, 0.75, 1.0]
    error_min = stats['error_min']
    distance_km = stats['distance_km']
    for pct in checkpoints:
        idx = int(len(error_min) * pct) - 1
        if idx >= 0 and idx < len(error_min):
            print(f"  At {pct*100:.0f}% ({distance_km[idx]:.1f} km): {error_min[idx]:.2f} minutes error")
    
    return {
        'pred_total': pred_total,
        'actual_total': actual_total,
        'total_error': total_error,
        'total_error_pct': total_error_pct,
        'checkpoints': checkpoints
    }

def save_cold_start_summary(summary_file, all_predictions, chunk_errors, stats, summary_stats,
                           session_id=None, encoder_length=1, max_prediction_length=200, ckpt_path=None):
    """
    Save cold-start inference summary to a markdown file.
    
    Args:
        summary_file: Path to save the summary file
        all_predictions: Dict with target names as keys and list of predictions as values
        chunk_errors: List of chunk info dicts
        stats: Dict from plot_accumulated_duration_error
        summary_stats: Dict from print_cold_start_summary
        session_id: Optional session ID
        encoder_length: Encoder length used
        max_prediction_length: Max prediction length per chunk
        ckpt_path: Path to model checkpoint
    """
    with open(summary_file, 'w') as f:
        f.write("# Cold-Start Sequential Inference Summary\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Encoder Length:** {encoder_length} (minimal for cold-start)\n")
        f.write(f"- **Prediction Length per Chunk:** {max_prediction_length}\n")
        if ckpt_path:
            f.write(f"- **Model Checkpoint:** {ckpt_path}\n\n")
        
        f.write("## Session Information\n\n")
        if session_id:
            f.write(f"- **Session ID:** {session_id}\n")
        f.write(f"- **Total Distance:** {stats['distance_km'][-1]:.2f} km\n")
        f.write(f"- **Total Steps:** {len(all_predictions['duration_diff'])}\n")
        f.write(f"- **Number of Chunks:** {len(chunk_errors)}\n\n")
        
        f.write("## Duration Prediction Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Actual Total Duration | {summary_stats['actual_total']/60:.2f} min ({summary_stats['actual_total']/3600:.2f} h) |\n")
        f.write(f"| Predicted Total Duration | {summary_stats['pred_total']/60:.2f} min ({summary_stats['pred_total']/3600:.2f} h) |\n")
        f.write(f"| Absolute Error | {abs(summary_stats['total_error'])/60:.2f} min |\n")
        f.write(f"| Percentage Error | {abs(summary_stats['total_error_pct']):.2f}% |\n\n")
        
        f.write("## Per-Chunk MAE (duration_diff)\n\n")
        f.write("| Chunk | Steps | MAE |\n")
        f.write("|-------|-------|-----|\n")
        for chunk_info in chunk_errors:
            f.write(f"| {chunk_info['chunk']} | {chunk_info['start_idx']}-{chunk_info['end_idx']} | {chunk_info['mae']:.4f} |\n")
        
        f.write("\n## Error Evolution\n\n")
        f.write("| Progress | Distance (km) | Error (min) |\n")
        f.write("|----------|---------------|-------------|\n")
        error_min = stats['error_min']
        distance_km = stats['distance_km']
        for pct in summary_stats['checkpoints']:
            idx = int(len(error_min) * pct) - 1
            if idx >= 0 and idx < len(error_min):
                f.write(f"| {pct*100:.0f}% | {distance_km[idx]:.1f} | {error_min[idx]:.2f} |\n")
        
        f.write("\n## Methodology\n\n")
        f.write("1. **Cold-Start Initialization:** Synthetic encoder created using weighted average ")
        f.write("of first samples from all training sessions (chronological weighting)\n")
        f.write("2. **Known Terrain Data:** Actual altitude, elevation_diff, elevation_gain, elevation_loss ")
        f.write("used from target session (simulating GPS/route preview)\n")
        f.write("3. **Sequential Prediction:** Each chunk uses predicted values from previous chunks ")
        f.write("to inform the encoder for the next prediction\n")
        f.write("4. **Accumulated Duration:** Duration predictions (duration_diff) summed across all chunks\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("- `cold_start_chunks.png`: Per-chunk duration_diff predictions vs actuals\n")
        f.write("- `cold_start_accumulated_error.png`: Accumulated duration and error over distance\n")

    print(f"\nSummary saved to: {summary_file}")