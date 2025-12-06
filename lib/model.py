from typing import Optional, List
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE, MultiLoss
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import InterpretableMultiHeadAttention


class AsymmetricSMAPE(MultiHorizonMetric):
    """
    Asymmetric SMAPE loss that penalizes under-predictions more than over-predictions.
    
    This is compatible with pytorch-forecasting's MultiLoss and TFT architecture.
    
    With alpha > 0.5, the model is penalized more for under-predictions,
    encouraging predictions to be higher than actuals (conservative estimates).
    
    For trail running: alpha=0.7 means under-predictions are penalized ~2.3x more
    than over-predictions, which helps address the observed cold-start bias.
    """
    
    def __init__(self, alpha: float = 0.7, **kwargs):
        """
        Initialize asymmetric SMAPE loss.
        
        Args:
            alpha: Asymmetry factor (0.5 = symmetric SMAPE, >0.5 = penalize under-prediction more)
        """
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate asymmetric SMAPE loss.
        
        Args:
            y_pred: Predicted values (may have extra dimension from model output)
            target: Target values
        
        Returns:
            Asymmetric SMAPE loss per sample
        """
        # Convert to point prediction (same as SMAPE does)
        y_pred = self.to_prediction(y_pred)
        
        # Standard SMAPE denominator
        denominator = (torch.abs(y_pred) + torch.abs(target) + 1e-8)
        
        # Calculate error
        error = target - y_pred  # Positive when under-predicting
        
        # Asymmetric weighting
        # When error > 0 (under-prediction): weight = alpha
        # When error < 0 (over-prediction): weight = (1 - alpha)
        weights = torch.where(error > 0, self.alpha, 1 - self.alpha)
        
        # Asymmetric SMAPE: weight * |error| / denominator * 2
        loss = weights * torch.abs(error) / denominator * 2
        
        return loss


class WeightedMultiTargetSMAPE(nn.Module):
    """
    Weighted SMAPE loss for multi-target forecasting.
    Applies different weights to different target variables.
    """
    
    def __init__(self, target_weights: List[float], target_names: List[str] = None):
        """
        Initialize weighted SMAPE loss.
        
        Args:
            target_weights: List of weights for each target variable (should sum to 1.0)
            target_names: Optional list of target names for debugging
        """
        super().__init__()
        
        self.target_names = target_names or [f"target_{i}" for i in range(len(target_weights))]
        self.target_weights = torch.tensor(target_weights, dtype=torch.float32)
        
        # Ensure weights sum to 1.0
        if not abs(sum(target_weights) - 1.0) < 1e-6:
            print(f"Warning: Target weights sum to {sum(target_weights):.6f}, not 1.0")
        
        print(f"Initialized WeightedMultiTargetSMAPE with weights:")
        for name, weight in zip(self.target_names, target_weights):
            print(f"  {name}: {weight:.1%}")
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted SMAPE loss.
        
        Args:
            prediction: Predicted values [batch_size, seq_len, num_targets]
            target: Target values [batch_size, seq_len, num_targets]
        
        Returns:
            Weighted SMAPE loss
        """
        # Move weights to the same device as inputs
        weights = self.target_weights.to(prediction.device)
        
        # Calculate SMAPE for each target
        # SMAPE = 2 * |prediction - target| / (|prediction| + |target|)
        smape_per_target = 2 * torch.abs(prediction - target) / (torch.abs(prediction) + torch.abs(target) + 1e-8)
        
        # Average over batch and sequence dimensions, keep target dimension
        smape_per_target = smape_per_target.mean(dim=(0, 1))  # [num_targets]
        
        # Apply weights and sum
        weighted_loss = (smape_per_target * weights).sum()
        
        return weighted_loss

class TrailRunningTFT(TemporalFusionTransformer):
    """Temporal Fusion Transformer for Trail Running Time Prediction."""

    def __init__(self, mask_bias: float = -1e4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._override_mask_bias(mask_bias)

    # def _override_mask_bias(self, mask_bias_value: float):
    #     """Override mask_bias in all InterpretableMultiHeadAttention modules."""
    #     for module in self.modules():
    #         if isinstance(module, InterpretableMultiHeadAttention):
    #             # Override the mask_bias used in its internal attention
    #             if hasattr(module, "mask_bias"):
    #                 module.mask_bias = mask_bias_value
    #             # Also override in the nested MultiHeadAttention if accessible
    #             if hasattr(module, "_attention"):
    #                 attention_module = getattr(module, "_attention", None)
    #                 if attention_module is not None and hasattr(attention_module, "mask_bias"):
    #                     attention_module.mask_bias = mask_bias_value

    @classmethod
    def from_dataset(cls, dataset, use_quantile_loss: bool = False, quantile_alpha: float = 0.7, **kwargs):
        """
        Instantiate TrailRunningTFT from a TimeSeriesDataSet with optimized defaults for trail running.
        
        Args:
            dataset: TimeSeriesDataSet to create model from
            use_quantile_loss: If True, use asymmetric quantile loss instead of SMAPE
            quantile_alpha: Alpha for quantile loss (0.7 = penalize under-predictions more)
            **kwargs: Additional model arguments
        """
        # Set our preferred defaults only if not already provided
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = 4e-05
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = 45
        if 'attention_head_size' not in kwargs:
            kwargs['attention_head_size'] = 3
        if 'dropout' not in kwargs:
            kwargs['dropout'] = 0.25
        if 'hidden_continuous_size' not in kwargs:
            kwargs['hidden_continuous_size'] = 37
        if 'output_size' not in kwargs:
            kwargs['output_size'] = len(dataset.target)  # Multi-target output
        if 'lstm_layers' not in kwargs:
            kwargs['lstm_layers'] = 1
        if 'weight_decay' not in kwargs:
            kwargs['weight_decay'] = 5e-04  # L2 regularization
        if 'loss' not in kwargs:
            target_names = dataset.target
            
            # Define weights for multi-target forecasting
            target_weights = []
            for name in target_names:
                if name == "duration_diff":
                    target_weights.append(0.85)  # 85% weight for duration_diff
                else:
                    target_weights.append(0.05)  # 5% weight for each other variable
            
            if use_quantile_loss:
                # Use asymmetric SMAPE loss to bias toward over-prediction
                # This helps address the cold-start under-prediction bias
                print(f"\n=== Using Asymmetric SMAPE Loss (alpha={quantile_alpha}) ===")
                print(f"Under-prediction penalty: {quantile_alpha / (1 - quantile_alpha):.2f}x more than over-prediction")
                
                kwargs['loss'] = MultiLoss(
                    metrics=[AsymmetricSMAPE(alpha=quantile_alpha) for _ in target_names],
                    weights=target_weights
                )
            else:
                # Use standard SMAPE loss
                print(f"\n=== Using Standard SMAPE Loss ===")
                kwargs['loss'] = MultiLoss(
                    metrics=[SMAPE() for _ in target_names],
                    weights=target_weights
                )
            
            print(f"Target weights:")
            for name, weight in zip(target_names, target_weights):
                print(f"  {name}: {weight:.1%}")
                
        if 'logging_metrics' not in kwargs:
            kwargs['logging_metrics'] = [SMAPE(), MAE(), RMSE()]
        if 'reduce_on_plateau_patience' not in kwargs:
            kwargs['reduce_on_plateau_patience'] = 4
        
        # Call parent's from_dataset
        return super().from_dataset(dataset, **kwargs)


def evaluate_full_session_sequential(
    model, 
    test_data, 
    train_data, 
    session_id,
    calculate_weighted_first_sample_fn,
    max_pred_length: int = 200, 
    encoder_length: int = 1,
    target_names: list = None,
    verbose: bool = False
):
    """
    Evaluate FULL SESSION using sequential chunking.
    Previous predictions feed into subsequent chunks as encoder data.
    
    This function is designed for cold-start evaluation where:
    1. First chunk uses a synthetic encoder (no prior data)
    2. Subsequent chunks use previous predictions as encoder input
    3. Predictions accumulate across the entire session
    
    Args:
        model: TFT model for inference (should be on CPU and in eval mode)
        test_data: DataFrame containing test session data
        train_data: DataFrame containing training data (for synthetic encoder)
        session_id: ID of the session to evaluate
        calculate_weighted_first_sample_fn: Function to calculate synthetic encoder
        max_pred_length: Maximum prediction steps per chunk (default 200)
        encoder_length: Encoder length for subsequent chunks (default 1)
        target_names: List of target variable names (default: standard 4 targets)
        verbose: If True, print progress for each chunk
        
    Returns:
        Dictionary with evaluation results including:
        - session_id, session_length, steps_predicted, chunks_processed
        - final_error_pct, bias, mae
        - actual_duration_min, pred_duration_min
        - all_predictions, all_actuals (dict of lists per target)
        - chunk_errors (list of per-chunk error info)
        - chunk_boundaries (list of chunk start indices)
        
        Returns None if session is too short or evaluation fails.
    """
    import pandas as pd
    import numpy as np
    import torch
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
    from pytorch_forecasting.data.encoders import NaNLabelEncoder
    
    if target_names is None:
        target_names = ['duration_diff', 'heartRate', 'temperature', 'cadence']
    
    session_data = test_data[test_data['session_id'] == session_id].copy()
    session_data = session_data.reset_index(drop=True)
    session_length = len(session_data)
    
    if verbose:
        print(f"Starting evaluation for session: {session_id}")
        print(f"Session length: {session_length} steps")
    
    if session_length < 50:
        if verbose:
            print(f"Session too short ({session_length} < 50), returning None")
        return None
    
    session_encoded = session_data['session_id_encoded'].iloc[0]
    
    # Create synthetic encoder for cold-start
    synthetic_encoder = calculate_weighted_first_sample_fn(train_data)
    
    # Override with actual known terrain values
    known_future_vars = ['altitude', 'elevation_diff', 'elevation_gain', 'elevation_loss', 'distance']
    for var in known_future_vars:
        if var in session_data.columns:
            synthetic_encoder[var] = session_data.iloc[0][var]
    
    # Storage for sequential predictions (per-target)
    all_predictions = {name: [] for name in target_names}
    all_actuals = {name: [] for name in target_names}
    accumulated_predictions = {name: [] for name in target_names}
    chunk_errors = []
    chunk_boundaries = []
    
    # Time-varying variable lists
    time_varying_known_reals = ["altitude", "elevation_diff", "elevation_gain", "elevation_loss"]
    time_varying_unknown_reals = target_names + ["speed", "avg_heart_rate_so_far", "duration"]
    
    chunk_idx = 0
    start_idx = 0
    
    while start_idx < session_length:
        chunk_idx += 1
        
        # Calculate chunk boundaries
        pred_start = start_idx
        pred_end = min(start_idx + max_pred_length, session_length)
        pred_length = pred_end - pred_start
        
        if pred_length < 10:
            break
        
        if verbose:
            print(f"\n--- Chunk {chunk_idx} ---")
            print(f"  Prediction range: steps {pred_start} to {pred_end-1} ({pred_length} steps)")
            if 'distance' in session_data.columns:
                print(f"  Distance: {session_data.iloc[pred_start]['distance']/1000:.2f} km to "
                      f"{session_data.iloc[pred_end-1]['distance']/1000:.2f} km")
        
        try:
            if chunk_idx == 1:
                # Cold-start: synthetic encoder + prediction data
                encoder_df = pd.DataFrame([synthetic_encoder])
                encoder_df['time_idx'] = 0
                encoder_df['session_id_encoded'] = int(session_encoded)
                encoder_df['session_id'] = session_id
                
                pred_df = session_data.iloc[pred_start:pred_end].copy()
                pred_df['time_idx'] = range(1, pred_length + 1)
                
                chunk_data = pd.concat([encoder_df, pred_df], ignore_index=True)
                actual_encoder_len = 1
            else:
                # Use previous predictions as encoder
                encoder_steps = min(encoder_length, pred_start)
                encoder_start = max(0, pred_start - encoder_steps)
                
                chunk_data = session_data.iloc[encoder_start:pred_end].copy()
                chunk_data = chunk_data.reset_index(drop=True)
                chunk_data['time_idx'] = range(len(chunk_data))
                
                # Replace encoder targets with accumulated predictions
                if len(accumulated_predictions['duration_diff']) >= encoder_steps:
                    for i in range(encoder_steps):
                        global_pred_idx = pred_start - encoder_steps + i
                        for target in target_names:
                            if global_pred_idx < len(accumulated_predictions[target]):
                                chunk_data.loc[i, target] = accumulated_predictions[target][global_pred_idx]
                
                actual_encoder_len = encoder_steps
            
            # Ensure consistent types
            chunk_data['session_id_encoded'] = chunk_data['session_id_encoded'].astype(int)
            chunk_data['time_idx'] = chunk_data['time_idx'].astype(int)
            
            # Create TimeSeriesDataSet
            chunk_dataset = TimeSeriesDataSet(
                chunk_data,
                time_idx="time_idx",
                target=target_names,
                group_ids=["session_id_encoded"],
                min_encoder_length=1,
                max_encoder_length=max(1, actual_encoder_len),
                min_prediction_length=pred_length,
                max_prediction_length=pred_length,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=MultiNormalizer(
                    [GroupNormalizer(groups=["session_id_encoded"], transformation=None) for _ in target_names]
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                categorical_encoders={"session_id_encoded": NaNLabelEncoder(add_nan=True)},
                add_encoder_length=True,
                predict_mode=True
            )
            
            chunk_loader = chunk_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            x, y = next(iter(chunk_loader))
            
            # Move inputs to model's device (recursive for nested structures)
            device = next(model.parameters()).device
            
            def move_to_device(obj, device):
                """Recursively move tensors in dict/list/tuple to device."""
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, device) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(move_to_device(v, device) for v in obj)
                else:
                    return obj
            
            x = move_to_device(x, device)
            
            with torch.no_grad():
                output = model(x)
            
            predictions = output['prediction']
            
            # Store predictions and actuals for all targets
            min_len = None
            for i, target in enumerate(target_names):
                if isinstance(predictions, list):
                    pred_vals = predictions[i][0, :, 0].cpu().numpy()
                else:
                    pred_vals = predictions[0, :, i].cpu().numpy()
                
                actual_vals = session_data.iloc[pred_start:pred_start + len(pred_vals)][target].values
                
                current_min_len = min(len(pred_vals), len(actual_vals))
                if min_len is None:
                    min_len = current_min_len
                else:
                    min_len = min(min_len, current_min_len)
                
                pred_vals = pred_vals[:current_min_len]
                actual_vals = actual_vals[:current_min_len]
                
                all_predictions[target].extend(pred_vals)
                all_actuals[target].extend(actual_vals)
                accumulated_predictions[target].extend(pred_vals)
            
            # Calculate chunk error for duration_diff
            pred_duration = np.array(all_predictions['duration_diff'][-min_len:])
            actual_duration = np.array(all_actuals['duration_diff'][-min_len:])
            chunk_mae = np.mean(np.abs(pred_duration - actual_duration))
            
            chunk_errors.append({
                'chunk': chunk_idx,
                'start_idx': pred_start,
                'end_idx': pred_start + min_len,
                'mae': chunk_mae,
                'accumulated_pred': np.sum(all_predictions['duration_diff']),
                'accumulated_actual': np.sum(all_actuals['duration_diff'])
            })
            
            chunk_boundaries.append(pred_start)
            
            if verbose:
                print(f"  Predictions obtained: {min_len} steps")
                print(f"  Chunk MAE (duration_diff): {chunk_mae:.4f}")
            
        except Exception as e:
            if verbose:
                print(f"  Error in chunk {chunk_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
            break
        
        start_idx = pred_end
    
    if len(all_predictions['duration_diff']) == 0:
        return None
    
    # Calculate final accumulated duration error
    pred_arr = np.array(all_predictions['duration_diff'])
    actual_arr = np.array(all_actuals['duration_diff'])
    
    pred_accumulated = np.sum(pred_arr)
    actual_accumulated = np.sum(actual_arr)
    
    if actual_accumulated != 0:
        final_error_pct = (pred_accumulated - actual_accumulated) / actual_accumulated * 100
    else:
        final_error_pct = 0
    
    errors = pred_arr - actual_arr
    bias = np.mean(errors)
    mae = np.mean(np.abs(errors))
    
    return {
        'session_id': session_id,
        'session_length': session_length,
        'steps_predicted': len(all_predictions['duration_diff']),
        'chunks_processed': chunk_idx,
        'final_error_pct': final_error_pct,
        'bias': bias,
        'mae': mae,
        'actual_duration_min': actual_accumulated / 60,
        'pred_duration_min': pred_accumulated / 60,
        # Additional data for detailed analysis
        'all_predictions': all_predictions,
        'all_actuals': all_actuals,
        'chunk_errors': chunk_errors,
        'chunk_boundaries': chunk_boundaries
    }
    