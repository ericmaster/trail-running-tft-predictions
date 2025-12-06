"""
TFT Model V3 - Fine-tuning with Sparse Features

This module provides the TrailRunningTFTV3 model for fine-tuning with Garmin data.
Key features:
1. Load pre-trained V2 weights with partial matching
2. Layer freezing strategy for transfer learning
3. Support for new sparse features (rpe, water_intake, etc.)
4. Lower learning rate and higher regularization for fine-tuning

The model extends TrailRunningTFT but adds fine-tuning capabilities
for the new Garmin data with nutritional tracking.
"""

from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE, MultiLoss
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

# Import base model components
from lib.model import TrailRunningTFT, AsymmetricSMAPE


class TrailRunningTFTV3(TrailRunningTFT):
    """
    Temporal Fusion Transformer V3 for Fine-tuning with Garmin Data.
    
    This model:
    1. Can load V2 pre-trained weights
    2. Supports layer freezing for transfer learning
    3. Uses lower learning rate for fine-tuning
    4. Has higher regularization to prevent overfitting on small dataset
    """
    
    # Default fine-tuning hyperparameters
    FINETUNE_LEARNING_RATE = 1e-6  # 10x lower than V2
    FINETUNE_WEIGHT_DECAY = 0.01   # 2x higher than V2
    FINETUNE_DROPOUT = 0.30        # Higher than V2's 0.25
    
    def __init__(
        self,
        freeze_encoder: bool = False,
        freeze_variable_selection: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize TFT V3 for fine-tuning.
        
        Args:
            freeze_encoder: If True, freeze LSTM encoder layers
            freeze_variable_selection: If True, freeze variable selection networks
            *args, **kwargs: Passed to parent TrailRunningTFT
        """
        super().__init__(*args, **kwargs)
        
        self.freeze_encoder = freeze_encoder
        self.freeze_variable_selection = freeze_variable_selection
        
        # Apply freezing if requested
        if freeze_encoder:
            self._freeze_encoder_layers()
        if freeze_variable_selection:
            self._freeze_variable_selection()
    
    def _freeze_encoder_layers(self):
        """Freeze LSTM encoder layers to preserve learned representations."""
        frozen_count = 0
        for name, param in self.named_parameters():
            if 'lstm_encoder' in name or 'encoder_lstm' in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} encoder parameters")
    
    def _freeze_variable_selection(self):
        """Freeze variable selection networks."""
        frozen_count = 0
        for name, param in self.named_parameters():
            if 'variable_selection' in name or 'vsn' in name.lower():
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} variable selection parameters")
    
    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("Unfrozen all parameters")
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": 100 * trainable / total if total > 0 else 0
        }
    
    @classmethod
    def from_dataset(
        cls,
        dataset,
        use_asymmetric_loss: bool = True,
        asymmetric_alpha: float = 0.6,
        freeze_encoder: bool = False,
        freeze_variable_selection: bool = False,
        **kwargs
    ):
        """
        Create V3 model from dataset with fine-tuning defaults.
        
        Args:
            dataset: TimeSeriesDataSet to configure model from
            use_asymmetric_loss: Use asymmetric SMAPE (same as V2)
            asymmetric_alpha: Alpha for asymmetric loss
            freeze_encoder: Freeze LSTM encoder
            freeze_variable_selection: Freeze variable selection networks
            **kwargs: Additional model arguments
        """
        # Set fine-tuning defaults (lower LR, higher regularization)
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = cls.FINETUNE_LEARNING_RATE
        if 'dropout' not in kwargs:
            kwargs['dropout'] = cls.FINETUNE_DROPOUT
        if 'weight_decay' not in kwargs:
            kwargs['weight_decay'] = cls.FINETUNE_WEIGHT_DECAY
        
        # Match V2 architecture defaults
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = 57  # Match V2
        if 'attention_head_size' not in kwargs:
            kwargs['attention_head_size'] = 3  # Match V2
        if 'hidden_continuous_size' not in kwargs:
            kwargs['hidden_continuous_size'] = 44  # Match V2
        if 'lstm_layers' not in kwargs:
            kwargs['lstm_layers'] = 1
        
        # Output size for multi-target
        if 'output_size' not in kwargs:
            kwargs['output_size'] = [1] * len(dataset.target)
        
        # Loss function (same as V2)
        if 'loss' not in kwargs:
            target_names = dataset.target
            target_weights = []
            for name in target_names:
                if name == "duration_diff":
                    target_weights.append(0.85)
                else:
                    target_weights.append(0.05)
            
            if use_asymmetric_loss:
                print(f"\n=== V3 Fine-tuning with Asymmetric SMAPE (alpha={asymmetric_alpha}) ===")
                kwargs['loss'] = MultiLoss(
                    metrics=[AsymmetricSMAPE(alpha=asymmetric_alpha) for _ in target_names],
                    weights=target_weights
                )
            else:
                kwargs['loss'] = MultiLoss(
                    metrics=[SMAPE() for _ in target_names],
                    weights=target_weights
                )
            
            print(f"Target weights: {dict(zip(target_names, target_weights))}")
        
        if 'logging_metrics' not in kwargs:
            kwargs['logging_metrics'] = [SMAPE(), MAE(), RMSE()]
        
        if 'reduce_on_plateau_patience' not in kwargs:
            kwargs['reduce_on_plateau_patience'] = 5  # More aggressive LR reduction
        
        if 'reduce_on_plateau_min_lr' not in kwargs:
            kwargs['reduce_on_plateau_min_lr'] = 1e-7
        
        # Store freezing config
        kwargs['freeze_encoder'] = freeze_encoder
        kwargs['freeze_variable_selection'] = freeze_variable_selection
        
        return cls.from_dataset_base(dataset, **kwargs)
    
    @classmethod
    def from_dataset_base(cls, dataset, freeze_encoder=False, freeze_variable_selection=False, **kwargs):
        """Base method to create model from dataset."""
        # Remove our custom kwargs before passing to parent
        model = TemporalFusionTransformer.from_dataset(dataset, **kwargs)
        
        # Apply freezing after model creation
        if freeze_encoder:
            for name, param in model.named_parameters():
                if 'lstm_encoder' in name or 'encoder_lstm' in name:
                    param.requires_grad = False
        
        if freeze_variable_selection:
            for name, param in model.named_parameters():
                if 'variable_selection' in name or 'vsn' in name.lower():
                    param.requires_grad = False
        
        return model


def load_v2_weights_into_v3(
    v3_model: nn.Module,
    v2_checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load V2 pre-trained weights into V3 model.
    
    Handles mismatched layers gracefully (e.g., new sparse feature embeddings).
    
    Args:
        v3_model: The V3 model to load weights into
        v2_checkpoint_path: Path to V2 checkpoint file
        strict: If False, allow missing/unexpected keys
        verbose: Print detailed loading information
    
    Returns:
        Dictionary with loading statistics
    """
    print(f"\n=== Loading V2 Weights from: {v2_checkpoint_path} ===")
    
    # Load checkpoint
    checkpoint = torch.load(v2_checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get state dict
    if 'state_dict' in checkpoint:
        v2_state_dict = checkpoint['state_dict']
    else:
        v2_state_dict = checkpoint
    
    # Get V3 model state dict
    v3_state_dict = v3_model.state_dict()
    
    # Track what gets loaded
    loaded_keys = []
    skipped_keys = []
    missing_in_v2 = []
    shape_mismatch = []
    
    # Try to match keys
    for key in v3_state_dict.keys():
        if key in v2_state_dict:
            v2_shape = v2_state_dict[key].shape
            v3_shape = v3_state_dict[key].shape
            
            if v2_shape == v3_shape:
                v3_state_dict[key] = v2_state_dict[key]
                loaded_keys.append(key)
            else:
                shape_mismatch.append((key, v2_shape, v3_shape))
                if verbose:
                    print(f"  Shape mismatch: {key} - V2:{v2_shape} vs V3:{v3_shape}")
        else:
            missing_in_v2.append(key)
    
    # Keys in V2 but not in V3
    for key in v2_state_dict.keys():
        if key not in v3_state_dict:
            skipped_keys.append(key)
    
    # Load matched weights
    v3_model.load_state_dict(v3_state_dict, strict=False)
    
    # Summary
    stats = {
        "loaded": len(loaded_keys),
        "skipped": len(skipped_keys),
        "missing_in_v2": len(missing_in_v2),
        "shape_mismatch": len(shape_mismatch),
        "total_v3_params": len(v3_state_dict),
        "total_v2_params": len(v2_state_dict),
    }
    
    if verbose:
        print(f"\n=== Weight Loading Summary ===")
        print(f"  Loaded: {stats['loaded']}/{stats['total_v3_params']} parameters")
        print(f"  Shape mismatch: {stats['shape_mismatch']}")
        print(f"  Missing in V2: {stats['missing_in_v2']}")
        print(f"  Skipped (not in V3): {stats['skipped']}")
        
        if missing_in_v2 and verbose:
            print(f"\n  New V3 parameters (randomly initialized):")
            for key in missing_in_v2[:10]:  # Show first 10
                print(f"    - {key}")
            if len(missing_in_v2) > 10:
                print(f"    ... and {len(missing_in_v2) - 10} more")
    
    return stats


def create_v3_model_from_v2(
    v2_checkpoint_path: str,
    v3_dataset,
    freeze_encoder: bool = False,
    freeze_variable_selection: bool = False,
    learning_rate: float = 1e-6,
    **kwargs
) -> TrailRunningTFTV3:
    """
    Convenience function to create V3 model and load V2 weights.
    
    Args:
        v2_checkpoint_path: Path to V2 checkpoint
        v3_dataset: TimeSeriesDataSet for V3 configuration
        freeze_encoder: Freeze LSTM encoder layers
        freeze_variable_selection: Freeze variable selection networks
        learning_rate: Fine-tuning learning rate
        **kwargs: Additional model arguments
    
    Returns:
        V3 model with V2 weights loaded
    """
    # Create V3 model from dataset
    model = TrailRunningTFTV3.from_dataset(
        v3_dataset,
        freeze_encoder=freeze_encoder,
        freeze_variable_selection=freeze_variable_selection,
        learning_rate=learning_rate,
        **kwargs
    )
    
    # Load V2 weights
    load_v2_weights_into_v3(model, v2_checkpoint_path, strict=False, verbose=True)
    
    # Print trainable parameter info
    param_info = model.get_trainable_parameters()
    print(f"\n=== Trainable Parameters ===")
    print(f"  Trainable: {param_info['trainable']:,} ({param_info['trainable_pct']:.1f}%)")
    print(f"  Frozen: {param_info['frozen']:,}")
    print(f"  Total: {param_info['total']:,}")
    
    return model
