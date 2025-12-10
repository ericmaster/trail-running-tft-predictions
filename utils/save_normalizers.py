"""
Save only the normalizers and encoders from training data for efficient cold-start inference.

This script extracts just the essential components needed for normalization,
avoiding the need to save the entire 45MB training dataset.
"""

import os
import sys
from pathlib import Path
import torch
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.data import TFTDataModule

def save_normalizers_only(
    data_dir: str = "./data/resampled",
    output_path: str = "./checkpoints_v2/normalizers.pkl",
    max_encoder_length: int = 400,
    max_prediction_length: int = 200
):
    """
    Extract and save only normalizers and encoders (not the full dataset).
    
    Args:
        data_dir: Directory containing training CSV files
        output_path: Where to save the normalizers
        max_encoder_length: Encoder length used in training
        max_prediction_length: Prediction length used in training
    """
    print("="*80)
    print("EXTRACTING NORMALIZERS FOR COLD-START INFERENCE")
    print("="*80)
    
    # Create data module
    print(f"\nLoading training data from: {data_dir}")
    data_module = TFTDataModule(
        data_dir=data_dir,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=64,
        use_sliding_windows=False
    )
    data_module.setup()
    
    # Get the training dataset
    training_dataset = data_module.training
    
    print(f"\nTraining dataset loaded:")
    print(f"  Total samples: {len(training_dataset)}")
    print(f"  Time-varying known reals: {training_dataset.time_varying_known_reals}")
    print(f"  Time-varying unknown reals: {training_dataset.time_varying_unknown_reals}")
    print(f"  Targets: {training_dataset.target_names}")
    
    # Extract only what we need for normalization
    normalizers_data = {
        'target_normalizer': training_dataset.target_normalizer,
        'categorical_encoders': training_dataset.categorical_encoders,
        'time_varying_known_reals': training_dataset.time_varying_known_reals,
        'time_varying_unknown_reals': training_dataset.time_varying_unknown_reals,
        'time_varying_known_categoricals': training_dataset.time_varying_known_categoricals,
        'target_names': training_dataset.target_names,
        'group_ids': training_dataset.group_ids,
        'max_encoder_length': training_dataset.max_encoder_length,
        'max_prediction_length': training_dataset.max_prediction_length,
        'static_categoricals': training_dataset.static_categoricals,
        'static_reals': training_dataset.static_reals,
        'add_relative_time_idx': training_dataset.add_relative_time_idx,
        'add_target_scales': training_dataset.add_target_scales,
        'add_encoder_length': training_dataset.add_encoder_length,
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(normalizers_data, f)
    
    print(f"\n✓ Normalizers saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    # Verify it can be loaded
    print("\nVerifying saved normalizers...")
    with open(output_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"✓ Normalizers loaded successfully")
    print(f"  Target normalizer preserved: {loaded_data['target_normalizer'] is not None}")
    print(f"  Categorical encoders preserved: {len(loaded_data['categorical_encoders'])} encoders")
    print(f"  Configuration preserved: {len(loaded_data)} fields")
    
    print("\n" + "="*80)
    print("DONE! Use these normalizers in API for efficient cold-start inference.")
    print("File is much smaller than full dataset (KB vs MB).")
    print("="*80)

if __name__ == "__main__":
    save_normalizers_only()
