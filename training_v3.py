#!/usr/bin/env python3
"""
Training Script V3 - Fine-tuning with Garmin Data

This script fine-tunes the V2 model on Garmin data with sparse nutritional
and RPE features. Uses transfer learning approach:

1. Load V2 pre-trained weights
2. Use lower learning rate (1e-6) to avoid catastrophic forgetting
3. Higher regularization (dropout=0.3, weight_decay=0.01)
4. Manual 5/1/1 session split for train/val/test
5. Early stopping with patience to prevent overfitting

Usage:
    python training_v3.py
    python training_v3.py --freeze-encoder  # Freeze LSTM encoder layers
    python training_v3.py --no-sparse       # Exclude sparse features (baseline)
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lib.data_v3 import TFTDataModuleV3
from lib.model_v3 import (
    TrailRunningTFTV3,
    load_v2_weights_into_v3,
)
from lib.model import TrailRunningTFT


# ============================================================================
# Configuration
# ============================================================================

# V2 checkpoint to load pre-trained weights from
V2_CHECKPOINT_PATH = "./checkpoints_v2/best-checkpoint_v2-epoch=27-val_loss=0.12-v1.ckpt"

# V3 checkpoint directory
V3_CHECKPOINT_DIR = "./checkpoints_v3"

# Garmin data directory
GARMIN_DATA_DIR = "./data/fit-resampled"

# Training hyperparameters (fine-tuning optimized)
FINE_TUNE_CONFIG = {
    "learning_rate": 1e-7,       # 10x lower than V2
    "weight_decay": 0.01,        # 2x higher than V2
    "dropout": 0.35,             # Higher than V2's 0.25
    "batch_size": 16,            # Smaller for limited data
    "max_epochs": 100,           # Enough epochs with early stopping
    "patience": 10,              # Early stopping patience
    "min_delta": 0.001,          # Minimum improvement for early stopping
    "gradient_clip_val": 0.02,    # Gradient clipping
    "accumulate_grad_batches": 2,  # Effective batch size = 32
}

# Model architecture (must match V2)
MODEL_CONFIG = {
    "hidden_size": 57,
    "attention_head_size": 3,
    "hidden_continuous_size": 44,
    "lstm_layers": 1,
    "max_encoder_length": 400,
    "max_prediction_length": 200,
}

# Loss configuration
LOSS_CONFIG = {
    "use_asymmetric_loss": True,
    "asymmetric_alpha": 0.51,  # Same as V2
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="V3 Fine-tuning with Garmin Data")
    
    parser.add_argument(
        "--v2-checkpoint",
        type=str,
        default=V2_CHECKPOINT_PATH,
        help="Path to V2 checkpoint for weight initialization"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=GARMIN_DATA_DIR,
        help="Directory containing Garmin resampled data"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=V3_CHECKPOINT_DIR,
        help="Directory to save V3 checkpoints"
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze LSTM encoder layers"
    )
    parser.add_argument(
        "--freeze-vsn",
        action="store_true",
        help="Freeze variable selection networks"
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Exclude sparse features (rpe, intake counters)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=FINE_TUNE_CONFIG["learning_rate"],
        help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=FINE_TUNE_CONFIG["max_epochs"],
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=FINE_TUNE_CONFIG["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to V3 checkpoint to resume training from"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in directory by modification time."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith('.ckpt')
    ]
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("TFT V3 Fine-tuning with Garmin Data")
    print("=" * 60)
    
    # ========================================================================
    # Setup directories
    # ========================================================================
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # ========================================================================
    # Initialize DataModule
    # ========================================================================
    print(f"\n=== Loading Garmin Data ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Include sparse features: {not args.no_sparse}")
    
    data_module = TFTDataModuleV3(
        data_dir=args.data_dir,
        min_encoder_length=1,
        max_encoder_length=MODEL_CONFIG["max_encoder_length"],
        max_prediction_length=MODEL_CONFIG["max_prediction_length"],
        batch_size=args.batch_size,
        num_workers=4,
        include_sparse_features=not args.no_sparse,
        train_sessions=5,
        val_sessions=1,
        test_sessions=1,
    )
    
    # Setup data
    data_module.prepare_data()
    data_module.setup(stage='fit')
    
    # Get feature info
    feature_info = data_module.get_feature_info()
    print(f"\nFeature configuration:")
    for key, value in feature_info.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print(f"\n=== Initializing V3 Model ===")
    print(f"Loading V2 weights from: {args.v2_checkpoint}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Freeze variable selection: {args.freeze_vsn}")
    
    # Check if V2 checkpoint exists
    if not os.path.exists(args.v2_checkpoint):
        raise FileNotFoundError(f"V2 checkpoint not found: {args.v2_checkpoint}")
    
    # Create model from dataset
    model = TrailRunningTFT.from_dataset(
        data_module.training,
        learning_rate=args.learning_rate,
        hidden_size=MODEL_CONFIG["hidden_size"],
        attention_head_size=MODEL_CONFIG["attention_head_size"],
        hidden_continuous_size=MODEL_CONFIG["hidden_continuous_size"],
        lstm_layers=MODEL_CONFIG["lstm_layers"],
        dropout=FINE_TUNE_CONFIG["dropout"],
        weight_decay=FINE_TUNE_CONFIG["weight_decay"],
        use_quantile_loss=LOSS_CONFIG["use_asymmetric_loss"],
        quantile_alpha=LOSS_CONFIG["asymmetric_alpha"],
        output_size=[1] * len(data_module.target_names),
    )
    
    # Load V2 weights
    load_stats = load_v2_weights_into_v3(
        model,
        args.v2_checkpoint,
        strict=False,
        verbose=True
    )
    
    # Apply freezing if requested
    if args.freeze_encoder:
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'lstm_encoder' in name or 'encoder_lstm' in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} encoder parameters")
    
    if args.freeze_vsn:
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'variable_selection' in name or 'vsn' in name.lower():
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} variable selection parameters")
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"\nParameter summary:")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  Frozen: {frozen:,}")
    print(f"  Total: {total:,}")
    
    # ========================================================================
    # Setup Callbacks
    # ========================================================================
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            patience=FINE_TUNE_CONFIG["patience"],
            min_delta=FINE_TUNE_CONFIG["min_delta"],
            mode="min",
            verbose=True,
        ),
        # Model checkpoint
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="best-checkpoint_v3-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # ========================================================================
    # Setup Loggers
    # ========================================================================
    loggers = [
        CSVLogger(save_dir="./logs", name="tft_model_v3"),
        TensorBoardLogger(save_dir="./logs", name="tft_model_v3_tb"),
    ]
    
    # ========================================================================
    # Setup Trainer
    # ========================================================================
    print(f"\n=== Training Configuration ===")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {FINE_TUNE_CONFIG['weight_decay']}")
    print(f"  Dropout: {FINE_TUNE_CONFIG['dropout']}")
    print(f"  Gradient accumulation: {FINE_TUNE_CONFIG['accumulate_grad_batches']}")
    print(f"  Gradient clip: {FINE_TUNE_CONFIG['gradient_clip_val']}")
    print(f"  Early stopping patience: {FINE_TUNE_CONFIG['patience']}")
    
    # Determine device
    if torch.cuda.is_available() and args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
        print(f"  Using {devices} GPU(s)")
    else:
        accelerator = "cpu"
        devices = 1
        print("  Using CPU")
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=FINE_TUNE_CONFIG["gradient_clip_val"],
        accumulate_grad_batches=FINE_TUNE_CONFIG["accumulate_grad_batches"],
        precision="32-true",  # Full precision for stability
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
        enable_progress_bar=True,
    )
    
    # ========================================================================
    # Resume from checkpoint if specified
    # ========================================================================
    resume_path = args.resume
    if resume_path is None:
        # Check for existing V3 checkpoints
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest:
            print(f"\nFound existing V3 checkpoint: {latest}")
            user_input = input("Resume from this checkpoint? [y/N]: ")
            if user_input.lower() == 'y':
                resume_path = latest
    
    if resume_path:
        print(f"\nResuming from: {resume_path}")
    
    # ========================================================================
    # Train
    # ========================================================================
    print("\n" + "=" * 60)
    print("Starting V3 Fine-tuning...")
    print("=" * 60 + "\n")
    
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
        ckpt_path=resume_path,
    )
    
    # ========================================================================
    # Print Results
    # ========================================================================
    print("\n" + "=" * 60)
    print("V3 Fine-tuning Complete!")
    print("=" * 60)
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nBest model saved to: {best_model_path}")
    print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    
    # Test on held-out session
    print("\n=== Testing on held-out session ===")
    test_results = trainer.test(
        model,
        dataloaders=data_module.test_dataloader(),
        ckpt_path=best_model_path,
    )
    print(f"Test results: {test_results}")
    
    return best_model_path


if __name__ == "__main__":
    best_path = main()
