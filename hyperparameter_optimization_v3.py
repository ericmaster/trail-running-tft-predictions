"""
Hyperparameter Optimization for V3 Fine-tuning with Garmin Data.

This script runs hyperparameter optimization using Optuna for the V3 model,
which fine-tunes the V2 pre-trained weights on Garmin data with sparse features.

Key differences from V2 optimization:
1. Much smaller dataset (5 train sessions vs 47+ for V2)
2. Focus on preventing overfitting (higher regularization search range)
3. Lower learning rates (1e-7 to 1e-5 vs 5e-6 to 5e-4)
4. Shorter epochs per trial (quick overfitting expected)
5. Optional layer freezing search

Run with: python hyperparameter_optimization_v3.py
"""

import os
import pickle
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import torch

from lib.data_v3 import TFTDataModuleV3
from lib.model import TrailRunningTFT
from lib.model_v3 import load_v2_weights_into_v3

# Set random seed for reproducibility
pl.seed_everything(42, workers=True)

# V2 checkpoint for weight initialization
V2_CHECKPOINT_PATH = "./checkpoints_v2/best-checkpoint_v2-epoch=27-val_loss=0.12-v1.ckpt"


def objective(
    trial: optuna.Trial,
    train_dataloader,
    val_dataloader,
    training_dataset,
    target_names: list,
    max_epochs: int = 15,
    use_quantile_loss: bool = True,
):
    """
    Optuna objective function for V3 hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        training_dataset: Training TimeSeriesDataSet (for model creation)
        target_names: List of target variable names
        max_epochs: Maximum epochs per trial (shorter for V3 due to quick overfitting)
        use_quantile_loss: Whether to use asymmetric SMAPE loss
    
    Returns:
        Validation loss (to minimize)
    """
    # Sample hyperparameters - V3-specific ranges (focus on regularization)
    
    # Architecture must match V2 for weight loading
    hidden_size = 57  # Fixed to match V2
    hidden_continuous_size = 44  # Fixed to match V2
    attention_head_size = 3  # Fixed to match V2
    # hidden_size = trial.suggest_int ("hidden_size", 55, 64)
    # hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 40, 48)
    # attention_head_size = trial.suggest_int("attention_head_size", 3, 5)
    
    # Fine-tuning hyperparameters (main search space)
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 5e-5, log=True)
    dropout = trial.suggest_float("dropout", 0.30, 0.40)  # Higher range for overfitting prevention
    weight_decay = trial.suggest_float("weight_decay", 0.005, 0.02, log=True)  # L2 regularization
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 0.1)
    
    # Asymmetric loss hyperparameters
    if use_quantile_loss:
        quantile_alpha = trial.suggest_float("quantile_alpha", 0.50, 0.55)
    else:
        quantile_alpha = 0.5
    
    # Layer freezing options
    freeze_encoder = trial.suggest_categorical("freeze_encoder", [True, False])
    freeze_vsn = trial.suggest_categorical("freeze_vsn", [True, False])
    
    # Create model with sampled hyperparameters
    model = TrailRunningTFT.from_dataset(
        training_dataset,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        learning_rate=learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
        output_size=[1] * len(target_names),
        use_quantile_loss=use_quantile_loss,
        quantile_alpha=quantile_alpha,
    )
    
    # Load V2 weights
    try:
        load_v2_weights_into_v3(model, V2_CHECKPOINT_PATH, strict=False, verbose=False)
    except Exception as e:
        print(f"  Warning: Could not load V2 weights: {e}")
    
    # Apply layer freezing
    if freeze_encoder:
        for name, param in model.named_parameters():
            if 'lstm_encoder' in name or 'encoder_lstm' in name:
                param.requires_grad = False
    
    if freeze_vsn:
        for name, param in model.named_parameters():
            if 'variable_selection' in name or 'vsn' in name.lower():
                param.requires_grad = False
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,  # Very aggressive early stopping for small dataset
        verbose=False,
        mode="min",
        min_delta=0.001,
    )
    
    # Optuna pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,  # Single GPU for Optuna
        gradient_clip_val=gradient_clip_val,
        precision="32-true",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        callbacks=[early_stopping, pruning_callback],
    )
    
    # Train
    try:
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')
    
    # Get best validation loss
    val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
    
    return float(val_loss)


def run_hyperparameter_optimization_v3(
    data_dir: str = "./data/fit-resampled",
    n_trials: int = 50,
    max_epochs: int = 15,
    batch_size: int = 16,
    min_encoder_length: int = 1,
    max_encoder_length: int = 400,
    max_prediction_length: int = 200,
    use_quantile_loss: bool = True,
    include_sparse_features: bool = True,
    output_file: str = "./optuna_study_v3.pkl",
    study_name: str = "tft_v3_finetuning"
):
    """
    Run hyperparameter optimization for V3 fine-tuning.
    
    Args:
        data_dir: Directory containing Garmin resampled data
        n_trials: Number of hyperparameter combinations to test
        max_epochs: Maximum epochs per trial (shorter for V3)
        batch_size: Batch size for training
        min_encoder_length: Minimum encoder length
        max_encoder_length: Maximum encoder length
        max_prediction_length: Prediction horizon length
        use_quantile_loss: Whether to use asymmetric SMAPE loss
        include_sparse_features: Include RPE and intake features
        output_file: Path to save the Optuna study results
        study_name: Name for the Optuna study
    """
    
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION FOR V3 FINE-TUNING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Number of trials: {n_trials}")
    print(f"  - Max epochs per trial: {max_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Encoder length: {min_encoder_length}-{max_encoder_length}")
    print(f"  - Prediction length: {max_prediction_length}")
    print(f"  - Include sparse features: {include_sparse_features}")
    print(f"  - Loss function: {'Asymmetric SMAPE' if use_quantile_loss else 'Standard SMAPE'}")
    print(f"  - V2 checkpoint: {V2_CHECKPOINT_PATH}")
    print(f"  - Results will be saved to: {output_file}\n")
    
    # Check V2 checkpoint exists
    if not os.path.exists(V2_CHECKPOINT_PATH):
        raise FileNotFoundError(f"V2 checkpoint not found: {V2_CHECKPOINT_PATH}")
    
    # Create data module
    print("Loading Garmin data...")
    data_module = TFTDataModuleV3(
        data_dir=data_dir,
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size,
        num_workers=4,
        include_sparse_features=include_sparse_features,
        train_sessions=5,
        val_sessions=1,
        test_sessions=1,
    )
    
    # Setup data
    data_module.prepare_data()
    data_module.setup(stage="fit")
    
    # Get feature info
    feature_info = data_module.get_feature_info()
    target_names = feature_info["target_names"]
    
    print(f"\nFeature configuration:")
    print(f"  - Targets ({len(target_names)}): {target_names}")
    print(f"  - Include sparse: {feature_info['include_sparse_features']}")
    
    # Get dataloaders
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    
    print(f"\nDataloaders:")
    print(f"  - Training batches: {len(train_dataloader)}")
    print(f"  - Validation batches: {len(val_dataloader)}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    
    print("\nStarting hyperparameter optimization...\n")
    print("=" * 80)
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            train_dataloader,
            val_dataloader,
            data_module.training,
            target_names=target_names,
            max_epochs=max_epochs,
            use_quantile_loss=use_quantile_loss,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Save study results
    print(f"\n{'=' * 80}")
    print("OPTIMIZATION COMPLETED!")
    print(f"{'=' * 80}\n")
    
    with open(output_file, "wb") as fout:
        pickle.dump(study, fout)
    
    print(f"Study results saved to: {output_file}\n")
    
    # Display best hyperparameters
    print("Best hyperparameters found:")
    print("=" * 80)
    for param, value in study.best_trial.params.items():
        if isinstance(value, float):
            print(f"  {param:35}: {value:.8f}")
        else:
            print(f"  {param:35}: {value}")
    print("=" * 80)
    
    # Display best trial info
    print(f"\nBest validation loss: {study.best_trial.value:.6f}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"\nTotal trials completed: {len([t for t in study.trials if t.value is not None])}")
    
    # Show top 5 trials
    print("\nTop 5 trials:")
    print("=" * 80)
    completed_trials = [t for t in study.trials if t.value is not None]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)
    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"{i}. Trial {trial.number}: val_loss = {trial.value:.6f}")
        print(f"   lr={trial.params.get('learning_rate', 0):.2e}, "
              f"dropout={trial.params.get('dropout', 0):.2f}, "
              f"freeze_encoder={trial.params.get('freeze_encoder', False)}, "
              f"freeze_vsn={trial.params.get('freeze_vsn', False)}")
    print("=" * 80)
    
    # Generate suggested training command
    print("\n" + "=" * 80)
    print("SUGGESTED TRAINING COMMAND:")
    print("=" * 80)
    best_params = study.best_trial.params
    print(f"""
Update training_v3.py FINE_TUNE_CONFIG with:
    learning_rate = {best_params.get('learning_rate', 1e-6):.8f}
    dropout = {best_params.get('dropout', 0.30):.4f}
    weight_decay = {best_params.get('weight_decay', 0.01):.6f}
    gradient_clip_val = {best_params.get('gradient_clip_val', 0.5):.4f}
    quantile_alpha = {best_params.get('quantile_alpha', 0.55):.4f}
    
Add flags:
    --freeze-encoder: {best_params.get('freeze_encoder', False)}
    --freeze-vsn: {best_params.get('freeze_vsn', False)}

Example:
    python training_v3.py {'--freeze-encoder' if best_params.get('freeze_encoder', False) else ''} {'--freeze-vsn' if best_params.get('freeze_vsn', False) else ''}
""")
    print("=" * 80)
    
    return study


if __name__ == "__main__":
    study = run_hyperparameter_optimization_v3(
        data_dir="./data/fit-resampled",
        n_trials=50,  # Fewer trials since quick evaluation
        max_epochs=15,  # Short - overfitting happens fast
        batch_size=16,
        min_encoder_length=1,
        max_encoder_length=400,
        max_prediction_length=200,
        use_quantile_loss=True,
        include_sparse_features=True,  # Test with sparse features
        output_file="./optuna_study_v3.pkl",
        study_name="tft_v3_finetuning_v1"
    )
    
    print("\nV3 Hyperparameter optimization completed successfully!")
