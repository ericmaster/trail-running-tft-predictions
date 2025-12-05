"""
Hyperparameter Optimization for Trail Running TFT Model using Optuna.

This script runs hyperparameter optimization using Optuna directly,
supporting the custom AsymmetricSMAPE loss function for cold-start bias correction.

Run with: python hyperparameter_optimization.py
"""

import os
import pickle
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lib.data import TFTDataModule
from lib.model import TrailRunningTFT

# Set random seed for reproducibility
pl.seed_everything(42, workers=True)


def objective(
    trial: optuna.Trial,
    train_dataloader,
    val_dataloader,
    training_dataset,
    max_epochs: int = 25,
    use_quantile_loss: bool = True,
):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        training_dataset: Training TimeSeriesDataSet (for model creation)
        max_epochs: Maximum epochs per trial
        use_quantile_loss: Whether to use asymmetric SMAPE loss
    
    Returns:
        Validation loss (to minimize)
    """
    # Sample hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 32, 80)
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 20, 64)
    attention_head_size = trial.suggest_int("attention_head_size", 2, 5)
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.15, 0.45)
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 0.2)
    
    # Asymmetric loss hyperparameters (only if using quantile loss)
    if use_quantile_loss:
        # Search for optimal alpha - key parameter for bias correction
        # 0.5 = symmetric, 0.55 = slight under-prediction penalty
        quantile_alpha = trial.suggest_float("quantile_alpha", 0.50, 0.55)
    else:
        quantile_alpha = 0.5  # Symmetric (standard SMAPE)
    
    # Create model with sampled hyperparameters
    model = TrailRunningTFT.from_dataset(
        training_dataset,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        learning_rate=learning_rate,
        dropout=dropout,
        output_size=[1] * 4,  # Multi-target: duration_diff, heartRate, temperature, cadence
        use_quantile_loss=use_quantile_loss,
        quantile_alpha=quantile_alpha,
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=False,
        mode="min",
    )
    
    # Optuna pruning callback - prune unpromising trials early
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,  # Single GPU for Optuna (multi-GPU causes issues)
        gradient_clip_val=gradient_clip_val,
        limit_train_batches=80,  # Limit batches for faster trials
        limit_val_batches=40,
        precision="32-true",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,  # Disable logging for cleaner output
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


def run_hyperparameter_optimization(
    data_dir: str = "./data/resampled",
    n_trials: int = 100,
    max_epochs: int = 25,
    batch_size: int = 64,
    min_encoder_length: int = 1,
    max_encoder_length: int = 400,
    max_prediction_length: int = 200,
    use_quantile_loss: bool = True,
    output_file: str = "./optuna_study_asymmetric.pkl",
    study_name: str = "tft_asymmetric_smape"
):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        data_dir: Directory containing resampled training data
        n_trials: Number of hyperparameter combinations to test
        max_epochs: Maximum epochs per trial
        batch_size: Batch size for training
        min_encoder_length: Minimum encoder length (for cold-start)
        max_encoder_length: Maximum encoder length
        max_prediction_length: Prediction horizon length
        use_quantile_loss: Whether to use asymmetric SMAPE loss
        output_file: Path to save the Optuna study results
        study_name: Name for the Optuna study
    """
    
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION FOR TRAIL RUNNING TFT MODEL")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Number of trials: {n_trials}")
    print(f"  - Max epochs per trial: {max_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Encoder length: {min_encoder_length}-{max_encoder_length}")
    print(f"  - Prediction length: {max_prediction_length}")
    print(f"  - Total sequence length: {max_encoder_length + max_prediction_length}")
    print(f"  - Loss function: {'Asymmetric SMAPE' if use_quantile_loss else 'Standard SMAPE'}")
    print(f"  - Using 1 GPU (multi-GPU not compatible with Optuna)")
    print(f"  - Results will be saved to: {output_file}\n")
    
    # Create data module
    print("Loading data...")
    data_module = TFTDataModule(
        data_dir=data_dir,
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size,
        num_workers=4,
    )
    
    # Setup data
    data_module.setup(stage="fit")
    
    # Get dataloaders
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )
    
    print("\nStarting hyperparameter optimization...\n")
    print("="*80)
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            train_dataloader,
            val_dataloader,
            data_module.training,
            max_epochs=max_epochs,
            use_quantile_loss=use_quantile_loss,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Save study results
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETED!")
    print(f"{'='*80}\n")
    
    with open(output_file, "wb") as fout:
        pickle.dump(study, fout)
    
    print(f"Study results saved to: {output_file}\n")
    
    # Display best hyperparameters
    print("Best hyperparameters found:")
    print("=" * 80)
    for param, value in study.best_trial.params.items():
        if isinstance(value, float):
            print(f"  {param:35}: {value:.6f}")
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
        if use_quantile_loss and 'quantile_alpha' in trial.params:
            print(f"   quantile_alpha = {trial.params['quantile_alpha']:.4f}")
    print("=" * 80)
    
    # Generate suggested training command
    print("\n" + "="*80)
    print("SUGGESTED TRAINING COMMAND:")
    print("="*80)
    best_params = study.best_trial.params
    print(f"""
python training.py with:
    hidden_size={best_params.get('hidden_size', 45)}
    hidden_continuous_size={best_params.get('hidden_continuous_size', 37)}
    attention_head_size={best_params.get('attention_head_size', 3)}
    learning_rate={best_params.get('learning_rate', 1e-5):.6f}
    dropout (in model.py defaults)={best_params.get('dropout', 0.25):.4f}
    gradient_clip_val={best_params.get('gradient_clip_val', 0.04):.4f}
    quantile_alpha={best_params.get('quantile_alpha', 0.6):.4f}
""")
    print("="*80)
    
    return study


if __name__ == "__main__":
    study = run_hyperparameter_optimization(
        data_dir="./data/resampled",
        n_trials=80,  # Good number for exploration
        max_epochs=30,  # Balanced for thorough training
        batch_size=64,
        min_encoder_length=1,
        max_encoder_length=400,  # Same as training.py
        max_prediction_length=200,
        use_quantile_loss=True,  # Use asymmetric SMAPE loss
        output_file="./optuna_study_asymmetric.pkl",
        study_name="tft_asymmetric_smape_v1"
    )
    
    print("\nHyperparameter optimization completed successfully!")
