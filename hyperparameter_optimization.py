"""
Hyperparameter Optimization for Trail Running TFT Model using Optuna.

This script runs hyperparameter optimization using the pytorch-forecasting's
built-in Optuna integration.

Run with: python hyperparameter_optimization.py
"""

import pickle
import lightning.pytorch as pl
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from lib.data import TFTDataModule

# Set random seed for reproducibility
pl.seed_everything(42, workers=True)


def run_hyperparameter_optimization(
    data_dir: str = "./data/resampled",
    n_trials: int = 100,  # More trials for broader search
    max_epochs: int = 25,  # Balanced training time
    batch_size: int = 64,
    min_encoder_length: int = 1,
    max_encoder_length: int = 300,  # Middle ground: 50% increase from v9's 200
    max_prediction_length: int = 200,
    output_file: str = "./optuna_study.pkl"
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
        output_file: Path to save the Optuna study results
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
    print(f"  - Strategy: Broader search with moderate context increase")
    print(f"  - Using 1 GPU (multi-GPU not compatible with Optuna's dynamic sampling)")
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
    print("\nStarting hyperparameter optimization...\n")
    
    # Run Optuna study for hyperparameter optimization
    # Broader search around version_9 values with architectural exploration
    # version_9 baseline: lr=2.5e-5, dropout=0.25, hidden_size=32,
    # hidden_continuous_size=29, attention_head_size=3, lstm_layers=1
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="./optuna_checkpoints",
        n_trials=n_trials,
        max_epochs=max_epochs,
        gradient_clip_val_range=(0.01, 0.15),
        hidden_size_range=(35, 64),
        hidden_continuous_size_range=(25, 64),
        attention_head_size_range=(3, 5),  # Keep focused on proven range
        learning_rate_range=(1e-5, 1e-4),
        dropout_range=(0.2, 0.40),
        trainer_kwargs=dict(
            limit_train_batches=100,
            devices=1,  # Single GPU
            accelerator='gpu',
            precision="32-true",
        ),
        reduce_on_plateau_patience=5,
        use_learning_rate_finder=False,
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
        print(f"  {param:35}: {value}")
    print("=" * 80)
    
    # Display best trial info
    print(f"\nBest validation loss: {study.best_trial.value:.6f}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"\nTotal trials completed: {len(study.trials)}")
    
    # Show top 5 trials
    print("\nTop 5 trials:")
    print("=" * 80)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, trial in enumerate(sorted_trials[:5], 1):
        if trial.value is not None:
            print(f"{i}. Trial {trial.number}: val_loss = {trial.value:.6f}")
    print("=" * 80)
    
    return study


if __name__ == "__main__":
    study = run_hyperparameter_optimization(
        data_dir="./data/resampled",
        n_trials=100,  # More trials for broader exploration
        max_epochs=40,  # Balanced for 400 encoder context
        batch_size=64,
        min_encoder_length=1,
        max_encoder_length=400,  # Moderate increase: 2x version_9's 200
        max_prediction_length=200,
        output_file="./optuna_study_broader.pkl"  # New strategy file
    )
    
    print("\nHyperparameter optimization completed successfully!")
