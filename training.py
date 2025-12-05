import glob
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lib.model import TrailRunningTFT
from lib.data import TFTDataModule

pl.seed_everything(42, workers=True)

def find_latest_checkpoint(checkpoint_dir: str = "./checkpoints/"):
    """Find the latest checkpoint file by modification time."""
    all_checkpoints = []
    for filename in glob.glob(os.path.join(checkpoint_dir, '*.ckpt')):
        all_checkpoints.append(filename)
    
    if all_checkpoints:
        return max(all_checkpoints, key=os.path.getmtime)
    return None

# Training function
def train_tft_model(
    data_dir: str = "./data/resampled",
    max_epochs: int = 60,
    min_encoder_length: int = 1, # For cold-start scenarios
    max_encoder_length: int = 400, # Increased for longer context windows
    max_prediction_length: int = 200,
    batch_size: int = 64,
    hidden_size: int = 45,
    hidden_continuous_size: int = 37,
    attention_head_size: int = 3,
    learning_rate: float = 1e-05,
    # New parameters for quantile loss and cold-start emphasis
    use_quantile_loss: bool = False,
    quantile_alpha: float = 0.6,
    cold_start_weight: float = 0.0,  # Weight for emphasizing cold-start samples (0.0-1.0)
    checkpoint_version: str = "",  # Version suffix for checkpoint (e.g., "_v2", "_quantile")
    resume_from_checkpoint: bool = True,  # Whether to resume from latest checkpoint
):
    """
    Train the TFT model.
    
    Args:
        data_dir: Directory containing the training data
        max_epochs: Maximum number of training epochs
        min_encoder_length: Minimum encoder length (1 for cold-start support)
        max_encoder_length: Maximum encoder length (longer context window)
        max_prediction_length: Length of prediction horizon
        batch_size: Batch size for training
        hidden_size: Hidden size of the model
        learning_rate: Learning rate
        use_quantile_loss: If True, use asymmetric quantile loss for cold-start bias correction
        quantile_alpha: Alpha for quantile loss (0.7 = penalize under-predictions 2.3x more)
        cold_start_weight: Weight for emphasizing cold-start samples during training (0.0-1.0)
        checkpoint_version: Version suffix for checkpoint filename (e.g., "_v2", "_quantile")
        resume_from_checkpoint: Whether to resume from latest checkpoint
    
    Returns:
        Trained model and data module
    """
    # Determine checkpoint directory based on version
    checkpoint_dir = "checkpoints" if not checkpoint_version else f"checkpoints{checkpoint_version}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create data module
    # Note: use_sliding_windows=False for training (only used for inference)
    # randomize_length=True provides data augmentation during training
    data_module = TFTDataModule(
        data_dir=data_dir,
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size,
        use_sliding_windows=False,  # Disabled for training
        cold_start_weight=cold_start_weight,  # Pass cold-start weight to data module
    )
    
    # Setup data
    data_module.setup(stage="fit")
    
    # Create model
    model = TrailRunningTFT.from_dataset(
        data_module.training,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        output_size=[1] * 4, # Multi-target output: duration_diff, heartRate, temperature, cadence
        use_quantile_loss=use_quantile_loss,  # Enable asymmetric loss
        quantile_alpha=quantile_alpha,  # Control over/under prediction bias
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"best-checkpoint{checkpoint_version}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min",
        min_delta=0.001,  # Minimum change to qualify as improvement
        strict=True
    )

    learning_rate_callback = LearningRateMonitor(logging_interval="step")
    
    # Add StochasticWeightAveraging for better generalization
    # from lightning.pytorch.callbacks import StochasticWeightAveraging
    # swa_callback = StochasticWeightAveraging(
    #     swa_lrs=1e-5,  # Lower learning rate for SWA
    #     swa_epoch_start=0.8,  # Start SWA at 80% of training
    #     annealing_epochs=5
    # )

    logger = pl.loggers.CSVLogger("logs", name="tft_model")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        strategy='auto',  # Let Lightning choose the best strategy
        logger=logger,
        devices=2,
        gradient_clip_val=0.04,
        # limit_train_batches=50,  # Limit for faster training during development
        enable_checkpointing=True,
        precision="32-true",  # Use full precision for better accuracy
        # precision="16-mixed",  # Mixed precision for speed and resource efficiency
        # accumulate_grad_batches=2,  # Gradient accumulation for effective larger batch size
        # val_check_interval=0.25,  # Check validation more frequently (4 times per epoch)
        # log_every_n_steps=10,  # Log more frequently
        callbacks=[
            early_stopping_callback,
            learning_rate_callback,
            checkpoint_callback,
            # swa_callback
        ]
    )

    ckpt_path = None
    if resume_from_checkpoint:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            print(f"Cargando modelo desde checkpoint: {ckpt_path}")
        else:
            print(f"No se encontró checkpoint en {checkpoint_dir}, entrenando desde cero.")
    else:
        print(f"Entrenando desde cero (resume_from_checkpoint=False)")
    
    # Train model
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    
    return model, data_module, trainer


if __name__ == "__main__":
    # Train the model with quantile loss for cold-start bias correction
    # Version _v2: Uses asymmetric quantile loss (alpha=0.7) to penalize under-predictions
    model, data_module, trainer = train_tft_model(
        data_dir="./data/resampled",
        max_epochs=60,  # Likely early stop will trigger
        min_encoder_length=1,  # For cold-start scenarios
        max_encoder_length=400,  # Longer context window for better predictions
        max_prediction_length=200,
        batch_size=64,
        hidden_size=57,
        hidden_continuous_size=44,
        attention_head_size=3,
        learning_rate=5e-06,
        # New parameters for cold-start bias correction
        use_quantile_loss=True,  # Enable asymmetric quantile loss
        quantile_alpha=0.51,  # Penalize under-predictions more than over-predictions
        cold_start_weight=0.3,  # 30% more emphasis on cold-start samples
        checkpoint_version="_v2",  # Save to checkpoints_v2/
        resume_from_checkpoint=False,  # Start fresh training
    )
    
    print("Training completed!")
    
    # Make predictions on test set
    # test_predictions = model.predict(data_module.test_dataloader())
    # print(f"Test predictions shape: {test_predictions.shape}")