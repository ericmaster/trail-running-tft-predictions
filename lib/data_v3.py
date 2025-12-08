"""
TFT DataModule V3 - Fine-tuning with Garmin FIT data

This module extends the base TFTDataModule to support:
1. Garmin FIT resampled data with sparse nutritional/RPE fields
2. Missing temperature handling (synthetic constant)
3. Manual train/val/test split for small datasets (5/1/1)
4. New feature handling:
   - RPE: time_varying_known_real (runner decides effort level)
   - Intake counters: Additional targets (model predicts consumption)

For fine-tuning the V2 model with new Garmin data that includes sparse
nutrition and perceived exertion tracking from the NutritionLogger app.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import warnings

warnings.filterwarnings("ignore")


class TFTDataModuleV3(pl.LightningDataModule):
    """
    Lightning DataModule for TFT V3 fine-tuning with Garmin data.
    
    Key differences from TFTDataModule:
    - Supports Garmin FIT resampled data with sparse fields
    - Handles missing temperature (fills with constant)
    - Manual session split for small datasets
    - RPE as known real (you decide effort level before/during race)
    - Intake counters as additional targets (model predicts consumption)
    """
    
    # Default temperature value when not available (°C)
    DEFAULT_TEMPERATURE = 15.0
    
    # Garmin session files in chronological order
    GARMIN_SESSIONS = [
        "garmin-20150439204_ACTIVITY.csv",  # Session 1 - Train
        "garmin-20225156421_ACTIVITY.csv",  # Session 2 - Train
        "garmin-20331019679_ACTIVITY.csv",  # Session 3 - Train
        "garmin-20458469606_ACTIVITY.csv",  # Session 4 - Train
        "garmin-20526112104_ACTIVITY.csv",  # Session 5 - Train
        "garmin-20616576637_ACTIVITY.csv",  # Session 6 - Validation
        "garmin-20659499932_ACTIVITY.csv",  # Session 7 - Test
    ]
    
    def __init__(
        self,
        data_dir: str = "./data/fit-resampled",
        min_encoder_length: int = 1,
        max_encoder_length: int = 400,  # Match V2 model
        max_prediction_length: int = 200,
        batch_size: int = 16,  # Smaller batch for limited data
        num_workers: int = 4,
        time_idx: str = "time_idx",
        group_ids: List[str] = None,
        random_seed: int = 42,
        include_sparse_features: bool = True,
        default_temperature: float = DEFAULT_TEMPERATURE,
        # Manual split configuration (5 train, 1 val, 1 test)
        train_sessions: int = 5,
        val_sessions: int = 1,
        test_sessions: int = 1,
    ):
        """
        Initialize the TFT V3 DataModule for Garmin data fine-tuning.
        
        Args:
            data_dir: Directory containing Garmin FIT resampled CSV files
            min_encoder_length: Minimum encoder length (for cold-start)
            max_encoder_length: Maximum encoder length (should match V2: 400)
            max_prediction_length: Prediction horizon (should match V2: 200)
            batch_size: Batch size (smaller for limited data)
            num_workers: DataLoader workers
            time_idx: Time index column name
            group_ids: Group identifier columns
            random_seed: Random seed for reproducibility
            include_sparse_features: Whether to include rpe, intake counters
            default_temperature: Temperature value to use when missing
            train_sessions: Number of sessions for training
            val_sessions: Number of sessions for validation
            test_sessions: Number of sessions for testing
        """
        super().__init__()
        self.data_dir = data_dir
        self.min_encoder_length = min_encoder_length
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_idx = time_idx
        self.group_ids = group_ids or ["session_id"]
        self.random_seed = random_seed
        self.include_sparse_features = include_sparse_features
        self.default_temperature = default_temperature
        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions
        
        # Base target variables (same as V2 for compatibility)
        self.base_target_names = ['duration_diff', 'heartRate', 'temperature', 'cadence']
        
        # Sparse features from Garmin/NutritionLogger
        # RPE is a known real (runner decides effort level before/during race)
        self.rpe_feature = 'rpe'
        # Intake counters as known reals (runner follows nutrition plan)
        # These are NOT targets - adding targets would break V2 weight compatibility
        self.intake_features = ['water_intake', 'electrolytes_intake', 'food_intake']
        
        # ALWAYS use 4 targets for V2 compatibility (fine-tuning requires same architecture)
        # Sparse features (RPE, intake) are added as known inputs, not as additional targets
        self.target_names = self.base_target_names.copy()
        
        # Data storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.full_data = None
        
        # Datasets
        self.training = None
        self.validation = None
        self.test = None
    
    def _add_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add missing columns required by V2 model with default values.
        
        Args:
            df: Input DataFrame from Garmin data
            
        Returns:
            DataFrame with all required columns
        """
        # Add temperature if missing (Garmin FIT files don't have it)
        if 'temperature' not in df.columns:
            df['temperature'] = self.default_temperature
            print(f"  Added synthetic temperature column: {self.default_temperature}°C")
        
        # Ensure session_id_encoded is integer
        if 'session_id_encoded' in df.columns:
            df['session_id_encoded'] = df['session_id_encoded'].astype(int)
        
        # Convert RPE to string for categorical encoding (levels 0-4)
        if self.rpe_feature in df.columns:
            # Ensure integer values, then convert to string
            df[self.rpe_feature] = df[self.rpe_feature].fillna(0).astype(int).astype(str)
        
        return df
    
    def _validate_garmin_data(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Validate that Garmin data has required columns.
        
        Args:
            df: DataFrame to validate
            filename: Source filename for error messages
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_cols = [
            'distance', 'altitude', 'heartRate', 'cadence', 'speed',
            'duration', 'duration_diff', 'time_idx', 'session_id',
            'elevation_diff', 'elevation_gain', 'elevation_loss',
            'avg_heart_rate_so_far', 'session_id_encoded'
        ]
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {filename}: {missing}")
        
        # Check sparse features if enabled
        if self.include_sparse_features:
            # Check RPE (categorical: levels 0-4)
            if self.rpe_feature not in df.columns:
                print(f"  Warning: Missing RPE in {filename}, adding '0'")
                df[self.rpe_feature] = "0"  # String for categorical
            
            # Check intake features (known reals)
            for col in self.intake_features:
                if col not in df.columns:
                    print(f"  Warning: Missing {col} in {filename}, adding zeros")
                    df[col] = 0
        
        return True
    
    def prepare_data(self):
        """Load and prepare Garmin data with proper handling of missing columns."""
        all_sessions = []
        
        # Get available files
        available_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        
        print(f"Loading Garmin sessions from {self.data_dir}...")
        print(f"Found {len(available_files)} session files")
        
        for file in available_files:
            file_path = os.path.join(self.data_dir, file)
            print(f"  Loading: {file}")
            
            df = pd.read_csv(file_path)
            
            # Validate and add missing columns
            self._validate_garmin_data(df, file)
            df = self._add_missing_columns(df)
            
            all_sessions.append(df)
            print(f"    Rows: {len(df)}, Distance: {df['distance'].max():.0f}m")
        
        # Combine all sessions
        self.full_data = pd.concat(all_sessions, ignore_index=True)
        
        print(f"\nLoaded {len(all_sessions)} sessions with {len(self.full_data)} total data points")
        print(f"Columns: {list(self.full_data.columns)}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup train, validation, and test datasets with manual split."""
        
        if stage == 'fit' and self.training is not None:
            return
        if stage == 'test' and self.test is not None:
            return
        
        if self.full_data is None:
            self.prepare_data()
        
        # Get unique sessions in order
        session_ids = sorted(self.full_data['session_id'].unique())
        n_sessions = len(session_ids)
        
        print(f"\n=== Manual Session Split (5/1/1) ===")
        print(f"Total sessions: {n_sessions}")
        
        # Validate split configuration
        total_split = self.train_sessions + self.val_sessions + self.test_sessions
        if total_split > n_sessions:
            raise ValueError(f"Split requires {total_split} sessions but only {n_sessions} available")
        
        # Manual split
        train_session_ids = session_ids[:self.train_sessions]
        val_session_ids = session_ids[self.train_sessions:self.train_sessions + self.val_sessions]
        test_session_ids = session_ids[self.train_sessions + self.val_sessions:
                                        self.train_sessions + self.val_sessions + self.test_sessions]
        
        print(f"Train sessions ({len(train_session_ids)}): {train_session_ids}")
        print(f"Val sessions ({len(val_session_ids)}): {val_session_ids}")
        print(f"Test sessions ({len(test_session_ids)}): {test_session_ids}")
        
        # Create data splits
        self.train_data = self.full_data[self.full_data['session_id'].isin(train_session_ids)].copy()
        self.val_data = self.full_data[self.full_data['session_id'].isin(val_session_ids)].copy()
        self.test_data = self.full_data[self.full_data['session_id'].isin(test_session_ids)].copy()
        
        print(f"\nData points - Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        
        # Filter sessions that are too short
        min_required = self.max_encoder_length + self.max_prediction_length
        print(f"Minimum required sequence length: {min_required}")
        
        for split_name, split_data in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            session_lengths = split_data.groupby('session_id_encoded').size()
            short_sessions = session_lengths[session_lengths < min_required]
            if len(short_sessions) > 0:
                print(f"  Warning: {len(short_sessions)} {split_name} sessions too short (< {min_required} steps)")
        
        # Define feature groups (compatible with V2 model)
        time_varying_known_reals = [
            "altitude",
            "elevation_diff",
            "elevation_gain",
            "elevation_loss",
        ]
        
        # Time-varying known categoricals (discrete levels, not continuous)
        time_varying_known_categoricals = []
        
        # Add sparse features as known inputs for V2 compatibility
        if self.include_sparse_features:
            # RPE: discrete effort levels 0-4, better as categorical (learns embeddings per level)
            time_varying_known_categoricals.append(self.rpe_feature)
            # Intake counters: runner follows a nutrition plan (known beforehand)
            time_varying_known_reals.extend(self.intake_features)
            print(f"\nSparse features:")
            print(f"  RPE as categorical (5 levels): {self.rpe_feature}")
            print(f"  Intake as known reals: {self.intake_features}")
            print(f"NOTE: Keeping 4 targets for V2 weight compatibility")
        
        # Unknown reals: targets + context variables
        # Targets are ONLY the base 4 targets to maintain V2 compatibility
        time_varying_unknown_reals = self.target_names + [
            "speed",
            "avg_heart_rate_so_far",
            "duration",
        ]
        
        print(f"Time-varying known reals: {time_varying_known_reals}")
        print(f"Time-varying known categoricals: {time_varying_known_categoricals}")
        print(f"Time-varying unknown reals: {time_varying_unknown_reals}")
        print(f"Targets ({len(self.target_names)}): {self.target_names}")
        
        # Create normalizer for multi-target
        group_id_col = "session_id_encoded"
        target_normalizer = MultiNormalizer(
            [GroupNormalizer(groups=[group_id_col], transformation=None) 
             for _ in range(len(self.target_names))]
        )
        
        # Categorical encoders: session_id + RPE if sparse features enabled
        cat_encoders = {group_id_col: NaNLabelEncoder(add_nan=True)}
        if self.include_sparse_features:
            cat_encoders[self.rpe_feature] = NaNLabelEncoder(add_nan=True)
        
        # Create training dataset
        self.training = TimeSeriesDataSet(
            self.train_data,
            time_idx="time_idx",
            target=self.target_names,
            group_ids=[group_id_col],
            min_encoder_length=self.min_encoder_length,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=True,  # Data augmentation
            allow_missing_timesteps=False,
            categorical_encoders=cat_encoders,
            add_encoder_length=True,
        )
        
        # Create validation dataset
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training,
            self.val_data,
            predict=False,
            stop_randomization=True,
        )
        
        # Create test dataset
        self.test = TimeSeriesDataSet.from_dataset(
            self.training,
            self.test_data,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            predict=True,
            stop_randomization=True,
        )
        
        print(f"\nDataset samples - Training: {len(self.training)}, Validation: {len(self.validation)}, Test: {len(self.test)}")
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self.training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return self.validation.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return self.test.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def get_feature_info(self) -> Dict:
        """Return information about features for model configuration."""
        return {
            "target_names": self.target_names,
            "base_target_names": self.base_target_names,
            "rpe_feature": self.rpe_feature if self.include_sparse_features else None,
            "rpe_is_categorical": True,  # RPE has discrete levels 0-4
            "rpe_levels": ["0", "1", "2", "3", "4"],
            "intake_features": self.intake_features if self.include_sparse_features else [],
            "n_targets": len(self.target_names),  # Always 4 for V2 compatibility
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "include_sparse_features": self.include_sparse_features,
        }


class TFTDataModuleV3Combined(TFTDataModuleV3):
    """
    Extended DataModule that can load both Polar and Garmin data.
    
    Useful for testing if fine-tuned model still works on original data
    (catastrophic forgetting evaluation).
    """
    
    def __init__(
        self,
        polar_data_dir: str = "./data/resampled",
        garmin_data_dir: str = "./data/fit-resampled",
        use_polar_for_training: bool = False,
        **kwargs
    ):
        """
        Initialize combined data module.
        
        Args:
            polar_data_dir: Directory with Polar resampled data
            garmin_data_dir: Directory with Garmin resampled data
            use_polar_for_training: If True, include Polar data in training
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(data_dir=garmin_data_dir, **kwargs)
        self.polar_data_dir = polar_data_dir
        self.garmin_data_dir = garmin_data_dir
        self.use_polar_for_training = use_polar_for_training
        self.polar_data = None
    
    def load_polar_data(self) -> pd.DataFrame:
        """Load Polar data for comparison/evaluation."""
        all_sessions = []
        csv_files = [f for f in os.listdir(self.polar_data_dir) if f.endswith('.csv')]
        
        print(f"\nLoading {len(csv_files)} Polar sessions for comparison...")
        
        for file in csv_files[:5]:  # Load just a few for testing
            file_path = os.path.join(self.polar_data_dir, file)
            df = pd.read_csv(file_path)
            
            # Polar data doesn't have sparse features, add them with defaults
            if self.rpe_feature not in df.columns:
                df[self.rpe_feature] = "0"  # String for categorical
            for col in self.intake_features:
                if col not in df.columns:
                    df[col] = 0
            
            all_sessions.append(df)
        
        self.polar_data = pd.concat(all_sessions, ignore_index=True)
        print(f"Loaded {len(all_sessions)} Polar sessions with {len(self.polar_data)} data points")
        
        return self.polar_data


class TFTDataModuleGarminForV2(pl.LightningDataModule):
    """
    DataModule that loads Garmin data but uses V2 feature structure.
    
    This allows evaluating the V2 model (trained on Polar data) on Garmin data
    to see how well it generalizes without sparse features.
    
    Uses only 4 targets: duration_diff, heartRate, temperature, cadence
    No sparse features (RPE, intake counters) are included.
    """
    
    def __init__(
        self,
        data_dir: str = "./data/fit-resampled",
        batch_size: int = 32,
        num_workers: int = 0,
        max_encoder_length: int = 400,
        min_encoder_length: int = 1,
        max_prediction_length: int = 200,
        train_sessions: int = 5,
        val_sessions: int = 1,
        test_sessions: int = 1,
    ):
        """
        Initialize Garmin DataModule for V2 model evaluation.
        
        Args:
            data_dir: Directory containing resampled Garmin CSVs
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            max_encoder_length: Maximum encoder sequence length
            min_encoder_length: Minimum encoder sequence length
            max_prediction_length: Maximum prediction horizon
            train_sessions: Number of training sessions
            val_sessions: Number of validation sessions
            test_sessions: Number of test sessions
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions
        
        # V2-compatible target names (4 targets, no sparse features)
        self.target_names = ['duration_diff', 'heartRate', 'temperature', 'cadence']
        
        # Data storage
        self.full_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.training = None
        self.validation = None
        self.test = None
    
    def prepare_data(self):
        """Load and prepare Garmin data for V2-compatible evaluation."""
        print(f"\n=== Loading Garmin Data for V2 Evaluation ===")
        print(f"Data directory: {self.data_dir}")
        
        all_sessions = []
        csv_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        
        print(f"Found {len(csv_files)} session files")
        
        for file in csv_files:
            file_path = os.path.join(self.data_dir, file)
            print(f"  Loading: {file}")
            
            df = pd.read_csv(file_path)
            
            # Check required columns (V2-compatible)
            required_cols = [
                'distance', 'altitude', 'heartRate', 'cadence', 'speed',
                'duration', 'duration_diff', 'time_idx', 'session_id',
                'elevation_diff', 'elevation_gain', 'elevation_loss',
                'avg_heart_rate_so_far', 'session_id_encoded'
            ]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in {file}: {missing}")
            
            all_sessions.append(df)
            print(f"    Rows: {len(df)}, Distance: {df['distance'].max():.0f}m")
        
        self.full_data = pd.concat(all_sessions, ignore_index=True)
        print(f"\nLoaded {len(all_sessions)} sessions with {len(self.full_data)} total data points")
    
    def setup(self, stage: Optional[str] = None):
        """Setup train, validation, and test datasets with V2 feature structure."""
        
        if stage == 'fit' and self.training is not None:
            return
        if stage == 'test' and self.test is not None:
            return
        
        if self.full_data is None:
            self.prepare_data()
        
        # Get unique sessions in order
        session_ids = sorted(self.full_data['session_id'].unique())
        n_sessions = len(session_ids)
        
        print(f"\n=== Garmin Data for V2 Model (Session Split) ===")
        print(f"Total sessions: {n_sessions}")
        
        # Validate split configuration
        total_split = self.train_sessions + self.val_sessions + self.test_sessions
        if total_split > n_sessions:
            raise ValueError(f"Split requires {total_split} sessions but only {n_sessions} available")
        
        # Manual split
        train_session_ids = session_ids[:self.train_sessions]
        val_session_ids = session_ids[self.train_sessions:self.train_sessions + self.val_sessions]
        test_session_ids = session_ids[self.train_sessions + self.val_sessions:
                                        self.train_sessions + self.val_sessions + self.test_sessions]
        
        print(f"Train sessions ({len(train_session_ids)}): {train_session_ids}")
        print(f"Val sessions ({len(val_session_ids)}): {val_session_ids}")
        print(f"Test sessions ({len(test_session_ids)}): {test_session_ids}")
        
        # Create data splits
        self.train_data = self.full_data[self.full_data['session_id'].isin(train_session_ids)].copy()
        self.val_data = self.full_data[self.full_data['session_id'].isin(val_session_ids)].copy()
        self.test_data = self.full_data[self.full_data['session_id'].isin(test_session_ids)].copy()
        
        print(f"\nData points - Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        
        # V2-compatible feature groups (no RPE or intake features)
        time_varying_known_reals = [
            "altitude",
            "elevation_diff",
            "elevation_gain",
            "elevation_loss",
        ]
        
        time_varying_unknown_reals = self.target_names + [
            "speed",
            "avg_heart_rate_so_far",
            "duration",
        ]
        
        print(f"Time-varying known reals: {time_varying_known_reals}")
        print(f"Time-varying unknown reals: {time_varying_unknown_reals}")
        print(f"Targets (V2-compatible): {self.target_names}")
        
        # Create normalizer for multi-target
        group_id_col = "session_id_encoded"
        target_normalizer = MultiNormalizer(
            [GroupNormalizer(groups=[group_id_col], transformation=None) 
             for _ in range(len(self.target_names))]
        )
        
        # Create training dataset
        self.training = TimeSeriesDataSet(
            self.train_data,
            time_idx="time_idx",
            target=self.target_names,
            group_ids=[group_id_col],
            min_encoder_length=self.min_encoder_length,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=True,
            allow_missing_timesteps=False,
            categorical_encoders={group_id_col: NaNLabelEncoder(add_nan=True)},
            add_encoder_length=True,
        )
        
        # Create validation dataset
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training,
            self.val_data,
            predict=False,
            stop_randomization=True,
        )
        
        # Create test dataset
        self.test = TimeSeriesDataSet.from_dataset(
            self.training,
            self.test_data,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            predict=True,
            stop_randomization=True,
        )
        
        print(f"\nDataset samples - Training: {len(self.training)}, Validation: {len(self.validation)}, Test: {len(self.test)}")
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self.training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return self.validation.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return self.test.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def get_feature_info(self) -> Dict:
        """Return information about features for model configuration."""
        return {
            "target_names": self.target_names,
            "n_targets": len(self.target_names),
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "include_sparse_features": False,  # Always False for V2 compatibility
        }
