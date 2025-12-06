"""
FIT Data Processor

A library class for processing trail running data from Garmin FIT files
(captured with NutritionLogger app) and converting them to structured CSV format.

Features:
- Extracts standard record data (heart rate, altitude, distance, speed, cadence, power)
- Extracts custom developer fields (RPE, water/electrolytes/food intake counts)
- Processes high-frequency accelerometer/gyroscope data with spectral feature extraction
- Resamples data to 5-meter distance intervals (matching existing pipeline)
- Calculates fatigue proxy features (elevation changes, avg HR, etc.)

Data Source: Garmin Fenix 7 Pro via NutritionLogger app
https://github.com/ericmaster/NutritionLogger
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from fitparse import FitFile
except ImportError:
    raise ImportError("fitparse is required. Install with: pip install fitparse")

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
except ImportError:
    raise ImportError("scipy is required. Install with: pip install scipy")


class FitDataProcessor:
    """
    A class for processing FIT files from Garmin watches with NutritionLogger app.
    
    This class handles:
    - Extracting record messages (1 Hz data: HR, altitude, distance, speed, cadence, power)
    - Extracting developer fields (RPE, water, electrolytes, food intake)
    - Processing high-frequency accelerometer/gyroscope data (100 Hz)
    - Computing spectral features (PSD, dominant frequencies, energy bands)
    - Resampling to 5-meter distance intervals
    - Feature engineering (elevation diff, fatigue proxies, etc.)
    """
    
    # Standard fields to extract from FIT record messages
    RECORD_FIELDS = [
        "timestamp",
        "distance",
        "enhanced_altitude",
        "enhanced_speed",
        "heart_rate",
        "cadence",
        "power",
        "position_lat",
        "position_long",
    ]
    
    # Developer fields from NutritionLogger app
    DEVELOPER_FIELDS = [
        "rate_of_perceived_exertion",
        "water_intake_count",
        "electrolytes_intake_count",
        "food_intake_count",
    ]
    
    # High-frequency sensor data fields
    ACCELEROMETER_FIELDS = ["calibrated_accel_x", "calibrated_accel_y", "calibrated_accel_z"]
    GYROSCOPE_FIELDS = ["calibrated_gyro_x", "calibrated_gyro_y", "calibrated_gyro_z"]
    
    # Spectral analysis parameters
    ACCEL_GYRO_SAMPLE_RATE = 100  # Hz (samples per record are at ~10ms intervals)
    
    def __init__(
        self,
        input_dir: str = "./data/fitfiles",
        output_dir: str = "./data/fit-processed",
        resampled_output_dir: str = "./data/fit-resampled",
        distance_interval: float = 5.0,  # meters
        spectral_window_seconds: float = 5.0,  # Window for spectral analysis
    ):
        """
        Initialize the FitDataProcessor.
        
        Args:
            input_dir: Directory containing input FIT files
            output_dir: Directory to save processed CSV files (before resampling)
            resampled_output_dir: Directory to save resampled CSV files (5m intervals)
            distance_interval: Resampling interval in meters (default: 5.0)
            spectral_window_seconds: Window size for spectral feature extraction
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.resampled_output_dir = resampled_output_dir
        self.distance_interval = distance_interval
        self.spectral_window_seconds = spectral_window_seconds
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.resampled_output_dir, exist_ok=True)
        
        # Metadata file paths
        self.metadata_file = os.path.join(self.output_dir, "sessions_metadata.csv")
    
    def _semicircles_to_degrees(self, semicircles: Optional[int]) -> Optional[float]:
        """Convert FIT semicircles to degrees."""
        if semicircles is None:
            return None
        return semicircles * (180.0 / 2**31)
    
    def _extract_session_metadata(self, fitfile: FitFile, file_name: str) -> Dict[str, Any]:
        """
        Extract session-level metadata from FIT file.
        
        Args:
            fitfile: Parsed FitFile object
            file_name: Name of the source FIT file
            
        Returns:
            Dictionary containing session metadata
        """
        metadata = {"file_name": file_name}
        
        for session in fitfile.get_messages("session"):
            for field in session.fields:
                # Map common session fields
                field_mapping = {
                    "timestamp": "end_time",
                    "start_time": "start_time",
                    "total_elapsed_time": "duration_seconds",
                    "total_distance": "distance_meters",
                    "total_ascent": "ascent_meters",
                    "total_descent": "descent_meters",
                    "total_calories": "kilo_calories",
                    "avg_heart_rate": "avg_heart_rate",
                    "max_heart_rate": "max_heart_rate",
                    "avg_running_cadence": "avg_cadence",
                    "max_running_cadence": "max_cadence",
                    "enhanced_avg_speed": "avg_speed",
                    "enhanced_max_speed": "max_speed",
                    "avg_power": "avg_power",
                    "max_power": "max_power",
                    "sport": "sport",
                    "sub_sport": "sub_sport",
                    "avg_vertical_oscillation": "avg_vertical_oscillation",
                    "avg_stance_time": "avg_stance_time",
                    "avg_step_length": "avg_step_length",
                }
                if field.name in field_mapping:
                    metadata[field_mapping[field.name]] = field.value
        
        return metadata
    
    def _extract_record_data(self, fitfile: FitFile) -> pd.DataFrame:
        """
        Extract record-level data (1 Hz) from FIT file.
        
        Returns:
            DataFrame with timestamp, distance, altitude, speed, HR, cadence, power,
            and developer fields (RPE, nutrition intake).
        """
        records = []
        
        for record in fitfile.get_messages("record"):
            row = {}
            for field in record.fields:
                if field.name in self.RECORD_FIELDS:
                    value = field.value
                    # Handle position fields (convert semicircles to degrees)
                    if field.name in ["position_lat", "position_long"]:
                        value = self._semicircles_to_degrees(value)
                    row[field.name] = value
                elif field.name in self.DEVELOPER_FIELDS:
                    row[field.name] = field.value
            
            if row:  # Only add if we got some data
                records.append(row)
        
        df = pd.DataFrame(records)
        
        # Rename columns to match existing pipeline conventions
        column_mapping = {
            "enhanced_altitude": "altitude",
            "enhanced_speed": "speed",
            "heart_rate": "heartRate",
            "position_lat": "latitude",
            "position_long": "longitude",
            "rate_of_perceived_exertion": "rpe",
            "water_intake_count": "water_intake",
            "electrolytes_intake_count": "electrolytes_intake",
            "food_intake_count": "food_intake",
        }
        df = df.rename(columns=column_mapping)
        
        return df
    
    def _extract_high_freq_data(
        self, fitfile: FitFile, message_type: str, fields: List[str]
    ) -> pd.DataFrame:
        """
        Extract high-frequency sensor data (accelerometer or gyroscope).
        
        Args:
            fitfile: Parsed FitFile object
            message_type: 'accelerometer_data' or 'gyroscope_data'
            fields: List of field names to extract
            
        Returns:
            DataFrame with timestamp and sensor values (expanded from arrays)
        """
        all_samples = []
        
        for record in fitfile.get_messages(message_type):
            timestamp = None
            timestamp_ms = 0
            sample_offsets = None
            sensor_data = {f: None for f in fields}
            
            for field in record.fields:
                if field.name == "timestamp":
                    timestamp = field.value
                elif field.name == "timestamp_ms":
                    timestamp_ms = field.value if field.value else 0
                elif field.name == "sample_time_offset":
                    sample_offsets = field.value
                elif field.name in fields:
                    sensor_data[field.name] = field.value
            
            if timestamp and sample_offsets:
                # Expand array data into individual samples
                for i, offset in enumerate(sample_offsets):
                    sample = {
                        "timestamp": timestamp,
                        "timestamp_ms": timestamp_ms + offset,
                    }
                    for f in fields:
                        if sensor_data[f] is not None and i < len(sensor_data[f]):
                            sample[f] = sensor_data[f][i]
                    all_samples.append(sample)
        
        return pd.DataFrame(all_samples)
    
    def _compute_spectral_features(
        self,
        data: np.ndarray,
        sample_rate: float = 100.0
    ) -> Dict[str, float]:
        """
        Compute spectral features from a 1D signal.
        
        Features extracted:
        - Power Spectral Density (PSD) in key frequency bands
        - Dominant frequency
        - Spectral entropy
        - Total signal power/energy
        
        Args:
            data: 1D numpy array of sensor values
            sample_rate: Sampling rate in Hz
            
        Returns:
            Dictionary of spectral features
        """
        if len(data) < 10:
            return self._empty_spectral_features()
        
        # Remove DC component and normalize
        data = data - np.mean(data)
        
        # Compute FFT
        n = len(data)
        frequencies = fftfreq(n, 1/sample_rate)
        fft_values = fft(data)
        
        # Power spectral density (one-sided)
        psd = np.abs(fft_values[:n//2])**2 / n
        freqs = frequencies[:n//2]
        
        # Frequency bands for running/movement analysis
        # Low freq (0.5-2 Hz): Step/stride frequency
        # Mid freq (2-5 Hz): Arm swing, body oscillation
        # High freq (5-15 Hz): Impact, vibrations
        bands = {
            "low": (0.5, 2.0),
            "mid": (2.0, 5.0),
            "high": (5.0, 15.0),
        }
        
        features = {}
        
        # Band powers
        for band_name, (f_low, f_high) in bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)
            features[f"psd_{band_name}"] = np.sum(psd[band_mask]) if np.any(band_mask) else 0.0
        
        # Total power
        features["psd_total"] = np.sum(psd)
        
        # Dominant frequency (excluding DC)
        valid_mask = freqs > 0.1
        if np.any(valid_mask):
            valid_psd = psd[valid_mask]
            valid_freqs = freqs[valid_mask]
            features["dominant_freq"] = valid_freqs[np.argmax(valid_psd)]
        else:
            features["dominant_freq"] = 0.0
        
        # Spectral entropy (measure of signal complexity)
        psd_norm = psd / (np.sum(psd) + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        features["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Spectral centroid (weighted average frequency)
        features["spectral_centroid"] = (
            np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        )
        
        return features
    
    def _empty_spectral_features(self) -> Dict[str, float]:
        """Return empty spectral features dictionary."""
        return {
            "psd_low": 0.0,
            "psd_mid": 0.0,
            "psd_high": 0.0,
            "psd_total": 0.0,
            "dominant_freq": 0.0,
            "spectral_entropy": 0.0,
            "spectral_centroid": 0.0,
        }
    
    def _compute_aggregate_spectral_features(
        self,
        accel_df: pd.DataFrame,
        gyro_df: pd.DataFrame,
        record_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute spectral features for each 1-second window aligned with record data.
        
        Optimized version that uses vectorized timestamp matching and processes
        data in chunks rather than individual records.
        
        Args:
            accel_df: Accelerometer data with timestamps
            gyro_df: Gyroscope data with timestamps
            record_df: Record-level data with timestamps
            
        Returns:
            DataFrame with spectral features for each record timestamp
        """
        # Define all feature names
        feature_names = []
        for prefix in ["accel", "gyro"]:
            for axis in ["x", "y", "z", "magnitude"]:
                for feat in ["psd_low", "psd_mid", "psd_high", "psd_total", 
                             "dominant_freq", "spectral_entropy", "spectral_centroid"]:
                    feature_names.append(f"{prefix}_{axis}_{feat}")
        
        if accel_df.empty and gyro_df.empty:
            # No high-frequency data, return empty features
            empty_features = {f: 0.0 for f in feature_names}
            return pd.DataFrame([empty_features] * len(record_df))
        
        # Convert timestamps once (avoid repeated conversions)
        if not record_df.empty and "timestamp" in record_df.columns:
            record_timestamps = pd.to_datetime(record_df["timestamp"]).values
        else:
            return pd.DataFrame()
        
        # Prepare sensor data with numeric timestamps for faster comparison
        accel_ts_numeric = None
        gyro_ts_numeric = None
        
        if not accel_df.empty:
            accel_df = accel_df.copy()
            accel_df["ts_numeric"] = pd.to_datetime(accel_df["timestamp"]).values.astype(np.int64)
            accel_ts_numeric = accel_df["ts_numeric"].values
        
        if not gyro_df.empty:
            gyro_df = gyro_df.copy()
            gyro_df["ts_numeric"] = pd.to_datetime(gyro_df["timestamp"]).values.astype(np.int64)
            gyro_ts_numeric = gyro_df["ts_numeric"].values
        
        # Convert record timestamps to numeric (nanoseconds)
        record_ts_numeric = record_timestamps.astype(np.int64)
        window_half = int(0.5 * 1e9)  # 0.5 seconds in nanoseconds
        
        # Pre-allocate result arrays for efficiency
        n_records = len(record_df)
        results = {f: np.zeros(n_records) for f in feature_names}
        
        # Process in batches for memory efficiency
        for i in range(n_records):
            ts = record_ts_numeric[i]
            window_start = ts - window_half
            window_end = ts + window_half
            
            # Process accelerometer
            if accel_ts_numeric is not None:
                mask = (accel_ts_numeric >= window_start) & (accel_ts_numeric < window_end)
                if mask.any():
                    for axis in ["x", "y", "z"]:
                        field_name = f"calibrated_accel_{axis}"
                        if field_name in accel_df.columns:
                            window_data = accel_df.loc[mask, field_name].values
                            axis_features = self._compute_spectral_features(
                                window_data, self.ACCEL_GYRO_SAMPLE_RATE
                            )
                            for feat_name, feat_value in axis_features.items():
                                results[f"accel_{axis}_{feat_name}"][i] = feat_value
                    
                    # Magnitude
                    ax = accel_df.loc[mask, "calibrated_accel_x"].values
                    ay = accel_df.loc[mask, "calibrated_accel_y"].values
                    az = accel_df.loc[mask, "calibrated_accel_z"].values
                    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
                    mag_features = self._compute_spectral_features(
                        magnitude, self.ACCEL_GYRO_SAMPLE_RATE
                    )
                    for feat_name, feat_value in mag_features.items():
                        results[f"accel_magnitude_{feat_name}"][i] = feat_value
            
            # Process gyroscope
            if gyro_ts_numeric is not None:
                mask = (gyro_ts_numeric >= window_start) & (gyro_ts_numeric < window_end)
                if mask.any():
                    for axis in ["x", "y", "z"]:
                        field_name = f"calibrated_gyro_{axis}"
                        if field_name in gyro_df.columns:
                            window_data = gyro_df.loc[mask, field_name].values
                            axis_features = self._compute_spectral_features(
                                window_data, self.ACCEL_GYRO_SAMPLE_RATE
                            )
                            for feat_name, feat_value in axis_features.items():
                                results[f"gyro_{axis}_{feat_name}"][i] = feat_value
                    
                    # Magnitude
                    gx = gyro_df.loc[mask, "calibrated_gyro_x"].values
                    gy = gyro_df.loc[mask, "calibrated_gyro_y"].values
                    gz = gyro_df.loc[mask, "calibrated_gyro_z"].values
                    magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
                    mag_features = self._compute_spectral_features(
                        magnitude, self.ACCEL_GYRO_SAMPLE_RATE
                    )
                    for feat_name, feat_value in mag_features.items():
                        results[f"gyro_magnitude_{feat_name}"][i] = feat_value
        
        return pd.DataFrame(results)
    
    # Cumulative intake counters (monotonically increasing integers)
    INTAKE_COUNTER_FIELDS = ["water_intake", "electrolytes_intake", "food_intake"]
    
    # RPE is a state field (0-4 levels, can go up and down)
    RPE_FIELD = "rpe"
    
    # All sparse event fields (integers, not interpolated)
    SPARSE_EVENT_FIELDS = [RPE_FIELD] + INTAKE_COUNTER_FIELDS
    
    def _correct_accidental_changes(
        self, 
        df: pd.DataFrame, 
        threshold_seconds: float = 10.0
    ) -> pd.DataFrame:
        """
        Correct accidental increases in intake counter fields that were quickly reverted.
        
        For intake counters: If a value increases and then decreases back within 
        threshold_seconds, the increase is considered accidental and corrected.
        
        For RPE: No correction needed as it's a state that can legitimately change.
        
        Args:
            df: DataFrame with timestamp and sparse event fields
            threshold_seconds: Maximum time for a change to be considered accidental
            
        Returns:
            DataFrame with corrected sparse event fields
        """
        if df.empty or "timestamp" not in df.columns:
            return df
        
        df = df.copy()
        timestamps = pd.to_datetime(df["timestamp"])
        
        # Only correct intake counters (not RPE which can legitimately go up/down)
        for field in self.INTAKE_COUNTER_FIELDS:
            if field not in df.columns:
                continue
            
            values = df[field].values.copy()
            
            # Ensure integer type
            values = np.round(values).astype(int)
            
            i = 0
            while i < len(values) - 1:
                current_val = values[i]
                
                # Look for an increase
                if i + 1 < len(values) and values[i + 1] > current_val:
                    increased_val = values[i + 1]
                    increase_idx = i + 1
                    
                    # Look for a decrease back to original within threshold
                    for j in range(increase_idx + 1, len(values)):
                        time_diff = (timestamps.iloc[j] - timestamps.iloc[increase_idx]).total_seconds()
                        
                        if time_diff > threshold_seconds:
                            # Too much time has passed, this was a real change
                            break
                        
                        if values[j] < increased_val:
                            # Found a decrease - check if it goes back to or below original
                            if values[j] <= current_val:
                                # This was an accidental increase, correct it
                                # Set all values from increase_idx to j-1 back to current_val
                                for k in range(increase_idx, j):
                                    values[k] = current_val
                                break
                            else:
                                # Partial decrease, might be complex - skip
                                break
                
                i += 1
            
            # Ensure monotonically non-decreasing (forward fill any remaining decreases)
            for i in range(1, len(values)):
                if values[i] < values[i - 1]:
                    values[i] = values[i - 1]
            
            df[field] = values
        
        # For RPE: just ensure it's an integer (0-4 range)
        if self.RPE_FIELD in df.columns:
            df[self.RPE_FIELD] = np.clip(
                np.round(df[self.RPE_FIELD].values).astype(int), 0, 4
            )
        
        return df
    
    def _resample_to_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to fixed distance intervals (5 meters).
        
        Follows the same approach as data_resampling.py for consistency
        with the existing TFT pipeline.
        """
        if df.empty or "distance" not in df.columns:
            return df
        
        # Separate sparse event fields (should not be interpolated)
        sparse_cols = [c for c in self.SPARSE_EVENT_FIELDS if c in df.columns]
        # Continuous cols excludes distance (will be index) and sparse cols
        continuous_cols = [c for c in df.columns if c not in sparse_cols and c != "distance"]
        
        # Fill missing values systematically for continuous columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != "distance" and col not in sparse_cols:
                df[col] = df[col].interpolate().ffill().bfill()
        
        # For sparse event fields: just forward fill (no interpolation)
        for col in sparse_cols:
            df[col] = df[col].ffill().bfill().astype(int)
        
        # Remove duplicate distances (keep first)
        df = df.drop_duplicates(subset="distance", keep="first")
        
        # Define resampling grid
        min_dist = np.floor(df["distance"].min())
        max_dist = np.ceil(df["distance"].max())
        target_distances = np.arange(min_dist, max_dist + self.distance_interval, self.distance_interval)
        
        # Set distance as index for resampling
        df_indexed = df.set_index("distance")
        
        # Reindex and interpolate continuous columns
        continuous_df = df_indexed[continuous_cols]
        continuous_resampled = (
            continuous_df
            .reindex(target_distances)
            .interpolate(method="linear")
        )
        
        # Reindex and forward-fill sparse event columns (no interpolation)
        if sparse_cols:
            sparse_df = df_indexed[sparse_cols]
            sparse_resampled = (
                sparse_df
                .reindex(target_distances)
                .ffill()
                .bfill()
                .astype(int)
            )
            # Combine
            result_df = pd.concat([continuous_resampled, sparse_resampled], axis=1)
        else:
            result_df = continuous_resampled
        
        result_df = result_df.reset_index().rename(columns={"index": "distance"})
        
        return result_df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features matching the existing pipeline.
        
        Features:
        - elevation_diff: Change in altitude between steps
        - elevation_gain: Cumulative positive elevation change
        - elevation_loss: Cumulative negative elevation change
        - duration_diff: Time between distance steps
        - avg_heart_rate_so_far: Running average of heart rate
        - elevation_gain/loss_of_last_100m: Rolling window terrain features
        """
        if df.empty:
            return df
        
        # Calculate duration from timestamp if available
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["duration"] = (
                df["timestamp"].diff().dt.total_seconds().cumsum().fillna(0)
            )
        
        # Elevation difference
        if "altitude" in df.columns:
            df["elevation_diff"] = df["altitude"].diff().fillna(0)
            
            # Cumulative elevation gain (positive changes)
            df["elevation_gain"] = (
                df["elevation_diff"].clip(lower=0).cumsum().fillna(0)
            )
            
            # Cumulative elevation loss (negative changes, stored as positive)
            df["elevation_loss"] = (
                df["elevation_diff"].clip(upper=0).abs().cumsum().fillna(0)
            )
        
        # Duration difference
        if "duration" in df.columns:
            df["duration_diff"] = df["duration"].diff().fillna(0)
        
        # === FATIGUE PROXY FEATURES ===
        if "heartRate" in df.columns:
            df["avg_heart_rate_so_far"] = df["heartRate"].expanding().mean().fillna(0)
        
        # Elevation features for last 100m (rolling window)
        window_size = int(100 / self.distance_interval)  # 100m / 5m = 20 points
        if "elevation_diff" in df.columns:
            df["elevation_gain_of_last_100m"] = (
                df["elevation_diff"]
                .clip(lower=0)
                .rolling(window=window_size, min_periods=1)
                .sum()
                .fillna(0)
            )
            
            df["elevation_loss_of_last_100m"] = (
                df["elevation_diff"]
                .clip(upper=0)
                .abs()
                .rolling(window=window_size, min_periods=1)
                .sum()
                .fillna(0)
            )
        
        return df
    
    def _extract_session_id(self, file_name: str) -> str:
        """Generate session ID from FIT file name."""
        # Remove .fit extension and any path components
        base_name = os.path.basename(file_name).replace(".fit", "").replace(".FIT", "")
        return f"garmin-{base_name}"
    
    def _extract_session_timestamp(self, session_id: str) -> int:
        """Extract a numeric identifier from session ID for encoding."""
        # For Garmin files, use the activity ID which is numeric
        try:
            # Extract numeric part from garmin-XXXXXX_ACTIVITY format
            parts = session_id.replace("garmin-", "").split("_")
            if parts:
                return int(parts[0])
        except (ValueError, IndexError):
            pass
        # Fallback to hash
        return abs(hash(session_id)) % (10**10)
    
    def _append_metadata(self, metadata: Dict[str, Any]) -> None:
        """Append session metadata to the metadata CSV file."""
        metadata_df = pd.DataFrame([metadata])
        
        if os.path.exists(self.metadata_file):
            metadata_df.to_csv(self.metadata_file, mode="a", header=False, index=False)
        else:
            metadata_df.to_csv(self.metadata_file, mode="w", header=True, index=False)
    
    def process_single_file(
        self,
        file_name: str,
        verbose: bool = True,
        extract_spectral: bool = True,
    ) -> bool:
        """
        Process a single FIT file and save as CSV.
        
        Args:
            file_name: Name of the FIT file to process
            verbose: Whether to print processing information
            extract_spectral: Whether to extract spectral features from high-freq data
            
        Returns:
            True if file was processed successfully
        """
        file_path = os.path.join(self.input_dir, file_name)
        
        # Check if output file already exists
        session_id = self._extract_session_id(file_name)
        output_file_name = os.path.join(self.output_dir, f"{session_id}.csv")
        resampled_file_name = os.path.join(self.resampled_output_dir, f"{session_id}.csv")
        
        if os.path.exists(resampled_file_name):
            if verbose:
                print(f"Skipping {file_name} - already processed")
            return False
        
        try:
            if verbose:
                print(f"Processing file: {file_name}")
            
            # Parse FIT file
            fitfile = FitFile(file_path)
            
            # Extract session metadata
            metadata = self._extract_session_metadata(fitfile, file_name)
            
            # Need to re-parse for each message type due to generator exhaustion
            fitfile = FitFile(file_path)
            record_df = self._extract_record_data(fitfile)
            
            if record_df.empty:
                if verbose:
                    print(f"No record data found in {file_name}")
                return False
            
            if verbose:
                print(f"  Extracted {len(record_df)} records")
            
            # Extract and process high-frequency sensor data
            if extract_spectral:
                fitfile = FitFile(file_path)
                accel_df = self._extract_high_freq_data(
                    fitfile, "accelerometer_data", self.ACCELEROMETER_FIELDS
                )
                
                fitfile = FitFile(file_path)
                gyro_df = self._extract_high_freq_data(
                    fitfile, "gyroscope_data", self.GYROSCOPE_FIELDS
                )
                
                if verbose:
                    print(f"  Extracted {len(accel_df)} accelerometer samples")
                    print(f"  Extracted {len(gyro_df)} gyroscope samples")
                
                # Compute spectral features for each record
                spectral_df = self._compute_aggregate_spectral_features(
                    accel_df, gyro_df, record_df
                )
                
                # Merge spectral features with record data
                if not spectral_df.empty:
                    record_df = pd.concat(
                        [record_df.reset_index(drop=True), spectral_df.reset_index(drop=True)],
                        axis=1
                    )
            
            # Add session identifier
            record_df["session_id"] = session_id
            record_df["session_id_encoded"] = self._extract_session_timestamp(session_id)
            
            # Calculate duration from timestamps
            if "timestamp" in record_df.columns:
                record_df["timestamp"] = pd.to_datetime(record_df["timestamp"])
                record_df["duration"] = (
                    record_df["timestamp"].diff().dt.total_seconds().cumsum().fillna(0)
                )
            
            # Trim initial samples where distance is 0.0 (GPS initialization)
            if "distance" in record_df.columns:
                zero_dist_mask = (record_df["distance"] == 0.0) | (record_df["distance"].isna())
                if zero_dist_mask.any():
                    first_moving_idx = None
                    for i in range(len(record_df)):
                        if not zero_dist_mask.iloc[i]:
                            first_moving_idx = i
                            break
                    
                    if first_moving_idx is not None and first_moving_idx > 0:
                        trim_idx = max(0, first_moving_idx - 1)
                        if verbose:
                            print(f"  Trimming {trim_idx} initial samples with distance=0")
                        record_df = record_df.iloc[trim_idx:].reset_index(drop=True)
                        
                        # Recalculate duration
                        if "timestamp" in record_df.columns:
                            record_df["duration"] = (
                                record_df["timestamp"].diff().dt.total_seconds().cumsum().fillna(0)
                            )
            
            # Correct accidental increases in sparse event fields before saving
            record_df = self._correct_accidental_changes(record_df)
            
            # Save raw processed data (before resampling)
            record_df.to_csv(output_file_name, index=False)
            if verbose:
                print(f"  Saved raw data to: {output_file_name}")
            
            # Resample to distance intervals
            resampled_df = self._resample_to_distance(record_df.copy())
            
            # Add derived features
            resampled_df = self._add_derived_features(resampled_df)
            
            # Add time index for TimeSeriesDataSet
            resampled_df = resampled_df.sort_values("distance").reset_index(drop=True)
            resampled_df["time_idx"] = range(len(resampled_df))
            
            # Final cleanup - fill any remaining NaN values
            resampled_df = resampled_df.ffill().bfill()
            
            # Ensure sparse event fields remain integers
            for col in self.SPARSE_EVENT_FIELDS:
                if col in resampled_df.columns:
                    resampled_df[col] = resampled_df[col].astype(int)
            
            # Save resampled data
            resampled_df.to_csv(resampled_file_name, index=False)
            if verbose:
                print(f"  Saved resampled data to: {resampled_file_name}")
                print(f"  Final shape: {resampled_df.shape}")
            
            # Save metadata
            self._append_metadata(metadata)
            
            return True
            
        except Exception as e:
            if verbose:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f"Error processing {file_name}: {exc_type}: {exc_value}")
                traceback.print_exception(exc_type, exc_value, exc_traceback)
            return False
    
    def process_files(
        self,
        verbose: bool = True,
        extract_spectral: bool = True,
        limit: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Process all FIT files in the input directory.
        
        Args:
            verbose: Whether to print processing information
            extract_spectral: Whether to extract spectral features
            limit: Maximum number of files to process (None for all)
            
        Returns:
            Tuple of (processed_count, skipped_count)
        """
        files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(".fit")
        ]
        
        if not files:
            print(f"No FIT files found in {self.input_dir}")
            return 0, 0
        
        processed_count = 0
        skipped_count = 0
        
        iterator = tqdm(files, desc="Processing FIT files") if verbose else files
        
        for file_name in iterator:
            if self.process_single_file(file_name, verbose=verbose, extract_spectral=extract_spectral):
                processed_count += 1
                if limit is not None and processed_count >= limit:
                    print(f"Reached processing limit of {limit}")
                    break
            else:
                skipped_count += 1
        
        print(f"\nProcessed {processed_count} files, skipped {skipped_count} files")
        return processed_count, skipped_count
    
    def get_processed_files(self) -> List[str]:
        """Get list of already processed CSV files."""
        if not os.path.exists(self.resampled_output_dir):
            return []
        return [f for f in os.listdir(self.resampled_output_dir) if f.endswith(".csv")]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing status."""
        total_fit_files = len([
            f for f in os.listdir(self.input_dir) if f.lower().endswith(".fit")
        ])
        processed_files = len(self.get_processed_files())
        
        return {
            "total_input_files": total_fit_files,
            "processed_files": processed_files,
            "remaining_files": total_fit_files - processed_files,
            "input_directory": self.input_dir,
            "raw_output_directory": self.output_dir,
            "resampled_output_directory": self.resampled_output_dir,
            "distance_interval": self.distance_interval,
        }


def main():
    """Main entry point for FIT data processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process FIT files from Garmin watches with NutritionLogger app"
    )
    parser.add_argument(
        "--input-dir",
        default="./data/fitfiles",
        help="Directory containing FIT files",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/fit-processed",
        help="Directory for raw processed CSV files",
    )
    parser.add_argument(
        "--resampled-dir",
        default="./data/fit-resampled",
        help="Directory for resampled CSV files",
    )
    parser.add_argument(
        "--no-spectral",
        action="store_true",
        help="Skip spectral feature extraction (faster processing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print processing statistics and exit",
    )
    
    args = parser.parse_args()
    
    processor = FitDataProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        resampled_output_dir=args.resampled_dir,
    )
    
    if args.stats:
        stats = processor.get_processing_stats()
        print("Processing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    processor.process_files(
        verbose=args.verbose,
        extract_spectral=not args.no_spectral,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
