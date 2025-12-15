"""
FastAPI server for Trail Running TFT predictions.

This API provides cold-start inference for trail running race predictions
using GPX files as input.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gpxpy
import gpxpy.gpx

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.model import TrailRunningTFT
from lib.data_v3 import TFTDataModuleGarminForV2

app = FastAPI(
    title="Trail Running TFT API",
    description="Cold-start race prediction using Temporal Fusion Transformers",
    version="1.0.0"
)

# CORS middleware for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = Path(__file__).parent.parent / "checkpoints_v2" / "best-checkpoint_v2-epoch=27-val_loss=0.12-v1.ckpt"
NORMALIZERS_PATH = Path(__file__).parent.parent / "checkpoints_v2" / "normalizers.pkl"
GPX_PRESETS_PATH = Path(__file__).parent.parent / "data" / "gpx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and normalizers at startup
model = None
normalizers_data = None

@app.on_event("startup")
async def load_model():
    """Load the V2 model and normalizers on startup."""
    global model, normalizers_data
    
    print(f"Loading V2 model from {MODEL_PATH}...")
    model = TrailRunningTFT.load_from_checkpoint(str(MODEL_PATH), map_location=DEVICE)
    model.eval()
    print(f"Model loaded successfully on {DEVICE}")
    
    print(f"Loading pre-fitted normalizers from {NORMALIZERS_PATH}...")
    import pickle
    with open(NORMALIZERS_PATH, 'rb') as f:
        normalizers_data = pickle.load(f)
    print(f"Normalizers loaded successfully ({len(normalizers_data)} configuration fields)")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    distance_km: List[float]
    altitude: List[float]
    elevation_gain: List[float]
    predicted_duration: List[float]
    accumulated_duration: List[float]
    predicted_heart_rate: List[float]
    predicted_cadence: List[float]
    total_distance_km: float
    total_predicted_time_min: float
    elevation_stats: Dict[str, float]
    

def parse_gpx_file(gpx_content: str) -> pd.DataFrame:
    """
    Parse GPX file and extract elevation profile at 5m intervals.
    
    Args:
        gpx_content: GPX file content as string
        
    Returns:
        DataFrame with columns: distance, altitude, elevation_diff, elevation_gain, elevation_loss
    """
    gpx = gpxpy.parse(gpx_content)
    
    # Extract track points
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'elevation': point.elevation
                })
    
    if not points:
        raise ValueError("No track points found in GPX file")
    
    df = pd.DataFrame(points)
    
    # Calculate distance between consecutive points using Haversine formula
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates in meters."""
        R = 6371000  # Earth radius in meters
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    # Calculate cumulative distance
    distances = [0]
    for i in range(1, len(df)):
        d = haversine_distance(
            df.iloc[i-1]['lat'], df.iloc[i-1]['lon'],
            df.iloc[i]['lat'], df.iloc[i]['lon']
        )
        distances.append(distances[-1] + d)
    
    df['distance'] = distances
    df['altitude'] = df['elevation']
    
    # Resample to 5m intervals
    max_distance = df['distance'].max()
    target_distances = np.arange(0, max_distance, 5)
    
    # Interpolate elevation at target distances
    resampled_altitude = np.interp(target_distances, df['distance'], df['altitude'])
    
    # Create resampled dataframe
    resampled_df = pd.DataFrame({
        'distance': target_distances,
        'altitude': resampled_altitude
    })
    
    # Calculate elevation changes
    resampled_df['elevation_diff'] = resampled_df['altitude'].diff().fillna(0)
    resampled_df['elevation_gain'] = resampled_df['elevation_diff'].apply(lambda x: max(0, x))
    resampled_df['elevation_loss'] = resampled_df['elevation_diff'].apply(lambda x: max(0, -x))
    
    return resampled_df


def create_cold_start_encoder(historical_data: pd.DataFrame, target_session_data: pd.DataFrame) -> pd.Series:
    """
    Create synthetic encoder for cold-start inference.
    
    Uses weighted average of historical first samples, with terrain from target session.
    
    Args:
        historical_data: DataFrame with all historical training sessions
        target_session_data: DataFrame with target session terrain data
        
    Returns:
        Series with synthetic encoder values
    """
    # For this API, we'll use simple averages from training data
    # In production, you'd load actual historical sessions
    
    # Create synthetic encoder with terrain from target session
    encoder = pd.Series({
        'distance': 0.0,
        'time_idx': 0,
        'altitude': target_session_data.iloc[0]['altitude'],
        'elevation_diff': 0.0,
        'elevation_gain': 0.0,
        'elevation_loss': 0.0,
        'duration_diff': 4.0,  # Average from training data
        'heartRate': 140.0,  # Average resting HR during activity
        'temperature': 20.0,  # Default temperature
        'cadence': 80.0,  # Average cadence
        'speed': 3.0,  # Average speed m/s
        'avg_heart_rate_so_far': 140.0,
        'duration': 0.0,
        'session_id_encoded': 0,
        'elevation_gain_of_last_100m': 0.0,
        'elevation_loss_of_last_100m': 0.0
    })
    
    return encoder


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "normalizers_loaded": normalizers_data is not None,
        "device": DEVICE
    }


class PresetInfo(BaseModel):
    """Info about a preset GPX file."""
    filename: str
    name: str


def prepare_session_data(session_data: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """
    Prepare session data for inference by adding required columns.
    
    Args:
        session_data: DataFrame with distance, altitude, elevation_diff, elevation_gain, elevation_loss
        session_id: Identifier for the session
        
    Returns:
        DataFrame ready for model inference
    """
    session_data = session_data.copy()
    
    # Add required columns
    session_data['time_idx'] = np.arange(len(session_data))
    session_data['session_id'] = session_id
    session_data['session_id_encoded'] = 0
    
    # Add synthetic physiological values (cold-start)
    session_data['duration_diff'] = 4.0
    session_data['heartRate'] = 140.0
    session_data['temperature'] = 20.0
    session_data['cadence'] = 80.0
    session_data['speed'] = 3.0
    session_data['avg_heart_rate_so_far'] = 140.0
    session_data['duration'] = session_data.index * 4.0
    
    # Calculate fatigue proxy features
    window_size = 20  # 100m / 5m = 20 steps
    session_data['elevation_gain_of_last_100m'] = session_data['elevation_gain'].rolling(
        window=window_size, min_periods=1
    ).sum()
    session_data['elevation_loss_of_last_100m'] = session_data['elevation_loss'].rolling(
        window=window_size, min_periods=1
    ).sum()
    
    return session_data


def calculate_weighted_first_sample(train_data: pd.DataFrame) -> Dict[str, float]:
    """Synthetic encoder function for cold-start inference."""
    return {
        'duration_diff': 4.0,
        'heartRate': 140.0,
        'temperature': 20.0,
        'cadence': 80.0,
        'speed': 3.0,
        'avg_heart_rate_so_far': 140.0,
        'duration': 0.0
    }


def run_inference(session_data: pd.DataFrame, session_id: str) -> PredictionResponse:
    """
    Run model inference on prepared session data.
    
    Args:
        session_data: Prepared DataFrame with all required columns
        session_id: Identifier for the session
        
    Returns:
        PredictionResponse with all predictions
    """
    from lib.model import evaluate_full_session_sequential
    
    print(f"Running cold-start inference...")
    
    result = evaluate_full_session_sequential(
        model=model,
        test_data=session_data.copy(),
        train_data=pd.DataFrame(),
        session_id=session_id,
        calculate_weighted_first_sample_fn=calculate_weighted_first_sample,
        max_pred_length=200,
        encoder_length=400,
        normalizers_data=normalizers_data,
        use_model_features=True,
        verbose=True
    )
    
    if result is None:
        raise ValueError("Model inference failed - session may be too short")
    
    # Extract predictions
    predictions = result['all_predictions']['duration_diff']
    heart_rates = result['all_predictions']['heartRate']
    cadences = result['all_predictions']['cadence']
    
    print(f"Predictions generated: {len(predictions)} steps")
    print(f"Duration range: {min(predictions):.2f} - {max(predictions):.2f}s")
    print(f"HR range: {min(heart_rates):.1f} - {max(heart_rates):.1f} bpm")
    print(f"Chunks processed: {result['chunks_processed']}")
    
    # Calculate accumulated duration
    accumulated_duration = np.cumsum(predictions) / 60  # Convert to minutes
    
    # Get the corresponding session data for predicted steps
    num_predicted = len(predictions)
    predicted_session_data = session_data.iloc[:num_predicted].copy()
    
    return PredictionResponse(
        distance_km=(predicted_session_data['distance'] / 1000).tolist(),
        altitude=predicted_session_data['altitude'].tolist(),
        elevation_gain=predicted_session_data['elevation_gain'].cumsum().tolist(),
        predicted_duration=predictions,
        accumulated_duration=accumulated_duration.tolist(),
        predicted_heart_rate=heart_rates,
        predicted_cadence=cadences,
        total_distance_km=float(predicted_session_data['distance'].max() / 1000),
        total_predicted_time_min=float(accumulated_duration[-1]),
        elevation_stats={
            'min_altitude': float(predicted_session_data['altitude'].min()),
            'max_altitude': float(predicted_session_data['altitude'].max()),
            'total_gain': float(predicted_session_data['elevation_gain'].sum()),
            'total_loss': float(predicted_session_data['elevation_loss'].sum())
        }
    )


@app.get("/presets", response_model=List[PresetInfo])
async def list_presets():
    """List available preset GPX files for demo."""
    presets = []
    
    if GPX_PRESETS_PATH.exists():
        for gpx_file in sorted(GPX_PRESETS_PATH.glob("*.gpx")):
            # Create display name from filename
            name = gpx_file.stem.replace("-", " ").replace("_", " ").title()
            presets.append(PresetInfo(filename=gpx_file.name, name=name))
    
    return presets


@app.get("/presets/{filename}", response_model=PredictionResponse)
async def predict_from_preset(filename: str):
    """
    Run prediction on a preset GPX file.
    
    Args:
        filename: Name of the preset GPX file
        
    Returns:
        Prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    gpx_path = GPX_PRESETS_PATH / filename
    
    if not gpx_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset not found: {filename}")
    
    if not filename.endswith('.gpx'):
        raise HTTPException(status_code=400, detail="File must be a GPX file")
    
    try:
        with open(gpx_path, 'r', encoding='utf-8') as f:
            gpx_content = f.read()
        
        session_data = parse_gpx_file(gpx_content)
        print(f"Parsed preset GPX '{filename}': {len(session_data)} points, {session_data['distance'].max()/1000:.2f} km")
        
        session_data = prepare_session_data(session_data, 'gpx_preset')
        return run_inference(session_data, 'gpx_preset')
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_race_time(file: UploadFile = File(...)):
    """
    Predict race time and metrics from GPX file using cold-start inference.
    
    Args:
        file: GPX file upload
        
    Returns:
        Prediction results with elevation profile and predicted metrics
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.gpx'):
        raise HTTPException(status_code=400, detail="File must be a GPX file")
    
    try:
        content = await file.read()
        gpx_content = content.decode('utf-8')
        
        session_data = parse_gpx_file(gpx_content)
        print(f"Parsed GPX: {len(session_data)} points, {session_data['distance'].max()/1000:.2f} km")
        
        session_data = prepare_session_data(session_data, 'gpx_upload')
        return run_inference(session_data, 'gpx_upload')
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
