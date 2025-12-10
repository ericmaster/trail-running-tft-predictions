# Trail Running TFT API

FastAPI server for cold-start race predictions using GPX files.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
cd /home/eaguayo/DeepLearning/ProyectoFinal
python api/main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### GET /
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

### POST /predict
Upload a GPX file and get race predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (GPX file)

**Response:**
```json
{
  "distance_km": [0.0, 0.005, 0.01, ...],
  "altitude": [2500, 2501, 2502, ...],
  "elevation_gain": [0, 1, 3, ...],
  "predicted_duration": [4.0, 4.1, 4.2, ...],
  "accumulated_duration": [0.067, 0.135, 0.205, ...],
  "predicted_heart_rate": [140, 142, 145, ...],
  "predicted_cadence": [80, 81, 82, ...],
  "total_distance_km": 20.5,
  "total_predicted_time_min": 165.3,
  "elevation_stats": {
    "min_altitude": 2500,
    "max_altitude": 3500,
    "total_gain": 1200,
    "total_loss": 800
  }
}
```

## Testing

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@gpx/20km.gpx"
```
