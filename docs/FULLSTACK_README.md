# Trail Running Race Predictor - Full Stack App

Complete AI-powered trail running race time prediction system with FastAPI backend and Next.js frontend.

## 🎯 Overview

This application uses a Temporal Fusion Transformer (TFT) deep learning model to predict race completion times, heart rate, and cadence from GPX route files. It provides cold-start predictions without requiring any prior race data from the athlete.

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Next.js Web   │─────▶│   FastAPI        │─────▶│   TFT Model     │
│   Frontend      │◀─────│   Backend        │◀─────│   (PyTorch)     │
│   (Port 3000)   │      │   (Port 8000)    │      │   V2 Checkpoint │
└─────────────────┘      └──────────────────┘      └─────────────────┘
        │                        │
        │                        │
        ▼                        ▼
   GPX Upload              GPX Parsing
   Visualization           Cold-start Inference
```

## 📦 Components

### 1. FastAPI Backend (`/api`)
- Endpoint: `POST /predict`
- GPX file parsing and resampling to 5m intervals
- Cold-start inference using V2 model
- Returns predictions: duration, heart rate, cadence

### 2. Next.js Frontend (`/web-app`)
- Drag-and-drop GPX file upload
- Interactive charts (elevation, duration, HR, cadence)
- Real-time prediction visualization
- Responsive design with gradient UI

### 3. TFT Model
- Pre-trained on 106 Polar Vantage V sessions
- Asymmetric SMAPE loss (α=0.51)
- Multi-target forecasting
- Cold-start capable with synthetic encoder

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (optional, CPU works too)

### 1. Start the API Server

```bash
cd /home/eaguayo/DeepLearning/ProyectoFinal

# Install Python dependencies
pip install -r api/requirements.txt

# Start FastAPI server
python api/main.py
```

API will be available at: http://localhost:8000

### 2. Start the Web App

```bash
# Open new terminal
cd /home/eaguayo/DeepLearning/ProyectoFinal/web-app

# Install dependencies
npm install

# Start development server
npm run dev
```

Web app will be available at: http://localhost:3000

### 3. Test with Sample GPX Files

Use any GPX file from the `/gpx` directory:
- `20km.gpx` - 20km route
- `30kmchota2025-oficial.gpx` - 30km race route
- `80km.gpx` - Ultra distance route

## 🎨 Features

### Backend Features
- ✅ GPX file parsing with elevation extraction
- ✅ Haversine distance calculation
- ✅ 5m interval resampling
- ✅ Cold-start synthetic encoder
- ✅ Sequential chunk prediction
- ✅ Multi-target forecasting (duration, HR, cadence)
- ✅ CORS enabled for local development

### Frontend Features
- ✅ Drag-and-drop file upload
- ✅ Real-time prediction loading states
- ✅ Summary statistics cards
- ✅ Elevation profile with duration overlay
- ✅ Heart rate prediction chart
- ✅ Cadence prediction chart
- ✅ Elevation statistics breakdown
- ✅ Responsive gradient UI
- ✅ Error handling and validation

## 📊 Sample Output

```json
{
  "total_distance_km": 20.5,
  "total_predicted_time_min": 165.3,
  "elevation_stats": {
    "min_altitude": 2980,
    "max_altitude": 3911,
    "total_gain": 1200,
    "total_loss": 800
  }
}
```

## 🧪 Testing

### Test API Endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@gpx/20km.gpx"
```

### Health Check
```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

## 📁 Project Structure

```
ProyectoFinal/
├── api/                          # FastAPI backend
│   ├── main.py                   # API server
│   ├── requirements.txt          # Python dependencies
│   └── README.md
├── web-app/                      # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Main page
│   │   │   ├── layout.tsx       # Root layout
│   │   │   └── globals.css      # Global styles
│   │   └── components/
│   │       ├── GpxUploader.tsx  # Upload component
│   │       └── PredictionResults.tsx  # Visualization
│   ├── package.json
│   └── README.md
├── gpx/                          # Sample GPX files
├── checkpoints_v2/               # Model checkpoint
└── lib/                          # Model code
```

## 🔧 Configuration

### API Configuration
Edit `api/main.py`:
- `MODEL_PATH`: Path to V2 checkpoint
- `DEVICE`: "cuda" or "cpu"
- CORS origins for frontend

### Web App Configuration
Create `web-app/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🐛 Troubleshooting

### API Issues
- **Model not loading**: Check `checkpoints_v2/best-checkpoint_v2-epoch=27-val_loss=0.12-v1.ckpt` exists
- **CUDA errors**: Set `DEVICE = "cpu"` in `api/main.py`
- **GPX parsing errors**: Ensure GPX file has valid track points

### Web App Issues
- **CORS errors**: Check API is running and CORS is enabled
- **Connection refused**: Ensure API is running on port 8000
- **Chart not rendering**: Check browser console for errors

## 📝 License

This project is part of an academic research project for trail running performance prediction using Temporal Fusion Transformers.

## 🙏 Acknowledgments

- V2 Model trained on 106 Polar Vantage V sessions
- Temporal Fusion Transformer architecture by Lim et al.
- PyTorch Lightning framework
- FastAPI and Next.js frameworks
