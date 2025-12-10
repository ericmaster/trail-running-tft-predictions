# Trail Running Race Predictor - Web App

A Next.js React application for visualizing AI-powered race time predictions from GPX files.

## Features

- 📤 Drag-and-drop GPX file upload
- 📊 Interactive elevation profile visualization
- ⏱️ Predicted race time and accumulated duration
- ❤️ Predicted heart rate throughout the race
- 🏃 Predicted cadence visualization
- 📈 Real-time charts using Recharts

## Installation

```bash
cd web-app
npm install
```

## Configuration

Create a `.env.local` file (optional):

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the App

Development mode:
```bash
npm run dev
```

Production build:
```bash
npm run build
npm start
```

The app will be available at http://localhost:3000

## Usage

1. Make sure the API server is running at http://localhost:8000
2. Open the web app at http://localhost:3000
3. Drag and drop a GPX file or click to browse
4. Click "Get Predictions" to analyze the route
5. View the predicted race time, elevation profile, heart rate, and cadence

## GPX Files

Test GPX files are available in the `/gpx` directory:
- `15kmchota2025-oficial.gpx`
- `20km.gpx`
- `30kmchota2025-oficial.gpx`
- `80km.gpx`

## Technology Stack

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Recharts** - Chart visualization library
- **Axios** - HTTP client
- **CSS-in-JS** - Inline styling

## Architecture

```
web-app/
├── src/
│   ├── app/
│   │   ├── layout.tsx       # Root layout
│   │   ├── page.tsx         # Main page component
│   │   └── globals.css      # Global styles
│   └── components/
│       ├── GpxUploader.tsx      # File upload component
│       └── PredictionResults.tsx # Results visualization
├── package.json
├── tsconfig.json
└── next.config.js
```

## API Integration

The app communicates with the FastAPI backend at `http://localhost:8000/predict`:

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: GPX file

**Response:**
```json
{
  "distance_km": [...],
  "altitude": [...],
  "predicted_duration": [...],
  "accumulated_duration": [...],
  "predicted_heart_rate": [...],
  "predicted_cadence": [...],
  "total_distance_km": 20.5,
  "total_predicted_time_min": 165.3,
  "elevation_stats": {...}
}
```
