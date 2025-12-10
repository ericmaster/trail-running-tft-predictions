#!/bin/bash

# Trail Running Predictor - Startup Script
# This script starts both the API server and the web app

echo "🏃 Starting Trail Running Race Predictor..."
echo ""

# Check if we're in the right directory
if [ ! -d "api" ] || [ ! -d "web-app" ]; then
    echo "❌ Error: Please run this script from the ProyectoFinal directory"
    exit 1
fi

# Check if Python dependencies are installed
echo "📦 Checking Python dependencies..."
if ! python3 -c "import fastapi, uvicorn, gpxpy" 2>/dev/null; then
    echo "⚠️  Installing Python dependencies..."
    pip install -r api/requirements.txt
fi

# Check if Node dependencies are installed
echo "📦 Checking Node.js dependencies..."
if [ ! -d "web-app/node_modules" ]; then
    echo "⚠️  Installing Node.js dependencies..."
    cd web-app
    npm install
    cd ..
fi

# Start API server in background
echo ""
echo "🚀 Starting API server on port 8000..."
python3 api/main.py &
API_PID=$!

# Wait for API to be ready
sleep 3

# Check if API is running
if ! curl -s http://localhost:8000/ > /dev/null; then
    echo "❌ Failed to start API server"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo "✅ API server running at http://localhost:8000"

# Start web app
echo ""
echo "🚀 Starting web app on port 3000..."
cd web-app
npm run dev &
WEB_PID=$!
cd ..

# Wait for web app to be ready
sleep 5

echo ""
echo "✅ Web app running at http://localhost:3000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Both services are running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 API:     http://localhost:8000"
echo "📍 Web App: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID 2>/dev/null
    kill $WEB_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Wait for user to stop
wait
