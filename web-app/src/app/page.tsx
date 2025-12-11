'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import PredictionResults from '@/components/PredictionResults';
import Navbar from '@/components/Navbar';

export interface PredictionData {
  distance_km: number[];
  altitude: number[];
  elevation_gain: number[];
  predicted_duration: number[];
  accumulated_duration: number[];
  predicted_heart_rate: number[];
  predicted_cadence: number[];
  total_distance_km: number;
  total_predicted_time_min: number;
  elevation_stats: {
    min_altitude: number;
    max_altitude: number;
    total_gain: number;
    total_loss: number;
  };
}

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);

  const handlePrediction = (data: PredictionData) => {
    setPrediction(data);
  };

  return (
    <div className="app-container">
      <Navbar />

      {/* Main Layout */}
      <div className="main-layout">
        <Sidebar 
          onPrediction={handlePrediction}
          loading={loading}
          setLoading={setLoading}
        />

        <main className="main-content">
          {prediction ? (
            <PredictionResults data={prediction} />
          ) : (
            <div className="hero is-medium">
              <div className="hero-body has-text-centered">
                <div className="container">
                  <span className="icon is-large has-text-info mb-4">
                    <i className="fas fa-route fa-4x"></i>
                  </span>
                  <h2 className="title is-3">Welcome to Trail Running Predictor</h2>
                  <p className="subtitle is-5">
                    Upload a GPX file from the sidebar to get AI-powered predictions for your race
                  </p>
                  <div className="content has-text-left mt-6" style={{ maxWidth: '600px', margin: '0 auto' }}>
                    <h3 className="title is-5">What you'll get:</h3>
                    <ul>
                      <li>📊 Predicted race duration and pace</li>
                      <li>❤️ Heart rate predictions along the route</li>
                      <li>🏃 Cadence estimates for different terrain</li>
                      <li>⛰️ Detailed elevation profile analysis</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
