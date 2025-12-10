'use client';

import { useState } from 'react';
import GpxUploader from '@/components/GpxUploader';
import PredictionResults from '@/components/PredictionResults';

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
    <main style={styles.main}>
      <div style={styles.container}>
        <header style={styles.header}>
          <h1 style={styles.title}>🏃‍♂️ Trail Running Race Predictor</h1>
          <p style={styles.subtitle}>
            AI-powered race time predictions using Temporal Fusion Transformers
          </p>
        </header>

        <GpxUploader 
          onPrediction={handlePrediction} 
          loading={loading}
          setLoading={setLoading}
        />

        {prediction && <PredictionResults data={prediction} />}
      </div>
    </main>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  main: {
    minHeight: '100vh',
    padding: '2rem',
  },
  container: {
    maxWidth: '1400px',
    margin: '0 auto',
  },
  header: {
    textAlign: 'center' as const,
    marginBottom: '3rem',
    color: 'white',
  },
  title: {
    fontSize: '3rem',
    fontWeight: 'bold',
    marginBottom: '1rem',
    textShadow: '2px 2px 4px rgba(0,0,0,0.3)',
  },
  subtitle: {
    fontSize: '1.2rem',
    opacity: 0.9,
  },
};
