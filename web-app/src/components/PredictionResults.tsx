'use client';

import { PredictionData } from '@/app/page';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from 'recharts';

interface PredictionResultsProps {
  data: PredictionData;
}

export default function PredictionResults({ data }: PredictionResultsProps) {
  // Prepare data for charts
  const chartData = data.distance_km.map((dist, idx) => ({
    distance: parseFloat(dist.toFixed(2)),
    altitude: data.altitude[idx],
    duration: parseFloat(data.accumulated_duration[idx].toFixed(2)),
    heartRate: Math.round(data.predicted_heart_rate[idx]),
    cadence: Math.round(data.predicted_cadence[idx]),
  }));

  // Sample data for visualization (every 10th point to reduce rendering load)
  const sampledData = chartData.filter((_, idx) => idx % 10 === 0);

  const formatTime = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = Math.floor(minutes % 60);
    return `${hours}h ${mins}m`;
  };

  return (
    <div style={styles.container}>
      {/* Summary Stats */}
      <div style={styles.statsGrid}>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Total Distance</div>
          <div style={styles.statValue}>
            {data.total_distance_km.toFixed(2)} km
          </div>
        </div>
        
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Predicted Time</div>
          <div style={styles.statValue}>
            {formatTime(data.total_predicted_time_min)}
          </div>
        </div>
        
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Elevation Gain</div>
          <div style={styles.statValue}>
            {Math.round(data.elevation_stats.total_gain)} m
          </div>
        </div>
        
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Avg Pace</div>
          <div style={styles.statValue}>
            {(data.total_predicted_time_min / data.total_distance_km).toFixed(1)} min/km
          </div>
        </div>
      </div>

      {/* Elevation Profile with Duration Overlay */}
      <div style={styles.chartContainer}>
        <h3 style={styles.chartTitle}>📈 Elevation Profile & Predicted Duration</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={sampledData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="distance" 
              label={{ value: 'Distance (km)', position: 'insideBottom', offset: -5, fill: 'white' }}
              stroke="white"
            />
            <YAxis 
              yAxisId="left"
              label={{ value: 'Altitude (m)', angle: -90, position: 'insideLeft', fill: 'white' }}
              stroke="white"
            />
            <YAxis 
              yAxisId="right"
              orientation="right"
              label={{ value: 'Duration (min)', angle: 90, position: 'insideRight', fill: 'white' }}
              stroke="white"
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(0, 0, 0, 0.8)', 
                border: '1px solid rgba(255, 255, 255, 0.3)',
                borderRadius: '8px',
                color: 'white'
              }}
            />
            <Legend wrapperStyle={{ color: 'white' }} />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="altitude"
              fill="rgba(34, 197, 94, 0.3)"
              stroke="rgb(34, 197, 94)"
              name="Altitude"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="duration"
              stroke="rgb(239, 68, 68)"
              strokeWidth={2}
              dot={false}
              name="Accumulated Duration"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Heart Rate Prediction */}
      <div style={styles.chartContainer}>
        <h3 style={styles.chartTitle}>❤️ Predicted Heart Rate</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={sampledData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="distance" 
              label={{ value: 'Distance (km)', position: 'insideBottom', offset: -5, fill: 'white' }}
              stroke="white"
            />
            <YAxis 
              label={{ value: 'Heart Rate (bpm)', angle: -90, position: 'insideLeft', fill: 'white' }}
              stroke="white"
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(0, 0, 0, 0.8)', 
                border: '1px solid rgba(255, 255, 255, 0.3)',
                borderRadius: '8px',
                color: 'white'
              }}
            />
            <Line
              type="monotone"
              dataKey="heartRate"
              stroke="rgb(239, 68, 68)"
              strokeWidth={2}
              dot={false}
              name="Heart Rate"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Cadence Prediction */}
      <div style={styles.chartContainer}>
        <h3 style={styles.chartTitle}>🏃 Predicted Cadence</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={sampledData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="distance" 
              label={{ value: 'Distance (km)', position: 'insideBottom', offset: -5, fill: 'white' }}
              stroke="white"
            />
            <YAxis 
              label={{ value: 'Cadence (spm)', angle: -90, position: 'insideLeft', fill: 'white' }}
              stroke="white"
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(0, 0, 0, 0.8)', 
                border: '1px solid rgba(255, 255, 255, 0.3)',
                borderRadius: '8px',
                color: 'white'
              }}
            />
            <Line
              type="monotone"
              dataKey="cadence"
              stroke="rgb(59, 130, 246)"
              strokeWidth={2}
              dot={false}
              name="Cadence"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Elevation Stats */}
      <div style={styles.elevationStats}>
        <h3 style={styles.chartTitle}>⛰️ Elevation Statistics</h3>
        <div style={styles.elevStatsGrid}>
          <div>
            <strong>Min Altitude:</strong> {Math.round(data.elevation_stats.min_altitude)}m
          </div>
          <div>
            <strong>Max Altitude:</strong> {Math.round(data.elevation_stats.max_altitude)}m
          </div>
          <div>
            <strong>Total Gain:</strong> {Math.round(data.elevation_stats.total_gain)}m
          </div>
          <div>
            <strong>Total Loss:</strong> {Math.round(data.elevation_stats.total_loss)}m
          </div>
        </div>
      </div>
    </div>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    marginTop: '2rem',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '1rem',
    marginBottom: '2rem',
  },
  statCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
    borderRadius: '12px',
    padding: '1.5rem',
    textAlign: 'center' as const,
    border: '1px solid rgba(255, 255, 255, 0.2)',
  },
  statLabel: {
    fontSize: '0.9rem',
    color: 'rgba(255, 255, 255, 0.7)',
    marginBottom: '0.5rem',
  },
  statValue: {
    fontSize: '1.8rem',
    fontWeight: 'bold' as const,
    color: 'white',
  },
  chartContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
    borderRadius: '12px',
    padding: '1.5rem',
    marginBottom: '1.5rem',
    border: '1px solid rgba(255, 255, 255, 0.2)',
  },
  chartTitle: {
    color: 'white',
    marginBottom: '1rem',
    fontSize: '1.2rem',
  },
  elevationStats: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
    borderRadius: '12px',
    padding: '1.5rem',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    color: 'white',
  },
  elevStatsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '1rem',
    marginTop: '1rem',
  },
};
