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
    <div className="prediction-results">
      <h2 className="title is-3 has-text-centered mb-5">
        <span className="icon-text">
          <span className="icon has-text-info">
            <i className="fas fa-chart-line"></i>
          </span>
          <span>Race Predictions</span>
        </span>
      </h2>

      {/* Summary Stats */}
      <div className="columns is-multiline mb-5">
        <div className="column is-3">
          <div className="box has-background-info-light stat-card">
            <p className="heading">Total Distance</p>
            <p className="title is-4 has-text-info">
              {data.total_distance_km.toFixed(2)} km
            </p>
          </div>
        </div>
        
        <div className="column is-3">
          <div className="box has-background-primary-light stat-card">
            <p className="heading">Predicted Time</p>
            <p className="title is-4 has-text-primary">
              {formatTime(data.total_predicted_time_min)}
            </p>
          </div>
        </div>
        
        <div className="column is-3">
          <div className="box has-background-success-light stat-card">
            <p className="heading">Elevation Gain</p>
            <p className="title is-4 has-text-success">
              {Math.round(data.elevation_stats.total_gain)} m
            </p>
          </div>
        </div>
        
        <div className="column is-3">
          <div className="box has-background-link-light stat-card">
            <p className="heading">Avg Pace</p>
            <p className="title is-4 has-text-link">
              {(data.total_predicted_time_min / data.total_distance_km).toFixed(1)} min/km
            </p>
          </div>
        </div>
      </div>

      {/* Elevation Profile with Duration Overlay */}
      <div className="box chart-container mb-5">
        <h3 className="title is-5 mb-4">
          <span className="icon-text">
            <span className="icon has-text-success">
              <i className="fas fa-mountain"></i>
            </span>
            <span>Elevation Profile & Predicted Duration</span>
          </span>
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={sampledData}>
            <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
            <XAxis 
              dataKey="distance" 
              label={{ value: 'Distance (km)', position: 'insideBottom', offset: -5 }}
              className="chart-axis"
            />
            <YAxis 
              yAxisId="left"
              label={{ value: 'Altitude (m)', angle: -90, position: 'insideLeft' }}
              className="chart-axis"
            />
            <YAxis 
              yAxisId="right"
              orientation="right"
              label={{ value: 'Duration (min)', angle: 90, position: 'insideRight' }}
              className="chart-axis"
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'var(--tooltip-bg)', 
                border: '1px solid var(--tooltip-border)',
                borderRadius: '6px',
                color: 'var(--text-color)'
              }}
            />
            <Legend />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="altitude"
              fill="rgba(72, 199, 142, 0.3)"
              stroke="#48c78e"
              name="Altitude"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="duration"
              stroke="#3e8ed0"
              strokeWidth={2}
              dot={false}
              name="Accumulated Duration"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Heart Rate Prediction */}
      <div className="box chart-container mb-5">
        <h3 className="title is-5 mb-4">
          <span className="icon-text">
            <span className="icon has-text-danger">
              <i className="fas fa-heartbeat"></i>
            </span>
            <span>Predicted Heart Rate</span>
          </span>
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={sampledData}>
            <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
            <XAxis 
              dataKey="distance" 
              label={{ value: 'Distance (km)', position: 'insideBottom', offset: -5 }}
              className="chart-axis"
            />
            <YAxis 
              label={{ value: 'Heart Rate (bpm)', angle: -90, position: 'insideLeft' }}
              className="chart-axis"
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'var(--tooltip-bg)', 
                border: '1px solid var(--tooltip-border)',
                borderRadius: '6px',
                color: 'var(--text-color)'
              }}
            />
            <Line
              type="monotone"
              dataKey="heartRate"
              stroke="#f14668"
              strokeWidth={2}
              dot={false}
              name="Heart Rate"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Cadence Prediction */}
      <div className="box chart-container mb-5">
        <h3 className="title is-5 mb-4">
          <span className="icon-text">
            <span className="icon has-text-link">
              <i className="fas fa-running"></i>
            </span>
            <span>Predicted Cadence</span>
          </span>
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={sampledData}>
            <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
            <XAxis 
              dataKey="distance" 
              label={{ value: 'Distance (km)', position: 'insideBottom', offset: -5 }}
              className="chart-axis"
            />
            <YAxis 
              label={{ value: 'Cadence (spm)', angle: -90, position: 'insideLeft' }}
              className="chart-axis"
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'var(--tooltip-bg)', 
                border: '1px solid var(--tooltip-border)',
                borderRadius: '6px',
                color: 'var(--text-color)'
              }}
            />
            <Line
              type="monotone"
              dataKey="cadence"
              stroke="#3e8ed0"
              strokeWidth={2}
              dot={false}
              name="Cadence"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Elevation Stats */}
      <div className="box">
        <h3 className="title is-5 mb-4">
          <span className="icon-text">
            <span className="icon has-text-warning">
              <i className="fas fa-chart-area"></i>
            </span>
            <span>Elevation Statistics</span>
          </span>
        </h3>
        <div className="columns is-multiline">
          <div className="column is-6">
            <div className="content">
              <p><strong>Min Altitude:</strong> <span className="tag is-info is-medium">{Math.round(data.elevation_stats.min_altitude)}m</span></p>
            </div>
          </div>
          <div className="column is-6">
            <div className="content">
              <p><strong>Max Altitude:</strong> <span className="tag is-info is-medium">{Math.round(data.elevation_stats.max_altitude)}m</span></p>
            </div>
          </div>
          <div className="column is-6">
            <div className="content">
              <p><strong>Total Gain:</strong> <span className="tag is-success is-medium">{Math.round(data.elevation_stats.total_gain)}m</span></p>
            </div>
          </div>
          <div className="column is-6">
            <div className="content">
              <p><strong>Total Loss:</strong> <span className="tag is-danger is-medium">{Math.round(data.elevation_stats.total_loss)}m</span></p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
