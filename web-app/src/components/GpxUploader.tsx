'use client';

import { useRef, useState, useEffect } from 'react';
import axios from 'axios';
import { PredictionData } from '@/app/page';

interface GpxUploaderProps {
  onPrediction: (data: PredictionData) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

interface Preset {
  filename: string;
  name: string;
}

export default function GpxUploader({ onPrediction, loading, setLoading }: GpxUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [loadingPresets, setLoadingPresets] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load presets on mount
  useEffect(() => {
    const fetchPresets = async () => {
      try {
        const response = await axios.get<Preset[]>('/api/presets');
        setPresets(response.data);
      } catch (err) {
        console.error('Failed to load presets:', err);
      } finally {
        setLoadingPresets(false);
      }
    };
    fetchPresets();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.gpx')) {
        setError('Please select a GPX file');
        return;
      }
      setSelectedFile(file);
      setSelectedPreset(''); // Clear preset selection
      setError(null);
    }
  };

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedPreset(e.target.value);
    setSelectedFile(null); // Clear file selection
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile && !selectedPreset) {
      setError('Please select a file or a preset route');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let response;
      
      if (selectedPreset) {
        // Load from preset
        response = await axios.get<PredictionData>(`/api/presets/${selectedPreset}`);
      } else if (selectedFile) {
        // Upload file
        const formData = new FormData();
        formData.append('file', selectedFile);
        response = await axios.post<PredictionData>(
          '/api/predict',
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          }
        );
      }

      if (response) {
        onPrediction(response.data);
      }
    } catch (err: any) {
      console.error('Upload error:', err);
      setError(
        err.response?.data?.detail || 
        'Failed to get predictions. Please make sure the API server is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.gpx')) {
      setSelectedFile(file);
      setError(null);
    } else {
      setError('Please drop a GPX file');
    }
  };

  return (
    <div className="gpx-uploader">
      {/* Loading Overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-content">
            <div className="loading-spinner">
              <i className="fas fa-spinner fa-spin fa-3x"></i>
            </div>
            <p className="loading-text mt-4">Analyzing your route...</p>
            <p className="loading-subtext">This may take a moment for longer routes</p>
            <progress className="progress is-info mt-4" max="100">Processing</progress>
          </div>
        </div>
      )}

      {/* Preset Selector */}
      <div className="field mb-4">
        <label className="label has-text-grey-light is-size-7">
          <span className="icon-text">
            <span className="icon">
              <i className="fas fa-route"></i>
            </span>
            <span>Demo Routes</span>
          </span>
        </label>
        <div className="control has-icons-left">
          <div className={`select is-fullwidth ${loadingPresets ? 'is-loading' : ''}`}>
            <select 
              value={selectedPreset} 
              onChange={handlePresetChange}
              disabled={loading || loadingPresets}
            >
              <option value="">Select a preset route...</option>
              {presets.map((preset) => (
                <option key={preset.filename} value={preset.filename}>
                  {preset.name}
                </option>
              ))}
            </select>
          </div>
          <span className="icon is-left">
            <i className="fas fa-map-marked-alt"></i>
          </span>
        </div>
      </div>

      <div className="divider-text mb-4">
        <span>or upload your own</span>
      </div>

      <div
        className={`file has-name is-fullwidth is-boxed is-info dropzone ${selectedPreset ? 'is-dimmed' : ''}`}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <label className="file-label">
          <input
            ref={fileInputRef}
            className="file-input"
            type="file"
            accept=".gpx"
            onChange={handleFileChange}
            disabled={loading}
          />
          <span className="file-cta">
            <span className="file-icon">
              <i className="fas fa-cloud-upload-alt"></i>
            </span>
            <span className="file-label">
              {selectedFile ? 'Change file' : 'Choose a file…'}
            </span>
          </span>
          {selectedFile && (
            <span className="file-name">
              {selectedFile.name}
            </span>
          )}
        </label>
      </div>

      <div className="content has-text-centered mt-3 mb-4">
        <p className="is-size-7 has-text-grey-light">
          Drag and drop your GPX file or click to browse
        </p>
      </div>

      {error && (
        <article className="message is-danger mb-4">
          <div className="message-body">
            <span className="icon">
              <i className="fas fa-exclamation-triangle"></i>
            </span>
            <span>{error}</span>
          </div>
        </article>
      )}

      <button
        className={`button is-info is-fullwidth ${loading ? 'is-loading' : ''}`}
        onClick={handleUpload}
        disabled={(!selectedFile && !selectedPreset) || loading}
      >
        <span className="icon">
          <i className="fas fa-chart-line"></i>
        </span>
        <span>{loading ? 'Analyzing route...' : 'Get Predictions'}</span>
      </button>
    </div>
  );
}
