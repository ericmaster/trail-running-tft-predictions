'use client';

import { useRef, useState } from 'react';
import axios from 'axios';
import { PredictionData } from '@/app/page';

interface GpxUploaderProps {
  onPrediction: (data: PredictionData) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

export default function GpxUploader({ onPrediction, loading, setLoading }: GpxUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.gpx')) {
        setError('Please select a GPX file');
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post<PredictionData>(
        '/api/predict',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      onPrediction(response.data);
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
      <div
        className="file has-name is-fullwidth is-boxed is-info dropzone"
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
        disabled={!selectedFile || loading}
      >
        <span className="icon">
          <i className="fas fa-chart-line"></i>
        </span>
        <span>{loading ? 'Analyzing route...' : 'Get Predictions'}</span>
      </button>
    </div>
  );
}
