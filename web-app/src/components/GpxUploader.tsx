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
    <div style={styles.container}>
      <div
        style={styles.dropzone}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".gpx"
          onChange={handleFileChange}
          style={styles.hiddenInput}
        />
        
        <div style={styles.dropzoneContent}>
          <svg
            style={styles.icon}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          
          {selectedFile ? (
            <p style={styles.fileName}>📄 {selectedFile.name}</p>
          ) : (
            <>
              <p style={styles.dropText}>
                Drop your GPX file here or click to browse
              </p>
              <p style={styles.dropSubtext}>
                Upload a race route to get AI-powered time predictions
              </p>
            </>
          )}
        </div>
      </div>

      {error && (
        <div style={styles.error}>
          ⚠️ {error}
        </div>
      )}

      <button
        onClick={handleUpload}
        disabled={!selectedFile || loading}
        style={{
          ...styles.button,
          ...((!selectedFile || loading) && styles.buttonDisabled),
        }}
      >
        {loading ? (
          <>
            <div style={styles.spinner} />
            Analyzing route...
          </>
        ) : (
          'Get Predictions'
        )}
      </button>
    </div>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    marginBottom: '2rem',
  },
  dropzone: {
    border: '3px dashed rgba(255, 255, 255, 0.4)',
    borderRadius: '12px',
    padding: '3rem',
    textAlign: 'center' as const,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    backdropFilter: 'blur(10px)',
  },
  hiddenInput: {
    display: 'none',
  },
  dropzoneContent: {
    color: 'white',
  },
  icon: {
    width: '64px',
    height: '64px',
    margin: '0 auto 1rem',
    opacity: 0.8,
  },
  dropText: {
    fontSize: '1.2rem',
    fontWeight: 'bold' as const,
    marginBottom: '0.5rem',
  },
  dropSubtext: {
    fontSize: '0.9rem',
    opacity: 0.7,
  },
  fileName: {
    fontSize: '1.1rem',
    fontWeight: 'bold' as const,
  },
  error: {
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    border: '2px solid rgba(239, 68, 68, 0.5)',
    borderRadius: '8px',
    padding: '1rem',
    marginTop: '1rem',
    color: 'white',
    fontSize: '0.9rem',
  },
  button: {
    width: '100%',
    padding: '1rem 2rem',
    marginTop: '1rem',
    fontSize: '1.1rem',
    fontWeight: 'bold' as const,
    color: 'white',
    backgroundColor: 'rgba(16, 185, 129, 0.8)',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  spinner: {
    width: '20px',
    height: '20px',
    border: '3px solid rgba(255, 255, 255, 0.3)',
    borderTop: '3px solid white',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
};
