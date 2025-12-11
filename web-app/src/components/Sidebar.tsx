'use client';

import { useState } from 'react';
import GpxUploader from './GpxUploader';
import { PredictionData } from '@/app/page';

interface SidebarProps {
  onPrediction: (data: PredictionData) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

export default function Sidebar({ onPrediction, loading, setLoading }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <>
      {/* Sidebar Toggle Button */}
      <button
        className={`sidebar-toggle ${isCollapsed ? 'collapsed' : ''}`}
        onClick={() => setIsCollapsed(!isCollapsed)}
        aria-label="Toggle sidebar"
      >
        <span className="icon">
          <i className={`fas fa-chevron-${isCollapsed ? 'right' : 'left'}`}></i>
        </span>
      </button>

      {/* Sidebar */}
      <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-content">
          <div className="sidebar-header">
            <h2 className="title is-5 has-text-white">
              <span className="icon-text">
                <span className="icon">
                  <i className="fas fa-upload"></i>
                </span>
                <span>Upload GPX</span>
              </span>
            </h2>
          </div>
          
          <div className="sidebar-body">
            <GpxUploader 
              onPrediction={onPrediction}
              loading={loading}
              setLoading={setLoading}
            />
          </div>
        </div>
      </aside>
    </>
  );
}
