'use client';

import { useState, useEffect } from 'react';
import GpxUploader from './GpxUploader';
import { PredictionData } from '@/app/page';

interface SidebarProps {
  onPrediction: (data: PredictionData) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

export default function Sidebar({ onPrediction, loading, setLoading }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(true); // Start collapsed to prevent flash
  const [isMobile, setIsMobile] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const checkMobile = () => {
      const mobile = window.innerWidth <= 768;
      setIsMobile(mobile);
      // Only expand on desktop after mount
      if (!mobile && !mounted) {
        setIsCollapsed(false);
      }
    };
    
    checkMobile();
    // Expand on desktop after initial check
    if (window.innerWidth > 768) {
      setIsCollapsed(false);
    }
    
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const handleToggle = () => {
    setIsCollapsed(!isCollapsed);
  };

  // Don't render sidebar content until mounted to prevent hydration mismatch
  if (!mounted) {
    return null;
  }

  return (
    <>
      {/* Mobile Overlay */}
      {isMobile && !isCollapsed && (
        <div 
          className="sidebar-overlay" 
          onClick={() => setIsCollapsed(true)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Toggle Button */}
      <button
        className={`sidebar-toggle ${isCollapsed ? 'collapsed' : ''}`}
        onClick={handleToggle}
        aria-label="Toggle sidebar"
      >
        <span className="icon">
          {isCollapsed ? (
            <i className="fas fa-bars"></i>
          ) : (
            <i className="fas fa-times"></i>
          )}
        </span>
      </button>

      {/* Sidebar */}
      <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''} ${isMobile ? 'mobile' : ''}`}>
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
