'use client';

import { useTheme } from '@/context/ThemeContext';

export default function Navbar() {
  const { theme, toggleTheme } = useTheme();

  return (
    <nav className="navbar is-fixed-top" role="navigation" aria-label="main navigation">
      <div className="navbar-brand">
        <div className="navbar-item">
          <span className="icon-text">
            <span className="icon has-text-info">
              <i className="fas fa-mountain"></i>
            </span>
            <span className="title is-4 has-text-white ml-2">Trail Running Race Predictor</span>
          </span>
        </div>
      </div>

      <div className="navbar-menu">
        <div className="navbar-end">
          <div className="navbar-item">
            <p className="subtitle is-6 has-text-grey-light mr-4">
              AI-powered predictions using Temporal Fusion Transformers
            </p>
            <button 
              className="button is-rounded theme-toggle"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              <span className="icon">
                {theme === 'light' ? (
                  <i className="fas fa-moon"></i>
                ) : (
                  <i className="fas fa-sun"></i>
                )}
              </span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
