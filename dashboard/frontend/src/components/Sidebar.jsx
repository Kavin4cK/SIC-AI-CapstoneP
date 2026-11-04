import React from 'react';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: '📊' },
  { id: 'review', label: 'Review queue', icon: '🔍', showBadge: true },
  { id: 'labelled', label: 'Labelled cases', icon: '🏷️' },
  { id: 'model', label: 'Model and threshold', icon: '⚙️' },
  { id: 'retraining', label: 'Retraining', icon: '🔄' },
];

export default function Sidebar({ activeTab, onSelectTab, pendingCount = 0 }) {
  return (
    <aside className="sidebar" aria-label="Main Navigation">
      <div className="sidebar-header">
        <span className="brand-badge">SIC Fraud Guard</span>
        <h2 className="brand-title">Fraud Operator</h2>
        <p className="brand-subtitle">CatBoost Benchmark · t = 0.144</p>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              className={`nav-item-btn ${isActive ? 'active' : ''}`}
              onClick={() => onSelectTab(item.id)}
              aria-current={isActive ? 'page' : undefined}
            >
              <div className="nav-item-content">
                <span className="nav-icon" aria-hidden="true">{item.icon}</span>
                <span>{item.label}</span>
              </div>
              {item.showBadge && pendingCount > 0 && (
                <span className="nav-badge" aria-label={`${pendingCount} pending reviews`}>
                  {pendingCount}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="operator-badge">
          <span className="status-dot-live" title="API Online"></span>
          <span>Operator: <strong>Active Session</strong></span>
        </div>
      </div>
    </aside>
  );
}

