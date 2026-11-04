import React from 'react';

const TAB_TITLES = {
  overview: {
    title: 'Operations Overview',
    subtitle: 'Cost-sensitive fraud monitoring, live triage statistics, and incoming alerts',
  },
  review: {
    title: 'Review Queue',
    subtitle: 'Evaluate suspicious transactions exceeding the 0.144 decision threshold',
  },
  labelled: {
    title: 'Labelled Cases',
    subtitle: 'Historical operator review decisions and exportable feedback logs',
  },
  model: {
    title: 'Model and Threshold Specification',
    subtitle: 'Frozen CatBoost production parameters, test evaluation benchmarks, and confusion matrix',
  },
  retraining: {
    title: 'Model Retraining Pipeline',
    subtitle: 'Continuous feedback loop handoff status and retraining schedule',
  },
};

export default function Header({ activeTab, onRefresh, isRefreshing }) {
  const current = TAB_TITLES[activeTab] || { title: 'Dashboard', subtitle: '' };

  return (
    <header className="top-header">
      <div className="header-title-wrap">
        <h1>{current.title}</h1>
        <p className="header-subtitle">{current.subtitle}</p>
      </div>
      <div className="header-actions">
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onRefresh}
          disabled={isRefreshing}
          aria-label="Refresh data from server"
        >
          {isRefreshing ? 'Refreshing...' : '🔄 Refresh Data'}
        </button>
      </div>
    </header>
  );
}

