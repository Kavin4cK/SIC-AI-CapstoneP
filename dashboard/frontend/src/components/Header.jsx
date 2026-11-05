import React from 'react';

const TAB_TITLES = {
  overview: {
    title: 'Operations Overview',
    subtitle: 'Cost-sensitive fraud monitoring, live triage statistics, and incoming alerts',
  },
  'dataset-upload': {
    title: 'Dataset Upload & Fraud Detection',
    subtitle: 'Upload transaction CSV or evaluate the ULB benchmark dataset with the trained CatBoost model',
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

export default function Header({
  activeTab,
  onRefresh,
  isRefreshing,
  activeDatasetId = 'default',
  onSelectDataset,
  availableDatasets = [],
}) {
  const current = TAB_TITLES[activeTab] || { title: 'Dashboard', subtitle: '' };

  // Prepare dropdown options ensuring activeDatasetId is present
  const optionsMap = new Map();
  optionsMap.set('default', {
    dataset_id: 'default',
    label: 'Default Benchmark (sample_input.csv)',
  });

  availableDatasets.forEach((ds) => {
    const label = ds.filename
      ? `${ds.filename} (${ds.total_rows != null ? `${ds.total_rows.toLocaleString()} rows` : ds.dataset_id})`
      : ds.dataset_id;
    optionsMap.set(ds.dataset_id, {
      dataset_id: ds.dataset_id,
      label,
    });
  });

  if (activeDatasetId && !optionsMap.has(activeDatasetId)) {
    optionsMap.set(activeDatasetId, {
      dataset_id: activeDatasetId,
      label: `Dataset: ${activeDatasetId}`,
    });
  }

  const datasetList = Array.from(optionsMap.values());

  return (
    <header className="top-header">
      <div className="header-title-wrap">
        <h1>{current.title}</h1>
        <p className="header-subtitle">{current.subtitle}</p>
      </div>
      <div className="header-actions" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* Active Dataset Context Selector */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: '#F8FAFC',
            padding: '4px 10px',
            borderRadius: '6px',
            border: '1px solid #E2E8F0',
          }}
        >
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#475569' }}>
            Active Dataset:
          </span>
          <select
            id="active-dataset-select"
            value={activeDatasetId}
            onChange={(e) => onSelectDataset && onSelectDataset(e.target.value)}
            style={{
              padding: '4px 8px',
              fontSize: '12px',
              fontWeight: 600,
              color: '#0F3D3E',
              backgroundColor: '#FFFFFF',
              border: '1px solid #CBD5E1',
              borderRadius: '4px',
              cursor: 'pointer',
              outline: 'none',
              maxWidth: '280px',
            }}
            aria-label="Active Dataset Context"
          >
            {datasetList.map((item) => (
              <option key={item.dataset_id} value={item.dataset_id}>
                {item.label}
              </option>
            ))}
          </select>
        </div>

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
