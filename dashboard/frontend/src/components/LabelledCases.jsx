import React, { useState, useMemo } from 'react';

export default function LabelledCases({ labels = [] }) {
  const [filterType, setFilterType] = useState('all'); // 'all', 'fraud', 'genuine'

  const filteredLabels = useMemo(() => {
    if (filterType === 'all') return labels;
    return labels.filter((l) => l.label.toLowerCase() === filterType);
  }, [labels, filterType]);

  const handleExportCSV = () => {
    if (!labels || labels.length === 0) return;
    const header = ['TransactionID', 'label', 'operator', 'created_at'].join(',');
    const rows = labels.map((l) => [l.TransactionID, l.label, l.operator, l.created_at].join(','));
    const csvContent = 'data:text/csv;charset=utf-8,' + [header, ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `operator_labels_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="page-container">
      {/* Top action and filter bar */}
      <div className="overview-top-bar">
        <div className="filter-pills" role="radiogroup" aria-label="Filter labels">
          <button
            type="button"
            className={`filter-pill ${filterType === 'all' ? 'active' : ''}`}
            onClick={() => setFilterType('all')}
          >
            All Cases ({labels.length})
          </button>
          <button
            type="button"
            className={`filter-pill ${filterType === 'fraud' ? 'active' : ''}`}
            onClick={() => setFilterType('fraud')}
          >
            Confirmed Fraud ({labels.filter((l) => l.label === 'fraud').length})
          </button>
          <button
            type="button"
            className={`filter-pill ${filterType === 'genuine' ? 'active' : ''}`}
            onClick={() => setFilterType('genuine')}
          >
            Marked Genuine ({labels.filter((l) => l.label === 'genuine').length})
          </button>
        </div>

        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleExportCSV}
          disabled={labels.length === 0}
          aria-label="Export labelled cases to CSV"
        >
          📥 Export to CSV
        </button>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        {filteredLabels.length > 0 ? (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Transaction ID</th>
                  <th>Operator Decision</th>
                  <th>Reviewing Operator</th>
                  <th>Labelled Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {filteredLabels.map((l) => (
                  <tr key={l.TransactionID}>
                    <td>
                      <strong style={{ color: '#0F3D3E' }}>{l.TransactionID}</strong>
                    </td>
                    <td>
                      {l.label === 'fraud' ? (
                        <span className="badge badge-fraud">CONFIRMED FRAUD</span>
                      ) : (
                        <span className="badge badge-genuine">CONFIRMED GENUINE</span>
                      )}
                    </td>
                    <td>
                      <span style={{ fontSize: '13px', color: '#475569' }}>
                        {l.operator || 'operator'}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '12px', color: '#64748B' }}>
                        {new Date(l.created_at).toLocaleString()}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">🏷️</div>
            <div className="empty-state-title">No Labelled Cases Yet</div>
            <div className="empty-state-text">
              Transactions triaged in the Review Queue will automatically be cataloged here with full operator audit tracking.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

