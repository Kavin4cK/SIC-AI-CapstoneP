import React, { useState, useMemo } from 'react';

export default function ReviewQueue({
  transactions,
  selectedId,
  onSelectTransaction,
  onApplyLabel,
  isSubmitting,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('pending'); // 'all', 'pending', 'fraud', 'genuine'

  // Filter transactions
  const filtered = useMemo(() => {
    return transactions.filter((t) => {
      // Status filter
      if (statusFilter === 'pending' && t.assessment_status !== 'SUSPICIOUS_PENDING_REVIEW') {
        return false;
      }
      if (statusFilter === 'fraud' && t.assessment_status !== 'CONFIRMED_FRAUD') {
        return false;
      }
      if (statusFilter === 'genuine' && t.assessment_status !== 'CONFIRMED_GENUINE') {
        return false;
      }

      // Search filter
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        return t.TransactionID.toLowerCase().includes(q);
      }
      return true;
    });
  }, [transactions, statusFilter, searchQuery]);

  // Selected item: either specified by selectedId or the first in filtered list
  const selectedItem = useMemo(() => {
    if (selectedId) {
      const found = transactions.find((t) => t.TransactionID === selectedId);
      if (found) return found;
    }
    return filtered.length > 0 ? filtered[0] : null;
  }, [transactions, selectedId, filtered]);

  const threshold = selectedItem ? selectedItem.selected_threshold : 0.144;
  const score = selectedItem ? selectedItem.fraud_probability : 0;
  const deltaFromThreshold = score - threshold;

  return (
    <div className="page-container">
      {/* Search and Filter Controls */}
      <div className="filter-bar">
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <input
            type="text"
            className="search-input"
            placeholder="Search by Transaction ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            aria-label="Search by Transaction ID"
          />
          <div className="filter-pills" role="radiogroup" aria-label="Filter status">
            <button
              type="button"
              className={`filter-pill ${statusFilter === 'pending' ? 'active' : ''}`}
              onClick={() => setStatusFilter('pending')}
            >
              Pending ({transactions.filter((t) => t.assessment_status === 'SUSPICIOUS_PENDING_REVIEW').length})
            </button>
            <button
              type="button"
              className={`filter-pill ${statusFilter === 'all' ? 'active' : ''}`}
              onClick={() => setStatusFilter('all')}
            >
              All ({transactions.length})
            </button>
            <button
              type="button"
              className={`filter-pill ${statusFilter === 'fraud' ? 'active' : ''}`}
              onClick={() => setStatusFilter('fraud')}
            >
              Fraud ({transactions.filter((t) => t.assessment_status === 'CONFIRMED_FRAUD').length})
            </button>
            <button
              type="button"
              className={`filter-pill ${statusFilter === 'genuine' ? 'active' : ''}`}
              onClick={() => setStatusFilter('genuine')}
            >
              Genuine ({transactions.filter((t) => t.assessment_status === 'CONFIRMED_GENUINE').length})
            </button>
          </div>
        </div>
      </div>

      {filtered.length > 0 ? (
        <div className="review-queue-layout">
          {/* Left Table */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div className="table-wrap">
              <table className="data-table" role="table">
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Fraud Probability</th>
                    <th>Amount</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((t) => {
                    const isSelected = selectedItem && selectedItem.TransactionID === t.TransactionID;
                    return (
                      <tr
                        key={t.TransactionID}
                        className={isSelected ? 'row-selected' : ''}
                        onClick={() => onSelectTransaction(t.TransactionID)}
                        style={{ cursor: 'pointer' }}
                        tabIndex={0}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            onSelectTransaction(t.TransactionID);
                          }
                        }}
                        aria-label={`Select transaction ${t.TransactionID}`}
                      >
                        <td>
                          <strong style={{ color: '#0F3D3E' }}>{t.TransactionID}</strong>
                        </td>
                        <td>
                          <div className="prob-cell">
                            <div className="prob-bar-track">
                              <div
                                className={`prob-bar-fill ${t.fraud_probability > 0.8 ? 'prob-high' : ''}`}
                                style={{ width: `${Math.min(t.fraud_probability * 100, 100)}%` }}
                              />
                            </div>
                            <span className="prob-text">
                              {(t.fraud_probability * 100).toFixed(2)}%
                            </span>
                          </div>
                        </td>
                        <td>
                          {t.amount !== null && t.amount !== undefined
                            ? `₹${t.amount.toFixed(2)}`
                            : '—'}
                        </td>
                        <td>
                          {t.assessment_status === 'CONFIRMED_FRAUD' ? (
                            <span className="badge badge-fraud">CONFIRMED FRAUD</span>
                          ) : t.assessment_status === 'CONFIRMED_GENUINE' ? (
                            <span className="badge badge-genuine">GENUINE</span>
                          ) : (
                            <span className="badge badge-pending">PENDING</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right Detail Panel */}
          {selectedItem && (
            <div className="detail-panel">
              <div className="detail-panel-header">
                <span style={{ fontSize: '12px', textTransform: 'uppercase', color: '#64748B', fontWeight: 600 }}>
                  Case Investigation
                </span>
                <h3 className="detail-txn-id">{selectedItem.TransactionID}</h3>
                <div style={{ marginTop: '6px' }}>
                  {selectedItem.assessment_status === 'CONFIRMED_FRAUD' ? (
                    <span className="badge badge-fraud">Labeled: Confirmed Fraud</span>
                  ) : selectedItem.assessment_status === 'CONFIRMED_GENUINE' ? (
                    <span className="badge badge-genuine">Labeled: Genuine</span>
                  ) : (
                    <span className="badge badge-pending">Action Required: Pending Review</span>
                  )}
                </div>
              </div>

              {/* Score vs Threshold 0.144 Meter */}
              <div className="threshold-meter-card">
                <div className="meter-title">Decision Threshold Benchmark</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600 }}>
                  <span>CatBoost Risk Score: {(score * 100).toFixed(2)}%</span>
                  <span style={{ color: '#0F3D3E' }}>Frozen Cutoff: {(threshold * 100).toFixed(1)}%</span>
                </div>

                <div className="meter-bar-container">
                  <div
                    className="meter-score-fill"
                    style={{ width: `${Math.min(score * 100, 100)}%` }}
                  />
                  <div
                    className="meter-threshold-pin"
                    style={{ left: `${threshold * 100}%` }}
                  />
                  <div
                    className="meter-threshold-label"
                    style={{ left: `${threshold * 100}%` }}
                  >
                    t = {threshold.toFixed(3)}
                  </div>
                </div>

                <div style={{ fontSize: '12px', color: deltaFromThreshold >= 0 ? '#B45309' : '#047857', fontWeight: 600 }}>
                  {deltaFromThreshold >= 0
                    ? `▲ Exceeds cost threshold by +${(deltaFromThreshold * 100).toFixed(2)}% (High Priority Alert)`
                    : `▼ Below threshold`}
                </div>
              </div>

              {/* Transaction Attributes */}
              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-item-label">Transaction Amount</span>
                  <span className="detail-item-value">
                    {selectedItem.amount !== null && selectedItem.amount !== undefined
                      ? `₹${selectedItem.amount.toFixed(2)}`
                      : '—'}
                  </span>
                </div>

                <div className="detail-item">
                  <span className="detail-item-label">Elapsed Time</span>
                  <span className="detail-item-value">
                    {selectedItem.time !== null && selectedItem.time !== undefined
                      ? `${selectedItem.time.toFixed(1)}s (${(selectedItem.time / 3600).toFixed(2)}h)`
                      : '—'}
                  </span>
                </div>

                <div className="detail-item">
                  <span className="detail-item-label">Scoring Model</span>
                  <span className="detail-item-value" style={{ fontSize: '14px' }}>
                    {selectedItem.selected_model}
                  </span>
                </div>

                <div className="detail-item">
                  <span className="detail-item-label">Original Status</span>
                  <span className="detail-item-value" style={{ fontSize: '12px', color: '#64748B' }}>
                    SUSPICIOUS_PENDING_REVIEW
                  </span>
                </div>
              </div>

              {/* Cost If Wrong Box */}
              <div className="cost-penalty-box">
                <div className="cost-penalty-title">Asymmetric Financial Cost Impact</div>
                <div className="cost-row">
                  <span>Missed Fraud Loss (FN penalty):</span>
                  <strong>₹500.00</strong>
                </div>
                <div className="cost-row">
                  <span>False Alarm Review (FP penalty):</span>
                  <strong>₹5.00</strong>
                </div>
                <p style={{ fontSize: '11px', color: '#78350F', marginTop: '6px' }}>
                  *Missed fraud carries a 100x higher penalty than manual review. The model optimizes to minimize this financial loss.
                </p>
              </div>

              {/* Action Buttons */}
              <div className="detail-actions">
                <button
                  type="button"
                  className="btn btn-fraud"
                  onClick={() => onApplyLabel(selectedItem.TransactionID, 'fraud')}
                  disabled={isSubmitting}
                  aria-label={`Confirm transaction ${selectedItem.TransactionID} as fraud`}
                >
                  Confirm fraud
                </button>
                <button
                  type="button"
                  className="btn btn-genuine"
                  onClick={() => onApplyLabel(selectedItem.TransactionID, 'genuine')}
                  disabled={isSubmitting}
                  aria-label={`Mark transaction ${selectedItem.TransactionID} as genuine`}
                >
                  Mark genuine
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">🔍</div>
            <div className="empty-state-title">No Transactions Match Filter</div>
            <div className="empty-state-text">
              Try adjusting your search query or status filter to view transactions.
            </div>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => {
                setSearchQuery('');
                setStatusFilter('all');
              }}
            >
              Reset Filters
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
