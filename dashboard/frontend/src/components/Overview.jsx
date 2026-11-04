import React from 'react';

export default function Overview({
  transactions,
  metrics,
  flagsPerHour,
  onNavigateToReview,
  onSelectTransactionForReview,
}) {
  const pendingTransactions = transactions.filter(
    (t) => t.assessment_status === 'SUSPICIOUS_PENDING_REVIEW'
  );
  const pendingCount = pendingTransactions.length;

  // Real metrics from backend /metrics (never hardcoded)
  const recallVal = metrics ? (metrics.recall * 100).toFixed(2) : '—';
  const costVal = metrics ? `₹${metrics.total_hypothetical_cost.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—';
  const avgCostVal = metrics ? `₹${metrics.average_cost_per_tx.toFixed(4)} avg / tx` : '—';

  // Top needs attention (top 5 pending)
  const topAttention = pendingTransactions.slice(0, 5);

  // Maximum value for bar chart scaling
  const maxFlags = flagsPerHour && flagsPerHour.length > 0
    ? Math.max(...flagsPerHour.map((b) => b.flags), 1)
    : 1;

  return (
    <div className="page-container">
      {/* Top action bar with primary Amber button */}
      <div className="overview-top-bar">
        <div>
          <h2 style={{ fontSize: '22px', fontWeight: 700, color: '#0F3D3E' }}>
            Operational Triage
          </h2>
          <p style={{ fontSize: '13px', color: '#64748B' }}>
            Current queue status under frozen CatBoost threshold 0.144
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={onNavigateToReview}
          aria-label={`Review ${pendingCount} pending suspicious transactions`}
        >
          Review {pendingCount} pending {pendingCount === 1 ? 'transaction' : 'transactions'} →
        </button>
      </div>

      {/* 3 KPI Cards */}
      <div className="kpi-grid">
        {/* Card 1: Flagged Suspicious (Amber top border) */}
        <div className="kpi-card kpi-flagged">
          <div className="kpi-label">Flagged Suspicious</div>
          <div className="kpi-value" style={{ color: '#B45309' }}>
            {pendingCount}
          </div>
          <div className="kpi-subtext">
            {pendingCount > 0 ? 'Pending manual operator review' : 'Queue cleared — no pending alerts'}
          </div>
        </div>

        {/* Card 2: Recall (Teal top border) */}
        <div className="kpi-card">
          <div className="kpi-label">Recall (Test Benchmark)</div>
          <div className="kpi-value" style={{ color: '#0F3D3E' }}>
            {metrics ? `${recallVal}%` : '—'}
          </div>
          <div className="kpi-subtext">
            {metrics ? `${metrics.confusion_matrix.tp} of 74 rare fraud cases captured` : 'Loading benchmark metrics...'}
          </div>
        </div>

        {/* Card 3: Hypothetical Cost (Teal top border) */}
        <div className="kpi-card">
          <div className="kpi-label">Hypothetical Cost</div>
          <div className="kpi-value" style={{ color: '#0F3D3E' }}>
            {costVal}
          </div>
          <div className="kpi-subtext">
            {avgCostVal} (FN ₹500 / FP ₹5 ratio)
          </div>
        </div>
      </div>

      {/* Overview 2-Column Grid: Flags per Hour + Needs Attention */}
      <div className="overview-grid">
        {/* Flags per Hour Bar Chart */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Flags per Hour</h3>
            <span style={{ fontSize: '12px', color: '#64748B' }}>
              Elapsed Time (0–48h)
            </span>
          </div>

          {flagsPerHour && flagsPerHour.length > 0 ? (
            <div className="chart-container">
              <div className="bar-chart" role="img" aria-label="Bar chart showing flags per hour">
                {flagsPerHour.map((item) => {
                  const heightPct = (item.flags / maxFlags) * 100;
                  return (
                    <div key={item.hour} className="bar-column">
                      <div className="bar-tooltip">
                        Hour {item.hour}: {item.flags} {item.flags === 1 ? 'flag' : 'flags'}
                      </div>
                      <div
                        className="bar-fill"
                        style={{
                          height: `${Math.max(heightPct, item.flags > 0 ? 8 : 2)}%`,
                          opacity: item.flags > 0 ? 1 : 0.25,
                        }}
                      />
                      <span className="bar-label">{item.hour}h</span>
                    </div>
                  );
                })}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748B' }}>
                <span>Start: 0h</span>
                <span>Peak: {maxFlags} flags/hr</span>
                <span>Time axis represents elapsed seconds</span>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <div className="empty-state-title">No Flags Data Available</div>
              <div className="empty-state-text">
                Flags per hour will populate as transactions are ingested and scored.
              </div>
            </div>
          )}
        </div>

        {/* Needs Attention List */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Needs Attention</h3>
            <span style={{ fontSize: '12px', color: '#B45309', fontWeight: 600 }}>
              Top Suspicious
            </span>
          </div>

          {topAttention.length > 0 ? (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Risk Score</th>
                    <th>Amount</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {topAttention.map((t) => (
                    <tr key={t.TransactionID}>
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
                        <button
                          type="button"
                          className="btn btn-primary btn-sm"
                          onClick={() => onSelectTransactionForReview(t.TransactionID)}
                          aria-label={`Review transaction ${t.TransactionID}`}
                        >
                          Review
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">✅</div>
              <div className="empty-state-title">All Caught Up!</div>
              <div className="empty-state-text">
                There are no pending suspicious transactions requiring attention right now.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

