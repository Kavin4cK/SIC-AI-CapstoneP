import React from 'react';

export default function ModelThreshold({ metrics }) {
  if (!metrics) {
    return (
      <div className="page-container">
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">⚙️</div>
            <div className="empty-state-title">Loading Model Benchmarks...</div>
            <div className="empty-state-text">
              Reading verified performance figures from evaluation outputs and project documentation.
            </div>
          </div>
        </div>
      </div>
    );
  }

  const {
    model_name,
    decision_threshold,
    recall,
    precision,
    accuracy,
    f1_score,
    pr_auc,
    roc_auc,
    total_hypothetical_cost,
    average_cost_per_tx,
    confusion_matrix,
    source_file,
  } = metrics;

  return (
    <div className="page-container">
      {/* Top Config Card */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Deployed ML Model Architecture</h3>
            <p style={{ fontSize: '13px', color: '#64748B' }}>
              Audited production parameters for asymmetric fraud classification
            </p>
          </div>
          <span className="badge badge-genuine" style={{ fontSize: '12px', padding: '6px 12px' }}>
            Production Benchmark
          </span>
        </div>

        <div className="detail-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
          <div className="detail-item">
            <span className="detail-item-label">Model Engine</span>
            <span className="detail-item-value" style={{ color: '#0F3D3E' }}>
              {model_name} Classifier
            </span>
          </div>

          <div className="detail-item">
            <span className="detail-item-label">Frozen Decision Threshold (t*)</span>
            <span className="detail-item-value" style={{ color: '#B45309' }}>
              {decision_threshold.toFixed(4)}
            </span>
          </div>

          <div className="detail-item">
            <span className="detail-item-label">Optimization Criterion</span>
            <span className="detail-item-value">
              Cost-Sensitive Loss
            </span>
          </div>

          <div className="detail-item">
            <span className="detail-item-label">Verified Data Source</span>
            <span className="detail-item-value" style={{ fontSize: '13px', color: '#64748B' }}>
              {source_file}
            </span>
          </div>
        </div>
      </div>

      {/* Test Partition Performance Metrics */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Unbiased Test Set Evaluation (42,722 Transactions)</h3>
            <p style={{ fontSize: '13px', color: '#64748B' }}>
              Verified out-of-sample metrics (untouched test partition; 74 fraud cases)
            </p>
          </div>
        </div>

        <div className="kpi-grid">
          <div className="kpi-card">
            <div className="kpi-label">Recall (Sensitivity)</div>
            <div className="kpi-value" style={{ color: '#0F3D3E' }}>
              {(recall * 100).toFixed(2)}%
            </div>
            <div className="kpi-subtext">
              {confusion_matrix.tp} / 74 fraudulent charges intercepted
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Precision</div>
            <div className="kpi-value" style={{ color: '#0F3D3E' }}>
              {(precision * 100).toFixed(2)}%
            </div>
            <div className="kpi-subtext">
              {confusion_matrix.tp} TP / {confusion_matrix.flagged_transactions || 133} total alerts
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">F1-Score</div>
            <div className="kpi-value" style={{ color: '#0F3D3E' }}>
              {f1_score.toFixed(4)}
            </div>
            <div className="kpi-subtext">
              Harmonic balance under extreme imbalance
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">PR-AUC</div>
            <div className="kpi-value" style={{ color: '#0F3D3E' }}>
              {pr_auc.toFixed(4)}
            </div>
            <div className="kpi-subtext">
              Precision-Recall area under curve
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">ROC-AUC</div>
            <div className="kpi-value" style={{ color: '#0F3D3E' }}>
              {roc_auc.toFixed(4)}
            </div>
            <div className="kpi-subtext">
              Discrimination threshold capacity
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Total Hypothetical Cost</div>
            <div className="kpi-value" style={{ color: '#B45309' }}>
              ₹{total_hypothetical_cost.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div className="kpi-subtext">
              ₹{average_cost_per_tx.toFixed(4)} average / transaction
            </div>
          </div>
        </div>
      </div>

      {/* Confusion Matrix Section */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Test Set Confusion Matrix</h3>
            <p style={{ fontSize: '13px', color: '#64748B' }}>
              Evaluating alert outcomes at frozen decision cutoff t* = 0.144
            </p>
          </div>
        </div>

        <div className="confusion-matrix-grid">
          <div></div>
          <div style={{ textAlign: 'center', fontSize: '12px', fontWeight: 700, color: '#64748B' }}>
            PREDICTED GENUINE
          </div>
          <div style={{ textAlign: 'center', fontSize: '12px', fontWeight: 700, color: '#64748B' }}>
            PREDICTED FRAUD
          </div>

          <div style={{ display: 'flex', alignItems: 'center', fontSize: '12px', fontWeight: 700, color: '#64748B' }}>
            ACTUAL GENUINE
          </div>
          <div className="cm-cell cm-tn">
            <div className="cm-value">{confusion_matrix.tn.toLocaleString()}</div>
            <div className="cm-label">True Negative (TN)</div>
            <span style={{ fontSize: '10px', color: '#047857' }}>Cost: ₹0</span>
          </div>
          <div className="cm-cell cm-fp">
            <div className="cm-value">{confusion_matrix.fp}</div>
            <div className="cm-label">False Positive (FP)</div>
            <span style={{ fontSize: '10px', color: '#B45309' }}>Review: ₹5/alert (₹350)</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', fontSize: '12px', fontWeight: 700, color: '#64748B' }}>
            ACTUAL FRAUD
          </div>
          <div className="cm-cell cm-fn">
            <div className="cm-value">{confusion_matrix.fn}</div>
            <div className="cm-label">False Negative (FN)</div>
            <span style={{ fontSize: '10px', color: '#B91C1C' }}>Missed: ₹500/miss (₹5,500)</span>
          </div>
          <div className="cm-cell cm-tp">
            <div className="cm-value">{confusion_matrix.tp}</div>
            <div className="cm-label">True Positive (TP)</div>
            <span style={{ fontSize: '10px', color: '#047857' }}>Caught: ₹0 loss</span>
          </div>
        </div>

        {/* Governance Notice */}
        <div style={{ marginTop: '24px', padding: '16px', background: '#F8FAFC', borderRadius: '8px', borderLeft: '4px solid #0F3D3E', fontSize: '12px', color: '#475569' }}>
          <strong>Academic Benchmark Disclaimer:</strong> All cost figures (₹5 investigation, ₹500 missed fraud) are academic assumptions formulated for experimental methodology verification. Real-world financial deployment necessitates device intelligence, IP telemetry, and sub-50ms core-banking switch protocols not present in anonymized tabular PCA datasets.
        </div>
      </div>
    </div>
  );
}

