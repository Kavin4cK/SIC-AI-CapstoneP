import React from 'react';

export default function Retraining({ labels = [] }) {
  const fraudCount = labels.filter((l) => l.label === 'fraud').length;
  const genuineCount = labels.filter((l) => l.label === 'genuine').length;
  const totalCount = labels.length;

  return (
    <div className="page-container">
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Retraining Pipeline Status</h3>
            <p style={{ fontSize: '13px', color: '#64748B' }}>
              Continuous learning feedback loop & model recalibration
            </p>
          </div>
          <span className="badge badge-pending" style={{ fontSize: '12px', padding: '6px 12px' }}>
            Handled by ML Teammates
          </span>
        </div>

        <div style={{ margin: '20px 0', padding: '20px', background: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
          <h4 style={{ fontSize: '16px', fontWeight: 600, color: '#0F3D3E', marginBottom: '8px' }}>
            ℹ️ Engineering Notice
          </h4>
          <p style={{ fontSize: '14px', color: '#475569', lineHeight: 1.6 }}>
            The automated model retraining pipeline and scheduled threshold recalibration are managed by our teammates on the ML Engineering & LLM-as-a-Judge team. Verified operator determinations recorded in this dashboard feed directly into the retraining feedback database without altering production baseline weights until scheduled release audits.
          </p>
        </div>

        {/* Live Label Metrics */}
        <h4 style={{ fontSize: '15px', fontWeight: 600, color: '#1E293B', marginBottom: '14px' }}>
          Feedback Dataset Ingestion
        </h4>

        <div className="kpi-grid">
          <div className="kpi-card">
            <div className="kpi-label">Total Operator Labels</div>
            <div className="kpi-value" style={{ color: '#0F3D3E' }}>
              {totalCount}
            </div>
            <div className="kpi-subtext">
              Awaiting next retraining batch ingestion
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Confirmed Fraud Labels</div>
            <div className="kpi-value" style={{ color: '#B91C1C' }}>
              {fraudCount}
            </div>
            <div className="kpi-subtext">
              High-value true positives for boosting weights
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Marked Genuine Labels</div>
            <div className="kpi-value" style={{ color: '#047857' }}>
              {genuineCount}
            </div>
            <div className="kpi-subtext">
              False positive corrections to reduce review volume
            </div>
          </div>
        </div>

        {/* Feedback Loop Architecture Overview */}
        <div style={{ marginTop: '24px', borderTop: '1px solid #E2E8F0', paddingTop: '20px' }}>
          <h4 style={{ fontSize: '15px', fontWeight: 600, color: '#1E293B', marginBottom: '12px' }}>
            Operator Feedback Architecture
          </h4>
          <ol style={{ paddingLeft: '20px', fontSize: '13px', color: '#64748B', lineHeight: 1.8 }}>
            <li>
              <strong>Operator Decision Ingestion:</strong> Decisions recorded in SQLite (`labels.sqlite3`) preserve transaction IDs without mutating input predictions.
            </li>
            <li>
              <strong>Audit & LLM Judge Validation:</strong> Discrepancies between model probabilities and human determinations are audited.
            </li>
            <li>
              <strong>Scheduled Batch Retraining:</strong> The CatBoost model (`catboost_model.cbm`) is incrementally refitted with balanced class weights.
            </li>
            <li>
              <strong>Threshold Recalibration:</strong> The optimal threshold ($t^* = 0.144$) is re-evaluated across asymmetric loss scenarios (₹5 review vs ₹500 missed fraud).
            </li>
          </ol>
        </div>
      </div>
    </div>
  );
}

