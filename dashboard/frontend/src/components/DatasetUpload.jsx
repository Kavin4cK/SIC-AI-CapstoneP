import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  uploadDataset,
  useExistingDataset,
  runDatasetPrediction,
  fetchDatasetResults,
  getDownloadResultsUrl,
  getDownloadTemplateUrl,
} from '../api';

export default function DatasetUpload({
  onApplyLabel,
  isSubmittingLabel,
  activeDatasetId,
  onSelectDataset,
}) {
  // Mode selection: 'existing' vs 'upload'
  const [datasetSource, setDatasetSource] = useState('existing');

  // Upload & File State
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationError, setValidationError] = useState(null);

  // Loaded Dataset State (Preview)
  const [loadedDataset, setLoadedDataset] = useState(null); // DatasetUploadResponse

  // Inference State
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionSummary, setPredictionSummary] = useState(null); // PredictionResponse

  // Results & Table State
  const [resultsData, setResultsData] = useState(null); // DatasetResultsResponse
  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [statusFilter, setStatusFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('fraud_probability');
  const [sortOrder, setSortOrder] = useState('desc');

  // Detail Modal State
  const [inspectedTransaction, setInspectedTransaction] = useState(null);

  // 1. Handle Option A: Select Existing ULB Dataset
  const handleSelectExistingULB = async () => {
    setValidationError(null);
    setIsValidating(true);
    setPredictionSummary(null);
    setResultsData(null);
    setSelectedFile(null);
    try {
      const data = await useExistingDataset();
      setLoadedDataset(data);
    } catch (err) {
      setValidationError(err.message || 'Failed to load existing ULB dataset.');
    } finally {
      setIsValidating(false);
    }
  };

  // 2. Handle Option B: Upload Custom CSV
  const handleFileChange = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setValidationError('Invalid file format. Please upload a standard comma-separated CSV file (.csv).');
      return;
    }

    setSelectedFile(file);
    setValidationError(null);
    setIsValidating(true);
    setLoadedDataset(null);
    setPredictionSummary(null);
    setResultsData(null);

    try {
      const data = await uploadDataset(file);
      setLoadedDataset(data);
    } catch (err) {
      setValidationError(err.message || 'Dataset validation failed. Please check required columns and values.');
    } finally {
      setIsValidating(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  // 3. Execute Fraud Detection Inference
  const handleRunPrediction = async () => {
    if (!loadedDataset) return;
    setIsPredicting(true);
    setValidationError(null);

    try {
      const pred = await runDatasetPrediction(loadedDataset.dataset_id, 0.144);
      setPredictionSummary(pred);
      if (onSelectDataset) {
        onSelectDataset(loadedDataset.dataset_id);
      }
      setCurrentPage(1);
      // Fetch first page of results
      await loadResults(loadedDataset.dataset_id, 1, pageSize, statusFilter, searchQuery, sortBy, sortOrder);
    } catch (err) {
      setValidationError(err.message || 'Error executing fraud inference on dataset.');
    } finally {
      setIsPredicting(false);
    }
  };

  // 4. Fetch Paginated Results
  const loadResults = useCallback(
    async (
      datasetId,
      page = 1,
      size = 25,
      status = 'all',
      search = '',
      sort = 'fraud_probability',
      order = 'desc'
    ) => {
      setIsLoadingResults(true);
      try {
        const res = await fetchDatasetResults(datasetId, {
          page,
          pageSize: size,
          status,
          search,
          sortBy: sort,
          sortOrder: order,
        });
        setResultsData(res);
      } catch (err) {
        console.error('Failed to fetch dataset results:', err);
      } finally {
        setIsLoadingResults(false);
      }
    },
    []
  );

  // Trigger results reload when pagination / filters change
  useEffect(() => {
    if (loadedDataset && predictionSummary) {
      loadResults(loadedDataset.dataset_id, currentPage, pageSize, statusFilter, searchQuery, sortBy, sortOrder);
    }
  }, [loadedDataset, predictionSummary, currentPage, pageSize, statusFilter, searchQuery, sortBy, sortOrder, loadResults]);

  // Handle inline operator labeling directly from the dataset results table
  const handleInlineLabel = async (txnId, label) => {
    if (onApplyLabel) {
      await onApplyLabel(txnId, label, loadedDataset?.dataset_id || activeDatasetId);
      // Refresh current results page to show updated assessment_status
      if (loadedDataset) {
        loadResults(loadedDataset.dataset_id, currentPage, pageSize, statusFilter, searchQuery, sortBy, sortOrder);
      }
    }
  };

  // Reset to initial state
  const handleReset = () => {
    setLoadedDataset(null);
    setPredictionSummary(null);
    setResultsData(null);
    setSelectedFile(null);
    setValidationError(null);
    setCurrentPage(1);
  };

  // Evaluation metrics if present
  const evalMetrics = predictionSummary?.evaluation_metrics;
  const charts = predictionSummary?.charts;
  const summary = predictionSummary?.summary;

  return (
    <div className="page-container">
      {/* Top Title & Operational Banner */}
      <div className="overview-top-bar">
        <div>
          <h2 style={{ fontSize: '22px', fontWeight: 700, color: '#0F3D3E' }}>
            Batch Dataset Fraud Detection
          </h2>
          <p style={{ fontSize: '13px', color: '#64748B' }}>
            Score transactions through the frozen CatBoost model (threshold = 0.144) with full ground-truth evaluation
          </p>
        </div>
        {resultsData && (
          <div style={{ display: 'flex', gap: '8px' }}>
            <a
              href={getDownloadResultsUrl(loadedDataset.dataset_id)}
              download={`predictions_${loadedDataset.dataset_id}.csv`}
              className="btn btn-primary"
              style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
            >
              📥 Download Prediction Results CSV
            </a>
            <button type="button" className="btn btn-secondary" onClick={handleReset}>
              Upload Another
            </button>
          </div>
        )}
      </div>

      {/* STEP 1: DATASET SELECTION (Only shown before prediction) */}
      {!predictionSummary && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginBottom: '24px' }}>
          {/* Option Selector Toggle Tabs */}
          <div className="dataset-mode-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={datasetSource === 'existing'}
              className={`dataset-tab-btn ${datasetSource === 'existing' ? 'active' : ''}`}
              onClick={() => {
                setDatasetSource('existing');
                setValidationError(null);
              }}
            >
              <span className="tab-icon">🗄️</span>
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontWeight: 600 }}>Option A: Use Existing Dataset</div>
                <div style={{ fontSize: '11px', opacity: 0.8 }}>Original ULB Credit Card Fraud Dataset (284,807 tx)</div>
              </div>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={datasetSource === 'upload'}
              className={`dataset-tab-btn ${datasetSource === 'upload' ? 'active' : ''}`}
              onClick={() => {
                setDatasetSource('upload');
                setValidationError(null);
              }}
            >
              <span className="tab-icon">📤</span>
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontWeight: 600 }}>Option B: Upload Custom Dataset</div>
                <div style={{ fontSize: '11px', opacity: 0.8 }}>Upload compatible transaction CSV (V1–V28, Amount, Time)</div>
              </div>
            </button>
          </div>

          {/* Option A Body */}
          {datasetSource === 'existing' && (
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Original ULB Credit Card Benchmark Dataset</h3>
                  <p style={{ fontSize: '13px', color: '#64748B', marginTop: '2px' }}>
                    Available locally at <code>data/creditcard.csv</code> (150 MB). Contains 284,807 transactions with 492 fraud cases (0.172%).
                  </p>
                </div>
                <span className="badge badge-genuine">Local Benchmark Ready</span>
              </div>
              <div style={{ padding: '16px 20px', background: '#F8FAFC', borderRadius: '8px', margin: '0 0 16px 0', border: '1px solid #E2E8F0' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', fontSize: '13px' }}>
                  <div><strong>Total Transactions:</strong> 284,807</div>
                  <div><strong>Required Features:</strong> 30 (Time, Amount, V1–V28)</div>
                  <div><strong>Ground Truth:</strong> Class column included</div>
                  <div><strong>Model:</strong> CatBoost (models/catboost_model.cbm)</div>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={handleSelectExistingULB}
                  disabled={isValidating}
                >
                  {isValidating ? 'Loading Benchmark Dataset...' : 'Select & Preview ULB Dataset'}
                </button>
              </div>
            </div>
          )}

          {/* Option B Body */}
          {datasetSource === 'upload' && (
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Upload Custom Transaction Dataset</h3>
                  <p style={{ fontSize: '13px', color: '#64748B', marginTop: '2px' }}>
                    Must match the ULB feature representation (V1–V28 PCA, Amount, Time).
                  </p>
                </div>
                <a
                  href={getDownloadTemplateUrl()}
                  download="credit_card_transactions_template.csv"
                  className="btn btn-secondary btn-sm"
                  style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
                >
                  📄 Download CSV Template
                </a>
              </div>

              {/* Strict Compatibility Explanation Banner */}
              <div className="compatibility-notice">
                <div className="compatibility-title">⚠️ Strict Feature Compatibility Invariant:</div>
                <p>
                  Our CatBoost model expects exactly <strong>30 numerical input features</strong>: <code>Time</code>, <code>Amount</code>, and <code>V1</code> through <code>V28</code>.
                  Optional columns: <code>TransactionID</code> (preserved or generated) and <code>Class</code> (strictly excluded from model features; used exclusively for post-prediction evaluation).
                </p>
                <div style={{ marginTop: '8px', padding: '8px 12px', background: '#FFFBEB', borderRadius: '4px', border: '1px solid #FDE68A', fontSize: '12px', color: '#92400E' }}>
                  <strong>Important:</strong> V1–V28 are anonymized PCA-transformed features derived exclusively from the European credit card distribution. 
                  Arbitrary banking CSV files (e.g. raw transaction logs, customer IDs, merchant names, or arbitrary columns named V1–V28 from other datasets) are <strong>NOT directly supported</strong>. 
                  Matching column names alone does not guarantee compatible PCA feature representations.
                </div>
              </div>

              {/* Drag-and-Drop Dropzone */}
              <div
                className={`dropzone ${isDragOver ? 'dropzone-active' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => document.getElementById('csv-file-input').click()}
                tabIndex={0}
                role="button"
                aria-label="Upload CSV file"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    document.getElementById('csv-file-input').click();
                  }
                }}
              >
                <input
                  id="csv-file-input"
                  type="file"
                  accept=".csv"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      handleFileChange(e.target.files[0]);
                    }
                  }}
                />
                <div style={{ fontSize: '36px', marginBottom: '8px' }}>📂</div>
                <div style={{ fontWeight: 600, fontSize: '15px', color: '#0F3D3E' }}>
                  {selectedFile ? selectedFile.name : 'Drag & drop your CSV file here, or click to browse'}
                </div>
                <div style={{ fontSize: '12px', color: '#64748B', marginTop: '4px' }}>
                  Supported format: Comma-Separated Values (.csv) up to 250 MB
                </div>
                {isValidating && (
                  <div style={{ marginTop: '12px', color: '#B45309', fontWeight: 600, fontSize: '13px' }}>
                    Validating dataset schema and feature columns...
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Validation Error Banner */}
          {validationError && (
            <div className="alert-banner alert-danger" role="alert">
              <span style={{ fontSize: '18px' }}>❌</span>
              <div>
                <strong>Validation Failed:</strong> {validationError}
              </div>
            </div>
          )}

          {/* Dataset Preview Card (Shown once validated) */}
          {loadedDataset && (
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Dataset Preview & Compatibility Check</h3>
                  <p style={{ fontSize: '13px', color: '#64748B' }}>
                    Dataset: <strong>{loadedDataset.filename}</strong> · {loadedDataset.total_rows.toLocaleString()} total transactions
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <span className="badge badge-genuine">✓ 30 Model Features Valid</span>
                  {loadedDataset.has_class ? (
                    <span className="badge badge-genuine">✓ Ground Truth (Class) Included</span>
                  ) : (
                    <span className="badge badge-pending">Ground Truth (Class) Not Provided</span>
                  )}
                  {loadedDataset.has_transaction_id ? (
                    <span className="badge badge-genuine">✓ TransactionID Present</span>
                  ) : (
                    <span className="badge badge-pending">Auto-Generating Unique IDs</span>
                  )}
                </div>
              </div>

              {/* Preview Table */}
              <div className="table-wrap" style={{ maxHeight: '240px', overflowY: 'auto', marginBottom: '16px' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>TransactionID</th>
                      <th>Time</th>
                      <th>Amount</th>
                      <th>V1</th>
                      <th>V2</th>
                      <th>V3</th>
                      <th>V4</th>
                      {loadedDataset.has_class && <th>Class</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {loadedDataset.preview_rows.map((row, idx) => (
                      <tr key={idx}>
                        <td><strong>{row.TransactionID || `TXN_${idx + 1}`}</strong></td>
                        <td>{row.Time !== undefined ? Number(row.Time).toFixed(1) : '—'}</td>
                        <td>₹{row.Amount !== undefined ? Number(row.Amount).toFixed(2) : '—'}</td>
                        <td>{row.V1 !== undefined ? Number(row.V1).toFixed(4) : '—'}</td>
                        <td>{row.V2 !== undefined ? Number(row.V2).toFixed(4) : '—'}</td>
                        <td>{row.V3 !== undefined ? Number(row.V3).toFixed(4) : '—'}</td>
                        <td>{row.V4 !== undefined ? Number(row.V4).toFixed(4) : '—'}</td>
                        {loadedDataset.has_class && (
                          <td>
                            <span className={row.Class === 1 ? 'badge badge-fraud' : 'badge badge-genuine'}>
                              {row.Class === 1 ? '1 (Fraud)' : '0 (Genuine)'}
                            </span>
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Primary Action Button */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '8px', borderTop: '1px solid #E2E8F0' }}>
                <span style={{ fontSize: '13px', color: '#64748B' }}>
                  Trained CatBoost inference with threshold <code>t* = 0.144</code>
                </span>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={handleRunPrediction}
                  disabled={isPredicting}
                  style={{ minWidth: '220px' }}
                >
                  {isPredicting ? 'Scoring Transactions...' : '🚀 Detect Fraud (CatBoost t=0.144)'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* STEP 2: PROCESSING INDICATOR */}
      {isPredicting && (
        <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
          <div className="spinner" style={{ margin: '0 auto 16px' }} />
          <h3 style={{ fontSize: '18px', color: '#0F3D3E', fontWeight: 700 }}>
            Processing Transactions Through CatBoost Model
          </h3>
          <p style={{ fontSize: '13px', color: '#64748B', maxWidth: '500px', margin: '8px auto 0' }}>
            Applying Phase 1 RobustScaler preprocessing and scoring probabilities against decision threshold 0.144...
          </p>
        </div>
      )}

      {/* STEP 3: RESULTS DASHBOARD */}
      {predictionSummary && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* 4 Summary KPI Cards */}
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">Total Transactions</div>
              <div className="kpi-value" style={{ color: '#0F3D3E' }}>
                {summary.total_transactions.toLocaleString()}
              </div>
              <div className="kpi-subtext">All processed through CatBoost</div>
            </div>

            <div className="kpi-card kpi-flagged">
              <div className="kpi-label">Suspicious Transactions</div>
              <div className="kpi-value" style={{ color: '#B45309' }}>
                {summary.suspicious_count.toLocaleString()}
              </div>
              <div className="kpi-subtext">Probability ≥ 0.144 (Pending Review)</div>
            </div>

            <div className="kpi-card">
              <div className="kpi-label">Not Flagged Transactions</div>
              <div className="kpi-value" style={{ color: '#047857' }}>
                {summary.not_flagged_count.toLocaleString()}
              </div>
              <div className="kpi-subtext">Probability &lt; 0.144 (Automated Pass)</div>
            </div>

            <div className="kpi-card">
              <div className="kpi-label">Flagged Percentage</div>
              <div className="kpi-value" style={{ color: '#0F3D3E' }}>
                {summary.suspicious_percentage}%
              </div>
              <div className="kpi-subtext">Of total uploaded batch</div>
            </div>
          </div>

          {/* Notice when Ground-Truth Class is absent */}
          {!evalMetrics && (
            <div className="card" style={{ borderLeft: '4px solid #64748B', background: '#F8FAFC' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '4px 0' }}>
                <span style={{ fontSize: '22px' }}>ℹ️</span>
                <div>
                  <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#1E293B' }}>
                    Prediction Statistics Only (No Ground-Truth Labels)
                  </h4>
                  <p style={{ fontSize: '13px', color: '#64748B', marginTop: '2px' }}>
                    No <code>Class</code> column was provided with this dataset. Displaying model prediction statistics only.
                    Evaluation benchmarks (Confusion Matrix, Precision, Recall, F1, ROC-AUC) require verified ground-truth labels and are never fabricated.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Optional Ground-Truth Evaluation Card */}
          {evalMetrics && (
            <div className="card" style={{ borderLeft: '4px solid #0F3D3E' }}>
              <div className="card-header">
                <div>
                  <h3 className="card-title">
                    {evalMetrics.is_ulb_benchmark
                      ? 'Full-Dataset Descriptive Evaluation (In-Sample + Out-of-Sample Batch)'
                      : 'Uploaded Dataset Ground-Truth Evaluation'}
                  </h3>
                  <p style={{ fontSize: '13px', color: '#64748B' }}>
                    {evalMetrics.is_ulb_benchmark
                      ? 'Descriptive whole-dataset scoring across all 284,807 transactions (training + validation + test partitions combined).'
                      : 'Evaluated post-prediction using the uploaded Class column (strictly omitted from model inputs).'}
                  </p>
                </div>
                <span className="badge badge-genuine">
                  {evalMetrics.is_ulb_benchmark ? 'Full Dataset Benchmark' : 'Verified Ground Truth'}
                </span>
              </div>

              {evalMetrics.is_ulb_benchmark && (
                <div style={{ background: '#EFF6FF', border: '1px solid #BFDBFE', borderRadius: '6px', padding: '10px 14px', fontSize: '12px', color: '#1E40AF', marginBottom: '16px' }}>
                  <strong>Important Evaluation Scope Distinction:</strong> The original CatBoost model was trained using 70% of this ULB dataset.
                  The metrics below represent <em>descriptive in-sample + out-of-sample batch scoring</em> across the full dataset, not independent test-set generalization.
                  For audited out-of-sample test set benchmarks (<strong>Recall: 85.14%, Precision: 47.37%, ROC-AUC: 0.9710, PR-AUC: 0.8298</strong>), see the <strong>Model and threshold</strong> or <strong>Overview</strong> tabs.
                </div>
              )}

              {/* Confusion Matrix & Metrics Grid */}
              <div className="evaluation-grid">
                {/* 2x2 Confusion Matrix */}
                <div className="cm-container">
                  <div className="cm-header-label">Confusion Matrix (Threshold = 0.144)</div>
                  <div className="cm-table">
                    <div className="cm-cell cm-tp">
                      <span className="cm-val">{evalMetrics.confusion_matrix.tp.toLocaleString()}</span>
                      <span className="cm-tag">True Positive (Captured Fraud)</span>
                    </div>
                    <div className="cm-cell cm-fp">
                      <span className="cm-val">{evalMetrics.confusion_matrix.fp.toLocaleString()}</span>
                      <span className="cm-tag">False Positive (False Alarm)</span>
                    </div>
                    <div className="cm-cell cm-fn">
                      <span className="cm-val">{evalMetrics.confusion_matrix.fn.toLocaleString()}</span>
                      <span className="cm-tag">False Negative (Missed Fraud)</span>
                    </div>
                    <div className="cm-cell cm-tn">
                      <span className="cm-val">{evalMetrics.confusion_matrix.tn.toLocaleString()}</span>
                      <span className="cm-tag">True Negative (Correct Genuine)</span>
                    </div>
                  </div>
                </div>

                {/* Performance Metrics Badges */}
                <div className="metrics-col">
                  <div className="metric-row">
                    <span className="metric-name">Recall (Capture Rate):</span>
                    <strong className="metric-val" style={{ color: '#0F3D3E' }}>
                      {(evalMetrics.recall * 100).toFixed(2)}%
                    </strong>
                  </div>
                  <div className="metric-row">
                    <span className="metric-name">Precision (Alert Accuracy):</span>
                    <strong className="metric-val" style={{ color: '#0F3D3E' }}>
                      {(evalMetrics.precision * 100).toFixed(2)}%
                    </strong>
                  </div>
                  <div className="metric-row">
                    <span className="metric-name">F1-Score:</span>
                    <strong className="metric-val">{evalMetrics.f1_score.toFixed(4)}</strong>
                  </div>
                  {evalMetrics.roc_auc !== null && (
                    <div className="metric-row">
                      <span className="metric-name">ROC-AUC:</span>
                      <strong className="metric-val">{evalMetrics.roc_auc.toFixed(4)}</strong>
                    </div>
                  )}
                  {evalMetrics.pr_auc !== null && (
                    <div className="metric-row">
                      <span className="metric-name">PR-AUC:</span>
                      <strong className="metric-val">{evalMetrics.pr_auc.toFixed(4)}</strong>
                    </div>
                  )}
                  <div className="metric-row" style={{ borderBottom: 'none' }}>
                    <span className="metric-name">Total Actual Fraud in Batch:</span>
                    <strong className="metric-val" style={{ color: '#B91C1C' }}>
                      {evalMetrics.confusion_matrix.total_actual_fraud}
                    </strong>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Visualizations / Charts Section */}
          <div className="overview-grid">
            {/* Chart 1: Fraud Probability Distribution Histogram */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Fraud Probability Distribution</h3>
                <span style={{ fontSize: '12px', color: '#B45309', fontWeight: 600 }}>
                  Threshold Marker: 0.144
                </span>
              </div>
              {charts && charts.probability_histogram ? (
                <div className="chart-container">
                  <div className="bar-chart" role="img" aria-label="Histogram of fraud probabilities">
                    {charts.probability_histogram.map((bin, idx) => {
                      const maxBinCount = Math.max(...charts.probability_histogram.map((b) => b.count), 1);
                      const heightPct = (bin.count / maxBinCount) * 100;
                      return (
                        <div key={idx} className="bar-column">
                          <div className="bar-tooltip">
                            Bin {bin.bin}: {bin.count.toLocaleString()} txns
                            {bin.contains_threshold ? ' (Threshold 0.144)' : ''}
                          </div>
                          <div
                            className="bar-fill"
                            style={{
                              height: `${Math.max(heightPct, bin.count > 0 ? 8 : 2)}%`,
                              backgroundColor: bin.contains_threshold ? '#B45309' : idx >= 2 ? '#B91C1C' : '#0F3D3E',
                              opacity: bin.count > 0 ? 1 : 0.2,
                            }}
                          />
                          <span className="bar-label">{bin.bin.split('–')[0]}</span>
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748B', marginTop: '4px' }}>
                    <span>0.0 (Genuine)</span>
                    <span style={{ color: '#B45309', fontWeight: 600 }}>▲ Decision Threshold 0.144</span>
                    <span>1.0 (High Risk)</span>
                  </div>
                </div>
              ) : null}
            </div>

            {/* Chart 2: Suspicious vs Not-Flagged Distribution & Top Risk */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Detection Classification Split</h3>
                <span style={{ fontSize: '12px', color: '#64748B' }}>
                  CatBoost Decision Boundary
                </span>
              </div>
              {charts && charts.status_distribution ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', padding: '12px 0' }}>
                  {/* Proportional Split Bar */}
                  <div style={{ height: '32px', display: 'flex', borderRadius: '6px', overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${Math.max(charts.status_distribution[0].percentage, 1)}%`,
                        backgroundColor: '#047857',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: '#FFF',
                        fontSize: '12px',
                        fontWeight: 600,
                      }}
                      title={`Not Flagged: ${charts.status_distribution[0].count.toLocaleString()}`}
                    >
                      {charts.status_distribution[0].percentage > 10 ? `${charts.status_distribution[0].percentage}% Not Flagged` : ''}
                    </div>
                    <div
                      style={{
                        width: `${Math.max(charts.status_distribution[1].percentage, 1)}%`,
                        backgroundColor: '#B45309',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: '#FFF',
                        fontSize: '12px',
                        fontWeight: 600,
                      }}
                      title={`Suspicious: ${charts.status_distribution[1].count.toLocaleString()}`}
                    >
                      {charts.status_distribution[1].percentage > 3 ? `${charts.status_distribution[1].percentage}% Suspicious` : ''}
                    </div>
                  </div>

                  {/* Legend / Metrics */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ width: '12px', height: '12px', borderRadius: '3px', backgroundColor: '#047857' }} />
                      <span>Not Flagged: <strong>{charts.status_distribution[0].count.toLocaleString()}</strong> ({charts.status_distribution[0].percentage}%)</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ width: '12px', height: '12px', borderRadius: '3px', backgroundColor: '#B45309' }} />
                      <span>Suspicious: <strong>{charts.status_distribution[1].count.toLocaleString()}</strong> ({charts.status_distribution[1].percentage}%)</span>
                    </div>
                  </div>

                  {/* Top Highest Risk Preview */}
                  {charts.top_risk_transactions && charts.top_risk_transactions.length > 0 && (
                    <div style={{ marginTop: '8px' }}>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: '#64748B', marginBottom: '6px' }}>
                        Highest Risk Flagged Cases (Top Probability):
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {charts.top_risk_transactions.slice(0, 4).map((t) => (
                          <div
                            key={t.TransactionID}
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              fontSize: '12px',
                              padding: '4px 8px',
                              background: '#FFFBEB',
                              borderRadius: '4px',
                              border: '1px solid #FDE68A',
                            }}
                          >
                            <span style={{ fontWeight: 600, color: '#0F3D3E' }}>{t.TransactionID}</span>
                            <span>Amount: ₹{t.amount.toFixed(2)}</span>
                            <span style={{ fontWeight: 700, color: '#B45309' }}>
                              {(t.fraud_probability * 100).toFixed(2)}% Risk
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          </div>

          {/* Interactive Transaction Results Table */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid #E2E8F0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h3 className="card-title">Transaction-Level Prediction Results</h3>
                <span style={{ fontSize: '12px', color: '#64748B' }}>
                  Showing {resultsData ? resultsData.total_records.toLocaleString() : 0} matching transactions
                </span>
              </div>

              {/* Filters & Search */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                <input
                  type="text"
                  className="search-input"
                  style={{ width: '220px' }}
                  placeholder="Search Transaction ID..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setCurrentPage(1);
                  }}
                  aria-label="Search Transaction ID"
                />

                <select
                  className="search-input"
                  style={{ width: '160px' }}
                  value={statusFilter}
                  onChange={(e) => {
                    setStatusFilter(e.target.value);
                    setCurrentPage(1);
                  }}
                  aria-label="Filter status"
                >
                  <option value="all">All Statuses</option>
                  <option value="suspicious">Suspicious (≥ 0.144)</option>
                  <option value="not_flagged">Not Flagged (&lt; 0.144)</option>
                  <option value="fraud">Confirmed Fraud</option>
                  <option value="genuine">Confirmed Genuine</option>
                </select>

                <select
                  className="search-input"
                  style={{ width: '160px' }}
                  value={sortBy}
                  onChange={(e) => {
                    setSortBy(e.target.value);
                    setCurrentPage(1);
                  }}
                  aria-label="Sort by column"
                >
                  <option value="fraud_probability">Sort: Probability</option>
                  <option value="Amount">Sort: Amount</option>
                  <option value="Time">Sort: Time</option>
                  <option value="TransactionID">Sort: Transaction ID</option>
                </select>

                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => setSortOrder((prev) => (prev === 'desc' ? 'asc' : 'desc'))}
                  title="Toggle Ascending/Descending"
                >
                  {sortOrder === 'desc' ? '↓ Desc' : '↑ Asc'}
                </button>
              </div>
            </div>

            {/* Table Content */}
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Fraud Probability</th>
                    <th>Amount</th>
                    <th>Elapsed Time</th>
                    <th>Prediction Label</th>
                    <th>Assessment Status</th>
                    {evalMetrics && <th>Ground Truth</th>}
                    <th>Operator Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoadingResults ? (
                    <tr>
                      <td colSpan={evalMetrics ? 8 : 7} style={{ textAlign: 'center', padding: '32px' }}>
                        <div className="spinner" style={{ margin: '0 auto 8px' }} />
                        <span style={{ fontSize: '13px', color: '#64748B' }}>Loading transaction records...</span>
                      </td>
                    </tr>
                  ) : resultsData && resultsData.transactions.length > 0 ? (
                    resultsData.transactions.map((tx) => {
                      const isSuspicious = tx.prediction_label === 'SUSPICIOUS';
                      const isConfirmedFraud = tx.assessment_status === 'CONFIRMED_FRAUD';
                      const isConfirmedGenuine = tx.assessment_status === 'CONFIRMED_GENUINE';

                      return (
                        <tr key={tx.TransactionID}>
                          <td>
                            <strong style={{ color: '#0F3D3E' }}>{tx.TransactionID}</strong>
                          </td>
                          <td>
                            <div className="prob-cell">
                              <div className="prob-bar-track">
                                <div
                                  className={`prob-bar-fill ${tx.fraud_probability >= 0.144 ? 'prob-high' : ''}`}
                                  style={{ width: `${Math.min(tx.fraud_probability * 100, 100)}%` }}
                                />
                              </div>
                              <span className="prob-text">
                                {(tx.fraud_probability * 100).toFixed(2)}%
                              </span>
                            </div>
                          </td>
                          <td>{tx.amount !== null ? `₹${tx.amount.toFixed(2)}` : '—'}</td>
                          <td>{tx.time !== null ? `${tx.time}s` : '—'}</td>
                          <td>
                            <span className={isSuspicious ? 'badge badge-pending' : 'badge badge-genuine'}>
                              {tx.prediction_label}
                            </span>
                          </td>
                          <td>
                            {isConfirmedFraud ? (
                              <span className="badge badge-fraud">CONFIRMED FRAUD</span>
                            ) : isConfirmedGenuine ? (
                              <span className="badge badge-genuine">CONFIRMED GENUINE</span>
                            ) : isSuspicious ? (
                              <span className="badge badge-pending">PENDING REVIEW</span>
                            ) : (
                              <span className="badge" style={{ backgroundColor: '#F1F5F9', color: '#475569' }}>
                                NOT FLAGGED
                              </span>
                            )}
                          </td>
                          {evalMetrics && (
                            <td>
                              {tx.ground_truth_class !== undefined ? (
                                <span className={tx.ground_truth_class === 1 ? 'badge badge-fraud' : 'badge badge-genuine'}>
                                  {tx.ground_truth_class === 1 ? 'Actual Fraud (1)' : 'Genuine (0)'}
                                </span>
                              ) : (
                                '—'
                              )}
                            </td>
                          )}
                          <td>
                            {/* Operator inline decision actions */}
                            <div style={{ display: 'flex', gap: '6px' }}>
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() => setInspectedTransaction(tx)}
                                title="Inspect transaction features"
                              >
                                👁️ View
                              </button>
                              {isSuspicious && !isConfirmedFraud && !isConfirmedGenuine && (
                                <>
                                  <button
                                    type="button"
                                    className="btn btn-primary btn-sm"
                                    style={{ padding: '4px 8px', fontSize: '11px', minHeight: '30px' }}
                                    onClick={() => handleInlineLabel(tx.TransactionID, 'fraud')}
                                    disabled={isSubmittingLabel}
                                    title="Confirm as Fraudulent"
                                  >
                                    Fraud
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-secondary btn-sm"
                                    style={{ padding: '4px 8px', fontSize: '11px', minHeight: '30px', color: '#047857' }}
                                    onClick={() => handleInlineLabel(tx.TransactionID, 'genuine')}
                                    disabled={isSubmittingLabel}
                                    title="Dismiss as Genuine"
                                  >
                                    Dismiss
                                  </button>
                                </>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={evalMetrics ? 8 : 7} style={{ textAlign: 'center', padding: '32px', color: '#64748B' }}>
                        No transactions matched the current search or status filter.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Bar */}
            {resultsData && resultsData.total_pages > 1 && (
              <div className="pagination-bar">
                <div style={{ fontSize: '13px', color: '#64748B' }}>
                  Page {resultsData.page} of {resultsData.total_pages} ({resultsData.total_records.toLocaleString()} items)
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={currentPage <= 1 || isLoadingResults}
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  >
                    ← Previous
                  </button>
                  <span style={{ fontSize: '13px', fontWeight: 600 }}>{currentPage}</span>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={currentPage >= resultsData.total_pages || isLoadingResults}
                    onClick={() => setCurrentPage((p) => Math.min(resultsData.total_pages, p + 1))}
                  >
                    Next →
                  </button>
                  <select
                    className="search-input"
                    style={{ width: '80px', padding: '4px 8px', minHeight: '32px' }}
                    value={pageSize}
                    onChange={(e) => {
                      setPageSize(Number(e.target.value));
                      setCurrentPage(1);
                    }}
                  >
                    <option value={10}>10</option>
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                  </select>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Individual Transaction Details Modal */}
      {inspectedTransaction && (
        <div
          className="modal-overlay"
          onClick={() => setInspectedTransaction(null)}
          tabIndex={-1}
          role="dialog"
          aria-modal="true"
        >
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="card-header" style={{ padding: '16px 20px', borderBottom: '1px solid #E2E8F0' }}>
              <h3 className="card-title">Transaction Details: {inspectedTransaction.TransactionID}</h3>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setInspectedTransaction(null)}
              >
                ✕ Close
              </button>
            </div>
            <div style={{ padding: '20px', maxHeight: '70vh', overflowY: 'auto' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '16px' }}>
                <div><strong>Transaction ID:</strong> {inspectedTransaction.TransactionID}</div>
                <div><strong>Fraud Probability:</strong> {(inspectedTransaction.fraud_probability * 100).toFixed(4)}%</div>
                <div><strong>Amount:</strong> ₹{inspectedTransaction.amount !== null ? inspectedTransaction.amount.toFixed(2) : '—'}</div>
                <div><strong>Elapsed Time:</strong> {inspectedTransaction.time !== null ? `${inspectedTransaction.time}s` : '—'}</div>
                <div><strong>Selected Model:</strong> {inspectedTransaction.selected_model}</div>
                <div><strong>Selected Threshold:</strong> {inspectedTransaction.selected_threshold}</div>
                <div><strong>Prediction Label:</strong> {inspectedTransaction.prediction_label}</div>
                <div><strong>Assessment Status:</strong> {inspectedTransaction.assessment_status}</div>
                {inspectedTransaction.operator_label && (
                  <div><strong>Operator Decision:</strong> {inspectedTransaction.operator_label} (by {inspectedTransaction.operator})</div>
                )}
                {inspectedTransaction.ground_truth_class !== undefined && (
                  <div><strong>Ground Truth (Class):</strong> {inspectedTransaction.ground_truth_class === 1 ? 'Fraud (1)' : 'Genuine (0)'}</div>
                )}
              </div>

              {/* Operator Action from modal */}
              {inspectedTransaction.prediction_label === 'SUSPICIOUS' && (
                <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid #E2E8F0', display: 'flex', gap: '12px' }}>
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={async () => {
                      await handleInlineLabel(inspectedTransaction.TransactionID, 'fraud');
                      setInspectedTransaction(null);
                    }}
                    disabled={isSubmittingLabel}
                  >
                    Confirm as Fraud
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={async () => {
                      await handleInlineLabel(inspectedTransaction.TransactionID, 'genuine');
                      setInspectedTransaction(null);
                    }}
                    disabled={isSubmittingLabel}
                  >
                    Dismiss as Genuine
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
