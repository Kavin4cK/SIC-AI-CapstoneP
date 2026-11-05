/**
 * API client for Fraud Operator Dashboard
 */

const API_BASE = (typeof window !== 'undefined' && (window.location.port === '5173' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
  ? 'http://localhost:8000'
  : '/api';

export async function fetchTransactions(status = null, datasetId = null) {
  const params = new URLSearchParams();
  if (status && status !== 'all') params.append('status', status);
  if (datasetId) params.append('dataset_id', datasetId);

  const qs = params.toString();
  const url = qs ? `${API_BASE}/transactions?${qs}` : `${API_BASE}/transactions`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch transactions: ${res.statusText}`);
  return await res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  return await res.json();
}

export async function fetchFlagsPerHour(datasetId = null) {
  const url = datasetId
    ? `${API_BASE}/flags-per-hour?dataset_id=${encodeURIComponent(datasetId)}`
    : `${API_BASE}/flags-per-hour`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch flags per hour: ${res.statusText}`);
  return await res.json();
}

export async function fetchLabels(datasetId = null) {
  const url = datasetId
    ? `${API_BASE}/labels?dataset_id=${encodeURIComponent(datasetId)}`
    : `${API_BASE}/labels`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch labels: ${res.statusText}`);
  return await res.json();
}

export async function submitLabel(transactionId, label, operator = 'operator_1', datasetId = null) {
  const res = await fetch(`${API_BASE}/labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      TransactionID: transactionId,
      label,
      operator,
      dataset_id: datasetId || 'default',
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to submit label: ${res.statusText}`);
  }
  return await res.json();
}

export async function fetchAvailableDatasets() {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) throw new Error(`Failed to fetch datasets: ${res.statusText}`);
  return await res.json();
}

export function getDownloadLabelsUrl(datasetId = null) {
  return datasetId
    ? `${API_BASE}/labels/export.csv?dataset_id=${encodeURIComponent(datasetId)}`
    : `${API_BASE}/labels/export.csv`;
}

export async function triggerSeed() {
  const res = await fetch(`${API_BASE}/seed`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to seed data: ${res.statusText}`);
  return await res.json();
}

// =============================================================================
// DATASET UPLOAD & FRAUD DETECTION APIS
// =============================================================================

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/datasets/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${res.statusText}`);
  }
  return await res.json();
}

export async function useExistingDataset() {
  const res = await fetch(`${API_BASE}/datasets/use-existing`, {
    method: 'POST',
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to load existing dataset: ${res.statusText}`);
  }
  return await res.json();
}

export async function runDatasetPrediction(datasetId, threshold = 0.144) {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/predict?threshold=${threshold}`, {
    method: 'POST',
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Prediction failed: ${res.statusText}`);
  }
  return await res.json();
}

export async function fetchDatasetResults(datasetId, {
  page = 1,
  pageSize = 25,
  status = '',
  search = '',
  sortBy = 'fraud_probability',
  sortOrder = 'desc',
} = {}) {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
    sort_by: sortBy,
    sort_order: sortOrder,
  });
  if (status && status !== 'all') params.append('status', status);
  if (search && search.trim()) params.append('search', search.trim());

  const res = await fetch(`${API_BASE}/datasets/${datasetId}/results?${params.toString()}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch results: ${res.statusText}`);
  }
  return await res.json();
}

export function getDownloadResultsUrl(datasetId) {
  return `${API_BASE}/datasets/${datasetId}/download`;
}

export function getDownloadTemplateUrl() {
  return `${API_BASE}/datasets/template.csv`;
}
