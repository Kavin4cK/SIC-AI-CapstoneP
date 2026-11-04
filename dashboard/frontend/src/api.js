/**
 * API client for Fraud Operator Dashboard
 */

const API_BASE = (typeof window !== 'undefined' && (window.location.port === '5173' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
  ? 'http://localhost:8000'
  : '/api';

export async function fetchTransactions(status = null) {
  const url = status && status !== 'all' 
    ? `${API_BASE}/transactions?status=${encodeURIComponent(status)}`
    : `${API_BASE}/transactions`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch transactions: ${res.statusText}`);
  return await res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  return await res.json();
}

export async function fetchFlagsPerHour() {
  const res = await fetch(`${API_BASE}/flags-per-hour`);
  if (!res.ok) throw new Error(`Failed to fetch flags per hour: ${res.statusText}`);
  return await res.json();
}

export async function fetchLabels() {
  const res = await fetch(`${API_BASE}/labels`);
  if (!res.ok) throw new Error(`Failed to fetch labels: ${res.statusText}`);
  return await res.json();
}

export async function submitLabel(transactionId, label, operator = 'operator_1') {
  const res = await fetch(`${API_BASE}/labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      TransactionID: transactionId,
      label,
      operator,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to submit label: ${res.statusText}`);
  }
  return await res.json();
}

export async function triggerSeed() {
  const res = await fetch(`${API_BASE}/seed`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to seed data: ${res.statusText}`);
  return await res.json();
}
