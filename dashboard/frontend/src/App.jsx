import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Overview from './components/Overview';
import DatasetUpload from './components/DatasetUpload';
import ReviewQueue from './components/ReviewQueue';
import LabelledCases from './components/LabelledCases';
import ModelThreshold from './components/ModelThreshold';
import Retraining from './components/Retraining';
import Toast from './components/Toast';
import {
  fetchTransactions,
  fetchMetrics,
  fetchFlagsPerHour,
  fetchLabels,
  fetchAvailableDatasets,
  submitLabel,
} from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [activeDatasetId, setActiveDatasetId] = useState(() => {
    try {
      return localStorage.getItem('sic_active_dataset_id') || 'default';
    } catch {
      return 'default';
    }
  });
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [flagsPerHour, setFlagsPerHour] = useState([]);
  const [labels, setLabels] = useState([]);
  const [selectedTransactionId, setSelectedTransactionId] = useState(null);

  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toasts, setToasts] = useState([]);

  // Toast notification helper
  const addToast = useCallback((message, type = 'success') => {
    const id = Date.now() + Math.random().toString();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Handle dataset switching with persistent localStorage context
  const handleSelectDataset = useCallback((newDatasetId) => {
    if (!newDatasetId) return;
    setActiveDatasetId(newDatasetId);
    try {
      localStorage.setItem('sic_active_dataset_id', newDatasetId);
    } catch (err) {
      console.error('Failed to save active dataset to localStorage:', err);
    }
    setSelectedTransactionId(null);
  }, []);

  // Load all dashboard data isolated to the active dataset
  const loadDashboardData = useCallback(async (isSilent = false, datasetId = activeDatasetId) => {
    if (!isSilent) setIsLoading(true);
    setIsRefreshing(true);
    try {
      const [txnsData, metricsData, hourlyData, labelsData, datasetsData] = await Promise.all([
        fetchTransactions(null, datasetId),
        fetchMetrics(),
        fetchFlagsPerHour(datasetId),
        fetchLabels(datasetId),
        fetchAvailableDatasets().catch(() => []),
      ]);
      setTransactions(txnsData || []);
      setMetrics(metricsData || null);
      setFlagsPerHour(hourlyData || []);
      setLabels(labelsData || []);
      setAvailableDatasets(datasetsData || []);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      addToast(err.message || 'Failed to connect to backend API', 'error');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [activeDatasetId, addToast]);

  // Reload dashboard data whenever active dataset context changes
  useEffect(() => {
    loadDashboardData(false, activeDatasetId);
  }, [activeDatasetId]);

  // Optimistic Operator Labeling with dataset context
  const handleApplyLabel = async (transactionId, label, targetDatasetId = activeDatasetId) => {
    if (isSubmitting) return;

    // Snapshot current state for rollback if needed
    const prevTransactions = [...transactions];
    const prevLabels = [...labels];

    const cleanLabel = label.toLowerCase();
    const newStatus = cleanLabel === 'fraud' ? 'CONFIRMED_FRAUD' : 'CONFIRMED_GENUINE';

    // 1. Optimistic Update in UI state
    setTransactions((prev) =>
      prev.map((t) =>
        t.TransactionID === transactionId
          ? {
              ...t,
              assessment_status: newStatus,
              operator_label: cleanLabel,
              operator: 'operator_1',
              labelled_at: new Date().toISOString(),
            }
          : t
      )
    );

    setLabels((prev) => {
      const filtered = prev.filter(
        (l) => !(l.TransactionID === transactionId && (l.dataset_id || activeDatasetId) === targetDatasetId)
      );
      return [
        {
          dataset_id: targetDatasetId,
          TransactionID: transactionId,
          label: cleanLabel,
          operator: 'operator_1',
          created_at: new Date().toISOString(),
        },
        ...filtered,
      ];
    });

    addToast(`Transaction ${transactionId} confirmed as ${cleanLabel}.`, 'success');

    // 2. Perform API call in background with target dataset_id
    setIsSubmitting(true);
    try {
      await submitLabel(transactionId, cleanLabel, 'operator_1', targetDatasetId);
    } catch (err) {
      console.error('Failed to submit label to backend:', err);
      // Rollback optimistic update
      setTransactions(prevTransactions);
      setLabels(prevLabels);
      addToast(`Error saving decision for ${transactionId}: ${err.message}`, 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  const pendingCount = transactions.filter(
    (t) => t.assessment_status === 'SUSPICIOUS_PENDING_REVIEW'
  ).length;

  return (
    <div className="app-container">
      {/* Deep Teal Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        pendingCount={pendingCount}
      />

      {/* Main App Content */}
      <div className="main-wrapper">
        <Header
          activeTab={activeTab}
          onRefresh={() => loadDashboardData(true, activeDatasetId)}
          isRefreshing={isRefreshing}
          activeDatasetId={activeDatasetId}
          onSelectDataset={handleSelectDataset}
          availableDatasets={availableDatasets}
        />

        <main id="main-content">
          {activeTab === 'overview' && (
            <Overview
              transactions={transactions}
              metrics={metrics}
              flagsPerHour={flagsPerHour}
              onNavigateToReview={() => setActiveTab('review')}
              onSelectTransactionForReview={(id) => {
                setSelectedTransactionId(id);
                setActiveTab('review');
              }}
            />
          )}

          {activeTab === 'dataset-upload' && (
            <DatasetUpload
              onApplyLabel={handleApplyLabel}
              isSubmittingLabel={isSubmitting}
              activeDatasetId={activeDatasetId}
              onSelectDataset={handleSelectDataset}
            />
          )}

          {activeTab === 'review' && (
            <ReviewQueue
              transactions={transactions}
              selectedId={selectedTransactionId}
              onSelectTransaction={setSelectedTransactionId}
              onApplyLabel={handleApplyLabel}
              isSubmitting={isSubmitting}
            />
          )}

          {activeTab === 'labelled' && (
            <LabelledCases
              labels={labels}
              activeDatasetId={activeDatasetId}
            />
          )}

          {activeTab === 'model' && (
            <ModelThreshold metrics={metrics} />
          )}

          {activeTab === 'retraining' && (
            <Retraining labels={labels} />
          )}
        </main>
      </div>

      {/* Floating Action Notifications */}
      <Toast toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
