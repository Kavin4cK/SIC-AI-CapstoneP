import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Overview from './components/Overview';
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
  submitLabel,
} from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
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

  // Load all initial data from backend
  const loadDashboardData = useCallback(async (isSilent = false) => {
    if (!isSilent) setIsLoading(true);
    setIsRefreshing(true);
    try {
      const [txnsData, metricsData, hourlyData, labelsData] = await Promise.all([
        fetchTransactions(),
        fetchMetrics(),
        fetchFlagsPerHour(),
        fetchLabels(),
      ]);
      setTransactions(txnsData || []);
      setMetrics(metricsData || null);
      setFlagsPerHour(hourlyData || []);
      setLabels(labelsData || []);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      addToast(err.message || 'Failed to connect to backend API', 'error');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [addToast]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Optimistic Operator Labeling
  const handleApplyLabel = async (transactionId, label) => {
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
      const filtered = prev.filter((l) => l.TransactionID !== transactionId);
      return [
        {
          TransactionID: transactionId,
          label: cleanLabel,
          operator: 'operator_1',
          created_at: new Date().toISOString(),
        },
        ...filtered,
      ];
    });

    addToast(`Transaction ${transactionId} confirmed as ${cleanLabel}.`, 'success');

    // 2. Perform API call in background
    setIsSubmitting(true);
    try {
      await submitLabel(transactionId, cleanLabel, 'operator_1');
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
          onRefresh={() => loadDashboardData(true)}
          isRefreshing={isRefreshing}
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
            <LabelledCases labels={labels} />
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

