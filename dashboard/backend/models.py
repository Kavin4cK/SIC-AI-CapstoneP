"""Pydantic data models for the Operator Dashboard API."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    TransactionID: str
    fraud_probability: float
    selected_threshold: float = 0.144
    selected_model: str = "CatBoost"
    assessment_status: str  # SUSPICIOUS_PENDING_REVIEW, CONFIRMED_FRAUD, CONFIRMED_GENUINE
    amount: Optional[float] = None
    time: Optional[float] = None
    operator_label: Optional[str] = None  # fraud | genuine
    operator: Optional[str] = None
    labelled_at: Optional[str] = None


class LabelRequest(BaseModel):
    TransactionID: str
    label: str = Field(..., description="'fraud' or 'genuine'")
    operator: Optional[str] = "operator_1"


class LabelRecord(BaseModel):
    TransactionID: str
    label: str
    operator: str
    created_at: str


class FlagPerHour(BaseModel):
    hour: int
    flags: int


class MetricSummary(BaseModel):
    model_name: str
    decision_threshold: float
    recall: float
    precision: float
    accuracy: float
    f1_score: float
    pr_auc: float
    roc_auc: float
    total_hypothetical_cost: float
    average_cost_per_tx: float
    confusion_matrix: Dict[str, int]
    cost_matrix: Dict[str, Any]
    source_file: str
    raw_metrics: Dict[str, Any]
