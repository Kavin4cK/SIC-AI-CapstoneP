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
    dataset_id: Optional[str] = "default"


class LabelRequest(BaseModel):
    TransactionID: str
    label: str = Field(..., description="'fraud' or 'genuine'")
    operator: Optional[str] = "operator_1"
    dataset_id: Optional[str] = "default"


class LabelRecord(BaseModel):
    TransactionID: str
    label: str
    operator: str
    created_at: str
    dataset_id: str = "default"


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


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    total_rows: int
    columns: List[str]
    has_class: bool
    has_transaction_id: bool
    preview_rows: List[Dict[str, Any]]
    validation_status: str = "VALID"
    message: str


class PredictionSummary(BaseModel):
    dataset_id: str
    filename: str
    total_transactions: int
    suspicious_count: int
    not_flagged_count: int
    suspicious_percentage: float
    threshold: float = 0.144
    model_name: str = "CatBoost"
    has_ground_truth: bool = False


class PredictionResponse(BaseModel):
    dataset_id: str
    summary: PredictionSummary
    charts: Dict[str, Any]
    evaluation_metrics: Optional[Dict[str, Any]] = None


class DatasetResultsResponse(BaseModel):
    dataset_id: str
    summary: Optional[Dict[str, Any]] = None
    charts: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    page: int
    page_size: int
    total_pages: int
    total_records: int
    transactions: List[Dict[str, Any]]
