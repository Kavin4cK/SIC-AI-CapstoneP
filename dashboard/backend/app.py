"""FastAPI backend for Fraud Detection Operator Dashboard.
Endpoints:
- GET /transactions (filter by status, sorted by probability)
- GET /metrics
- GET /flags-per-hour
- POST /labels {TransactionID, label: fraud|genuine, operator}
- GET /labels
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from dashboard.backend.models import (
    Transaction,
    LabelRequest,
    LabelRecord,
    FlagPerHour,
    MetricSummary,
)
from dashboard.backend.database import (
    init_db,
    save_label,
    get_all_labels,
    get_label_map,
)
from dashboard.backend.data_loader import (
    load_transactions,
    load_real_metrics,
    get_flags_per_hour_data,
    run_seed_mode_if_needed,
    get_active_predictions_file,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure DB exists and seed predictions if needed
    init_db()
    run_seed_mode_if_needed()
    yield


app = FastAPI(
    title="Fraud Operator Dashboard API",
    description="Operational review API for cost-sensitive fraud detection module (CatBoost, threshold 0.144)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    active_pred = get_active_predictions_file()
    return {
        "status": "healthy",
        "service": "Fraud Operator Dashboard API",
        "predictions_source": str(active_pred) if active_pred else "None",
    }


@app.get("/transactions", response_model=List[Transaction])
def get_transactions(
    status: Optional[str] = Query(
        None,
        description="Filter status: 'all', 'pending', 'fraud', 'genuine', or exact status string",
    )
):
    """Retrieve transactions sorted by fraud_probability descending, with Amount/Time joined
    and operator labels applied.
    """
    labels = get_label_map()
    all_txns = load_transactions(label_map=labels)

    if not status or status.strip().lower() in ("all", ""):
        return all_txns

    clean_status = status.strip().lower()
    filtered: List[Dict[str, Any]] = []

    for tx in all_txns:
        tx_status = tx["assessment_status"].upper()
        if clean_status in ("pending", "suspicious_pending_review"):
            if tx_status == "SUSPICIOUS_PENDING_REVIEW":
                filtered.append(tx)
        elif clean_status in ("fraud", "confirmed_fraud"):
            if tx_status == "CONFIRMED_FRAUD":
                filtered.append(tx)
        elif clean_status in ("genuine", "confirmed_genuine"):
            if tx_status == "CONFIRMED_GENUINE":
                filtered.append(tx)
        else:
            if tx_status == status.strip().upper():
                filtered.append(tx)

    return filtered


@app.get("/metrics", response_model=MetricSummary)
def get_metrics():
    """Dynamically read and return verified performance and cost metrics from outputs/ and README.md."""
    metrics = load_real_metrics()
    return metrics


@app.get("/flags-per-hour", response_model=List[FlagPerHour])
def get_flags_per_hour():
    """Return hourly flagged transactions distribution across elapsed time."""
    labels = get_label_map()
    txns = load_transactions(label_map=labels)
    hourly = get_flags_per_hour_data(txns)
    return hourly


@app.post("/labels", response_model=Dict[str, Any])
def submit_label(payload: LabelRequest):
    """Record an operator review decision in the separate SQLite labels store.
    Guarantees no overwrite of the predictions file.
    """
    label_norm = payload.label.strip().lower()
    if label_norm not in ("fraud", "genuine"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label '{payload.label}'. Must be 'fraud' or 'genuine'.",
        )

    try:
        record = save_label(
            transaction_id=payload.TransactionID,
            label=label_norm,
            operator=payload.operator,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "success": True,
        "message": f"Transaction {payload.TransactionID} marked as {label_norm}.",
        "data": record,
    }


@app.get("/labels", response_model=List[LabelRecord])
def get_labels():
    """Retrieve all recorded operator feedback decisions."""
    labels = get_all_labels()
    return labels


@app.get("/labels/export.csv", response_class=PlainTextResponse)
def export_labels_csv():
    """Export operator decisions as downloadable CSV."""
    labels = get_all_labels()
    lines = ["TransactionID,label,operator,created_at"]
    for item in labels:
        lines.append(f"{item['TransactionID']},{item['label']},{item['operator']},{item['created_at']}")
    csv_content = "\n".join(lines)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=operator_labels.csv"},
    )


@app.post("/seed")
def seed_demo_data():
    """Explicitly trigger seed mode to ensure sample predictions are available."""
    did_seed = run_seed_mode_if_needed()
    return {"seeded": did_seed, "message": "Predictions checked/seeded."}
