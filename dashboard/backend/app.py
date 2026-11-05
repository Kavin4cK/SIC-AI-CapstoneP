"""FastAPI backend for Fraud Detection Operator Dashboard.
Endpoints:
- GET /transactions (filter by status, sorted by probability, isolated by dataset_id)
- GET /metrics
- GET /flags-per-hour (isolated by dataset_id)
- POST /labels {TransactionID, label: fraud|genuine, operator, dataset_id}
- GET /labels (isolated by dataset_id)
- GET /labels/export.csv
- GET /datasets (list all active/available datasets)
- POST /datasets/upload (custom CSV upload & validation)
- POST /datasets/use-existing (Option A: original ULB benchmark)
- POST /datasets/{dataset_id}/predict (CatBoost inference at t=0.144)
- GET /datasets/{dataset_id}/results (paginated, filtered, with charts & evaluation)
- GET /datasets/{dataset_id}/download (full results CSV download)
- GET /datasets/template.csv (downloadable 30-feature CSV template)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from dashboard.backend.models import (
    Transaction,
    LabelRequest,
    LabelRecord,
    FlagPerHour,
    MetricSummary,
    DatasetUploadResponse,
    PredictionResponse,
    DatasetResultsResponse,
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
from dashboard.backend.dataset_service import (
    DATASET_SERVICE,
    DatasetValidationException,
    FROZEN_THRESHOLD,
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
    version="1.2.0",
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
    ),
    dataset_id: Optional[str] = Query(
        None,
        description="Optional dataset ID to fetch transactions from an active/evaluated dataset",
    ),
):
    """Retrieve transactions sorted by fraud_probability descending, with Amount/Time joined
    and operator labels applied, strictly isolated to the specified dataset_id.
    """
    clean_did = (dataset_id or "default").strip()
    labels = get_label_map(dataset_id=clean_did)

    if clean_did == "default":
        all_txns = load_transactions(label_map=labels)
        for tx in all_txns:
            tx["dataset_id"] = "default"
    else:
        try:
            record = DATASET_SERVICE.get_dataset(clean_did)
            if not record.is_predicted or record.results_df is None:
                DATASET_SERVICE.predict_dataset(clean_did)
                record = DATASET_SERVICE.get_dataset(clean_did)

            all_txns: List[Dict[str, Any]] = []
            for _, row in record.results_df.iterrows():
                tid = str(row["TransactionID"])
                prob = float(row.get("fraud_probability", 0.0))
                th = float(row.get("selected_threshold", FROZEN_THRESHOLD))
                model = str(row.get("selected_model", "CatBoost"))
                st = str(row.get("assessment_status", "NOT_FLAGGED"))

                op_info = labels.get(tid)
                op_label = None
                op_name = None
                lbl_at = None
                if op_info:
                    op_label = op_info["label"]
                    op_name = op_info.get("operator")
                    lbl_at = op_info.get("created_at")
                    st = "CONFIRMED_FRAUD" if op_label == "fraud" else "CONFIRMED_GENUINE"

                # Review Queue and triage strictly display only suspicious transactions (>= 0.144)
                # or transactions with operator feedback. Sub-threshold transactions are excluded.
                if prob < th and not op_info:
                    continue

                all_txns.append({
                    "dataset_id": clean_did,
                    "TransactionID": tid,
                    "fraud_probability": round(prob, 6),
                    "selected_threshold": round(th, 4),
                    "selected_model": model,
                    "assessment_status": st,
                    "amount": round(float(row["Amount"]), 2) if pd.notna(row.get("Amount")) else None,
                    "time": round(float(row["Time"]), 1) if pd.notna(row.get("Time")) else None,
                    "operator_label": op_label,
                    "operator": op_name,
                    "labelled_at": lbl_at,
                })
            all_txns.sort(key=lambda x: x["fraud_probability"], reverse=True)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Dataset '{clean_did}' not found.")

    if not status or status.strip().lower() in ("all", ""):
        return all_txns

    clean_status = status.strip().lower()
    filtered: List[Dict[str, Any]] = []

    for tx in all_txns:
        tx_status = tx["assessment_status"].upper()
        if clean_status in ("pending", "suspicious_pending_review"):
            # Exclude anything below threshold (must be >= 0.144 and pending review)
            if tx_status == "SUSPICIOUS_PENDING_REVIEW" and tx["fraud_probability"] >= FROZEN_THRESHOLD:
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
def get_flags_per_hour(
    dataset_id: Optional[str] = Query(None, description="Dataset ID for hourly flag distribution"),
):
    """Return hourly flagged transactions distribution across elapsed time for the specified dataset."""
    clean_did = (dataset_id or "default").strip()
    txns = get_transactions(status="all", dataset_id=clean_did)
    hourly = get_flags_per_hour_data(txns)
    return hourly


@app.post("/labels", response_model=Dict[str, Any])
def submit_label(payload: LabelRequest):
    """Record an operator review decision in the separate SQLite labels store.
    Guarantees isolation by dataset_id and never mutates predictions files.
    """
    label_norm = payload.label.strip().lower()
    if label_norm not in ("fraud", "genuine"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label '{payload.label}'. Must be 'fraud' or 'genuine'.",
        )

    clean_dataset_id = (payload.dataset_id or "default").strip() or "default"
    try:
        record = save_label(
            transaction_id=payload.TransactionID,
            label=label_norm,
            operator=payload.operator,
            dataset_id=clean_dataset_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "success": True,
        "message": f"Transaction {payload.TransactionID} in dataset '{clean_dataset_id}' marked as {label_norm}.",
        "data": record,
    }


@app.get("/labels", response_model=List[LabelRecord])
def get_labels(
    dataset_id: Optional[str] = Query(None, description="Filter labels by dataset_id"),
):
    """Retrieve recorded operator feedback decisions, isolated by dataset_id."""
    labels = get_all_labels(dataset_id=dataset_id)
    return labels


@app.get("/labels/export.csv", response_class=PlainTextResponse)
def export_labels_csv(
    dataset_id: Optional[str] = Query(None, description="Filter exported labels by dataset_id"),
):
    """Export operator decisions as downloadable CSV."""
    labels = get_all_labels(dataset_id=dataset_id)
    lines = ["dataset_id,TransactionID,label,operator,created_at"]
    for item in labels:
        lines.append(f"{item.get('dataset_id', 'default')},{item['TransactionID']},{item['label']},{item['operator']},{item['created_at']}")
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


# =============================================================================
# DATASET UPLOAD & FRAUD DETECTION ENDPOINTS
# =============================================================================

@app.get("/datasets")
@app.get("/api/datasets")
def list_datasets():
    """List all available datasets in the current session."""
    return DATASET_SERVICE.list_datasets()


async def _process_upload(file: UploadFile) -> DatasetUploadResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a CSV file with '.csv' extension.",
        )

    content = await file.read()
    try:
        record = DATASET_SERVICE.register_uploaded_dataset(
            file_bytes=content,
            filename=file.filename,
        )
    except DatasetValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")

    preview = record.df_original.head(5).to_dict(orient="records")

    return DatasetUploadResponse(
        dataset_id=record.dataset_id,
        filename=record.filename,
        total_rows=record.total_rows,
        columns=record.columns,
        has_class=record.has_class,
        has_transaction_id=record.has_transaction_id,
        preview_rows=preview,
        validation_status="VALID",
        message="Dataset validated successfully. All 30 required model features are present and numerical.",
    )


@app.post("/datasets/upload", response_model=DatasetUploadResponse)
@app.post("/api/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and strictly validate a credit card transaction CSV dataset."""
    return await _process_upload(file)


def _load_ulb() -> DatasetUploadResponse:
    try:
        record = DATASET_SERVICE.load_existing_ulb_dataset()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ULB dataset: {str(e)}")

    preview = record.df_original.head(5).to_dict(orient="records")
    return DatasetUploadResponse(
        dataset_id=record.dataset_id,
        filename=record.filename,
        total_rows=record.total_rows,
        columns=record.columns,
        has_class=record.has_class,
        has_transaction_id=record.has_transaction_id,
        preview_rows=preview,
        validation_status="VALID",
        message="Loaded original ULB benchmark dataset (284,807 transactions, 30 features + Class ground truth).",
    )


@app.post("/datasets/use-existing", response_model=DatasetUploadResponse)
@app.post("/api/datasets/use-existing", response_model=DatasetUploadResponse)
def use_existing_dataset():
    """Option A: Select the original ULB Credit Card Fraud Detection dataset."""
    return _load_ulb()


def _run_prediction(dataset_id: str, threshold: float = FROZEN_THRESHOLD) -> PredictionResponse:
    try:
        record = DATASET_SERVICE.predict_dataset(dataset_id=dataset_id, threshold=threshold)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return PredictionResponse(
        dataset_id=record.dataset_id,
        summary=record.summary_stats,
        charts=record.charts_data,
        evaluation_metrics=record.evaluation_metrics,
    )


@app.post("/datasets/{dataset_id}/predict", response_model=PredictionResponse)
@app.post("/api/datasets/{dataset_id}/predict", response_model=PredictionResponse)
def predict_dataset(
    dataset_id: str,
    threshold: float = Query(FROZEN_THRESHOLD, description="Decision threshold (frozen 0.144)"),
):
    """Run CatBoost fraud inference on all transactions in the specified dataset."""
    return _run_prediction(dataset_id=dataset_id, threshold=threshold)


def _get_results(
    dataset_id: str,
    page: int = 1,
    page_size: int = 25,
    status: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "fraud_probability",
    sort_order: str = "desc",
) -> DatasetResultsResponse:
    try:
        res = DATASET_SERVICE.get_results_page(
            dataset_id=dataset_id,
            page=page,
            page_size=page_size,
            status_filter=status,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return DatasetResultsResponse(**res)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")


@app.get("/datasets/{dataset_id}/results", response_model=DatasetResultsResponse)
@app.get("/api/datasets/{dataset_id}/results", response_model=DatasetResultsResponse)
def get_dataset_results(
    dataset_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter: all, suspicious, not_flagged, fraud, genuine"),
    search: Optional[str] = Query(None, description="Search by TransactionID"),
    sort_by: str = Query("fraud_probability", description="Column to sort by"),
    sort_order: str = Query("desc", description="desc or asc"),
):
    """Fetch paginated, filtered, and sorted prediction results with operator review overlays."""
    return _get_results(
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
        status=status,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
    )


def _download_results(dataset_id: str) -> Response:
    try:
        csv_content = DATASET_SERVICE.generate_results_csv(dataset_id)
        filename = f"predictions_{dataset_id}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating CSV export: {str(e)}")


@app.get("/datasets/{dataset_id}/download")
@app.get("/api/datasets/{dataset_id}/download")
def download_results(dataset_id: str):
    """Download the complete prediction results CSV containing all original columns plus predictions."""
    return _download_results(dataset_id)


def _download_template() -> Response:
    csv_content = DATASET_SERVICE.generate_template_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=credit_card_transactions_template.csv"},
    )


@app.get("/datasets/template.csv")
@app.get("/api/datasets/template.csv")
def download_template():
    """Download sample CSV template containing all 30 required model features."""
    return _download_template()
