# Fraud Detection Operator Dashboard

A dedicated human-in-the-loop operator review dashboard for the **Cost-Sensitive Credit Card Fraud Detection System** (CatBoost Classifier, frozen optimal decision threshold $t^* = 0.144$).

---

> [!IMPORTANT]
> ### Academic Benchmark & Governance Notice
> This dashboard and its underlying machine learning pipeline operate strictly as an **academic and methodological benchmark** for cost-sensitive learning and ML-to-LLM / human operator handoff. It is **not** an online core banking authorization engine.
>
> 1. **Hypothetical Cost Matrix**: Cost parameters ($C_{\text{FP}} = \text{₹5.00}$ per manual review alert, $C_{\text{FN}} = \text{₹500.00}$ per missed fraud) are hypothetical research assumptions designed to evaluate asymmetric loss minimization (100:1 ratio). They do not reflect proprietary banking cost ledgers or real institutional chargebacks.
> 2. **Alert Interpretation**: Transactions flagged by the model are designated as `SUSPICIOUS_PENDING_REVIEW` (statistical anomaly alerts). Under no circumstances are alerts automatically classified as confirmed fraud without operator adjudication.
> 3. **Production Banking Boundaries**: Real-world payment authorization requires streaming sub-50ms latency SLAs, cardholder behavior profiles, device fingerprinting, fraud velocity counters, IP reputation, and integration with card clearing networks (ISO 8583/20022), which are absent from this anonymized tabular PCA dataset.

---

## 1. Quick Start (One Command)

You can launch the full system (FastAPI backend + Vite React frontend) with a single command:

```bash
# Using Make
make dev

# Or directly using the runner script
./run.sh
```

- **Frontend Application**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 2. System Architecture

The dashboard is organized cleanly into decoupled backend and frontend packages:

```
dashboard/
├── backend/
│   ├── app.py              # FastAPI endpoints, CORS middleware, startup lifecycle
│   ├── data_loader.py      # Dynamic metrics parser, Amount/Time feature joining, seed mode
│   ├── database.py         # SQLite persistence layer for operator feedback labels
│   ├── models.py           # Pydantic schemas for request/response validation
│   └── requirements.txt    # Pinned backend dependencies
├── frontend/
│   ├── index.html          # Fraunces & Inter typography setup
│   ├── vite.config.js      # Vite build configuration with API reverse proxy
│   ├── package.json        # React + Vite dependencies
│   └── src/
│       ├── App.jsx         # Root dashboard controller with optimistic updates & toast
│       ├── api.js          # REST client wrapper
│       ├── index.css       # Option B (Teal overview) responsive CSS design
│       └── components/
│           ├── Sidebar.jsx        # Deep teal #0F3D3E navigation with live count badge
│           ├── Header.jsx         # Section titles & refresh controls
│           ├── Overview.jsx       # 3 KPI cards, Flags per hour bar chart, Needs Attention
│           ├── ReviewQueue.jsx    # Split table & detail panel with score-vs-threshold meter
│           ├── LabelledCases.jsx  # Operator audit table with client CSV export
│           ├── ModelThreshold.jsx # Test metrics (Recall, Precision, Cost) & Confusion Matrix
│           ├── Retraining.jsx     # Feedback dataset tracker & ML team handoff workflow
│           └── Toast.jsx          # Accessible floating action toasts
├── data/
│   └── labels.sqlite3      # Isolated SQLite store for operator labels (never overwrites predictions)
├── tests/
│   ├── conftest.py         # Pytest configuration & sys.path setup
│   └── test_api.py         # API test suite (transactions, metrics, labels, immutability)
└── README.md               # Dashboard documentation
```

---

## 3. Backend REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/transactions` | List scored transactions sorted by `fraud_probability` descending. Supports optional `?status=pending\|fraud\|genuine\|all` filter. Features `Amount` and `Time` are dynamically joined. |
| `GET` | `/metrics` | Read verified out-of-sample test and validation metrics dynamically from `outputs/evaluation/` and `README.md`. Never hardcoded. |
| `GET` | `/flags-per-hour` | Hourly distribution of flagged alerts across the 48-hour elapsed time window. |
| `POST` | `/labels` | Record operator adjudication (`label: "fraud" \| "genuine"`). Persists exclusively to SQLite; strictly never mutates input predictions. |
| `GET` | `/labels` | Retrieve all recorded operator review feedback records ordered by timestamp descending. |
| `GET` | `/labels/export.csv` | Download all operator feedback decisions as a CSV file. |
| `POST` | `/seed` | Idempotently trigger inference on `outputs/predictions/sample_input.csv` via `src/predict.py` if predictions are missing. |
| `GET` | `/health` | Service health status and active predictions file path. |

---

## 4. Key Design Specifications (Option B: Teal Overview)

- **Color Palette**:
  - Sidebar: Deep Teal `#0F3D3E` with lighter active selection `#1B5254` and amber accents.
  - Content Canvas: Soft Pale Teal `#EEF4F2`.
  - Cards: Pure White `#FFFFFF`, 12px corner radius, 4px top border (`#B45309` Amber for Flagged items, `#0F3D3E` Teal for standard cards).
  - Primary Action Button: High-visibility Amber `#B45309`, white text, 44px minimum touch height.
- **Typography**:
  - Headings & KPI Numbers: `Fraunces` serif.
  - Body & Tabular Content: `Inter` sans-serif.
- **Accessibility**:
  - Native semantic `<button>` elements across all interactive controls.
  - Visible focus indicators (`:focus-visible` outline in amber) for complete keyboard navigation.
  - ARIA attributes on dynamic panels, alerts, and live regions.
- **Operator Workflow**:
  - **Optimistic State Updates**: Triage decisions immediately update UI status to eliminate perceived latency, confirmed by floating success toasts. If a backend write fails, changes rollback automatically.
  - **Zero Data Leakage**: Ground-truth target labels are excluded from inference inputs and review screens.

---

## 5. Clean Seed / Demo Mode

If the prediction output (`outputs/predictions/flagged_transactions.csv`) does not exist on disk, the system automatically detects the absence and executes:

```bash
python src/predict.py \
  --input outputs/predictions/sample_input.csv \
  --output outputs/predictions/flagged_transactions.csv \
  --threshold 0.144
```

This guarantees an immediate, working out-of-the-box demo with realistic transactions, full probability bars, Amount, and Time features.

---

## 6. Running Tests

Run the backend test suite:

```bash
# Run dashboard API tests
make test

# Or directly with pytest
PYTHONPATH=. .venv/bin/pytest dashboard/tests/test_api.py

# Run all repository tests (ML pipeline + Dashboard API)
make test-all
```

