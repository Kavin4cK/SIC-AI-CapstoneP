#!/usr/bin/env bash
# ==============================================================================
# Fraud Operator Dashboard — Unified Dev Runner
# Runs FastAPI backend (:8000) and Vite React frontend (:5173) concurrently.
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo " Starting Fraud Operator Dashboard (Benchmark CatBoost 0.144)"
echo "======================================================================"

# 1. Ensure Python Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[*] Creating Python virtual environment in .venv..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r dashboard/backend/requirements.txt catboost scikit-learn
fi

# 2. Ensure Frontend Node Dependencies
if [ ! -d "dashboard/frontend/node_modules" ]; then
    echo "[*] Installing frontend npm dependencies..."
    (cd dashboard/frontend && npm install)
fi

# 3. Clean Seed / Demo Mode Check
if [ ! -f "outputs/predictions/flagged_transactions.csv" ] || [ ! -s "outputs/predictions/flagged_transactions.csv" ]; then
    echo "[*] Predictions CSV absent. Running seed demo inference via src/predict.py..."
    .venv/bin/python src/predict.py \
        --input outputs/predictions/sample_input.csv \
        --output outputs/predictions/flagged_transactions.csv \
        --threshold 0.144
fi

# 4. Cleanup function on exit
cleanup() {
    echo ""
    echo "[*] Shutting down dashboard servers..."
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup INT TERM EXIT

# 5. Start Backend
echo "[*] Starting FastAPI Backend on http://localhost:8000..."
export PYTHONPATH="$REPO_ROOT"
.venv/bin/uvicorn dashboard.backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "[*] Waiting for backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "[+] Backend is ready on http://localhost:8000"
        break
    fi
    sleep 0.5
done

# 6. Start Frontend
echo "[*] Starting Vite React Frontend on http://localhost:5173..."
(cd dashboard/frontend && npm run dev -- --host 0.0.0.0 --port 5173) &
FRONTEND_PID=$!

echo "======================================================================"
echo " Dashboard running!"
echo "   - Frontend UI:  http://localhost:5173"
echo "   - Backend API: http://localhost:8000 (Docs: http://localhost:8000/docs)"
echo " Press Ctrl+C to stop all servers."
echo "======================================================================"

# Wait for processes
wait

