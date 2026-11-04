"""SQLite storage layer for operator review labels.
Guarantees persistent separation between prediction outputs and operator decisions.
Never writes to or overwrites ML prediction files.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "labels.sqlite3"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite tables for operator feedback."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_labels (
                transaction_id TEXT PRIMARY KEY,
                label TEXT NOT NULL CHECK(label IN ('fraud', 'genuine', 'CONFIRMED_FRAUD', 'CONFIRMED_GENUINE')),
                operator TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def save_label(
    transaction_id: str,
    label: str,
    operator: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Insert or update an operator decision for a transaction."""
    init_db(db_path)
    clean_label = label.strip().lower()
    if clean_label not in ("fraud", "genuine"):
        raise ValueError(f"Invalid label '{label}'. Allowed: 'fraud', 'genuine'")

    op_name = (operator or "operator").strip() or "operator"
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO operator_labels (transaction_id, label, operator, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(transaction_id) DO UPDATE SET
                label=excluded.label,
                operator=excluded.operator,
                created_at=excluded.created_at;
            """,
            (transaction_id, clean_label, op_name, now_iso),
        )
        conn.commit()

    return {
        "TransactionID": transaction_id,
        "label": clean_label,
        "operator": op_name,
        "created_at": now_iso,
    }


def get_all_labels(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all recorded operator labels ordered by created_at desc."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT transaction_id, label, operator, created_at
            FROM operator_labels
            ORDER BY created_at DESC;
            """
        )
        rows = cursor.fetchall()
        return [
            {
                "TransactionID": row["transaction_id"],
                "label": row["label"],
                "operator": row["operator"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


def get_label_map(db_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Return a dictionary mapping transaction_id to label info."""
    labels = get_all_labels(db_path)
    return {item["TransactionID"]: item for item in labels}

