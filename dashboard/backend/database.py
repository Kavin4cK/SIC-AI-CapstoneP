"""SQLite storage layer for operator review labels.
Guarantees persistent separation between prediction outputs and operator decisions.
Supports multi-dataset isolation with (dataset_id, transaction_id) composite keys.
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
    """Initialize SQLite tables for operator feedback, with automatic schema migration."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='operator_labels'")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            cursor.execute("PRAGMA table_info(operator_labels)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "dataset_id" not in columns:
                # Migrate to compound primary key (dataset_id, transaction_id)
                cursor.execute(
                    """
                    CREATE TABLE operator_labels_new (
                        dataset_id TEXT NOT NULL DEFAULT 'default',
                        transaction_id TEXT NOT NULL,
                        label TEXT NOT NULL CHECK(label IN ('fraud', 'genuine', 'CONFIRMED_FRAUD', 'CONFIRMED_GENUINE')),
                        operator TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (dataset_id, transaction_id)
                    );
                    """
                )
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO operator_labels_new (dataset_id, transaction_id, label, operator, created_at)
                    SELECT 'default', transaction_id, label, operator, created_at FROM operator_labels;
                    """
                )
                cursor.execute("DROP TABLE operator_labels")
                cursor.execute("ALTER TABLE operator_labels_new RENAME TO operator_labels")
                conn.commit()
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS operator_labels (
                    dataset_id TEXT NOT NULL DEFAULT 'default',
                    transaction_id TEXT NOT NULL,
                    label TEXT NOT NULL CHECK(label IN ('fraud', 'genuine', 'CONFIRMED_FRAUD', 'CONFIRMED_GENUINE')),
                    operator TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, transaction_id)
                );
                """
            )
            conn.commit()


def save_label(
    transaction_id: str,
    label: str,
    operator: Optional[str] = None,
    dataset_id: Optional[str] = "default",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Insert or update an operator decision for a transaction within a specific dataset."""
    init_db(db_path)
    clean_label = label.strip().lower()
    if clean_label not in ("fraud", "genuine"):
        raise ValueError(f"Invalid label '{label}'. Allowed: 'fraud', 'genuine'")

    clean_dataset_id = (dataset_id or "default").strip() or "default"
    op_name = (operator or "operator").strip() or "operator"
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO operator_labels (dataset_id, transaction_id, label, operator, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, transaction_id) DO UPDATE SET
                label=excluded.label,
                operator=excluded.operator,
                created_at=excluded.created_at;
            """,
            (clean_dataset_id, transaction_id, clean_label, op_name, now_iso),
        )
        conn.commit()

    return {
        "dataset_id": clean_dataset_id,
        "TransactionID": transaction_id,
        "label": clean_label,
        "operator": op_name,
        "created_at": now_iso,
    }


def get_all_labels(
    dataset_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recorded operator labels, optionally filtered by dataset_id."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        if dataset_id and dataset_id.strip() and dataset_id.strip().lower() != "all":
            cursor = conn.execute(
                """
                SELECT dataset_id, transaction_id, label, operator, created_at
                FROM operator_labels
                WHERE dataset_id = ?
                ORDER BY created_at DESC;
                """,
                (dataset_id.strip(),),
            )
        else:
            cursor = conn.execute(
                """
                SELECT dataset_id, transaction_id, label, operator, created_at
                FROM operator_labels
                ORDER BY created_at DESC;
                """
            )
        rows = cursor.fetchall()
        return [
            {
                "dataset_id": row["dataset_id"],
                "TransactionID": row["transaction_id"],
                "label": row["label"],
                "operator": row["operator"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


def get_label_map(
    dataset_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return a dictionary mapping transaction_id to label info for the specified dataset."""
    labels = get_all_labels(dataset_id=dataset_id, db_path=db_path)
    return {item["TransactionID"]: item for item in labels}
