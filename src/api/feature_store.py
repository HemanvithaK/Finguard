# src/api/feature_store.py
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path(__file__).resolve().parent / "feature_store.db"


class FeatureStore:
    """
    Lightweight SQLite-based feature store that tracks recent transactions
    per user and computes rolling velocity features in real-time.
    
    Why SQLite instead of Redis:
    - Zero external dependencies (no Redis server to install/manage)
    - Runs anywhere Docker runs
    - Fast enough for our throughput (thousands of txns/sec on SQLite)
    - In production you'd swap this for Redis or a streaming engine (Flink),
      but the interface stays the same — that's the point of abstracting it.
    """

    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create the transactions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recent_transactions (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                merchant_id TEXT NOT NULL,
                amount REAL NOT NULL,
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                inserted_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_timestamp 
            ON recent_transactions(user_id, timestamp)
        """)
        conn.commit()
        conn.close()

    def record_transaction(self, txn: dict):
        """
        Record a transaction so future feature lookups can use it.
        Called after every /predict request — builds up the history
        that makes velocity features accurate.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO recent_transactions 
            (transaction_id, user_id, merchant_id, amount, timestamp, device_id, inserted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            txn["transaction_id"],
            txn["user_id"],
            txn["merchant_id"],
            txn["amount"],
            txn["timestamp"],
            txn["device_id"],
            time.time(),
        ))
        conn.commit()
        conn.close()

    def get_velocity_features(self, user_id: str, current_timestamp: str) -> dict:
        """
        Compute rolling velocity features for a user at a given timestamp.
        These match EXACTLY what the training pipeline computes:
        - time_since_prev_txn (seconds)
        - txns_last_10min (count)
        - distinct_merchants_1hr (count of unique merchants)
        """
        conn = sqlite3.connect(self.db_path)
        current_dt = datetime.fromisoformat(current_timestamp)

        # 1. Time since previous transaction
        prev_txn = conn.execute("""
            SELECT timestamp FROM recent_transactions
            WHERE user_id = ? AND timestamp < ?
            ORDER BY timestamp DESC LIMIT 1
        """, (user_id, current_timestamp)).fetchone()

        if prev_txn:
            prev_dt = datetime.fromisoformat(prev_txn[0])
            time_since_prev = (current_dt - prev_dt).total_seconds()
        else:
            time_since_prev = 999999.0  # no history — same default as training

        # 2. Transaction count in last 10 minutes
        ten_min_ago = (current_dt - timedelta(minutes=10)).isoformat()
        txn_count_10min = conn.execute("""
            SELECT COUNT(*) FROM recent_transactions
            WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
        """, (user_id, ten_min_ago, current_timestamp)).fetchone()[0]
        txns_last_10min = txn_count_10min + 1  # +1 for the current transaction

        # 3. Distinct merchants in last hour
        one_hour_ago = (current_dt - timedelta(hours=1)).isoformat()
        distinct_merchants = conn.execute("""
            SELECT COUNT(DISTINCT merchant_id) FROM recent_transactions
            WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
        """, (user_id, one_hour_ago, current_timestamp)).fetchone()[0]
        distinct_merchants_1hr = distinct_merchants + 1  # +1 for current merchant

        conn.close()

        return {
            "time_since_prev_txn": time_since_prev,
            "txns_last_10min": float(txns_last_10min),
            "distinct_merchants_1hr": float(distinct_merchants_1hr),
        }

    def cleanup_old(self, max_age_hours=24):
        """
        Remove transactions older than max_age_hours.
        Keeps the database small — we only need recent history
        for velocity features, not a permanent archive.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        conn = sqlite3.connect(self.db_path)
        deleted = conn.execute("""
            DELETE FROM recent_transactions WHERE inserted_at < ?
        """, (cutoff,)).rowcount
        conn.commit()
        conn.close()
        return deleted