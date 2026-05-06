"""
Lightweight SQLite event store for near real-time monitoring + continuous learning.

This is additive: the existing JSONL alert bus remains the cross-process ring buffer.
SQLite provides durable history and low-cost querying for dashboards and model drift.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config.settings import get_settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(obj), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class StoredEvent:
    ts: str
    risk_class: str
    risk_code: int
    anomaly_flag: int
    anomaly_score: float
    cvss_predicted: float
    confidence: float
    source: str
    subject_key: str
    cve_id: Optional[str]
    lstm_risk_class: Optional[str]
    lstm_confidence: Optional[float]
    explanation_json: str
    live_meta_json: str


class SqliteEventStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        con.row_factory = sqlite3.Row
        # Low-resource friendly durability/perf tradeoff.
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    source TEXT NOT NULL,
                    subject_key TEXT NOT NULL,
                    cve_id TEXT,
                    risk_class TEXT NOT NULL,
                    risk_code INTEGER NOT NULL,
                    anomaly_flag INTEGER NOT NULL,
                    anomaly_score REAL NOT NULL,
                    cvss_predicted REAL NOT NULL,
                    confidence REAL NOT NULL,
                    lstm_risk_class TEXT,
                    lstm_confidence REAL,
                    explanation_json TEXT NOT NULL,
                    live_meta_json TEXT NOT NULL
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_subject ON events(subject_key, ts);")

            con.execute(
                """
                CREATE TABLE IF NOT EXISTS packets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    src_ip TEXT NOT NULL,
                    dst_ip TEXT NOT NULL,
                    src_port INTEGER NOT NULL,
                    dst_port INTEGER NOT NULL,
                    proto TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    summary TEXT NOT NULL
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_packets_ts ON packets(ts);")

    def insert_packet_snapshot(self, snap: Dict[str, Any]) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO packets(ts, src_ip, dst_ip, src_port, dst_port, proto, size, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(snap.get("ts") or _utc_now_iso()),
                    str(snap.get("src_ip") or ""),
                    str(snap.get("dst_ip") or ""),
                    int(snap.get("src_port") or 0),
                    int(snap.get("dst_port") or 0),
                    str(snap.get("proto") or ""),
                    int(snap.get("size") or 0),
                    str(snap.get("summary") or "")[:800],
                ),
            )

    def insert_scored_event(
        self,
        scored_row: Dict[str, Any],
        *,
        source: str,
        subject_key: str,
        ts: Optional[str] = None,
    ) -> None:
        ts_s = ts or str(scored_row.get("timestamp") or scored_row.get("ts") or _utc_now_iso())
        risk_class = str(scored_row.get("risk_class") or "Safe")
        risk_code = int(scored_row.get("risk_code") or 0)
        anomaly_flag = int(scored_row.get("anomaly_score_flag") or scored_row.get("anomaly") or 1)
        anomaly_score = float(scored_row.get("anomaly_score") or 0.0)
        cvss_pred = float(scored_row.get("cvss_predicted") or 0.0)
        confidence = float(scored_row.get("confidence") or 0.0)
        cve_id = scored_row.get("cve_id")
        cve_id = None if cve_id is None else str(cve_id)
        lstm_risk_class = scored_row.get("lstm_risk_class")
        lstm_risk_class = None if lstm_risk_class is None else str(lstm_risk_class)
        lstm_confidence = scored_row.get("lstm_confidence")
        try:
            lstm_confidence_f = None if lstm_confidence is None else float(lstm_confidence)
        except (TypeError, ValueError):
            lstm_confidence_f = None

        expl = scored_row.get("explanation") or {}
        live_meta = scored_row.get("live_meta") or {}

        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO events(
                    ts, source, subject_key, cve_id,
                    risk_class, risk_code, anomaly_flag, anomaly_score,
                    cvss_predicted, confidence, lstm_risk_class, lstm_confidence,
                    explanation_json, live_meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_s,
                    str(source),
                    str(subject_key),
                    cve_id,
                    risk_class,
                    int(risk_code),
                    int(anomaly_flag),
                    float(anomaly_score),
                    float(cvss_pred),
                    float(confidence),
                    lstm_risk_class,
                    lstm_confidence_f,
                    _json_dumps(expl),
                    _json_dumps(live_meta),
                ),
            )

    def tail_events(self, n: int = 200) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                (int(max(1, min(n, 2000))),),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in reversed(rows):
            d = dict(r)
            for k in ("explanation_json", "live_meta_json"):
                try:
                    d[k] = json.loads(d.get(k) or "{}")
                except Exception:
                    d[k] = {}
            out.append(d)
        return out


_STORE: Optional[SqliteEventStore] = None
_STORE_LOCK = threading.Lock()


def get_event_store() -> Optional[SqliteEventStore]:
    """
    Lazy singleton store. Returns None when disabled to preserve existing behavior.
    """
    global _STORE
    s = get_settings()
    if not getattr(s, "sqlite_enabled", True):
        return None
    with _STORE_LOCK:
        if _STORE is None:
            db_path = Path(getattr(s, "sqlite_path", "")) if getattr(s, "sqlite_path", "") else None
            if db_path is None:
                db_path = Path(s.project_root) / "logs" / "aura_events.db"
            _STORE = SqliteEventStore(db_path)
        return _STORE

