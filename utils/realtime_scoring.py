"""
Near real-time scoring loop for packet capture.

Design goals:
- Additive: does not replace existing UI simulation or API prediction flow.
- Lightweight: small batches, minimal locks, safe defaults.
- Durable: writes alerts (JSONL) + optional SQLite event store.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import get_settings
from utils.alert_bus import append_alert
from utils.behavioral_profiles import combine_scores, get_behavior_store
from utils.firewall import get_firewall_manager
from utils.helpers import LOGS_DIR, setup_logging
from utils.packet_monitor import PacketSnapshot, get_or_create_monitor
from utils.prediction import AuraPredictor
from utils.preprocessing import RISK_NAMES, build_feature_frame, transform_with_preprocessor
from utils.sqlite_store import get_event_store

logger = setup_logging("aura.realtime", log_file=LOGS_DIR / "realtime.log")


@dataclass
class RealtimeState:
    running: bool
    queued: int
    scored: int
    last_error: Optional[str]


class RealtimeScoringService:
    def __init__(self, predictor: AuraPredictor, df_raw: pd.DataFrame) -> None:
        self._predictor = predictor
        self._df_raw = df_raw
        self._queue: Deque[Tuple[Dict[str, Any], PacketSnapshot]] = deque(maxlen=2000)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._scored = 0
        self._last_err: Optional[str] = None
        self._seq_cache: Dict[str, Deque[np.ndarray]] = {}

    def _subject_key(self, rec: Dict[str, Any]) -> str:
        lm = rec.get("_live_meta") if isinstance(rec.get("_live_meta"), dict) else {}
        ip = (lm or {}).get("observed_source_ip")
        if ip:
            return str(ip)
        return str(rec.get("Data") or rec.get("cve_id") or "unknown")

    def _enqueue(self, rec: Dict[str, Any], snap: PacketSnapshot) -> None:
        with self._lock:
            self._queue.append((rec, snap))

    def start(self, bpf_filter: str = "ip") -> bool:
        s = get_settings()
        if not s.realtime_scoring_enabled:
            self._last_err = "Realtime scoring disabled by settings"
            return False
        mon = get_or_create_monitor(self._df_raw)
        mon.set_on_record(lambda rec, snap: self._enqueue(rec, snap))
        ok = mon.start(bpf_filter=bpf_filter)
        if not ok:
            self._last_err = mon.state().last_error
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def state(self) -> RealtimeState:
        with self._lock:
            qn = len(self._queue)
        return RealtimeState(
            running=self._thread is not None and self._thread.is_alive(),
            queued=qn,
            scored=self._scored,
            last_error=self._last_err,
        )

    def _loop(self) -> None:
        s = get_settings()
        event_store = get_event_store()
        while not self._stop.is_set():
            batch: List[Tuple[Dict[str, Any], PacketSnapshot]] = []
            with self._lock:
                while self._queue and len(batch) < 16:
                    batch.append(self._queue.popleft())
            if not batch:
                time.sleep(0.25)
                continue

            try:
                records = [b[0] for b in batch]
                snaps = [b[1] for b in batch]

                clean_records: List[Dict[str, Any]] = []
                live_metas: List[Dict[str, Any]] = []
                keys: List[str] = []
                for r in records:
                    rr = dict(r)
                    lm = rr.pop("_live_meta", {}) if isinstance(rr.get("_live_meta"), dict) else {}
                    clean_records.append(rr)
                    live_metas.append(lm if isinstance(lm, dict) else {})
                    keys.append(self._subject_key(r))

                # Precompute transformed vectors for LSTM windows (per subject).
                df = pd.DataFrame(clean_records)
                feat = build_feature_frame(df)
                X = transform_with_preprocessor(self._predictor.risk.preprocessor, feat)
                X = np.asarray(X, dtype=np.float32)

                scored = self._predictor.predict_records(records, include_explanation=True, use_llm=False)
                now = datetime.now(timezone.utc).isoformat()
                behavior_store = get_behavior_store()
                fw = get_firewall_manager()

                for i, row in enumerate(scored):
                    key = keys[i] if i < len(keys) else "unknown"
                    row["timestamp"] = row.get("timestamp") or now
                    if i < len(live_metas) and live_metas[i]:
                        row["live_meta"] = live_metas[i]

                    # Behavioral update (existing logic).
                    dev = behavior_store.deviation_score(key, str(row.get("risk_class", "Safe")))
                    behavior_store.record_event(key, str(row.get("risk_class", "Safe")), now)
                    row["behavioral"] = combine_scores(int(row.get("risk_code", 0)), dev)

                    # LSTM: per-subject sliding window over transformed features.
                    try:
                        vec = X[i]
                        dq = self._seq_cache.setdefault(key, deque(maxlen=int(self._predictor.risk.lstm_seq_len)))
                        dq.append(vec)
                        if len(dq) >= int(self._predictor.risk.lstm_seq_len):
                            seq = np.stack(list(dq), axis=0)[np.newaxis, ...]
                            prob = self._predictor.predict_lstm_sequence(seq)
                            li = int(np.argmax(prob[0]))
                            row["lstm_risk_class"] = RISK_NAMES[li]
                            row["lstm_confidence"] = float(np.max(prob[0]))
                            expl = row.get("explanation") if isinstance(row.get("explanation"), dict) else {}
                            expl = dict(expl)
                            expl.setdefault(
                                "sequence_context",
                                f"LSTM sequence signal: {row['lstm_risk_class']} (conf {row['lstm_confidence']:.2f}) over the last {self._predictor.risk.lstm_seq_len} observations for {key}.",
                            )
                            row["explanation"] = expl
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("LSTM sequence enrichment failed: %s", str(exc)[:200])

                    append_alert(
                        {
                            "ts": now,
                            "cve_id": row.get("cve_id"),
                            "risk_class": row.get("risk_class"),
                            "source": "realtime_packet",
                            "subject_key": key,
                        }
                    )

                    if event_store is not None:
                        try:
                            event_store.insert_scored_event(row, source="realtime_packet", subject_key=key, ts=now)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("sqlite insert failed: %s", str(exc)[:240])

                    # Optional blocking (kept conservative).
                    if s.firewall_auto_block and row.get("live_meta"):
                        ip = (row.get("live_meta") or {}).get("observed_source_ip")
                        if ip and str(row.get("risk_class")) in ("Malicious", "Critical"):
                            fw.block_observed_ip(str(ip), str(row.get("risk_class")), "realtime")

                    # Store packet snapshot too (for dashboard context).
                    if event_store is not None:
                        try:
                            event_store.insert_packet_snapshot(snap=snaps[i].__dict__)
                        except Exception:
                            pass

                self._scored += len(scored)
                self._last_err = None
            except Exception as exc:  # noqa: BLE001
                self._last_err = str(exc)[:400]
                logger.warning("realtime worker error: %s", self._last_err)
                time.sleep(0.5)


_SVC: Optional[RealtimeScoringService] = None
_SVC_LOCK = threading.Lock()


def get_or_create_realtime_service(predictor: AuraPredictor, df_raw: pd.DataFrame) -> RealtimeScoringService:
    global _SVC
    with _SVC_LOCK:
        if _SVC is None:
            _SVC = RealtimeScoringService(predictor, df_raw)
        return _SVC

