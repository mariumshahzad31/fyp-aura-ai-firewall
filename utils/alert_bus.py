"""Cross-process alert ring (JSONL) for Streamlit + FastAPI + packet worker."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List

from utils.helpers import LOGS_DIR

ALERT_PATH = LOGS_DIR / "aura_alerts.jsonl"
_LOCK = threading.Lock()


def append_alert(payload: Dict[str, Any]) -> None:
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with _LOCK:
        with ALERT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def read_alerts_tail(n: int = 200) -> List[Dict[str, Any]]:
    if not ALERT_PATH.exists():
        return []
    lines = ALERT_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]
    out: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
