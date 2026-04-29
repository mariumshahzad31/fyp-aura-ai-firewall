"""
Shared helpers: paths, structured logging, exports, and API-oriented DTOs.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATASET_FILENAME = "Dataset-Attacks-Firewall.csv"
DEFAULT_DATA_PATH = DATA_DIR / DATASET_FILENAME


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON for structured ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    name: str = "aura",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    json_logs: bool = True,
) -> logging.Logger:
    """Configure root-style logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(console)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = log_file or (LOGS_DIR / "aura.log")
    try:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("Could not attach file handler: %s", exc)

    return logger


def ensure_directories() -> None:
    for d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictionRequest:
    """API-ready request body for batch or single-record scoring."""

    records: List[Dict[str, Any]]
    include_explanation: bool = True


@dataclass
class PredictionResponse:
    """API-ready response with scores and optional narrative fields."""

    predictions: List[Dict[str, Any]]
    model_versions: Dict[str, str] = field(default_factory=dict)


def export_threat_report_csv(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    """Write threat history rows to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_threat_report_pdf(rows: Sequence[Dict[str, Any]], out_path: Path, title: str = "AURA Threat Report") -> None:
    """Write a simple PDF summary of recent threats using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise RuntimeError("fpdf2 is required for PDF export. Install with: pip install fpdf2") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.now(timezone.utc).isoformat()} UTC", ln=True)
    pdf.ln(4)

    max_rows = 80
    display = list(rows[:max_rows])
    for i, row in enumerate(display, start=1):
        line = f"{i}. " + " | ".join(f"{k}={v}" for k, v in list(row.items())[:12])
        if len(line) > 120:
            line = line[:117] + "..."
        pdf.multi_cell(0, 6, line)
    if len(rows) > max_rows:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, f"... truncated; total rows: {len(rows)}")
    pdf.output(str(out_path))


def dto_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError("Expected dataclass instance")


RISK_COLORS = {
    "Safe": "#22c55e",
    "Suspicious": "#eab308",
    "Malicious": "#ef4444",
    "Critical": "#7f1d1d",
}
