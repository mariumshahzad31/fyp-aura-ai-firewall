"""
AURA FastAPI application: predictions, alerts, logs, system status, threat intel.
Mobile-ready: JSON + CORS; pair with Streamlit UI or native clients.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from api.auth import authenticate_user, create_access_token, get_current_user, require_admin
from api.schemas import PredictRequest, SystemStatus, Token
from config.settings import get_settings
from utils.alert_bus import append_alert, read_alerts_tail
from utils.behavioral_profiles import combine_scores, get_behavior_store
from utils.firewall import get_firewall_manager
from utils.helpers import LOGS_DIR, setup_logging
from utils.packet_monitor import get_or_create_monitor
from utils.prediction import AuraPredictor
from utils.preprocessing import load_raw_dataset
from utils.threat_intel import correlate_dataset_cve

logger = setup_logging("aura.api", log_file=LOGS_DIR / "api.log")

app = FastAPI(title="AURA API", version="2.0.0", docs_url="/docs", redoc_url="/redoc")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins if _settings.cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Optional[AuraPredictor] = None


def get_predictor() -> AuraPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AuraPredictor()
    return _predictor


@app.post("/api/v1/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(
        {"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=_settings.access_token_expire_minutes),
    )
    return Token(access_token=token)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "aura"}


@app.get("/api/v1/status", response_model=SystemStatus)
async def status_endpoint(user: Dict[str, Any] = Depends(get_current_user)) -> SystemStatus:
    try:
        await run_in_threadpool(get_predictor)
        ready = True
    except Exception:  # noqa: BLE001
        ready = False
    fw = get_firewall_manager().state()
    df = await run_in_threadpool(load_raw_dataset)
    mon = get_or_create_monitor(df).state()
    beh = get_behavior_store().snapshot()
    return SystemStatus(
        model_ready=ready,
        firewall={
            "dry_run": fw.dry_run,
            "platform": fw.platform_name,
            "blocked_ips": fw.blocked_ips,
            "recent_actions": fw.last_actions,
        },
        packet_monitor={
            "running": mon.running,
            "packets_captured": mon.packets_captured,
            "last_error": mon.last_error,
        },
        behavioral_profiles=beh,
    )


@app.post("/api/v1/predict")
async def predict_endpoint(
    body: PredictRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Predict endpoint with comprehensive error handling"""
    try:
        pred = await run_in_threadpool(get_predictor)
        out = await run_in_threadpool(
            pred.predict_records,
            body.records,
            include_explanation=body.include_explanation,
            use_llm=body.use_llm,
        )
        now = datetime.now(timezone.utc).isoformat()
        
        for row in out:
            try:
                lm = row.get("live_meta") or {}
                
                # Get CVE ID with fallback detection
                cve_id = (
                    row.get("cve_id") or 
                    row.get("Data") or 
                    row.get("CVE") or 
                    row.get("vulnerability") or 
                    str(lm.get("observed_cve_id", "unknown"))
                )
                
                key = str(lm.get("observed_source_ip") or cve_id or "unknown")
                store = get_behavior_store()
                risk_class = str(row.get("risk_class", "Safe"))
                dev = store.deviation_score(key, risk_class)
                store.record_event(key, risk_class, now)
                row["behavioral"] = combine_scores(int(row.get("risk_code", 0)), dev)
                
                # Alert with proper CVE field
                alert_data = {
                    "ts": now,
                    "cve_id": cve_id,
                    "risk_class": risk_class,
                    "user": user.get("username"),
                    "source": "api",
                }
                append_alert(alert_data)
                
                # Firewall auto-blocking
                if _settings.firewall_auto_block and row.get("live_meta"):
                    ip = row["live_meta"].get("observed_source_ip")
                    if ip and risk_class in ("Malicious", "Critical"):
                        get_firewall_manager().block_observed_ip(str(ip), risk_class, "api_auto")
            except Exception as e:
                logger.warning(f"Error processing prediction row: {e}")
                continue
        
        return {"predictions": out, "model_versions": {"risk": "risk_model.pkl", "cvss": "cvss_model.pkl"}}
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/v1/alerts")
async def alerts(user: Dict[str, Any] = Depends(get_current_user), n: int = 100) -> Dict[str, Any]:
    return {"items": read_alerts_tail(n)}


@app.get("/api/v1/logs/tail")
async def logs_tail(
    user: Dict[str, Any] = Depends(require_admin),
    path: str = "app.log",
    lines: int = 120,
) -> Dict[str, Any]:
    """Return last N lines from logs/{path} (admin only)."""
    safe = Path(path).name
    fp = LOGS_DIR / safe
    if not fp.exists():
        return {"lines": [], "path": str(fp)}
    text = fp.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:]
    return {"lines": text, "path": str(fp)}


@app.get("/api/v1/intel/cve/{cve_id}")
async def intel_cve(cve_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get CVE intelligence with dynamic column detection"""
    try:
        df = load_raw_dataset()
        
        # Dynamic column detection
        cve_columns = ["Data", "cve_id", "cve", "CVE"]
        cve_col = None
        for col in cve_columns:
            if col in df.columns:
                cve_col = col
                break
        
        if not cve_col:
            cve_col = next((c for c in df.columns if "cve" in c.lower()), None)
        
        if cve_col:
            sub = df[df[cve_col].astype(str).str.upper() == cve_id.upper()]
            if not sub.empty:
                cvss_columns = ["cvss", "cvss_score", "severity"]
                cvss_col = next((c for c in cvss_columns if c in sub.columns), "cvss")
                cvss_val = float(sub.iloc[0][cvss_col]) if cvss_col in sub.columns else None
                return correlate_dataset_cve(str(sub.iloc[0][cve_col]), cvss_val if cvss_val else 0)
        
        return {"error": "CVE not in local dataset", "cve_id": cve_id}
    except Exception as e:
        logger.error(f"CVE lookup error for {cve_id}: {e}")
        return {"error": "CVE lookup failed", "cve_id": cve_id, "detail": str(e)}


@app.post("/api/v1/firewall/block")
async def firewall_block(
    ip: str,
    user: Dict[str, Any] = Depends(require_admin),
    reason: str = "manual",
) -> Dict[str, Any]:
    res = get_firewall_manager().block_observed_ip(ip, "Critical", reason)
    return {"ok": res.ok, "message": res.message}


@app.post("/api/v1/firewall/unblock")
async def firewall_unblock(ip: str, user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    res = get_firewall_manager().unblock(ip)
    return {"ok": res.ok, "message": res.message}


@app.on_event("startup")
async def startup() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("AURA API startup complete")
