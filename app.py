#!/usr/bin/env python3
"""
AURA Streamlit control center: production-style dashboard shell, dataset-driven analytics,
monitoring, workflow visualization, exports, and retrain orchestration.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_metrics import compute_dataset_summary
from utils.helpers import setup_logging, LOGS_DIR
from utils.prediction import AuraPredictor
from utils.preprocessing import RISK_NAMES, load_raw_dataset

from ui.layout import (
    architecture_panel,
    header_bar,
    init_shell_state,
    sidebar_branding,
    sidebar_footer,
    sidebar_navigation,
)
from ui.pages import (
    emit_simulation_batch,
    filter_history,
    render_ai_analyst_chat,
    render_ai_insights,
    render_analytics,
    render_architecture_full,
    render_dashboard,
    render_firewall_controls,
    render_live_monitoring,
    render_logs,
    render_threat_intel,
    render_anomaly_check,
    run_retrain,
    sensitivity_min_risk_index,
)
from ui.styles import hide_streamlit_chrome, inject_global_styles

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logging("aura.app", log_file=LOGS_DIR / "app.log")

st.set_page_config(
    page_title="AURA — Behavioral Firewall",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def cached_predictor() -> AuraPredictor:
    return AuraPredictor()


@st.cache_data(show_spinner=False)
def cached_raw_dataframe():
    return load_raw_dataset()


def _mode_to_arch_index(mode: str) -> int:
    if mode == "Learning":
        return 1
    if mode == "Monitoring":
        return 3
    return 2


def main() -> None:
    init_shell_state()
    inject_global_styles()
    hide_streamlit_chrome()

    df_raw = cached_raw_dataframe()
    summary = compute_dataset_summary(df_raw)

    try:
        predictor = cached_predictor()
        model_ready = True
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError) as exc:
        predictor = None
        model_ready = False
        st.error(f"Models are not available ({exc}). Run `python train_model.py` first.")

    with st.sidebar:
        sidebar_branding()
        page = sidebar_navigation()
        mode = st.selectbox("System mode", ["Protection", "Learning", "Monitoring"], key="aura_system_mode")
        st.session_state.aura_arch_active = _mode_to_arch_index(mode)
        threat_filter = st.multiselect("Threat levels", RISK_NAMES, default=RISK_NAMES, key="aura_threat_levels")

        smin = sensitivity_min_risk_index(st.session_state.aura_sensitivity)
        floor = set(RISK_NAMES[smin:])
        levels = [x for x in threat_filter if x in floor] or list(floor)

        hist = filter_history(st.session_state.threat_history, levels)
        sec_hist = hist[-500:] if hist else []
        under_attack = any(row.get("risk_class") in ("Malicious", "Critical") for row in sec_hist)
        status_text = "ALERT" if under_attack else "NOMINAL"
        status_color = "#f97316" if under_attack else "#22c55e"

        model_state = {"Protection": "Active", "Learning": "Learning", "Monitoring": "Idle"}.get(mode, "Active")

        def on_run_scan() -> None:
            if not model_ready or predictor is None:
                st.session_state.aura_toast = "Models unavailable for scan."
                st.rerun()
                return
            n = emit_simulation_batch(predictor, model_ready, df_raw, 5, logger)
            st.session_state.aura_toast = f"Scan ingested {n} events."
            st.rerun()

        def on_refresh() -> None:
            cached_raw_dataframe.clear()
            st.session_state.aura_toast = "Dataset cache cleared."
            st.rerun()

        sidebar_footer(status_text, status_color, model_state, on_run_scan, on_refresh)

        st.divider()
        if st.button("Run model retraining", type="primary", key="retrain_sidebar"):
            run_retrain(cached_predictor, logger)

    search_q = str(st.session_state.get("aura_search", ""))

    crit_recent = sum(1 for r in hist if r.get("risk_class") == "Critical")
    st.session_state.aura_notification_count = crit_recent

    toast_msg = st.session_state.pop("aura_toast", None)
    if toast_msg:
        st.success(toast_msg)

    title_map = {
        "Dashboard": "Dashboard",
        "Anomaly Check": "Real-Time Anomaly Verification",
        "Live Monitoring": "Live Monitoring",
        "AI Insights": "AI Insights",
        "AI Analyst Chat": "AI Analyst Chat",
        "Threat Intelligence": "Threat Intelligence",
        "System Architecture": "System Architecture",
        "Analytics": "Analytics",
        "Firewall Controls": "Firewall Controls",
        "Logs": "Logs",
    }
    header_bar(title_map.get(page, "AURA"))


    if page == "Dashboard":
        render_dashboard(
            df_raw,
            summary,
            predictor,
            model_ready,
            hist,
            levels,
            cached_raw_dataframe,
            cached_predictor,
            logger,
        )
    elif page == "Anomaly Check":
        render_anomaly_check(df_raw, predictor, model_ready)
    elif page == "Live Monitoring":
        render_live_monitoring(df_raw, hist, predictor, model_ready)
    elif page == "AI Insights":
        render_ai_insights(df_raw, hist)
    elif page == "AI Analyst Chat":
        render_ai_analyst_chat()
    elif page == "System Architecture":
        st.subheader("Full pipeline")
        architecture_panel(compact=False, mode=mode, active_stage_index=int(st.session_state.aura_arch_active))
        render_architecture_full(df_raw, mode, model_ready)
    elif page == "Analytics":
        render_analytics(df_raw)
    elif page == "Threat Intelligence":
        render_threat_intel(df_raw)
    elif page == "Firewall Controls":
        render_firewall_controls(df_raw)
    elif page == "Logs":
        render_logs(df_raw, hist, search_q)

    st.divider()
    st.caption(
        "Production API: `uvicorn api.main:app --host 0.0.0.0 --port 8000` — OpenAPI at `/docs` (JWT auth). "
        "Streamlit and API share models, dataset pipeline, alerts JSONL, and firewall orchestration."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logger.exception("App failure: %s", exc)
        raise
