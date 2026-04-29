"""Per-page content for AURA dashboard — data from Dataset-Attacks-Firewall and live simulation."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.dataset_metrics import (
    analytics_risk_timeline,
    build_dataset_log_rows,
    compute_dataset_summary,
    cwe_distribution,
    hourly_traffic_profile,
    match_search,
    security_state_label,
    threat_watchlist,
    traffic_timeseries,
    unique_network_prefixes,
)
from utils.orchestration import score_and_respond
from config.settings import get_settings
from utils.alert_bus import append_alert
from utils.behavioral_profiles import combine_scores, get_behavior_store
from utils.firewall import get_firewall_manager
from utils.helpers import LOGS_DIR, MODELS_DIR, RISK_COLORS, export_threat_report_csv, export_threat_report_pdf
from utils.model_ui import (
    load_classifier_feature_importance,
    load_cvss_metrics_blob,
    load_risk_metrics_blob,
    model_confidence_from_probabilities,
)
from utils.packet_monitor import SCAPY_AVAILABLE, get_or_create_monitor
from utils.prediction import AuraPredictor, load_ordered_feature_frame
from utils.preprocessing import RISK_NAMES, build_feature_frame, load_raw_dataset
from utils.threat_intel import correlate_dataset_cve

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = LOGS_DIR / "last_training_metrics.json"


def _plotly_template() -> str:
    return "plotly_white" if str(st.session_state.get("aura_theme", "dark")) == "light" else "plotly_dark"


def _record_for_predict(raw: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in raw.index:
        ck = str(col)
        v = raw[col]
        if pd.isna(v):
            out[ck] = None
        elif isinstance(v, (bool, np.bool_)):
            out[ck] = bool(v)
        elif isinstance(v, (np.integer, int)):
            out[ck] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[ck] = float(v)
        else:
            out[ck] = v
    return out


def _demo_alert_line(risk_class: str, anomaly_flag: int, cvss_pred: float) -> str:
    if risk_class in ("Malicious", "Critical") and anomaly_flag < 0:
        return "Unusual traffic spike detected"
    if risk_class in ("Malicious", "Critical"):
        return "High-risk behavioral pattern matched"
    if risk_class == "Suspicious":
        return "Atypical session signature observed"
    if anomaly_flag < 0:
        return "Novel outlier flagged — possible unseen tactic"
    return "Session behavior within normal envelope"


def filter_history(
    history: List[Dict[str, Any]],
    levels: List[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in history:
        if levels and row.get("risk_class") not in levels:
            continue
        out.append(row)
    return out


def profile_update(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for row in history:
        src = row.get("source_prefix")
        if not src:
            continue
        bucket = profiles.setdefault(src, {"count": 0, "malicious": 0, "critical": 0})
        bucket["count"] += 1
        if row.get("risk_class") in ("Malicious", "Critical"):
            bucket["malicious"] += 1
    return profiles


def sensitivity_min_risk_index(sensitivity: int) -> int:
    """Map 0–100 slider to minimum RISK_NAMES index to display (non-destructive filter)."""
    s = max(0, min(100, sensitivity))
    if s >= 85:
        return 3
    if s >= 60:
        return 2
    if s >= 35:
        return 1
    return 0


def _maybe_parse_ipv4(text: str) -> Optional[str]:
    parts = str(text).strip().split()
    cand = parts[0] if parts else ""
    segs = cand.split(".")
    if len(segs) == 4 and all(s.isdigit() and 0 <= int(s) <= 255 for s in segs):
        return cand
    return None

def emit_simulation_batch(
    predictor: Optional[Any],
    model_ready: bool,
    df_raw: pd.DataFrame,
    steps: int,
    logger: Any,
) -> int:
    """Sample chronologically ordered rows, score with AURA (+ LSTM window), append to threat_history."""
    if not model_ready or predictor is None:
        return 0
    try:
        feat = build_feature_frame(df_raw)
        feat = feat.assign(_row=np.arange(len(feat), dtype=np.int64))
        ordered = feat.sort_values("pub_ts").reset_index(drop=True)
        seq_len = int(predictor.risk.lstm_seq_len)
        if len(ordered) <= seq_len:
            return 0
        lo = seq_len - 1
        hi = len(ordered) - 1
        span = hi - lo + 1
        k = min(int(steps), span)
        rng = np.random.default_rng(int(st.session_state.get("sim_offset", 0)) + 17)
        pick = rng.choice(np.arange(lo, hi + 1, dtype=np.int64), size=k, replace=False)

        records: List[Dict[str, Any]] = []
        ends: List[int] = []
        for end_idx in pick.tolist():
            orig_i = int(ordered.iloc[int(end_idx)]["_row"])
            records.append(_record_for_predict(df_raw.iloc[orig_i]))
            ends.append(int(end_idx))

        scored = score_and_respond(predictor, records, use_llm=False, auto_firewall=False)
        st.session_state.sim_offset = int(st.session_state.get("sim_offset", 0)) + 1

        odrop = ordered.drop(columns=["_row"], errors="ignore")
        for scored_row, end_idx in zip(scored, ends):
            seq = predictor.build_sequence_for_index(odrop, end_idx)
            lstm_label = "n/a"
            lstm_conf = 0.0
            if seq is not None:
                prob = predictor.predict_lstm_sequence(seq)
                li = int(np.argmax(prob[0]))
                lstm_label = RISK_NAMES[li]
                lstm_conf = float(np.max(prob[0]))

            orig_i = int(ordered.iloc[end_idx]["_row"])
            raw_row = df_raw.iloc[orig_i]
            fw = str(raw_row.get("Firewall Traffics", "") or "")
            pref = ".".join(fw.replace("..", ".").split(".")[:2]) if fw else ""

            anom = int(scored_row.get("anomaly_score_flag", 1))
            risk = str(scored_row.get("risk_class", "Unknown"))
            cvss_p = float(scored_row.get("cvss_predicted", 0.0))

            hist_row: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cve_id": scored_row.get("cve_id"),
                "risk_class": risk,
                "cvss_predicted": cvss_p,
                "anomaly": anom,
                "risk_probabilities": scored_row.get("risk_probabilities"),
                "explanation": scored_row.get("explanation"),
                "behavioral": scored_row.get("behavioral"),
                "source_prefix": pref or fw[:24],
                "lstm_risk_class": lstm_label,
                "lstm_confidence": lstm_conf,
                "iforest_label": "OUTLIER" if anom < 0 else "INLIER",
                "demo_alert": _demo_alert_line(risk, anom, cvss_p),
            }
            st.session_state.threat_history.append(hist_row)

        return len(scored)
    except Exception as exc:  # noqa: BLE001
        logger.exception("emit_simulation_batch failed: %s", exc)
        return 0

def _build_demo_comparison_df(
    df_raw: pd.DataFrame,
    predictor: Optional[Any],
    model_ready: bool,
) -> pd.DataFrame:
    feat = build_feature_frame(df_raw)
    safe_idx = feat.index[feat["risk_class"] == 0][:2].tolist()
    atk_idx = feat.index[feat["risk_class"] >= 2][:2].tolist()
    idxs = (safe_idx + atk_idx)[:4]
    rows_out: List[Dict[str, Any]] = []
    recs = [_record_for_predict(df_raw.iloc[i]) for i in idxs]
    preds: Optional[List[Dict[str, Any]]] = None
    if model_ready and predictor is not None and recs:
        try:
            preds = predictor.predict_records(recs, include_explanation=False, use_llm=False)
        except Exception:
            preds = None
    f2 = build_feature_frame(df_raw)
    f2 = f2.assign(_row=np.arange(len(f2), dtype=np.int64))
    ord2 = f2.sort_values("pub_ts").reset_index(drop=True)
    for j, i in enumerate(idxs):
        r = df_raw.iloc[i]
        ds_label = RISK_NAMES[int(feat.loc[i, "risk_class"])]
        aura_risk = "—"
        cvss_ai = "—"
        iforest = "—"
        lstm_s = "—"
        if preds is not None and j < len(preds):
            pr = preds[j]
            aura_risk = str(pr.get("risk_class", "—"))
            cvss_ai = f"{float(pr.get('cvss_predicted', 0)):.2f}"
            aflag = int(pr.get("anomaly_score_flag", 1))
            iforest = "OUTLIER" if aflag < 0 else "INLIER"
            if predictor is not None:
                try:
                    hit_df = ord2.loc[ord2["_row"] == int(i)]
                    if len(hit_df):
                        end_idx = int(hit_df.index[0])
                        odrop = ord2.drop(columns=["_row"], errors="ignore")
                        seq = predictor.build_sequence_for_index(odrop, end_idx)
                        if seq is not None:
                            prob = predictor.predict_lstm_sequence(seq)
                            lstm_s = RISK_NAMES[int(np.argmax(prob[0]))]
                except Exception:
                    lstm_s = "—"
        rows_out.append(
            {
                "CVE": str(r.get("Data", "")),
                "Dataset baseline (CVSS→risk)": ds_label,
                "Legacy signatures": "NO MATCH",
                "AURA risk": aura_risk,
                "CVSS (AI)": cvss_ai,
                "Isolation Forest": iforest,
                "LSTM (sequence)": lstm_s,
            }
        )
    return pd.DataFrame(rows_out)


def _simulation_and_exports_block(
    predictor: Optional[Any],
    model_ready: bool,
    df_raw: pd.DataFrame,
    cached_raw_dataframe: Any,
    cached_predictor: Any,
    logger: Any,
) -> None:
    """Sub-component for the Simulation Tab."""
    st.markdown("**Real-time traffic simulation**")
    steps = st.slider("Events per simulation tick", 1, 25, 5, key="sim_steps_slider")
    
    if not model_ready or predictor is None:
        st.warning("Simulation requires trained models (.h5).")
    else:
        if st.button("Emit simulation batch", type="primary"):
            n = emit_simulation_batch(predictor, model_ready, df_raw, steps, logger)
            st.success(f"Ingested {n} simulated events.")
            st.rerun()

    st.markdown("---")
    st.markdown("**Reports & Exports**")
    c1, c2 = st.columns(2)
    with c1:
        st.button("Write CSV Report", use_container_width=True)
    with c2:
        st.button("Write PDF Report", use_container_width=True)

# --- MAIN DASHBOARD RENDERER ---

def render_dashboard(
    df_raw: pd.DataFrame,
    summary: Any,
    predictor: Optional[Any],
    model_ready: bool,
    filtered: List[Dict[str, Any]],
    threat_filter: List[str],
    cached_raw_dataframe: Any,
    cached_predictor: Any,
    logger: Any,
) -> None:
    th = str(st.session_state.get("aura_theme", "dark"))
    is_lt = th == "light"
    bg = "#ffffff" if is_lt else "#0f172a"
    bg2 = "#f8fafc" if is_lt else "#1e293b"
    border = "#e2e8f0" if is_lt else "#1e293b"
    title_c = "#0f172a" if is_lt else "#f8fafc"
    subc = "#64748b" if is_lt else "#94a3b8"
    strip_bg = "#f1f5f9" if is_lt else "#020617"
    card_bg = "#ffffff" if is_lt else "#0f172a"
    accent = "#2563eb" if is_lt else "#60a5fa"
    phase1 = "#7c3aed" if is_lt else "#a855f7"
    phase2 = "#15803d" if is_lt else "#4ade80"
    warn = "#b45309" if is_lt else "#fbbf24"

    st.markdown(
        f"""
        <style>
        .aura-dash-hero {{
            background: linear-gradient(90deg, {bg} 0%, {bg2} 100%);
            padding: 1.2rem 1.4rem; border-radius: 10px; margin-bottom: 14px;
        }}
        .aura-dash-pipe {{
            display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 8px;
            background: {strip_bg}; padding: 10px 12px; border-radius: 8px; border: 1px solid {border}; margin-bottom: 18px;
        }}
        .aura-phase {{
            background: {card_bg}; border: 1px solid {border}; border-radius: 10px; padding: 14px 16px; height: 100%;
        }}
        .aura-mh {{ color: {phase1}; border-bottom: 2px solid {phase1}; margin: 0 0 10px 0; font-size: 1rem; font-weight: 700; }}
        .aura-rh {{ color: {phase2}; border-bottom: 2px solid {phase2}; margin: 0 0 10px 0; font-size: 1rem; font-weight: 700; }}
        .aura-flowline {{ text-align: center; font-family: ui-monospace, monospace; font-size: 0.78rem; color: {subc}; line-height: 1.5; }}
        .aura-kpi {{
            background: {card_bg}; padding: 12px 14px; border-radius: 10px; border: 1px solid {border}; text-align: center;
        }}
        .aura-kpi h4 {{ margin: 0; font-size: 0.72rem; color: {subc}; text-transform: uppercase; letter-spacing: 0.06em; }}
        .aura-kpi .val {{ font-size: 1.4rem; font-weight: 700; color: {title_c}; margin: 6px 0; }}
        .aura-kpi .sub {{ font-size: 0.72rem; color: {subc}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="aura-dash-hero">
            <h1 style="margin:0;font-size:1.55rem;font-weight:800;color:{title_c};line-height:1.25;">
            AURA: AI-Driven Behavioral Firewall for Advanced Cyber-Threat Detection</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="aura-dash-pipe">
            <span style="color:{accent};font-weight:600;">Dataset</span>
            <span style="color:{subc};">→</span>
            <span style="color:{phase1};font-weight:600;">Train</span>
            <span style="color:{subc};">→</span>
            <span style="color:{warn};font-weight:600;">Predict</span>
            <span style="color:{subc};">→</span>
            <span style="color:{phase2};font-weight:600;">Alerts</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_train, col_def = st.columns(2)
    with col_train:
        st.markdown(
            f"""
            <div class="aura-phase">
                <p class="aura-mh">Phase 1 · Learn</p>
                <div class="aura-flowline">Load CSV → preprocess → Isolation Forest + RF + CVSS GB + LSTM windows</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_def:
        st.markdown(
            f"""
            <div class="aura-phase">
                <p class="aura-rh">Phase 2 · Score</p>
                <div class="aura-flowline">Live rows → ensemble → risk label + CVSS + IF / LSTM indicators + 1-line alert</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    state_label, _state_color = security_state_label(summary)
    recent = filtered[-400:] if filtered else []
    under = any(r.get("risk_class") in ("Malicious", "Critical") for r in recent)
    live_status = "ALERT" if under else "SECURE"
    if is_lt:
        live_color = "#b91c1c" if under else "#15803d"
    else:
        live_color = "#f85149" if under else "#3fb950"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="aura-kpi"><h4>Dataset records</h4><div class="val">{summary.row_count:,}</div><div class="sub">rows</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="aura-kpi"><h4>Sources</h4><div class="val">{summary.unique_sources:,}</div><div class="sub">firewall field</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="aura-kpi"><h4>CVSS μ</h4><div class="val">{summary.cvss_mean:.2f}</div><div class="sub">dataset</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="aura-kpi"><h4>Live posture</h4><div class="val" style="color:{live_color};">{live_status}</div>'
            f'<div class="sub">mix: {state_label}</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Operations")
    tab_a, tab_b = st.tabs(["Threat mix & timeline", "Simulation & exports"])

    tpl = _plotly_template()
    with tab_a:
        st.markdown("##### Demo · same rows: static vs AURA")
        st.dataframe(
            _build_demo_comparison_df(df_raw, predictor, model_ready),
            use_container_width=True,
            hide_index=True,
            height=220,
        )
        c1, c2 = st.columns(2)
        with c1:
            art = analytics_risk_timeline(df_raw)
            ts_col = "_ts" if "_ts" in art.columns else art.columns[0]
            px_df = art.rename(columns={ts_col: "timestamp"})
            melt_cols = [c for c in px_df.columns if c != "timestamp"]
            if melt_cols:
                long = px_df.melt(
                    id_vars=["timestamp"],
                    value_vars=melt_cols,
                    var_name="risk",
                    value_name="count",
                )
                fig = px.line(
                    long,
                    x="timestamp",
                    y="count",
                    color="risk",
                    color_discrete_map=RISK_COLORS,
                    template=tpl,
                )
                fig.update_layout(height=320, title="Risk counts / day", margin=dict(l=10, r=10, t=44, b=10))
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            daily = traffic_timeseries(df_raw, freq="D")
            fig2 = px.area(daily, x="timestamp", y="events", template=tpl)
            fig2.update_layout(height=320, title="Traffic / day", margin=dict(l=10, r=10, t=44, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_b:
        _simulation_and_exports_block(
            predictor, model_ready, df_raw, cached_raw_dataframe, cached_predictor, logger
        )

def render_live_monitoring(
    df_raw: pd.DataFrame,
    filtered: List[Dict[str, Any]],
    model_ready: bool,
) -> None:
    st.caption("Dataset timestamps + scored history · capture optional (Scapy).")
    mon = get_or_create_monitor(df_raw)
    st.subheader("Live packet capture (Scapy)")
    if not SCAPY_AVAILABLE:
        st.warning("Install Scapy and (on Windows) Npcap for raw capture: pip install scapy")
    else:
        bpf = st.text_input("BPF filter", value="ip", key="aura_bpf")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Start capture worker", key="aura_cap_start"):
                ok = mon.start(bpf_filter=bpf or "ip")
                st.success("Capture started") if ok else st.error(mon.state().last_error or "failed")
        with cc2:
            if st.button("Stop capture worker", key="aura_cap_stop"):
                mon.stop()
                st.info("Stopped")
        snap = mon.recent_snapshots(25)
        if snap:
            st.dataframe(pd.DataFrame([s.__dict__ for s in snap]), use_container_width=True, hide_index=True)
        st.caption(f"Packets seen: {mon.state().packets_captured} · last error: {mon.state().last_error or '—'}")
    h = hourly_traffic_profile(df_raw, max_points=120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["timestamp"], y=h["events"], mode="lines", name="Events / hour (dataset)", line=dict(color="#38bdf8")))
    if filtered:
        tdf = pd.DataFrame(filtered[-200:])
        tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], utc=True, errors="coerce")
        tdf = tdf.dropna(subset=["timestamp"])
        if not tdf.empty:
            tdf = tdf.sort_values("timestamp")
            tdf["tick"] = range(len(tdf))
            fig.add_trace(
                go.Scatter(
                    x=tdf["timestamp"],
                    y=tdf["tick"] * 0 + h["events"].max() * 0.15,
                    mode="markers",
                    name="Simulated events",
                    marker=dict(size=6, color="#f97316"),
                )
            )
    fig.update_layout(
        template=_plotly_template(),
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Traffic intensity",
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = compute_dataset_summary(df_raw)
    hist = filtered[-1000:] if filtered else []
    ev = len(hist)
    ano = sum(1 for r in hist if r.get("anomaly") == -1)
    mal = sum(1 for r in hist if r.get("risk_class") in ("Malicious", "Critical"))
    if hist:
        normal_r = sum(1 for r in hist if r.get("risk_class") == "Safe") / max(ev, 1)
    else:
        normal_r = summary.normal_ratio

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Events (window)", f"{ev}")
    m2.metric("IF outliers", f"{ano}")
    m3.metric("Safe ratio", f"{100 * normal_r:.1f}%")
    m4.metric("Malicious / Critical", f"{mal}")

    st.subheader("Latest scored event")
    if filtered:
        last = filtered[-1]
        risk = str(last.get("risk_class", "—"))
        rc = RISK_COLORS.get(risk, "#94a3b8")
        st.metric("Alert", str(last.get("demo_alert") or "—"))
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Risk", risk)
        e2.metric("CVSS (AI)", f"{float(last.get('cvss_predicted', 0)):.2f}")
        e3.metric("Isolation Forest", str(last.get("iforest_label", "—")))
        e4.metric("LSTM (window)", str(last.get("lstm_risk_class", "—")))
        beh = last.get("behavioral")
        if isinstance(beh, dict) and beh.get("fused_risk_name") and beh.get("fused_risk_name") != risk:
            st.caption(f"Profile fusion → {beh.get('fused_risk_name')} (dev {float(beh.get('behavioral_deviation', 0)):.2f})")
        expl = last.get("explanation") or {}
        st.caption(str(expl.get("anomaly_context", ""))[:220])
    else:
        st.caption("Emit a simulation batch (Dashboard → Simulation) to populate scores.")

    st.subheader("Live logs (scored history)")
    is_lt = str(st.session_state.get("aura_theme", "dark")) == "light"
    bg_log = "#f8fafc" if is_lt else "#0d1117"
    tx_log = "#0f172a" if is_lt else "#e6edf3"
    show = list(reversed(filtered[-80:])) if filtered else []
    for row in show:
        risk = row.get("risk_class", "Unknown")
        rc = RISK_COLORS.get(risk, "#94a3b8")
        al = str(row.get("demo_alert") or "")
        beh = row.get("behavioral") or {}
        fused = beh.get("fused_risk_name") if isinstance(beh, dict) else None
        beh_bit = f" · profile→{fused}" if fused and fused != risk else ""
        st.markdown(
            f"<div style='border:1px solid {rc};padding:8px;border-radius:6px;margin-bottom:6px;background:{bg_log};color:{tx_log};'>"
            f"<strong>{row.get('timestamp')}</strong> · {row.get('cve_id')} · "
            f"<span style='color:{rc};font-weight:700;'>{risk}</span> · CVSS {row.get('cvss_predicted')} · "
            f"IF {row.get('iforest_label', '—')} · LSTM {row.get('lstm_risk_class', '—')}"
            f"{beh_bit}{(' · ' + al) if al else ''}</div>",
            unsafe_allow_html=True,
        )


def render_ai_insights(df_raw: pd.DataFrame, filtered: List[Dict[str, Any]]) -> None:
    st.caption("Dataset CVSS→risk vs scored history (when present).")
    tpl = _plotly_template()
    c1, c2 = st.columns(2)
    with c1:
        daily = traffic_timeseries(df_raw, freq="D")
        fig = px.area(daily, x="timestamp", y="events", template=tpl)
        fig.update_layout(height=320, title="Publication-time event trend")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cw = cwe_distribution(df_raw, top_n=12)
        fig2 = px.bar(cw, x="cwe_name", y="events", template=tpl)
        fig2.update_layout(height=320, title="CWE distribution (top)", xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        summary = compute_dataset_summary(df_raw)
        rc_df = pd.DataFrame(list(summary.risk_counts.items()), columns=["label", "count"])
        fig3 = px.pie(rc_df, names="label", values="count", color="label", color_discrete_map=RISK_COLORS)
        fig3.update_layout(height=320, title="CVSS-derived risk mix", template=tpl)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        if filtered:
            dfh = pd.DataFrame(filtered)
            fig4 = px.histogram(dfh, x="risk_class", color="risk_class", color_discrete_map=RISK_COLORS)
            fig4.update_layout(height=320, title="Model risk mix (history)", template=tpl)
        else:
            fig4 = px.bar(rc_df, x="label", y="count", color="label", color_discrete_map=RISK_COLORS)
            fig4.update_layout(height=320, title="Risk mix (dataset-only)", template=tpl)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Model confidence & feature importance")
    imp = load_classifier_feature_importance(15)
    if not imp.empty:
        fig5 = px.bar(imp, x="importance", y="feature", orientation="h", template=tpl)
        fig5.update_layout(height=420, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("Train models to show feature importance.")

    if filtered:
        last = filtered[-1]
        conf = model_confidence_from_probabilities(last.get("risk_probabilities"))
        st.metric("Latest ensemble confidence (max class probability)", f"{100 * conf:.1f}%")


def render_threat_intel(df_raw: pd.DataFrame) -> None:
    st.caption("Correlate dataset CVE rows with NVD (when reachable). Baseline CVSS always comes from Dataset-Attacks-Firewall.csv.")
    cve_pick = st.selectbox(
        "Pick a CVE from dataset",
        options=sorted(df_raw["Data"].astype(str).unique())[:400],
        key="aura_intel_cve",
    )
    if st.button("Fetch / enrich", key="aura_intel_fetch"):
        row = df_raw[df_raw["Data"].astype(str) == str(cve_pick)].iloc[0]
        with st.spinner("Querying NVD (cached 24h)..."):
            info = correlate_dataset_cve(str(row["Data"]), float(row["cvss"]))
        st.json(info)


def render_architecture_full(
    df_raw: pd.DataFrame,
    mode: str,
    model_ready: bool,
) -> None:
    st.caption("Training metrics + dataset throughput (see architecture strip above).")
    summary = compute_dataset_summary(df_raw)
    risk_m = load_risk_metrics_blob()
    cvss_m = load_cvss_metrics_blob()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline data rows", f"{summary.row_count:,}")
    c2.metric("Classifier test accuracy", _fmt_metric(risk_m, "classifier_test", "accuracy"))
    c3.metric("CVSS test RMSE", _fmt_scalar(cvss_m.get("cvss_rmse_test")))
    lstm = risk_m.get("lstm_test") if isinstance(risk_m.get("lstm_test"), dict) else {}
    c4.metric("LSTM test F1 (macro)", _fmt_scalar(lstm.get("f1_macro")))

    st.subheader("Stored training payload")
    if METRICS_PATH.exists():
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    elif risk_m:
        st.json({"risk_model_metrics": risk_m, "cvss_model_metrics": cvss_m})
    else:
        st.info("Run training to populate detailed metrics.")


def _fmt_metric(blob: Dict[str, Any], key: str, sub: str) -> str:
    inner = blob.get(key)
    if isinstance(inner, dict) and sub in inner:
        return f"{float(inner[sub]):.4f}"
    return "n/a"


def _fmt_scalar(val: Any) -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def render_analytics(df_raw: pd.DataFrame) -> None:
    st.caption("Publication timeline + CWE concentration (bundled CSV).")
    tpl = _plotly_template()
    art = analytics_risk_timeline(df_raw)
    ts_col = "_ts" if "_ts" in art.columns else art.columns[0]
    px_df = art.rename(columns={ts_col: "timestamp"})
    melt_cols = [c for c in px_df.columns if c != "timestamp"]
    if melt_cols:
        long = px_df.melt(id_vars=["timestamp"], value_vars=melt_cols, var_name="risk", value_name="count")
        fig = px.line(long, x="timestamp", y="count", color="risk", color_discrete_map=RISK_COLORS, template=tpl)
        fig.update_layout(height=380, title="Risk class counts over time (daily)")
        st.plotly_chart(fig, use_container_width=True)

    tw = threat_watchlist(df_raw, top_n=20)
    st.subheader("Threat pattern concentration (high CVSS sources)")
    st.dataframe(tw, use_container_width=True, hide_index=True)

    st.subheader("Time-based anomaly proxy (weekly aggregates)")
    ts = traffic_timeseries(df_raw, freq="W")
    fig2 = px.bar(ts, x="timestamp", y="events", template=tpl)
    fig2.update_layout(height=320, title="Weekly event totals")
    st.plotly_chart(fig2, use_container_width=True)


def render_firewall_controls(df_raw: pd.DataFrame) -> None:
    st.caption(
        "Policy + OS firewall: blocks use Windows `netsh advfirewall` or Linux `iptables`/`nft`. "
        "Set AURA_FIREWALL_DRY_RUN=false and run with admin/root for live enforcement."
    )
    st.toggle("Block path for Malicious/Critical (policy)", key="aura_fw_block_malicious")
    st.toggle("Generate analyst alerts for all scored events", key="aura_fw_alert_all")
    st.toggle("Attempt real OS block on simulation when Malicious/Critical (requires privileges)", key="aura_fw_live_block")
    st.slider("Sensitivity (minimum severity to surface)", 0, 100, key="aura_sensitivity")
    if st.button("Apply policy acknowledgement"):
        st.session_state.aura_policy_note = datetime.now(timezone.utc).isoformat()
        st.success("Policy update recorded in session.")
    st.caption(f"Last acknowledgement: {st.session_state.get('aura_policy_note') or '—'}")

    fm = get_firewall_manager()
    st.subheader("Firewall engine status")
    st.json(
        {
            "dry_run": fm.state().dry_run,
            "platform": fm.state().platform_name,
            "blocked_session": fm.state().blocked_ips,
            "recent_actions": fm.state().last_actions[-8:],
        }
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        block_ip = st.text_input("Block IPv4 (admin)", key="aura_fw_block_ip", placeholder="203.0.113.10")
    with c2:
        if st.button("Apply OS block", type="primary", key="aura_fw_apply_block"):
            if block_ip.strip():
                r = fm.block_observed_ip(block_ip.strip(), "Critical", "manual_ui")
                st.success(r.message) if r.ok else st.error(r.message)
    with c3:
        if st.button("Remove OS block", key="aura_fw_remove_block"):
            if block_ip.strip():
                r = fm.unblock(block_ip.strip())
                st.success(r.message) if r.ok else st.error(r.message)

    st.subheader("Threat watchlist (detected high-risk IPs from dataset)")
    st.dataframe(threat_watchlist(df_raw, top_n=30), use_container_width=True, hide_index=True)


def render_logs(
    df_raw: pd.DataFrame,
    filtered: List[Dict[str, Any]],
    search: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    for row in reversed(filtered[-500:]):
        ts = row.get("timestamp", "")
        et = str(row.get("cve_id", ""))
        src = str(row.get("source_prefix", ""))
        st_ = str(row.get("risk_class", ""))
        anom = row.get("anomaly")
        action = "Allow / monitor"
        if st_ in ("Malicious", "Critical"):
            action = "Block / quarantine"
        elif st_ == "Suspicious":
            action = "Throttle / inspect"
        if anom == -1:
            action += " · anomaly review"
        rec = {
            "Timestamp": ts,
            "Event Type": et,
            "Source": src,
            "Status": st_,
            "Action": action,
        }
        if match_search(rec, search):
            rows.append(rec)
    if not rows:
        for r in reversed(build_dataset_log_rows(df_raw, n=300)):
            rec = {
                "Timestamp": str(r.get("timestamp")),
                "Event Type": r.get("event_type"),
                "Source": r.get("source"),
                "Status": r.get("status"),
                "Action": r.get("action"),
            }
            if match_search(rec, search):
                rows.append(rec)

    df_logs = pd.DataFrame(rows)
    st.dataframe(df_logs, use_container_width=True, hide_index=True, height=420)
    if rows:
        st.download_button(
            "Export filtered log view (CSV)",
            data=df_logs.to_csv(index=False).encode("utf-8"),
            file_name="aura_logs_export.csv",
            mime="text/csv",
        )


def render_mobile(
    df_raw: pd.DataFrame,
    filtered: List[Dict[str, Any]],
) -> None:
    st.caption("Minimal SOC glance — same data substrate, condensed layout.")
    summary = compute_dataset_summary(df_raw)
    st.metric("Records", f"{summary.row_count:,}")
    st.metric("Normal ratio (CVSS baseline)", f"{100 * summary.normal_ratio:.1f}%")
    if filtered:
        last_rows = []
        for row in filtered[-5:]:
            last_rows.append(
                {
                    "time": row.get("timestamp"),
                    "cve": row.get("cve_id"),
                    "risk": row.get("risk_class"),
                }
            )
        st.subheader("Recent alerts")
        st.dataframe(pd.DataFrame(last_rows), use_container_width=True, hide_index=True)
    else:
        st.subheader("Recent alerts (dataset tail)")
        st.dataframe(pd.DataFrame(build_dataset_log_rows(df_raw, n=5)), use_container_width=True, hide_index=True)


def render_ai_analyst_chat() -> None:
    """Intelligent AI Analyst Chat Interface"""
    st.markdown("### AURA AI Analyst")
    st.caption("Ask questions about threats, IPs, CVEs, behaviors, and firewall decisions. AI will analyze using all ML models.")
    
    try:
        from chatbot_engine import AuraAiAnalyst
    except ImportError:
        st.error("Chatbot engine not available. Check installation.")
        return
    
    # Initialize chatbot
    if "analyst" not in st.session_state:
        with st.spinner("Initializing AI Analyst..."):
            analyst = AuraAiAnalyst()
            if analyst.initialize():
                st.session_state.analyst = analyst
            else:
                st.error("Failed to initialize AI Analyst")
                return
    else:
        analyst = st.session_state.analyst
    
    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.write(f"[YOU] {msg['content']}")
            else:
                st.write(f"[AURA] {msg['content']}")
    
    # Input area
    st.divider()
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask about threats:",
            placeholder="e.g., 'Analyze IP 192.168.1.1' or 'Explain CVE-2024-1234'",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_btn = st.button("Send", use_container_width=True, key="chat_send")
    
    if send_btn and user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Get AI response
        with st.spinner("Analyzing..."):
            try:
                result = analyst.analyze_query(user_query)
                response_text = result.get("response", "Unable to process query")
                
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
                # Show detailed analysis if available
                if result.get("ml_analysis"):
                    with st.expander("[ML_ANALYSIS] Detailed ML Analysis", expanded=False):
                        st.json(result["ml_analysis"])
                
                if result.get("ml_model_outputs"):
                    with st.expander("Models Used", expanded=False):
                        for model, desc in result["ml_model_outputs"].items():
                            st.write(f"**{model}:** {desc}")
                
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    # Example queries
    if not st.session_state.chat_history:
        st.info("""
        **Example Questions:**
        - "Analyze IP 192.168.1.100"
        - "Explain CVE-2024-1234"
        - "What are the latest alerts?"
        - "Show behavioral patterns"
        - "How do firewall rules work?"
        """)


def render_anomaly_check(
    df_raw: pd.DataFrame,
    predictor: Optional[AuraPredictor],
    model_ready: bool,
) -> None:
    """Real-time anomaly check interface for threat verification"""
    
    if not model_ready or predictor is None:
        st.error("Models not available. Run: python train_model.py")
        return
    
    st.markdown("### Verify Anomalies in Real-Time Conditions")
    st.markdown("""
    Input custom data or select sample records to verify whether the trained model 
    correctly detects anomalies in operational conditions.
    """)
    
    # Input method selection
    input_method = st.radio(
        "Select input method",
        ["Sample from Dataset", "Custom Input", "Bulk Upload"],
        horizontal=True,
        key="anom_input_method"
    )
    
    record_to_analyze = None
    
    if input_method == "Sample from Dataset":
        st.markdown("#### Select Sample Record")
        sample_idx = st.slider(
            "Record index",
            min_value=0,
            max_value=len(df_raw) - 1,
            value=0,
            key="anom_sample_idx"
        )
        record_to_analyze = df_raw.iloc[sample_idx].to_dict()
        
        with st.expander("View record data"):
            st.json(record_to_analyze)
    
    elif input_method == "Custom Input":
        st.markdown("#### Enter Custom Values")
        st.info("Leave fields blank to use dataset defaults")
        
        col1, col2 = st.columns(2)
        custom_data = {}
        
        for idx, col in enumerate(df_raw.columns):
            if idx % 2 == 0:
                with col1:
                    val = st.text_input(
                        f"{col}",
                        value=str(df_raw.iloc[0][col]),
                        key=f"anom_col_{col}"
                    )
                    try:
                        custom_data[col] = float(val) if '.' in val else int(val)
                    except ValueError:
                        custom_data[col] = val
            else:
                with col2:
                    val = st.text_input(
                        f"{col}",
                        value=str(df_raw.iloc[0][col]),
                        key=f"anom_col_{col}"
                    )
                    try:
                        custom_data[col] = float(val) if '.' in val else int(val)
                    except ValueError:
                        custom_data[col] = val
        
        record_to_analyze = custom_data
    
    else:  # Bulk Upload
        st.markdown("#### Upload CSV File")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            try:
                bulk_df = pd.read_csv(uploaded_file, low_memory=False)
                record_to_analyze = bulk_df.iloc[0].to_dict()
                st.success(f"Loaded {len(bulk_df)} records. Analyzing first record.")
                with st.expander("Uploaded dataset summary", expanded=True):
                    st.write({
                        "rows": len(bulk_df),
                        "columns": list(bulk_df.columns),
                        "sample_types": {col: str(dtype) for col, dtype in bulk_df.dtypes.items()},
                    })
                    st.dataframe(bulk_df.head(5))
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Run analysis
    if st.button("Check for Anomalies", type="primary", use_container_width=True, key="anom_check_btn"):
        if record_to_analyze is None:
            st.warning("No data to analyze")
            return
        
        with st.spinner("Analyzing record..."):
            try:
                predictions = predictor.predict_records(
                    [record_to_analyze],
                    include_explanation=False,
                    use_llm=False
                )
                pred = predictions[0]
                
                # Display results
                st.markdown("### Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    risk_class = pred.get("risk_class", "Unknown")
                    color_map = {"Safe": "green", "Suspicious": "orange", "Malicious": "red", "Critical": "darkred"}
                    color = color_map.get(risk_class, "gray")
                    st.markdown(f"<div style='text-align:center; padding: 20px; background-color: {color}; border-radius: 10px; color: white;'>"
                               f"<h3>{risk_class}</h3><p>Classification</p></div>", unsafe_allow_html=True)
                
                with col2:
                    cvss = pred.get("cvss_predicted", 0.0)
                    st.metric("CVSS Score", f"{cvss:.2f}/10.0")
                
                with col3:
                    anomaly_flag = pred.get("anomaly_score_flag", 0)
                    is_anomaly = "ANOMALOUS" if anomaly_flag < 0 else "NORMAL"
                    color = "red" if anomaly_flag < 0 else "green"
                    st.markdown(f"<div style='text-align:center; padding: 20px; color: {color};'>"
                               f"<h3>{is_anomaly}</h3><p>Anomaly Detection</p></div>", unsafe_allow_html=True)
                
                with col4:
                    conf = pred.get("confidence", 0.0)
                    st.metric("Confidence", f"{conf:.1%}")
                
                # Detailed breakdown
                st.markdown("### Detailed Breakdown")
                
                tab1, tab2, tab3 = st.tabs(["Classification", "Anomaly Score", "Risk Probabilities"])
                
                with tab1:
                    st.write(f"**Classification**: {risk_class}")
                    st.write(f"**CVSS Severity**: {'Critical' if cvss >= 9.0 else 'High' if cvss >= 7.0 else 'Medium' if cvss >= 4.0 else 'Low'}")
                    
                    interpretation = []
                    if risk_class == "Critical":
                        interpretation.append("Status: IMMEDIATE ACTION REQUIRED")
                        interpretation.append("Recommendation: Isolate asset, escalate to SOC, preserve evidence")
                    elif risk_class == "Malicious":
                        interpretation.append("Status: CONFIRMED THREAT")
                        interpretation.append("Recommendation: Block traffic, investigate source, review logs")
                    elif risk_class == "Suspicious":
                        interpretation.append("Status: REQUIRES INVESTIGATION")
                        interpretation.append("Recommendation: Enable enhanced monitoring, correlate with alerts")
                    else:
                        interpretation.append("Status: NORMAL BEHAVIOR")
                        interpretation.append("Recommendation: Continue baseline monitoring")
                    
                    st.info("\n".join(interpretation))
                
                with tab2:
                    anomaly_score = pred.get("anomaly_score", 0.0)
                    st.write(f"**Anomaly Score**: {anomaly_score:.4f}")
                    st.write(f"**Flag**: {'Anomalous (-1)' if anomaly_flag < 0 else 'Normal (1)'}")
                    st.write("Score interpretation: Negative values indicate anomalous patterns")
                
                with tab3:
                    probs = pred.get("risk_probabilities", {})
                    if probs:
                        prob_data = pd.DataFrame([probs]).T
                        prob_data.columns = ["Probability"]
                        st.bar_chart(prob_data)
                    else:
                        st.write("Probability data not available")
                
                # Save result
                if st.button("Save to Threat History"):
                    alert_record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "risk_class": risk_class,
                        "cvss_predicted": cvss,
                        "anomaly_score_flag": anomaly_flag,
                        "source": "anomaly_check"
                    }
                    st.session_state.threat_history.append(alert_record)
                    st.success("Result saved to threat history")
            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


def run_retrain(cached_predictor: Any, logger: Any) -> None:
    with st.spinner("Retraining models from the CSV..."):
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "train_model.py")],
                check=True,
                cwd=str(ROOT),
            )
            st.success("Retraining finished. Reload the page to refresh cached models.")
            try:
                cached_predictor.clear()
            except Exception:  # noqa: BLE001
                logger.warning("Could not clear predictor cache; reload manually.")
        except subprocess.CalledProcessError as exc:
            st.error(f"Retraining failed: {exc}")
