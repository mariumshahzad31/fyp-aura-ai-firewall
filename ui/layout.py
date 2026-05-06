"""Shell layout: sidebar chrome, header row, persistent architecture strip."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import streamlit as st

from ui.styles import inject_global_styles


ARCH_STAGES: List[Tuple[str, List[str]]] = [
    (
        "Input",
        ["User Behavior Data", "Network Traffic", "Device Activity", "Session Patterns"],
    ),
    (
        "Preprocessing",
        ["Data Cleaning", "Feature Extraction", "Session Structuring"],
    ),
    (
        "AI Engine",
        ["Pattern Detection", "LSTM (Temporal Analysis)", "Isolation Forest (Outlier Detection)"],
    ),
    (
        "Decision",
        ["Anomaly Detection", "Risk Evaluation", "Threat Classification"],
    ),
    (
        "Response",
        [
            "Alert Generation",
            "OS Firewall (iptables / Windows)",
            "REST API + Mobile Clients",
            "Logging",
        ],
    ),
]


def init_shell_state() -> None:
    if "threat_history" not in st.session_state:
        st.session_state.threat_history = []
    if "profiles" not in st.session_state:
        st.session_state.profiles = {}
    if "sim_offset" not in st.session_state:
        st.session_state.sim_offset = 0
    if "aura_nav" not in st.session_state:
        st.session_state.aura_nav = "Dashboard"
    if "aura_search" not in st.session_state:
        st.session_state.aura_search = ""
    if "aura_theme" not in st.session_state:
        st.session_state.aura_theme = "light"
    if "aura_fw_block_malicious" not in st.session_state:
        st.session_state.aura_fw_block_malicious = True
    if "aura_fw_alert_all" not in st.session_state:
        st.session_state.aura_fw_alert_all = True
    if "aura_sensitivity" not in st.session_state:
        st.session_state.aura_sensitivity = 50
    if "aura_policy_note" not in st.session_state:
        st.session_state.aura_policy_note = ""
    if "aura_arch_active" not in st.session_state:
        st.session_state.aura_arch_active = 2


def sidebar_branding() -> None:
    st.markdown(
        '<p class="aura-brand-title">AURA</p>'
        '<p class="aura-brand-sub">AI-Driven Behavioral Firewall</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


def sidebar_navigation() -> str:
    pages = [
        "Dashboard",
        "Anomaly Check",
        "Live Monitoring",
        "AI Insights",
        "AI Analyst Chat",
        "Threat Intelligence",
        "System Architecture",
        "Analytics",
        "Firewall Controls",
        "Logs",
    ]
    cur = st.session_state.get("aura_nav", "Dashboard")
    idx = pages.index(cur) if cur in pages else 0
    choice = st.radio("Navigation", pages, index=idx, label_visibility="collapsed", key="aura_nav_radio")
    st.session_state.aura_nav = choice
    return choice


def sidebar_footer(
    system_status_text: str,
    status_color: str,
    model_state: str,
    on_run_scan: Callable[[], None],
    on_refresh: Callable[[], None],
) -> None:
    st.markdown("---")
    st.caption("System status")
    st.markdown(
        f'<span class="aura-status-pill" style="border-color:{status_color};color:{status_color};">'
        f"{system_status_text}</span>",
        unsafe_allow_html=True,
    )
    st.caption("Model state")
    st.markdown(f'<p class="aura-muted" style="margin:0;">{model_state}</p>', unsafe_allow_html=True)
    st.caption("Quick actions")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run Scan", use_container_width=True, key="qa_scan"):
            on_run_scan()
    with c2:
        if st.button("Refresh Data", use_container_width=True, key="qa_refresh"):
            on_refresh()


def header_bar(page_title: str) -> None:
    left, right = st.columns([4.2, 1.0])
    with left:
        st.markdown(f'<p class="aura-page-title">{page_title}</p>', unsafe_allow_html=True)
    with right:
        cur = st.session_state.get("aura_theme", "light")
        is_light = st.toggle(
            "Light mode",
            value=(cur == "light"),
            key="aura_theme_toggle",
            help="Global light/dark UI",
        )
        st.session_state.aura_theme = "light" if is_light else "dark"


def architecture_panel(
    compact: bool,
    mode: str,
    active_stage_index: Optional[int] = None,
) -> None:
    """Render pipeline: Input → … → Response. Highlights one stage when provided."""
    title = "System Architecture" + (" (compact)" if compact else "")
    st.markdown(f'<div class="aura-pipe-title">{title}</div>', unsafe_allow_html=True)
    if active_stage_index is None:
        active_stage_index = int(st.session_state.get("aura_arch_active", 2)) % len(ARCH_STAGES)

    parts: List[str] = []
    for idx, (name, items) in enumerate(ARCH_STAGES):
        active = idx == active_stage_index
        cls = "aura-stage active" if active else "aura-stage"
        li = "".join(f"<li title=\"{mode}: {name}\">{it}</li>" for it in items)
        parts.append(
            f'<div class="{cls}"><h5>{name}</h5><ul>{li}</ul></div>'
        )
    arrow = '<span class="aura-arrow">→</span>'
    inner = arrow.join(parts)
    st.markdown(
        f'<div class="aura-pipe-wrap"><div class="aura-flow">{inner}</div></div>',
        unsafe_allow_html=True,
    )
