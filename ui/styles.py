"""Global dashboard CSS — theme-aware contrast, spacing, no box shadows."""

from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    theme = st.session_state.get("aura_theme", "light")
    css = _light_css() if theme == "light" else _dark_css()
    st.markdown(css, unsafe_allow_html=True)


def _dark_css() -> str:
    return """
<style>
  /* ROOT DARK THEME - COMPLETE VISIBILITY */
  .stApp { background-color: #0E1117; color: #e5e7eb; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
  header[data-testid="stHeader"] { background: transparent; }

  [data-testid="stSidebar"] {
    background: #0E1117;
    border-right: 1px solid #1f2937;
  }

  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] h4,
  [data-testid="stSidebar"] h5,
  [data-testid="stSidebar"] h6 {
    color: #e5e7eb !important;
  }

  /* TEXT & LABELS - DARK MODE */
  body, h1, h2, h3, h4, h5, h6, p, span, label, div { color: #e5e7eb !important; }
  .aura-brand-title { font-size: 1.35rem; font-weight: 700; color: #f9fafb; }
  .aura-brand-sub { font-size: 0.78rem; color: #cbd5e1; }
  .aura-page-title { font-size: 1.45rem; font-weight: 700; color: #f9fafb; }
  .aura-muted { color: #94a3b8; font-size: 0.85rem; }

  /* STREAMLIT INPUTS & BUTTONS - DARK MODE */
  input, textarea, select {
    color: #e5e7eb !important;
    background-color: #1e293b !important;
    border: 1px solid #374151 !important;
    caret-color: #e5e7eb !important;
  }
  input::placeholder, textarea::placeholder { color: #6b7280 !important; }

  .stButton > button {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
  }
  .stButton > button:hover {
    background-color: #2563eb !important;
  }

  .stNumberInput input, .stTextInput input, .stTextArea textarea {
    color: #e5e7eb !important;
    background-color: #1e293b !important;
    border-color: #374151 !important;
  }

  .stSelectbox div, .stMultiSelect div {
    background-color: #1e293b !important;
    border-color: #374151 !important;
    color: #e5e7eb !important;
  }

  .stMetric { color: #e5e7eb !important; }
  .stMetric label { color: #cbd5e1 !important; }

  /* AURA COMPONENTS - DARK MODE */
  .aura-card,
  .aura-pipe-wrap,
  .aura-stage,
  .aura-dash-hero,
  .aura-phase,
  .aura-kpi,
  div[data-testid="stMetric"] {
    box-shadow: none !important;
    filter: none !important;
    border: none !important;
    border-left: none !important;
    border-right: none !important;
    border-top: none !important;
    border-bottom: none !important;
    outline: none !important;
  }

  /* HERO CARD */
  .aura-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    color: #e5e7eb !important;
  }

  .aura-card h4 {
    margin: 0 0 6px 0;
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #8b949e;
  }

  .aura-card .val {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f0f6fc;
  }

  .aura-card .sub {
    font-size: 0.8rem;
    color: #8b949e;
  }

  /* PIPELINE */
  .aura-pipe-wrap {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 10px 12px;
  }

  .aura-stage {
    flex: 1 1 120px;
    min-width: 100px;
    border-radius: 8px;
    padding: 8px 10px;
    border: 1px solid #30363d;
    background: #0d1117;
  }

  .aura-stage.active { border-color: #58a6ff; }

  /* METRICS */
  div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    padding: 8px 12px;
    border-radius: 8px;
  }

  /* DIVIDERS & SEPARATORS */
  hr { border-color: #30363d !important; }
  .stDivider { color: #30363d !important; }

  /* TABS */
  [data-testid="stTabs"] button { color: #e5e7eb !important; }
  [data-testid="stTabs"] button:hover { background-color: #1f2937 !important; }

  /* EXPANDERS */
  [data-testid="stExpander"] { background-color: #161b22 !important; border-color: #30363d !important; }
  [data-testid="stExpander"] p, [data-testid="stExpander"] span { color: #e5e7eb !important; }

  /* CONTAINERS */
  [data-testid="stContainer"] { background-color: #0E1117 !important; }
  [data-testid="stColumn"] { background-color: transparent !important; }

  /* REMOVE ALL DECORATIVE PSEUDO-ELEMENTS */
  .aura-card::before,
  .aura-card::after,
  .aura-dash-hero::before,
  .aura-dash-hero::after,
  .aura-phase::before,
  .aura-phase::after,
  .aura-kpi::before,
  .aura-kpi::after {
    content: none !important;
    display: none !important;
    background: none !important;
    box-shadow: none !important;
    border: none !important;
    height: 0 !important;
    width: 0 !important;
  }

  /* STREAMLIT MARKDOWN - DARK MODE */
  div[data-testid="stMarkdownContainer"] {
    border: none !important;
    border-left: none !important;
    border-right: none !important;
    border-top: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #e5e7eb !important;
  }

  /* CHARTS & TABLES - DARK MODE */
  .stPlotlyChart, .stDataFrame, table { color: #e5e7eb !important; }
  table, table tr, table td, table th {
    background-color: #1f2937 !important;
    color: #e5e7eb !important;
    border-color: #374151 !important;
  }
  thead th { background-color: #374151 !important; color: #e5e7eb !important; }

  /* TOGGLE & CHECKBOXES - DARK MODE */
  .stCheckbox label, .stRadio label, .stSelectbox label, .stMultiSelect label {
    color: #e5e7eb !important;
  }

  /* ALERTS & MESSAGES */
  [data-testid="stAlert"] { background-color: #1f2937 !important; border-color: #374151 !important; color: #e5e7eb !important; }

  /* GLOBAL TEXT CLEAN */
  * { text-shadow: none !important; }

  /* FOCUS GLOW KILL */
  button:focus, button:active, button:focus-visible,
  input:focus, textarea:focus, select:focus,
  .stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
  }

  /* PREVENT BLUE LINES FROM ANY SOURCE */
  [style*="border-left"],
  [style*="border-right"],
  [style*="#2563eb"],
  [style*="#58a6ff"] {
    border-left: none !important;
    border-right: none !important;
  }

</style>
"""


def _light_css() -> str:
    return """
<style>
  /* ROOT LIGHT THEME - COMPLETE VISIBILITY */
  .stApp { background-color: #ffffff; color: #111827; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
  header[data-testid="stHeader"] { background: transparent; }

  [data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
  }

  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] h4,
  [data-testid="stSidebar"] h5,
  [data-testid="stSidebar"] h6 {
    color: #111827 !important;
  }

  /* TEXT & LABELS - LIGHT MODE */
  body, h1, h2, h3, h4, h5, h6, p, span, label, div { color: #111827 !important; }
  .aura-brand-title { font-size: 1.35rem; font-weight: 700; color: #0f172a; }
  .aura-brand-sub { font-size: 0.78rem; color: #475569; }
  .aura-page-title { font-size: 1.45rem; font-weight: 700; color: #0f172a; }
  .aura-muted { color: #64748b; font-size: 0.85rem; }

  /* STREAMLIT INPUTS & BUTTONS - LIGHT MODE */
  input, textarea, select {
    color: #0f172a !important;
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    caret-color: #0f172a !important;
  }
  input::placeholder, textarea::placeholder { color: #94a3b8 !important; }

  .stButton > button {
    background-color: #1e40af !important;
    color: #ffffff !important;
    border: none !important;
  }
  .stButton > button:hover {
    background-color: #1e3a8a !important;
  }

  .stNumberInput input, .stTextInput input, .stTextArea textarea {
    color: #0f172a !important;
    background-color: #f8fafc !important;
    border-color: #cbd5e1 !important;
  }

  .stSelectbox div, .stMultiSelect div {
    background-color: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
  }

  .stMetric { color: #0f172a !important; }
  .stMetric label { color: #475569 !important; }

  /* AURA COMPONENTS - LIGHT MODE */
  .aura-card,
  .aura-pipe-wrap,
  .aura-stage,
  .aura-dash-hero,
  .aura-phase,
  .aura-kpi,
  div[data-testid="stMetric"] {
    box-shadow: none !important;
    filter: none !important;
    border: none !important;
    border-left: none !important;
    border-right: none !important;
    border-top: none !important;
    border-bottom: none !important;
    outline: none !important;
  }

  .aura-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    color: #0f172a !important;
  }

  .aura-card h4 { color: #475569 !important; }
  .aura-card .val { color: #0f172a !important; }
  .aura-card .sub { color: #64748b !important; }

  /* DIVIDERS & SEPARATORS */
  hr { border-color: #e5e7eb !important; }
  .stDivider { color: #e5e7eb !important; }

  /* TABS */
  [data-testid="stTabs"] button { color: #111827 !important; }
  [data-testid="stTabs"] button:hover { background-color: #f1f5f9 !important; }

  /* EXPANDERS */
  [data-testid="stExpander"] { background-color: #ffffff !important; border-color: #e2e8f0 !important; }
  [data-testid="stExpander"] p, [data-testid="stExpander"] span { color: #111827 !important; }

  /* CONTAINERS */
  [data-testid="stContainer"] { background-color: #ffffff !important; }
  [data-testid="stColumn"] { background-color: transparent !important; }

  /* REMOVE ALL DECORATIVE PSEUDO-ELEMENTS */
  .aura-card::before,
  .aura-card::after,
  .aura-dash-hero::before,
  .aura-dash-hero::after,
  .aura-phase::before,
  .aura-phase::after,
  .aura-kpi::before,
  .aura-kpi::after {
    content: none !important;
    display: none !important;
    background: none !important;
    box-shadow: none !important;
    border: none !important;
    height: 0 !important;
    width: 0 !important;
  }

  /* STREAMLIT MARKDOWN - LIGHT MODE */
  div[data-testid="stMarkdownContainer"] {
    border: none !important;
    border-left: none !important;
    border-right: none !important;
    border-top: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #0f172a !important;
  }

  /* CHARTS & TABLES - LIGHT MODE */
  .stPlotlyChart, .stDataFrame, table { color: #0f172a !important; }
  table, table tr, table td, table th {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-color: #e5e7eb !important;
  }
  thead th { background-color: #f1f5f9 !important; color: #0f172a !important; }

  /* TOGGLE & CHECKBOXES - LIGHT MODE */
  .stCheckbox label, .stRadio label, .stSelectbox label, .stMultiSelect label {
    color: #0f172a !important;
  }

  /* ALERTS & MESSAGES */
  [data-testid="stAlert"] { background-color: #f8fafc !important; border-color: #e5e7eb !important; color: #0f172a !important; }

  /* GLOBAL TEXT CLEAN */
  * { text-shadow: none !important; }

  /* FOCUS GLOW KILL */
  button:focus, button:active, button:focus-visible,
  input:focus, textarea:focus, select:focus,
  .stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
  }

  /* PREVENT BLUE LINES FROM ANY SOURCE */
  [style*="border-left"],
  [style*="border-right"],
  [style*="#2563eb"],
  [style*="#58a6ff"] {
    border-left: none !important;
    border-right: none !important;
  }

</style>
"""


def hide_streamlit_chrome() -> None:
    st.markdown(
        """
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
        """,
        unsafe_allow_html=True,
    )