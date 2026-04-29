# AURA Production Architecture

## Overview

AURA is an AI-driven behavioral firewall stack: **dataset-backed training** (CSV-only features), **ensemble ML** (RandomForest, Isolation Forest, CVSS regression, optional LSTM), **OS firewall integration**, **live packet capture** with dataset-aligned proxy rows, **LLM explanations**, **REST API** with JWT roles, **behavioral adaptive profiles**, and **NVD threat intelligence** enrichment.

## Data Flow (Text Diagram)

```
Dataset-Attacks-Firewall.csv
        │
        ▼
┌───────────────────┐
│  preprocessing    │  build_feature_frame, chronological split
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐     ┌─────────────────────┐
│  train_model.py   │────▶│  models/*.pkl, .h5 │
└───────────────────┘     └─────────────────────┘
          │
          ▼
┌───────────────────┐     ┌─────────────────────┐
│  AuraPredictor    │◀───│  live_bridge /      │
│  predict_*        │    │  packet monitor     │
└─────────┬─────────┘    └─────────────────────┘
          │
          ├──▶ LLM layer (optional) ──▶ structured explanations
          ├──▶ Behavioral profiles (EWMA + deviation fusion)
          ├──▶ Threat intel (NVD CVE) correlation
          │
          ▼
┌───────────────────┐     ┌─────────────────────┐
│  FirewallManager  │     │  alert_bus JSONL    │
│  (Win/Linux)      │     │  logs/*             │
└───────────────────┘     └─────────────────────┘
          │
          ▼
┌───────────────────┐     ┌─────────────────────┐
│  Streamlit UI     │     │  FastAPI `/api/v1` │
└───────────────────┘     └─────────────────────┘
```

## Components

| Module | Role |
|--------|------|
| `utils/preprocessing.py` | Single source of truth for CSV schema and features |
| `utils/prediction.py` | Loads joblib/keras; exposes `predict_records` |
| `utils/live_bridge.py` | Maps live IPs to **full** dataset rows (strict feature lineage) |
| `utils/packet_monitor.py` | Scapy sniff + dataset-proxy records |
| `utils/firewall.py` | Windows `netsh` / Linux `iptables`/`nft` |
| `utils/llm_explainer.py` | OpenAI-compatible JSON + deterministic fallback |
| `utils/behavioral_profiles.py` | Per-key EWMA + optional SGD auxiliary |
| `utils/threat_intel.py` | NVD CVE 2.0 API (cached) |
| `api/main.py` | FastAPI: auth, predict, alerts, logs, intel, firewall |

## Deployment

- **Docker**: `docker compose up --build` (API on `:8000`, UI on `:8501`).
- **Host**: `uvicorn api.main:app --host 0.0.0.0 --port 8000` and `streamlit run app.py`.
- Secrets: set `AURA_JWT_SECRET`, `OPENAI_API_KEY`, `NVD_API_KEY` as env vars (see `.env.example`).

## Backward Compatibility

- `train_model.py` and `utils/preprocessing.py` column contracts unchanged.
- `AuraPredictor.predict_records` adds optional `use_llm` and `_live_meta` stripping; existing callers unchanged.
