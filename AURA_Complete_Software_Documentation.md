AURA Complete Software Documentation, Architecture, Reverse Engineering & Functional Analysis
Document version: 1.0 (reverse-engineered from repository contents)  
Repository: `mariumshahzad31/aura` (local workspace copy)  
Primary language: Python  
Primary interfaces: Streamlit dashboard, FastAPI REST API, CLI SOC simulator, AI Analyst chatbot  
## 1. Complete Introduction
### What AURA software is
AURA (Advanced Unified Resilience Architecture) is an **AI-driven behavioral firewall stack** that combines:

- **Dataset-backed ML training** from a single canonical CSV dataset (`data/Dataset-Attacks-Firewall.csv`).
- **An ensemble detection engine** that produces risk classification, anomaly signals, and severity scores:
  - Random Forest classifier for discrete **risk class**.
  - Isolation Forest for **outlier/anomaly** detection.
  - Gradient Boosting regressor for **CVSS-like severity prediction** on a 0–10 scale.
  - Optional LSTM for **temporal/sequence-based** analysis using ordered windows.
- **Operational response** capabilities:
  - Structured alerting (JSONL ring buffer shared by UI/API/workers).
  - OS firewall integration (Windows Defender Firewall via `netsh`, Linux via `iptables` or optional `nftables`), with safe dry-run defaults.
  - Behavioral adaptive profiles (EWMA-based deviation scoring to surface “unexpected maliciousness” for previously benign entities).
  - Threat intelligence enrichment via NVD CVE 2.0 API with disk cache.
- **Multi-interface accessibility**:
  - Streamlit web dashboard (`app.py`).
  - FastAPI backend (`api/main.py`) with JWT authentication and role gating.
  - CLI step-by-step SOC simulator (`cli_engine.py`).
  - AI Analyst chatbot (`chatbot_engine.py`) with offline retrieval augmentation (RAG) and intent routing.

In practice, AURA is designed as a **blue-team defensive analytics system** that can be deployed in a lab, training/demo environment, or production-like setup (Docker Compose provided). It centralizes detection, explanation, and response workflows in one coherent architecture.

### Why it was created
Traditional security controls (signature IDS, static rule-based firewalls, and manual monitoring) are excellent for known patterns but degrade when confronted with unknown threats, low-and-slow behavioral abuse, and human-scale monitoring constraints. AURA was built to demonstrate a coherent, production-style architecture where **ML classification**, **unsupervised anomaly detection**, **behavioral baselining**, and **policy enforcement** cooperate to produce explainable, auditable security outcomes.

### Which domain it belongs to
Cybersecurity defensive analytics:

- Behavioral firewalling and response automation
- Anomaly detection and outlier scoring
- UEBA-style behavioral profiling
- Threat scoring, severity ranking, and enrichment

### Which real-world problem it solves
Operationally, AURA addresses: **how to convert raw security-relevant records into prioritized, explainable decisions and (optionally) automated response actions**.

### Why this software matters in modern systems
Modern systems are dynamic (cloud, remote endpoints, rapid change). Static rules and signatures struggle with novel behaviors and drift. AURA demonstrates an approach where “unknown” can still be surfaced via anomaly signals and adaptive baselines, with outputs consumable by humans and automation.

### Who uses it
- SOC analysts (triage, investigation, reporting)
- Security engineers (pipeline integration, response automation)
- Researchers/learners (security ML and architecture demonstrations)

### Why it is valuable
It is end-to-end: **train → load artifacts → score → explain → alert → optionally enforce firewall**, delivered via **Streamlit UI + REST API + CLI + chatbot**.

---

## 2. Full Identity of Software

### Meaning of AURA name
**AURA** stands for **Advanced Unified Resilience Architecture**.

### Exact software category
**AI-driven behavioral firewall and threat scoring platform** (feature-based scoring with enforcement hooks).

### System identity (what it is in practice)
- **Anomaly detection platform**: Isolation Forest (and generic anomaly fallback)
- **Behavioral analytics engine**: per-entity EWMA deviation fusion
- **Threat intelligence enrichment**: NVD CVE 2.0 API (cached)
- **Firewall controller**: Windows `netsh`, Linux `iptables`/optional `nftables` (safe dry-run default)

### Research prototype or deployable system?
Deployable architecture (Docker Compose + FastAPI + Streamlit), with a demo-friendly live-telemetry bridge that maps live observations to dataset-backed proxy rows to preserve feature lineage.

---

## 3. Problem Statement (Detailed)

### Traditional security systems limitations
- Rule-only firewalls lack behavioral context and adapt poorly to drift.
- Signature systems fail against new exploits, obfuscation, and benign-looking abuse.
- Manual monitoring does not scale to modern event volumes.

### Signature-based systems limitations
They only detect what is already known and encoded. Novel techniques and slight variants can bypass signatures while still being operationally harmful.

### Human monitoring limitations
Analysts cannot inspect every event; inconsistent triage and alert fatigue reduce effectiveness.

### Unknown threats and insider abuse
When payload-based indicators are missing, deviations from baseline and unusual combinations of attributes become critical early indicators.

### Need for behavior intelligence and real-time anomaly detection
Modern security requires fast scoring with explainable outputs and an operational pipeline that supports alerting and response.

---

## 4. Objectives (Detailed)

### Primary objectives
- Train and persist deployable ML artifacts.
- Score records into risk classes with probabilities.
- Detect anomalies/outliers.
- Produce explanations and operational outputs (alerts, optional firewall actions).

### Secondary objectives
- Provide multiple interfaces (UI/API/CLI/chat).
- Provide threat intel enrichment and caching.
- Operate safely when optional dependencies are absent.

### Security objectives
- JWT auth with admin/user roles.
- Safe enforcement defaults (firewall dry-run).
- Audit trails via JSONL logs and structured logging.

### Technical objectives
- Single source of truth for preprocessing and feature engineering.
- Shared inference engine across all interfaces.
- Thin interface layers using shared utilities and orchestration.

### Business objectives
- Demonstrate a production-style security ML system suitable for portfolio, interviews, and extension into SIEM/SOAR.

---

## 5. Full Software Features (Every Feature)

### 5.1 Multi-interface AI-driven platform
- **Streamlit Dashboard** (`app.py`, `ui/`)
- **FastAPI REST API** (`api/`)
- **CLI SOC Simulator** (`cli_engine.py`)
- **AI Analyst Chatbot** (`chatbot_engine.py`)

### 5.2 Anomaly detection
- Isolation Forest anomaly flagging on canonical dataset features.
- Generic anomaly fallback path for non-canonical schemas (Isolation Forest + optional OneClassSVM + timing deviation).

### 5.3 Threat scoring and classification
- Random Forest classification into `Safe`, `Suspicious`, `Malicious`, `Critical`.
- Class probabilities and confidence.

### 5.4 Severity prediction (CVSS-like)
- Gradient Boosting regression predicts `cvss_predicted` in the 0–10 range for ranking and policy thresholds.

### 5.5 Temporal analysis (optional)
- LSTM trained on sliding windows for sequence-based classification.

### 5.6 Explainability layer
- Deterministic explanations (`utils.prediction.explain_decision`).
- Optional OpenAI-compatible structured explanations (`utils.llm_explainer.py`).

### 5.7 Behavioral profiles (adaptive UEBA)
- EWMA per-subject stats; deviation boosts when maliciousness appears unexpectedly.

### 5.8 Threat intelligence enrichment
- NVD CVE 2.0 fetch + disk cache; dataset vs intel CVSS delta.

### 5.9 Firewall logic (response)
- Windows and Linux enforcement backends with dry-run safety.
- Admin endpoints and UI controls for manual actions.

### 5.10 Alerts, logs, and reports
- JSONL alert bus shared by UI/API.
- Firewall action logs.
- CSV/PDF exports for reporting (UI hooks + helper functions).

---

## 6. Full End-to-End Working Process

### 6.1 Inputs
- **Dataset-backed records** from `data/Dataset-Attacks-Firewall.csv` (primary).
- **API JSON records** posted to `/api/v1/predict`.
- **CLI custom or sampled inputs**.
- **Optional live packet observations** mapped to dataset-backed proxy rows.

### 6.2 Processing steps (canonical path)
1. Load raw records (UI/API/CLI).
2. Feature engineering and schema enforcement (`utils.preprocessing.build_feature_frame`).
3. Preprocessor transform (scaler + one-hot).
4. Random Forest risk classification + probabilities.
5. Isolation Forest outlier flag + score.
6. Gradient Boosting severity prediction (0–10).
7. Optional LSTM sequence label (UI simulation).
8. Explanation generation (deterministic; optional LLM enhancement).
9. Behavioral deviation fusion (API/orchestration).
10. Persist alert to JSONL.
11. Optional firewall block (requires live meta + config).

### 6.3 Internal flow between files/modules (high-level)
- UI (`app.py`, `ui/pages.py`) and API (`api/main.py`) both call `utils.prediction.AuraPredictor`.
- Shared response loop exists in `utils/orchestration.score_and_respond`.
- Alerts are persisted and shared via `utils/alert_bus`.
- Enforcement is implemented in `utils/firewall` and is invoked by API/UI/orchestration depending on configuration.

---

## 7. Deep Anomaly Detection Engine (Very Important)

### What anomaly detection means in AURA
It is the detection of **out-of-distribution** events relative to the learned feature manifold, independent of the supervised class label.

### Why used here
It provides a route to surface unknown or novel patterns that a classifier may not have seen in training.

### How implemented (canonical schema)
- Trained Isolation Forest in `train_model.py`, persisted in `models/risk_model.pkl`.
- Inference in `utils.prediction.AuraPredictor.predict_tabular` using:
  - `predict()` → `anomaly_score_flag` (`-1` outlier, `1` inlier)
  - `score_samples()` → `anomaly_score` (best-effort)

### Behavioral baselines and adaptive anomaly interpretation
`utils.behavioral_profiles` adds a second “anomaly” axis: a deviation score when a historically benign entity produces a malicious event. This supports insider-style or first-time-abuse scenarios.

### Generic anomaly scoring (non-canonical schema)
When inputs don’t match the canonical schema, AURA constructs a generic feature matrix and computes an ensemble anomaly score, then maps that score to risk levels and a CVSS-like severity estimate.

---

## 8. ML Models Used (Based on Repo)

### Random Forest (classification)
Primary risk classifier; outputs class probabilities for confidence and explainability.

### Isolation Forest (anomaly detection)
Outlier detector; flags novel patterns.

### Gradient Boosting (regression)
Predicts severity (`cvss_predicted`) on 0–10.

### LSTM (sequence model, optional)
Temporal classifier trained on sliding windows of ordered events.

### OneClassSVM (fallback only)
Secondary anomaly signal in the generic scoring path when batch size permits.

---

## 9. Why Anomaly Check Feature Is Powerful
It is a direct operational mechanism to validate suspicious inputs, outperforming purely static signature/rule systems by combining supervised risk classification with unsupervised novelty detection and (optionally) temporal modeling.

---

## 10. Full Tech Stack
- **Python**: pandas, numpy
- **ML**: scikit-learn, TensorFlow (optional)
- **UI**: Streamlit, Plotly
- **API**: FastAPI, Uvicorn, pydantic, python-jose, passlib
- **Packet capture**: Scapy (optional)
- **Deployment**: Docker, Docker Compose
- **Storage**: CSV dataset, JSON/JSONL logs, joblib + Keras artifacts

---

## 11. Full Folder Structure Analysis

### `api/`
FastAPI application, JWT auth, API schemas.

### `config/`
Environment-backed configuration and defaults.

### `docs/`
Architecture documentation.

### `mobile/`
Mobile/client integration guide for the API.

### `ui/`
Streamlit layout, pages, styles.

### `utils/`
Core engine: preprocessing, prediction, orchestration, firewall, alert bus, behavioral profiles, live monitor bridge, intel, analytics, explainers.

### Runtime directories (expected)
Even if not committed in this snapshot, AURA expects:

- `data/` for `Dataset-Attacks-Firewall.csv`
- `models/` for persisted artifacts
- `logs/` for JSONL logs, profile store, and intel cache

---

## 12. FULL FILE-BY-FILE ANALYSIS (CRITICAL)

| File Name | Purpose | What Code Likely Does | Why Needed | Connected Modules |
|---|---|---|---|---|
| `README.md` | Project overview | Run instructions and system description | Operator entrypoint | All |
| `requirements.txt` | Dependencies | Declares runtime deps | Reproducible environment | All |
| `Dockerfile` | Container build | Builds Python image and installs deps | Deployability | API/UI |
| `docker-compose.yml` | Orchestration | Runs API and UI services | One-command deploy | Docker |
| `.gitignore` | VCS hygiene | Ignores env/log/model artifacts | Prevent secrets/artifacts | Repo |
| `app.py` | Streamlit entry | Loads dataset/models and routes pages | Main UI | `ui/*`, `utils/*` |
| `train_model.py` | Training | Trains RF/IF/GBDT + optional LSTM | Produces artifacts | `utils.preprocessing` |
| `cli_engine.py` | CLI | Step-by-step SOC simulation | Debug/education | `utils.prediction` |
| `chatbot_engine.py` | Chatbot | Intent + RAG + ML-backed answers | SOC copilot | `utils/*` |
| `docs/ARCHITECTURE.md` | Arch docs | Flow diagram and module roles | Design guide | All |
| `mobile/README.md` | Client docs | API endpoint usage | Integration | `api/main.py` |
| `api/__init__.py` | Package | Marks API package | Imports | API |
| `api/main.py` | API app | Auth, predict, status, alerts, logs, intel, firewall | Integration surface | `utils/*` |
| `api/auth.py` | JWT auth | Token issuance and role checks | Security | `api/main.py` |
| `api/schemas.py` | API models | Predict/status/token DTOs | Contract | `api/main.py` |
| `config/__init__.py` | Package | Exports settings | Imports | `api/*`, `utils/*` |
| `config/settings.py` | Settings | Env config for secrets/features | Central config | All |
| `ui/__init__.py` | Package | Marks UI package | Imports | `app.py` |
| `ui/layout.py` | UI shell | Sidebar/header/pipeline strip/state init | UX | `ui/pages.py` |
| `ui/pages.py` | UI pages | Dashboard, anomaly check, monitoring, firewall controls, logs, chat UI | Core UI | `utils/*` |
| `ui/styles.py` | CSS | Light/dark theme and chrome hiding | UX | UI |
| `utils/__init__.py` | Package | Marks utils package | Imports | All |
| `utils/helpers.py` | Helpers | Paths, JSON logging, exports | Plumbing | All |
| `utils/preprocessing.py` | Features | Schema enforcement + engineered features + LSTM split | ML correctness | Train/predict |
| `utils/prediction.py` | Predictor | Load artifacts + inference + explanations + fallback | Core engine | UI/API/CLI/chat |
| `utils/orchestration.py` | Orchestrator | Score + behavior + alert + firewall | Reuse | UI/CLI |
| `utils/alert_bus.py` | Alerts | JSONL append/tail | Shared comms | UI/API |
| `utils/firewall.py` | Firewall | netsh/iptables/nft + logging + dry-run | Response | UI/API |
| `utils/behavioral_profiles.py` | UEBA | EWMA baseline + deviation fusion | Context | UI/API |
| `utils/packet_monitor.py` | Capture | Scapy sniff + record buffer | Optional live | UI/API status |
| `utils/live_bridge.py` | Mapping | Live context → dataset proxy row | Feature lineage | Packet monitor |
| `utils/threat_intel.py` | Intel | NVD fetch + cache + merge | Enrichment | UI/API/chat |
| `utils/dataset_metrics.py` | Analytics | Summary/timeline/watchlists/log rows | Dashboard | UI |
| `utils/model_ui.py` | Model metrics | Feature importance + metrics blobs | Observability | UI |
| `utils/llm_explainer.py` | LLM layer | Optional structured explanations with fallback | Explainability | Predictor |

---

## 13. Code Architecture Analysis
### 13.1 Architectural pattern
AURA is built as a **shared-core / multi-adapter** system:

- **Shared core** (`utils/`): all detection, preprocessing, inference, response primitives.
- **Adapters**:
  - Streamlit UI (`app.py`, `ui/`) for interactive SOC workflows.
  - FastAPI (`api/`) for programmatic integration.
  - CLI (`cli_engine.py`) for step-by-step pipeline execution.
  - Chatbot (`chatbot_engine.py`) for natural-language interactions.
- **Configuration** (`config/`): environment-driven settings for security-sensitive and deployment-specific behavior.

This is a deliberate design choice: **one engine, multiple entrypoints**, which reduces divergence and “it works in the UI but not in the API” class of failures.

### 13.2 Module responsibilities (reverse-engineered boundaries)
- **`utils/preprocessing.py`**: schema contract + feature engineering + preprocessor construction + chronological split + sequence building.
- **`utils/prediction.py`**: artifact loading + transformation + inference + explanation + fallback inference for non-canonical schemas.
- **`utils/orchestration.py`**: “score → behavioral update → alert → optional firewall” loop for reuse by interfaces.
- **`utils/behavioral_profiles.py`**: per-subject baseline + deviation scoring + fused risk view.
- **`utils/alert_bus.py`**: minimal event bus via JSONL append/tail (cross-process).
- **`utils/firewall.py`**: enforcement abstraction with platform-specific backends + action logging.
- **`utils/threat_intel.py`**: optional enrichment and caching.
- **`utils/packet_monitor.py` + `utils/live_bridge.py`**: optional live capture + dataset-aligned record generation.
- **`utils/dataset_metrics.py` + `utils/model_ui.py`**: observability and analytics for UI.
- **`utils/llm_explainer.py`**: optional explanation enhancement with deterministic fallback.

### 13.3 Key runtime object graph
Across interfaces, the dominant runtime objects are:

- **`AuraPredictor`**: loads model bundles once and scores records.
- **`FirewallManager`**: singleton used by UI/API/orchestration for block/unblock and state reporting.
- **`BehavioralProfileStore`**: singleton used to record events and compute deviation.
- **Packet monitor service** (optional): singleton that manages background sniff thread and ring buffers.

### 13.4 Data lineage principle (critical design decision)
AURA enforces a strong lineage rule: **feature vectors must be derived from rows compatible with the canonical dataset schema**.

- When live packets are captured, AURA does not invent missing engineered columns.
- Instead, it selects a **deterministic dataset proxy row** and attaches live-only metadata in `_live_meta`.

This preserves reproducibility and model correctness at the expense of real-telemetry richness. The design is suitable for demonstration and controlled environments and provides a stable base for evolving toward real feature ingestion later.

---

## 14. Security Architecture
### 14.1 Identity and access control (API layer)
The FastAPI service uses:

- OAuth2 password flow at `POST /api/v1/auth/token`
- JWT issuance and validation (`python-jose`)
- Role claims: `role = admin | user`

Role enforcement:

- `get_current_user` validates JWT and returns `{"username", "role"}`
- `require_admin` blocks access to admin-only endpoints (logs tail, firewall actions)

Credentials are loaded from environment variables (`AURA_ADMIN_*`, `AURA_USER_*`). Password hashes are computed at runtime via `passlib`.

### 14.2 Security controls for enforcement (firewall)
Firewall actions are designed to be safe by default:

- **Dry-run enabled by default** (`AURA_FIREWALL_DRY_RUN=true`).
- Auto-blocking is disabled by default (`AURA_FIREWALL_AUTO_BLOCK=false`).
- Windows and Linux backends are separated behind an interface; commands are logged.

### 14.3 Auditability and forensic logging
AURA’s audit trail is built from:

- Structured JSON logs (UI/API/training logs).
- Append-only JSONL:
  - `logs/aura_alerts.jsonl` (alerts)
  - `logs/firewall_actions.jsonl` (enforcement actions)
- Behavioral profile persistence (`logs/behavioral_profiles.json`)
- Intel cache persistence (`logs/intel_cache/*.json`)

These files are intentionally human-readable and ingestible by log shipping tools.

### 14.4 LLM and threat-intel security posture
Optional external calls:

- **LLM explanations**: OpenAI-compatible API; controlled by `AURA_LLM_ENABLED` and API key env vars; deterministic fallback prevents failure.
- **NVD intel**: httpx calls to NVD CVE 2.0; optional API key; 24h cache reduces exposure and rate-limit issues.

### 14.5 Known security gaps (as implemented)
- Default credential values exist; production must override them.
- There is no built-in rate limiting, brute-force protection, or account lockout for the auth endpoint.
- JSONL storage has no tamper-proofing; enterprise deployments would add integrity controls and external log retention.

---

## 15. Dataset Analysis
### 15.1 Canonical dataset contract (strict)
The preprocessing layer requires a specific schema (a “contract”). Missing required columns raise a hard error, preventing silent model misuse. Required columns include:

- CVE/event identifier: `Data`
- Dates: `pub_date`, `mod_date` (parsed day-first; converted to epoch seconds)
- Severity: `cvss` (numeric; required)
- Source-like field: `Firewall Traffics` (parsed into octet features)
- CWE metadata: `cwe_code`, `cwe_name`
- Text: `summary`
- Access/impact categorical vectors:
  - `access_authentication`, `access_complexity`, `access_vector`
  - `impact_availability`, `impact_confidentiality`, `impact_integrity`

### 15.2 Derived engineered features (explicit)
From the canonical columns AURA derives:

- `summary_len`: length of summary (clipped)
- `fw_o1..fw_o4`: numeric components parsed from `Firewall Traffics`
- `pub_ts`, `mod_ts`: epoch seconds from dates
- `risk_class` (training label): derived from CVSS banding:
  - Safe < 4.0
  - Suspicious 4.0–<6.0
  - Malicious 6.0–<9.0
  - Critical ≥ 9.0

### 15.3 Dataset-driven analytics (UI-facing)
`utils/dataset_metrics.py` derives:

- dataset summary (row counts, unique CVEs/sources, CVSS stats, risk distribution)
- time-series event volumes (daily/hourly/weekly)
- watchlists (sources with concentration of high CVSS)
- CWE distribution and risk timelines
- log-like rows for UI when no simulated history exists

### 15.4 What the dataset represents (interpretation)
Although named “firewall,” the dataset is closer to a **CVE/vulnerability + derived “firewall source” field** dataset than raw packet telemetry. AURA uses it as a stable substrate for:

- supervised risk band classification
- severity regression
- unsupervised outlier detection
- temporal sequencing (ordered by publication time)

---

## 16. Model Training Pipeline
### 16.1 Artifact outputs
The training pipeline produces:

- `models/risk_model.pkl`
  - RandomForest classifier
  - IsolationForest
  - fitted preprocessor (ColumnTransformer)
  - `lstm_seq_len`
  - metrics blob
- `models/cvss_model.pkl`
  - GradientBoosting regressor
  - fitted preprocessor
  - RMSE metric
- `models/lstm_model.h5` (optional; only if TensorFlow available)
- `logs/last_training_metrics.json`

### 16.2 Step-by-step training flow (actual code path)
1. Ensure directories exist (`utils.helpers.ensure_directories`).
2. Load raw dataset (`load_raw_dataset`) and build feature frame (`build_feature_frame`).
3. Chronologically split by `pub_ts` into train/test.
4. Extract labels:
   - `y_risk` from `risk_class` (derived from CVSS)
   - `y_cvss` from `cvss`
5. Fit and apply preprocessor on train only; transform test.
6. Train models:
   - IsolationForest (contamination 0.06)
   - RandomForestClassifier (balanced_subsample)
   - GradientBoostingRegressor (depth 5, 200 estimators)
7. Prepare LSTM tensors:
   - build sliding windows of length 12 across ordered data
   - split sequences by the same chronological cut index
   - subsample sequences to caps for efficiency
8. Train LSTM (if TensorFlow present), save model, store metrics.
9. Persist artifacts to `models/` and metrics file to `logs/`.

### 16.3 Why this pipeline design matters
- Chronological split reduces “future leakage.”
- Persisting the preprocessor with artifacts prevents transformation drift.
- Storing metrics and exposing them in UI supports governance and observability.

---

## 17. Chatbot Module Analysis
### 17.1 Operational purpose
The chatbot provides a conversational interface for:

- CVE explanations and severity context
- IP threat assessment (based on dataset matches or fallback reasoning)
- alerts/incident summaries (based on sample scoring)
- behavioral profiling status
- firewall decision explanation
- project usage and architecture guidance (offline RAG)

### 17.2 Retrieval-Augmented Generation (offline)
The offline RAG index is intentionally lightweight:

- Sources: README, requirements, Docker files, entrypoints, `docs/*.md`
- Vectorization: TF-IDF with uni/bi-grams, stop words removed
- Output: multi-source snippet bullets, explicitly showing the source file path

This allows the chatbot to answer “how do I run this?” without needing external LLM calls.

### 17.3 ML-backed answers (when artifacts + dataset exist)
When available, the chatbot:

- Locates dataset columns dynamically (schema detector) to find CVE/IP/CVSS fields.
- Subsets matching rows and calls `AuraPredictor.predict_records`.
- Summarizes risk class distribution, average predicted CVSS, anomaly counts.
- Optionally enriches CVE context via `correlate_dataset_cve`.

### 17.4 Fallback behavior and safety
If models or dataset are unavailable:

- It returns deterministic fallback responses and conservative “AI reasoning” templates.
- It does not claim ground truth about a CVE not present in data; it frames it as an estimate.

---

## 18. Dashboard Module Analysis
### 18.1 Dashboard as a SOC control center
The Streamlit UI is designed to look and behave like an operational console:

- A persistent navigation sidebar across functional pages.
- A system “mode” selector (Protection / Learning / Monitoring) that influences which architecture stage is highlighted (presentation + operator context).
- A threat-level filter and sensitivity slider controlling what is surfaced.
- Quick actions (scan ingestion, refresh dataset).

### 18.2 Threat history model
The UI maintains an in-session “threat history” with fields such as:

- timestamp, cve_id, risk_class, cvss_predicted, anomaly/inlier label
- risk probabilities, deterministic/LLM explanation output
- behavioral fusion output
- (in simulation) LSTM label and confidence

This history powers:

- the live logs view
- charts/histograms
- exports

### 18.3 Simulation workflow (important)
The simulation is not random noise; it is dataset-aligned and sequence-aware:

- Features are built and ordered by publication timestamp.
- Random end indices are selected ensuring a full LSTM window exists.
- The predictor scores the corresponding raw dataset rows.
- For each scored event, the UI optionally runs LSTM classification on the associated window.

This creates a realistic “stream” for UI demonstrations without requiring live traffic.

### 18.4 Pages implemented (functional inventory)
`ui/pages.py` implements renderers for:

- Dashboard
- Anomaly Check (sample/custom/bulk upload)
- Live Monitoring (packet capture if available + dataset timeseries)
- AI Insights (risk mix, confidence, feature importance)
- AI Analyst Chat (embedded chatbot UI)
- Threat Intelligence (NVD enrichment)
- System Architecture (metrics blob display)
- Analytics (watchlists and timelines)
- Firewall Controls (policy toggles + manual block/unblock)
- Logs (filter/search/export)
- Mobile View (condensed status)

### 18.5 UI security posture
The UI does not implement its own auth; it is assumed to run in a trusted operator context (local or behind an access-controlled environment). The API is the primary security boundary.

---

## 19. CLI Module Analysis
### 19.1 CLI as an explainable pipeline runner
The CLI provides a structured walkthrough of the system with explicit “Step 1…Step N” execution. It exposes:

- intermediate data shapes and null counts
- IsolationForest score distributions (mean/std)
- RandomForest probability breakdown per class
- CVSS regression distribution
- optional LSTM output shape and score

### 19.2 Why this matters
Security ML systems fail in production most often due to:

- schema mismatch
- preprocessing drift
- artifact load errors
- misunderstood model outputs

The CLI makes these failure modes visible and provides a deterministic way to validate each stage independently.

---

## 20. API Module Analysis
### 20.1 API design principles
The API is designed to be:

- **Mobile-ready**: JSON responses, CORS configurable.
- **Thin**: wraps shared engine objects rather than re-implementing logic.
- **Secure**: JWT auth with admin-only endpoints.

### 20.2 Endpoint catalog (implemented behavior)
- `POST /api/v1/auth/token`
  - Validates credentials, issues JWT with role.
- `GET /api/v1/status`
  - Reports:
    - model readiness
    - firewall state (dry run, blocked session IPs, recent actions)
    - packet monitor state (running, captured count, last error)
    - behavioral profile snapshot
- `POST /api/v1/predict`
  - Batch scoring with optional explanations and optional LLM enhancement.
  - For each result:
    - behavioral deviation computed and stored
    - alert appended to JSONL
    - optional firewall auto-block on malicious/critical when live meta exists
- `GET /api/v1/alerts`
  - Tail N recent alerts from shared JSONL ring.
- `GET /api/v1/logs/tail` (admin)
  - Returns last N lines from a log file in `logs/`.
- `GET /api/v1/intel/cve/{cve_id}`
  - Attempts to find CVE in local dataset using dynamic column detection; if found, correlates with NVD and returns merged output.
- `POST /api/v1/firewall/block` (admin)
  - Applies block rule (dry-run aware).
- `POST /api/v1/firewall/unblock` (admin)
  - Removes rule (dry-run aware).

### 20.3 Data validation considerations
`PredictRequest.records` is schema-free by design (list of dicts). This allows flexibility but shifts responsibility to preprocessing:

- Canonical path strictly enforces required dataset columns.
- Non-canonical path uses generic feature inference and anomaly scoring.

Enterprise hardening would add explicit schemas per telemetry type and stricter input validation, but AURA intentionally demonstrates both strict and flexible modes.

---

## 21. Pros of Software
End-to-end system, modular reuse, safe enforcement defaults, deterministic fallbacks, and multiple operational interfaces.

---

## 22. Cons / Limitations
Live telemetry is dataset-proxy based; storage is file-based; production hardening and scaling would require stronger secrets management and persistent stores/queues.

---

## 23. Improvement Suggestions
SIEM/SOAR integrations, streaming ingestion, stronger RBAC, Kubernetes deployment, explainable AI (SHAP), drift detection and retraining automation, richer telemetry features.

---

## 24. Real World Use Cases
Banking, healthcare, cloud, e-commerce, government, campus networks, and SMEs, especially where explainable scoring and safe response automation are required.

---

## 25. Resume / Interview Value
Demonstrates full-stack security ML engineering: training, inference, APIs, UI, access control, deployment, and operational response.

---

## 26. 20 Interview Questions with Answers

1. **What is AURA?** An AI-driven behavioral firewall and threat scoring platform with UI/API/CLI/chat interfaces.
2. **Why Random Forest + Isolation Forest?** Classification + novelty/outlier detection for unknowns.
3. **Why chronological split?** Reduces leakage; approximates real deployment.
4. **What is `cvss_predicted`?** Continuous severity ranking output (0–10).
5. **What is the anomaly flag meaning?** `-1` outlier, `1` inlier (Isolation Forest).
6. **How does AURA enforce firewall decisions safely?** Dry-run default; explicit enable for auto-blocking.
7. **How are alerts stored?** Append-only JSONL ring shared across processes.
8. **What is UEBA in AURA?** EWMA behavioral profiles and deviation boosts.
9. **How does the chatbot work offline?** TF-IDF retrieval over repo docs + deterministic reasoning.
10. **What are the main API endpoints?** Auth token, predict, status, alerts, logs tail, intel, firewall block/unblock.
11. **How is API secured?** JWT with role checks; admin-only endpoints.
12. **How is training reproducible?** Fixed preprocessing pipeline and persisted preprocessor in artifacts.
13. **What happens if TensorFlow is missing?** LSTM training/loading is skipped; system continues.
14. **What happens if Scapy is missing?** Packet capture disabled; system continues.
15. **What is the generic scoring path?** Anomaly ensemble for non-canonical schemas.
16. **What is the single source of truth for features?** `utils/preprocessing.build_feature_frame`.
17. **How do you export reports?** CSV/PDF helper functions; UI provides controls.
18. **How do you integrate threat intel?** NVD CVE 2.0 API with cache.
19. **What are the biggest limitations?** Dataset-proxy live mapping; file-based persistence.
20. **How would you scale it?** Queue/DB, worker processes, Kafka, Kubernetes, model registry.

---

## 27. Final 60-Second Project Explanation
AURA trains an ensemble security ML model on a canonical dataset and exposes a unified scoring engine across a Streamlit SOC dashboard, FastAPI REST API with JWT roles, a CLI simulator, and a natural-language AI Analyst chatbot. For each event it outputs a risk class, anomaly signal, severity score, and explanation, persists alerts for audit, and can optionally trigger OS firewall blocks with safe defaults.

---

## 28. Final Verdict
AURA is a strong end-to-end security ML platform demonstrating production-style architecture and operational workflows. Its core strength is the shared engine design (`utils/`) reused consistently across UI/API/CLI/chat, with safe enforcement defaults and auditable outputs.
