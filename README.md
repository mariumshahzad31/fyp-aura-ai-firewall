# AURA: Advanced Unified Resilience Architecture  
## Multi-Interface AI-Driven Behavioral Firewall System  

### Technical Report and Implementation Guide  

---

## Executive Summary  

AURA is a production-grade behavioral anomaly detection and firewall system designed for real-time threat identification in operational environments. The system integrates multiple machine learning models—Random Forest, Isolation Forest, Gradient Boosting, and LSTM—along with natural language processing capabilities to deliver comprehensive threat analysis.

AURA provides multi-interface accessibility through:
- Web Dashboard (Streamlit)
- Command-Line Interface (CLI)
- Intelligent Chatbot
- REST API

This document serves as complete technical documentation suitable for academic, research, and professional use.

---

### Installation
bash
cd aura
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt


### Train Models (One-time)
bash
python train_model.py


### Launch Interfaces
bash
# Web Dashboard
streamlit run app.py

# CLI Simulator
python cli_engine.py

# Chatbot
python chatbot_engine.py

# REST API
uvicorn api.main:app --host 0.0.0.0 --port 8000