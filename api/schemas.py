"""Pydantic models for AURA REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    include_explanation: bool = True
    use_llm: bool = False


class SystemStatus(BaseModel):
    model_ready: bool
    firewall: Dict[str, Any]
    packet_monitor: Dict[str, Any]
    behavioral_profiles: Dict[str, Any]


class AlertItem(BaseModel):
    timestamp: str
    cve_id: Optional[str] = None
    risk_class: str
    cvss_predicted: Optional[float] = None
    source_prefix: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
