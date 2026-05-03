"""Graph anomaly detection across entities without replacing ML stack."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from platform.config import platform_settings


def _norm_ip(ip: Optional[str]) -> str:
    s = str(ip or "").strip()
    return s or "unknown_ip"


class ThreatCorrelationGraph:
    def __init__(self, window: int = 5000) -> None:
        self._edges: Dict[Tuple[str, str], float] = {}
        self._deg: defaultdict[str, float] = defaultdict(float)
        self._window_events: List[str] = []
        self._window_cap = window

    def clear(self) -> None:
        self._edges.clear()
        self._deg.clear()
        self._window_events.clear()

    def ingest(self, evt: Dict[str, Any]) -> None:
        if not platform_settings()["graph_detection"]:
            return
        tenant = str(evt.get("tenant_id") or "_default")
        ip = _norm_ip(evt.get("source_ip") or evt.get("observed_source_ip"))
        sess = str(evt.get("session_id") or "").strip()
        uid = str(evt.get("user_id") or "").strip()
        behave = str(evt.get("risk_class") or "Safe")

        nid_ip = f"ip:{tenant}:{ip}"
        nid_user = f"user:{tenant}:{uid or sess or 'anonymous'}"
        nid_behave = f"behavior:{tenant}:{behave}"

        w = 1.0
        rc = behave
        if rc in ("Malicious", "Critical"):
            w = 3.0
        elif rc == "Suspicious":
            w = 1.5

        def add_edge(a: str, b: str, wt: float) -> None:
            key = tuple(sorted([a, b]))
            prev = float(self._edges.get(key, 0.0))
            self._edges[key] = prev + wt
            self._deg[a] += wt
            self._deg[b] += wt

        add_edge(nid_ip, nid_user, w)
        add_edge(nid_ip, nid_behave, w * 0.6)

        key_ip = nid_ip[:48]
        self._window_events.append(key_ip)
        if len(self._window_events) > self._window_cap:
            self._window_events.pop(0)

    def pagerank(self, damping: float = 0.85, iters: int = 12) -> Dict[str, float]:
        nodes = sorted(self._deg.keys())
        if not nodes:
            return {}
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        out_adj: defaultdict[int, List[Tuple[int, float]]] = defaultdict(list)

        for (a, b), wt in self._edges.items():
            ia, ib = idx[a], idx[b]
            out_adj[ia].append((ib, wt))
            out_adj[ib].append((ia, wt))

        out_sum = []
        for i in range(n):
            s = sum(w for _, w in out_adj[i]) or 1.0
            out_sum.append(s)

        r = [1.0 / n for _ in range(n)]
        teleport = (1.0 - damping) / n
        for _ in range(iters):
            nxt = [teleport] * n
            for i in range(n):
                if not out_adj[i]:
                    contrib = damping * r[i] / n
                    for j in range(n):
                        nxt[j] += contrib
                    continue
                share = damping * r[i] / out_sum[i]
                for j, wt in out_adj[i]:
                    nxt[j] += share * wt
            ssum = sum(nxt)
            r = [x / ssum for x in nxt]
        return {nodes[i]: r[i] for i in range(n)}

    def multi_stage_signals(self, top_k: int = 5) -> Dict[str, Any]:
        rank = self.pagerank()
        items = sorted(rank.items(), key=lambda kv: kv[1], reverse=True)
        hotspots = [{"node": k, "score": v} for k, v in items[:top_k]]
        bursts: Dict[str, int] = {}
        for k in self._window_events[-200:]:
            bursts[k] = bursts.get(k, 0) + 1

        anomaly_nodes = [{"node": nk, "burst": c} for nk, c in sorted(bursts.items(), key=lambda x: x[1], reverse=True)[:5]]
        max_burst = float(max([c for _, c in bursts.items()] or [0]))
        churn = math.log1p(max_burst)

        verdict = churn > 4.5 or float(items[0][1]) > 0.25 if items else False
        return {"graph_hotspots": hotspots, "ip_burst_signals": anomaly_nodes, "burst_churn": churn, "graph_threat_hint": verdict}
