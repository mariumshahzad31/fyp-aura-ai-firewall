"""
Live packet capture (Scapy) with ring buffer; feeds dataset-aligned records via live_bridge.
Requires admin/root for raw sockets. Disabled by default (AURA_PACKET_CAPTURE).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

import pandas as pd

from utils.live_bridge import build_dataset_index, packet_context_to_record

logger = logging.getLogger("aura.packet_monitor")

try:
    from scapy.all import IP, TCP, UDP  # type: ignore
    from scapy.all import sniff as scapy_sniff  # type: ignore

    SCAPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    SCAPY_AVAILABLE = False
    scapy_sniff = None  # type: ignore
    IP = TCP = UDP = None  # type: ignore


@dataclass
class PacketSnapshot:
    ts: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: str
    size: int
    summary: str


@dataclass
class PacketMonitorState:
    running: bool
    packets_captured: int
    last_error: Optional[str]
    buffer: List[Dict[str, Any]] = field(default_factory=list)


class PacketMonitorService:
    """
    Background sniff thread; on each packet builds a dataset-aligned record for ML scoring.
    """

    def __init__(self, df: pd.DataFrame, on_record: Optional[Callable[[Dict[str, Any], PacketSnapshot], None]] = None):
        self._df = df
        self._index = build_dataset_index(df)
        self._on_record = on_record
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._count = 0
        self._last_err: Optional[str] = None
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._snap_buffer: Deque[PacketSnapshot] = deque(maxlen=200)

    def _handle_packet(self, pkt: Any) -> None:
        if IP is None or not pkt.haslayer(IP):
            return
        ip = pkt[IP]
        src = str(ip.src)
        dst = str(ip.dst)
        sport = dport = 0
        proto = "ip"
        if pkt.haslayer(TCP):
            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)
            proto = "tcp"
        elif pkt.haslayer(UDP):
            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)
            proto = "udp"
        snap = PacketSnapshot(
            ts=datetime.now(timezone.utc).isoformat(),
            src_ip=src,
            dst_ip=dst,
            src_port=sport,
            dst_port=dport,
            proto=proto,
            size=len(pkt),
            summary=str(pkt.summary()),
        )
        self._snap_buffer.append(snap)
        rec = packet_context_to_record(self._df, self._index, src, dst, sport, dport, proto)
        self._buffer.append(rec)
        self._count += 1
        if self._on_record:
            try:
                self._on_record(rec, snap)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_record failed: %s", exc)

    def _loop(self, bpf_filter: str) -> None:
        if not SCAPY_AVAILABLE or scapy_sniff is None:
            self._last_err = "Scapy not installed"
            return
        while not self._stop.is_set():
            try:
                scapy_sniff(
                    filter=bpf_filter,
                    prn=self._handle_packet,
                    store=False,
                    stop_filter=lambda _: self._stop.is_set(),
                    timeout=2,
                )
            except Exception as exc:  # noqa: BLE001
                self._last_err = str(exc)[:500]
                logger.warning("sniff error: %s", self._last_err)
                time.sleep(1.0)

    def start(self, bpf_filter: str = "ip") -> bool:
        if self._thread and self._thread.is_alive():
            return True
        if not SCAPY_AVAILABLE:
            self._last_err = "Scapy not available; pip install scapy"
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(bpf_filter,), daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def state(self) -> PacketMonitorState:
        return PacketMonitorState(
            running=self._thread is not None and self._thread.is_alive(),
            packets_captured=self._count,
            last_error=self._last_err,
            buffer=list(self._buffer)[-80:],
        )

    def recent_snapshots(self, n: int = 50) -> List[PacketSnapshot]:
        return list(self._snap_buffer)[-n:]


_GLOBAL_MONITOR: Optional[PacketMonitorService] = None
_GLOBAL_LOCK = threading.Lock()


def get_or_create_monitor(df: pd.DataFrame) -> PacketMonitorService:
    global _GLOBAL_MONITOR
    with _GLOBAL_LOCK:
        if _GLOBAL_MONITOR is None:
            _GLOBAL_MONITOR = PacketMonitorService(df)
        return _GLOBAL_MONITOR
