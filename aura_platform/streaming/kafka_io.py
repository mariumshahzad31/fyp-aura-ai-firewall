"""Kafka ingestion + threat fan-out using optional kafka-python."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, Optional

from aura_platform.config import platform_settings


class AuraKafkaProducer:
    def __init__(self, bootstrap_servers: Optional[str] = None) -> None:
        cfg = platform_settings()
        self._servers = bootstrap_servers or cfg["kafka_bootstrap"]
        self._topic = cfg["kafka_topic_in"]
        self._producer = None

    def _ensure(self):
        if self._producer is not None:
            return
        from kafka import KafkaProducer  # type: ignore

        self._producer = KafkaProducer(
            bootstrap_servers=self._servers.split(","),
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            linger_ms=1,
            acks=1,
        )

    def send_event(self, payload: Dict[str, Any]) -> None:
        if not platform_settings()["kafka_enabled"]:
            return
        self._ensure()
        enriched = dict(payload)
        enriched.setdefault("ts_wall", time.time())
        fut = self._producer.send(self._topic, value=enriched)
        fut.add_errback(lambda exc: None)

    def flush(self) -> None:
        if self._producer:
            self._producer.flush(timeout=10)


class AuraKafkaConsumer(threading.Thread):
    def __init__(self, on_message: Callable[[Dict[str, Any]], None], topic: Optional[str] = None) -> None:
        super().__init__(daemon=True)
        self._on_message = on_message
        cfg = platform_settings()
        self._bootstrap = cfg["kafka_bootstrap"]
        self._topic = topic or cfg["kafka_topic_out"]
        self._group = cfg["kafka_consumer_group"]
        self._stop = threading.Event()

    def run(self) -> None:
        cfg = platform_settings()
        if not cfg["kafka_enabled"]:
            return
        from kafka import KafkaConsumer  # type: ignore

        consumer = KafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap.split(","),
            group_id=self._group,
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )
        while not self._stop.is_set():
            batch = consumer.poll(timeout_ms=1000)
            for _tp, msgs in batch.items():
                for m in msgs:
                    try:
                        self._on_message(m.value if isinstance(m.value, dict) else {})
                    except Exception:
                        continue
            time.sleep(0.01)


def kafka_emit_threat(summary: Dict[str, Any]) -> None:
    if not platform_settings()["kafka_enabled"]:
        return
    from kafka import KafkaProducer  # type: ignore

    cfg = platform_settings()
    p = KafkaProducer(
        bootstrap_servers=cfg["kafka_bootstrap"].split(","),
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        linger_ms=0,
        acks=1,
    )
    try:
        p.send(cfg["kafka_topic_out"], value=summary).get(timeout=5)
    finally:
        p.flush()
        p.close()
