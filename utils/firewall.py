"""
OS-level firewall integration: Windows Defender Firewall (netsh) and Linux iptables/nftables.
Dry-run mode logs actions without executing. Malicious predictions can trigger blocks on observed IPs.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger("aura.firewall")

RULE_PREFIX = "AURA-BLOCK-"


@dataclass
class FirewallActionResult:
    ok: bool
    message: str
    command: Optional[str] = None


@dataclass
class FirewallState:
    dry_run: bool
    platform_name: str
    blocked_ips: List[str] = field(default_factory=list)
    last_actions: List[Dict[str, str]] = field(default_factory=list)


class FirewallBackend(ABC):
    @abstractmethod
    def block_ip(self, ip: str, reason: str = "") -> FirewallActionResult:
        ...

    @abstractmethod
    def unblock_ip(self, ip: str) -> FirewallActionResult:
        ...

    def list_blocked(self) -> List[str]:
        return []


def _validate_ip(ip: str) -> bool:
    return bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", ip.strip()))


class WindowsFirewallBackend(FirewallBackend):
    """Windows Defender Firewall via netsh advfirewall."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run

    def _rule_name(self, ip: str) -> str:
        safe = ip.replace(".", "-")
        return f"{RULE_PREFIX}{safe}"

    def block_ip(self, ip: str, reason: str = "") -> FirewallActionResult:
        if not _validate_ip(ip):
            return FirewallActionResult(False, f"Invalid IPv4 for block: {ip}")
        name = self._rule_name(ip)
        cmd = [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name={name}",
            "dir=in",
            "action=block",
            f"remoteip={ip}",
            "protocol=any",
            "enable=yes",
        ]
        if self.dry_run:
            logger.info("[dry-run] Would block %s (%s)", ip, reason)
            return FirewallActionResult(True, "dry-run: no rule added", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return FirewallActionResult(True, f"Blocked {ip} inbound", " ".join(cmd))
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or str(exc))[:500]
            return FirewallActionResult(False, f"netsh failed: {msg}", " ".join(cmd))

    def unblock_ip(self, ip: str) -> FirewallActionResult:
        name = self._rule_name(ip)
        cmd = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={name}"]
        if self.dry_run:
            return FirewallActionResult(True, "dry-run: no rule deleted", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return FirewallActionResult(True, f"Removed rule for {ip}", " ".join(cmd))
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or str(exc))[:500]
            return FirewallActionResult(False, f"netsh delete failed: {msg}", " ".join(cmd))


class LinuxIptablesBackend(FirewallBackend):
    """Linux iptables DROP for inbound from IP (requires CAP_NET_ADMIN / root)."""

    def __init__(self, dry_run: bool, chain: str = "INPUT") -> None:
        self.dry_run = dry_run
        self.chain = chain

    def block_ip(self, ip: str, reason: str = "") -> FirewallActionResult:
        if not _validate_ip(ip):
            return FirewallActionResult(False, f"Invalid IPv4 for block: {ip}")
        cmd = ["iptables", "-I", self.chain, "1", "-s", ip, "-j", "DROP"]
        if self.dry_run:
            logger.info("[dry-run] Would iptables DROP %s (%s)", ip, reason)
            return FirewallActionResult(True, "dry-run: iptables not run", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return FirewallActionResult(True, f"iptables DROP {ip}", " ".join(cmd))
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or str(exc))[:500]
            return FirewallActionResult(False, f"iptables failed: {msg}", " ".join(cmd))

    def unblock_ip(self, ip: str) -> FirewallActionResult:
        cmd = ["iptables", "-D", self.chain, "-s", ip, "-j", "DROP"]
        if self.dry_run:
            return FirewallActionResult(True, "dry-run: iptables not run", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return FirewallActionResult(True, f"iptables removed DROP for {ip}", " ".join(cmd))
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or str(exc))[:500]
            return FirewallActionResult(False, f"iptables delete failed: {msg}", " ".join(cmd))


class LinuxNftablesBackend(FirewallBackend):
    """Optional nftables drop (inet filter)."""

    def __init__(self, dry_run: bool, table: str = "inet", family_filter: str = "filter") -> None:
        self.dry_run = dry_run
        self.table = table
        self.family_filter = family_filter

    def block_ip(self, ip: str, reason: str = "") -> FirewallActionResult:
        if not _validate_ip(ip):
            return FirewallActionResult(False, f"Invalid IPv4 for block: {ip}")
        cmd = ["nft", "add", "rule", self.table, self.family_filter, "input", "ip", "saddr", ip, "drop"]
        if self.dry_run:
            logger.info("[dry-run] Would nft drop %s (%s)", ip, reason)
            return FirewallActionResult(True, "dry-run: nft not run", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return FirewallActionResult(True, f"nft drop {ip}", " ".join(cmd))
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or str(exc))[:500]
            return FirewallActionResult(False, f"nft failed: {msg}", " ".join(cmd))

    def unblock_ip(self, ip: str) -> FirewallActionResult:
        return FirewallActionResult(
            False,
            "nftables removal requires handle from `nft -a list`; use iptables backend or delete manually.",
        )


class FirewallManager:
    """
    Routes block/unblock to the correct backend; tracks blocked IPs in-memory and optional JSON log.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blocked: Dict[str, datetime] = {}
        s = get_settings()
        self._dry = s.firewall_dry_run
        sysname = platform.system().lower()
        if sysname == "windows":
            self._backend: FirewallBackend = WindowsFirewallBackend(self._dry)
        elif (Path("/sbin/nft").exists() or Path("/usr/sbin/nft").exists()) and (
            os.getenv("AURA_USE_NFTABLES", "").lower() in ("1", "true", "yes")
        ):
            self._backend = LinuxNftablesBackend(self._dry)
        else:
            self._backend = LinuxIptablesBackend(self._dry)
        self._log_path = Path(__file__).resolve().parents[1] / "logs" / "firewall_actions.jsonl"

    def state(self) -> FirewallState:
        with self._lock:
            ips = list(self._blocked.keys())
        return FirewallState(
            dry_run=self._dry,
            platform_name=platform.system(),
            blocked_ips=ips,
            last_actions=self._read_last_actions(20),
        )

    def _read_last_actions(self, n: int) -> List[Dict[str, str]]:
        if not self._log_path.exists():
            return []
        lines = self._log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]
        out: List[Dict[str, str]] = []
        for ln in lines:
            try:
                import json

                out.append(json.loads(ln))
            except Exception:
                continue
        return out

    def _append_log(self, payload: Dict[str, str]) -> None:
        import json

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def block_observed_ip(
        self,
        ip: str,
        risk_class: str,
        reason: str = "ml_policy",
    ) -> FirewallActionResult:
        if not _validate_ip(ip):
            return FirewallActionResult(False, "skip: invalid IP")
        with self._lock:
            if ip in self._blocked:
                return FirewallActionResult(True, "already blocked (session)")
        res = self._backend.block_ip(ip, reason=reason)
        if res.ok:
            with self._lock:
                self._blocked[ip] = datetime.now(timezone.utc)
            self._append_log(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "action": "block",
                    "ip": ip,
                    "risk_class": risk_class,
                    "reason": reason,
                    "message": res.message,
                }
            )
        return res

    def unblock(self, ip: str) -> FirewallActionResult:
        res = self._backend.unblock_ip(ip)
        if res.ok:
            with self._lock:
                self._blocked.pop(ip, None)
            self._append_log(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "action": "unblock",
                    "ip": ip,
                    "message": res.message,
                }
            )
        return res


# Lazy singleton
_manager: Optional[FirewallManager] = None
_mgr_lock = threading.Lock()


def get_firewall_manager() -> FirewallManager:
    global _manager
    with _mgr_lock:
        if _manager is None:
            _manager = FirewallManager()
        return _manager
