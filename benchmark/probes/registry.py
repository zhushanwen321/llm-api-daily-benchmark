"""Probe registry - manages all available probes."""

from __future__ import annotations

import threading
from typing import Any

from benchmark.probes import BaseProbe
from benchmark.probes.safety.safety_probe import SafetyProbe
from benchmark.probes.fingerprint.fingerprint_probe import FingerprintProbe
from benchmark.probes.consistency.consistency_probe import ConsistencyProbe
from benchmark.probes.uncertainty.logprobs_probe import LogprobsProbe


class ProbeRegistry:
    """Registry for managing and executing all probes."""

    _instance: ProbeRegistry | None = None
    _lock: threading.Lock = threading.Lock()
    _probes: dict[str, BaseProbe]

    def __new__(cls) -> ProbeRegistry:
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._probes = {}
                    cls._instance._register_default_probes()
        return cls._instance

    def _register_default_probes(self) -> None:
        """Register all built-in probes."""
        self.register("safety", SafetyProbe())
        self.register("fingerprint", FingerprintProbe())
        self.register("consistency", ConsistencyProbe())
        self.register("uncertainty", LogprobsProbe())

    def register(self, name: str, probe: BaseProbe) -> None:
        """Register a probe."""
        self._probes[name] = probe

    def get(self, name: str) -> BaseProbe | None:
        """Get a probe by name."""
        return self._probes.get(name)

    def list_probes(self) -> list[str]:
        """List all registered probe names."""
        return list(self._probes.keys())

    def get_by_frequency(self, frequency: str) -> dict[str, BaseProbe]:
        """Get probes by frequency level."""
        return {
            name: probe
            for name, probe in self._probes.items()
            if probe.frequency == frequency
        }

    def get_all_probes(self) -> dict[str, BaseProbe]:
        """Get all registered probes."""
        return self._probes.copy()

    def get_probe_summary(self) -> dict[str, Any]:
        """Get summary of all probes."""
        summary = {}
        for name, probe in self._probes.items():
            tasks = probe.load_probes()
            summary[name] = {
                "frequency": probe.frequency,
                "task_count": len(tasks),
                "categories": list(set(t.dataset for t in tasks)),
            }
        return summary
