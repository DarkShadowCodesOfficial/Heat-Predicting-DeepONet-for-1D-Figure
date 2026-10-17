"""Thread-safe training progress, polled by the demo UI."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

log = logging.getLogger("heat_operator")


class Progress:
    """Tracks weighted progress units, the current stage and a rolling log."""

    def __init__(self, total_units: float = 1.0, max_log: int = 300):
        self.total = max(float(total_units), 1.0)
        self.done = 0.0
        self.stage = "starting"
        self.message = ""
        self.started = time.time()
        self._log: deque[str] = deque(maxlen=max_log)
        self._lock = threading.Lock()

    def set_total(self, total_units: float):
        with self._lock:
            self.total = max(float(total_units), 1.0)

    def advance(self, units: float = 1.0):
        with self._lock:
            self.done = min(self.done + units, self.total)

    def set_stage(self, stage: str, message: str = ""):
        with self._lock:
            self.stage = stage
            self.message = message
        self.add_log(f"[{stage}] {message}" if message else f"[{stage}]")

    def add_log(self, line: str):
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._log.append(f"{stamp}  {line}")
        log.info(line)

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = time.time() - self.started
            frac = self.done / self.total
            eta = elapsed * (1 - frac) / frac if frac > 0.02 else None
            return {
                "fraction": frac,
                "stage": self.stage,
                "message": self.message,
                "elapsed_s": elapsed,
                "eta_s": eta,
                "log": list(self._log)[-60:],
            }
