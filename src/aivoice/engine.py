from __future__ import annotations

import asyncio
import time

from .adapters.llm import MLXLLM
from .adapters.stt import ParakeetSTT
from .adapters.tts import KokoroTTS
from .config import Config


class Engine:
    """Owns the model adapters and the on/off lifecycle the control page drives."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = "stopped"  # stopped | starting | running | error
        self.error = ""
        self.started_at: float | None = None
        self.sessions: set = set()
        self.turns = 0
        self.last_latency_ms: float | None = None
        self._lock = asyncio.Lock()
        self._build_adapters()

    def _build_adapters(self) -> None:
        self.stt = ParakeetSTT(self.config.stt_model)
        self.llm = MLXLLM(self.config.llm_model)
        self.tts = KokoroTTS(self.config.tts_model)

    @property
    def running(self) -> bool:
        return self.state == "running"

    async def start(self) -> None:
        async with self._lock:
            if self.state in ("running", "starting"):
                return
            self.state = "starting"
            self.error = ""
            # Model loads are CPU-bound and slow; keep the event loop answering the UI.
            for adapter in (self.stt, self.llm, self.tts):
                await asyncio.to_thread(adapter.load)
            failed = [a.status() for a in (self.stt, self.llm, self.tts) if not a.status().ready]
            self.state = "running"
            self.started_at = time.time()
            if failed:
                self.error = "; ".join(f"{s.name}: {s.detail}" for s in failed)

    async def stop(self) -> None:
        async with self._lock:
            for adapter in (self.stt, self.llm, self.tts):
                adapter.unload()
            self.state = "stopped"
            self.started_at = None

    async def apply_config(self, updates: dict) -> None:
        known = set(self.config.__dataclass_fields__)
        changed_models = False
        for key, value in updates.items():
            if key not in known or key in ("host", "port"):
                continue
            if key.endswith("_model") and getattr(self.config, key) != value:
                changed_models = True
            setattr(self.config, key, value)
        self.config.save()
        if changed_models:
            was_running = self.running
            await self.stop()
            self._build_adapters()
            if was_running:
                await self.start()

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "uptime_s": None if self.started_at is None else round(time.time() - self.started_at, 1),
            "sessions": len(self.sessions),
            "turns": self.turns,
            "last_latency_ms": self.last_latency_ms,
            "adapters": [a.status().as_dict() for a in (self.stt, self.llm, self.tts)],
            "config": self.config.as_dict(),
        }
