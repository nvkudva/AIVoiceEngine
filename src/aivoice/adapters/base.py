from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import numpy as np


@dataclass
class AdapterStatus:
    name: str
    backend: str
    ready: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "backend": self.backend, "ready": self.ready, "detail": self.detail}


class Adapter:
    """Base for lazily-loaded model adapters.

    Loading is deferred so the control page comes up even when a model is missing
    or still downloading. `status()` is what the PWA renders.
    """

    kind = "adapter"

    def __init__(self, model: str) -> None:
        self.model = model
        self._loaded = False
        self._error = ""

    def load(self) -> None:
        if self._loaded or self._error:
            return
        try:
            self._load()
            self._loaded = True
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, never fatal
            self._error = f"{type(exc).__name__}: {exc}"

    def _load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self._loaded = False
        self._error = ""

    def status(self) -> AdapterStatus:
        backend = self.model if self._loaded else "stub"
        detail = self._error or ("" if self._loaded else "not loaded")
        return AdapterStatus(self.kind, backend, self._loaded, detail)


class STTAdapter(Adapter):
    kind = "stt"

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError


class LLMAdapter(Adapter):
    kind = "llm"

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        raise NotImplementedError


class TTSAdapter(Adapter):
    kind = "tts"

    async def stream(self, text: str, voice: str) -> AsyncIterator[np.ndarray]:
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator
