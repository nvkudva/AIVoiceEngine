from __future__ import annotations

import asyncio
import queue
import threading
from typing import AsyncIterator

import numpy as np

from ..audio import resample
from ..config import SAMPLE_RATE
from .base import TTSAdapter

_SENTINEL = object()


class KokoroTTS(TTSAdapter):
    """Kokoro-82M via mlx-audio. Natively 24 kHz, which is the wire rate."""

    def _load(self) -> None:
        from mlx_audio.tts.utils import load_model

        self._model = load_model(self.model)

    async def stream(self, text: str, voice: str) -> AsyncIterator[np.ndarray]:
        if not self._loaded or not text.strip():
            return

        loop = asyncio.get_running_loop()
        chunks: queue.Queue = queue.Queue()

        def produce() -> None:
            try:
                for segment in self._model.generate(text=text, voice=voice, speed=1.0):
                    audio = np.asarray(segment.audio, dtype=np.float32).reshape(-1)
                    rate = getattr(segment, "sample_rate", SAMPLE_RATE)
                    chunks.put(resample(audio, rate, SAMPLE_RATE))
            except Exception as exc:  # noqa: BLE001 - a failed utterance must not kill the session
                chunks.put(exc)
            finally:
                chunks.put(_SENTINEL)

        threading.Thread(target=produce, daemon=True).start()

        while True:
            item = await loop.run_in_executor(None, chunks.get)
            if item is _SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item
