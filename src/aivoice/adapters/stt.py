from __future__ import annotations

import numpy as np

from ..audio import resample
from .base import STTAdapter


class ParakeetSTT(STTAdapter):
    """Parakeet TDT v3 via parakeet-mlx (Apache-2.0, runs on the Neural Engine).

    Parakeet expects 16 kHz mono; the wire format is 24 kHz, so we downsample here.
    """

    MODEL_RATE = 16000

    def _load(self) -> None:
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(self.model)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if not self._loaded:
            return ""
        import mlx.core as mx

        audio = resample(samples, sample_rate, self.MODEL_RATE)
        if audio.size < self.MODEL_RATE // 10:
            return ""
        result = self._model.transcribe(mx.array(audio))
        return (getattr(result, "text", "") or "").strip()
