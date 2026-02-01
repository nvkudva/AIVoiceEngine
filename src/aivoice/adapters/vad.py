from __future__ import annotations

import numpy as np

from ..audio import rms
from ..config import SAMPLE_RATE


class EnergyVAD:
    """Turn detection on short-term energy.

    Deliberately dependency-free so the engine detects turns on first run, before
    any model has been downloaded. Silero replaces this once it is wired up.
    """

    def __init__(self, threshold: float, silence_ms: int, min_speech_ms: int) -> None:
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self._speech_ms = 0
        self._silence_ms = 0
        self.level = 0.0

    def push(self, samples: np.ndarray) -> str | None:
        """Feed a chunk. Returns 'start', 'stop', or None."""
        if samples.size == 0:
            return None
        chunk_ms = 1000.0 * samples.size / SAMPLE_RATE
        level = rms(samples)
        # Slow attack, fast decay: keeps the meter readable without lagging the gate.
        self.level = max(level, self.level * 0.7)
        loud = level >= self.threshold

        if loud:
            self._silence_ms = 0
            self._speech_ms += chunk_ms
            if not self.speaking and self._speech_ms >= self.min_speech_ms:
                self.speaking = True
                return "start"
        else:
            if self.speaking:
                self._silence_ms += chunk_ms
                if self._silence_ms >= self.silence_ms:
                    self.speaking = False
                    self._speech_ms = 0
                    self._silence_ms = 0
                    return "stop"
            else:
                self._speech_ms = 0
        return None
