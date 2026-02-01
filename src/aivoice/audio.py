from __future__ import annotations

import base64

import numpy as np


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def b64_to_float(payload: str) -> np.ndarray:
    return pcm16_to_float(base64.b64decode(payload))


def float_to_b64(samples: np.ndarray) -> str:
    return base64.b64encode(float_to_pcm16(samples)).decode("ascii")


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. Good enough for speech; avoids a scipy dependency."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / src_rate
    dst_len = int(round(duration * dst_rate))
    src_idx = np.linspace(0, samples.size - 1, dst_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(samples.size), samples).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
