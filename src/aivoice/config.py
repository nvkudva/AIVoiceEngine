from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SAMPLE_RATE = 24000

if os.uname().sysname == "Darwin":
    STATE_DIR = Path.home() / "Library" / "Application Support" / "AIVoiceEngine"
else:
    STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "aivoice"

CONFIG_PATH = STATE_DIR / "config.json"


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8791
    autostart: bool = True

    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    llm_model: str = "mlx-community/Qwen3-8B-4bit"
    tts_model: str = "prince-canuma/Kokoro-82M-4bit"
    tts_voice: str = "af_heart"

    instructions: str = (
        "You are a warm, quick-witted voice assistant. Speak the way a person does out "
        "loud: short sentences, contractions, no lists, no markdown, no emoji. Keep "
        "answers to a couple of sentences unless asked for more."
    )
    temperature: float = 0.8
    max_tokens: int = 300

    vad_threshold: float = 0.02
    vad_silence_ms: int = 600
    vad_min_speech_ms: int = 200

    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        if CONFIG_PATH.exists():
            try:
                stored = json.loads(CONFIG_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                stored = {}
            known = {f for f in cls.__dataclass_fields__}
            for key, value in stored.items():
                if key in known:
                    setattr(cfg, key, value)
        for key in cls.__dataclass_fields__:
            env = os.environ.get(f"AIVOICE_{key.upper()}")
            if env is None:
                continue
            current = getattr(cfg, key)
            if isinstance(current, bool):
                setattr(cfg, key, env.lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                try:
                    setattr(cfg, key, int(env))
                except ValueError:
                    continue
            elif isinstance(current, float):
                try:
                    setattr(cfg, key, float(env))
                except ValueError:
                    continue
            elif isinstance(current, str):
                setattr(cfg, key, env)
        return cfg

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    def as_dict(self) -> dict:
        return asdict(self)
