# AI Voice Engine

Local voice engine: OpenAI Realtime-compatible speech API running on your own machine. No cloud, no per-minute billing.

Install once, get a WebSocket endpoint at `ws://127.0.0.1:8791`. Existing OpenAI Realtime client SDKs work by changing the URL.

## Pipeline

```
mic audio (pcm16 24k) ──ws──> RealtimeSession
                                 │
                              VAD (silero | energy fallback)
                                 │ speech_stopped
                              STT adapter  ──partial/final transcript──> client
                                 │
                              LLM adapter  ──token deltas──> client
                                 │ sentence chunks
                              TTS adapter  ──pcm16 24k deltas──> client
```

`engine.py` owns the turn state machine. Adapters are swappable and know nothing about the wire protocol. Every adapter is lazily imported: a missing dependency degrades that stage to a stub, reported by `/v1/health`, rather than crashing the server.

## Default models

All free, permissive, Apple-Silicon-accelerated:

- STT: `mlx-community/parakeet-tdt-0.6b-v3`
- LLM: `mlx-community/Qwen3-8B-4bit` (`Qwen3-30B-A3B` on 32 GB+)
- TTS: `Kokoro-82M`
- VAD: Silero, with built-in energy VAD fallback

Audio: pcm16, 24 kHz, mono, little-endian, base64 on the wire.

## Run

Requires Python 3.11+.

```bash
pip install -e ".[mac]"
aivoice
```

Control page: `http://127.0.0.1:8791` — start/stop the engine, pick models, watch live meters. State lives in the server, not the page.

## Layout

- `src/aivoice/realtime.py` — OpenAI Realtime-compatible WebSocket server
- `src/aivoice/engine.py` — turn state machine
- `src/aivoice/adapters/` — STT / LLM / TTS / VAD adapters
- `src/aivoice/config.py` — config, defaults to port 8791
- `src/aivoice/audio.py` — pcm16 helpers

See [PLAN.md](PLAN.md) for architecture and decisions, [TODO.md](TODO.md) for progress.
