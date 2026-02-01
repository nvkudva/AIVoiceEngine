# AI Voice Engine — Plan

## Goal
A locally-installed voice engine that gives any application a Google-Assistant-class
speech API on the user's own machine. Install once, get a WebSocket endpoint.
No cloud, no per-minute billing, no manual model plumbing.

## Product shape
- macOS `.app` first (Linux/Windows later, same Python core).
- Tiny PWA control page at `http://127.0.0.1:8791` — start/stop the engine, pick models, watch live meters.
- The engine itself is a local HTTP + WebSocket server; the PWA is just its remote control.

## Architecture
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
- `engine.py` owns the turn state machine; adapters are swappable and know nothing about the wire protocol.
- Every adapter is lazily imported. A missing dependency degrades that stage to a stub and is reported by `/v1/health` rather than crashing the server — the control page must always come up.

## Decisions
- **API: OpenAI Realtime-compatible WebSocket.** Existing client SDKs work by changing a URL. A native protocol was rejected: it would make every integration bespoke for no functional gain.
- **Runtime: bundled Python + MLX.** Rejected Ollama (forces a second install, breaks the one-click promise) and native Swift (smallest binary, but locks the core to macOS and blocks the Linux/Windows goal).
- **Audio: pcm16, 24 kHz, mono, little-endian, base64 on the wire.** Matches the OpenAI Realtime default and Kokoro's native output rate, so the TTS path needs no resampling.
- **Default models** (all free, permissive, Apple-Silicon-accelerated):
  - STT `mlx-community/parakeet-tdt-0.6b-v3` — Apache-2.0, streaming, runs on the Neural Engine.
  - LLM `mlx-community/Qwen3-8B-4bit` — best latency/quality in its class on MLX; `Qwen3-30B-A3B` on 32 GB+.
  - TTS `Kokoro-82M` — 82M params, faster than real time on MPS, Apache-2.0.
  - VAD Silero, with a built-in energy VAD so first run works before any download.
- **State lives in the server, not the PWA.** The page can be closed, reopened, or opened on a phone on the same LAN without disturbing a running session.

## Non-goals (for now)
- Speaker diarization, voice cloning, wake-word.
- Multi-tenant auth. Binds to 127.0.0.1; a LAN mode with a token comes later.
- Windows/Linux packaging (the Python core stays portable, the installers do not exist yet).

## Revisions
