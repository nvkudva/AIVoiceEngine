# Code review — 2026-02-03

Scope: `src/aivoice/` as of `13173f5`. Read every file. No tests exist; findings below are from reading, not from running.

## Blocking

1. Declared entry point is missing. `pyproject.toml` ships `aivoice = "aivoice.__main__:main"`, but there is no `src/aivoice/__main__.py` and nothing constructs the FastAPI app, the WebSocket route, `/v1/health`, or the control page. Nothing here can boot.
   - Verify: `python -c "import aivoice.__main__"` fails; `pip install -e .` leaves `aivoice` broken.
   - Smallest fix: add `__main__.py` + server wiring, or drop the script entry until it exists.

2. Documented HTTP surface does not exist in code. `PLAN.md` and `readme.md` promise `/v1/health` and a control page on port 8791; `realtime.py` only implements the socket session, and no file serves HTTP.
   - Verify: grep for `health` finds only comments/error strings.
   - Smallest fix: either land the server file or mark those docs as unimplemented.

## Should fix soon

3. `realtime.py:220` — LLM tokens pulled via `loop.run_in_executor(None, lambda: next(stream, None))` per token. Works, but one thread-pool round trip per token and the sync generator holds a worker the whole turn. Prefer an async bridge or a dedicated thread with a queue (same pattern `tts.py` already uses).
   - Verify: stream N tokens, count executor submissions; currently N.

4. `realtime.py:233` — sentence flush splits on `[.!?;:\n]` with `MIN_SENTENCE_CHARS = 12`. Colons, semicolons, and short fragments flush early, so TTS speaks fragments ("e.g.", "12:30", "well: ..."). Narrow the split to `.`, `!`, `?` + newline, or require a longer tail before flushing on `;`/`:`.
   - Verify: feed `"meet at 12:30 sharp"` and observe two TTS calls.

5. `tts.py:32-46` — cancelled utterance can strand the producer thread. On `response.cancel`, `_respond` stops consuming while `produce()` keeps `chunks.put(...)` into a `maxsize=8` queue and then blocks forever. Daemon thread, so exit is fine, but repeated barge-ins leak threads.
   - Verify: start/cancel 50 utterances, count live threads.
   - Smallest fix: `put_nowait` + drop on full, or a stop flag checked in `produce()`.

6. `engine.py:44-48` — `start()` reports `state = "running"` even when adapters failed and only stubs are up. `error` is set, but `running` is true, so the socket's "engine stopped" warning never fires. Decide: `running` means models ready, or keep degraded-running and say so explicitly in `status()`.
   - Verify: break one model load, read `status()` — `state` says running with `adapters[].ready = false`.

7. `audio.py:25-32` — `resample()` can build a zero-length index for sub-sample inputs (`dst_len = 0`), and `np.interp` then fails. Short VAD chunks hit this first.
   - Verify: `resample(np.zeros(1, dtype=np.float32), 24000, 16000)`.
   - Smallest fix: return early when `dst_len < 2`.

## Minor

8. `config.py:62-65` — `int(env)` / `float(env)` raise on a typo'd `AIVOICE_*` value and kill startup. Wrap and ignore with a warning, matching the tolerant JSON-load path above.
9. `realtime.py:97-101` — `session.update` accepts `turn_detection` without validating `type`; a typo silently becomes `server_vad`. Reject unknown types.
10. `engine.py:57-72` — any `_model` config change unloads and reloads all three adapters. Fine for now; note it, don't add hot-swap until startup is measured slow.

## What is good

- Adapter seam is small and honest: lazy `load()`, stub fallback, `status()` for the UI. Matches the plan.
- Turn lifecycle reads cleanly: VAD edge → commit → transcribe → stream LLM → sentence-chunked TTS, with barge-in wired at the VAD start edge.
- Audio helpers stay dependency-free (`audio.py`, `vad.py`). No scipy for one lerp.

## Suggested next steps

1. Land `__main__.py` + HTTP/health/control-page wiring or unship the claim.
2. Add the first test: VAD start/stop edges + `resample()` boundaries. No framework needed beyond what ships.
3. Fix TTS thread leak and resample edge before soak-testing barge-in.
