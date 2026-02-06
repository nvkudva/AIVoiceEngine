from __future__ import annotations

import asyncio
import re
import time
import uuid

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .adapters.vad import EnergyVAD
from .audio import b64_to_float, float_to_b64
from .config import SAMPLE_RATE
from .engine import Engine

SENTENCE_END = re.compile(r"[.!?;:\n]")
MIN_SENTENCE_CHARS = 12


def _eid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class RealtimeSession:
    """One client connection, speaking a subset of the OpenAI Realtime protocol.

    Audio is pcm16 / 24 kHz / mono, base64-encoded, in both directions.
    """

    def __init__(self, ws: WebSocket, engine: Engine) -> None:
        self.ws = ws
        self.engine = engine
        self.id = _eid("sess")
        cfg = engine.config
        self.instructions = cfg.instructions
        self.voice = cfg.tts_voice
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.turn_detection = "server_vad"
        self.history: list[dict] = []
        self.buffer: list[np.ndarray] = []
        self.vad = EnergyVAD(cfg.vad_threshold, cfg.vad_silence_ms, cfg.vad_min_speech_ms)
        self.response_task: asyncio.Task | None = None

    # ---------- wire helpers ----------

    async def send(self, type_: str, **payload) -> None:
        try:
            await self.ws.send_json({"event_id": _eid("evt"), "type": type_, **payload})
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def error(self, message: str, code: str = "engine_error") -> None:
        await self.send("error", error={"type": "invalid_request_error", "code": code, "message": message})

    def session_payload(self) -> dict:
        return {
            "id": self.id,
            "object": "realtime.session",
            "model": self.engine.config.llm_model,
            "modalities": ["text", "audio"],
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "sample_rate": SAMPLE_RATE,
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_tokens,
            "turn_detection": None if self.turn_detection is None else {"type": self.turn_detection},
        }

    # ---------- main loop ----------

    async def run(self) -> None:
        await self.ws.accept()
        self.engine.sessions.add(self)
        await self.send("session.created", session=self.session_payload())
        if not self.engine.running:
            await self.error("Engine is stopped. Start it from the control page.", "engine_stopped")
        try:
            while True:
                event = await self.ws.receive_json()
                await self.handle(event)
        except WebSocketDisconnect:
            pass
        finally:
            await self.cancel_response()
            self.engine.sessions.discard(self)

    async def handle(self, event: dict) -> None:
        type_ = event.get("type", "")
        if type_ == "session.update":
            patch = event.get("session", {})
            self.instructions = patch.get("instructions", self.instructions)
            self.voice = patch.get("voice", self.voice)
            self.temperature = patch.get("temperature", self.temperature)
            if "max_response_output_tokens" in patch:
                self.max_tokens = int(patch["max_response_output_tokens"])
            if "turn_detection" in patch:
                td = patch["turn_detection"]
                if td is None:
                    self.turn_detection = None
                elif isinstance(td, dict) and td.get("type") in ("server_vad",):
                    self.turn_detection = td["type"]
            await self.send("session.updated", session=self.session_payload())

        elif type_ == "input_audio_buffer.append":
            await self.append_audio(event.get("audio", ""))

        elif type_ == "input_audio_buffer.commit":
            await self.commit(create_response=False)

        elif type_ == "input_audio_buffer.clear":
            self.buffer.clear()
            self.vad.reset()
            await self.send("input_audio_buffer.cleared")

        elif type_ == "conversation.item.create":
            item = event.get("item", {})
            text = "".join(part.get("text", "") for part in item.get("content", []))
            if text:
                self.history.append({"role": item.get("role", "user"), "content": text})
            await self.send("conversation.item.created", item=item)

        elif type_ == "response.create":
            await self.start_response()

        elif type_ == "response.cancel":
            await self.cancel_response()

        else:
            await self.error(f"Unknown event type: {type_}", "unknown_event")

    # ---------- audio in ----------

    async def append_audio(self, payload: str) -> None:
        if not payload:
            return
        samples = b64_to_float(payload)
        self.buffer.append(samples)
        if self.turn_detection is None:
            return
        edge = self.vad.push(samples)
        if edge == "start":
            await self.send("input_audio_buffer.speech_started", audio_start_ms=self._buffered_ms())
            if self.response_task and not self.response_task.done():
                await self.cancel_response()  # barge-in
        elif edge == "stop":
            await self.send("input_audio_buffer.speech_stopped", audio_end_ms=self._buffered_ms())
            await self.commit(create_response=True)

    def _buffered_ms(self) -> int:
        total = sum(chunk.size for chunk in self.buffer)
        return int(1000 * total / SAMPLE_RATE)

    def take_buffer(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self.buffer)
        self.buffer.clear()
        self.vad.reset()
        return audio

    async def commit(self, create_response: bool) -> None:
        audio = self.take_buffer()
        item_id = _eid("item")
        await self.send("input_audio_buffer.committed", item_id=item_id)
        if audio.size == 0:
            return

        text = await asyncio.to_thread(self.engine.stt.transcribe, audio, SAMPLE_RATE)
        if not text:
            await self.send(
                "conversation.item.input_audio_transcription.failed",
                item_id=item_id,
                error={"code": "no_speech", "message": "Nothing transcribed from the committed audio."},
            )
            return

        self.history.append({"role": "user", "content": text})
        await self.send(
            "conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            content_index=0,
            transcript=text,
        )
        if create_response:
            await self.start_response()

    # ---------- response out ----------

    async def start_response(self) -> None:
        if self.response_task and not self.response_task.done():
            return
        self.response_task = asyncio.create_task(self._respond())

    async def cancel_response(self) -> None:
        task = self.response_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _respond(self) -> None:
        started = time.time()
        response_id = _eid("resp")
        item_id = _eid("item")
        await self.send("response.created", response={"id": response_id, "status": "in_progress"})

        messages = [{"role": "system", "content": self.instructions}, *self.history[-20:]]
        spoken = ""
        pending = ""
        first_token_at: float | None = None

        try:
            loop = asyncio.get_running_loop()
            stream = self.engine.llm.stream(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            while True:
                token = await loop.run_in_executor(None, lambda: next(stream, None))
                if token is None:
                    break
                if first_token_at is None:
                    first_token_at = time.time()
                spoken += token
                pending += token
                await self.send(
                    "response.audio_transcript.delta",
                    response_id=response_id,
                    item_id=item_id,
                    delta=token,
                )
                if SENTENCE_END.search(token) and len(pending) >= MIN_SENTENCE_CHARS:
                    await self._speak(pending, response_id, item_id)
                    pending = ""

            if pending.strip():
                await self._speak(pending, response_id, item_id)

            self.history.append({"role": "assistant", "content": spoken})
            await self.send(
                "response.audio_transcript.done",
                response_id=response_id,
                item_id=item_id,
                transcript=spoken,
            )
            await self.send("response.audio.done", response_id=response_id, item_id=item_id)
            await self.send(
                "response.done",
                response={"id": response_id, "status": "completed", "output_text": spoken},
            )
            self.engine.turns += 1
            self.engine.last_latency_ms = round(
                1000 * ((first_token_at or time.time()) - started), 1
            )
        except asyncio.CancelledError:
            await self.send("response.done", response={"id": response_id, "status": "cancelled"})
            raise
        except Exception as exc:  # noqa: BLE001 - report and keep the socket alive
            await self.error(f"{type(exc).__name__}: {exc}")
            await self.send("response.done", response={"id": response_id, "status": "failed"})

    async def _speak(self, text: str, response_id: str, item_id: str) -> None:
        async for chunk in self.engine.tts.stream(text.strip(), self.voice):
            await self.send(
                "response.audio.delta",
                response_id=response_id,
                item_id=item_id,
                delta=float_to_b64(chunk),
            )
