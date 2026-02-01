from __future__ import annotations

from typing import Iterator

from .base import LLMAdapter


class MLXLLM(LLMAdapter):
    """Local chat model via mlx-lm."""

    def _load(self) -> None:
        from mlx_lm import load

        self._model, self._tokenizer = load(self.model)

    def stream(self, messages: list[dict], temperature: float = 0.8, max_tokens: int = 300) -> Iterator[str]:
        if not self._loaded:
            yield "The language model is not loaded yet."
            return

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        sampler = make_sampler(temp=temperature)
        for chunk in stream_generate(
            self._model, self._tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            if chunk.text:
                yield chunk.text
