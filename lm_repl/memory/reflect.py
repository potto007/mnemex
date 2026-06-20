"""A reflect function backed by an OpenAI-compatible chat endpoint.

The :class:`~lm_repl.memory.distill.TraceDistiller` needs a ``prompt -> text``
callable to distill experiences. This adapter implements it over the same kind
of OpenAI-compatible server lm-repl drives, so distillation can use a dedicated
judge/memory model. Injected client keeps it unit-testable without network.
"""
from __future__ import annotations

from typing import Any


class OpenAIReflectFn:
    """Callable ``prompt -> str`` over ``client.chat.completions.create``."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float = 0.3,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

    def __call__(self, prompt: str) -> str:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        return resp.choices[0].message.content or ""

    @classmethod
    def from_config(
        cls, *, base_url: str, model: str, api_key: str = "EMPTY", **kwargs: Any
    ) -> OpenAIReflectFn:
        """Build a reflect fn backed by a real ``openai.OpenAI`` client."""
        import openai

        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        return cls(client, model=model, **kwargs)
