"""Tests for the OpenAI-compatible reflect function (distiller's LLM call)."""
from __future__ import annotations

from types import SimpleNamespace

from lm_repl.memory.reflect import OpenAIReflectFn


class FakeChatCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(content))


def test_returns_message_content():
    client = FakeClient('{"key_insight": "k"}')
    reflect = OpenAIReflectFn(client, model="judge-model")
    assert reflect("distill this") == '{"key_insight": "k"}'


def test_sends_prompt_as_user_message_to_model():
    client = FakeClient("ok")
    reflect = OpenAIReflectFn(client, model="judge-model")
    reflect("the reflect prompt")
    call = client.chat.completions.create.__self__.calls[0]
    assert call["model"] == "judge-model"
    assert call["messages"][-1]["role"] == "user"
    assert call["messages"][-1]["content"] == "the reflect prompt"


def test_returns_empty_string_when_content_is_none():
    client = FakeClient(None)
    reflect = OpenAIReflectFn(client, model="m")
    assert reflect("x") == ""
