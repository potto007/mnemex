"""Tests for the runaway-generation guards: subcall_max_tokens, run deadlines,
and in-flight cancellation (born from the 2026-06-11 zombie-generation incident:
a timed-out ask left sub-calls generating 35K+ tokens server-side)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lm_repl.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from lm_repl.core.lm_handler import LMHandler
from lm_repl.utils.exceptions import CancellationError, TimeoutExceededError
from tests.mock_lm import MockLM


# ---------------------------------------------------------------------------
# subcall_max_tokens plumbing (handler -> client)
# ---------------------------------------------------------------------------


def test_single_subcall_gets_max_tokens_cap():
    mock = MockLM(responses=["ok"])
    with LMHandler(client=mock, subcall_max_tokens=2048) as handler:
        request = LMRequest(prompt="hi")
        response = send_lm_request(handler.address, request)
    assert response.success
    assert mock.seen_max_tokens == [2048]


def test_batched_subcalls_get_max_tokens_cap():
    mock = MockLM(responses=["a", "b", "c"])
    with LMHandler(client=mock, subcall_max_tokens=512) as handler:
        responses = send_lm_request_batched(handler.address, ["p1", "p2", "p3"])
    assert all(r.success for r in responses)
    assert mock.seen_max_tokens == [512, 512, 512]


def test_no_cap_by_default():
    mock = MockLM(responses=["ok"])
    with LMHandler(client=mock) as handler:
        response = send_lm_request(handler.address, LMRequest(prompt="hi"))
    assert response.success
    assert mock.seen_max_tokens == [None]


def test_root_completion_not_capped_by_subcall_limit():
    """subcall_max_tokens does not touch the root orchestrator path."""
    mock = MockLM(responses=["ok"])
    handler = LMHandler(client=mock, subcall_max_tokens=128)
    assert handler.completion("root prompt") == "ok"
    assert mock.seen_max_tokens == [None]


def test_root_completion_capped_by_root_limit():
    """root_max_tokens bounds root orchestrator generations: the forced final
    REDUCE on 2026-06-11 ran away to ~50K tokens (n_tokens 65024 at deadline
    cancel) because the root path had no cap at all."""
    mock = MockLM(responses=["ok"])
    handler = LMHandler(client=mock, subcall_max_tokens=128, root_max_tokens=8192)
    assert handler.completion("root prompt") == "ok"
    assert mock.seen_max_tokens == [8192]


def test_root_limit_does_not_leak_into_subcalls():
    """Sub-calls keep their own (tighter) cap when both are set."""
    mock = MockLM(responses=["ok"])
    with LMHandler(client=mock, subcall_max_tokens=128, root_max_tokens=8192) as handler:
        response = send_lm_request(handler.address, LMRequest(prompt="hi"))
    assert response.success
    assert mock.seen_max_tokens == [128]


def test_rlm_wires_root_max_tokens_through_to_root_calls():
    import lm_repl.core.rlm as rlm_module
    from lm_repl import RLM
    from tests.test_subcall import create_mock_lm, final

    with patch.object(rlm_module, "get_client") as mock_get_client:
        mock_lm = create_mock_lm([final("answer")])
        mock_get_client.return_value = mock_lm
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "m"},
            root_max_tokens=8192,
        )
        rlm.completion("context", root_prompt="q")
        root_call_kwargs = mock_lm.completion.call_args_list[0].kwargs
        assert root_call_kwargs.get("max_tokens") == 8192


# ---------------------------------------------------------------------------
# cancel_inflight / set_run_deadline fan-out
# ---------------------------------------------------------------------------


def _patched_openai_client(**kwargs):
    from lm_repl.clients.openai import OpenAIClient

    with patch("lm_repl.clients.openai.openai.OpenAI"), patch(
        "lm_repl.clients.openai.openai.AsyncOpenAI"
    ):
        return OpenAIClient(api_key="test-key", model_name="m", **kwargs)


def test_cancel_inflight_sets_event_on_all_clients():
    a = _patched_openai_client()
    b = _patched_openai_client()
    handler = LMHandler(client=a, other_backend_client=b)
    assert not a.cancel_event.is_set() and not b.cancel_event.is_set()
    handler.cancel_inflight()
    assert a.cancel_event.is_set() and b.cancel_event.is_set()


def test_cancel_inflight_tolerates_clients_without_event():
    mock = MockLM()
    handler = LMHandler(client=mock)
    handler.cancel_inflight()  # must not raise


def test_set_run_deadline_arms_clients():
    a = _patched_openai_client()
    handler = LMHandler(client=a)
    handler.set_run_deadline(300.0)
    assert a._deadline is not None
    handler.set_run_deadline(None)
    assert a._deadline is None


# ---------------------------------------------------------------------------
# OpenAIClient abort behavior
# ---------------------------------------------------------------------------


def _chunk(content=None, usage=None):
    choices = []
    if content is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=choices, usage=usage)


_USAGE = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class _FakeStream:
    """Iterable chat-completion stream that records close()."""

    def __init__(self, chunks, on_yield=None):
        self._chunks = list(chunks)
        self._on_yield = on_yield
        self.closed = False

    def __iter__(self):
        for i, chunk in enumerate(self._chunks):
            if self._on_yield:
                self._on_yield(i)
            yield chunk

    def close(self):
        self.closed = True


def test_stream_completion_assembles_chunks_and_tracks_usage():
    client = _patched_openai_client(stream=True)
    stream = _FakeStream([_chunk("hel"), _chunk("lo"), _chunk(usage=_USAGE)])
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = stream

    assert client.completion("hi") == "hello"
    assert stream.closed
    assert client.client.chat.completions.create.call_args.kwargs["stream"] is True
    assert client.model_output_tokens["m"] == 5


def test_stream_aborts_on_cancel_event():
    client = _patched_openai_client(stream=True)
    # Set the event after the first chunk is yielded
    stream = _FakeStream(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        on_yield=lambda i: client.cancel_event.set() if i == 1 else None,
    )
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = stream

    with pytest.raises(CancellationError):
        client.completion("hi")
    assert stream.closed  # the server-side generation is torn down


def test_stream_aborts_when_deadline_expires_mid_generation():
    client = _patched_openai_client(stream=True)
    stream = _FakeStream(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        on_yield=lambda i: client.set_deadline(0.0) if i == 1 else None,
    )
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = stream

    with pytest.raises(TimeoutExceededError):
        client.completion("hi")
    assert stream.closed


def test_expired_deadline_prevents_new_request():
    client = _patched_openai_client(stream=True)
    client.client = MagicMock()
    client.set_deadline(0.0)  # already expired before the call
    with pytest.raises(TimeoutExceededError):
        client.completion("hi")
    client.client.chat.completions.create.assert_not_called()


def test_cancelled_call_never_starts_new_generation():
    """A call that was queued past cancellation must not hit the backend."""
    client = _patched_openai_client(stream=True)
    client.client = MagicMock()
    client.cancel_event.set()
    with pytest.raises(CancellationError):
        client.completion("hi")
    client.client.chat.completions.create.assert_not_called()


def test_nonstream_call_also_checks_abort_before_request():
    client = _patched_openai_client()  # stream=False
    client.client = MagicMock()
    client.cancel_event.set()
    with pytest.raises(CancellationError):
        client.completion("hi")
    client.client.chat.completions.create.assert_not_called()


def test_completion_max_tokens_lands_in_request():
    client = _patched_openai_client()
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=_USAGE,
    )
    client.completion("hi", max_tokens=777)
    assert client.client.chat.completions.create.call_args.kwargs["max_tokens"] == 777
