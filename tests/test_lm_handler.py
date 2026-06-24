"""Tests for LMHandler using MockLM (no real LM required)."""

from prehend.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from prehend.core.lm_handler import LMHandler
from tests.mock_lm import MockLM


def test_lm_handler_single_request():
    """Single prompt request returns success and echo-style content."""
    mock = MockLM(responses=["hello back"])
    with LMHandler(client=mock) as handler:
        request = LMRequest(prompt="hello")
        response = send_lm_request(handler.address, request)
    assert response.success
    assert response.chat_completion is not None
    assert response.chat_completion.response == "hello back"


def test_lm_handler_batched_request():
    """Batched prompts return one response per prompt in order."""
    responses = [f"r{i}" for i in range(5)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=3) as handler:
        prompts = [f"prompt-{i}" for i in range(5)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == 5
    for i, resp in enumerate(result):
        assert resp.success, resp.error
        assert resp.chat_completion is not None
        assert resp.chat_completion.response == f"r{i}"


def test_lm_handler_batched_one_failure_does_not_poison_siblings():
    """One sub-call raising must NOT turn the whole batch into errors.

    Regression (2026-06-24 multihop epic-fail RCA): _handle_batched used
    asyncio.gather without return_exceptions, so a single oversized chunk's 400
    tore down the event loop and every in-flight sibling came back as
    APIConnectionError ("Connection error."). The good chunks must still return
    their answers; the bad one degrades to an "Error:" string that map_reduce
    already filters out of the reduce.
    """
    def fn(prompt):
        if "POISON" in str(prompt):
            raise RuntimeError("Connection error.")
        return f"ok:{prompt}"

    mock = MockLM(response_fn=fn)
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        prompts = ["p-0", "p-1", "POISON", "p-3", "p-4"]
        result = send_lm_request_batched(handler.address, prompts)

    assert len(result) == 5
    # Good siblings preserved (the poison must not cascade onto them).
    for i in (0, 1, 3, 4):
        assert result[i].success, result[i].error
        assert result[i].chat_completion.response == f"ok:p-{i}"
    # The failing chunk surfaces as a per-prompt error string, not a batch-wide kill.
    assert result[2].chat_completion is not None
    assert result[2].chat_completion.response.startswith("Error:")


def test_lm_handler_batched_many_prompts_semaphore_cap():
    """Many prompts complete successfully with semaphore limiting concurrency."""
    # 50 prompts, max 4 concurrent: should still all complete
    count = 50
    responses = [f"resp-{i}" for i in range(count)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        prompts = [f"p-{i}" for i in range(count)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == count
    for i, resp in enumerate(result):
        assert resp.success, (i, resp.error)
        assert resp.chat_completion.response == f"resp-{i}"
