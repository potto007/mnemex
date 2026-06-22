"""
Deterministic input-size guard for sub-calls (reject-with-hint).

The RLM premise is context-by-reference: large context lives in a REPL variable
and is sliced/queried, never fed whole to a sub-model. When a sub-call prompt
exceeds the sub-model's context window the server 400s ("exceeds available
context size") and the trajectory spins to a hard timeout. This module provides
a PURE, arithmetic guard that the orchestrator can act on: rather than failing
open like an LM verifier, it returns an actionable rejection string telling the
model to chunk the context and map-reduce via rlm_query_batched.

Wording deliberately mirrors the strategy-verifier rejection style
("... rejected this call: <reason>") but is self-contained and actionable.
"""

import math

from prehend.utils.token_utils import (
    CONSERVATIVE_CHARS_PER_TOKEN,
    count_tokens,
)


def safe_chunk_chars(limit: int, model: str, margin_frac: float = 0.15) -> int:
    """
    Return K: the max chunk size in CHARACTERS that safely fits one sub-call.

    Derived from the safe token budget (limit minus a margin reserved for the
    system+user prompt envelope and tokenizer skew) converted to chars with the
    conservative chars-per-token. Pure. Always >= 1.
    """
    safe_tokens = math.floor(limit * (1 - margin_frac))
    chars = int(safe_tokens * CONSERVATIVE_CHARS_PER_TOKEN)
    return max(1, chars)


def oversize_rejection(
    prompt: str,
    *,
    limit: int,
    model: str,
    margin_frac: float = 0.15,
) -> str | None:
    """
    Return None if the prompt fits the safe budget, else an actionable rejection.

    Fits when count_tokens([{user: prompt}], model) <= floor(limit*(1-margin_frac)).
    Otherwise returns a string that (a) names the limit and the prompt's estimated
    token size and (b) instructs the model to split the context into chunks of
    <= K characters and map-reduce via rlm_query_batched.
    """
    est_tokens = count_tokens([{"role": "user", "content": prompt}], model)
    safe_tokens = math.floor(limit * (1 - margin_frac))
    if est_tokens <= safe_tokens:
        return None
    chunk_chars = safe_chunk_chars(limit, model, margin_frac)
    return (
        f"Sub-call input guard rejected this call: the prompt is ~{est_tokens} "
        f"tokens, which exceeds the safe budget of {safe_tokens} tokens "
        f"(sub-model context limit {limit} tokens, with a {int(margin_frac * 100)}% "
        f"margin reserved for the prompt envelope and tokenizer skew). Do NOT pass "
        f"this much context to a single sub-call. Instead, split the context into "
        f"chunks of <= {chunk_chars} characters each and map-reduce them via "
        f"rlm_query_batched, then combine the per-chunk results."
    )
