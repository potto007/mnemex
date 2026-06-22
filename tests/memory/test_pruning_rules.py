"""Tests for the prehend anti-give-up write-time filter."""
from __future__ import annotations

import pytest

from prehend.memory.pruning_rules import is_anti_give_up


@pytest.mark.parametrize("text", [
    "data not available, return unknown",
    "Insufficient data to answer.",
    "Cannot determine the result.",
    "no data available for this query",
])
def test_capitulation_text_is_flagged(text):
    assert is_anti_give_up(text) is True


# Real give-up insights that poisoned a v13 multihop bank (2026-06-21). The
# guard's "data ... missing/unavailable" patterns missed the "information is
# missing" / "context is garbled/ciphertext/nonsensical/unintelligible" family,
# so a self-reinforcing capitulation cascade filled the bank. These must flag.
@pytest.mark.parametrize("text", [
    "When a query asks for specific information not present in the provided context, "
    "explicitly state that the information is missing and briefly summarize what is available.",
    "When the provided context contains only nonsensical, randomized, or irrelevant data "
    "that fails to address the query, explicitly state that the information is missing.",
    "When the provided context contains only garbled text, ciphertext, or nonsensical data "
    "that lacks any human-readable information, explicitly state that the information is missing.",
    "When the provided context is completely unintelligible, garbled, or contains no "
    "human-readable information, do not attempt to hallucinate or infer details.",
])
def test_garbled_context_give_up_is_flagged(text):
    assert is_anti_give_up(text) is True


@pytest.mark.parametrize("text", [
    "first compute the ratio, then divide by the base",
    "When the value is missing, retry with a wider window before concluding.",
    "Do not give up; re-read the context from the start.",
    # Mentions a missing cell but is a constructive cross-reference strategy.
    "Cross-reference the two tables to fill a missing cell, then sum the rows.",
    # Mentions "garbled" but instructs to re-read -> protective override keeps it.
    "If a section looks garbled, re-read it carefully; it usually decodes to a table.",
])
def test_useful_experience_is_not_flagged(text):
    assert is_anti_give_up(text) is False


def test_protective_override_beats_capitulation_phrase():
    # Mentions "data not available" but instructs to retry first -> keep it.
    text = "If data not available, retry with a different parameter before concluding."
    assert is_anti_give_up(text) is False


# Contrastive failure channel (ADR-0010 / 2026-06-22 spec): the FAILURE distiller
# emits behavioral premature-stop guards that is_anti_give_up (tuned for
# capitulation WORDING) does NOT catch. is_premature_stop is a SEPARATE filter
# applied ONLY to failure-channel content (folding into is_anti_give_up would
# regress the success path).
from prehend.memory.pruning_rules import is_premature_stop


@pytest.mark.parametrize("text", [
    "When chunks conflict, prefer the first match and stop searching.",
    "When the answer is uncertain, return the best available estimate rather than over-searching.",
    "Avoid exhaustive search; once a plausible candidate is found, return it.",
    "When you cannot verify across all chunks, commit to the most frequent value.",
    "When a multi-hop lookup is slow, narrow scope and answer with the partial result.",
    "When the question is ambiguous, pick the simplest interpretation and stop.",
])
def test_premature_stop_guards_flagged(text):
    assert is_premature_stop(text) is True


@pytest.mark.parametrize("text", [
    "When chunks conflict, re-read and cross-check across all chunks before concluding.",
    "When a lookup seems slow, decompose into per-entity sub-queries and verify each.",
    "When the first match is found, keep scanning the remaining chunks to confirm.",
])
def test_constructive_failure_guard_survives(text):
    assert is_premature_stop(text) is False
