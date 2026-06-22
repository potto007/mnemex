"""Tests for token counting, context limits, and subcall limit resolution."""

from prehend.utils.token_utils import (
    CONSERVATIVE_CHARS_PER_TOKEN,
    count_tokens,
    get_context_limit,
    resolve_subcall_limit,
)


class TestGetContextLimit:
    """Tests for get_context_limit, especially gemma keys."""

    def test_gemma_4_sft_kb_v13_not_default_128k(self):
        # The v13 sft model name must NOT silently fall back to 128000.
        limit = get_context_limit("gemma-4-12b-it-sft-kb-v13-sft")
        assert limit != 128_000
        assert limit == 262_144

    def test_bare_gemma_key(self):
        assert get_context_limit("gemma") == 262_144

    def test_gemma_4_key(self):
        assert get_context_limit("gemma-4") == 262_144

    def test_unknown_model_still_default(self):
        assert get_context_limit("totally-unknown-model-xyz") == 128_000

    def test_empty_and_unknown_sentinel(self):
        assert get_context_limit("") == 128_000
        assert get_context_limit("unknown") == 128_000

    def test_longest_key_wins_preserved(self):
        # gpt-4o-mini (longer key) must beat gpt-4 / gpt-4o.
        assert get_context_limit("@openai/gpt-4o-mini") == 128_000


class TestCountTokensConservative:
    """count_tokens must NOT undercount for gemma (dense tokenizer)."""

    def test_gemma_does_not_undercount_vs_naive_char4(self):
        # Dense structured text: gemma estimate must be strictly larger than the
        # naive char/4 count (the old undercount path).
        text = "def f(x): return {'a': [1,2,3], 'b': x*x}  # dense_code_$%^&*()"
        text = text * 50
        messages = [{"role": "user", "content": text}]
        naive_char4 = (len(text) + 3) // 4
        est = count_tokens(messages, "gemma-4-12b-it-sft-kb-v13-sft")
        assert est > naive_char4

    def test_gemma_estimate_at_least_conservative_lower_bound(self):
        text = "The quick brown fox jumps over the lazy dog. " * 100
        messages = [{"role": "user", "content": text}]
        est = count_tokens(messages, "gemma-4-12b-it-sft-kb-v13-sft")
        # Conservative lower bound: chars / CONSERVATIVE_CHARS_PER_TOKEN.
        lower = int(len(text) / CONSERVATIVE_CHARS_PER_TOKEN)
        assert est >= lower

    def test_conservative_constant_is_below_average(self):
        # Must over-estimate vs the 4.0 average to bias toward over-counting.
        assert CONSERVATIVE_CHARS_PER_TOKEN < 4.0
        assert CONSERVATIVE_CHARS_PER_TOKEN > 0

    def test_empty_messages_zero(self):
        assert count_tokens([], "gemma-4-12b-it-sft-kb-v13-sft") == 0


class TestResolveSubcallLimit:
    """resolve_subcall_limit precedence: explicit > runtime_ctx > table."""

    def test_explicit_wins(self):
        assert (
            resolve_subcall_limit(
                "gemma-4-12b-it-sft-kb-v13-sft", explicit=98_304, runtime_ctx=50_000
            )
            == 98_304
        )

    def test_runtime_ctx_when_no_explicit(self):
        assert (
            resolve_subcall_limit(
                "gemma-4-12b-it-sft-kb-v13-sft", explicit=None, runtime_ctx=50_000
            )
            == 50_000
        )

    def test_falls_back_to_table_when_all_none(self):
        assert (
            resolve_subcall_limit("gemma-4-12b-it-sft-kb-v13-sft")
            == 262_144
        )

    def test_fallback_uses_get_context_limit_for_unknown(self):
        assert resolve_subcall_limit("unknown-model") == 128_000

    def test_never_raises_on_weird_input(self):
        # Should not raise even with empty model name.
        assert resolve_subcall_limit("") == 128_000
