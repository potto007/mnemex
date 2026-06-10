"""Tests for the SRLM subclass - context-length routing, direct mode, and selection."""
from unittest.mock import MagicMock, patch

from lm_repl.core.rlm import RLM
from lm_repl.core.srlm import SRLM, _choose_mode, _build_direct_messages, _select_best
from lm_repl.core.types import RLMChatCompletion, UsageSummary


class TestChooseMode:
    def test_rlm_when_threshold_zero(self):
        assert _choose_mode(100, 0) == "rlm"

    def test_rlm_when_threshold_none(self):
        assert _choose_mode(100, None) == "rlm"

    def test_direct_when_below(self):
        assert _choose_mode(5000, 30000) == "direct"

    def test_rlm_when_at_threshold(self):
        assert _choose_mode(30000, 30000) == "rlm"

    def test_rlm_when_above(self):
        assert _choose_mode(50000, 30000) == "rlm"


class TestBuildDirectMessages:
    def test_roles(self):
        msgs = _build_direct_messages("ctx", "q")
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_content(self):
        msgs = _build_direct_messages("my data", "find X")
        user = msgs[1]["content"]
        assert "my data" in user
        assert "find X" in user


class TestSRLMInit:
    def test_inherits_rlm(self):
        from lm_repl import RLM
        assert issubclass(SRLM, RLM)

    def test_accepts_srlm_params(self):
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            direct_threshold=30000,
            n_candidates=4,
        )
        assert srlm.direct_threshold == 30000
        assert srlm.n_candidates == 4

    def test_defaults(self):
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
        )
        assert srlm.direct_threshold == 0
        assert srlm.n_candidates == 1


def _make_completion(response: str, exec_time: float) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="test",
        prompt="test prompt",
        response=response,
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=exec_time,
    )


class TestSelectBest:
    def test_single_candidate(self):
        c = _make_completion("42", 1.0)
        assert _select_best([c]) is c

    def test_majority_vote(self):
        c1 = _make_completion("42", 1.0)
        c2 = _make_completion("42", 2.0)
        c3 = _make_completion("99", 0.5)
        result = _select_best([c1, c2, c3])
        assert result.response == "42"

    def test_picks_shortest_trace_among_consistent(self):
        c1 = _make_completion("42", 3.0)
        c2 = _make_completion("42", 1.0)
        c3 = _make_completion("42", 2.0)
        assert _select_best([c1, c2, c3]) is c2

    def test_all_different_picks_any(self):
        c1 = _make_completion("a", 1.0)
        c2 = _make_completion("b", 2.0)
        c3 = _make_completion("c", 3.0)
        result = _select_best([c1, c2, c3])
        assert result in [c1, c2, c3]

    def test_case_insensitive_consistency(self):
        c1 = _make_completion("YES", 2.0)
        c2 = _make_completion("yes", 1.0)
        result = _select_best([c1, c2])
        assert result.execution_time == 1.0


class TestCandidateTemperature:
    def test_default_is_none(self):
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
        )
        assert srlm.candidate_temperature is None

    def test_accepts_temperature(self):
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            candidate_temperature=0.7,
        )
        assert srlm.candidate_temperature == 0.7

    def test_temperature_injected_during_multi_trajectory(self):
        """When candidate_temperature is set, backend_kwargs should get temperature
        injected into default_extra_body during multi-trajectory runs, then restored."""
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            n_candidates=2,
            candidate_temperature=0.8,
        )
        original_extra = dict(srlm.backend_kwargs.get("default_extra_body", {}))

        captured_temps = []
        original_completion = RLM.completion

        def mock_completion(self_inner, prompt, root_prompt=None):
            extra = self_inner.backend_kwargs.get("default_extra_body", {})
            captured_temps.append(extra.get("temperature"))
            from lm_repl.core.types import RLMChatCompletion, UsageSummary
            return RLMChatCompletion(
                root_model="test", prompt=prompt, response="42",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=1.0,
            )

        import unittest.mock
        with unittest.mock.patch.object(RLM, 'completion', mock_completion):
            srlm.completion("test prompt")

        assert all(t == 0.8 for t in captured_temps), f"Expected 0.8, got {captured_temps}"
        assert srlm.backend_kwargs.get("default_extra_body", {}) == original_extra

    def test_no_temperature_injection_when_none(self):
        """When candidate_temperature is None, no temperature is injected."""
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            n_candidates=2,
        )

        captured_temps = []
        def mock_completion(self_inner, prompt, root_prompt=None):
            extra = self_inner.backend_kwargs.get("default_extra_body", {})
            captured_temps.append(extra.get("temperature"))
            from lm_repl.core.types import RLMChatCompletion, UsageSummary
            return RLMChatCompletion(
                root_model="test", prompt=prompt, response="42",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=1.0,
            )

        import unittest.mock
        with unittest.mock.patch.object(RLM, 'completion', mock_completion):
            srlm.completion("test prompt")

        assert all(t is None for t in captured_temps)


# --- Verbalized confidence & joint scoring tests ---

import math
from lm_repl.core.srlm import _parse_confidence_scores, _compute_vc_score


class TestParseConfidenceScores:
    def test_single_score(self):
        text = 'I found the answer. {"confidence": 85}'
        assert _parse_confidence_scores(text) == [85.0]

    def test_multiple_scores(self):
        text = '{"confidence": 90}\nsome code\n{"confidence": 70}'
        assert _parse_confidence_scores(text) == [90.0, 70.0]

    def test_no_scores(self):
        assert _parse_confidence_scores("just regular text") == []

    def test_handles_whitespace_variants(self):
        text = '{"confidence" : 75}'
        assert _parse_confidence_scores(text) == [75.0]

    def test_handles_integer_and_float(self):
        text = '{"confidence": 80}\n{"confidence": 92.5}'
        scores = _parse_confidence_scores(text)
        assert scores == [80.0, 92.5]

    def test_clamps_to_range(self):
        text = '{"confidence": 0}\n{"confidence": 100}\n{"confidence": 150}'
        scores = _parse_confidence_scores(text)
        assert scores[0] == 0.0
        assert scores[1] == 100.0
        assert scores[2] == 100.0  # clamped


class TestComputeVCScore:
    def test_perfect_confidence(self):
        text = '{"confidence": 100}\n{"confidence": 100}'
        assert _compute_vc_score(text) == 0.0  # log(1) + log(1) = 0

    def test_partial_confidence(self):
        text = '{"confidence": 50}'
        score = _compute_vc_score(text)
        assert score < 0  # log(0.5) is negative
        assert abs(score - math.log(0.5)) < 1e-6

    def test_no_scores_returns_neg_inf(self):
        assert _compute_vc_score("no confidence here") == float('-inf')

    def test_zero_confidence_clamps(self):
        text = '{"confidence": 0}'
        score = _compute_vc_score(text)
        assert score == float('-inf')  # log(0) is -inf, use floor


class TestSelectBestWithConfidence:
    def test_confidence_mode_prefers_high_vc(self):
        """High VC score (closer to 0) wins over low VC score."""
        c1 = _make_completion("42", 2.0)
        c1.metadata = {"trajectory_text": '{"confidence": 95}\n{"confidence": 90}'}
        c2 = _make_completion("42", 2.0)
        c2.metadata = {"trajectory_text": '{"confidence": 40}\n{"confidence": 30}'}

        result = _select_best([c1, c2], use_confidence=True)
        assert result is c1

    def test_confidence_mode_joint_score(self):
        """Joint score VC*Len: high confidence + short trace beats low confidence + short trace."""
        c1 = _make_completion("42", 1.0)
        c1.metadata = {"trajectory_text": '{"confidence": 95}'}
        c2 = _make_completion("42", 1.0)
        c2.metadata = {"trajectory_text": '{"confidence": 50}'}

        result = _select_best([c1, c2], use_confidence=True)
        assert result is c1

    def test_confidence_off_ignores_metadata(self):
        """Without confidence mode, selection uses execution_time only."""
        c1 = _make_completion("42", 2.0)
        c1.metadata = {"trajectory_text": '{"confidence": 95}'}
        c2 = _make_completion("42", 1.0)
        c2.metadata = {"trajectory_text": '{"confidence": 30}'}

        result = _select_best([c1, c2], use_confidence=False)
        assert result is c2  # shorter time wins

    def test_falls_back_to_time_when_no_confidence_data(self):
        """If metadata has no trajectory_text, fall back to time-based selection."""
        c1 = _make_completion("42", 2.0)
        c2 = _make_completion("42", 1.0)

        result = _select_best([c1, c2], use_confidence=True)
        assert result is c2
