"""Tests for the mnemex embedding helpers."""
from __future__ import annotations

import math

from lm_repl.memory.embed import cosine


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_is_scale_invariant():
    # Direction matters, magnitude does not.
    assert math.isclose(cosine([1.0, 1.0], [3.0, 3.0]), 1.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert math.isclose(cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)
