"""mnemex embedding helpers and the pluggable embedding backend.

FinAcumen resolved embeddings from pre-baked ``datasets/*_emb.npy`` keyed by a
benchmark ``target_id`` (zero API calls during eval). mnemex instead embeds
arbitrary queries live through an injected :class:`EmbeddingBackend`, so the
memory layer works on open-ended inputs rather than a fixed benchmark set.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = Sequence[float]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors. Scale-invariant.

    Returns 0.0 when either vector has zero magnitude (no defined direction).
    """
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that turns text into a fixed-dimension embedding vector."""

    def embed(self, text: str) -> list[float]:
        ...
