"""mnemex retrieval: rank bank entries against a live query embedding.

Single-stage cosine ranking (matching FinAcumen's shipped ``fm/retrieve.py``,
not the unimplemented 3-stage tagger/rerank in its docs), with the key change
that the query is embedded live through an injected backend rather than resolved
from a pre-baked, id-keyed matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lm_repl.memory.bank import Bank
from lm_repl.memory.embed import EmbeddingBackend, cosine

DEFAULT_K_MAX = 3
DEFAULT_MIN_COSINE = 0.65


@dataclass
class RetrievalResult:
    """Outcome of a retrieval call.

    ``mode`` is ``"with-memory"`` when at least one entry cleared the threshold,
    else ``"no-memory"`` (the agent then runs with no experience injected).
    """

    mode: str
    entries: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


def _no_memory() -> RetrievalResult:
    return RetrievalResult(mode="no-memory", entries=[], scores=[])


def retrieve(
    query: str,
    bank: Bank,
    backend: EmbeddingBackend,
    *,
    k_max: int = DEFAULT_K_MAX,
    min_cosine: float = DEFAULT_MIN_COSINE,
) -> RetrievalResult:
    """Return up to ``k_max`` bank entries most similar to ``query``.

    Entries scoring below ``min_cosine`` are dropped; duplicates sharing an
    ``id`` are collapsed to their highest-scoring occurrence.
    """
    entries = bank.load()
    if not entries:
        return _no_memory()

    query_vec = backend.embed(query)

    scored: list[tuple[dict, float]] = []
    for entry in entries:
        emb = entry.get("embedding")
        if not emb:
            continue
        score = cosine(query_vec, emb)
        if score >= min_cosine:
            scored.append((entry, score))

    if not scored:
        return _no_memory()

    scored.sort(key=lambda pair: pair[1], reverse=True)

    selected: list[tuple[dict, float]] = []
    seen_ids: set[str] = set()
    for entry, score in scored:
        eid = entry.get("id")
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        selected.append((entry, score))
        if len(selected) >= k_max:
            break

    return RetrievalResult(
        mode="with-memory",
        entries=[e for e, _ in selected],
        scores=[s for _, s in selected],
    )
