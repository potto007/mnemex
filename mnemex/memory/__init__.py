"""mnemex: self-evolving experience memory for mnemex.

A domain-agnostic port of FinAcumen's FM subsystem. Wraps a context-offloading
solver (an :class:`~mnemex.SRLM`) so it accumulates and reuses verified,
polarity-tagged experience across tasks: retrieve -> inject -> solve -> collect.

Quick start::

    from mnemex import SRLM
    from mnemex.memory import build_memory_harness_from_config

    srlm = SRLM(backend="openai", backend_kwargs={...})
    harness = build_memory_harness_from_config(
        srlm, "memory_bank",
        base_url="http://localhost:8080/v1",
        embed_model="nv-embed-v2", reflect_model="my-judge",
    )
    result = harness.answer(context=long_context, question="...")
"""
from mnemex.memory.bank import Bank
from mnemex.memory.distill import TraceDistiller
from mnemex.memory.embed import EmbeddingBackend, HashingEmbeddingBackend, cosine
from mnemex.memory.embed_openai import OpenAIEmbeddingBackend
from mnemex.memory.factory import (
    build_memory_harness,
    build_memory_harness_from_config,
)
from mnemex.memory.harness import Distiller, MemoryHarness, Solver
from mnemex.memory.inject import render_memory_block
from mnemex.memory.pruning_rules import is_anti_give_up
from mnemex.memory.reflect import OpenAIReflectFn
from mnemex.memory.retrieve import RetrievalResult, retrieve
from mnemex.memory.tagger import NullTagger, Tagger

__all__ = [
    "Bank",
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "OpenAIEmbeddingBackend",
    "OpenAIReflectFn",
    "cosine",
    "MemoryHarness",
    "Distiller",
    "Solver",
    "Tagger",
    "NullTagger",
    "TraceDistiller",
    "build_memory_harness",
    "build_memory_harness_from_config",
    "is_anti_give_up",
    "render_memory_block",
    "RetrievalResult",
    "retrieve",
]
