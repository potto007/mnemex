"""High-level Harness API: owns orchestration strategy, runtime detection, and
memory composition so clients do not hand-assemble SRLM. See
docs/superpowers/specs/2026-06-21-prehend-harness-api-design.md and ADR-0008."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import urllib.request

from prehend.core.srlm import SRLM

_log = logging.getLogger("prehend.harness")


@dataclass(frozen=True)
class Defaults:
    """Vetted Tier-A strategy/reliability defaults the Harness applies to SRLM."""
    max_output_chars: int = 500
    max_iterations: int = 10
    max_depth: int = 2
    max_errors: int = 3
    max_retries: int = 0
    stream: bool = False
    subcall_enable_thinking: bool = False
    max_concurrent_subcalls: int = 4
    soft_timeout_pct: float | None = None


VETTED = Defaults()


@dataclass(frozen=True)
class Runtime:
    """Resolved server facts (Tier B). slots drives map-reduce fan-out."""
    slots: int
    ctx: int | None = None


@dataclass(frozen=True)
class MemoryConfig:
    """ADR-0005 memory wiring, mapped to build_memory_harness_from_config."""
    bank_dir: str
    embed_model: str
    reflect_model: str
    embed_url: str | None = None
    embed_api_key: str | None = None
    k_max: int | None = None
    min_cosine: float | None = None


def _default_probe(base_url: str, api_key: str) -> Runtime | None:
    """Best-effort llama-server probe. Returns None if facts are unavailable."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(f"{root}/props", timeout=5) as r:
            props = json.loads(r.read())
        gen = props.get("default_generation_settings", {}) or {}
        ctx = gen.get("n_ctx") or None
        slots = props.get("total_slots") or gen.get("n_parallel") or 0
        if not slots or slots <= 0:
            return None
        return Runtime(slots=int(slots), ctx=int(ctx) if ctx else None)
    except Exception:
        return None


def detect_runtime(
    base_url: str,
    *,
    api_key: str = "not-needed",
    probe: Callable[[str, str], Runtime | None] | None = None,
) -> Runtime | None:
    """Hybrid Tier-B detection. None means 'ambiguous, caller should fall back'."""
    p = probe or _default_probe
    try:
        rt = p(base_url, api_key)
    except Exception:
        return None
    if rt is None or rt.slots <= 0:
        return None
    return rt


class Harness:
    """High-level entry point that assembles SRLM from vetted defaults + resolved runtime."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str = "not-needed",
        timeout: float | None = None,
        runtime: "Runtime | str" = "auto",
        defaults: Defaults | None = None,
        system_addendum: str | None = None,
        subcall_verifier=None,
        answer_verifier=None,
        max_answer_retries: int | None = None,
        custom_tools: dict | None = None,
        observability: Callable[[object], None] | None = None,
        logger=None,
        memory=None,            # behavior added in Task 4
    ):
        d = defaults or VETTED
        self.runtime = self._resolve_runtime(runtime, base_url, api_key, d)

        backend_kwargs = {
            "model_name": model, "base_url": base_url, "api_key": api_key,
            "max_retries": d.max_retries, "stream": d.stream,
        }
        subcall_kwargs = dict(backend_kwargs)
        subcall_kwargs["default_extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": d.subcall_enable_thinking}
        }
        srlm_kwargs = dict(
            backend="openai",
            backend_kwargs=backend_kwargs,
            other_backends=["openai"],
            other_backend_kwargs=[subcall_kwargs],
            environment="local",
            environment_kwargs={"max_output_chars": d.max_output_chars},
            max_iterations=d.max_iterations,
            max_depth=d.max_depth,
            max_errors=d.max_errors,
            max_timeout=timeout,
            max_concurrent_subcalls=self.runtime.slots,
            soft_timeout_pct=d.soft_timeout_pct,
            logger=logger,
            verbose=False,
        )
        if system_addendum is not None:
            srlm_kwargs["custom_system_prompt"] = system_addendum
        if subcall_verifier is not None:
            srlm_kwargs["subcall_verifier"] = subcall_verifier
        if answer_verifier is not None:
            srlm_kwargs["answer_verifier"] = answer_verifier
        if max_answer_retries is not None:
            srlm_kwargs["max_answer_retries"] = max_answer_retries
        if custom_tools is not None:
            srlm_kwargs["custom_tools"] = custom_tools

        self.srlm = SRLM(**srlm_kwargs)
        if observability is not None:
            observability(self.srlm)
        self.solver = self.srlm     # Task 4 may wrap this

    def _resolve_runtime(self, runtime, base_url, api_key, d: Defaults) -> Runtime:
        if isinstance(runtime, Runtime):
            return runtime
        detected = detect_runtime(base_url, api_key=api_key)
        if detected is not None:
            return detected
        _log.info("harness: runtime probe ambiguous; falling back to slots=%d",
                  d.max_concurrent_subcalls)
        return Runtime(slots=d.max_concurrent_subcalls)

    def completion(self, context: str, query: str) -> str:
        return self.solver.completion(context, query)
