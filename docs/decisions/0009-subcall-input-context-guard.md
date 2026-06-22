---
status: "accepted"
date: "2026-06-22"
deciders: "potto"
---

# Sub-call input-size context guard (reject-with-hint on oversized llm_query / rlm_query)

## Context and Problem Statement

prehend is context-by-reference: a large context is offloaded into a `context` REPL
variable and the model writes programs that slice/search/recursively-query it via
`llm_query` / `rlm_query`, rather than attending over the whole thing (README; RLM/SRLM
papers, N >> L). On large plain-multihop tasks (~317,768-char / ~150K-token contexts) the
v13 model instead inlined the **entire** context into a single
`llm_query(f"...Context:\n{context}")`, producing a request of ~150,261 tokens against a
server whose window is 98,304. The server returns `400 exceeds the available context size`,
the openai client recognizes the unfittable request but only retries once and propagates,
the solve spins, and the 660s hard-kill fires. The cold (memory-off) path hits it too, so it
is a base-harness defect, not a memory/observability artifact.

The documented guard stack bounds **output** (ADR-0002 decode ceiling) and **call count**
(ADR-0003 subcall circuit-breaker), and the strategy verifier (prehend ADR / spec) rejects
whole-**task** delegation. Nothing governs sub-call **input size**, and the strategy verifier
explicitly **exempts** `llm_query` (it is output-token-capped) - which is exactly the surface
that overflowed. The orchestrator system prompt actively encouraged the anti-pattern
("the sub-LLM can handle around 500K chars ... don't be afraid to put a lot of context into
them"), contradicting the recursive-rlm-refactor spec ("`llm_query` for SHORT text only;
NEVER pass large chunks; large chunks -> `rlm_query`; chunk to ~50000 chars"). How do we stop
a sub-call from exceeding the sub-model's real context window?

## Decision Drivers

- The RLM premise: no single sub-LLM call may exceed the sub-model's token window; decompose
  first. A 150K-token call against a 98K window defeats the whole design.
- The probe is router-fragile: `/props` returns `n_ctx=0` in router mode, so `Runtime.ctx`
  is often unavailable; the limit must be supplyable explicitly and degrade safely.
- Papers favor auto-chunk or an **actionable** reject-with-hint over a bare reject that
  derails trajectories; chunking is the model's job, taught via the prompt.
- harness-api spec: Tier-B config is explicit constructor args, "no env hack".

## Considered Options

- **Reject-with-actionable-hint at the sub-call seam** (chosen): an arithmetic guard that, when
  a sub-call's input exceeds the resolved limit (minus margin), returns an instructive string
  telling the model to chunk and map-reduce via `rlm_query_batched`, reusing the verifier's
  reject -> error-string -> orchestrator-adapt channel.
- Auto-chunk the oversized payload transparently inside the harness.
- Truncate the input to fit.
- Prompt-only fix (correct the "500K chars" claim) with no guard.

## Decision Outcome

Chosen option: **reject-with-actionable-hint**, plus the prompt realignment and a correct,
configurable limit. Rationale: it realigns the implementation with the documented by-reference
design without taking decomposition control away from the model (which is trained/prompted to
chunk), reuses the existing rejection channel, and is deterministic (unlike the LM verifier it
does not fail open). Auto-chunk was rejected for v1 as too invasive (it changes solve
semantics and obscures the model's strategy); truncation was rejected as silently lossy; a
prompt-only fix leaves no safety net when the model overshoots anyway.

Concretely:

1. **Limit resolution.** `subcall_context_limit` is a constructor param + `Defaults` field on
   the high-level Harness; effective limit = first non-None of (explicit param, probed
   `Runtime.ctx`, `get_context_limit(model)`). `get_context_limit` gains gemma entries (it was
   silently returning the 128000 default), and `count_tokens` is made conservative for
   non-tiktoken models so the guard does not under-estimate gemma. Threaded into the prompt
   builder, the LocalREPL (`llm_query` seam), and `RLM._subcall` (`rlm_query` seam).
2. **Guard at both seams.** `local_repl._llm_query` / `_llm_query_batched` and `rlm._subcall`
   call a pure `oversize_rejection(prompt, limit, model)` before sending. It intentionally
   **breaks the verifier's `llm_query` exemption** because the live bug is an oversized
   `llm_query`. The hint names the limit and the offending size and instructs chunk-and-
   map-reduce.
3. **Prompt realignment.** The "~500K chars / don't be afraid" guidance is replaced by
   refactor-spec-aligned text parameterized by the resolved limit (`llm_query` short-only;
   large context -> chunk to ~K chars -> `rlm_query_batched`).
4. **No env var in prehend core.** Per the harness-api Tier-B principle, the limit is an
   explicit arg. Operational env overrides (e.g. `PREHEND_SUBCALL_CONTEXT_LIMIT`) live only in
   the rlm-trainer benchmark driver, which passes the value as the constructor param.

### Consequences

- Good, because a sub-call can no longer silently overflow the sub-model window; the model
  gets an actionable correction and recovers within the same solve.
- Good, because it realigns three audited deviations (prompt, limit, missing guard) with the
  documented design and the RLM/SRLM papers.
- Good, because it is deterministic and reuses the established rejection channel.
- Bad, because the margin is heuristic (covers prompt envelope + tokenizer skew); too tight
  risks residual overflow, too loose over-chunks. Tuned at ~15% with conservative token counts.
- Bad, because experiences distilled before this fix (e.g. bank entry `exp_840d65c8`, which
  advises single-shot whole-context `llm_query`) encode the anti-pattern and violate ADR-0005
  "learn only from correct solves"; the contaminated bank must be regenerated, not reused.

## More Information

- Companions: ADR-0002 (decode ceiling, output axis), ADR-0003 (subcall caps, count axis),
  ADR-0008 (Harness / Runtime.ctx). This is the first guard on the **input** axis.
- Specs: docs/superpowers/specs/2026-06-11-strategy-verifier-design.md (reject channel +
  llm_query exemption), rlm-trainer docs/superpowers/specs/2026-05-27-recursive-rlm-refactor-design.md
  (chunk-to-rlm_query), docs/superpowers/specs/2026-06-21-prehend-harness-api-design.md (Tier-B
  no-env, Runtime.ctx deferred-YAGNI seam). Plan:
  docs/superpowers/plans/2026-06-22-subcall-context-limit-coherence.md.
