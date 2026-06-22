---
status: "accepted"
date: "2026-06-22"
deciders: "potto"
---

# Auto-chunk enforcement for oversized sub-calls (`context=` map-reduce)

## Context and Problem Statement

ADR-0009 added a deterministic input-size guard: an oversized sub-call prompt is not
sent; instead the harness returns an actionable reject-with-hint telling the model to
chunk and map-reduce. That eliminated the context-overflow bug (plain-multihop cold 40%
-> warm 53.3%, +13.3pp, 0 overflow) and was the right v1 call - ADR-0009 explicitly
DEFERRED auto-chunk as "too invasive".

But reject-with-hint leaves decomposition as the model's job, and the 12B orchestrator
(`gemma-4-12b-it-sft-kb-v13-sft`) does not reliably honor the advised chunk size on hard
tasks: `multihop_002` made only 2 giant chunks and timed out at 660s, and the chunk-size
tune (prehend `b150371`) was inconclusive on latency precisely because the bottleneck is
model behavior, not the advised number. A context of ~200K chars *fits* the 98,304-token
window (so ADR-0009 never rejects it) yet prefills slowly as one giant call - the residual
latency tail. How do we make decomposition reliable without depending on the model
chunking perfectly?

## Decision Drivers

- The latency tail is model-behavior; advice does not fix it. The harness must be able to
  decompose mechanically.
- Zero regression: a bare oversized `prompt` (no declared data) must still hit ADR-0009
  reject-with-hint and never overflow.
- The harness must not GUESS what is instruction vs data inside an opaque prompt string.
- The papers put the summary-agent / chunk-the-input pattern on the harness as a valid
  mechanism (MIT RLM Appendix A; SRLM s2.1/s3.5).

## Considered Options

1. **Optional `context=` data channel + harness map-reduce** (chosen). The model declares
   the large data via `context=`; when oversized the harness chunks it, maps the
   instruction over each chunk in parallel, and tree-reduces.
2. **New `map_reduce_query` primitive.** Leaves existing primitives untouched but the model
   must learn a new call; no reuse of the trained `llm_query` shape.
3. **Heuristic split of the opaque `prompt`.** No API change but the harness must guess the
   instruction/data boundary - fragile.

## Decision Outcome

Chosen: **option 1**, auto-chunk enforcement on a new optional `context=` argument,
superseding ADR-0009's "auto-chunk deferred" stance FOR THE `context=` PATH ONLY. ADR-0009's
reject-with-hint remains the behavior for a bare oversized `prompt`.

- **API**: `llm_query` / `llm_query_batched` / `rlm_query` / `rlm_query_batched` gain
  keyword-only `context=None` and `reduce=None`. The pre-existing `priority` param on the
  `llm_*` primitives is preserved. `prompt` is the map instruction; `context` is the data;
  `reduce` is the combine instruction (defaults to `prompt`).
- **Dispatch** (per single call): no `context` -> unchanged ADR-0009 behavior; `context`
  whose composed prompt fits the RECOMMENDED size `R` (`recommended_chunk_chars`) -> inline
  one send; composed prompt over `R` -> map-reduce. The threshold is `R`, not the `fits`
  ceiling, so the ~88K..250K char band (fits the window but prefills slowly) is decomposed
  instead of sent as one slow call.
- **Engine** (`prehend/utils/mapreduce.py`, pure): split data at a per-chunk budget
  (`R` minus the compose envelope); MAP the instruction over chunks via a CONTEXT-FREE
  batched send; hierarchical tree-REDUCE bounded by `max_reduce_depth=3` (the unconditional
  termination backstop), with a progress invariant (partials hard-cut to the data budget so
  `2*budget+envelope < ceiling` guarantees groups hold >= 2). Control/error/budget strings
  are filtered out of the reduce so they cannot poison the answer; the budget message stops
  further fan-out and flags `budget_exhausted`.
- **Re-entrancy**: the engine's batched send is a private context-free helper
  (`_send_batched` / `_rlm_send_batched`), never the public `*_batched` primitive, so a
  mis-sized group can never re-enter map-reduce.
- **Budget / circuit-breaker (ADR-0003)** and the **per-prompt guard (ADR-0009)** are
  inherited by delegating to the existing send path; no new accounting.
- **No chunk overlap** in v1 (documented future lever if multi-hop accuracy regresses).
- **`_subcall` (RLM) is unchanged**: the REPL composes per-chunk prompts and calls
  `subcall_fn` positionally, so the data channel lives entirely in the REPL primitives.

### Consequences

- Good: the timeout tail no longer depends on the model chunking well; a large `context=`
  fans out across server slots. Additive and backward compatible (keyword-only new args;
  no current caller passes them).
- Bad / risks: (1) **adoption** - the v13 SFT model was trained on `llm_query(selected_docs)`
  and may not emit `context=`; mitigated because the path is additive (reject-with-hint
  still prevents overflow) and adoption is a validation question. (2) **partition validity**
  - splitting can sever a multi-hop link across a chunk boundary (MIT paper caveat); v1 has
  no overlap, so validation watches accuracy, not just latency. (3) **reduce information
  loss** - tree-reduce over partial answers can drop a detail; the depth bound + truncation
  note make loss visible.
- Guard-disabled (`subcall_context_limit is None`): a `context=` is inlined and can
  overflow - the caller opted out of the guard (consistent with ADR-0009).

## More Information

- Spec (source of truth, incl. 2 adversarial reviews):
  `docs/superpowers/specs/2026-06-22-auto-chunk-enforcement-design.md`
- Supersedes the "auto-chunk deferred" stance of [ADR-0009](0009-subcall-input-context-guard.md)
  for the `context=` path; companions [ADR-0003](0003-runaway-generation-guards.md),
  [ADR-0002](0002-hard-per-generation-decode-token-ceiling.md).
- Plan/history: `docs/superpowers/plans/2026-06-22-subcall-context-limit-coherence.md`
- Papers: rlm-trainer `docs/recursive_language_models_mit_paper.md` (Appendix A; D.1),
  `docs/srlm_apple_paper.md` (s2.1, s3.5).
