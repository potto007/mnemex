---
status: "accepted"
date: "2026-06-19"
deciders: "potto"
---

# Reasoning-loop repeat-guard + escalation (4th runaway-generation guard)

## Context and Problem Statement

ADR-0003 added three runaway-generation guards (hard subcall circuit-breaker,
time-based soft-budget, KV-contention retry). None of them stops a **single**
completion that spins inside its own reasoning channel: the gemma fine-tune can
emit repeated tokens/phrases (no content, no tool call) until `max_tokens` or the
run deadline. In the rlm-trainer 2026-06-18 eval this wasted 245-407s on ~6% of
reps. The subcall cap counts CALLS (a spin is one call); the soft-budget needs the
model to FINISH a generation and then cooperate (a spinning generation never
finishes); contention-retry is for KV 500s. How do we bound a single spinning
generation, and an ask that keeps re-entering the spin?

## Considered Options

- **Client stream-abort on a no-progress signal + orchestrator escalation** (a 4th
  guard, two layers).
- Server-side sampler only (llama.cpp DRY / repetition penalty in `rlm-models.ini`).
- Lower `max_tokens` / `max_decode_tokens` (ADR-0002) further.

## Decision Outcome

Chosen option: **a 4th guard, in two layers**, both default-off:

1. **Client repeat-guard (`repeat_guard_threshold`).** During a streamed completion,
   while still reasoning-only (no answer content yet), re-check every ~300 chars
   whether the trailing reasoning window's word-4gram repeat rate crosses the
   threshold; if so, abort the stream (server frees the slot on disconnect) and
   return the tagged reasoning tail early. Evidence-backed: legit reasoning measured
   repeat rate <=0.006, every degeneration >=0.42, so ~0.35 separates them widely.
2. **Orchestrator escalation (`repeat_guard_abort_limit`).** The client counts
   per-run aborts; once the cumulative count for one completion reaches the limit,
   the RLM injects the soft-budget wrap-up message (answer-from-context or refuse) -
   the count-based sibling of the time-based soft-budget. Forwarded to SRLM
   candidates (a candidate is a root clone), not to recursion children, matching the
   soft-budget's scoping.

Rejected the server-side sampler as the sole fix: it is global (penalizes
legitimate repetition such as citation lists), invisible to the harness, and cannot
drive the orchestrator-level escalate-to-refuse. The client signal is per-completion
and steers the orchestrator.

### Consequences

- Good: live A/B (2026-06-19, v13-ep1, focused degeneration-prone set) cut total
  wall-clock on the failure tail -76% (7476s -> 1782s; mean 227->81s, max 537->259s)
  with fabrication held at 0 and no truncation of legitimate answers (the guard never
  fires while answer content is flowing).
- Good: the escalation converts a persistent looper into a clean answer/refusal
  rather than a fast-but-empty degeneration.
- Bad/limit: the client guard makes degeneration CHEAP but does not lower its rate
  (30%->32% in the run); the escalation limit is a blunt knob that, set too low,
  could force a refusal on an ask that would have recovered - hence default-off and
  A/B before enabling.

## More Information

- Extends lm-repl ADR-0003 (the three prior guards); companion ADR-0002 (decode
  ceiling). rlm-trainer #6 (scorer precision + reasoning-loop degeneration).
- Config: `repeat_guard_threshold`, `repeat_guard_abort_limit` on `RLM`/`SRLM`;
  librarian envs `KB_REPEAT_GUARD_THRESHOLD`, `KB_REPEAT_GUARD_ABORT_LIMIT`.
- Live validation writeup: rlm-trainer `docs/eval-artifacts/repeat-guard-live-2026-06-19.md`.
- Commits: `36db22d` (client guard), `767d983` (escalation).
