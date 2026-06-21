---
status: "accepted"
date: "2026-06-20"
deciders: "potto"
---

# Full rename `lm-repl` -> `mnemex` (package, PyPI, repo)

## Context and Problem Statement

ADR-0005 added an experience-memory layer and named that capability **mnemex**
("from mnemonic; a harness that learns"). At that point only the memory
subsystem carried the name - the Python package was still `lm_repl`, PyPI
distribution `lm-repl`, the repo `potto007/lm-repl`, and the README led with
"LM-REPL: Recursive Language Models". The `lm-repl` label names the *mechanism*
(an LM driving a REPL), but the project's defining capability is now the
learning/memory axis, not the REPL trick. Should the whole project adopt the
`mnemex` name, and if so, how do we handle the downstream repos that import it?

## Decision Drivers

- The name should describe what the project *is* (a harness that learns), not
  one implementation mechanism (LM-in-REPL). The REPL remains a mechanism, not
  the brand.
- A half-rename (subsystem named mnemex, package named lm_repl) is a standing
  source of confusion in imports, docs, and conversation.
- Two repos import the package (`~/src/knowledge-base` kb-librarian,
  `~/src/rlm-trainer`); a rename must not silently break them.

## Considered Options

- **Full rename now, hard cut, no compat shim** - rename package/imports, PyPI
  name, repo, README; migrate both downstream consumers in the same effort.
- Rename the distribution/repo but keep the `lm_repl` import path.
- Ship a `lm_repl` -> `mnemex` compatibility shim and migrate consumers later.
- Keep `lm-repl`; use `mnemex` only for the memory subsystem (status quo).

## Decision Outcome

Chosen option: **full rename, hard cut**. `lm_repl` -> `mnemex` across the
package and import paths; `pyproject` `name` `lm-repl` -> `mnemex`; README
retitled and reframed around "a harness that learns"; repo `potto007/lm-repl` ->
`potto007/mnemex`; new PyPI `mnemex`. No compatibility shim - the two downstream
importers (`knowledge-base`, `rlm-trainer`) are migrated in the same effort.
The behavioral contracts those consumers depend on (custom_tools dict form, root
`TimeoutExceededError`, OpenAI-SDK error surface) are unchanged by a rename; only
the module path moves.

Deliberately *not* renamed, because they name the mechanism or are historical:
the `lm-repl` Makefile target (runs the `lm_in_repl` example), the `REPL`
concept in prose, on-disk paths still literally at `/home/potto/src/lm-repl`, and
accepted ADRs 0001-0005 (immutable; they keep the name they were accepted under).

### Consequences

- Good, because the project's name now matches its defining capability and the
  import path stops contradicting the brand.
- Good, because a hard cut leaves no dual-name ambiguity or shim to maintain.
- Bad, because it is a breaking change for any external consumer of the old
  `lm-repl` PyPI package / `lm_repl` import path (mitigated for the known
  in-tree consumers by migrating them together; GitHub redirects the old repo
  URL).
- Bad, because old ADRs and historical design docs now reference a name the
  project no longer uses (accepted as the cost of ADR immutability).

## More Information

- Supersedes the naming scope of ADR-0005 (which named only the subsystem).
- Downstream consumers and the contracts they pin: see auto-memory
  `project_lm-repl-downstream-consumers`.
