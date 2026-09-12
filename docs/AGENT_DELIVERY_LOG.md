# Conversational agent delivery log

## C1 contract increment — 2026-09-12

Added versioned partial drafts, per-field provenance and confirmation, atomic
typed patches, target-dependent invalidation, and a portable full-dialogue state.
Policy suggestions remain unconfirmed. Persistence wiring follows in C3.

Validation: full regression suite 67 passed; Ruff check/format and diff checks.
No live provider calls. This is a contract increment, not C1-C6 final acceptance.

## C2 bounded tool increment — 2026-09-12

Added catalog-grounded counts, validation, canonical report generation, work/date
estimates, and first-24-month availability-aware evidence. Raw vectors require
explicit confirmation; synthetic generation provenance is not provider context.
Future-value poisoning and report invalidation tests pass. Full suite: 70 passed;
Ruff check/format and diff checks pass. No live provider calls.

## C3 multi-turn agent increment — 2026-09-12

Added the Gemini structured action/proposal service, a versioned packaged prompt,
explicit action branches in LangGraph, durable conversations, turn deduplication,
dataset binding, limited offline recovery, and permission checks outside the LLM.
New dialogue checkpoints are separate from legacy single-experiment checkpoints.
Full suite: 74 passed; Ruff checks pass. Gemini integration verified with mocks;
no configured local service credentials. Execution and UI wiring follow in C4-C5.

## C4-C5 integrated execution slice — 2026-09-12

Connected the chat-only UI, canonical specification downloads and read-only
`manto spec --conversation ID` terminal projection to approval-bound execution.
The preliminary engine removes holdout values/revisions before feature construction;
full mode preserves the existing numerical path. Added immutable parent links,
stored comparisons on compatible development outcomes, repeated-holdout warnings,
and explicit import of existing results into new conversations.

These two dependent slices were integrated together so the new chat never needs
the removed form to execute. Full regression: 76 passed in 19.42 seconds; Ruff
check/format pass. Browser: coherent chat layout and initial turn verified, no
target selector. End-to-end Streamlit tests cover execution, scatter controls,
saved-result reopening. Tests cover stale report IDs, replay, terminal JSON parity,
preliminary future-value poisoning, full branches, and no refit on result reads.
No live Gemini/Langfuse verification; C6 hardening remains in progress.

## C6 local acceptance — 2026-09-12

Local implementation and recovery checks completed. Final full regression:
**81 passed in 19.98 seconds**. Ruff lint/format and diff checks pass; changed
documentation links resolve. `uv build --wheel` succeeds and includes the
versioned agent prompt. No dependencies were changed.

Additional checks cover Polish turns, pending proposal confirmation with preserved
rationale, bounded provider failures, rejection of LLM-requested mutations on
read-only questions, stale approvals, and numerical failure/retry without secrets.
Holdout exposure checks now consider overlapping outcome months, conservatively
across snapshots of the same target; C_EXPOSURE explicitly overrides qualification.

Browser acceptance: the live local UI has no target/configuration form; a Polish
sales question receives a Polish follow-up, then "Ile mamy kandydatów?" returns
10 available and 0 selected without losing the earlier turn or conversation ID.
The existing Streamlit integration test covers a full run, axis changes, and reopen.

### Remaining acceptance boundary

Live Gemini and Langfuse verification is **pending**, not passed: local environment
inspection found neither a Gemini key/model nor Langfuse credentials. No old
chat-pasted credential was reused. C6 must not be described as fully accepted
until a bounded live dialogue/trace check succeeds with fresh local configuration.

Documented bounded scope: offline suggestions are catalog-order, evidence uses a
fixed first-24-month prefix, exact `run`/`uruchom` is required, cross-mode fits are
recomputed, and legacy checkpoint-only drafts are not automatically migrated.
Saved legacy results can be explicitly imported. External viewing of data and
renamed targets cannot be inferred by the exposure tracker. Full analysis means
the existing linear engine, not later ECM/seasonality/monitoring milestones.

All delivery commits are local; no remote push was performed for C1-C6.

## Contextual explanation and recovery graph — 2026-09-12

Fixed the reported `co robi initial_train?` generic offline reply. Added explicit
`explain_setting`, `what_if`, `clarify`, `help`, and `resume_setup` LangGraph nodes,
a versioned bilingual parameter/metric contract, schema-derived numeric bounds,
and checkpointed detour context. Known explanations work without a provider call.
Hypothetical numeric changes use the engine's real workload estimator on a copy;
they never train, modify configuration, approve defaults or predict model quality.
Ambiguous confirmations, invalid values, misspellings and multi-field edits fail
safely; typed JSON edits and stored-result inspection retain their existing paths.

Validation: **98 tests passed in 22.31 seconds**, including the Streamlit screenshot
regression, restart/follow-up continuity, immutable draft/report checks, no-provider
and no-fit assertions, and existing numerical/execution tests. Ruff lint/format and
diff checks pass. Browser verification repeated the exact Polish question in the
existing local conversation: the reply explains 60 monthly training pairs, the
expanding window, validation trade-off, current undiscussed value and lower bound
24. The previous transcript and configuration were preserved. No live provider
validation is claimed; Gemini/Langfuse acceptance remains a separate gate.

## English-only response policy — 2026-09-12

Unified new conversational output in English: removed bilingual response branches
and duplicate Polish parameter definitions, fixed the reply language to English,
and updated the Gemini system prompt and routing context. Polish commands remain
accepted. Legacy `language=pl` checkpoints migrate on load without changing stored
messages, draft revisions, report approvals or saved artifacts. Explanation events
now identify `local_parameter_contract_v2_en`.

Validation: **99 tests passed in 25.01 seconds**; Ruff lint/format and diff checks
pass. Coverage includes legacy checkpoint restart, Polish input with English output,
and the mocked Gemini English response contract. The running browser conversation
also answered `co robi initial_train?` in English after rerun. Historical messages
were preserved. Live Gemini language compliance was not tested. Local commit only.

## Target discovery and conversational setup recovery — 2026-09-12

The reported conversation exposed three routing failures: ordinary feature-selection
phrasing was unrecognized, target requests were confused with predictor proposals,
and near-miss analysis commands ended in generic recovery text. Added `targets`,
`propose_target`, `prepare_analysis`, `recover`, and `resume_candidate_request`
graph nodes. A target proposal requires confirmation; no-target feature requests
persist their requested count and resume after explicit target selection. Common
selection verbs and number words are supported without changing `model_size`.
Target suggestions use bounded Gemini metadata or a labeled offline catalog policy.
Proposal replies show readable selections rather than only a raw settings object.

`performe analys` and similar bounded variants now show the missing setup or review
report. They do not approve defaults or authorize execution. Unknown input gets a
next step based on the draft; negations and ambiguous counts do not trigger changes.
The exact run/report approval boundary remains intact. Offline selection is still
catalog-order, not evidence of relevance or predictive usefulness.

Validation: **121 tests passed in 32.11 seconds**; Ruff lint/format and diff checks
pass. New tests cover the exact screenshot sequence in dialogue and Streamlit,
restart/resumption, target vs. feature semantics, word counts, ready-report no-run
guards, and a mocked Gemini target proposal. The running browser initially served
old code while awaiting Rerun; after loading the update, the exact feature request
correctly remembered three candidates and displayed available target options.
The temporary browser tab was closed and the user's existing conversation rerun
without modifying its setup. No live Gemini acceptance or remote push is claimed.
