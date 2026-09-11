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
