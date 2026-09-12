# Conversational workbench

## Start and discuss

Run `uv run manto ui`, then open the local workbench. Start with your business
question. The conversation keeps its ID, transcript, settings, and results as you
reply, ask questions, or change an earlier choice. No configuration form is required.

Gemini is the conversational agent inside the executable LangGraph. Numerical
tools enforce catalog IDs, timing, feasible work, and report approval. The graph
is visible under **Decision trail**, together with C-prefixed conversational
events and the existing numerical decisions. The packaged `agent_prompt.txt`
is versioned by its hash in decision events.

## Configure Gemini and tracing locally

Set `MANTO_ENABLE_GEMINI=true`, `GEMINI_API_KEY`, and a supported `GEMINI_MODEL`
in your ignored local `.env`. Use a fresh credential, not one pasted into a chat.
The existing Google GenAI client is used for typed actions and bounded proposals.
Restart the local application after changing service configuration.

Langfuse remains independently configurable through `.env.example`. Neither
provider has been live-tested for this increment because credentials/model
configuration were absent. Mock tests verify integration contracts and failures;
they do not establish service availability or real conversational quality.

Gemini receives your messages, catalog metadata, configuration and bounded
statistical/result summaries. Numerical vectors are included only after explicitly
confirming `share_vectors=true`. Proposal evidence is restricted to the first 24
calendar months, with release-aware revisions and missingness retained. This is a
conservative training-prefix bound, not a full lag-selection statistical method.
No synthetic generator formula is supplied as evidence. Langfuse receives metadata,
not raw vectors or prompts, through the current conservative telemetry adapter.

Provider requests have a 64,000-character context bound, at most two attempts per
generation, and at most two generations for a proposal turn. Output is bounded.
Failure/budget exhaustion enters a disclosed recovery mode; it does not authorize
analysis or discard the draft. The full local transcript is retained, while the
provider gets at most 16 recent messages plus the authoritative draft.

## Offline recovery

Without a configured service, the chat clearly displays **Guided offline recovery**.
This is a small command parser, not an equivalent LLM. Open-ended explanations and
semantic ranking require Gemini. An offline shortlist is explicitly catalog-order,
not an assertion that those variables will predict the target.

Example commands:

- `Forecast sales with 3 variables, including inflation`
- `How many candidates?` / `Ile mamy kandydatów?`
- `List candidates` / `Pokaż listę`
- `Propose 5` / `Zaproponuj 5 kandydatów`, then `yes` / `tak`
- `all candidates` or `candidates: inflation demand marketing`
- `Propose lags` / `Zaproponuj lagi`, then confirm or say `lags 0 1 3`
- `settings` / `Pokaż ustawienia`
- `confirm settings` / `akceptuję ustawienia`
- `report` / `Pokaż raport`
- `run` / `uruchom` after a complete, current report
- `results` / `Jakie lagi ma rekomendowany model?`
- `full` to prepare a linked full-mode specification, then review and `run`
- `replace inflation with interest_rate`, then inspect and approve the new report
- `compare` to compare the latest result with its parent

For precise edits in recovery mode, use a typed command such as
`set {"initial_train":60,"holdout_periods":12}`. It is validated input, not
executable Python. Unknown settings and invalid values are rejected.

Explicit user choices confirm only the named settings. Default suggestions are
listed for discussion and group acceptance, never silently treated as consent.
`yes` confirms the displayed settings/proposal; this implementation deliberately
requires the separate word `run` or `uruchom` for execution. An invalid or ambiguous
instruction produces a clarification rather than guessing authorization.

## Specification and execution

Ask for a report whenever useful. Draft reports show unresolved settings and cannot
run. A ready report binds the draft revision, complete request, input fingerprint,
policy, run mode, evaluation dates, work estimate and exposure state. Proposals keep
their recorded rationale after acceptance. New settings invalidate old approval.

Download the specification as Markdown or JSON. To print the last current report
from a conversation without starting the app or running analysis:

```powershell
uv run manto spec --conversation YOUR_CONVERSATION_ID
uv run manto spec --conversation YOUR_CONVERSATION_ID --json
```

Use `--directory PATH` if the data directory is not `.manto`. The command opens
the report projection read-only. Request a fresh report after changing settings.

**Preliminary** evaluates the full approved development grid, saves its metrics
and recommendation, and excludes holdout outcomes/later revisions from feature
construction. It does not issue a champion or next forecast. **Full** uses the
existing linear engine's holdout audit and next forecast. It does not add ECM,
cointegration, seasonal families, or continuous monitoring.

Reapproving the same report reopens its immutable result rather than refitting.
A new mode or variable choice creates a linked experiment and a new report.
Preliminary-to-full currently recomputes development fits; cross-mode fit-cache
reuse is deferred, without changing either experiment's numerical contract.

## Compare, reopen, and recover

Ask `compare` after a child run, or `compare EXPERIMENT_ID` in recovery mode.
Comparison includes specification changes and development MAE recomputed from
stored predictions on common outcome months. Numerical ranking is unavailable if
the target or input fingerprint differs. Different data vintages are not silently
treated as comparable. The existing candidate scatter still has selectable axes.

Previously exposed holdout months are tracked through saved artifacts across
conversations for the same target. Overlapping full audits are labeled exploratory
and cannot qualify a champion. This conservatively treats snapshots with the same
target ID as related; it cannot identify data a user viewed outside Manto or a
renamed target. Users must disclose such exposure; it is not independent evidence.

**Saved analyses** restores the original snapshot and imports the immutable result
into a new chat. Old single-experiment checkpoints remain untouched in
`conversations.sqlite`; the new chat uses `dialogue.sqlite`. Resume the new chat
with its ID and the same input snapshot. Legacy results can be opened in the UI;
legacy checkpoint-only drafts are not automatically migrated.

Turns are serialized locally. Checkpoint replay or repeated approval reuses a
completed stored result. A failed numerical run has sanitized feedback and can
be retried against the same report. A crash before artifact completion may repeat
pure numerical work, but never intentionally create a second result ID. The UI
keeps the last saved result visible while you prepare a new draft.
