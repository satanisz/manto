# First Demo: Run, Inspect, and Verify

Implemented scope: sprint 6 linear analytical demonstration, 2026-09-08.
The broader [executive plan](EXECUTIVE_PLAN.md) remains the product roadmap.

## Start locally

```powershell
uv sync --locked --extra dev
uv run manto ui
```

Open [Manto](http://127.0.0.1:8501). The server binds to this computer only.
Alternatively, the existing virtual environment can run the app with
`.venv\Scripts\python.exe -m manto ui`.

1. Keep **Synthetic sales demo** selected.
2. Click **Try: forecast sales with 3 variables, including inflation**.
3. Review the target, all ten candidates, three-variable model size, inflation pin,
   and monthly lag zero. The catalog names deliberately say "Fictional".
4. Click **Run model comparison**. The requested 36 combinations are fully evaluated.
5. Inspect original-scale forecasts, the baseline, Pareto alternatives, metrics,
   transformations and coefficients. Change the displayed model without refitting it.
6. Open **Decision trail** for rules and evidence, the executable conversation graph,
   and the active policy. Open **Data & sources** for the catalog and CSV template.
7. Download JSON/HTML or reopen the run under **Saved analyses**. Reopening restores
   both the request and its exact input snapshot. A modified comparison creates a
   new experiment; it does not replace the saved result.

The example intentionally contains a recoverable relation between next-month sales
changes and inflation, demand and marketing. Recovering it validates software
behavior; it is not evidence about a real company or economy.

## A target in natural units

Users select sales, margin, an index, or any other supported numeric monthly `Y`
in its original units. They do not need to supply an already stationary target.
The demo evaluates ADF evidence on the initial training prefix and freezes an
identity or first-difference recipe. An explicit positive-only log-difference
recipe is also available. Automatic feature recipes follow the same bounded rule.

Stationarity is not guaranteed by a transform. Inconclusive post-transform evidence
is flagged and prevents qualified champion status, while exploratory forecasts
remain inspectable. KPSS, integration-order/cointegration routing and ECM belong
to subsequent sprints. The first implementation does not blindly repeat differencing.

Predicted differences are added to the known origin level; predicted log differences
are exponentiated against that anchor and labeled conditional medians. Comparison
errors are calculated on original target units. A log is never applied to a
nonpositive target, and normalizing/scaling is not presented as stationarity treatment.

The current forecast origin is calendar month-end. `Y(t)` must be available then
for the last-value baseline and inverse anchor. Delayed explanatory releases can
be handled by explicitly permitted lags. A delayed `Y(t)` blocks that specification;
there is no hidden nowcast or forward fill. Month labels in charts refer to outcomes.

## Data and revisions

Upload the documented [long-form CSV](CSV_FORMAT.md). Required columns are
`series_id`, `period`, `available_at`, and `value`. Missing observations remain
missing. Revised values use separate availability timestamps. Future releases
cannot enter earlier feature rows; training labels only use values known at the
current fit origin. Outcomes are scored against the stored first release.

The NBP connector provides a read-only real-data integration:

```powershell
uv run manto fetch-nbp --code EUR --start 2024-01-01 --end 2024-12-31
```

It downloads complete months, batches daily requests, records monthly mean FX
values and saves a content-addressed snapshot. It is not a general inflation/GDP
connector and does not by itself build a multivariate dataset. A live January 2024
EUR smoke check succeeded during implementation. All automated connector tests
use fixtures, so the suite works without network access.

User CSV release dates are supplied evidence, not independent verification of
historical vintages. CSV and latest-archive runs remain exploratory by default;
they can provide forecasts without receiving a verified point-in-time champion
claim. WIG20 historical-data access and ten real macro-series IDs are still sourcing
work. The synthetic catalog is complete and usable for the first demo.

## Agent and integration behavior

The application uses a real LangGraph state machine with a persisted SQLite
interrupt: intent → user review → validation → analysis → explanation. Restarting
the application preserves the pending request. A conversation ID is bound to its
dataset; repeated resume calls do not duplicate completed analysis.

Offline mode is an explicitly disclosed bounded English/Polish catalog parser.
The form gives full control over candidates and pins. Optional Gemini uses typed
intent extraction and catalog validation; failed/malformed responses have bounded
retries and fall back to the guided parser. The final numerical explanation is
deterministic in this milestone; LLM champion prose is a later step.

Copy `.env.example` to a local `.env`, provide fresh credentials, and set the
corresponding explicit enable switch to use a provider. Gemini requires
`MANTO_ENABLE_GEMINI=true`, `GEMINI_API_KEY`, and `GEMINI_MODEL`. Langfuse requires
`MANTO_ENABLE_LANGFUSE=true`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
`LANGFUSE_BASE_URL`. No credentials are bundled.

Live Gemini/Langfuse connections were not exercised during the offline acceptance
run. Their structured-response, redaction, fallback, and outage contracts are
tested with mocks. Langfuse receives metadata observations rather than raw client
time series. Local decision evidence remains available without telemetry.

## Validation and selection

The search covers every permitted subset/lag specification or refuses the scope
before work. With ten candidates and three variables: 120 subsets; with one pin:
36; with two possible lags per selected variable: 288 pinned specifications.
Budget counts reserve all development fits plus the selected holdout and final fit.

Each development origin uses an expanding past-only training window with at least
the requested sample size and ten training pairs per regression parameter. Missing
training pairs are counted; a missing required evaluation-origin feature rejects
that candidate instead of scoring it on an easier calendar. Scaling is fold-local.

The Pareto front minimizes development MAE, maximum fold VIF and block-MAE variability.
The frozen recommendation is chosen within a 5% development-MAE shortlist using
stability, simplicity and stable ID tie-breaking. Only that specification receives
the final holdout audit. A failure does not select a runner-up on the same holdout.

Qualified status requires beating the last-value baseline on development and
holdout, supported selected-series stationarity evidence, and supported provenance.
This is a heuristic policy, not statistical proof or a guarantee of future skill.
The prototype does not enforce a global cross-experiment holdout ledger: repeated
user experimentation is explicitly exploratory and needs fresh evidence.

Basic numeric residual checks are present for Pareto models. Gapped residual
calendars disable temporal tests instead of pretending adjacent rows are adjacent
months. Full visual residual analysis, ECM, seasonality, prediction intervals,
continuous monitoring, and automatic recalibration are not delivered in sprint 6.

## Reproduction and integrity

`.manto/experiments/<id>/` contains `result.json`, a manifest, and a hashed input
snapshot. The manifest records the result hash, policy hash/version, dependency
versions, available Git revision, and whether tracked code was dirty when saved.
Opening a corrupted result/snapshot fails visibly. This detects accidental
modification; it is not a signed, adversary-resistant audit system.

The dependency lock is `uv.lock`; Python 3.12 is pinned for the demo environment.
Secrets, inputs, checkpoints and generated reports are ignored by Git. No saved
experiment is overwritten with a different result.

## Verification commands and milestone evidence

Final acceptance run: **62 tests passed**. This includes numerical leakage and
inverse-transform tests, CSV/snapshot integrity, mocked provider failures, real
SQLite restart and rapid checkpoint/result writes, plus Streamlit interaction
tests through comparison and saved-input restoration. Lint passed and source/wheel
packaging succeeded. A real browser walkthrough confirmed the 36-model example,
five Pareto alternatives, and the original-scale forecast chart.

```powershell
uv run pytest -q
uv run ruff check src tests
uv run manto demo --pin inflation
```

| Milestone | Evidence implemented | Remaining limit |
| --- | --- | --- |
| 1: data/spec contract | Natural-scale target, monthly horizon, synthetic target + ten catalog predictors, explicit release timestamps | Real WIG20/macro catalog not yet populated |
| 2: observable conversation | Actual LangGraph interrupt/restart, optional Gemini/Langfuse, local decisions | Live provider credentials not tested |
| 3: data | CSV, revisions, immutable snapshots, tested NBP connector | No automated multi-source macro join UI |
| 4: linear validation | OLS, lag grid, ADF-guided recipes, inverse forecasts, past-only validation | Delayed Y and advanced stationarity/ECM deferred |
| 5: search control | Manual/all catalog, pins, exact/up-to counts, explicit budgets | Offline language understanding is intentionally bounded |
| 6: demo | Chat/form → comparison → Pareto → provisional explanation, save/reopen/export | Later-sprint product functions remain visible in roadmap |
