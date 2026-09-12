# manto

A conversational assistant for transparent time-series analysis, forecasting,
model comparison, and monitoring.

## Status

The linear demo now has a multi-turn chat workbench: Gemini action routing in
LangGraph, explicit setting review, approval-bound specification reports,
preliminary/full experiments, saved comparisons, and original-scale forecasts.
The interface no longer requires a target/configuration form. Without configured
Gemini credentials it uses clearly labeled, limited offline recovery commands.
Provider contracts are mock-tested; live Gemini/Langfuse acceptance is still pending.

Cointegration/ECM, advanced seasonal models, complete visual residual diagnostics,
and continuous monitoring remain later milestones, not features of this demo.

- [Executive plan and 14-sprint roadmap](docs/EXECUTIVE_PLAN.md)
- [Next increment: conversational Gemini agent and C1-C6 sprint plan](docs/CONVERSATIONAL_AGENT_PLAN.md)
- [Conversation commands, configuration, and limitations](docs/CHAT_GUIDE.md)
- [Implementation and validation ledger](docs/AGENT_DELIVERY_LOG.md)
- [Agent workflow and decision graph](docs/AGENT_WORKFLOW.md)
- [Proposed decision policy](config/decision_policy.toml)
- [Data-source assessment](docs/DATA_SOURCES.md)
- [Demo guide and validation evidence](docs/DEMO_GUIDE.md)
- [CSV input format](docs/CSV_FORMAT.md)

Code, documentation, identifiers, and default UI text are English. Conversation
and prompt content may be Polish.

## Run the demo

With Python 3.12 and `uv` installed:

```powershell
uv sync --locked --extra dev
uv run manto ui
```

Open [the local workbench](http://127.0.0.1:8501), select **Synthetic sales demo**, click
the example question, and continue in chat: `all candidates`, `confirm settings`,
then `run` after inspecting the report. This first run is preliminary; say `full`
and approve its new report to add the holdout audit and next forecast. Three
predictors from ten with inflation pinned produces 36 models at lag zero.
All example data are synthetic.

For a terminal-only demo or verification:

```powershell
uv run manto demo --pin inflation
uv run pytest -q
uv run ruff check src tests
```

Results, data snapshots and conversation checkpoints are saved under `.manto/`
(ignored by Git). Forecasts return to the original target scale after any internal
stationarity transformation. Monthly targets must be observed by the forecast
origin; delayed target releases require a later nowcasting extension.

Optional service settings are documented in [.env.example](.env.example). Use
fresh credentials in a local `.env`; never commit secrets. Live provider calls
were not needed for the offline acceptance run.
