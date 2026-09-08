# manto

A conversational assistant for transparent time-series analysis, forecasting,
model comparison, and monitoring.

## Status

The first analytical demo (sprints 1-6) is implemented: a local chat workbench,
monthly data import, past-only linear model search, original-scale forecasts,
Pareto comparison, a decision trail, and saved results. It runs without API keys.
Gemini intent extraction and Langfuse tracing are optional integrations.

Cointegration/ECM, advanced seasonal models, complete visual residual diagnostics,
and continuous monitoring remain later milestones, not features of this demo.

- [Executive plan and 14-sprint roadmap](docs/EXECUTIVE_PLAN.md)
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
the example question, then **Run model comparison**. Three predictors from ten
with inflation pinned produces 36 models. All example data are synthetic.

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
