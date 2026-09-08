# manto

A conversational assistant for transparent time-series analysis, forecasting,
model comparison, and monitoring.

## Status

Planning stage. Only the Python scaffold and design documents exist; the assistant,
analytical engine, integrations, and monitoring service are not implemented yet.

- [Executive plan and 14-sprint roadmap](docs/EXECUTIVE_PLAN.md)
- [Agent workflow and decision graph](docs/AGENT_WORKFLOW.md)
- [Proposed decision policy](config/decision_policy.toml)
- [Data-source assessment](docs/DATA_SOURCES.md)

Code, documentation, identifiers, and default UI text are English. Conversation
and prompt content may be Polish.

## Run the scaffold

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m manto
```

This currently prints a greeting; it does not launch the planned assistant.
