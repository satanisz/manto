# Sprint 6 implementation contract

The first demo uses `manto.domain` Pydantic contracts and a long-form Dataset.
`data.py` provides `demo_dataset(seed=42, periods=156)`,
`load_csv(source)`, `validate_dataset(dataset)`,
`snapshot_dataset(dataset, directory) -> str`,
`load_snapshot(path) -> Dataset`, and `fetch_nbp_monthly(code, start, end) -> Dataset`.
CSV input is long form: `series_id,period,available_at,value`, with optional title/unit.

`analysis.py` provides `enumerate_specs(request)`, `estimate_work(dataset, request)`,
and `run_analysis(dataset, request, experiment_id=None) -> AnalysisResult`.
All forecast metrics are on the original Y scale. No holdout score is used to
choose a model. Automatic transformation decisions only inspect training data.
Cointegration/ECM and advanced seasonality remain later-sprint functionality.

`workflow.py` provides `Conversation(database_path, dataset, analyzer=run_analysis)`
with `start(thread_id, message) -> dict`, `resume(thread_id, answer: dict) -> dict`,
`state(thread_id) -> dict`, and `graph_mermaid() -> str`.
Returned dicts contain `status`, `question` (dict when awaiting input), `request`
(dict when resolved), `result` (AnalysisResult dict when completed), and `messages`.
User review resumes with `{"request": <AnalysisRequest dict>, "approved": true}`.
Missing-field replies may carry the same complete request. Persist graph checkpoints.

`storage.py` provides `ResultStore(directory)` with `save(result, dataset=None) -> str`,
`load(experiment_id) -> AnalysisResult`, and `list_results() -> list[dict]`.
Storage belongs to the root integrator; `workflow.py` persists its own checkpoints.

The root integrator owns UI, CLI, packaging, docs and integration tests. Data,
analysis, and workflow agents own their files and focused tests. They must not
commit overlapping files or edit shared domain contracts without coordination.

Natural target units are the default. Stationarity transformations are internal,
audited decisions with invertible forecasts. The demo supports synthetic data and
user CSV; credentials are optional. Never reuse credentials pasted into conversation.
