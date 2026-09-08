"""Checkpointed, human-reviewed analytical conversation built with LangGraph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from manto.domain import AnalysisRequest, AnalysisResult, Dataset, Decision
from manto.providers import IntentService, Observability, redact


class ConversationState(TypedDict, total=False):
    message: str
    messages: list[dict[str, str]]
    status: str
    question: dict[str, Any] | None
    request: dict[str, Any]
    result: dict[str, Any] | None
    answer: Any
    errors: list[str]
    experiment_id: str
    events: list[dict[str, Any]]
    assistant_mode: str


class Conversation:
    """A durable conversation; reuse the same dataset and thread ID after restart.

    ``resume(id, {"request": request_dict, "approved": True})`` confirms a
    specification. ``{"cancelled": True}`` cancels. Completed calls are reads,
    not re-executions; use a new thread ID to branch a changed specification.
    """

    def __init__(
        self,
        database_path: str | Path,
        dataset: Dataset,
        analyzer: Callable[..., AnalysisResult] | None = None,
        *,
        intent_service: IntentService | None = None,
        observability: Observability | None = None,
    ):
        self.dataset = dataset
        self.analyzer = analyzer
        self.intent_service = intent_service or IntentService()
        self.observability = observability or Observability()
        if str(database_path) != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(database_path), check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS manto_threads "
            "(thread_id TEXT PRIMARY KEY, dataset_digest TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS manto_analysis_cache "
            "(experiment_id TEXT PRIMARY KEY, result_json TEXT NOT NULL)"
        )
        self.conn.commit()
        digest = hashlib.sha256()
        digest.update(pd.util.hash_pandas_object(dataset.observations, index=True).values.tobytes())
        digest.update(
            json.dumps([s.model_dump() for s in dataset.catalog], sort_keys=True).encode()
        )
        digest.update(f"{dataset.name}:{dataset.provenance}".encode())
        self.dataset_digest = digest.hexdigest()
        graph = StateGraph(ConversationState)
        for name, method in (
            ("intent", self._intent),
            ("review", self._review),
            ("validate", self._validate),
            ("analysis", self._analysis),
            ("explain", self._explain),
        ):
            graph.add_node(name, method)
        graph.add_edge(START, "intent")
        graph.add_edge("intent", "review")
        graph.add_edge("review", "validate")
        graph.add_conditional_edges(
            "validate", self._route, {"analysis": "analysis", "review": "review", "end": END}
        )
        graph.add_edge("analysis", "explain")
        graph.add_edge("explain", END)
        # LangGraph writes checkpoints in its background executor. Application
        # cache transactions must not share that connection: SqliteSaver's lock
        # protects only its own calls, not our cache inserts/commits.
        self.checkpoint_conn = sqlite3.connect(
            str(database_path), check_same_thread=False, timeout=30
        )
        self.checkpoint_conn.execute("PRAGMA journal_mode=WAL")
        self.graph = graph.compile(checkpointer=SqliteSaver(self.checkpoint_conn))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.observability.flush()
        self.checkpoint_conn.close()
        self.conn.close()

    @staticmethod
    def _config(thread_id: str) -> dict:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("A nonempty conversation ID is required")
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    def _check_dataset(self, thread_id: str, *, register: bool = False):
        self._config(thread_id)
        row = self.conn.execute(
            "SELECT dataset_digest FROM manto_threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row and row[0] != self.dataset_digest:
            raise ValueError(
                "This conversation belongs to a different dataset. Start a new thread."
            )
        if not row and register:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO manto_threads VALUES (?, ?)", (thread_id, self.dataset_digest)
                )

    def start(self, thread_id: str, message: str) -> dict:
        self._check_dataset(thread_id, register=True)
        if self.graph.get_state(self._config(thread_id)).values:
            return self.state(thread_id)
        safe_message = redact(str(message))[:20000]
        self.graph.invoke(
            {
                "message": safe_message,
                "messages": [{"role": "user", "content": safe_message}],
                "status": "running",
                "question": None,
                "request": {},
                "result": None,
                "errors": [],
                "events": [],
                "experiment_id": str(uuid.uuid4()),
            },
            self._config(thread_id),
        )
        return self.state(thread_id)

    def resume(self, thread_id: str, answer: dict) -> dict:
        self._check_dataset(thread_id)
        snapshot = self.graph.get_state(self._config(thread_id))
        if not snapshot.values:
            raise ValueError("Start this conversation before resuming it")
        if snapshot.values.get("status") in {"completed", "failed", "cancelled"}:
            return self.state(thread_id)
        # Do not place arbitrary keys or raw exception/input text in checkpoints.
        clean = self._clean_answer(answer)
        self.graph.invoke(Command(resume=clean), self._config(thread_id))
        return self.state(thread_id)

    @staticmethod
    def _clean_answer(answer: Any) -> dict:
        if not isinstance(answer, dict):
            return {"malformed": True}
        clean = {
            key: answer[key] is True for key in ("approved", "cancelled", "cancel") if key in answer
        }
        request = answer.get("request")
        if isinstance(request, dict):
            allowed = AnalysisRequest.model_fields
            clean["request"] = {key: value for key, value in request.items() if key in allowed}
            # Recursively redact credentials even when placed in a malformed field.
            try:
                clean["request"] = json.loads(redact(json.dumps(clean["request"])))
            except (TypeError, ValueError):
                return {"malformed": True}
        elif "request" in answer:
            clean["malformed"] = True
        return clean

    def state(self, thread_id: str) -> dict:
        self._check_dataset(thread_id)
        snapshot = self.graph.get_state(self._config(thread_id))
        values = dict(snapshot.values)
        question = None
        for task in snapshot.tasks:
            if task.interrupts:
                question = task.interrupts[0].value
                break
        status = "awaiting_input" if question is not None else values.get("status", "not_started")
        return {
            "status": status,
            "question": question,
            "request": values.get("request", {}),
            "result": values.get("result"),
            "messages": values.get("messages", []),
            "experiment_id": values.get("experiment_id"),
            "decisions": values.get("events", []),
            "assistant_mode": values.get("assistant_mode", self.intent_service.mode),
            "telemetry_status": self.observability.status,
        }

    def graph_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def _intent(self, state: ConversationState) -> dict:
        with self.observability.span("intent", state["experiment_id"]):
            proposal = self.intent_service.extract(state["message"], self.dataset.catalog)
        request: dict[str, Any] = {
            "candidate_ids": proposal.candidate_ids,
            "pinned_ids": proposal.pinned_ids,
        }
        if proposal.target_id:
            request["target_id"] = proposal.target_id
        if proposal.model_size:
            request["model_size"] = proposal.model_size
        errors = list(proposal.unresolved_terms)
        if not proposal.target_id:
            errors.append("Choose the target series to forecast.")
        if not proposal.model_size:
            errors.append("Choose how many external predictors each model should use (1 to 4).")
        if not proposal.pins_confirmed:
            errors.append(
                "Choose any predictors that must be fixed in every model, or confirm none."
            )
        event = Decision(
            rule_id="D01",
            outcome="review_required",
            explanation="Monthly, one-month-ahead forecasting; user reviews all choices.",
            evidence=self.intent_service.last_event,
        ).model_dump()
        content = (
            f"{self.intent_service.mode}. Review the target, candidate predictors, model size and "
            "fixed predictors. Forecasts are reported in the target's original units; training-only "
            "stationarity transformations and their inversion are recorded in the analysis."
        )
        return {
            "request": request,
            "errors": errors,
            "events": [event],
            "assistant_mode": self.intent_service.mode,
            "messages": state["messages"] + [{"role": "assistant", "content": content}],
        }

    def _review(self, state: ConversationState) -> dict:
        question = {
            "kind": "review",
            "message": "Review or complete the proposed analysis specification.",
            "errors": state.get("errors", []),
            "request": state.get("request", {}),
            "catalog": [item.model_dump() for item in self.dataset.catalog],
            "choices": "Submit the request with approved=true, or cancelled=true to stop.",
        }
        answer = interrupt(question)
        return {"answer": answer}

    def _validate(self, state: ConversationState) -> dict:
        answer = state.get("answer", {})
        if answer.get("cancelled") is True or answer.get("cancel") is True:
            return {
                "status": "cancelled",
                "question": None,
                "messages": state["messages"]
                + [{"role": "assistant", "content": "Analysis cancelled."}],
            }
        request_data = answer.get("request", state.get("request", {}))
        errors = []
        if answer.get("malformed"):
            errors.append("Submit a structured request using the review form.")
        if answer.get("approved") is not True:
            errors.append("Confirm the specification, including fixed predictors, before analysis.")
        request = None
        try:
            request = AnalysisRequest.model_validate(request_data)
        except (ValidationError, TypeError) as exc:
            if isinstance(exc, ValidationError):
                for item in exc.errors(include_input=False, include_url=False):
                    field = ".".join(str(part) for part in item["loc"]) or "request"
                    # Pydantic model validator messages are local trusted strings.
                    errors.append(f"{field}: {redact(item['msg'])}")
            else:
                errors.append("The analysis request must be an object.")
        if request:
            ids = {item.id for item in self.dataset.catalog}
            if request.target_id not in ids:
                errors.append("Choose a target present in this dataset catalog.")
            if not set(request.candidate_ids).issubset(ids):
                errors.append("All candidate predictors must be present in this dataset catalog.")
            # Use the same calendar/lag/anchor budget as the numerical implementation.
            from manto.analysis import estimate_work

            estimate = estimate_work(self.dataset, request)
            model_count = estimate["model_count"]
            upper_fits = estimate["estimated_fits"]
            if estimate["reason"]:
                errors.append(
                    {
                        "model_budget_exceeded": f"The request expands to {model_count} models; reduce candidates or lags.",
                        "fit_budget_exceeded": f"Estimated fit count ({upper_fits}) exceeds max_fits; narrow the request.",
                        "insufficient_history": "Not enough history remains for training, lags, development and holdout.",
                    }.get(
                        estimate["reason"],
                        "The specification is not feasible; review the data and settings.",
                    )
                )
        if errors:
            return {
                "status": "awaiting_input",
                "errors": errors,
                "request": request_data,
                "messages": state["messages"]
                + [{"role": "assistant", "content": " ".join(errors)}],
            }
        events = state["events"] + [
            Decision(
                rule_id="D02",
                outcome="user_approved",
                explanation="User approved the candidate universe and fixed predictors.",
                evidence={"candidate_ids": request.candidate_ids, "pinned_ids": request.pinned_ids},
            ).model_dump(),
            Decision(
                rule_id="D03",
                outcome="scope_feasible",
                explanation="Catalog, sample and conservative computation-budget checks passed.",
                evidence={"model_count": model_count, "upper_fit_count": upper_fits},
            ).model_dump(),
        ]
        return {"status": "ready", "request": request.model_dump(), "errors": [], "events": events}

    @staticmethod
    def _route(state: ConversationState) -> str:
        if state["status"] == "cancelled":
            return "end"
        return "analysis" if state["status"] == "ready" else "review"

    def _analysis(self, state: ConversationState) -> dict:
        with self.observability.span("analysis", state["experiment_id"]):
            cached = self.conn.execute(
                "SELECT result_json FROM manto_analysis_cache WHERE experiment_id = ?",
                (state["experiment_id"],),
            ).fetchone()
            if cached:
                return {"result": json.loads(cached[0]), "status": "analyzed"}
            try:
                analyzer = self.analyzer
                if analyzer is None:
                    from manto.analysis import run_analysis

                    analyzer = run_analysis
                result = analyzer(
                    self.dataset,
                    AnalysisRequest.model_validate(state["request"]),
                    experiment_id=state["experiment_id"],
                )
                result = AnalysisResult.model_validate(result)
                result.decisions = [
                    Decision.model_validate(event) for event in state["events"]
                ] + result.decisions
                payload = result.model_dump(mode="json")
                with self.conn:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO manto_analysis_cache VALUES (?, ?)",
                        (state["experiment_id"], json.dumps(payload, allow_nan=False)),
                    )
                return {"result": payload, "status": "analyzed"}
            except Exception as exc:  # noqa: BLE001 - graph boundary records a sanitized failure.
                return {
                    "status": "failed",
                    "result": None,
                    "errors": [
                        (
                            f"Analysis could not complete ({type(exc).__name__}). Review the dataset and specification "
                            "and start a new experiment. No champion has been selected."
                        )
                    ],
                }

    def _explain(self, state: ConversationState) -> dict:
        with self.observability.span("explain", state["experiment_id"]):
            if state["status"] == "failed":
                explanation = " ".join(state["errors"])
                status = "failed"
            else:
                result = state["result"]
                explanation = result.get("explanation") or (
                    f"Analysis completed with {len(result.get('models', []))} candidate models. "
                    "Inspect the numerical evidence and recorded decisions."
                )
                explanation += " Associations do not establish causality."
                status = "completed"
            return {
                "status": status,
                "question": None,
                "messages": state["messages"] + [{"role": "assistant", "content": explanation}],
            }
