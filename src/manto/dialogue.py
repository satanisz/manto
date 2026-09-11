"""Multi-turn Gemini/LangGraph conversation, independent of legacy checkpoints."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from manto.agent_service import (
    PROMPT_HASH,
    AgentAction,
    AgentService,
    explicit_confirmation,
    explicit_run,
)
from manto.agent_tools import (
    catalog_facts,
    dataset_digest,
    render_specification,
    specification_report,
    training_evidence,
    validate_catalog,
)
from manto.domain import AnalysisResult, Decision
from manto.drafts import AnalysisDraft, DialogueState
from manto.providers import Observability, _plain, redact
from manto.storage import ResultStore


class TurnState(TypedDict, total=False):
    dialogue: dict
    message: str
    turn_id: str
    action: dict
    reply: str


class AgentConversation:
    """One graph invocation per user turn; persisted experiment IDs outlive turns.

    A separate lock database serializes turns across processes without blocking
    the checkpointer's own database writes. Existing legacy checkpoints are untouched.
    """

    def __init__(self, directory, dataset, *, service=None, analyzer=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.fingerprint = dataset_digest(dataset)
        self.service = service or AgentService()
        self.analyzer = analyzer
        self.observability = Observability()
        self.store = ResultStore(self.directory / "experiments")
        self.conn = sqlite3.connect(
            self.directory / "dialogue.sqlite", check_same_thread=False, timeout=30
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS dialogue_datasets (id TEXT PRIMARY KEY, digest TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS dialogue_reports (thread_id TEXT PRIMARY KEY, report_json TEXT NOT NULL)"
        )
        self.conn.commit()
        self.lock = sqlite3.connect(self.directory / "dialogue_locks.sqlite", timeout=30)
        graph = StateGraph(TurnState)
        graph.add_node("gemini_agent", self._agent)
        graph.add_node("persist_reply", self._reply)
        actions = [
            "ask",
            "catalog",
            "propose",
            "lags",
            "patch",
            "settings",
            "confirm",
            "report",
            "run",
            "results",
            "compare",
            "branch",
        ]
        for action in actions:
            graph.add_node(action, self._tool)
            graph.add_edge(action, "persist_reply")
        graph.add_edge(START, "gemini_agent")
        graph.add_conditional_edges(
            "gemini_agent",
            lambda state: state["action"]["action"],
            {action: action for action in actions},
        )
        graph.add_edge("persist_reply", END)
        self.graph = graph.compile(checkpointer=SqliteSaver(self.conn))

    def close(self):
        self.observability.flush()
        self.conn.close()
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def _config(self, thread_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", thread_id):
            raise ValueError("Invalid conversation ID.")
        row = self.conn.execute(
            "SELECT digest FROM dialogue_datasets WHERE id=?", (thread_id,)
        ).fetchone()
        if row and row[0] != self.fingerprint:
            raise ValueError("This conversation belongs to a different input snapshot.")
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}

    def state(self, thread_id):
        values = self.graph.get_state(self._config(thread_id)).values
        if not values:
            raise ValueError(
                "Conversation not found. Legacy conversations must be imported explicitly."
            )
        return DialogueState.model_validate(values["dialogue"])

    def send(self, thread_id, message, *, turn_id=None):
        self.lock.execute("BEGIN IMMEDIATE")
        try:
            config = self._config(thread_id)
            self.conn.execute(
                "INSERT OR IGNORE INTO dialogue_datasets VALUES (?, ?)",
                (thread_id, self.fingerprint),
            )
            self.conn.commit()
            snapshot = self.graph.get_state(config)
            if snapshot.next:
                self.graph.invoke(None, config)
                snapshot = self.graph.get_state(config)
            state = (
                DialogueState.model_validate(snapshot.values["dialogue"])
                if snapshot.values
                else DialogueState()
            )
            turn_id = turn_id or str(uuid4())
            if turn_id in state.processed_turns:
                return state
            message = redact(str(message))[:12000]
            if not message.strip():
                return state
            plain = _plain(message)
            if any(
                word in plain
                for word in (
                    "prognoz",
                    "sprzedaz",
                    "pokaz",
                    "ile ",
                    "zmienn",
                    "uruchom",
                    "ustawien",
                    "zaproponuj",
                )
            ):
                state.language = "pl"
            state.messages.append({"role": "user", "content": message})
            self.graph.invoke(
                {"dialogue": state.model_dump(), "message": message, "turn_id": turn_id}, config
            )
            completed = self.state(thread_id)
            with self.conn:
                if completed.report:
                    self.conn.execute(
                        "INSERT INTO dialogue_reports VALUES (?, ?) ON CONFLICT(thread_id) DO UPDATE SET report_json=excluded.report_json",
                        (thread_id, json.dumps(completed.report)),
                    )
                else:
                    self.conn.execute(
                        "DELETE FROM dialogue_reports WHERE thread_id=?", (thread_id,)
                    )
            return completed
        finally:
            self.lock.rollback()

    def import_result(self, thread_id, experiment_id):
        """Explicitly open an old immutable result in a new conversation."""
        result = self.store.load(experiment_id)
        if dataset_digest(self.store.load_dataset(experiment_id)) != self.fingerprint:
            raise ValueError("Restore the saved input snapshot before importing its result.")
        self.lock.execute("BEGIN IMMEDIATE")
        try:
            config = self._config(thread_id)
            if self.graph.get_state(config).values:
                raise ValueError("Import requires a new conversation ID.")
            draft = AnalysisDraft.new().patch(result.request.model_dump(), "import")
            draft.parent_experiment_id = experiment_id
            state = DialogueState(
                draft=draft,
                result=result.model_dump(mode="json"),
                messages=[
                    {
                        "role": "assistant",
                        "content": "Saved analysis restored. Ask about its results or change a setting to prepare a linked experiment.",
                    }
                ],
            )
            self.conn.execute(
                "INSERT INTO dialogue_datasets VALUES (?, ?)", (thread_id, self.fingerprint)
            )
            self.conn.commit()
            self.graph.update_state(
                config, {"dialogue": state.model_dump()}, as_node="persist_reply"
            )
            return state
        finally:
            self.lock.rollback()

    def _agent(self, turn):
        state = DialogueState.model_validate(turn["dialogue"])
        with self.observability.span("conversation.route", turn["turn_id"]):
            action = self.service.route(
                turn["message"], state, catalog_facts(self.dataset, state.draft)
            )
        state.mode = self.service.mode
        return {
            "action": AgentAction.model_validate(action).model_dump(),
            "dialogue": state.model_dump(),
        }

    @staticmethod
    def _local(state, english, polish):
        return polish if state.language == "pl" else english

    def _settings(self, state):
        state.pending_fields = list(state.draft.unresolved)
        text = self._local(
            state,
            "These settings still need your confirmation:",
            "Te ustawienia wymagają jeszcze omówienia i potwierdzenia:",
        )
        lines = [
            f"- `{key}`: {json.dumps(state.draft.values[key], ensure_ascii=False)} ({state.draft.settings[key].status})"
            for key in state.pending_fields
        ]
        return (
            text
            + "\n\n"
            + "\n".join(lines)
            + "\n\n"
            + self._local(
                state,
                "Tell me what to change, or say 'confirm settings' to accept the listed values. Null values must first be supplied. This does not run analysis.",
                "Powiedz, co zmienić, albo napisz „akceptuję ustawienia”, aby przyjąć wymienione wartości. Brakujące wartości (null) trzeba najpierw podać. To nie uruchamia analizy.",
            )
        )

    def _next(self, state):
        values = state.draft.values
        if not values["target_id"]:
            state.pending_fields = ["target_id"]
            return self._local(
                state,
                "Which available series should we forecast? Ask me for the catalog if needed.",
                "Który szereg chcesz prognozować? Mogę przedstawić dostępny katalog.",
            )
        if values["candidate_ids"] is None:
            state.pending_fields = ["candidate_ids"]
            return self._local(
                state,
                "Which candidates should we consider? Ask me to list them or propose five with reasons.",
                "Jakich kandydatów rozważamy? Mogę pokazać listę albo zaproponować pięciu z uzasadnieniem.",
            )
        if state.draft.unresolved:
            return self._settings(state)
        state.pending_fields = []
        state.report = self._report(state)
        return render_specification(state.report)

    def _report(self, state):
        return specification_report(self.dataset, state.draft, exposure=self._exposed(state.draft))

    def _exposed(self, draft):
        # Cross-experiment audit tracking is added with execution; conservative
        # legacy handling already recognizes any saved audited matching target.
        for item in self.store.list_results():
            if item.get("target_id") == draft.values["target_id"]:
                result = self.store.load(item["experiment_id"])
                if any("holdout_mae" in model.metrics for model in result.models):
                    return True
        return False

    def _tool(self, turn):
        state = DialogueState.model_validate(turn["dialogue"])
        action = AgentAction.model_validate(turn["action"])
        original = state.model_copy(deep=True)
        evidence = {}
        try:
            reply, evidence = self._dispatch(action, state, turn)
        except (ValueError, TypeError, KeyError) as exc:
            state = original
            reply = (
                self._local(
                    state, "I could not apply that change: ", "Nie mogę zastosować tej zmiany: "
                )
                + redact(str(exc))[:2000]
            )
            evidence = {"rejected": True, "reason": type(exc).__name__}
        state.events.append(
            {
                "rule_id": "C_" + action.action,
                "turn_id": turn["turn_id"],
                "prompt_hash": PROMPT_HASH,
                "mode": state.mode,
                "revision_before": original.draft.revision,
                "revision_after": state.draft.revision,
                "outcome": "rejected" if evidence.get("rejected") else "completed",
                "evidence": evidence,
            }
        )
        return {"dialogue": state.model_dump(), "reply": reply}

    def _dispatch(self, action, state, turn):
        facts = catalog_facts(self.dataset, state.draft)
        evidence = {}
        if action.action == "catalog":
            reply = self._local(
                state,
                f"Available candidates: {facts['available_count']}; selected: {facts['selected_count']}.",
                f"Dostępni kandydaci: {facts['available_count']}; wybrani: {facts['selected_count']}.",
            )
            reply += "\n\n" + "\n".join(
                f"- `{item['id']}` — {item['title']} ({item['unit']})" for item in facts["entries"]
            )
            return reply, facts
        if action.action in {"propose", "lags"}:
            if not state.draft.values["target_id"]:
                return self._next(state), {}
            ids = [item["id"] for item in facts["entries"]][:20]
            evidence = training_evidence(self.dataset, state.draft, ids)
            proposal = self.service.propose(
                action.action,
                action.count,
                {
                    "catalog": facts,
                    "draft": state.draft.model_dump(),
                    "training_evidence": evidence,
                    "language": state.language,
                },
            )
            rationale = ""
            if action.action == "propose":
                if action.count > len(ids):
                    raise ValueError(f"Only {len(ids)} candidate series are available.")
                if proposal:
                    chosen = [item.series_id for item in proposal.items]
                    if (
                        len(chosen) != action.count
                        or len(set(chosen)) != len(chosen)
                        or not set(chosen).issubset(ids)
                    ):
                        raise ValueError(
                            "The provider proposal did not match the requested count/catalog. Ask again or choose manually."
                        )
                    rationale = "\n".join(
                        f"- `{item.series_id}`: {item.reason}" for item in proposal.items
                    )
                else:
                    chosen = list(dict.fromkeys([*state.draft.values["pinned_ids"], *ids]))[
                        : action.count
                    ]
                    rationale = self._local(
                        state,
                        "Offline catalog-order shortlist, not a Gemini relevance ranking. Each variable needs a business hypothesis and chronological validation.",
                        "Lista według kolejności katalogu w trybie offline, nie ranking trafności Gemini. Każda zmienna wymaga hipotezy biznesowej i walidacji chronologicznej.",
                    )
                if not set(state.draft.values["pinned_ids"]).issubset(chosen):
                    raise ValueError(
                        "The proposed shortlist omits a pinned variable; revise count or pins."
                    )
                changes = {"candidate_ids": chosen}
            else:
                changes = {"lag_menu": proposal.lag_menu if proposal else [0, 1]}
                rationale = (
                    proposal.explanation
                    if proposal
                    else self._local(
                        state,
                        "Offline suggestion: compare origin-month and one-month-lagged inputs. Availability must still be respected; this is not evidence of an optimal lag.",
                        "Propozycja offline: porównaj dane z miesiąca początkowego i opóźnione o miesiąc. Nadal obowiązują daty publikacji; to nie dowód optymalnego laga.",
                    )
                )
            state.draft = state.draft.patch(
                changes, turn["turn_id"], proposed=True, rationale=redact(rationale)
            )
            validate_catalog(self.dataset, state.draft)
            state.report = None
            state.pending_fields = list(changes)
            return rationale + "\n\n" + json.dumps(
                changes, ensure_ascii=False
            ) + "\n\n" + self._local(
                state,
                "Accept this proposal or tell me what to change?",
                "Akceptujesz tę propozycję, czy chcesz coś zmienić?",
            ), {"attachment": evidence, "proposal": changes, "rationale": rationale}
        if action.action == "patch":
            changes = action.changes.model_dump(exclude_none=True)
            if not changes:
                raise ValueError("No explicit setting change was identified.")
            state.draft = state.draft.patch(changes, turn["turn_id"])
            validate_catalog(self.dataset, state.draft)
            state.report = None
            state.pending_fields = []
            if state.result:
                state.draft.parent_experiment_id = state.result["experiment_id"]
            return self._next(state), {"changes": changes}
        if action.action == "settings":
            return self._settings(state), {}
        if action.action == "confirm":
            if not explicit_confirmation(turn["message"]):
                raise ValueError(
                    "Explicit confirmation is required. Say 'confirm settings' or 'run'."
                )
            if state.pending_fields:
                confirmed = list(state.pending_fields)
                state.draft = state.draft.confirm(confirmed, turn["turn_id"])
                validate_catalog(self.dataset, state.draft)
                state.report = None
                return self._next(state), {"confirmed_fields": confirmed}
            # A generic yes is not interpreted as execution; require explicit run.
            return self._local(
                state,
                "To execute the displayed ready report, say 'run'.",
                "Aby wykonać pokazany gotowy raport, napisz „uruchom”.",
            ), {}
        if action.action == "report":
            state.report = self._report(state)
            return render_specification(state.report), {"report_id": state.report["report_id"]}
        if action.action == "run":
            if not explicit_run(turn["message"]):
                raise ValueError(
                    "A direct run instruction is required; inspection never starts computation."
                )
            return self._execute(state, turn), {}
        if action.action in {"results", "ask"}:
            if action.action == "ask":
                return action.text or self._next(state), {}
            if not state.result:
                return self._local(
                    state,
                    "No analysis has run yet. Ask for the specification report.",
                    "Nie wykonano jeszcze analizy. Możesz poprosić o raport konfiguracji.",
                ), {}
            result = AnalysisResult.model_validate(state.result)
            model = next(
                (
                    model
                    for model in result.models
                    if model.model_id == (action.reference or result.recommended_id)
                ),
                None,
            )
            facts = {
                "experiment_id": result.experiment_id,
                "status": result.status,
                "explanation": result.explanation,
                "model_count": len(result.models),
                "model": model.model_dump(exclude={"predictions"}) if model else None,
            }
            return "```json\n" + json.dumps(facts, ensure_ascii=False, indent=2) + "\n```", {
                "experiment_id": result.experiment_id
            }
        if action.action == "branch":
            if not state.result:
                raise ValueError("Open an existing result first.")
            state.draft.parent_experiment_id = state.result["experiment_id"]
            state.report = None
            return self._next(state), {"parent": state.draft.parent_experiment_id}
        if action.action == "compare":
            return self._compare(state, action.reference), {}
        raise ValueError("Unsupported agent action.")

    def _execute(self, state, turn):
        report = state.report
        if not report or not report["ready"]:
            raise ValueError(
                "Request a complete specification report and confirm all settings first."
            )
        supplied = re.search(r"\b[a-f0-9]{64}\b", turn["message"])
        if supplied and supplied[0] != report["report_id"]:
            raise ValueError("That report is stale. Inspect and approve the current report.")
        mode = state.draft.values["run_mode"]
        plain = _plain(turn["message"])
        if ("full" in plain and mode != "full") or (
            "preliminary" in plain and mode != "preliminary"
        ):
            raise ValueError(
                "Requested mode differs from the report. Change run_mode and review first."
            )
        experiment_id = report["report_id"][:32]
        existing = self.store.directory / experiment_id / "manifest.json"
        if existing.exists():
            result = self.store.load(experiment_id)
            if result.specification_report != report:
                raise ValueError("Stored result does not match this approval report.")
        else:
            current = self._report(state)
            if current["report_id"] != report["report_id"]:
                state.report = current
                raise ValueError(
                    "Configuration, policy, data, or exposure changed. Request and approve a fresh report."
                )
            from manto.analysis import run_analysis

            try:
                with self.observability.span("conversation.analysis", experiment_id):
                    result = (self.analyzer or run_analysis)(
                        self.dataset,
                        state.draft.request(),
                        experiment_id=experiment_id,
                        run_mode=mode,
                    )
                result = AnalysisResult.model_validate(result)
            except Exception as exc:  # noqa: BLE001 - sanitize provider/numerical boundary
                raise ValueError(
                    f"Analysis failed ({type(exc).__name__}); no result was promoted. You may retry the same report."
                ) from None
            result.parent_experiment_id = state.draft.parent_experiment_id
            result.specification_report = report
            result.holdout_exposed = report["holdout_exposed"]
            if result.holdout_exposed and mode == "full":
                result.champion_id = None
                result.status = "exploratory_reused_holdout"
                result.explanation += " This holdout overlaps previously viewed evidence; the audit is exploratory, not independent qualification."
            result.decisions.insert(
                0,
                Decision(
                    rule_id="C_APPROVAL",
                    outcome="user_approved",
                    explanation="Execution is bound to the exact confirmed report and input/policy hashes.",
                    evidence={
                        "report_id": report["report_id"],
                        "turn_id": turn["turn_id"],
                        "run_mode": mode,
                        "holdout_exposed": result.holdout_exposed,
                        "prompt_hash": PROMPT_HASH,
                    },
                ),
            )
            self.store.save(result, self.dataset)
        state.result = result.model_dump(mode="json")
        state.pending_fields = []
        return (
            result.explanation
            + "\n\n"
            + self._local(
                state,
                "Saved. Ask about selected lags, inspect models, or change a variable for a linked experiment.",
                "Zapisano. Możesz zapytać o wybrane lagi, obejrzeć modele albo zmienić zmienną w nowym, powiązanym eksperymencie.",
            )
        )

    def _compare(self, state, reference):
        if not state.result:
            raise ValueError("Run or reopen an analysis first.")
        result = AnalysisResult.model_validate(state.result)
        parent_id = reference or result.parent_experiment_id
        if not parent_id:
            raise ValueError(
                "This result has no parent. Open a saved analysis and change a variable or run mode first."
            )
        parent = self.store.load(parent_id)
        same_target = parent.request.target_id == result.request.target_id
        same_data = dataset_digest(self.store.load_dataset(parent_id)) == self.fingerprint
        models = [
            next((model for model in item.models if model.model_id == item.recommended_id), None)
            for item in (parent, result)
        ]
        compatible = same_target and same_data and all(models)
        aligned = []
        if compatible:
            predictions = [
                {row["period"]: row for row in model.predictions if row["split"] == "development"}
                for model in models
            ]
            common = sorted(set(predictions[0]) & set(predictions[1]))
            compatible = bool(common) and all(
                predictions[0][period]["actual"] == predictions[1][period]["actual"]
                for period in common
            )
            if compatible:
                for rows in predictions:
                    aligned.append(
                        {
                            "origins": len(common),
                            "start": common[0],
                            "end": common[-1],
                            "mae": sum(
                                abs(rows[p]["actual"] - rows[p]["predicted"]) for p in common
                            )
                            / len(common),
                        }
                    )
        payload = {
            "parent": parent.experiment_id,
            "current": result.experiment_id,
            "run_modes": [parent.run_mode, result.run_mode],
            "changes": {
                key: {"before": getattr(parent.request, key), "after": getattr(result.request, key)}
                for key in type(parent.request).model_fields
                if getattr(parent.request, key) != getattr(result.request, key)
            },
            "comparable": bool(compatible),
            "common_development_scores": aligned,
            "holdout_exposed": result.holdout_exposed,
            "note": "Compare development only on identical target/data and common outcomes; holdout modes remain separate. This is exploratory, not causal evidence.",
        }
        return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"

    def _reply(self, turn):
        state = DialogueState.model_validate(turn["dialogue"])
        state.messages.append({"role": "assistant", "content": redact(turn["reply"])})
        state.processed_turns.append(turn["turn_id"])
        # Read-only CLI projection; the graph checkpoint remains authoritative.
        # This node can replay safely because the projection is an upsert.
        return {"dialogue": state.model_dump()}
