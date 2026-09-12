"""Gemini turn routing and evidence-backed proposals with an honest recovery mode."""

import json
import re
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from manto.conversation_help import help_action
from manto.domain import AnalysisRequest
from manto.drafts import digest
from manto.providers import IntentService, _mentions, _plain, redact

PROMPT = files("manto").joinpath("agent_prompt.txt").read_text(encoding="utf-8")
PROMPT_HASH = digest(PROMPT)

DraftPatch = create_model(
    "DraftPatch",
    __config__=ConfigDict(extra="forbid"),
    **{
        key: (field.rebuild_annotation() | None, None)
        for key, field in AnalysisRequest.model_fields.items()
    },
    run_mode=(Literal["preliminary", "full"] | None, None),
    share_vectors=(bool | None, None),
)


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
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
        "explain_setting",
        "what_if",
        "clarify",
        "help",
        "resume_setup",
    ]
    changes: DraftPatch = Field(default_factory=DraftPatch)
    count: int = Field(default=5, ge=1, le=20)
    text: str = Field(default="", max_length=6000)
    reference: str | None = Field(default=None, max_length=250)
    hypothetical_value: int | list[int] | None = None


class ProposalItem(BaseModel):
    series_id: str
    reason: str = Field(min_length=1, max_length=1500)


class AgentProposal(BaseModel):
    items: list[ProposalItem] = Field(default_factory=list, max_length=20)
    lag_menu: list[int] = Field(default_factory=list, max_length=13)
    explanation: str = Field(default="", max_length=3000)


def explicit_run(message: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:run|run analysis|run preliminary analysis|run full analysis|execute|uruchom|"
            r"uruchom analize|start|zatwierdz i uruchom)(?:\s+[a-f0-9]{64})?[.!]?",
            _plain(message).strip(),
        )
    )


def explicit_confirmation(message: str) -> bool:
    return _plain(message).strip(" .!") in {
        "yes",
        "tak",
        "ok",
        "accept",
        "confirm",
        "confirm settings",
        "accept settings",
        "accept all listed settings",
        "zatwierdz",
        "potwierdz",
        "akceptuje",
        "akceptuje ustawienia",
        "potwierdz ustawienia",
        "akceptuje pozostale ustawienia",
    }


class AgentService:
    def __init__(self, client=None, model=None):
        base = IntentService(client=client, model=model)
        self.client, self.model = base.client, base.model
        self.mode = "Gemini agent" if self.client and self.model else "Guided offline recovery"

    def _generate(self, schema, context):
        if self.client is None or not self.model:
            return None
        from google.genai import types

        contents = redact(json.dumps(context, ensure_ascii=False))
        if len(contents) > 64000:
            self.mode = "Guided offline recovery (provider context budget exceeded)"
            return None
        for _ in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=PROMPT,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0,
                        max_output_tokens=4096,
                    ),
                )
                self.mode = "Gemini agent"
                return (
                    schema.model_validate(response.parsed)
                    if response.parsed
                    else schema.model_validate_json(response.text)
                )
            except Exception:  # noqa: BLE001 - never surface provider input or credentials
                self.mode = "Guided offline recovery (provider response rejected)"
        self.mode = "Guided offline recovery (Gemini unavailable)"
        return None

    def route(self, message, state, catalog):
        # Run/confirmation permissions are resolved locally against the pending report.
        local_action = help_action(message, state)
        if local_action:
            return AgentAction.model_validate(local_action)
        if explicit_run(message):
            return AgentAction(action="run")
        if explicit_confirmation(message):
            return AgentAction(action="confirm")
        context = {
            "task": "Route the current user turn.",
            "message": message,
            "draft": state.draft.model_dump(),
            "pending_fields": state.pending_fields,
            "inquiry": state.inquiry,
            "report_ready": bool(state.report and state.report["ready"]),
            "recent_messages": state.messages[-16:],
            "catalog": catalog,
            "has_result": bool(state.result),
        }
        if state.result:
            result = state.result
            selected = next(
                (
                    model
                    for model in result["models"]
                    if model["model_id"] == result["recommended_id"]
                ),
                None,
            )
            context["stored_result_evidence"] = {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "explanation": result["explanation"],
                "run_mode": result.get("run_mode", "full"),
                "recommended_model": {
                    key: selected[key] for key in ("model_id", "lags", "metrics", "diagnostics")
                }
                if selected
                else None,
            }
        return self._generate(AgentAction, context) or self._offline(message, state, catalog)

    def propose(self, kind, count, context):
        return self._generate(
            AgentProposal,
            {
                "task": f"Propose {count} available candidate series"
                if kind == "propose"
                else "Propose a small monthly lag grid (0-12).",
                "evidence": context,
                "rules": "Honor pins, exclude the target. Reasons are hypotheses; do not invent statistics.",
            },
        )

    @staticmethod
    def _offline(message, state, catalog):
        plain = _plain(message).strip()
        from manto.domain import SeriesInfo

        entries = [
            SeriesInfo(**{k: v for k, v in item.items() if k in {"id", "title", "unit"}})
            for item in catalog["entries"]
        ]
        # Include the target in mention resolution as it is absent from candidate facts.
        target = state.draft.values["target_id"]
        if target:
            entries.append(SeriesInfo(id=target, title=target))
        mentioned = _mentions(message, entries)
        is_question = "?" in plain or plain.startswith(
            ("why", "what", "how", "czy", "dlaczego", "jaki", "jakie")
        )
        if any(word in plain for word in ("report", "raport", "print", "wydruk")):
            return AgentAction(action="report")
        if any(
            word in plain
            for word in ("undiscussed", "settings", "ustawien", "nie omow", "nie dyskut")
        ):
            return AgentAction(action="settings")
        if any(word in plain for word in ("compare", "porownaj")):
            reference = re.search(r"\b[a-f0-9]{32}\b", plain)
            return AgentAction(action="compare", reference=reference[0] if reference else None)
        if any(word in plain for word in ("result", "wynik", "selected lag", "wybrane lagi")):
            return AgentAction(action="results")
        if state.result and ("lag" in plain or "opozn" in plain) and is_question:
            return AgentAction(action="results")
        if any(word in plain for word in ("propose", "zaproponuj", "przygotuj")):
            if "lag" in plain or "opozn" in plain:
                return AgentAction(action="lags")
            match = re.search(r"\b(\d+)\b", plain)
            count = int(match[1]) if match else 5
            if not 1 <= count <= 20:
                return AgentAction(
                    action="ask", text="Choose between 1 and 20 candidate variables."
                )
            return AgentAction(action="propose", count=count)
        if (
            any(word in plain for word in ("how many", "ile ", "list", "lista", "kandydat"))
            and is_question
            or plain.startswith(("list", "show candidates", "pokaz liste", "ile "))
        ):
            return AgentAction(action="catalog")
        if is_question:
            return AgentAction(
                action="ask",
                text="Offline recovery supports catalog, settings, report, results, and explicit changes. Configure Gemini for open-ended analytical discussion.",
            )
        changes = {}
        if (
            not target or any(word in plain for word in ("forecast", "prognoz", "target", "cel "))
        ) and mentioned:
            changes["target_id"] = mentioned[0]
            target = mentioned[0]
        if any(
            word in plain for word in ("no pins", "bez stalych", "brak stalych", "bez pin", "none")
        ):
            changes["pinned_ids"] = []
        elif any(
            word in plain
            for word in ("pin ", "include", "including", "uwzglednij", "zawsze", "fix")
        ):
            changes["pinned_ids"] = [item for item in mentioned if item != target]
        if any(
            word in plain for word in ("all candidates", "wszystkich kandydat", "wszystkie zmienne")
        ):
            changes["candidate_ids"] = [
                item["id"] for item in catalog["entries"] if item["id"] != target
            ]
        elif any(
            word in plain
            for word in ("candidates:", "kandydaci:", "candidate variables", "zmienne:")
        ):
            changes["candidate_ids"] = [item for item in mentioned if item != target]
        if "replace" in plain or "zamien" in plain:
            if len(mentioned) == 2 and state.draft.values["candidate_ids"]:
                changes["candidate_ids"] = [
                    mentioned[1] if item == mentioned[0] else item
                    for item in state.draft.values["candidate_ids"]
                ]
            else:
                return AgentAction(
                    action="ask", text="Name the existing variable and its replacement."
                )
        if "lag" in plain or "opozn" in plain:
            changes["lag_menu"] = [int(value) for value in re.findall(r"\b\d+\b", plain)]
        else:
            numbers = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "jedna": 1,
                "dwie": 2,
                "dwa": 2,
                "trzy": 3,
                "cztery": 4,
            }
            match = re.search(r"\b([1-4]|one|two|three|four|jedna|dwie|dwa|trzy|cztery)\b", plain)
            if match and (
                any(
                    word in plain
                    for word in (
                        "variable",
                        "predictor",
                        "zmienn",
                        "per model",
                        "instead",
                        "zamiast",
                    )
                )
                or state.pending_fields == ["model_size"]
            ):
                changes["model_size"] = int(match[1]) if match[1].isdigit() else numbers[match[1]]
        if "preliminary" in plain or "wstepn" in plain:
            changes["run_mode"] = "preliminary"
        elif "full" in plain or "pelna" in plain:
            changes["run_mode"] = "full"
        if plain.startswith("set "):
            try:
                changes.update(json.loads(message[4:]))
            except (ValueError, TypeError):
                return AgentAction(
                    action="ask", text='Use set {"setting": value} in offline recovery.'
                )
        return (
            AgentAction(action="patch", changes=changes)
            if changes
            else AgentAction(
                action="ask",
                text="Could you clarify the target, candidate list, or setting to change? Offline recovery has limited language understanding.",
            )
        )
