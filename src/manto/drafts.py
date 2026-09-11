"""Versioned, partially discussed specifications; defaults never imply consent."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from manto.domain import AnalysisRequest
from manto.policy import load_policy


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


class DraftSetting(BaseModel):
    value: Any = None
    source: Literal["user", "agent_proposal", "policy_suggestion"] = "policy_suggestion"
    status: Literal["undiscussed", "proposed", "confirmed"] = "undiscussed"
    turn_id: str = ""
    rationale: str = ""


class AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    revision: int = 0
    settings: dict[str, DraftSetting] = Field(default_factory=dict)
    parent_experiment_id: str | None = None

    @classmethod
    def new(cls) -> AnalysisDraft:
        policy = load_policy()
        values = {
            key: field.get_default(call_default_factory=True)
            for key, field in AnalysisRequest.model_fields.items()
            if not field.is_required()
        }
        values.update(
            target_id=None,
            candidate_ids=None,
            pinned_ids=[],
            model_size=policy["default_model_size"],
            lag_menu=policy["default_lag_menu"],
            target_transform=policy["target_transform_default"],
            feature_transform=policy["feature_transform_default"],
            run_mode="preliminary",
            share_vectors=False,
        )
        for key in (
            "initial_train",
            "holdout_periods",
            "min_development",
            "max_models",
            "max_fits",
        ):
            values[key] = policy[key]
        return cls(settings={key: DraftSetting(value=value) for key, value in values.items()})

    @property
    def values(self) -> dict:
        return {key: setting.value for key, setting in self.settings.items()}

    @property
    def unresolved(self) -> list[str]:
        return [
            key
            for key, setting in self.settings.items()
            if setting.status != "confirmed" or setting.value is None
        ]

    def patch(self, changes: dict, turn_id: str, *, proposed=False, rationale="") -> AnalysisDraft:
        """Atomic typed update; changes invalidate dependent choices, not unrelated ones."""
        updated = self.model_copy(deep=True)
        for key, value in changes.items():
            if key not in self.settings:
                raise ValueError(f"Unsupported setting: {key}")
            if key == "run_mode":
                value = TypeAdapter(Literal["preliminary", "full"]).validate_python(value)
            elif key == "share_vectors":
                value = TypeAdapter(bool).validate_python(value, strict=True)
            else:
                annotation = AnalysisRequest.model_fields[key].rebuild_annotation()
                value = TypeAdapter(annotation).validate_python(value, strict=True)
            updated.settings[key] = DraftSetting(
                value=value,
                source="agent_proposal" if proposed else "user",
                status="proposed" if proposed else "confirmed",
                turn_id=turn_id,
                rationale=rationale,
            )
        if "target_id" in changes and changes["target_id"] != self.values["target_id"]:
            for key in (
                "candidate_ids",
                "pinned_ids",
                "lag_menu",
                "target_transform",
                "feature_transform",
            ):
                if key not in changes:
                    updated.settings[key].status = "undiscussed"
            if "candidate_ids" not in changes:
                updated.settings["candidate_ids"].value = None
            if "pinned_ids" not in changes:
                updated.settings["pinned_ids"].value = []
        if (
            "candidate_ids" in changes
            and "pinned_ids" not in changes
            and not set(updated.values["pinned_ids"]).issubset(changes["candidate_ids"])
        ):
            updated.settings["pinned_ids"].status = "undiscussed"
        if changes:
            updated.revision += 1
        return updated

    def confirm(self, displayed_fields: list[str], turn_id: str) -> AnalysisDraft:
        changes = {
            key: self.values[key] for key in displayed_fields if self.values[key] is not None
        }
        confirmed = self.patch(changes, turn_id)
        for key in changes:
            confirmed.settings[key].source = self.settings[key].source
            confirmed.settings[key].rationale = self.settings[key].rationale
        return confirmed

    def request(self) -> AnalysisRequest:
        if self.unresolved:
            raise ValueError("Undiscussed settings: " + ", ".join(self.unresolved))
        return AnalysisRequest.model_validate(
            {key: self.values[key] for key in AnalysisRequest.model_fields}
        )


class DialogueState(BaseModel):
    """Portable state, distinct from legacy single-experiment checkpoints."""

    schema_version: int = 1
    draft: AnalysisDraft = Field(default_factory=AnalysisDraft.new)
    messages: list[dict[str, str]] = Field(default_factory=list)
    pending_fields: list[str] = Field(default_factory=list)
    report: dict | None = None
    result: dict | None = None
    events: list[dict] = Field(default_factory=list)
    processed_turns: list[str] = Field(default_factory=list)
    language: Literal["en", "pl"] = "en"
    mode: str = "Guided offline recovery"
