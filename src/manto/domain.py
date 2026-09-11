"""Shared contracts for the first analytical demo; no framework state here."""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class SeriesInfo(BaseModel):
    id: str
    title: str
    unit: str = "units"
    source: str = "user"
    description: str = ""
    role: Literal["target", "predictor"] = "predictor"


@dataclass
class Dataset:
    """Long observations: series_id, period, available_at, value.

    period is a month-start Timestamp; available_at is UTC-naive Timestamp.
    Revisions have distinct availability timestamps. No future values may enter
    a forecast. Additional columns are permitted for provenance.
    """

    observations: pd.DataFrame
    catalog: list[SeriesInfo]
    name: str = "Dataset"
    provenance: str = "user_supplied_availability"


class AnalysisRequest(BaseModel):
    target_id: str
    candidate_ids: list[str]
    pinned_ids: list[str] = Field(default_factory=list)
    model_size: int = Field(default=3, ge=1, le=4)
    size_mode: Literal["exact", "up_to"] = "exact"
    lag_menu: list[int] = Field(default_factory=lambda: [0])
    target_transform: Literal["auto", "identity", "difference", "log_difference"] = "auto"
    feature_transform: Literal["auto", "identity", "difference"] = "auto"
    initial_train: int = Field(default=60, ge=24)
    holdout_periods: int = Field(default=12, ge=3)
    min_development: int = Field(default=24, ge=6)
    max_models: int = Field(default=5000, ge=1, le=10000)
    max_fits: int = Field(default=100000, ge=1, le=1000000)

    @model_validator(mode="after")
    def valid_constraints(self):
        for name in ("candidate_ids", "pinned_ids", "lag_menu"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique entries")
        if not self.lag_menu or any(lag < 0 or lag > 12 for lag in self.lag_menu):
            raise ValueError("Choose nonnegative monthly lags from 0 through 12")
        if self.target_id in self.candidate_ids:
            raise ValueError("The target cannot also be an external predictor")
        if not set(self.pinned_ids).issubset(self.candidate_ids):
            raise ValueError("Pinned series must belong to the candidate set")
        if len(self.pinned_ids) > self.model_size:
            raise ValueError("Pinned series exceed the requested model size")
        if len(self.candidate_ids) < self.model_size:
            raise ValueError("Not enough candidate series for the requested model size")
        return self


class Decision(BaseModel):
    rule_id: str
    outcome: str
    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class ModelResult(BaseModel):
    model_id: str
    features: list[str]
    lags: dict[str, int]
    status: str = "eligible"
    reason: str = ""
    metrics: dict[str, float | None] = Field(default_factory=dict)
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    coefficients: dict[str, float] = Field(default_factory=dict)
    transformations: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    is_pareto: bool = False


class AnalysisResult(BaseModel):
    experiment_id: str
    request: AnalysisRequest
    dataset_name: str
    provenance: str
    status: str
    models: list[ModelResult] = Field(default_factory=list)
    champion_id: str | None = None
    recommended_id: str | None = None
    explanation: str = ""
    decisions: list[Decision] = Field(default_factory=list)
    split: dict[str, Any] = Field(default_factory=dict)
    next_forecast: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    policy_version: str = "demo-0.1"
    run_mode: Literal["preliminary", "full"] = "full"
    parent_experiment_id: str | None = None
    specification_report: dict[str, Any] | None = None
    holdout_exposed: bool = False
