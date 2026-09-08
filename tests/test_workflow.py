import json

import pandas as pd
import pytest

from manto.domain import AnalysisRequest, AnalysisResult, Dataset, SeriesInfo
from manto.workflow import Conversation


@pytest.fixture
def dataset():
    months = pd.date_range("2010-01-01", periods=120, freq="MS")
    catalog = [
        SeriesInfo(id="sales", title="Sales", role="target"),
        SeriesInfo(id="inflation", title="Inflation"),
        SeriesInfo(id="demand", title="Demand"),
    ]
    observations = pd.DataFrame(
        [
            {"series_id": item.id, "period": month, "available_at": month, "value": float(i + 1)}
            for item in catalog
            for i, month in enumerate(months)
        ]
    )
    return Dataset(observations, catalog, "Test dataset")


@pytest.fixture
def request_data():
    return AnalysisRequest(
        target_id="sales",
        candidate_ids=["inflation", "demand"],
        model_size=2,
        pinned_ids=["inflation"],
    ).model_dump()


def result_for(dataset, request, experiment_id=None):
    return AnalysisResult(
        experiment_id=experiment_id,
        request=request,
        dataset_name=dataset.name,
        provenance=dataset.provenance,
        status="no_qualified_champion",
        explanation="No qualified champion.",
    )


def test_review_checkpoint_survives_restart_and_completion_is_idempotent(
    tmp_path, dataset, request_data
):
    calls = []

    def analyzer(dataset, request, experiment_id=None):
        calls.append(experiment_id)
        return result_for(dataset, request, experiment_id)

    path = tmp_path / "conversation.sqlite"
    with Conversation(path, dataset, analyzer) as conversation:
        state = conversation.start("one", "Forecast sales using two predictors; pin inflation")
        assert state["status"] == "awaiting_input"
        assert state["request"]["target_id"] == "sales"
        assert state["request"]["model_size"] == 2
        assert state["request"]["pinned_ids"] == ["inflation"]
        assert "review" in conversation.graph_mermaid()
    with Conversation(path, dataset, analyzer) as conversation:
        assert conversation.state("one")["status"] == "awaiting_input"
        completed = conversation.resume("one", {"request": request_data, "approved": True})
        assert completed["status"] == "completed"
        assert completed["result"]["status"] == "no_qualified_champion"
        assert [item["rule_id"] for item in completed["result"]["decisions"]] == [
            "D01",
            "D02",
            "D03",
        ]
        assert conversation.resume("one", {"request": request_data, "approved": True}) == completed
        assert conversation.start("one", "Another request") == completed
        assert len(calls) == 1


def test_missing_information_and_invalid_inputs_reinterrupt(tmp_path, dataset, request_data):
    with Conversation(tmp_path / "conversation.sqlite", dataset, result_for) as conversation:
        state = conversation.start("missing", "Please help me")
        assert len(state["question"]["errors"]) == 3
        for payload in [
            None,
            {"request": "bad", "approved": True},
            {"request": {**request_data, "target_id": "unknown"}, "approved": True},
            {
                "request": {**request_data, "candidate_ids": ["inflation", "missing"]},
                "approved": True,
            },
            {"request": {**request_data, "model_size": 4}, "approved": True},
            {"request": request_data, "approved": False},
        ]:
            state = conversation.resume("missing", payload)
            assert state["status"] == "awaiting_input"
            assert state["question"]["errors"]
        state = conversation.resume("missing", {"request": request_data, "approved": True})
        assert state["status"] == "completed"


def test_cancel_never_runs_analysis(tmp_path, dataset):
    def analyzer(*args, **kwargs):
        pytest.fail("Cancelled workflow must not execute analysis")

    with Conversation(tmp_path / "conversation.sqlite", dataset, analyzer) as conversation:
        conversation.start("cancel", "Forecast sales")
        assert conversation.resume("cancel", {"cancelled": True})["status"] == "cancelled"


def test_dataset_cannot_change_under_checkpoint(tmp_path, dataset):
    path = tmp_path / "conversation.sqlite"
    with Conversation(path, dataset, result_for) as conversation:
        conversation.start("one", "Forecast sales")
    dataset.observations.loc[0, "value"] = 999
    with (
        Conversation(path, dataset, result_for) as conversation,
        pytest.raises(ValueError, match="different dataset"),
    ):
        conversation.state("one")


def test_failed_analysis_does_not_persist_exception_secrets(
    tmp_path, dataset, request_data, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-secret")

    def analyzer(*args, **kwargs):
        raise RuntimeError("private-test-secret")

    with Conversation(tmp_path / "conversation.sqlite", dataset, analyzer) as conversation:
        started = conversation.start("one", "Forecast sales. api_key=private-test-secret")
        assert "private-test-secret" not in json.dumps(started)
        state = conversation.resume("one", {"request": request_data, "approved": True})
        assert state["status"] == "failed"
        assert "private-test-secret" not in json.dumps(state)


def test_compute_budget_and_sample_constraints_reinterrupt(tmp_path, dataset, request_data):
    with Conversation(tmp_path / "conversation.sqlite", dataset, result_for) as conversation:
        conversation.start("one", "Forecast sales")
        for invalid in [
            {"max_models": 1, "lag_menu": [0, 1]},
            {"max_fits": 1},
            {"initial_train": 120},
        ]:
            state = conversation.resume(
                "one", {"request": {**request_data, **invalid}, "approved": True}
            )
            assert state["status"] == "awaiting_input"
            assert state["question"]["errors"]
