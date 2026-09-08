import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from manto.domain import SeriesInfo
from manto.providers import IntentProposal, IntentService, Observability, offline_intent, redact


@pytest.fixture
def catalog():
    return [
        SeriesInfo(id="sales", title="Sales"),
        SeriesInfo(id="inflation", title="Inflation"),
        SeriesInfo(id="demand", title="Demand"),
    ]


def test_polish_and_english_offline(catalog):
    for message in [
        "Prognozuj sprzedaż z trzy zmienne i uwzględnij inflację",
        "Forecast sales with three predictors and pin inflation",
    ]:
        proposal = offline_intent(message, catalog)
        assert proposal.target_id == "sales"
        assert proposal.model_size == 3
        assert proposal.pinned_ids == ["inflation"]
        assert proposal.pins_confirmed
    assert offline_intent("Forecast sales; pin GDP", catalog).unresolved_terms


def test_explicit_enable_required(monkeypatch, catalog):
    monkeypatch.delenv("MANTO_ENABLE_GEMINI", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "private")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    service = IntentService()
    assert service.client is None
    assert service.extract("Forecast sales", catalog).target_id == "sales"
    assert service.mode == "Guided offline mode"


def test_structured_provider_success(catalog):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed=IntentProposal(
            target_id="sales", model_size=2, candidate_ids=["inflation", "demand"]
        ),
        text=None,
    )
    service = IntentService(client, "test-model")
    assert service.extract("sales", catalog).target_id == "sales"
    assert client.models.generate_content.call_count == 1
    assert service.last_event["attempts"] == 1
    assert (
        client.models.generate_content.call_args.kwargs["config"].response_schema is IntentProposal
    )


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(parsed=None, text="not json"),
        SimpleNamespace(parsed={"target_id": "invented"}, text=None),
        SimpleNamespace(parsed={"target_id": "sales", "model_size": 99}, text=None),
    ],
)
def test_malformed_provider_is_bounded_and_falls_back(catalog, response):
    client = Mock()
    client.models.generate_content.return_value = response
    service = IntentService(client, "test-model")
    assert service.extract("Forecast sales", catalog).target_id == "sales"
    assert client.models.generate_content.call_count == 2
    assert service.mode == "Guided offline mode"


def test_provider_exception_text_not_saved(catalog):
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError("super-secret-value")
    service = IntentService(client, "test-model")
    service.extract("sales", catalog)
    assert "super-secret-value" not in json.dumps(service.last_event)


def test_redacts_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "example-environment-secret")
    text = "example-environment-secret api_key=another-value AQ." + "x" * 30
    assert "example-environment-secret" not in redact(text)
    assert "another-value" not in redact(text)
    assert "AQ." not in redact(text)


def test_observability_outage_does_not_discard_work():
    client = Mock()
    client.start_as_current_observation.side_effect = RuntimeError("secret")
    observer = Observability(client)
    with observer.span("analysis", "run-1"):
        value = 42
    assert value == 42
    assert observer.status == "unavailable"


def test_observability_only_emits_metadata():
    captured = {}

    @contextmanager
    def observation(**kwargs):
        captured.update(kwargs)
        yield None

    observer = Observability(SimpleNamespace(start_as_current_observation=observation))
    with observer.span("intent", "run-1"):
        pass
    assert captured == {
        "name": "manto.intent",
        "as_type": "span",
        "metadata": {"experiment_id": "run-1", "node": "intent"},
    }
