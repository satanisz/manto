from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from manto.agent_service import AgentAction, AgentService
from manto.data import demo_dataset
from manto.dialogue import AgentConversation


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    monkeypatch.setenv("MANTO_ENABLE_LANGFUSE", "false")


def test_conversation_survives_questions_restart_and_duplicate_turn(tmp_path):
    data = demo_dataset()
    with AgentConversation(tmp_path, data) as chat:
        state = chat.send(
            "chat", "Forecast sales with three variables; pin inflation", turn_id="first"
        )
        assert state.draft.values["target_id"] == "sales"
        assert state.draft.values["model_size"] == 3
        assert state.draft.values["pinned_ids"] == ["inflation"]
        revision = state.draft.revision
        state = chat.send("chat", "How many candidates?", turn_id="count")
        assert "10" in state.messages[-1]["content"]
        assert state.draft.revision == revision
        assert chat.send("chat", "Ignored duplicate", turn_id="count") == state
        graph = chat.graph.get_graph().draw_mermaid()
        assert "propose" in graph and "report" in graph and "gemini_agent" in graph
    with AgentConversation(tmp_path, data) as chat:
        assert chat.state("chat") == state
        state = chat.send("chat", "Propose five")
        assert len(state.draft.values["candidate_ids"]) == 5
        assert "inflation" in state.draft.values["candidate_ids"]
        state = chat.send("chat", "yes")
        assert state.draft.settings["candidate_ids"].status == "confirmed"
        assert state.draft.settings["lag_menu"].status != "confirmed"
        state = chat.send("chat", "confirm settings")
        assert state.report["ready"]
        assert state.result is None


def test_gemini_structured_turn_uses_draft_and_bounded_context(tmp_path):
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed={"action": "patch", "changes": {"target_id": "sales"}}, text=None
    )
    service = AgentService(client=client, model="test-model")
    with AgentConversation(tmp_path, demo_dataset(), service=service) as chat:
        state = chat.send("live-contract", "Prognozuj sprzedaż")
        assert state.mode == "Gemini agent"
        assert state.language == "pl"
        assert state.draft.values["target_id"] == "sales"
    payload = client.models.generate_content.call_args.kwargs
    assert "sales_change" not in payload["contents"]
    assert payload["config"].response_schema == AgentAction


def test_untrusted_agent_cannot_execute_or_confirm_on_read_question(tmp_path):
    service = MagicMock(mode="Mock agent")
    service.route.return_value = AgentAction(action="run")
    with AgentConversation(tmp_path, demo_dataset(), service=service) as chat:
        state = chat.send("safe", "Show the report")
        assert state.result is None
        assert state.events[-1]["outcome"] == "rejected"
        service.route.return_value = AgentAction(action="confirm")
        state = chat.send("safe", "What is a lag?")
        assert state.draft.unresolved
        assert state.events[-1]["outcome"] == "rejected"


def test_dataset_mismatch_rejected(tmp_path):
    data = demo_dataset()
    with AgentConversation(tmp_path, data) as chat:
        chat.send("safe", "Forecast sales")
    data.observations.loc[0, "value"] += 1
    with AgentConversation(tmp_path, data) as chat, pytest.raises(ValueError, match="different"):
        chat.state("safe")


def test_polish_dialogue_and_proposal_confirmation_keeps_rationale(tmp_path):
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        chat.send("pl", "Chcę prognozować sprzedaż z trzema zmiennymi")
        counted = chat.send("pl", "Ile mamy kandydatów?")
        assert "Dostępni kandydaci: 10" in counted.messages[-1]["content"]
        proposal = chat.send("pl", "Zaproponuj 5 kandydatów")
        reason = proposal.draft.settings["candidate_ids"].rationale
        confirmed = chat.send("pl", "tak")
        assert confirmed.draft.settings["candidate_ids"].rationale == reason
        assert confirmed.draft.settings["candidate_ids"].source == "agent_proposal"
        chat.send("pl", "Zaproponuj lagi")
        state = chat.send("pl", "tak")
        assert state.draft.values["lag_menu"] == [0, 1]
        assert state.result is None


def test_malformed_provider_is_bounded_and_never_leaks_exception(tmp_path):
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("private-provider-secret")
    service = AgentService(client=client, model="test-model")
    with AgentConversation(tmp_path, demo_dataset(), service=service) as chat:
        state = chat.send("fallback", "Forecast sales")
    assert client.models.generate_content.call_count == 2
    assert "offline" in state.mode
    assert "private-provider-secret" not in state.model_dump_json()
    assert state.draft.values["target_id"] == "sales"


def test_provider_question_cannot_patch_settings(tmp_path):
    service = MagicMock(mode="Mock agent")
    service.route.return_value = AgentAction(action="patch", changes={"model_size": 4})
    with AgentConversation(tmp_path, demo_dataset(), service=service) as chat:
        state = chat.send("question", "Why use three variables?")
        assert state.draft.settings["model_size"].status == "undiscussed"
        assert state.events[-1]["outcome"] == "rejected"
