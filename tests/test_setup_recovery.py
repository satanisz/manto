from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from manto.agent_service import AgentService
from manto.data import demo_dataset
from manto.dialogue import AgentConversation


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    monkeypatch.setenv("MANTO_ENABLE_LANGFUSE", "false")


def no_analysis(*args, **kwargs):
    pytest.fail("Setup recovery must never run analysis")


def test_screenshot_sequence_preserves_feature_request_across_target_discussion(tmp_path):
    data = demo_dataset()
    with AgentConversation(tmp_path, data, analyzer=no_analysis) as chat:
        state = chat.send("flow", "choose for me three relative features")
        assert state.deferred_proposal_count == 3
        assert state.draft.values["target_id"] is None
        assert "Available target options" in state.messages[-1]["content"]
        state = chat.send("flow", "what target can be?")
        assert state.events[-1]["rule_id"] == "C_targets"
        assert state.draft.revision == 0
        state = chat.send("flow", "propose target")
        assert state.draft.values["target_id"] == "sales"
        assert state.draft.settings["target_id"].status == "proposed"
        assert state.draft.values["candidate_ids"] is None
    with AgentConversation(tmp_path, data, analyzer=no_analysis) as chat:
        state = chat.send("flow", "yes")
        assert state.draft.settings["target_id"].status == "confirmed"
        assert len(state.draft.values["candidate_ids"]) == 3
        assert state.draft.settings["candidate_ids"].status == "proposed"
        assert state.deferred_proposal_count is None
        assert [event["rule_id"] for event in state.events[-2:]] == ["C_confirm", "C_propose"]
        state = chat.send("flow", "performe analys")
        assert "review the analysis setup" in state.messages[-1]["content"]
        assert not state.report and not state.result
        assert state.draft.settings["candidate_ids"].status == "proposed"
        state = chat.send("flow", "confirm settings")
        assert state.report["ready"]
        state = chat.send("flow", "performe analys")
        assert "Analysis specification" in state.messages[-1]["content"]
        assert "Say 'run'" in state.messages[-1]["content"]
        assert not state.result
        assert "resume_candidate_request" in chat.graph.get_graph().draw_mermaid()


@pytest.mark.parametrize(
    "message",
    [
        "choose for me three relative features",
        "Please select three relevant predictors",
        "Could you suggest 3 candidate variables?",
        "Propose three",
        "pick three features",
    ],
)
def test_candidate_synonyms_preserve_count_and_model_size(tmp_path, message):
    with AgentConversation(tmp_path, demo_dataset(), analyzer=no_analysis) as chat:
        before = chat.send("features", 'set {"target_id":"sales","model_size":2}')
        state = chat.send("features", message)
        assert len(state.draft.values["candidate_ids"]) == 3
        assert "sales" not in state.draft.values["candidate_ids"]
        assert state.draft.settings["model_size"] == before.draft.settings["model_size"]


@pytest.mark.parametrize(
    "message", ["what target can be?", "show target options", "which targets are available?"]
)
def test_target_catalog_includes_current_target_without_mutation(tmp_path, message):
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        before = chat.send("options", "Forecast sales")
        state = chat.send("options", message)
        assert state.draft == before.draft
        assert state.events[-1]["rule_id"] == "C_targets"
        assert "`sales`" in state.messages[-1]["content"]
        assert "`inflation`" in state.messages[-1]["content"]
        assert chat.send("options", "yes").draft == before.draft


def test_target_choice_resumes_requested_candidates(tmp_path):
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        chat.send("choice", "choose three features")
        state = chat.send("choice", "choose demand as target")
        assert state.draft.values["target_id"] == "demand"
        assert len(state.draft.values["candidate_ids"]) == 3
        assert "demand" not in state.draft.values["candidate_ids"]


@pytest.mark.parametrize(
    "message",
    [
        "performe analys",
        "perform analysis",
        "do the analysis",
        "please begin analysis",
        "don't run analysis",
        "what if we run analysis?",
        "choose twenty one features",
        "choose zero features",
        "choose 21 features",
        "nonsense request",
    ],
)
def test_recovery_does_not_approve_or_execute_ready_report(tmp_path, message):
    with AgentConversation(tmp_path, demo_dataset(), analyzer=no_analysis) as chat:
        chat.send("safe", 'set {"target_id":"sales","candidate_ids":["inflation"],"model_size":1}')
        before = chat.send("safe", "confirm settings")
        state = chat.send("safe", message)
        assert state.draft == before.draft
        assert not state.result
        assert "Offline recovery supports" not in state.messages[-1]["content"]
        assert "Could you clarify the target" not in state.messages[-1]["content"]


def test_gemini_target_proposal_uses_target_contract_and_requires_confirmation(tmp_path):
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed={
            "items": [{"series_id": "demand", "reason": "A demand-planning objective to confirm."}]
        },
        text=None,
    )
    with AgentConversation(
        tmp_path, demo_dataset(), service=AgentService(client=client, model="test")
    ) as chat:
        state = chat.send("target", "propose target")
        assert state.draft.values["target_id"] == "demand"
        assert state.draft.settings["target_id"].status == "proposed"
        assert state.draft.values["candidate_ids"] is None
    assert (
        "Propose exactly one forecast target"
        in client.models.generate_content.call_args.kwargs["contents"]
    )


def test_screenshot_sequence_in_streamlit(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTO_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "src/manto/ui.py"), default_timeout=30
    ).run()
    for message in [
        "choose for me three relative features",
        "what target can be?",
        "propose target",
        "performe analys",
    ]:
        app.chat_input[0].set_value(message).run()
        assert not app.exception
        state = app.session_state["dialogue_state"]
        assert not state["result"]
        assert "Offline recovery supports" not in state["messages"][-1]["content"]
    assert state["draft"]["settings"]["target_id"]["status"] == "proposed"
