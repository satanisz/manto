from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from manto.agent_service import AgentService
from manto.conversation_help import TOPICS
from manto.data import demo_dataset
from manto.dialogue import AgentConversation
from manto.drafts import AnalysisDraft


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    monkeypatch.setenv("MANTO_ENABLE_LANGFUSE", "false")


@pytest.mark.parametrize(
    "question",
    [
        "co robi initial_train?",
        "Co oznacza initial_train",
        "wyjaśnij initial_train",
        "po co initial_train?",
    ],
)
def test_polish_question_has_local_english_answer(tmp_path, question):
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        state = chat.send("question", question)
    answer = state.messages[-1]["content"]
    assert "Minimum usable monthly training pairs" in answer
    assert "60" in answer and "five years" in answer
    assert "Offline recovery supports" not in answer
    assert state.events[-1]["rule_id"] == "C_explain_setting"
    assert state.draft.revision == 0


def test_every_execution_setting_has_a_definition():
    assert set(AnalysisDraft.new().settings).issubset(TOPICS)


def test_known_parameter_does_not_need_a_provider_even_when_configured(tmp_path):
    client = MagicMock()
    with AgentConversation(
        tmp_path, demo_dataset(), service=AgentService(client=client, model="test-model")
    ) as chat:
        state = chat.send("local", "co robi initial_train?")
        assert "≥ 24" in state.messages[-1]["content"]
        client.models.generate_content.assert_not_called()


def test_result_lag_question_is_not_intercepted_by_parameter_help(tmp_path):
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        state = chat.send("results", "Jakie wybrane lagi ma rekomendowany model?")
        assert state.events[-1]["rule_id"] == "C_results"
        assert not state.result


def test_ambiguous_multi_field_edit_does_not_apply_partial_change_or_confirm(tmp_path):
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        before = chat.send("ambiguous", "Forecast sales")
        state = chat.send("ambiguous", "ustaw holdout_periods 6 i lag_menu 1 3")
        assert state.events[-1]["rule_id"] == "C_clarify"
        assert state.draft == before.draft
        assert chat.send("ambiguous", "tak").draft == before.draft


def test_detour_preserves_report_and_pending_fields_then_resumes(tmp_path):
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        chat.send("flow", 'set {"target_id":"sales","candidate_ids":["inflation"],"model_size":1}')
        state = chat.send("flow", "report")
        before = state.model_copy(deep=True)
        state = chat.send("flow", "co robi initial_train?")
        assert state.draft == before.draft
        assert state.report == before.report
        assert state.pending_fields == before.pending_fields
        state = chat.send("flow", "tak")
        assert state.draft == before.draft
        state = chat.send("flow", "Ile mamy kandydatów?")
        assert "Available candidates" in state.messages[-1]["content"]
        assert state.events[-1]["rule_id"] == "C_catalog"
        chat.send("flow", "wróć do konfiguracji")
        state = chat.send("flow", "akceptuję ustawienia")
        assert state.report["ready"] and not state.result


def test_hypothesis_uses_real_estimator_without_mutating_or_fitting(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("A question must not fit models")

    with AgentConversation(tmp_path, demo_dataset(), analyzer=forbidden) as chat:
        chat.send("flow", 'set {"target_id":"sales","candidate_ids":["inflation"],"model_size":1}')
        ready = chat.send("flow", "confirm settings")
        chat.send("flow", "co robi initial_train?")
        state = chat.send("flow", "a gdy zwiększę do 72?")
        assert state.events[-1]["rule_id"] == "C_what_if"
        a, b = state.events[-1]["evidence"]["preview"]["estimates"]
        assert a["development_origins"] - b["development_origins"] == 12
        assert state.draft == ready.draft and state.report == ready.report
        state = chat.send("flow", "tak")
        assert state.draft == ready.draft and not state.result
        state = chat.send("flow", "run")
        assert not state.result
        state = chat.send("flow", "ustaw initial_train na 72")
        assert state.draft.values["initial_train"] == 72
        assert state.report["report_id"] != ready.report["report_id"]
        assert not state.result


@pytest.mark.parametrize(
    "message",
    [
        "co jeśli initial_train wynosi 12?",
        "ustaw initial_train 12",
        "co robi initial_trian?",
        "co robi unknown_setting?",
        "72",
    ],
)
def test_invalid_ambiguous_or_misspelled_input_is_recoverable(tmp_path, message):
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        state = chat.send("safe", message)
        assert state.draft.values["initial_train"] == 60
        assert state.messages[-1]["content"]
        assert state.result is None
        assert state.events[-1]["rule_id"] in {"C_what_if", "C_clarify"}


def test_explanation_context_survives_restart_and_new_proposal_can_be_confirmed(tmp_path):
    data = demo_dataset()
    with AgentConversation(tmp_path, data) as chat:
        chat.send("saved", "Forecast sales")
        chat.send("saved", "Explain lag_menu")
    with AgentConversation(tmp_path, data) as chat:
        state = chat.send("saved", "what if 0 1 3?")
        assert state.inquiry["changes"] == {"lag_menu": [0, 1, 3]}
        chat.send("saved", "Propose 5")
        state = chat.send("saved", "yes")
        assert state.draft.settings["candidate_ids"].status == "confirmed"
        graph = chat.graph.get_graph().draw_mermaid()
        for node in ("explain_setting", "what_if", "clarify", "help", "resume_setup"):
            assert node in graph


def test_ui_screenshot_regression(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTO_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "src" / "manto" / "ui.py"), default_timeout=30
    ).run()
    app.chat_input[0].set_value("co robi initial_train?").run()
    assert not app.exception
    answer = app.session_state["dialogue_state"]["messages"][-1]["content"]
    assert "Minimum usable monthly training pairs" in answer
    assert "Offline recovery supports" not in answer
