from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "src" / "manto" / "ui.py"


def test_offline_workbench_loads_and_opens_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    monkeypatch.setenv("MANTO_ENABLE_LANGFUSE", "false")
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "From a question to an explainable forecast."
    button = next(item for item in app.button if item.label.startswith("Try:"))
    button.click().run()
    assert not app.exception
    assert app.session_state["conversation_state"]["question"]
    assert app.session_state["thread_id"]
    assert app.session_state["conversation_state"]["request"]["pinned_ids"] == ["inflation"]


def test_offline_full_comparison_and_reopen(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    monkeypatch.setenv("MANTO_ENABLE_LANGFUSE", "false")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    next(item for item in app.button if item.label.startswith("Try:")).click().run()
    next(item for item in app.button if item.label == "Run model comparison").click().run()
    assert not app.exception
    result = app.session_state["analysis_result"]
    assert result["models"]
    assert result["recommended_id"]
    assert any(model["is_pareto"] for model in result["models"])
    assert list((tmp_path / "experiments").glob("*/manifest.json"))
    next(item for item in app.button if item.label == "Open saved analysis").click().run()
    assert not app.exception
    assert app.session_state["analysis_result"]["experiment_id"] == result["experiment_id"]
    assert app.session_state["source_choice"] == "Saved input snapshot"
    assert app.session_state["request_defaults"] == result["request"]
    assert not app.session_state.filtered_state.get("conversation_state")


def test_resuming_review_clears_unrelated_result(tmp_path, monkeypatch):
    from manto import ui
    from manto.data import demo_dataset

    state = {"analysis_result": {"experiment_id": "unrelated"}}
    monkeypatch.setattr(ui.st, "session_state", state)
    ui._accept_state({"status": "awaiting_input", "result": None}, tmp_path, demo_dataset())
    assert "analysis_result" not in state


def test_dataset_identity_includes_metadata():
    from manto.data import demo_dataset
    from manto.ui import _dataset_key

    data = demo_dataset()
    old = _dataset_key(data)
    data.provenance = "different_origin"
    assert _dataset_key(data) != old
