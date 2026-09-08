import json
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
    x_axis = next(item for item in app.selectbox if item.label == "X axis")
    y_axis = next(item for item in app.selectbox if item.label == "Y axis")
    assert x_axis.value == "development_mae"
    assert y_axis.value == "max_vif"
    assert "R² (training)" in x_axis.options
    assert "RMSE (validation)" in y_axis.options
    x_axis.select("train_r2")
    y_axis.select("development_rmse").run()
    assert not app.exception
    scatter = json.loads(app.get("plotly_chart")[-1].proto.spec)
    assert scatter["layout"]["xaxis"]["title"]["text"] == "R² (training)"
    assert scatter["layout"]["yaxis"]["title"]["text"] == "RMSE (validation)"
    assert app.session_state["analysis_result"] == result
    next(item for item in app.selectbox if item.label == "Y axis").select("train_r2").run()
    assert not app.exception  # The same metric on both axes is supported.
    next(item for item in app.selectbox if item.label == "X axis").select("holdout_mae")
    next(item for item in app.selectbox if item.label == "Y axis").select("holdout_rmse").run()
    assert not app.exception
    assert any("Showing 1 of" in item.value for item in app.caption)
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


def test_new_chat_target_replaces_previous_widget_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    monkeypatch.setenv("MANTO_ENABLE_LANGFUSE", "false")
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert app.session_state["target_choice"] == "sales"
    app.chat_input[0].set_value("Forecast inflation with 2 variables").run()
    assert not app.exception
    assert app.session_state["target_choice"] == "inflation"
