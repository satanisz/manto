import json
import subprocess
import sys

from manto.analysis import run_analysis
from manto.data import demo_dataset
from manto.dialogue import AgentConversation
from manto.domain import AnalysisRequest


def ready(chat):
    chat.send(
        "flow",
        'set {"target_id":"sales","candidate_ids":["inflation","demand","marketing"],"model_size":2,"pinned_ids":[]}',
    )
    return chat.send("flow", "confirm settings")


def test_preliminary_never_constructs_holdout_panel(tmp_path, monkeypatch):
    from manto import analysis

    data = demo_dataset()
    request = AnalysisRequest(
        target_id="sales", candidate_ids=["inflation", "demand"], model_size=2
    )
    end = sorted(data.observations.period.unique())[-13]
    original_panel = analysis._Panel

    def guarded_panel(dataset, request, calendar):
        assert dataset.observations.period.max() <= end
        assert calendar[-1] <= end
        return original_panel(dataset, request, calendar)

    monkeypatch.setattr(analysis, "_Panel", guarded_panel)
    before = run_analysis(data, request, run_mode="preliminary")
    data.observations.loc[data.observations.period > end, "value"] = 987654321
    after = run_analysis(data, request, run_mode="preliminary")
    assert before.models == after.models
    assert not before.champion_id and not before.next_forecast
    assert all(
        not any(key.startswith("holdout") for key in model.metrics) for model in before.models
    )


def test_approval_execution_lineage_and_terminal_report(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    monkeypatch.setenv("MANTO_ENABLE_LANGFUSE", "false")
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["experiment_id"])
        return run_analysis(*args, **kwargs)

    with AgentConversation(tmp_path, demo_dataset(), analyzer=counted) as chat:
        state = ready(chat)
        assert state.report["ready"]
        printed = subprocess.run(
            [
                sys.executable,
                "-m",
                "manto",
                "spec",
                "--conversation",
                "flow",
                "--directory",
                str(tmp_path),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(printed.stdout) == state.report
        report_id = state.report["report_id"]
        state = chat.send("flow", "run " + "0" * 64)
        assert state.result is None and not calls
        state = chat.send("flow", "run " + report_id)
        assert state.result["run_mode"] == "preliminary"
        first = state.result["experiment_id"]
        assert len(calls) == 1
        state = chat.send("flow", "run")
        assert len(calls) == 1
        state = chat.send("flow", "full")
        assert state.report["ready"] and state.draft.parent_experiment_id == first
        state = chat.send("flow", "run")
        assert state.result["run_mode"] == "full"
        assert state.result["parent_experiment_id"] == first
        assert len(calls) == 2
        state = chat.send("flow", "compare")
        assert '"comparable": true' in state.messages[-1]["content"]
        state = chat.send("flow", 'set {"lag_menu":[0,1]}')
        assert state.report["holdout_exposed"]
        state = chat.send("flow", "run")
        assert state.result["holdout_exposed"] and not state.result["champion_id"]
        assert state.result["status"] == "exploratory_reused_holdout"
        chat.send("flow", "results")
        assert len(calls) == 3
        assert chat.store.load(first).run_mode == "preliminary"


def test_failed_execution_can_retry_without_secret_or_duplicate_result(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    monkeypatch.setenv("MANTO_ENABLE_LANGFUSE", "false")

    def failed(*args, **kwargs):
        raise RuntimeError("sensitive failure details")

    data = demo_dataset()
    with AgentConversation(tmp_path, data, analyzer=failed) as chat:
        ready(chat)
        state = chat.send("flow", "run")
        assert not state.result
        assert "sensitive failure details" not in state.model_dump_json()
    with AgentConversation(tmp_path, data) as chat:
        state = chat.send("flow", "run")
        assert state.result["status"] == "preliminary_completed"
        assert len(chat.store.list_results()) == 1
        assert chat.send("flow", "run").result == state.result


def test_old_report_cannot_run_after_setting_change(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTO_ENABLE_GEMINI", "false")
    with AgentConversation(tmp_path, demo_dataset()) as chat:
        state = ready(chat)
        old_id = state.report["report_id"]
        chat.send("flow", 'set {"model_size":1}')
        state = chat.send("flow", "run " + old_id)
        assert state.result is None
        assert state.events[-1]["outcome"] == "rejected"
        assert not chat.store.list_results()
