import json

import pytest

from manto.data import demo_dataset, load_snapshot
from manto.domain import AnalysisRequest, AnalysisResult, Decision
from manto.reporting import render_report
from manto.storage import ResultStore


def sample_result(experiment_id="saved-example"):
    return AnalysisResult(
        experiment_id=experiment_id,
        request=AnalysisRequest(target_id="sales", candidate_ids=["inflation"], model_size=1),
        dataset_name="Synthetic sales",
        provenance="synthetic",
        status="no_qualified_champion",
        explanation="No model beat the reference.",
        decisions=[
            Decision(rule_id="D10", outcome="empty_front", explanation="No eligible model.")
        ],
    )


def test_result_and_data_roundtrip_and_immutable(tmp_path):
    store = ResultStore(tmp_path)
    result = sample_result()
    path = store.save(result, demo_dataset())
    assert store.load(result.experiment_id) == result
    assert store.save(result) == path
    manifest = store.list_results()[0]
    snapshot = tmp_path / result.experiment_id / manifest["dataset_snapshot"]
    assert load_snapshot(snapshot).name == demo_dataset().name
    result.explanation = "Changed history"
    with pytest.raises(ValueError, match="immutable"):
        store.save(result)


def test_tamper_and_path_escape_rejected(tmp_path):
    store = ResultStore(tmp_path)
    store.save(sample_result())
    path = tmp_path / "saved-example" / "result.json"
    payload = json.loads(path.read_text())
    payload["status"] = "forged"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity"):
        store.load("saved-example")
    with pytest.raises(ValueError, match="identifier"):
        store.save(sample_result("../escape"))


def test_html_escapes_untrusted_names():
    result = sample_result()
    result.dataset_name = '<script>alert("x")</script>'
    html = render_report(result)
    assert '<script>alert("x")</script>' not in html
    assert "&lt;script&gt;" in html
    assert "No model beat the reference" in html
