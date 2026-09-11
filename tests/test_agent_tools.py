import pytest

from manto.agent_tools import (
    catalog_facts,
    specification_report,
    training_evidence,
    validate_catalog,
)
from manto.data import demo_dataset
from manto.drafts import AnalysisDraft


def configured():
    draft = AnalysisDraft.new().patch(
        {"target_id": "sales", "candidate_ids": ["inflation", "demand"], "model_size": 2}, "1"
    )
    return draft.confirm(draft.unresolved, "2")


def test_report_binds_all_settings_and_catalog_counts():
    data = demo_dataset()
    draft = configured()
    facts = catalog_facts(data, draft)
    assert facts["available_count"] == 10
    assert facts["selected_count"] == 2
    report = specification_report(data, draft)
    assert report["ready"]
    assert report["work"]["model_count"] == 1
    assert report["dates"]["holdout_outcomes"]
    assert (
        specification_report(data, draft.patch({"lag_menu": [0, 1]}, "3"))["report_id"]
        != report["report_id"]
    )
    assert not specification_report(data, draft.patch({"max_fits": 1}, "4"))["ready"]


def test_evidence_ignores_future_values_revisions_and_generator_formula():
    data = demo_dataset()
    draft = configured()
    original = training_evidence(data, draft, ["sales", "inflation"])
    assert "vector" not in original["series"][0]
    cutoff = original["cutoff"]
    data.observations.loc[data.observations.available_at > cutoff, "value"] = 123456789
    data.provenance = "synthetic; secret generator formula"
    assert training_evidence(data, draft, ["sales", "inflation"]) == original
    vectors = training_evidence(data, draft.patch({"share_vectors": True}, "3"), ["sales"])
    assert len(vectors["series"][0]["vector"]) <= 24
    assert "secret generator" not in str(vectors)
    assert "secret generator" not in str(catalog_facts(data, draft))


def test_invalid_ids_lags_and_unconfirmed_report():
    data = demo_dataset()
    assert not specification_report(data, AnalysisDraft.new())["ready"]
    for patch in ({"candidate_ids": ["missing"]}, {"lag_menu": [13]}, {"candidate_ids": ["sales"]}):
        with pytest.raises(ValueError):
            validate_catalog(data, configured().patch(patch, "3"))
    with pytest.raises(ValueError):
        training_evidence(data, configured(), ["invented"])
