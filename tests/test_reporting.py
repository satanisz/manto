from types import SimpleNamespace

from manto.domain import ModelResult
from manto.reporting import candidate_metrics


def test_candidate_metrics_preserves_computed_metrics_and_filters_nonfinite():
    result = SimpleNamespace(
        models=[
            ModelResult(
                model_id="a",
                features=[],
                lags={},
                metrics={"development_mae": 2, "max_vif": float("inf"), "custom_score": 3},
            ),
            ModelResult(
                model_id="b",
                features=[],
                lags={},
                metrics={"development_mae": None, "max_vif": 4, "undefined": float("nan")},
            ),
            ModelResult(
                model_id="c",
                features=[],
                lags={},
                metrics={"max_vif": float("-inf")},
            ),
        ]
    )
    frame = candidate_metrics(result)
    assert list(frame.columns) == ["development_mae", "max_vif", "custom_score"]
    assert len(frame) == 3
    assert frame["max_vif"].isna().tolist() == [True, False, True]
    assert frame.dropna(subset=["development_mae", "max_vif"]).empty
    assert len(frame.dropna(subset=["development_mae", "custom_score"])) == 1


def test_candidate_metrics_handles_missing_metrics_and_models():
    assert candidate_metrics(SimpleNamespace(models=[])).empty
    assert candidate_metrics(
        SimpleNamespace(models=[ModelResult(model_id="a", features=[], lags={})])
    ).empty
