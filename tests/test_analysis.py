"""Forecast safety and mathematical contracts of the first demo."""

import numpy as np
import pandas as pd
import pytest

from manto.analysis import enumerate_specs, estimate_work, run_analysis
from manto.domain import AnalysisRequest, Dataset, SeriesInfo


def dataset(periods=112, identical=False, log=False):
    rng = np.random.default_rng(43)
    dates = pd.date_range("2010-01-01", periods=periods, freq="MS")
    x = rng.normal(size=periods)
    z = x.copy() if identical else rng.normal(size=periods)
    changes = np.r_[0.0, 2 * x[:-1] + rng.normal(scale=0.1, size=periods - 1)]
    y = 100 + np.cumsum(changes)
    if log:
        y = np.exp(4 + np.cumsum(changes) * 0.01)
    rows = [
        {
            "series_id": key,
            "period": date,
            "available_at": date.to_period("M").end_time,
            "value": float(value),
        }
        for key, values in [("y", y), ("x", x), ("z", z)]
        for date, value in zip(dates, values, strict=True)
    ]
    return Dataset(
        pd.DataFrame(rows),
        [SeriesInfo(id=k, title=k) for k in ["y", "x", "z"]],
        name="Synthetic",
        provenance="synthetic_point_in_time",
    )


def request(**kwargs):
    return AnalysisRequest(
        **{
            "target_id": "y",
            "candidate_ids": ["x", "z"],
            "model_size": 1,
            "target_transform": "difference",
            "feature_transform": "identity",
            **kwargs,
        }
    )


def test_complete_subset_counts_and_pins():
    req = request(candidate_ids=[f"x{i}" for i in range(10)], model_size=3)
    assert len(enumerate_specs(req)) == 120
    pinned = req.model_copy(update={"pinned_ids": ["x0"]})
    specs = enumerate_specs(pinned)
    assert len(specs) == 36
    assert all("x0" in spec["features"] for spec in specs)
    assert len(enumerate_specs(pinned.model_copy(update={"lag_menu": [0, 1]}))) == 288
    assert len(enumerate_specs(req.model_copy(update={"size_mode": "up_to"}))) == 175


def test_budget_preflight_never_silently_truncates():
    req = request(lag_menu=[0, 1], max_models=1)
    assert estimate_work(dataset(), req)["reason"] == "model_budget_exceeded"
    with pytest.raises(ValueError, match="model_budget_exceeded"):
        enumerate_specs(req)
    result = run_analysis(dataset(), req)
    assert result.status == "blocked" and not result.models
    assert run_analysis(dataset(), request(max_fits=1)).explanation == "fit_budget_exceeded"


def test_difference_predictions_reconstruct_original_units_and_baseline_audit():
    result = run_analysis(dataset(), request())
    assert result.champion_id == result.recommended_id == "ols|x@0"
    chosen = next(model for model in result.models if model.model_id == result.champion_id)
    assert chosen.metrics["development_mae"] < 0.3
    assert all(row["predicted"] > 50 for row in chosen.predictions)
    assert result.next_forecast["predicted"] > 50
    assert result.next_forecast["target_transform"] == "difference"
    assert sum(row["split"] == "holdout" for row in chosen.predictions) == 12
    assert all(
        not any(row["split"] == "holdout" for row in model.predictions)
        for model in result.models
        if model is not chosen
    )
    assert result.model_dump_json()


def test_log_difference_uses_observable_anchor_and_original_scale():
    result = run_analysis(dataset(log=True), request(target_transform="log_difference"))
    chosen = next(model for model in result.models if model.model_id == result.recommended_id)
    assert chosen.metrics["development_mae"] < 0.2
    assert result.next_forecast["estimand"] == "conditional_median"
    assert result.next_forecast["predicted"] > 10


def test_future_changes_cannot_change_first_development_prediction_or_recipe():
    data = dataset()
    req = request(target_transform="auto", feature_transform="auto")
    original = run_analysis(data, req)
    changed = Dataset(data.observations.copy(), data.catalog)
    cutoff = pd.Timestamp(original.split["development_start"]).to_period("M").end_time
    future = changed.observations.available_at > cutoff
    changed.observations.loc[future, "value"] *= 3
    modified = run_analysis(changed, req)
    for first, second in zip(original.models, modified.models, strict=True):
        assert first.transformations == second.transformations
        assert first.predictions[0]["predicted"] == second.predictions[0]["predicted"]


def test_holdout_does_not_choose_recommendation():
    data = dataset()
    original = run_analysis(data, request())
    changed = Dataset(data.observations.copy(), data.catalog)
    first_holdout_target = pd.Timestamp(original.split["holdout_start"]) + pd.offsets.MonthBegin(1)
    mask = (changed.observations.series_id == "y") & (
        changed.observations.period >= first_holdout_target
    )
    changed.observations.loc[mask, "value"] += 500
    modified = run_analysis(changed, request())
    assert original.recommended_id == modified.recommended_id
    assert [model.metrics["development_mae"] for model in original.models] == [
        model.metrics["development_mae"] for model in modified.models
    ]


def test_singular_candidates_are_reason_coded():
    result = run_analysis(dataset(identical=True), request(model_size=2))
    assert result.status == "no_eligible_models"
    assert result.champion_id is None
    assert result.models[0].reason == "singular_design"


def test_missing_release_is_not_backfilled():
    data = dataset()
    origin = 61
    mask = (data.observations.series_id == "x") & (
        data.observations.period == pd.Timestamp("2010-01-01") + pd.DateOffset(months=origin)
    )
    data.observations.loc[mask, "available_at"] += pd.DateOffset(months=1)
    result = run_analysis(data, request())
    x = next(model for model in result.models if model.features == ["x"])
    assert x.status == "ineligible"
    assert x.reason == "predictor_unavailable_at_origin"


def test_revision_published_later_cannot_change_earlier_fit():
    data = dataset()
    original = run_analysis(data, request())
    revision = data.observations.iloc[10].copy()
    revision["available_at"] = pd.Timestamp("2018-01-31 23:59:59")
    revision["value"] += 1000
    data.observations = pd.concat([data.observations, revision.to_frame().T], ignore_index=True)
    revised = run_analysis(data, request())
    assert (
        original.models[0].predictions[0]["predicted"]
        == revised.models[0].predictions[0]["predicted"]
    )


def test_latest_vintage_cannot_qualify_champion():
    data = dataset()
    data.provenance = "latest_vintage_exploratory"
    result = run_analysis(data, request())
    assert result.recommended_id and result.champion_id is None
    assert result.status == "no_qualified_champion"


def test_no_champion_when_persistence_is_perfect_and_mape_is_undefined():
    data = dataset()
    data.observations.loc[data.observations.series_id == "y", "value"] = 0.0
    result = run_analysis(data, request(target_transform="identity"))
    assert result.recommended_id and result.champion_id is None
    chosen = next(model for model in result.models if model.model_id == result.recommended_id)
    assert chosen.metrics["development_mape"] is None
    assert chosen.metrics["development_r2"] is None
    assert chosen.diagnostics["status"] == "constant_residuals_tests_unavailable"
    assert "NaN" not in result.model_dump_json()


def test_explicit_log_requires_positive_training_values_without_dropping_them():
    data = dataset()
    mask = (data.observations.series_id == "y") & (
        data.observations.period == pd.Timestamp("2011-01-01")
    )
    data.observations.loc[mask, "value"] = -1.0
    result = run_analysis(data, request(target_transform="log_difference"))
    assert result.status == "no_eligible_models"
    assert all(model.reason == "nonpositive_log_training_values" for model in result.models)


def test_lag_grid_compares_separate_forecasters():
    data = dataset()
    x = data.observations.loc[data.observations.series_id == "x", "value"].to_numpy()
    target = np.r_[100.0, 100.0, 100.0 + np.cumsum(2 * x[:-2])]
    data.observations.loc[data.observations.series_id == "y", "value"] = target
    result = run_analysis(data, request(candidate_ids=["x"], lag_menu=[0, 1]))
    assert result.recommended_id == "ols|x@1"
    chosen = next(model for model in result.models if model.model_id == result.recommended_id)
    assert chosen.metrics["development_mae"] < 1e-10
