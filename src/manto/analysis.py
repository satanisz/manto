"""Auditable one-month-ahead linear forecasting, with point-in-time inputs.

Each historical feature row is frozen at its own forecast origin. Training labels
use only revisions available at the fit origin. Scores use first-release outcomes.
Transformation recipes are selected once on the initial training prefix; neither
development outcomes nor the final holdout can change them.
"""

from __future__ import annotations

import warnings
from itertools import combinations, product
from math import comb
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller

from manto.domain import AnalysisRequest, AnalysisResult, Dataset, Decision, ModelResult
from manto.policy import load_policy

_POLICY = load_policy()


def _sizes(request):
    return (
        [request.model_size]
        if request.size_mode == "exact"
        else range(max(1, len(request.pinned_ids)), request.model_size + 1)
    )


def _count(request):
    pinned = len(request.pinned_ids)
    free = len(request.candidate_ids) - pinned
    return sum(
        comb(free, size - pinned) * len(request.lag_menu) ** size for size in _sizes(request)
    )


def enumerate_specs(request: AnalysisRequest) -> list[dict]:
    """Enumerate the complete requested subset/lag grid in stable order."""
    if _count(request) > request.max_models:
        raise ValueError("model_budget_exceeded: revise the explicit search scope")
    specs = []
    free = sorted(set(request.candidate_ids) - set(request.pinned_ids))
    for size in _sizes(request):
        for extra in combinations(free, size - len(request.pinned_ids)):
            features = sorted([*request.pinned_ids, *extra])
            for lags in product(sorted(request.lag_menu), repeat=size):
                lag_map = dict(zip(features, lags, strict=True))
                specs.append(
                    {
                        "model_id": "ols|"
                        + "|".join(f"{feature}@{lag_map[feature]}" for feature in features),
                        "features": features,
                        "lags": lag_map,
                    }
                )
    return specs


def _calendar(dataset, request):
    target = dataset.observations.loc[dataset.observations.series_id == request.target_id, "period"]
    if target.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(pd.Timestamp(target.min()), pd.Timestamp(target.max()), freq="MS")


def estimate_work(dataset: Dataset, request: AnalysisRequest) -> dict:
    """Conservative preflight, reserving a difference anchor for every recipe."""
    calendar = _calendar(dataset, request)
    start = max(request.lag_menu) + 1 + max(request.initial_train, 10 * (request.model_size + 1))
    development = max(0, len(calendar) - 1 - request.holdout_periods - start)
    count = _count(request)
    fits = count * development + request.holdout_periods + 1
    reason = (
        "model_budget_exceeded"
        if count > request.max_models
        else "fit_budget_exceeded"
        if fits > request.max_fits
        else "insufficient_history"
        if development < request.min_development
        else ""
    )
    return {
        "model_count": count,
        "development_origins": development,
        "holdout_origins": request.holdout_periods,
        "estimated_fits": fits,
        "within_budget": count <= request.max_models and fits <= request.max_fits,
        "reason": reason,
        "first_development_index": start,
    }


def _adf(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 24 or not np.isfinite(values).all() or np.ptp(values) < 1e-12:
        return {"pvalue": None, "status": "insufficient_or_constant"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = adfuller(values, regression="c", autolag="AIC", maxlag=min(12, len(values) // 4))
        return {
            "pvalue": float(test[1]),
            "status": "stationary_evidence"
            if test[1] < _POLICY["stationarity_alpha"]
            else "unit_root_not_rejected",
            "used_lag": int(test[2]),
        }
    except (ValueError, np.linalg.LinAlgError):
        return {"pvalue": None, "status": "test_unavailable"}


def _transform(values, recipe):
    if recipe == "identity":
        return np.asarray(values)
    if recipe == "difference":
        return np.diff(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(np.where(np.asarray(values) > 0, values, np.nan)))


def _recipe(values, mode):
    before = _adf(values)
    recipe = mode
    if mode == "auto":
        recipe = (
            "identity"
            if before["pvalue"] is not None and before["pvalue"] < _POLICY["stationarity_alpha"]
            else "difference"
        )
    after = _adf(_transform(values, recipe))
    return {
        "recipe": recipe,
        "before": before,
        "after": after,
        "selection_scope": "initial_training_prefix_only",
        "inverse": {
            "identity": "prediction",
            "difference": "observable_anchor + prediction",
            "log_difference": "observable_anchor * exp(prediction); conditional median",
        }[recipe],
    }


def _latest_contiguous(values):
    """Trim unpublished trailing periods without joining across internal gaps."""
    values = np.asarray(values, dtype=float)
    available = np.flatnonzero(np.isfinite(values))
    if not len(available):
        return values[:0]
    observed = values[: available[-1] + 1]
    gaps = np.flatnonzero(~np.isfinite(observed))
    return observed[gaps[-1] + 1 :] if len(gaps) else observed


class _Panel:
    def __init__(self, dataset, request, calendar):
        self.calendar = calendar
        self.series = [request.target_id, *sorted(request.candidate_ids)]
        self.index = {name: i for i, name in enumerate(self.series)}
        self.snapshots = np.full((len(calendar), len(calendar), len(self.series)), np.nan)
        self.truth = np.full(len(calendar), np.nan)
        data = dataset.observations.copy()
        data["period"] = pd.to_datetime(data.period)
        data["available_at"] = pd.to_datetime(data.available_at, utc=True).dt.tz_localize(None)
        for j, series in enumerate(self.series):
            rows = data.loc[data.series_id == series].sort_values("available_at")
            if j == 0:
                first = rows.drop_duplicates("period", keep="first").set_index("period").value
                self.truth = first.reindex(calendar).to_numpy(dtype=float)
            for i, period in enumerate(calendar):
                known = rows.loc[
                    (rows.available_at <= period.to_period("M").end_time) & (rows.period <= period)
                ]
                latest = known.drop_duplicates("period", keep="last").set_index("period").value
                self.snapshots[i, :, j] = latest.reindex(calendar).to_numpy(dtype=float)

    def feature(self, origin, feature, lag, recipe):
        period = origin - lag
        if period < 0:
            return np.nan
        current = self.snapshots[origin, period, self.index[feature]]
        if recipe == "identity":
            return current
        if period == 0:
            return np.nan
        previous = self.snapshots[origin, period - 1, self.index[feature]]
        return current - previous


def _fit(panel, origin, features, target_recipe, first_train, min_train):
    historical = np.arange(first_train, origin)
    labels = panel.snapshots[origin, historical + 1, 0]
    anchors = panel.snapshots[origin, historical, 0]
    if target_recipe == "difference":
        labels = labels - anchors
    elif target_recipe == "log_difference":
        if np.any(np.isfinite(labels) & (labels <= 0)) or np.any(
            np.isfinite(anchors) & (anchors <= 0)
        ):
            raise ValueError("nonpositive_log_training_values")
        with np.errstate(divide="ignore", invalid="ignore"):
            labels = np.log(np.where(labels > 0, labels, np.nan)) - np.log(
                np.where(anchors > 0, anchors, np.nan)
            )
    x = features[historical]
    usable = np.isfinite(labels) & np.isfinite(x).all(axis=1)
    if usable.sum() < min_train:
        raise ValueError("insufficient_available_training_pairs")
    x, y = x[usable], labels[usable]
    if not np.isfinite(features[origin]).all():
        raise ValueError("predictor_unavailable_at_origin")
    # Scaling is fitted to this training fold and avoids rank artefacts from units.
    mean, scale = x.mean(axis=0), x.std(axis=0)
    if (scale < 1e-12).any():
        raise ValueError("constant_predictor")
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("singular_design")
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    if not np.isfinite(coefficients).all():
        raise ValueError("nonfinite_coefficients")
    value = float(np.r_[1.0, (features[origin] - mean) / scale] @ coefficients)
    anchor = panel.snapshots[origin, origin, 0]
    if not np.isfinite(anchor):
        raise ValueError("baseline_or_inverse_anchor_unavailable")
    if target_recipe == "difference":
        value += anchor
    elif target_recipe == "log_difference":
        if anchor <= 0:
            raise ValueError("nonpositive_log_inverse_anchor")
        with np.errstate(over="ignore"):
            value = float(anchor * np.exp(value))
    if not np.isfinite(value):
        raise ValueError("nonfinite_forecast")
    correlation = design[:, 1:].T @ design[:, 1:] / len(x)
    max_vif = float(np.max(np.diag(np.linalg.inv(correlation))))
    residuals = y - design @ coefficients
    if not np.isfinite(residuals).all() or not np.isfinite(max_vif):
        raise ValueError("nonfinite_fit_statistics")
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(residuals @ residuals) / total if total > 1e-20 else None
    adjusted = (
        (1 - (1 - r2) * (len(y) - 1) / (len(y) - design.shape[1])) if r2 is not None else None
    )
    raw = coefficients[1:] / scale
    raw_intercept = float(coefficients[0] - raw @ mean)
    return (
        value,
        float(anchor),
        {"max_vif": max_vif, "train_r2": r2, "train_adjusted_r2": adjusted},
        {
            "intercept": raw_intercept,
            "slopes": raw.tolist(),
            "residuals": residuals,
            "design": design,
            "training_pairs": len(y),
            "omitted_training_pairs": int((~usable).sum()),
            "training_origin_indices": historical[usable],
        },
    )


def _scores(predictions, prefix):
    actual = np.array([row["actual"] for row in predictions])
    predicted = np.array([row["predicted"] for row in predictions])
    baseline = np.array([row["baseline"] for row in predictions])
    error = actual - predicted
    total = np.sum((actual - actual.mean()) ** 2)
    metrics = {
        f"{prefix}_mae": float(np.abs(error).mean()),
        f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_r2": float(1 - np.sum(error**2) / total) if total > 1e-20 else None,
        f"{prefix}_mape": float(np.mean(np.abs(error / actual)) * 100)
        if np.all(actual > 1e-8)
        else None,
    }
    metrics["baseline_mae" if prefix == "development" else "holdout_baseline_mae"] = float(
        np.abs(actual - baseline).mean()
    )
    if prefix == "development":
        blocks = [np.abs(error[i : i + 12]).mean() for i in range(0, len(error), 12)]
        metrics["block_mae_std"] = float(np.std(blocks))
    return metrics


def _prediction(panel, origin, predicted, baseline, split):
    truth = panel.truth[origin + 1]
    if not np.isfinite(truth):
        raise ValueError("evaluation_outcome_unavailable")
    return {
        "origin": panel.calendar[origin].to_period("M").end_time.date().isoformat(),
        "period": panel.calendar[origin + 1].date().isoformat(),
        "actual": float(truth),
        "predicted": predicted,
        "baseline": baseline,
        "split": split,
    }


def _diagnostics(fit):
    residuals, design = fit["residuals"], fit["design"]
    calendar_regular = bool(np.all(np.diff(fit["training_origin_indices"]) == 1))
    if np.ptp(residuals) < 1e-12:
        return {
            "scope": "last_development_training_fold_transformed_response",
            "status": "constant_residuals_tests_unavailable",
            "training_pairs": fit["training_pairs"],
            "omitted_training_pairs": fit["omitted_training_pairs"],
        }
    diagnostics = {
        "scope": "last_development_training_fold_transformed_response",
        "interpretation": "Post-selection descriptive diagnostics, not causal evidence.",
        "residual_adf": _adf(residuals)
        if calendar_regular
        else {"pvalue": None, "status": "unavailable_gapped_training_calendar"},
        "durbin_watson": float(durbin_watson(residuals)) if calendar_regular else None,
        "temporal_tests_status": "available"
        if calendar_regular
        else "unavailable_gapped_training_calendar",
        "shapiro_pvalue": float(stats.shapiro(residuals).pvalue),
        "training_pairs": fit["training_pairs"],
        "omitted_training_pairs": fit["omitted_training_pairs"],
    }
    try:
        diagnostics["breusch_pagan_pvalue"] = float(het_breuschpagan(residuals, design)[1])
        diagnostics["ljung_box_pvalue"] = (
            float(
                acorr_ljungbox(
                    residuals, lags=[min(12, len(residuals) // 5)], return_df=True
                ).lb_pvalue.iloc[0]
            )
            if calendar_regular
            else None
        )
    except (ValueError, np.linalg.LinAlgError):
        diagnostics["diagnostic_warning"] = "Some residual tests unavailable."
    return _finite_tree(diagnostics)


def _finite_tree(value):
    """Undefined diagnostic numbers are explicit nulls, never JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {key: _finite_tree(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _front(models):
    keys = _POLICY["pareto_minimize"]
    for model in models:
        a = np.array([model.metrics[key] for key in keys])
        dominated = False
        for other in models:
            if other is model:
                continue
            b = np.array([other.metrics[key] for key in keys])
            tolerance = 1e-12 + 1e-9 * np.maximum(np.abs(a), np.abs(b))
            if np.all(b <= a + tolerance) and np.any(b < a - tolerance):
                dominated = True
                break
        model.is_pareto = not dominated
    return [model for model in models if model.is_pareto]


def run_analysis(
    dataset: Dataset,
    request: AnalysisRequest,
    experiment_id: str | None = None,
    *,
    run_mode: str = "full",
) -> AnalysisResult:
    """Evaluate development models, optionally audit one frozen recommendation."""
    if run_mode not in {"preliminary", "full"}:
        raise ValueError("Unsupported execution mode")
    result = AnalysisResult(
        experiment_id=experiment_id or str(uuid4()),
        request=request,
        dataset_name=dataset.name,
        provenance=dataset.provenance,
        status="blocked",
        policy_version=_POLICY["version"],
        run_mode=run_mode,
    )

    def decide(rule, outcome, explanation, **evidence):
        result.decisions.append(
            Decision(rule_id=rule, outcome=outcome, explanation=explanation, evidence=evidence)
        )

    work = estimate_work(dataset, request)
    decide(
        "D03",
        "accepted" if work["within_budget"] else "blocked",
        "The complete candidate/lag grid and all forecast fits are budgeted.",
        **work,
    )
    if work["reason"]:
        result.explanation = work["reason"]
        if work["reason"] == "insufficient_history":
            decide("D04", "blocked", "Not enough monthly history for the requested split.", **work)
        return result
    required = {request.target_id, *request.candidate_ids}
    present = set(dataset.observations.series_id)
    if not required.issubset(present):
        result.explanation = "unknown_series_ids: " + ", ".join(sorted(required - present))
        decide("D04", "blocked", result.explanation)
        return result
    calendar = _calendar(dataset, request)
    start = work["first_development_index"]
    end = len(calendar) - 1 - request.holdout_periods
    if run_mode == "preliminary":
        # Remove holdout outcomes and later revisions before constructing any
        # feature snapshots. Merely skipping the final scoring loop is not enough.
        cutoff = calendar[end].to_period("M").end_time
        observations = dataset.observations.loc[
            (dataset.observations.period <= calendar[end])
            & (dataset.observations.available_at <= cutoff)
        ].copy()
        panel = _Panel(
            Dataset(observations, dataset.catalog, dataset.name, dataset.provenance),
            request,
            calendar[: end + 1],
        )
    else:
        panel = _Panel(dataset, request, calendar)
    development, holdout = range(start, end), range(end, len(calendar) - 1)
    first_train = max(request.lag_menu) + 1
    recipes = {}
    for series in panel.series:
        stop = start + 1 if series == request.target_id else start
        values = panel.snapshots[start, :stop, panel.index[series]]
        # Never bridge missing months: ADF needs a contiguous training segment.
        contiguous = _latest_contiguous(values)
        mode = (
            request.target_transform if series == request.target_id else request.feature_transform
        )
        recipes[series] = _recipe(contiguous, mode)
    target_recipe = recipes[request.target_id]["recipe"]
    result.split = {
        "initial_train": request.initial_train,
        "development_origins": len(development),
        "holdout_origins": len(holdout),
        "development_start": calendar[start].date().isoformat(),
        "development_end": calendar[end - 1].date().isoformat(),
        "holdout_start": calendar[end].date().isoformat(),
        "holdout_end": calendar[-2].date().isoformat(),
        "origin_convention": "Calendar month-end; labels are next month.",
        "truth_vintage": "first_release",
        "holdout_used_for_selection": False,
        "estimated_fits": work["estimated_fits"],
    }
    result.warnings.extend(
        [
            "Associations and coefficients do not establish causation.",
            "Cointegration/ECM, seasonal model families and prediction intervals are deferred beyond sprint 6.",
            "This demo does not implement live monitoring or automatic recalibration.",
            "Reusing this holdout in edited experiments is exploratory; fresh data are needed for confirmation.",
        ]
    )
    if "synthetic" in dataset.provenance or "synthetic" in dataset.name.lower():
        result.warnings.append(
            "Synthetic data demonstrate behavior, not real-world predictive validity."
        )
    if "latest" in dataset.provenance:
        result.warnings.append(
            "Latest-vintage inputs cannot substantiate a historical real-time backtest."
        )
    for series, recipe in recipes.items():
        if recipe["after"]["status"] != "stationary_evidence":
            result.warnings.append(
                f"{series}: stationarity after {recipe['recipe']} is unconfirmed; "
                "results require review, including possible cointegration or I(2)."
            )
    decide(
        "D04",
        "accepted",
        "Monthly inputs are selected as of each forecast origin; no backfill.",
        provenance=dataset.provenance,
        truth_vintage="first_release",
    )
    decide(
        "D05",
        "frozen",
        "Freeze common origins, recipes, lag grid and holdout before scoring.",
        split=result.split,
        policy_version=result.policy_version,
    )
    decide(
        "D20",
        "frozen",
        "ADF-guided recipes use the initial training prefix only; inverse forecasts retain Y units.",
        recipes=recipes,
        recipe_cutoff=calendar[start].to_period("M").end_time.isoformat(),
        auto_policy="ADF(c, AIC, maximum 12 lags); otherwise first difference",
        alpha=_POLICY["stationarity_alpha"],
    )
    decide(
        "D19",
        "frozen_grid",
        "Each external series uses one lag; lag zero must already be published.",
        lag_menu=request.lag_menu,
        convention="X(t-L) predicts Y(t+1)",
    )
    feature_cache = {}
    for series in request.candidate_ids:
        for lag in request.lag_menu:
            feature_cache[series, lag] = np.array(
                [
                    panel.feature(i, series, lag, recipes[series]["recipe"])
                    for i in range(len(panel.calendar))
                ]
            )
    fit_cache = {}
    for spec in enumerate_specs(request):
        model = ModelResult(
            **spec,
            transformations={key: recipes[key] for key in [request.target_id, *spec["features"]]},
        )
        features = np.column_stack(
            [feature_cache[series, spec["lags"][series]] for series in spec["features"]]
        )
        vif = 0.0
        try:
            for origin in development:
                predicted, baseline, train_metrics, fit = _fit(
                    panel,
                    origin,
                    features,
                    target_recipe,
                    first_train,
                    max(request.initial_train, 10 * (len(spec["features"]) + 1)),
                )
                model.predictions.append(
                    _prediction(panel, origin, predicted, baseline, "development")
                )
                vif = max(vif, train_metrics["max_vif"])
            model.metrics = {
                **_scores(model.predictions, "development"),
                **train_metrics,
                "max_vif": vif,
            }
            if any(
                value is not None and not np.isfinite(value) for value in model.metrics.values()
            ):
                model.metrics = {}
                raise ValueError("nonfinite_evaluation_metric")
            model.coefficients = {
                "intercept": fit["intercept"],
                **dict(zip(spec["features"], fit["slopes"], strict=True)),
            }
            fit_cache[model.model_id] = fit
        except (ValueError, np.linalg.LinAlgError) as exc:
            model.status = "ineligible"
            model.reason = str(exc)
            # Partial predictions remain visible, but never get comparable scores.
        result.models.append(model)
    eligible = [model for model in result.models if model.status == "eligible"]
    decide(
        "D09",
        "completed",
        "All specifications evaluated; any failed origin disqualifies its candidate.",
        eligible=len(eligible),
        rejected=len(result.models) - len(eligible),
        reasons={model.model_id: model.reason for model in result.models if model.reason},
    )
    front = _front(eligible)
    decide(
        "D10",
        "pareto_front" if front else "empty",
        "Minimize development MAE, worst fold VIF and block-MAE variability.",
        model_ids=[model.model_id for model in front],
    )
    if not front:
        result.status = "no_eligible_models"
        result.explanation = (
            "No candidate passed every common development origin. Review recorded failures."
        )
        return result
    for model in front:
        model.diagnostics = _diagnostics(fit_cache[model.model_id])
    best_mae = min(model.metrics["development_mae"] for model in front)
    shortlist = [
        model
        for model in front
        if model.metrics["development_mae"]
        <= best_mae * (1 + _POLICY["champion_shortlist_relative_mae_tolerance"]) + 1e-12
    ]
    chosen = min(
        shortlist,
        key=lambda model: (model.metrics["block_mae_std"], len(model.features), model.model_id),
    )
    result.recommended_id = chosen.model_id
    decide(
        "D12",
        "recommendation_frozen",
        "Among Pareto models within 5% of best development MAE, prefer stability, simplicity, then stable ID.",
        model_id=chosen.model_id,
        shortlist=[model.model_id for model in shortlist],
        development_mae=chosen.metrics["development_mae"],
        shortlist_relative_tolerance=_POLICY["champion_shortlist_relative_mae_tolerance"],
    )
    if run_mode == "preliminary":
        result.status = "preliminary_completed"
        result.explanation = (
            "Development-only comparison saved. The recommendation is exploratory; "
            "the holdout was not accessed and no champion or next forecast was issued."
        )
        result.split["holdout_evaluated"] = False
        decide(
            "D13", "not_requested", "Preliminary execution excludes the holdout and next forecast."
        )
        return result
    features = np.column_stack(
        [feature_cache[series, chosen.lags[series]] for series in chosen.features]
    )
    audit = []
    try:
        for origin in holdout:
            predicted, baseline, _, _ = _fit(
                panel,
                origin,
                features,
                target_recipe,
                first_train,
                max(request.initial_train, 10 * (len(chosen.features) + 1)),
            )
            audit.append(_prediction(panel, origin, predicted, baseline, "holdout"))
        chosen.predictions.extend(audit)
        chosen.metrics.update(_scores(audit, "holdout"))
        beats_baseline = (
            chosen.metrics["development_mae"] < chosen.metrics["baseline_mae"]
            and chosen.metrics["holdout_mae"] < chosen.metrics["holdout_baseline_mae"]
        )
        unsupported_series = [
            series
            for series in [request.target_id, *chosen.features]
            if recipes[series]["after"]["status"] != "stationary_evidence"
        ]
        provenance_supported = dataset.provenance in {
            "synthetic_point_in_time",
            "verified_point_in_time",
        } or dataset.provenance.startswith("synthetic;")
        if beats_baseline and provenance_supported and not unsupported_series:
            result.champion_id = chosen.model_id
        result.status = "completed" if result.champion_id else "no_qualified_champion"
        result.explanation = (
            "The frozen recommendation beat the persistence baseline on development and holdout. "
            "This is a heuristic eligibility check, not proof of significance."
            if result.champion_id
            else "The recommendation remains available for exploration, but did not qualify as champion under the baseline, provenance and stationarity audit."
        )
        decide(
            "D13",
            "passed" if result.champion_id else "not_qualified",
            "Only the frozen recommendation was audited; holdout scores never select a replacement.",
            model_id=chosen.model_id,
            holdout_mae=chosen.metrics["holdout_mae"],
            holdout_baseline_mae=chosen.metrics["holdout_baseline_mae"],
            stationarity_unconfirmed_series=unsupported_series,
            provenance_supported=provenance_supported,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        chosen.predictions.extend(audit)
        result.status = "no_qualified_champion"
        result.explanation = (
            f"The frozen recommendation could not complete its holdout audit: {exc}"
        )
        decide("D13", "audit_unavailable", result.explanation)
    try:
        origin = len(calendar) - 1
        predicted, baseline, _, fit = _fit(
            panel,
            origin,
            features,
            target_recipe,
            first_train,
            max(request.initial_train, 10 * (len(chosen.features) + 1)),
        )
        result.next_forecast = {
            "model_id": chosen.model_id,
            "origin": calendar[origin].to_period("M").end_time.date().isoformat(),
            "period": (calendar[origin] + pd.offsets.MonthBegin(1)).date().isoformat(),
            "predicted": predicted,
            "baseline": baseline,
            "unit": next(
                (item.unit for item in dataset.catalog if item.id == request.target_id), "units"
            ),
            "target_transform": target_recipe,
            "prediction_interval": None,
            "estimand": "conditional_median"
            if target_recipe == "log_difference"
            else "conditional_mean",
            "training_pairs": fit["training_pairs"],
            "coefficients": {
                "intercept": fit["intercept"],
                **dict(zip(chosen.features, fit["slopes"], strict=True)),
            },
        }
        later_evidence = {}
        for series in [request.target_id, *chosen.features]:
            values = panel.snapshots[origin, : origin + 1, panel.index[series]]
            contiguous = _latest_contiguous(values)
            later_evidence[series] = _adf(_transform(contiguous, recipes[series]["recipe"]))
            if later_evidence[series]["status"] != "stationary_evidence":
                result.warnings.append(
                    f"{series}: final training-window stationarity remains unconfirmed; "
                    "the frozen recipe was retained and requires review."
                )
        result.next_forecast["stationarity_reassessment"] = later_evidence
    except (ValueError, np.linalg.LinAlgError) as exc:
        result.warnings.append(f"Next forecast unavailable: {exc}")
    return result
