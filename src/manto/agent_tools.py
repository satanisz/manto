"""Bounded read tools and a canonical pre-execution specification report."""

import json

import pandas as pd

from manto.analysis import _calendar, estimate_work
from manto.domain import Dataset
from manto.drafts import AnalysisDraft, digest
from manto.policy import load_policy
from manto.providers import redact


def dataset_digest(dataset: Dataset) -> str:
    return digest(
        {
            "data": dataset.observations.to_csv(index=False),
            "catalog": [
                entry.model_dump() for entry in sorted(dataset.catalog, key=lambda item: item.id)
            ],
            "provenance": dataset.provenance,
        }
    )


def catalog_facts(dataset: Dataset, draft: AnalysisDraft) -> dict:
    target = draft.values["target_id"]
    selected = draft.values["candidate_ids"] or []
    # Never pass the synthetic generator's provenance/formula or private observations.
    entries = [
        {
            "id": item.id,
            "title": redact(item.title),
            "unit": redact(item.unit),
            "selected": item.id in selected,
        }
        for item in dataset.catalog
        if item.id != target
    ]
    return {
        "available_count": len(entries),
        "selected_count": len(selected),
        "entries": entries,
        "targets": [
            {"id": item.id, "title": redact(item.title), "unit": redact(item.unit)}
            for item in dataset.catalog
        ],
    }


def validate_catalog(dataset: Dataset, draft: AnalysisDraft) -> None:
    ids = {item.id for item in dataset.catalog}
    values = draft.values
    target = values["target_id"]
    if target is not None and target not in ids:
        raise ValueError("The target is not in this catalog.")
    for key in ("candidate_ids", "pinned_ids"):
        selected = values[key] or []
        if len(selected) != len(set(selected)) or not set(selected).issubset(ids):
            raise ValueError(f"{key}: select unique available catalog IDs.")
        if target in selected:
            raise ValueError("The target cannot be an external predictor.")
    if any(lag < 0 or lag > 12 for lag in values["lag_menu"]) or not values["lag_menu"]:
        raise ValueError("Choose monthly lags from 0 through 12.")


def training_evidence(dataset: Dataset, draft: AnalysisDraft, ids: list[str]) -> dict:
    """A fixed 24-month prefix remains inside every supported initial training window.

    No later values influence the payload, hash, sampling, or summaries. All
    revisions are selected as of the prefix cutoff. Vectors are separately opt-in.
    """
    target = draft.values["target_id"]
    known = {item.id: item for item in dataset.catalog}
    if target not in known or not ids or len(ids) > 20 or not set(ids).issubset(known):
        raise ValueError("Choose a target and at most 20 available evidence series.")
    frame = dataset.observations
    months = frame.loc[frame.series_id == target, "period"]
    start = pd.Timestamp(months.min())
    stop = start + pd.offsets.MonthBegin(24)
    cutoff = (start + pd.offsets.MonthBegin(23)).to_period("M").end_time
    subset = (
        frame.loc[
            frame.series_id.isin(ids)
            & (frame.period >= start)
            & (frame.period < stop)
            & (frame.available_at <= cutoff)
        ]
        .sort_values("available_at")
        .drop_duplicates(["series_id", "period"], keep="last")
    )
    calendar = pd.date_range(start, periods=24, freq="MS")
    records = []
    for series_id in ids:
        rows = subset.loc[subset.series_id == series_id].sort_values("period")
        series = rows.set_index("period").value.reindex(calendar)
        changes = series.diff()
        acf = {}
        for lag in (1, 2, 3, 6, 12):
            pairs = pd.concat([changes, changes.shift(lag)], axis=1).dropna()
            value = pairs.iloc[:, 0].corr(pairs.iloc[:, 1]) if len(pairs) >= 8 else None
            acf[str(lag)] = float(value) if value is not None and pd.notna(value) else None
        record = {
            "series_id": series_id,
            "unit": known[series_id].unit,
            "observed_months": len(rows),
            "missing_months": 24 - len(rows),
            "change_autocorrelation": acf,
            "interpretation": "Descriptive first-difference autocorrelation, not a causal lag test.",
            "transformation": "first_difference_for_summary_only",
        }
        if draft.values["share_vectors"] and draft.settings["share_vectors"].status == "confirmed":
            record["vector"] = json.loads(
                rows[["period", "available_at", "value"]].to_json(
                    orient="records", date_format="iso"
                )
            )
            record["vector_transformation"] = "original_values"
        records.append(record)
    payload = {
        "scope": "first_24_calendar_months_only",
        "cutoff": cutoff.isoformat(),
        "availability": "latest_revision_available_at_cutoff; no interpolation",
        "source": "local_input_snapshot_prefix",
        "truncation": "later_months_excluded",
        "series": records,
    }
    payload["evidence_id"] = digest(payload)
    return payload


def specification_report(dataset: Dataset, draft: AnalysisDraft, *, exposure=False) -> dict:
    policy = load_policy()
    errors = []
    request = None
    work = {}
    dates = {}
    try:
        validate_catalog(dataset, draft)
        request = draft.request()
        work = estimate_work(dataset, request)
        if work["reason"]:
            errors.append(work["reason"])
        calendar = _calendar(dataset, request)
        start = work["first_development_index"]
        end = len(calendar) - 1 - request.holdout_periods
        if 0 <= start < end < len(calendar):
            dates = {
                "development_outcomes": [
                    str(calendar[start + 1].date()),
                    str(calendar[end].date()),
                ],
                "holdout_outcomes": [str(calendar[end + 1].date()), str(calendar[-1].date())],
            }
    except ValueError as exc:
        errors.append(redact(str(exc)))
    payload = {
        "schema_version": 1,
        "target": next(
            (
                {"id": item.id, "title": item.title, "unit": item.unit}
                for item in dataset.catalog
                if item.id == draft.values["target_id"]
            ),
            None,
        ),
        "revision": draft.revision,
        "draft": draft.model_dump(),
        "dataset_hash": dataset_digest(dataset),
        "policy_hash": digest(policy),
        "policy_version": policy["version"],
        "request": request.model_dump() if request else None,
        "ready": not errors,
        "errors": errors,
        "work": work,
        "dates": dates,
        "holdout_exposed": exposure,
        "contract": "Monthly Y(t+1), original units, expanding windows, persistence baseline. "
        "X(t-L) must be published at origin t. One lag per external series. "
        "Auto transforms are fitted on initial training evidence; recipes are reported after fitting.",
        "scope": "Linear OLS only; preliminary skips holdout and next forecast. "
        "Full uses existing diagnostics, not ECM, seasonality, or monitoring.",
        "data_sharing": "Gemini receives messages, catalog metadata and bounded summaries; "
        "numerical vectors only when share_vectors is explicitly confirmed true. "
        "Langfuse receives metadata only.",
    }
    payload["report_id"] = digest(payload)
    return payload


def render_specification(report: dict) -> str:
    """Canonical text shared by chat, download and terminal printing."""
    lines = [
        "## Analysis specification",
        f"Report: `{report['report_id']}`",
        f"Target: {json.dumps(report.get('target'), ensure_ascii=False)}",
        f"Status: {'READY FOR APPROVAL' if report['ready'] else 'DRAFT — NOT EXECUTABLE'}",
    ]
    for key, setting in report["draft"]["settings"].items():
        value = json.dumps(setting["value"], ensure_ascii=False)
        lines.append(f"- **{key}**: `{value}` — {setting['status']} ({setting['source']})")
        if setting["rationale"]:
            lines.append(f"  Rationale: {setting['rationale']}")
    lines += [
        "",
        report["contract"],
        report["scope"],
        report["data_sharing"],
        f"Holdout previously exposed: {report['holdout_exposed']}",
        f"Evaluation dates: {json.dumps(report['dates'])}",
        f"Work estimate: {json.dumps(report['work'])}",
        "Issues: " + ("; ".join(report["errors"]) or "None"),
        "Say 'run' to approve this exact report, or request a change."
        if report["ready"]
        else "Discuss and confirm unresolved settings before running.",
    ]
    # No ANSI/control-sequence injection into terminal exports.
    return "\n\n".join(redact(line).replace("\x1b", "") for line in lines)
