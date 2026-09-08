"""Monthly inputs with explicit publication times and immutable data snapshots."""

import json
import re
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
import requests

from manto.domain import Dataset, SeriesInfo

REQUIRED_COLUMNS = ["series_id", "period", "available_at", "value"]
SNAPSHOT_SCHEMA = "manto.monthly.v1"


def validate_dataset(dataset: Dataset) -> Dataset:
    """Return a normalized copy, retaining publication vintages and missing months.

    Values are never interpolated. Availability before the observation month is
    rejected; revisions with different publication times remain separate rows.
    """
    frame = dataset.observations.copy()
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing observation columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("The dataset contains no observations")
    if not frame.columns.is_unique:
        raise ValueError("Observation column names must be unique")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required observation values must not be missing")
    if not frame.series_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("series_id must contain nonempty strings")
    if (frame.series_id != frame.series_id.str.strip()).any():
        raise ValueError("series_id must not have surrounding whitespace")
    try:
        # UTC conversion preserves the instant for timezone-aware release dates.
        frame["period"] = pd.to_datetime(frame.period, format="mixed", utc=True).dt.tz_localize(
            None
        )
        frame["available_at"] = pd.to_datetime(
            frame.available_at, format="mixed", utc=True
        ).dt.tz_localize(None)
        frame["value"] = pd.to_numeric(frame.value, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid dates or nonnumeric observation values") from exc
    if frame[["period", "available_at"]].isna().any().any():
        raise ValueError("Observation dates must not be missing")
    if not np.isfinite(frame.value).all():
        raise ValueError("Observation values must be finite")
    if not (frame.period == frame.period.dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("period must be a month-start date at midnight UTC")
    if (frame.available_at < frame.period).any():
        raise ValueError("available_at cannot precede the observation month")
    if frame.duplicated(["series_id", "period", "available_at"]).any():
        raise ValueError("Ambiguous duplicate series/period/available_at observation")
    catalog_ids = [item.id for item in dataset.catalog]
    if len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError("Catalog IDs must be unique")
    if set(frame.series_id) != set(catalog_ids):
        raise ValueError("Catalog IDs must exactly match the observed series")
    frame = frame.reindex(
        REQUIRED_COLUMNS + sorted(set(frame.columns) - set(REQUIRED_COLUMNS)), axis=1
    )
    frame = frame.sort_values(["series_id", "period", "available_at"]).reset_index(drop=True)
    return Dataset(frame, list(dataset.catalog), dataset.name, dataset.provenance)


def demo_dataset(seed: int = 42, periods: int = 156) -> Dataset:
    """Generate fictional business sales and ten fictional stationary predictors.

    Monthly sales changes use the *previous month's* inflation, demand, and
    marketing activity. The relationship is a simulation, not economic evidence.
    """
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 3:
        raise ValueError("periods must be an integer of at least three months")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", periods=periods, freq="MS")
    predictor_names = [
        ("inflation", "Inflation", "percentage points", 3.0, 0.8),
        ("demand", "Demand index", "index", 100.0, 3.0),
        ("marketing", "Marketing activity", "index", 50.0, 2.0),
        ("interest_rate", "Interest rate", "percent", 4.0, 0.4),
        ("unemployment", "Unemployment", "percent", 6.0, 0.5),
        ("exchange_rate", "Exchange rate", "PLN per currency unit", 4.2, 0.1),
        ("consumer_confidence", "Consumer confidence", "index", 100.0, 4.0),
        ("industrial_output", "Industrial output growth", "percent", 2.0, 1.0),
        ("energy_price", "Energy price index", "index", 100.0, 5.0),
        ("wage_growth", "Wage growth", "percent", 5.0, 1.0),
    ]
    predictors = {}
    for series_id, _, _, mean, scale in predictor_names:
        noise = rng.normal(0.0, scale, periods)
        ar = np.empty(periods)
        ar[0] = noise[0]
        for index in range(1, periods):
            ar[index] = 0.45 * ar[index - 1] + noise[index]
        predictors[series_id] = mean + ar
    sales = np.empty(periods)
    sales[0] = 1000.0
    for index in range(1, periods):
        sales[index] = sales[index - 1] + (
            8.0
            - 3.0 * (predictors["inflation"][index - 1] - 3.0)
            + 2.0 * (predictors["demand"][index - 1] - 100.0)
            + 1.5 * (predictors["marketing"][index - 1] - 50.0)
            + rng.normal(0.0, 1.5)
        )
    # Preserve positivity even for unusually adverse seeds without changing increments.
    sales += max(0.0, 1.0 - sales.min())
    values = {"sales": sales, **predictors}
    catalog = [
        SeriesInfo(
            id="sales",
            title="Fictional monthly sales",
            unit="thousand PLN",
            source="synthetic",
            role="target",
            description="Simulated sales; no real company or economic observations.",
        )
    ] + [
        SeriesInfo(
            id=series_id,
            title=f"Fictional {title.lower()}",
            unit=unit,
            source="synthetic",
            description="Seeded stationary simulation; no economic interpretation is implied.",
        )
        for series_id, title, unit, _, _ in predictor_names
    ]
    observations = pd.concat(
        [
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "period": dates,
                    "available_at": dates.to_period("M").end_time,
                    "value": series_values,
                }
            )
            for series_id, series_values in values.items()
        ],
        ignore_index=True,
    )
    return validate_dataset(
        Dataset(
            observations,
            catalog,
            "Synthetic monthly sales demo",
            f"synthetic; seed={seed}; periods={periods}; release=month_completion; "
            "sales_change[t+1]=8-3*(inflation[t]-3)+2*(demand[t]-100)"
            "+1.5*(marketing[t]-50)+noise; not_real_economic_data",
        )
    )


def load_csv(source: str | Path | IO) -> Dataset:
    """Read long-form CSV; publication dates are mandatory and never guessed."""
    frame = pd.read_csv(source, dtype={"series_id": str})
    if "series_id" not in frame or frame.empty:
        raise ValueError("CSV must contain series_id and observations")
    catalog = []
    for index, (series_id, group) in enumerate(
        frame.groupby("series_id", sort=False, dropna=False)
    ):
        metadata = {}
        for field in ("title", "unit"):
            if field in group:
                unique = group[field].dropna().astype(str).unique()
                if len(unique) > 1:
                    raise ValueError(f"Inconsistent {field} for series {series_id}")
                if len(unique):
                    metadata[field] = unique[0]
        if not isinstance(series_id, str) or not series_id.strip():
            raise ValueError("series_id must contain nonempty strings")
        catalog.append(
            SeriesInfo(
                id=series_id,
                title=metadata.get("title", series_id),
                unit=metadata.get("unit", "units"),
                source="user_csv",
                role="target" if index == 0 else "predictor",
            )
        )
    return validate_dataset(
        Dataset(
            frame,
            catalog,
            "Uploaded monthly data",
            "user_csv; user_supplied_availability",
        )
    )


def _snapshot_bytes(dataset: Dataset) -> bytes:
    dataset = validate_dataset(dataset)
    frame = dataset.observations.copy()
    for column in ("period", "available_at"):
        frame[column] = frame[column].map(lambda value: value.isoformat())
    frame = frame.reindex(sorted(frame.columns), axis=1)
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "name": dataset.name,
        "provenance": dataset.provenance,
        "catalog": [
            item.model_dump() for item in sorted(dataset.catalog, key=lambda item: item.id)
        ],
        "observations": frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"),
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def snapshot_dataset(dataset: Dataset, directory: str | Path) -> str:
    """Write an immutable canonical snapshot; its SHA-256 is its filename."""
    content = _snapshot_bytes(dataset)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sha256(content).hexdigest()}.json"
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError("Existing snapshot content does not match its digest")
    return str(path.resolve())


def load_snapshot(path: str | Path) -> Dataset:
    """Validate integrity and schema before returning a stored data vintage."""
    path = Path(path)
    content = path.read_bytes()
    if path.stem != sha256(content).hexdigest():
        raise ValueError("Snapshot integrity verification failed")
    payload = json.loads(content)
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported snapshot schema")
    return validate_dataset(
        Dataset(
            pd.DataFrame(payload["observations"]),
            [SeriesInfo.model_validate(item) for item in payload["catalog"]],
            payload["name"],
            payload["provenance"],
        )
    )


def fetch_nbp_monthly(code: str, start: str, end: str) -> Dataset:
    """Fetch completed-month mean table-A mid FX rates, in PLN per currency unit.

    Only full requested months are included. Daily publication dates are supplied
    by NBP; the completed aggregate is conservatively available at month-end.
    API documentation: https://api.nbp.pl/en.html (93-day request limit).
    """
    if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z]{3}", code):
        raise ValueError("Currency code must contain exactly three ASCII letters")
    try:
        first, last = pd.Timestamp(start), pd.Timestamp(end)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid NBP date range") from exc
    if pd.isna(first) or pd.isna(last) or first.tzinfo or last.tzinfo or first > last:
        raise ValueError("NBP dates must be ordered, timezone-naive dates")
    first = first.normalize()
    if first.day != 1:
        first = first + pd.offsets.MonthBegin(1)
    last = last.normalize()
    if not last.is_month_end:
        last = last - pd.offsets.MonthEnd(1)
    last_completed = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M").start_time
    last = min(last, last_completed - timedelta(days=1))
    if first > last:
        raise ValueError("Date range contains no completed full months")
    rows = []
    cursor = first
    while cursor <= last:
        batch_end = min(cursor + timedelta(days=92), last)
        url = (
            f"https://api.nbp.pl/api/exchangerates/rates/a/{code.lower()}/"
            f"{cursor:%Y-%m-%d}/{batch_end:%Y-%m-%d}/"
        )
        try:
            response = requests.get(url, params={"format": "json"}, timeout=(5, 30))
            if response.status_code != 404:
                response.raise_for_status()
                batch = response.json()["rates"]
                for item in batch:
                    date = pd.Timestamp(item["effectiveDate"])
                    value = float(item["mid"])
                    if date.tzinfo or not cursor <= date <= batch_end or not np.isfinite(value):
                        raise ValueError("Invalid date or value in NBP response")
                    rows.append({"date": date, "value": value})
        except requests.RequestException as exc:
            raise ValueError(
                f"NBP request failed for {cursor:%Y-%m-%d} to {batch_end:%Y-%m-%d}"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid NBP response") from exc
        cursor = batch_end + timedelta(days=1)
    if not rows:
        raise ValueError(f"No NBP table-A data found for {code.upper()}")
    daily = pd.DataFrame(rows)
    if daily.date.duplicated().any():
        raise ValueError("NBP returned duplicate publication dates")
    monthly = daily.groupby(daily.date.dt.to_period("M"))["value"].mean()
    expected = pd.period_range(first, last, freq="M")
    if not monthly.index.equals(expected.rename("date")):
        raise ValueError("NBP response is missing one or more requested months")
    series_id = f"nbp_{code.lower()}_pln"
    frame = pd.DataFrame(
        {
            "series_id": series_id,
            "period": monthly.index.start_time,
            "available_at": monthly.index.end_time,
            "value": monthly.to_numpy(),
        }
    )
    return validate_dataset(
        Dataset(
            frame,
            [
                SeriesInfo(
                    id=series_id,
                    title=f"NBP {code.upper()}/PLN monthly mean",
                    unit=f"PLN per {code.upper()}",
                    source="NBP table A",
                    description="Arithmetic mean of published daily mid rates.",
                )
            ],
            f"NBP {code.upper()}/PLN",
            "NBP_public_API; https://api.nbp.pl; table=A; "
            "aggregation=mean_published_daily_mid; availability=completed_month_end; "
            "full_months_only; current_archive_not_historical_vintages",
        )
    )
