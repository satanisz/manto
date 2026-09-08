import json
from datetime import timedelta
from hashlib import sha256
from io import BytesIO, StringIO
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from manto.data import (
    demo_dataset,
    fetch_nbp_monthly,
    load_csv,
    load_snapshot,
    snapshot_dataset,
    validate_dataset,
)


def test_demo_is_reproducible_positive_and_explicitly_synthetic():
    dataset = demo_dataset()
    pd.testing.assert_frame_equal(dataset.observations, demo_dataset().observations)
    assert len(dataset.catalog) == 11
    assert len([item for item in dataset.catalog if item.role == "predictor"]) == 10
    assert {item.source for item in dataset.catalog} == {"synthetic"}
    assert "not_real_economic_data" in dataset.provenance
    wide = dataset.observations.pivot(index="period", columns="series_id", values="value")
    assert (wide.sales > 0).all()
    design = np.column_stack(
        [
            np.ones(len(wide) - 1),
            wide.inflation.iloc[:-1] - 3,
            wide.demand.iloc[:-1] - 100,
            wide.marketing.iloc[:-1] - 50,
        ]
    )
    coefficients = np.linalg.lstsq(design, wide.sales.diff().iloc[1:], rcond=None)[0]
    np.testing.assert_allclose(coefficients, [8, -3, 2, 1.5], atol=0.3)


def test_revisions_and_future_publications_remain_distinct():
    data = load_csv(
        StringIO(
            "series_id,period,available_at,value\n"
            "sales,2024-01-01,2024-01-31T23:59:59Z,100\n"
            "sales,2024-01-01,2024-03-10T12:00:00+01:00,101\n"
            "sales,2024-02-01,2024-03-20,110\n"
        )
    )
    assert len(data.observations) == 3
    assert data.observations.available_at.iloc[1] == pd.Timestamp("2024-03-10 11:00:00")
    known = data.observations[data.observations.available_at <= pd.Timestamp("2024-02-29 23:59:59")]
    assert known.value.tolist() == [100]
    assert known.period.tolist() == [pd.Timestamp("2024-01-01")]


@pytest.mark.parametrize(
    "row,match",
    [
        ("sales,2024-01-02,2024-01-31,100", "month-start"),
        ("sales,2024-01-01,2023-12-31,100", "precede"),
        ("sales,2024-01-01,2024-01-31,inf", "finite"),
        ("sales,2024-01-01,2024-01-31,NaN", "missing"),
        ("sales,2024-01-01,not-a-date,100", "Invalid dates"),
        (",2024-01-01,2024-01-31,100", "nonempty"),
    ],
)
def test_invalid_csv(row, match):
    with pytest.raises(ValueError, match=match):
        load_csv(StringIO("series_id,period,available_at,value\n" + row))


def test_duplicate_vintage_rejected_even_when_values_agree():
    csv = "series_id,period,available_at,value\nsales,2024-01-01,2024-01-31,100\n"
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        load_csv(StringIO(csv + "sales,2024-01-01,2024-01-31,100\n"))


def test_bytes_csv_metadata_and_gaps_preserved():
    data = load_csv(
        BytesIO(
            b"series_id,period,available_at,value,title,unit\n"
            b"margin,2024-01-01,2024-02-15,3.4,Bank margin,percent\n"
            b"margin,2024-03-01,2024-04-15,3.7,Bank margin,percent\n"
            b"rate,2024-01-01,2024-01-31,5.4,Policy rate,percent\n"
        )
    )
    assert data.catalog[0].id == "margin"
    assert data.catalog[0].role == "target"
    assert data.catalog[0].unit == "percent"
    assert len(data.observations) == 3
    assert "user_supplied_availability" in data.provenance


def test_availability_required_and_metadata_consistent():
    with pytest.raises(ValueError, match="available_at"):
        load_csv(StringIO("series_id,period,value\nsales,2024-01-01,10"))
    with pytest.raises(ValueError, match="Inconsistent unit"):
        load_csv(
            StringIO(
                "series_id,period,available_at,value,unit\n"
                "x,2024-01-01,2024-01-31,1,PLN\n"
                "x,2024-02-01,2024-02-29,2,USD"
            )
        )


def test_validation_does_not_mutate_input_and_requires_catalog_match():
    dataset = demo_dataset(periods=3)
    before = dataset.observations.copy()
    normalized = validate_dataset(dataset)
    normalized.observations.loc[0, "value"] = 0
    pd.testing.assert_frame_equal(dataset.observations, before)
    dataset.catalog = dataset.catalog[:-1]
    with pytest.raises(ValueError, match="exactly match"):
        validate_dataset(dataset)


def test_content_addressed_snapshot_stable_roundtrip_and_tamper(tmp_path):
    dataset = demo_dataset(periods=12)
    path = snapshot_dataset(dataset, tmp_path)
    restored = load_snapshot(path)
    pd.testing.assert_frame_equal(dataset.observations, restored.observations)
    assert dataset.provenance == restored.provenance
    assert snapshot_dataset(restored, tmp_path) == path
    dataset.observations = dataset.observations.sample(frac=1, random_state=3)
    dataset.catalog.reverse()
    assert snapshot_dataset(dataset, tmp_path) == path
    # Simulate disk corruption, not a supported editing operation.
    from pathlib import Path

    snapshot = Path(path)
    snapshot.write_bytes(snapshot.read_bytes().replace(b"sales demo", b"sales test"))
    with pytest.raises(ValueError, match="integrity"):
        load_snapshot(path)
    with pytest.raises(ValueError, match="does not match"):
        snapshot_dataset(dataset, tmp_path)


def test_unknown_snapshot_schema_rejected(tmp_path):
    content = json.dumps({"schema": "other"}).encode()
    path = tmp_path / f"{sha256(content).hexdigest()}.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="schema"):
        load_snapshot(path)


def test_nbp_batches_and_aggregates_only_complete_months(monkeypatch):
    calls = []

    def get(url, **kwargs):
        parts = url.rstrip("/").split("/")
        first, last = pd.Timestamp(parts[-2]), pd.Timestamp(parts[-1])
        assert (last - first).days <= 92
        assert kwargs["timeout"] == (5, 30)
        calls.append((first, last))
        rates = [
            {"effectiveDate": date.strftime("%Y-%m-%d"), "mid": float(date.day)}
            for date in pd.date_range(first, last, freq="B")
        ]
        return Mock(status_code=200, json=lambda: {"rates": rates})

    monkeypatch.setattr("manto.data.requests.get", get)
    dataset = fetch_nbp_monthly("EUR", "2024-01-10", "2024-06-15")
    assert len(calls) == 2
    assert calls[0][0] == pd.Timestamp("2024-02-01")
    assert calls[-1][1] == pd.Timestamp("2024-05-31")
    assert calls[1][0] == calls[0][1] + timedelta(days=1)
    assert len(dataset.observations) == 4
    feb_days = pd.date_range("2024-02-01", "2024-02-29", freq="B").day
    assert dataset.observations.value.iloc[0] == np.mean(feb_days)
    assert dataset.observations.available_at.iloc[0] == pd.Period("2024-02").end_time
    assert "aggregation=mean" in dataset.provenance
    assert dataset.catalog[0].role == "predictor"


def test_nbp_no_data_and_timeout(monkeypatch):
    monkeypatch.setattr("manto.data.requests.get", lambda *args, **kwargs: Mock(status_code=404))
    with pytest.raises(ValueError, match="No NBP"):
        fetch_nbp_monthly("USD", "2024-01-01", "2024-01-31")

    def fail(*args, **kwargs):
        raise requests.Timeout("network timeout")

    monkeypatch.setattr("manto.data.requests.get", fail)
    with pytest.raises(ValueError, match="NBP request failed"):
        fetch_nbp_monthly("USD", "2024-01-01", "2024-01-31")


@pytest.mark.parametrize(
    "code,start,end",
    [
        ("../eur", "2024-01-01", "2024-01-31"),
        ("USD", "2024-02-01", "2024-01-01"),
        ("USD", "2024-01-02", "2024-01-30"),
        ("USD", "2099-01-01", "2099-12-31"),
    ],
)
def test_nbp_invalid_requests_never_call_network(monkeypatch, code, start, end):
    get = Mock()
    monkeypatch.setattr("manto.data.requests.get", get)
    with pytest.raises(ValueError):
        fetch_nbp_monthly(code, start, end)
    get.assert_not_called()
