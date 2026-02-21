"""Tests for FastAPI /cluster endpoint."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.ingest import RETAIL_COLUMNS, RETAIL_SHEETS

INVALID_XLSX_DETAIL = (
    "Invalid xlsx file upload. Please provide a valid .xlsx workbook "
    "with required sheets/columns."
)


def test_upload_form_route_renders_minimal_html() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    body = response.text
    assert "<form" in body
    assert 'action="/cluster/view"' in body
    assert 'enctype="multipart/form-data"' in body
    assert 'name="file"' in body


def _build_workbook_bytes() -> bytes:
    row = {
        "Invoice": "536365",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 6,
        "InvoiceDate": "2010-12-01 08:26:00",
        "Price": 2.55,
        "Customer ID": "17850",
        "Country": "United Kingdom",
    }
    df = pd.DataFrame([row], columns=RETAIL_COLUMNS)
    payload = BytesIO()
    with pd.ExcelWriter(payload, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=RETAIL_SHEETS[0], index=False)
        df.to_excel(writer, sheet_name=RETAIL_SHEETS[1], index=False)
    return payload.getvalue()


def _build_workbook_bytes_missing_columns() -> bytes:
    df = pd.DataFrame([{"Invoice": "536365"}])
    payload = BytesIO()
    with pd.ExcelWriter(payload, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=RETAIL_SHEETS[0], index=False)
        df.to_excel(writer, sheet_name=RETAIL_SHEETS[1], index=False)
    return payload.getvalue()


def test_cluster_endpoint_accepts_valid_xlsx(monkeypatch) -> None:
    client = TestClient(app)

    def _fake_extract(records: list[dict]) -> list[dict]:
        return [
            {
                "record_id": f"record-{index}",
                "description_norm": str(record.get("description", "")),
                "feature_vector": [1.0, 0.0],
                "unit_value": record.get("unit_value"),
                "unit_name": record.get("unit_name"),
                "unit_system": record.get("unit_system"),
            }
            for index, record in enumerate(records)
        ]

    monkeypatch.setattr("src.api.extract", _fake_extract)
    response = client.post(
        "/cluster",
        files={
            "file": (
                "demo.xlsx",
                _build_workbook_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "num_records",
        "num_clusters",
        "cluster_sizes",
        "labels",
        "cluster_stats",
        "suspect_clusters",
    }


def test_cluster_endpoint_rejects_invalid_extension() -> None:
    client = TestClient(app)
    response = client.post(
        "/cluster",
        files={"file": ("demo.csv", b"not,xlsx", "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Expected a .xlsx file upload."


def test_cluster_endpoint_requires_file() -> None:
    client = TestClient(app)
    response = client.post("/cluster")
    assert response.status_code == 422


def test_cluster_endpoint_rejects_malformed_workbook() -> None:
    client = TestClient(app)
    response = client.post(
        "/cluster",
        files={"file": ("broken.xlsx", b"not really an xlsx", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == INVALID_XLSX_DETAIL


def test_cluster_endpoint_rejects_workbook_missing_required_columns() -> None:
    client = TestClient(app)
    response = client.post(
        "/cluster",
        files={
            "file": (
                "missing_columns.xlsx",
                _build_workbook_bytes_missing_columns(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == INVALID_XLSX_DETAIL


def test_cluster_view_endpoint_renders_cluster_table(monkeypatch) -> None:
    client = TestClient(app)

    def _fake_extract(records: list[dict]) -> list[dict]:
        return [
            {
                "record_id": f"record-{index}",
                "description_norm": str(record.get("description", "")),
                "feature_vector": [1.0, 0.0],
                "unit_value": record.get("unit_value"),
                "unit_name": record.get("unit_name"),
                "unit_system": record.get("unit_system"),
            }
            for index, record in enumerate(records)
        ]

    monkeypatch.setattr("src.api.extract", _fake_extract)
    response = client.post(
        "/cluster/view",
        files={
            "file": (
                "demo.xlsx",
                _build_workbook_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    body = response.text
    assert "<table" in body
    assert "Cluster ID" in body
    assert "<td>0</td>" in body


def test_cluster_view_endpoint_rejects_malformed_workbook() -> None:
    client = TestClient(app)
    response = client.post(
        "/cluster/view",
        files={"file": ("broken.xlsx", b"not really an xlsx", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == INVALID_XLSX_DETAIL
