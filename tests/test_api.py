"""Tests for FastAPI /cluster endpoint."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.ingest import RETAIL_COLUMNS, RETAIL_SHEETS

INVALID_XLSX_DETAIL = (
    "Invalid xlsx file upload. Please provide a valid .xlsx workbook "
    "with required sheets/columns."
)


@pytest.fixture(autouse=True)
def _set_openai_api_key(monkeypatch) -> None:
    """Default tests to a valid API key so app startup preflight passes."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


@pytest.fixture(autouse=True)
def _set_runtime_config_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.api.load_runtime_config",
        lambda: {
            "cluster": {
                "similarity_threshold": 0.85,
                "conflicts": {"enforce_color": True, "enforce_quantity": True},
            },
            "normalize": {
                "number_words": True,
                "token_splits": True,
                "noise_tokens": True,
                "extract_color": True,
                "extract_quantity": True,
            },
        },
    )


def test_startup_preflight_fails_when_openai_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.api.load_dotenv", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="Missing required configuration: OPENAI_API_KEY"):
        with TestClient(app):
            pass


def test_health_config_reports_required_config_present() -> None:
    client = TestClient(app)
    response = client.get("/health/config")

    assert response.status_code == 200
    assert response.json() == {"required": {"OPENAI_API_KEY": True}, "ok": True}
    assert "test-api-key" not in response.text


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
        "unmatched_tokens",
    }


def test_cluster_endpoint_includes_edge_debug_when_enabled(monkeypatch) -> None:
    client = TestClient(app)

    def _fake_extract(records: list[dict]) -> list[dict]:
        return [
            {
                "record_id": "r0",
                "description_norm": "demo item",
                "feature_vector": [1.0, 0.0],
            },
            {
                "record_id": "r1",
                "description_norm": "demo item alt",
                "feature_vector": [0.99, 0.01],
            },
        ]

    def _fake_cluster(
        records: list[dict],
        *,
        similarity_threshold: float = 0.85,
        enforce_color_conflict: bool = True,
        enforce_quantity_conflict: bool = True,
        enable_blocking: bool = True,
        blocking_small_input_cutoff: int = 300,
        rare_token_max_frequency: int = 5,
        edge_debug: bool = False,
        edge_debug_top_k: int = 20,
        edge_debug_collector: dict[str, object] | None = None,
    ) -> list[dict]:
        if edge_debug and edge_debug_collector is not None:
            edge_debug_collector["candidate_pairs_evaluated"] = 1
            edge_debug_collector["top_k"] = edge_debug_top_k
            edge_debug_collector["top_candidate_pairs"] = [
                {
                    "record_id_i": "r0",
                    "record_id_j": "r1",
                    "description_i": "demo item",
                    "description_j": "demo item alt",
                    "similarity": 0.99,
                    "stock_code_match": False,
                    "attrs_i": {
                        "stock_code": None,
                        "color": None,
                        "quantity_total": None,
                        "unit_name": None,
                        "unit_system": None,
                        "unit_value": None,
                    },
                    "attrs_j": {
                        "stock_code": None,
                        "color": None,
                        "quantity_total": None,
                        "unit_name": None,
                        "unit_system": None,
                        "unit_value": None,
                    },
                    "conflict_reasons": [],
                    "edge_decision": True,
                }
            ]
        return [
            {
                "record_id": "r0",
                "cluster_id": 0,
                "description_norm": "demo item",
                "feature_vector": [1.0, 0.0],
            },
            {
                "record_id": "r1",
                "cluster_id": 0,
                "description_norm": "demo item alt",
                "feature_vector": [0.99, 0.01],
            },
        ]

    monkeypatch.setattr("src.api.extract", _fake_extract)
    monkeypatch.setattr("src.api.cluster", _fake_cluster)
    monkeypatch.setattr("src.api.canonicalize", lambda clusters: {0: "demo item"})
    monkeypatch.setattr(
        "src.api.evaluate",
        lambda clusters, labels: {
            "num_records": 2,
            "num_clusters": 1,
            "cluster_sizes": {"0": 2},
            "labels": {"0": "demo item"},
            "cluster_stats": {"total_clusters": 1, "avg_cluster_size": 2.0, "largest_cluster": 2},
            "suspect_clusters": [],
        },
    )

    response = client.post(
        "/cluster?edge_debug=true&edge_debug_top_k=7",
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
    assert "edge_debug" in payload
    assert payload["edge_debug"]["candidate_pairs_evaluated"] == 1
    assert payload["edge_debug"]["top_k"] == 7
    assert len(payload["edge_debug"]["top_candidate_pairs"]) == 1


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


def test_cluster_endpoint_returns_risk_and_explanation_in_suspects(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr("src.api.extract", lambda records: records)
    monkeypatch.setattr("src.api.cluster", lambda records, **kwargs: records)
    monkeypatch.setattr("src.api.canonicalize", lambda clusters: {0: "item"})
    monkeypatch.setattr(
        "src.api.evaluate",
        lambda clusters, labels: {
            "num_records": 2,
            "num_clusters": 1,
            "cluster_sizes": {"0": 2},
            "labels": {"0": "item"},
            "cluster_stats": {"total_clusters": 1, "avg_cluster_size": 2.0, "largest_cluster": 2},
            "suspect_clusters": [
                {
                    "cluster_id": "0",
                    "reasons": ["stock_code_mixed"],
                    "size": 2,
                    "risk_score": 0.175,
                    "explanation": (
                        "Low inconsistency risk (0.1750) for 'item': "
                        "detected stock_code_mixed."
                    ),
                }
            ],
        },
    )

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
    assert payload["suspect_clusters"][0]["risk_score"] == 0.175
    assert "detected stock_code_mixed" in payload["suspect_clusters"][0]["explanation"]


def test_cluster_endpoint_includes_ranked_unmatched_tokens(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "src.api.ingest",
        lambda _path: [
            {"Description": "Anchor rivet 2oz pack"},
            {"Description": "Rivet anchor pro"},
            {"Description": "Anchor and clamp"},
        ],
    )
    monkeypatch.setattr(
        "src.api.normalize",
        lambda raw_records, **kwargs: [
            {
                "description": str(row.get("Description", "")).lower(),
                "unit_value": None,
                "unit_name": None,
                "unit_system": None,
            }
            for row in raw_records
        ],
    )
    monkeypatch.setattr("src.api.extract", lambda records: records)
    monkeypatch.setattr("src.api.cluster", lambda records, **kwargs: records)
    monkeypatch.setattr("src.api.canonicalize", lambda clusters: {0: "item"})
    monkeypatch.setattr(
        "src.api.evaluate",
        lambda clusters, labels: {
            "num_records": len(clusters),
            "num_clusters": 1,
            "cluster_sizes": {"0": len(clusters)},
            "labels": {"0": "item"},
            "cluster_stats": {
                "total_clusters": 1,
                "avg_cluster_size": float(len(clusters)),
                "largest_cluster": len(clusters),
            },
            "suspect_clusters": [],
        },
    )

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
    assert response.json()["unmatched_tokens"] == [
        {"token": "anchor", "count": 3},
        {"token": "rivet", "count": 2},
        {"token": "clamp", "count": 1},
        {"token": "pro", "count": 1},
    ]


def test_cluster_endpoint_uses_shared_config_for_normalize_and_cluster(monkeypatch) -> None:
    client = TestClient(app)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "src.api.load_runtime_config",
        lambda: {
            "cluster": {
                "similarity_threshold": 0.91,
                "conflicts": {"enforce_color": False, "enforce_quantity": False},
            },
            "normalize": {
                "number_words": False,
                "token_splits": False,
                "noise_tokens": False,
                "extract_color": False,
                "extract_quantity": False,
            },
        },
    )
    def _fake_normalize(raw: list[dict], **kwargs) -> list[dict]:
        observed["normalize_kwargs"] = kwargs
        return [{"description": "item"}]

    monkeypatch.setattr("src.api.normalize", _fake_normalize)
    monkeypatch.setattr(
        "src.api.extract",
        lambda records: [{"record_id": "r0", "description_norm": "item", "feature_vector": [1.0]}],
    )

    def _fake_cluster(records: list[dict], **kwargs) -> list[dict]:
        observed["cluster_kwargs"] = kwargs
        return [{"record_id": "r0", "cluster_id": 0, "description_norm": "item", "feature_vector": [1.0]}]

    monkeypatch.setattr("src.api.cluster", _fake_cluster)
    monkeypatch.setattr("src.api.canonicalize", lambda clusters: {0: "item"})
    monkeypatch.setattr(
        "src.api.evaluate",
        lambda clusters, labels: {
            "num_records": 1,
            "num_clusters": 1,
            "cluster_sizes": {"0": 1},
            "labels": {"0": "item"},
            "cluster_stats": {"total_clusters": 1, "avg_cluster_size": 1.0, "largest_cluster": 1},
            "suspect_clusters": [],
        },
    )

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
    assert observed["normalize_kwargs"] == {
        "canonicalize_number_words": False,
        "canonicalize_token_splits": False,
        "remove_noise_tokens": False,
        "extract_color": False,
        "extract_quantity": False,
    }
    cluster_kwargs = observed["cluster_kwargs"]
    assert cluster_kwargs["similarity_threshold"] == 0.91
    assert cluster_kwargs["enforce_color_conflict"] is False
    assert cluster_kwargs["enforce_quantity_conflict"] is False
