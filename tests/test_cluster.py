"""Tests for clustering logic based on similarity and attributes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.cluster import cluster, generate_candidate_pairs
from src.extract import extract
from src.ingest import RETAIL_COLUMNS, RETAIL_SHEETS, ingest
from src.normalize import normalize


def test_cluster_empty_input_returns_empty_list() -> None:
    assert cluster([]) == []


def test_cluster_single_record_gets_cluster_id_zero() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "single item",
            "feature_vector": [1.0, 0.0],
        }
    ]

    result = cluster(records)

    assert result == [
        {
            "record_id": "r0",
            "cluster_id": 0,
            "description_norm": "single item",
            "feature_vector": [1.0, 0.0],
        }
    ]


def test_cluster_requires_stock_match_when_both_have_stock_codes() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0],
            "stock_code": "A1",
        },
        {
            "record_id": "r1",
            "description_norm": "b",
            "feature_vector": [0.95, 0.05],
            "stock_code": "B2",
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 1]


def test_cluster_uses_stock_code_match_path() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0],
            "stock_code": "A1",
        },
        {
            "record_id": "r1",
            "description_norm": "b",
            "feature_vector": [0.95, 0.05],
            "stock_code": "A1",
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_cluster_stock_code_match_creates_edge_below_similarity_threshold() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0],
            "stock_code": "A1",
        },
        {
            "record_id": "r1",
            "description_norm": "b",
            "feature_vector": [0.0, 1.0],
            "stock_code": "A1",
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_cluster_falls_back_to_unit_matching_without_stock_codes() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0],
            "unit_name": "g",
            "unit_system": "metric",
        },
        {
            "record_id": "r1",
            "description_norm": "b",
            "feature_vector": [0.95, 0.05],
            "unit_name": "g",
            "unit_system": "metric",
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_cluster_uses_similarity_path_when_attributes_are_missing_but_non_conflicting() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0],
        },
        {
            "record_id": "r1",
            "description_norm": "b",
            "feature_vector": [0.95, 0.05],
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_cluster_treats_null_unit_fields_as_optional_signal() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0],
            "unit_name": None,
            "unit_system": None,
            "unit_value": None,
        },
        {
            "record_id": "r1",
            "description_norm": "b",
            "feature_vector": [0.95, 0.05],
            "unit_name": None,
            "unit_system": None,
            "unit_value": None,
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_cluster_similarity_path_blocks_explicit_unit_conflict() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0],
            "unit_name": "ml",
        },
        {
            "record_id": "r1",
            "description_norm": "b",
            "feature_vector": [0.95, 0.05],
            "unit_name": "g",
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 1]


def test_cluster_similarity_path_blocks_explicit_color_conflict() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "red ceramic mug",
            "feature_vector": [1.0, 0.0],
            "color": "red",
        },
        {
            "record_id": "r1",
            "description_norm": "blue ceramic mug",
            "feature_vector": [0.95, 0.05],
            "color": "blue",
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 1]


def test_cluster_similarity_path_allows_missing_color_signal() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "red ceramic mug",
            "feature_vector": [1.0, 0.0],
            "color": "red",
        },
        {
            "record_id": "r1",
            "description_norm": "ceramic mug",
            "feature_vector": [0.95, 0.05],
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_cluster_similarity_path_allows_equivalent_quantity_totals() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "detergent pack 6x10",
            "feature_vector": [1.0, 0.0],
            "quantity_total": 60,
        },
        {
            "record_id": "r1",
            "description_norm": "detergent pack 60",
            "feature_vector": [0.95, 0.05],
            "quantity_total": 60,
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_cluster_similarity_path_blocks_explicit_quantity_conflict() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "detergent pack 60",
            "feature_vector": [1.0, 0.0],
            "quantity_total": 60,
        },
        {
            "record_id": "r1",
            "description_norm": "detergent pack 72",
            "feature_vector": [0.95, 0.05],
            "quantity_total": 72,
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 1]


def test_cluster_similarity_path_allows_missing_quantity_signal() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "detergent pack 60",
            "feature_vector": [1.0, 0.0],
            "quantity_total": 60,
        },
        {
            "record_id": "r1",
            "description_norm": "detergent",
            "feature_vector": [0.95, 0.05],
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_cluster_connected_components_are_transitive() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0, 0.0],
            "stock_code": "A1",
        },
        {
            "record_id": "r1",
            "description_norm": "b",
            "feature_vector": [0.86, 0.51, 0.0],
            "stock_code": "A1",
        },
        {
            "record_id": "r2",
            "description_norm": "c",
            "feature_vector": [0.5, 0.866, 0.0],
            "stock_code": "A1",
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0, 0]


def test_cluster_ids_are_deterministic_by_first_seen_component() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "x",
            "feature_vector": [1.0, 0.0],
            "stock_code": "X",
        },
        {
            "record_id": "r1",
            "description_norm": "a",
            "feature_vector": [1.0, 0.0],
            "stock_code": "A",
        },
        {
            "record_id": "r2",
            "description_norm": "b",
            "feature_vector": [0.95, 0.05],
            "stock_code": "A",
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 1, 1]


def test_cluster_raises_for_mismatched_vector_dimensions() -> None:
    records = [
        {"record_id": "r0", "description_norm": "a", "feature_vector": [1.0, 0.0]},
        {"record_id": "r1", "description_norm": "b", "feature_vector": [1.0, 0.0, 0.0]},
    ]

    with pytest.raises(ValueError, match="same dimension"):
        cluster(records)


def test_similar_items_are_grouped() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "white ceramic mug",
            "feature_vector": [1.0, 0.0],
            "unit_name": "ml",
            "unit_system": "metric",
            "unit_value": 350.0,
        },
        {
            "record_id": "r1",
            "description_norm": "white coffee mug",
            "feature_vector": [0.96, 0.04],
            "unit_name": "ml",
            "unit_system": "metric",
            "unit_value": 350.0,
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_different_size_items_are_separated() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "olive oil bottle 500 ml",
            "feature_vector": [1.0, 0.0],
            "unit_name": "ml",
            "unit_system": "metric",
            "unit_value": 500.0,
        },
        {
            "record_id": "r1",
            "description_norm": "olive oil bottle 1000 ml",
            "feature_vector": [0.96, 0.04],
            "unit_name": "ml",
            "unit_system": "metric",
            "unit_value": 1000.0,
        },
    ]

    result = cluster(records)

    assert [record["cluster_id"] for record in result] == [0, 1]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            {
                "record_id": "r0",
                "description_norm": "powder pack g",
                "feature_vector": [1.0, 0.0],
                "unit_name": "g",
                "unit_system": "metric",
                "unit_value": 250.0,
            },
            {
                "record_id": "r1",
                "description_norm": "powder pack ml",
                "feature_vector": [0.96, 0.04],
                "unit_name": "ml",
                "unit_system": "metric",
                "unit_value": 250.0,
            },
        ),
        (
            {
                "record_id": "r0",
                "description_norm": "flour bag metric",
                "feature_vector": [1.0, 0.0],
                "unit_name": "g",
                "unit_system": "metric",
                "unit_value": 500.0,
            },
            {
                "record_id": "r1",
                "description_norm": "flour bag imperial",
                "feature_vector": [0.96, 0.04],
                "unit_name": "g",
                "unit_system": "imperial",
                "unit_value": 500.0,
            },
        ),
    ],
)
def test_different_standard_items_are_separated(left: dict, right: dict) -> None:
    result = cluster([left, right])

    assert [record["cluster_id"] for record in result] == [0, 1]


def test_cluster_preserves_optional_attributes_for_downstream_labeling() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "sparkling water bottle",
            "feature_vector": [1.0, 0.0],
            "stock_code": "WATER-500",
            "color": "green",
            "quantity_total": 60,
            "unit_value": 500.0,
            "unit_name": "ml",
            "unit_system": "metric",
        },
        {
            "record_id": "r1",
            "description_norm": "sparkling water bottle",
            "feature_vector": [0.96, 0.04],
            "stock_code": "WATER-500",
            "color": "green",
            "quantity_total": 60,
            "unit_value": 500.0,
            "unit_name": "ml",
            "unit_system": "metric",
        },
    ]

    result = cluster(records)

    assert result[0]["stock_code"] == "WATER-500"
    assert result[0]["color"] == "green"
    assert result[0]["quantity_total"] == 60
    assert result[0]["unit_value"] == 500.0
    assert result[0]["unit_name"] == "ml"
    assert result[0]["unit_system"] == "metric"


def test_generate_candidate_pairs_includes_stock_code_block_pair() -> None:
    records = [
        {"description_norm": "alpha mug", "stock_code": "sku-1"},
        {"description_norm": "beta bowl", "stock_code": "sku-1"},
        {"description_norm": "gamma plate", "stock_code": "sku-2"},
    ]

    pairs = generate_candidate_pairs(records)

    assert (0, 1) in pairs


def test_generate_candidate_pairs_includes_prefix_block_pair() -> None:
    records = [
        {"description_norm": "space boy mug blue"},
        {"description_norm": "space boy mug red"},
        {"description_norm": "bird cage holder"},
    ]

    pairs = generate_candidate_pairs(records)

    assert (0, 1) in pairs


def test_generate_candidate_pairs_includes_shared_rare_token_pair() -> None:
    records = [
        {"description_norm": "alpha nebula mug"},
        {"description_norm": "beta nebula plate"},
        {"description_norm": "gamma common mug"},
        {"description_norm": "delta common plate"},
        {"description_norm": "epsilon common bowl"},
        {"description_norm": "zeta common cup"},
        {"description_norm": "eta common vase"},
        {"description_norm": "theta common lamp"},
    ]

    pairs = generate_candidate_pairs(records, rare_token_max_frequency=5)

    assert (0, 1) in pairs
    assert (2, 3) not in pairs


def test_blocking_mode_still_respects_compatibility_conflicts() -> None:
    records = [
        {
            "record_id": "r0",
            "description_norm": "space boy mug",
            "feature_vector": [1.0, 0.0],
            "unit_name": "ml",
        },
        {
            "record_id": "r1",
            "description_norm": "space boy mug",
            "feature_vector": [0.99, 0.01],
            "unit_name": "g",
        },
    ]

    result = cluster(records, blocking_small_input_cutoff=0)

    assert [record["cluster_id"] for record in result] == [0, 1]


def test_small_input_fallback_uses_all_pairs_path(monkeypatch) -> None:
    records = [
        {"record_id": "r0", "description_norm": "a", "feature_vector": [1.0, 0.0]},
        {"record_id": "r1", "description_norm": "b", "feature_vector": [0.95, 0.05]},
    ]

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("blocking generator should not run for small-input fallback")

    monkeypatch.setattr("src.cluster.generate_candidate_pairs", _fail_if_called)

    result = cluster(records, blocking_small_input_cutoff=1000)

    assert [record["cluster_id"] for record in result] == [0, 0]


def test_blocking_candidate_count_is_smaller_than_all_pairs_for_large_sparse_data() -> None:
    records: list[dict] = []
    for index in range(40):
        group = index // 4
        records.append({"description_norm": f"group{group} item{index}"})

    candidates = generate_candidate_pairs(records, rare_token_max_frequency=5)
    all_pairs = (len(records) * (len(records) - 1)) // 2

    assert len(candidates) < all_pairs


class _WorkbookFakeProvider:
    def __init__(self, vector_by_description: dict[str, list[float]]) -> None:
        self._vector_by_description = vector_by_description

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._vector_by_description]
        if missing:
            raise AssertionError(f"Missing test vector mappings for: {missing}")
        return [self._vector_by_description[text] for text in texts]


def _build_workbook_row(record_id: str, description: str) -> dict[str, object]:
    row: dict[str, object] = {
        "Invoice": f"INV-{record_id}",
        "StockCode": "",
        "Description": description,
        "Quantity": 1,
        "InvoiceDate": "2011-01-01 00:00:00",
        "Price": 1.0,
        "Customer ID": "10000",
        "Country": "United Kingdom",
        "record_id": record_id,
    }
    return row


def _write_sample_workbook(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    for column in RETAIL_COLUMNS:
        if column not in df.columns:
            df[column] = None
    ordered_columns = list(RETAIL_COLUMNS) + ["record_id"]
    df = df[ordered_columns]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=RETAIL_SHEETS[0], index=False)
        df.iloc[0:0].to_excel(writer, sheet_name=RETAIL_SHEETS[1], index=False)


def test_sample_workbook_expected_clustering(tmp_path: Path) -> None:
    rows = [
        _build_workbook_row("bird-a", "Decorative Birdcage Lantern"),
        _build_workbook_row("bird-b", "Decorative bird cage lantern"),
        _build_workbook_row("two-a", "Party candle two pack"),
        _build_workbook_row("two-b", "Party candle 2 pack"),
        _build_workbook_row("dup-a", "Vintage photo frame"),
        _build_workbook_row("dup-b", "Vintage photo frame"),
        _build_workbook_row("red-a", "Ceramic mug red"),
        _build_workbook_row("pink-a", "Ceramic mug pink"),
        _build_workbook_row("pack-a", "Storage box pack of 2"),
        _build_workbook_row("pack-b", "Storage box pack of 3"),
        _build_workbook_row("solo-a", "Striped table runner"),
    ]
    vector_by_description = {
        "decorative bird cage lantern": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "party candle 2 pack": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "vintage photo frame": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "ceramic mug red": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "ceramic mug pink": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "storage box pack 2": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "storage box pack 3": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "striped table runner": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    provider = _WorkbookFakeProvider(vector_by_description)

    workbook_path = tmp_path / "expected_clustering.xlsx"
    _write_sample_workbook(workbook_path, rows)
    raw = ingest(str(workbook_path))
    normalized = normalize(raw)
    for record in normalized:
        stock_code = str(record.get("stock_code", "")).strip().lower()
        if stock_code == "nan":
            record["stock_code"] = ""
    features = extract(normalized, provider=provider)
    clustered = cluster(features)

    cluster_by_record_id = {
        str(record["record_id"]): int(record["cluster_id"]) for record in clustered
    }
    num_clusters = len({int(record["cluster_id"]) for record in clustered})

    assert 7 <= num_clusters <= 8
    assert cluster_by_record_id["bird-a"] == cluster_by_record_id["bird-b"]
    assert cluster_by_record_id["two-a"] == cluster_by_record_id["two-b"]
    assert cluster_by_record_id["dup-a"] == cluster_by_record_id["dup-b"]
    assert cluster_by_record_id["red-a"] != cluster_by_record_id["pink-a"]
