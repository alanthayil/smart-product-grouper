"""Tests for clustering logic based on similarity and attributes."""

from __future__ import annotations

import pytest

from src.cluster import cluster


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
