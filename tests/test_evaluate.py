"""Tests for evaluation report generation."""

from __future__ import annotations

from src.evaluate import evaluate


def test_evaluate_returns_cluster_stats_for_multi_cluster_input() -> None:
    clusters = [
        {"record_id": "r0", "cluster_id": 0, "description_norm": "a", "feature_vector": [1.0]},
        {"record_id": "r1", "cluster_id": 0, "description_norm": "a", "feature_vector": [1.0]},
        {"record_id": "r2", "cluster_id": 1, "description_norm": "b", "feature_vector": [1.0]},
    ]
    labels = {0: "item a", 1: "item b"}

    report = evaluate(clusters, labels)

    assert report["num_records"] == 3
    assert report["num_clusters"] == 2
    assert report["cluster_sizes"] == {"0": 2, "1": 1}
    assert report["labels"] == {"0": "item a", "1": "item b"}
    assert report["cluster_stats"] == {
        "total_clusters": 2,
        "avg_cluster_size": 1.5,
        "largest_cluster": 2,
    }
    assert report["suspect_clusters"] == []


def test_evaluate_handles_empty_input() -> None:
    report = evaluate([], {})

    assert report["num_records"] == 0
    assert report["num_clusters"] == 0
    assert report["cluster_sizes"] == {}
    assert report["labels"] == {}
    assert report["cluster_stats"] == {
        "total_clusters": 0,
        "avg_cluster_size": 0.0,
        "largest_cluster": 0,
    }
    assert report["suspect_clusters"] == []


def test_evaluate_handles_single_cluster() -> None:
    clusters = [
        {"record_id": "r0", "cluster_id": 7, "description_norm": "x", "feature_vector": [1.0]},
        {"record_id": "r1", "cluster_id": 7, "description_norm": "x", "feature_vector": [1.0]},
    ]
    labels = {7: "item x"}

    report = evaluate(clusters, labels)

    assert report["num_records"] == 2
    assert report["num_clusters"] == 1
    assert report["cluster_sizes"] == {"7": 2}
    assert report["labels"] == {"7": "item x"}
    assert report["cluster_stats"] == {
        "total_clusters": 1,
        "avg_cluster_size": 2.0,
        "largest_cluster": 2,
    }
    assert report["suspect_clusters"] == []


def test_evaluate_flags_stock_code_mixed_cluster() -> None:
    clusters = [
        {"record_id": "r0", "cluster_id": 0, "description_norm": "x", "feature_vector": [1.0], "stock_code": "A1"},
        {"record_id": "r1", "cluster_id": 0, "description_norm": "x", "feature_vector": [1.0], "stock_code": "B2"},
    ]

    report = evaluate(clusters, {0: "item x"})

    assert report["suspect_clusters"] == [
        {"cluster_id": "0", "reasons": ["stock_code_mixed"], "size": 2}
    ]


def test_evaluate_flags_unit_mixed_cluster() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 1,
            "description_norm": "oil",
            "feature_vector": [1.0],
            "unit_name": "ml",
            "unit_system": "metric",
            "unit_value": 500,
        },
        {
            "record_id": "r1",
            "cluster_id": 1,
            "description_norm": "oil",
            "feature_vector": [1.0],
            "unit_name": "ml",
            "unit_system": "metric",
            "unit_value": 750,
        },
    ]

    report = evaluate(clusters, {1: "oil"})

    assert report["suspect_clusters"] == [
        {"cluster_id": "1", "reasons": ["unit_value_mixed"], "size": 2}
    ]


def test_evaluate_does_not_flag_consistent_cluster_attributes() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 2,
            "description_norm": "powder",
            "feature_vector": [1.0],
            "stock_code": "S1",
            "unit_name": "g",
            "unit_system": "metric",
            "unit_value": 250,
        },
        {
            "record_id": "r1",
            "cluster_id": 2,
            "description_norm": "powder",
            "feature_vector": [1.0],
            "stock_code": "S1",
            "unit_name": "g",
            "unit_system": "metric",
            "unit_value": 250.0,
        },
    ]

    report = evaluate(clusters, {2: "powder"})

    assert report["suspect_clusters"] == []


def test_evaluate_combines_multiple_mixed_reasons_for_cluster() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 3,
            "description_norm": "mix",
            "feature_vector": [1.0],
            "stock_code": "A",
            "unit_name": "ml",
            "unit_system": "metric",
            "unit_value": 500,
        },
        {
            "record_id": "r1",
            "cluster_id": 3,
            "description_norm": "mix",
            "feature_vector": [1.0],
            "stock_code": "B",
            "unit_name": "g",
            "unit_system": "imperial",
            "unit_value": 600,
        },
    ]

    report = evaluate(clusters, {3: "mix"})

    assert report["suspect_clusters"] == [
        {
            "cluster_id": "3",
            "reasons": [
                "stock_code_mixed",
                "unit_name_mixed",
                "unit_system_mixed",
                "unit_value_mixed",
            ],
            "size": 2,
        }
    ]
