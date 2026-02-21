"""Tests for evaluation report generation."""

from __future__ import annotations

from src.evaluate import (
    cluster_assignments_from_records,
    evaluate,
    pairwise_cluster_metrics,
)


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
        {
            "cluster_id": "0",
            "reasons": ["stock_code_mixed"],
            "size": 2,
            "risk_score": 0.175,
            "explanation": (
                "Low inconsistency risk (0.1750) for 'item x': "
                "detected stock_code_mixed."
            ),
        }
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
        {
            "cluster_id": "1",
            "reasons": ["unit_value_mixed"],
            "size": 2,
            "risk_score": 0.175,
            "explanation": (
                "Low inconsistency risk (0.1750) for 'oil': "
                "detected unit_value_mixed."
            ),
        }
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
            "risk_score": 0.7,
            "explanation": (
                "High inconsistency risk (0.7000) for 'mix': "
                "detected stock_code_mixed, unit_name_mixed, unit_system_mixed, "
                "unit_value_mixed."
            ),
        }
    ]


def test_cluster_assignments_from_records_maps_ids_to_cluster_ids() -> None:
    assignments = cluster_assignments_from_records(
        [
            {"record_id": "r0", "cluster_id": 2},
            {"record_id": "r1", "cluster_id": "A"},
        ]
    )

    assert assignments == {"r0": "2", "r1": "A"}


def test_pairwise_cluster_metrics_computes_precision_recall_and_f1() -> None:
    predicted = {"a": 0, "b": 0, "c": 1, "d": 1}
    labeled = {"a": "x", "b": "x", "c": "y", "d": "z"}

    metrics = pairwise_cluster_metrics(predicted, labeled)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == (2.0 / 3.0)
    assert metrics["tp_pairs"] == 1.0
    assert metrics["fp_pairs"] == 1.0
    assert metrics["fn_pairs"] == 0.0


def test_pairwise_cluster_metrics_returns_zero_without_enough_overlap() -> None:
    metrics = pairwise_cluster_metrics({"a": 0}, {"b": 1})

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["num_common_records"] == 0.0
