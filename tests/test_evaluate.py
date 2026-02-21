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
