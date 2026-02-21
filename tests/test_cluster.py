"""Tests for pairwise cosine nearest-neighbor baseline."""

from __future__ import annotations

import pytest

from src.cluster import cluster


def test_cluster_empty_input_returns_empty_list() -> None:
    assert cluster([]) == []


def test_cluster_single_record_has_no_neighbors() -> None:
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
            "description_norm": "single item",
            "feature_vector": [1.0, 0.0],
            "neighbors": [],
        }
    ]


def test_cluster_applies_similarity_threshold() -> None:
    records = [
        {"record_id": "r0", "description_norm": "a", "feature_vector": [1.0, 0.0]},
        {"record_id": "r1", "description_norm": "b", "feature_vector": [0.9, 0.1]},
        {"record_id": "r2", "description_norm": "c", "feature_vector": [0.0, 1.0]},
    ]

    result = cluster(records, top_k=5, similarity_threshold=0.80)
    r0_neighbors = result[0]["neighbors"]

    assert r0_neighbors == [{"record_id": "r1", "similarity": pytest.approx(0.993884)}]


def test_cluster_applies_top_k_cap() -> None:
    records = [
        {"record_id": "r0", "description_norm": "seed", "feature_vector": [1.0, 0.0]},
        {"record_id": "r1", "description_norm": "n1", "feature_vector": [0.95, 0.05]},
        {"record_id": "r2", "description_norm": "n2", "feature_vector": [0.9, 0.1]},
        {"record_id": "r3", "description_norm": "n3", "feature_vector": [0.85, 0.15]},
    ]

    result = cluster(records, top_k=2, similarity_threshold=0.0)
    r0_neighbors = result[0]["neighbors"]

    assert len(r0_neighbors) == 2
    assert [neighbor["record_id"] for neighbor in r0_neighbors] == ["r1", "r2"]


def test_cluster_tie_break_is_deterministic_by_input_index() -> None:
    records = [
        {"record_id": "r0", "description_norm": "seed", "feature_vector": [1.0, 0.0]},
        {"record_id": "r1", "description_norm": "same-a", "feature_vector": [1.0, 0.0]},
        {"record_id": "r2", "description_norm": "same-b", "feature_vector": [1.0, 0.0]},
    ]

    result = cluster(records, top_k=5, similarity_threshold=0.80)
    r0_neighbors = result[0]["neighbors"]

    assert [neighbor["record_id"] for neighbor in r0_neighbors] == ["r1", "r2"]
