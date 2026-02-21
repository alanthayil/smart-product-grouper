"""Tests for template-based canonical label generation."""

from __future__ import annotations

from src.canonicalize import canonicalize, canonicalize_with_confidence
from src.cluster import cluster
from src.extract import extract
from src.normalize import normalize


class _FakeProvider:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._vectors


def test_canonicalize_empty_input_returns_empty_mapping() -> None:
    assert canonicalize([]) == {}


def test_canonicalize_uses_template_b_when_units_are_consistent() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 0,
            "description_norm": "white ceramic mug",
            "unit_value": 350.0,
            "unit_name": "ml",
        },
        {
            "record_id": "r1",
            "cluster_id": 0,
            "description_norm": "white ceramic mug",
            "unit_value": 350.0,
            "unit_name": "ml",
        },
    ]

    assert canonicalize(clusters) == {0: "white ceramic mug 350 ml"}


def test_canonicalize_falls_back_when_any_unit_field_is_missing() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 0,
            "description_norm": "olive oil bottle",
            "unit_value": 1000.0,
            "unit_name": "ml",
        },
        {
            "record_id": "r1",
            "cluster_id": 0,
            "description_norm": "olive oil bottle",
            "unit_value": None,
            "unit_name": "ml",
        },
    ]

    assert canonicalize(clusters) == {0: "olive oil bottle"}


def test_canonicalize_falls_back_when_unit_values_conflict() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 0,
            "description_norm": "powder pack",
            "unit_value": 250.0,
            "unit_name": "g",
        },
        {
            "record_id": "r1",
            "cluster_id": 0,
            "description_norm": "powder pack",
            "unit_value": 500.0,
            "unit_name": "g",
        },
    ]

    assert canonicalize(clusters) == {0: "powder pack"}


def test_canonicalize_falls_back_when_unit_names_conflict() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 0,
            "description_norm": "powder pack",
            "unit_value": 250.0,
            "unit_name": "g",
        },
        {
            "record_id": "r1",
            "cluster_id": 0,
            "description_norm": "powder pack",
            "unit_value": 250.0,
            "unit_name": "ml",
        },
    ]

    assert canonicalize(clusters) == {0: "powder pack"}


def test_canonicalize_picks_most_frequent_description_with_lexical_tie_break() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 1,
            "description_norm": "beta mug",
            "unit_value": None,
            "unit_name": None,
        },
        {
            "record_id": "r1",
            "cluster_id": 1,
            "description_norm": "alpha mug",
            "unit_value": None,
            "unit_name": None,
        },
        {
            "record_id": "r2",
            "cluster_id": 1,
            "description_norm": "beta mug",
            "unit_value": None,
            "unit_name": None,
        },
        {
            "record_id": "r3",
            "cluster_id": 1,
            "description_norm": "alpha mug",
            "unit_value": None,
            "unit_name": None,
        },
    ]

    assert canonicalize(clusters) == {1: "alpha mug"}


def test_canonicalize_handles_multiple_clusters_deterministically() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 2,
            "description_norm": "sparkling water",
            "unit_value": 500.0,
            "unit_name": "ml",
        },
        {
            "record_id": "r1",
            "cluster_id": 0,
            "description_norm": "hexagon bolt",
            "unit_value": None,
            "unit_name": None,
        },
        {
            "record_id": "r2",
            "cluster_id": 2,
            "description_norm": "sparkling water",
            "unit_value": 500.0,
            "unit_name": "ml",
        },
    ]

    assert canonicalize(clusters) == {
        0: "hexagon bolt",
        2: "sparkling water 500 ml",
    }


def test_canonicalize_uses_units_from_cluster_output() -> None:
    feature_records = [
        {
            "record_id": "r0",
            "description_norm": "sparkling water",
            "feature_vector": [1.0, 0.0],
            "stock_code": "WATER-500",
            "unit_value": 500.0,
            "unit_name": "ml",
            "unit_system": "metric",
        },
        {
            "record_id": "r1",
            "description_norm": "sparkling water",
            "feature_vector": [0.96, 0.04],
            "stock_code": "WATER-500",
            "unit_value": 500.0,
            "unit_name": "ml",
            "unit_system": "metric",
        },
    ]

    clustered = cluster(feature_records)

    assert canonicalize(clustered) == {0: "sparkling water 500 ml"}


def test_phase5_pipeline_generates_template_based_canonical_label() -> None:
    raw_records = [
        {"Description": "Sparkling Water 500ml Bottle"},
        {"Description": "sparkling water 500 ml bottle"},
    ]
    normalized = normalize(raw_records)
    features = extract(normalized, provider=_FakeProvider([[1.0, 0.0], [0.96, 0.04]]))
    clustered = cluster(features)

    assert canonicalize(clustered) == {0: "sparkling water 500 ml bottle 500 ml"}


def test_phase5_pipeline_falls_back_to_description_when_units_conflict() -> None:
    raw_records = [
        {"Description": "Sparkling Water 500ml Bottle"},
        {"Description": "sparkling water 500 ml bottle"},
    ]
    normalized = normalize(raw_records)
    features = extract(normalized, provider=_FakeProvider([[1.0, 0.0], [0.96, 0.04]]))

    # Keep a single cluster via stock-code match, while unit metadata conflicts.
    features[0]["stock_code"] = "WATER-500"
    features[1]["stock_code"] = "WATER-500"
    features[1]["unit_value"] = 750.0

    clustered = cluster(features)

    assert canonicalize(clustered) == {0: "sparkling water 500 ml bottle"}


def test_canonicalize_missing_attributes_picks_most_frequent_cleaned_name() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 0,
            "description_norm": "  White   Ceramic   Mug ",
            "unit_value": 350.0,
            "unit_name": None,
        },
        {
            "record_id": "r1",
            "cluster_id": 0,
            "description_norm": "white ceramic mug",
            "unit_value": 350.0,
            "unit_name": None,
        },
        {
            "record_id": "r2",
            "cluster_id": 0,
            "description_norm": "white coffee mug",
            "unit_value": 350.0,
            "unit_name": None,
        },
    ]

    assert canonicalize(clusters) == {0: "white ceramic mug"}


def test_canonicalize_tie_after_cleaning_uses_lexicographic_order() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 3,
            "description_norm": "  Beta    Mug ",
            "unit_value": None,
            "unit_name": None,
        },
        {
            "record_id": "r1",
            "cluster_id": 3,
            "description_norm": "alpha mug",
            "unit_value": None,
            "unit_name": None,
        },
        {
            "record_id": "r2",
            "cluster_id": 3,
            "description_norm": "beta mug",
            "unit_value": None,
            "unit_name": None,
        },
        {
            "record_id": "r3",
            "cluster_id": 3,
            "description_norm": "  Alpha   Mug  ",
            "unit_value": None,
            "unit_name": None,
        },
    ]

    assert canonicalize(clusters) == {3: "alpha mug"}


def test_canonicalize_with_confidence_preserves_label_output() -> None:
    clusters = [
        {
            "record_id": "r0",
            "cluster_id": 0,
            "description_norm": "white ceramic mug",
            "feature_vector": [1.0, 0.0],
            "unit_value": 350.0,
            "unit_name": "ml",
            "unit_system": "metric",
            "stock_code": "MUG-350",
        },
        {
            "record_id": "r1",
            "cluster_id": 0,
            "description_norm": "white ceramic mug",
            "feature_vector": [0.99, 0.01],
            "unit_value": 350.0,
            "unit_name": "ml",
            "unit_system": "metric",
            "stock_code": "MUG-350",
        },
    ]

    labels = canonicalize(clusters)
    labels_with_confidence, confidences = canonicalize_with_confidence(clusters)

    assert labels_with_confidence == labels
    assert set(confidences) == set(labels)


def test_canonicalize_with_confidence_penalizes_missing_attributes() -> None:
    clusters = [
        {
            "record_id": "good-0",
            "cluster_id": 0,
            "description_norm": "sparkling water",
            "feature_vector": [1.0, 0.0],
            "unit_value": 500.0,
            "unit_name": "ml",
            "unit_system": "metric",
            "stock_code": "WATER-500",
        },
        {
            "record_id": "good-1",
            "cluster_id": 0,
            "description_norm": "sparkling water",
            "feature_vector": [0.98, 0.02],
            "unit_value": 500.0,
            "unit_name": "ml",
            "unit_system": "metric",
            "stock_code": "WATER-500",
        },
        {
            "record_id": "bad-0",
            "cluster_id": 1,
            "description_norm": "sparkling water",
            "feature_vector": [1.0, 0.0],
            "unit_value": 500.0,
            "unit_name": None,
            "unit_system": "metric",
            "stock_code": "WATER-500",
        },
        {
            "record_id": "bad-1",
            "cluster_id": 1,
            "description_norm": "sparkling water",
            "feature_vector": [0.98, 0.02],
            "unit_value": 500.0,
            "unit_name": None,
            "unit_system": "metric",
            "stock_code": "WATER-500",
        },
    ]

    _, confidences = canonicalize_with_confidence(clusters)
    assert confidences[0] > confidences[1]


def test_canonicalize_with_confidence_reflects_similarity_mean() -> None:
    clusters = [
        {
            "record_id": "high-0",
            "cluster_id": 0,
            "description_norm": "alpha item",
            "feature_vector": [1.0, 0.0],
            "unit_value": 250.0,
            "unit_name": "g",
            "unit_system": "metric",
            "stock_code": "ALPHA-250",
        },
        {
            "record_id": "high-1",
            "cluster_id": 0,
            "description_norm": "alpha item",
            "feature_vector": [0.99, 0.01],
            "unit_value": 250.0,
            "unit_name": "g",
            "unit_system": "metric",
            "stock_code": "ALPHA-250",
        },
        {
            "record_id": "low-0",
            "cluster_id": 1,
            "description_norm": "beta item",
            "feature_vector": [1.0, 0.0],
            "unit_value": 250.0,
            "unit_name": "g",
            "unit_system": "metric",
            "stock_code": "BETA-250",
        },
        {
            "record_id": "low-1",
            "cluster_id": 1,
            "description_norm": "beta item",
            "feature_vector": [-1.0, 0.0],
            "unit_value": 250.0,
            "unit_name": "g",
            "unit_system": "metric",
            "stock_code": "BETA-250",
        },
    ]

    _, confidences = canonicalize_with_confidence(clusters)
    assert confidences[0] > confidences[1]


def test_canonicalize_with_confidence_singleton_is_deterministic() -> None:
    clusters = [
        {
            "record_id": "solo",
            "cluster_id": 7,
            "description_norm": "single item",
            "feature_vector": [0.1, 0.2, 0.3],
            "unit_value": 100.0,
            "unit_name": "ml",
            "unit_system": "metric",
            "stock_code": "SINGLE-100",
        }
    ]

    _, confidences = canonicalize_with_confidence(clusters)
    assert confidences == {7: 1.0}
