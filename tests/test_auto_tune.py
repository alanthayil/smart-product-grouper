"""Tests for threshold auto-tuning utilities."""

from __future__ import annotations

from src.auto_tune import tune_similarity_threshold


def _feature(record_id: str, vector: list[float]) -> dict:
    return {
        "record_id": record_id,
        "description_norm": record_id,
        "feature_vector": vector,
        "unit_name": "ml",
        "unit_system": "metric",
        "unit_value": 1.0,
    }


def test_tune_similarity_threshold_selects_highest_threshold_on_f1_tie() -> None:
    features = [
        _feature("a", [1.0, 0.0]),
        _feature("b", [0.9, 0.1]),
        _feature("c", [0.7, 0.714]),
        _feature("d", [0.65, 0.76]),
    ]
    labeled = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}

    result = tune_similarity_threshold(features, labeled, [0.75, 0.8, 0.95])

    assert result["best_threshold"] == 0.95
    by_threshold = {entry["threshold"]: entry["metrics"] for entry in result["results"]}
    assert by_threshold[0.75]["f1"] < by_threshold[0.8]["f1"]
    assert by_threshold[0.8]["f1"] == by_threshold[0.95]["f1"] == 1.0
