"""Tests for cluster critic risk scoring and explanations."""

from __future__ import annotations

from src.cluster_critic import analyze_cluster


def test_analyze_cluster_returns_low_risk_for_consistent_records() -> None:
    records = [
        {
            "record_id": "r0",
            "cluster_id": 0,
            "description_norm": "powder",
            "feature_vector": [1.0, 0.0],
            "stock_code": "S1",
            "unit_name": "g",
            "unit_system": "metric",
            "unit_value": 250,
        },
        {
            "record_id": "r1",
            "cluster_id": 0,
            "description_norm": "powder",
            "feature_vector": [1.0, 0.0],
            "stock_code": "S1",
            "unit_name": "g",
            "unit_system": "metric",
            "unit_value": 250,
        },
    ]

    result = analyze_cluster(records, "powder")

    assert result["risk_score"] == 0.0
    assert result["signals"] == []
    assert result["explanation"] == (
        "Low inconsistency risk (0.0000) for 'powder': "
        "no mixed attribute signals detected."
    )


def test_analyze_cluster_returns_high_risk_for_multiple_conflicts() -> None:
    records = [
        {
            "record_id": "r0",
            "cluster_id": 3,
            "description_norm": "mix",
            "feature_vector": [1.0, 0.0],
            "stock_code": "A",
            "unit_name": "ml",
            "unit_system": "metric",
            "unit_value": 500,
        },
        {
            "record_id": "r1",
            "cluster_id": 3,
            "description_norm": "mix",
            "feature_vector": [1.0, 0.0],
            "stock_code": "B",
            "unit_name": "g",
            "unit_system": "imperial",
            "unit_value": 750,
        },
    ]

    result = analyze_cluster(records, "mix")

    assert result["risk_score"] == 0.7
    assert result["signals"] == [
        "stock_code_mixed",
        "unit_name_mixed",
        "unit_system_mixed",
        "unit_value_mixed",
    ]
    assert result["explanation"] == (
        "High inconsistency risk (0.7000) for 'mix': "
        "detected stock_code_mixed, unit_name_mixed, unit_system_mixed, unit_value_mixed."
    )
