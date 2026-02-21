"""Tests for run.py CLI behavior."""

from __future__ import annotations

import run


def test_run_main_auto_tuning_uses_best_threshold(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(run, "ingest", lambda _: [])
    monkeypatch.setattr(run, "normalize", lambda raw: raw)
    monkeypatch.setattr(
        run,
        "extract",
        lambda _: [
            {
                "record_id": "r0",
                "description_norm": "item",
                "feature_vector": [1.0],
                "unit_name": "ml",
                "unit_system": "metric",
                "unit_value": 1.0,
            }
        ],
    )

    called_thresholds: list[float] = []

    def fake_cluster(features: list[dict], *, similarity_threshold: float = 0.85) -> list[dict]:
        called_thresholds.append(similarity_threshold)
        return [
            {
                "record_id": "r0",
                "cluster_id": 0,
                "description_norm": "item",
                "feature_vector": [1.0],
            }
        ]

    monkeypatch.setattr(run, "cluster", fake_cluster)
    monkeypatch.setattr(run, "canonicalize", lambda _: {0: "item"})
    monkeypatch.setattr(run, "evaluate", lambda clusters, labels: {"num_records": 1})
    monkeypatch.setattr(
        run,
        "_load_labeled_assignments",
        lambda _: {"r0": "cluster-a"},
    )
    monkeypatch.setattr(
        run,
        "tune_similarity_threshold",
        lambda **_: {"best_threshold": 0.9, "best_metrics": {"f1": 1.0}, "results": []},
    )

    run.main(
        [
            "input.xlsx",
            "--auto-tune-thresholds",
            "--labels-path",
            "labels.json",
        ]
    )

    assert called_thresholds == [0.9]
    output = capsys.readouterr().out
    assert "'similarity_threshold': 0.9" in output
    assert "'tuning': {'best_threshold': 0.9" in output
