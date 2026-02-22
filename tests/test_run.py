"""Tests for run.py CLI behavior."""

from __future__ import annotations

import run


def test_run_main_auto_tuning_uses_best_threshold(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        run,
        "load_runtime_config",
        lambda: {
            "cluster": {"similarity_threshold": 0.77, "conflicts": {"enforce_color": True, "enforce_quantity": True}},
            "normalize": {
                "number_words": True,
                "token_splits": True,
                "noise_tokens": True,
                "extract_color": True,
                "extract_quantity": True,
            },
        },
    )
    monkeypatch.setattr(run, "ingest", lambda _: [])
    monkeypatch.setattr(run, "normalize", lambda raw, **_: raw)
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


def test_run_main_includes_edge_debug_when_enabled(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run,
        "load_runtime_config",
        lambda: {
            "cluster": {"similarity_threshold": 0.85, "conflicts": {"enforce_color": True, "enforce_quantity": True}},
            "normalize": {
                "number_words": True,
                "token_splits": True,
                "noise_tokens": True,
                "extract_color": True,
                "extract_quantity": True,
            },
        },
    )
    monkeypatch.setattr(run, "ingest", lambda _: [])
    monkeypatch.setattr(run, "normalize", lambda raw, **_: raw)
    monkeypatch.setattr(
        run,
        "extract",
        lambda _: [
            {
                "record_id": "r0",
                "description_norm": "item",
                "feature_vector": [1.0],
            }
        ],
    )

    observed: dict[str, object] = {}

    def fake_cluster(
        features: list[dict],
        *,
        similarity_threshold: float = 0.85,
        enforce_color_conflict: bool = True,
        enforce_quantity_conflict: bool = True,
        enable_blocking: bool = True,
        blocking_small_input_cutoff: int = 300,
        rare_token_max_frequency: int = 5,
        edge_debug: bool = False,
        edge_debug_top_k: int = 20,
        edge_debug_collector: dict[str, object] | None = None,
    ) -> list[dict]:
        observed["similarity_threshold"] = similarity_threshold
        observed["edge_debug"] = edge_debug
        observed["edge_debug_top_k"] = edge_debug_top_k
        if edge_debug_collector is not None:
            edge_debug_collector["candidate_pairs_evaluated"] = 0
            edge_debug_collector["top_k"] = edge_debug_top_k
            edge_debug_collector["top_candidate_pairs"] = []
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

    run.main(
        [
            "input.xlsx",
            "--edge-debug",
            "--edge-debug-top-k",
            "7",
        ]
    )

    assert observed == {
        "similarity_threshold": 0.85,
        "edge_debug": True,
        "edge_debug_top_k": 7,
    }
    output = capsys.readouterr().out
    assert "'edge_debug': {'candidate_pairs_evaluated': 0, 'top_k': 7, 'top_candidate_pairs': []}" in output


def test_run_main_uses_shared_config_for_normalize_and_cluster(monkeypatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        run,
        "load_runtime_config",
        lambda: {
            "cluster": {"similarity_threshold": 0.91, "conflicts": {"enforce_color": False, "enforce_quantity": False}},
            "normalize": {
                "number_words": False,
                "token_splits": False,
                "noise_tokens": False,
                "extract_color": False,
                "extract_quantity": False,
            },
        },
    )
    monkeypatch.setattr(run, "ingest", lambda _: [{"Description": "item"}])

    def _fake_normalize(raw: list[dict], **kwargs) -> list[dict]:
        observed["normalize_kwargs"] = kwargs
        return [{"description": "item"}]

    monkeypatch.setattr(run, "normalize", _fake_normalize)
    monkeypatch.setattr(
        run,
        "extract",
        lambda _: [{"record_id": "r0", "description_norm": "item", "feature_vector": [1.0]}],
    )

    def _fake_cluster(
        features: list[dict],
        *,
        similarity_threshold: float = 0.85,
        enforce_color_conflict: bool = True,
        enforce_quantity_conflict: bool = True,
        **_kwargs,
    ) -> list[dict]:
        observed["cluster_similarity_threshold"] = similarity_threshold
        observed["cluster_color_toggle"] = enforce_color_conflict
        observed["cluster_quantity_toggle"] = enforce_quantity_conflict
        return [{"record_id": "r0", "cluster_id": 0, "description_norm": "item", "feature_vector": [1.0]}]

    monkeypatch.setattr(run, "cluster", _fake_cluster)
    monkeypatch.setattr(run, "canonicalize", lambda _: {0: "item"})
    monkeypatch.setattr(run, "evaluate", lambda clusters, labels: {"num_records": 1})

    run.main(["input.xlsx"])

    assert observed["normalize_kwargs"] == {
        "canonicalize_number_words": False,
        "canonicalize_token_splits": False,
        "remove_noise_tokens": False,
        "extract_color": False,
        "extract_quantity": False,
    }
    assert observed["cluster_similarity_threshold"] == 0.91
    assert observed["cluster_color_toggle"] is False
    assert observed["cluster_quantity_toggle"] is False
