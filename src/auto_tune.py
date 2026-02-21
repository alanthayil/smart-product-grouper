"""Threshold auto-tuning utilities for clustering."""

from __future__ import annotations

from src.cluster import cluster
from src.evaluate import cluster_assignments_from_records, pairwise_cluster_metrics


def tune_similarity_threshold(
    features: list[dict],
    labeled_assignments: dict[str, object],
    candidate_thresholds: list[float],
) -> dict[str, object]:
    """Sweep thresholds and pick the best by F1 (tie-break: higher threshold)."""
    if not candidate_thresholds:
        raise ValueError("candidate_thresholds must not be empty.")

    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    sweep_results: list[dict[str, object]] = []

    for threshold in candidate_thresholds:
        clustered = cluster(features, similarity_threshold=float(threshold))
        predicted_assignments = cluster_assignments_from_records(clustered)
        metrics = pairwise_cluster_metrics(predicted_assignments, labeled_assignments)
        sweep_results.append({"threshold": float(threshold), "metrics": metrics})

        if best_threshold is None or best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue

        current_f1 = float(metrics["f1"])
        best_f1 = float(best_metrics["f1"])
        if (current_f1 > best_f1) or (
            current_f1 == best_f1 and float(threshold) > best_threshold
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return {
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "results": sweep_results,
    }
