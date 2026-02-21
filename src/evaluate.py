"""Evaluate clustering outputs and provide clustering-quality metrics."""

from __future__ import annotations

from itertools import combinations

from src.cluster_critic import analyze_cluster


def _normalized_optional_text(value: object) -> str:
    """Normalize optional text-like values for conflict checks."""
    return str(value or "").strip().lower()


def _normalized_unit_value(value: object) -> str:
    """Normalize unit value as a compact numeric string."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return ""


def _distinct_non_empty_values(records: list[dict], field: str) -> set[str]:
    """Collect distinct normalized non-empty values for a text field."""
    values: set[str] = set()
    for record in records:
        normalized = _normalized_optional_text(record.get(field))
        if normalized:
            values.add(normalized)
    return values


def _distinct_unit_values(records: list[dict]) -> set[str]:
    """Collect distinct normalized non-empty unit values."""
    values: set[str] = set()
    for record in records:
        normalized = _normalized_unit_value(record.get("unit_value"))
        if normalized:
            values.add(normalized)
    return values


def _suspect_reasons(records: list[dict]) -> list[str]:
    """Return reason codes for mixed attributes within a cluster."""
    reasons: list[str] = []

    stock_codes = _distinct_non_empty_values(records, "stock_code")
    if len(stock_codes) > 1:
        reasons.append("stock_code_mixed")

    unit_names = _distinct_non_empty_values(records, "unit_name")
    if len(unit_names) > 1:
        reasons.append("unit_name_mixed")

    unit_systems = _distinct_non_empty_values(records, "unit_system")
    if len(unit_systems) > 1:
        reasons.append("unit_system_mixed")

    unit_values = _distinct_unit_values(records)
    if len(unit_values) > 1:
        reasons.append("unit_value_mixed")

    return reasons


def cluster_assignments_from_records(records: list[dict]) -> dict[str, str]:
    """Build record_id -> cluster_id mapping from clustered records."""
    assignments: dict[str, str] = {}
    for record in records:
        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            continue
        assignments[record_id] = str(record["cluster_id"])
    return assignments


def pairwise_cluster_metrics(
    predicted_assignments: dict[str, object],
    labeled_assignments: dict[str, object],
) -> dict[str, float]:
    """Compute pairwise precision/recall/F1 for overlapping labeled records."""
    common_ids = sorted(
        set(predicted_assignments.keys()) & set(labeled_assignments.keys())
    )
    if len(common_ids) < 2:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp_pairs": 0.0,
            "fp_pairs": 0.0,
            "fn_pairs": 0.0,
            "num_common_records": float(len(common_ids)),
        }

    tp_pairs = 0
    fp_pairs = 0
    fn_pairs = 0
    for left_id, right_id in combinations(common_ids, 2):
        predicted_same = (
            str(predicted_assignments[left_id]) == str(predicted_assignments[right_id])
        )
        labeled_same = (
            str(labeled_assignments[left_id]) == str(labeled_assignments[right_id])
        )
        if predicted_same and labeled_same:
            tp_pairs += 1
        elif predicted_same and not labeled_same:
            fp_pairs += 1
        elif not predicted_same and labeled_same:
            fn_pairs += 1

    precision = tp_pairs / (tp_pairs + fp_pairs) if (tp_pairs + fp_pairs) else 0.0
    recall = tp_pairs / (tp_pairs + fn_pairs) if (tp_pairs + fn_pairs) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_pairs": float(tp_pairs),
        "fp_pairs": float(fp_pairs),
        "fn_pairs": float(fn_pairs),
        "num_common_records": float(len(common_ids)),
    }


def evaluate(clusters, canonical_labels):
    """Compute metrics and build report."""
    cluster_sizes: dict[str, int] = {}
    clusters_by_id: dict[str, list[dict]] = {}
    for record in clusters:
        cluster_id = str(int(record["cluster_id"]))
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        clusters_by_id.setdefault(cluster_id, []).append(record)

    num_records = len(clusters)
    num_clusters = len(cluster_sizes)
    avg_cluster_size = (num_records / num_clusters) if num_clusters else 0.0
    largest_cluster = max(cluster_sizes.values()) if cluster_sizes else 0

    labels = {
        str(int(cluster_id)): label
        for cluster_id, label in canonical_labels.items()
    }

    suspect_clusters: list[dict] = []
    for cluster_id in sorted(clusters_by_id, key=int):
        records = clusters_by_id[cluster_id]
        reasons = _suspect_reasons(records)
        if not reasons:
            continue
        critic_result = analyze_cluster(records, labels.get(cluster_id))
        suspect_clusters.append(
            {
                "cluster_id": cluster_id,
                "reasons": reasons,
                "size": len(records),
                "risk_score": critic_result["risk_score"],
                "explanation": critic_result["explanation"],
            }
        )

    return {
        "num_records": num_records,
        "num_clusters": num_clusters,
        "cluster_sizes": cluster_sizes,
        "labels": labels,
        "cluster_stats": {
            "total_clusters": num_clusters,
            "avg_cluster_size": avg_cluster_size,
            "largest_cluster": largest_cluster,
        },
        "suspect_clusters": suspect_clusters,
    }
