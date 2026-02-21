"""Produce canonical labels for each cluster."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

_WHITESPACE_RUNS = re.compile(r"\s+")


def _clean_description_norm(value: object) -> str:
    """Normalize description text for fallback-name selection."""
    normalized = _WHITESPACE_RUNS.sub(" ", str(value or "").strip().lower())
    return normalized.strip()


def _pick_base_description(records: list[dict]) -> str:
    """Pick the most frequent normalized description (stable ties)."""
    descriptions = [
        _clean_description_norm(record.get("description_norm", ""))
        for record in records
    ]
    counts = Counter(descriptions)
    if not counts:
        return ""
    max_frequency = max(counts.values())
    candidates = [text for text, count in counts.items() if count == max_frequency]
    return min(candidates)


def _try_consistent_unit(records: list[dict]) -> tuple[str, str] | None:
    """Return a consistent (unit_value, unit_name) pair when fully available."""
    unit_names: set[str] = set()
    unit_values: set[float] = set()
    for record in records:
        raw_name = record.get("unit_name")
        raw_value = record.get("unit_value")
        if raw_name is None or raw_value is None:
            return None

        unit_name = str(raw_name).strip().lower()
        if not unit_name:
            return None
        try:
            unit_value = float(raw_value)
        except (TypeError, ValueError):
            return None

        unit_names.add(unit_name)
        unit_values.add(unit_value)
        if len(unit_names) > 1 or len(unit_values) > 1:
            return None

    if not unit_names or not unit_values:
        return None
    return (f"{next(iter(unit_values)):g}", next(iter(unit_names)))


def _group_by_cluster_id(clusters: list[dict]) -> dict[int, list[dict]]:
    """Group records by cluster ID for per-cluster computations."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in clusters:
        grouped[int(record["cluster_id"])].append(record)
    return grouped


def _normalized_optional(value: object) -> str:
    """Normalize optional text values for consistency scoring."""
    return _clean_description_norm(value)


def _attribute_field_score(records: list[dict], field: str) -> float:
    """Score completeness and agreement for a single attribute field."""
    total = len(records)
    if total == 0:
        return 0.0

    normalized_values: list[str] = []
    for record in records:
        raw_value = record.get(field)
        if raw_value is None:
            continue
        if field == "unit_value":
            try:
                normalized_values.append(f"{float(raw_value):g}")
            except (TypeError, ValueError):
                continue
            continue
        cleaned = _normalized_optional(raw_value)
        if cleaned:
            normalized_values.append(cleaned)

    if not normalized_values:
        return 0.0
    completeness = len(normalized_values) / total
    is_consistent = len(set(normalized_values)) == 1
    return completeness if is_consistent else 0.0


def _attribute_consistency_score(records: list[dict]) -> float:
    """Compute attribute consistency score in [0, 1]."""
    scores = [
        _attribute_field_score(records, "unit_value"),
        _attribute_field_score(records, "unit_name"),
        _attribute_field_score(records, "unit_system"),
    ]
    if any(_normalized_optional(record.get("stock_code")) for record in records):
        scores.append(_attribute_field_score(records, "stock_code"))
    return sum(scores) / len(scores)


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("All feature vectors must have the same dimension.")
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(left * right for left, right in zip(vector_a, vector_b))
    return dot / (norm_a * norm_b)


def _similarity_mean_score(records: list[dict]) -> float:
    """Compute normalized mean pairwise cosine similarity in [0, 1]."""
    if len(records) <= 1:
        return 1.0
    vectors = [[float(value) for value in record.get("feature_vector", [])] for record in records]

    pair_scores: list[float] = []
    for index, vector_a in enumerate(vectors):
        for vector_b in vectors[index + 1 :]:
            cosine = _cosine_similarity(vector_a, vector_b)
            pair_scores.append((cosine + 1.0) / 2.0)
    if not pair_scores:
        return 1.0
    mean_score = sum(pair_scores) / len(pair_scores)
    return max(0.0, min(1.0, mean_score))


def canonicalize_with_confidence(clusters: list[dict]) -> tuple[dict[int, str], dict[int, float]]:
    """Generate labels and confidence scores for each cluster."""
    if not clusters:
        return {}, {}

    grouped = _group_by_cluster_id(clusters)

    labels: dict[int, str] = {}
    confidences: dict[int, float] = {}
    for cluster_id in sorted(grouped):
        records = grouped[cluster_id]
        base_description = _pick_base_description(records) or f"cluster {cluster_id}"
        unit_parts = _try_consistent_unit(records)
        labels[cluster_id] = (
            f"{base_description} {unit_parts[0]} {unit_parts[1]}"
            if unit_parts
            else base_description
        )
        attribute_consistency = _attribute_consistency_score(records)
        similarity_mean = _similarity_mean_score(records)
        confidence = (0.6 * attribute_consistency) + (0.4 * similarity_mean)
        confidences[cluster_id] = round(max(0.0, min(1.0, confidence)), 4)

    return labels, confidences


def canonicalize(clusters: list[dict]) -> dict[int, str]:
    """Generate a deterministic canonical label for each cluster."""
    labels, _ = canonicalize_with_confidence(clusters)
    return labels
