"""Clustering logic using cosine similarity and attribute matching."""

from __future__ import annotations

import math


def _vector_norm(vector: list[float]) -> float:
    """Compute Euclidean norm for a vector."""
    return math.sqrt(sum(value * value for value in vector))


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("All feature vectors must have the same dimension.")
    norm_a = _vector_norm(vector_a)
    norm_b = _vector_norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(left * right for left, right in zip(vector_a, vector_b))
    return dot / (norm_a * norm_b)


def _pairwise_cosine_similarity(vectors: list[list[float]]) -> list[list[float]]:
    """Build dense pairwise cosine similarity matrix."""
    similarities: list[list[float]] = []
    for vector_a in vectors:
        row: list[float] = []
        for vector_b in vectors:
            row.append(_cosine_similarity(vector_a, vector_b))
        similarities.append(row)
    return similarities


def _normalized_optional(value: object) -> str:
    """Normalize optional text-like values for null-safe comparisons."""
    return str(value or "").strip().lower()


def _attributes_match(record_a: dict, record_b: dict) -> bool:
    """Apply hybrid stock-code / unit attribute matching rule."""
    stock_a = _normalized_optional(record_a.get("stock_code"))
    stock_b = _normalized_optional(record_b.get("stock_code"))
    if stock_a and stock_b:
        return stock_a == stock_b

    unit_name_a = _normalized_optional(record_a.get("unit_name"))
    unit_name_b = _normalized_optional(record_b.get("unit_name"))
    unit_system_a = _normalized_optional(record_a.get("unit_system"))
    unit_system_b = _normalized_optional(record_b.get("unit_system"))
    return (
        bool(unit_name_a)
        and bool(unit_system_a)
        and unit_name_a == unit_name_b
        and unit_system_a == unit_system_b
    )


def _build_connected_components(adjacency: list[set[int]]) -> list[int]:
    """Return deterministic component IDs for each node index."""
    cluster_ids = [-1] * len(adjacency)
    visited: set[int] = set()
    next_cluster_id = 0

    for seed in range(len(adjacency)):
        if seed in visited:
            continue
        stack = [seed]
        visited.add(seed)
        while stack:
            node = stack.pop()
            cluster_ids[node] = next_cluster_id
            for neighbor in sorted(adjacency[node]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        next_cluster_id += 1
    return cluster_ids


def cluster(
    records_or_features: list[dict],
    *,
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Assign cluster IDs from pairwise similarity and attribute gates."""
    if not records_or_features:
        return []

    vectors = [
        [float(value) for value in record.get("feature_vector", [])]
        for record in records_or_features
    ]
    similarities = _pairwise_cosine_similarity(vectors)

    adjacency: list[set[int]] = [set() for _ in records_or_features]
    for source_index in range(len(records_or_features)):
        for target_index in range(source_index + 1, len(records_or_features)):
            similarity = similarities[source_index][target_index]
            if similarity < similarity_threshold:
                continue
            source_record = records_or_features[source_index]
            target_record = records_or_features[target_index]
            if not _attributes_match(source_record, target_record):
                continue
            adjacency[source_index].add(target_index)
            adjacency[target_index].add(source_index)

    cluster_ids = _build_connected_components(adjacency)
    return [
        {
            "record_id": str(record.get("record_id", f"record-{index}")),
            "cluster_id": cluster_ids[index],
            "description_norm": str(record.get("description_norm", "")),
            "feature_vector": vectors[index],
        }
        for index, record in enumerate(records_or_features)
    ]
