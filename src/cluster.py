"""Nearest-neighbor similarity search over extracted feature vectors."""

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


def cluster(
    records_or_features: list[dict],
    *,
    top_k: int = 5,
    similarity_threshold: float = 0.80,
) -> list[dict]:
    """Return top-k cosine neighbors for each feature record."""
    if not records_or_features:
        return []

    vectors = [
        [float(value) for value in record.get("feature_vector", [])]
        for record in records_or_features
    ]
    similarities = _pairwise_cosine_similarity(vectors)

    results: list[dict] = []
    for source_index, record in enumerate(records_or_features):
        ranked_candidates: list[tuple[float, int, str]] = []
        for target_index, similarity in enumerate(similarities[source_index]):
            if source_index == target_index:
                continue
            if similarity < similarity_threshold:
                continue
            target_record = records_or_features[target_index]
            target_record_id = str(target_record.get("record_id", f"record-{target_index}"))
            ranked_candidates.append((similarity, target_index, target_record_id))

        ranked_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        neighbors = [
            {"record_id": record_id, "similarity": similarity}
            for similarity, _, record_id in ranked_candidates[:top_k]
        ]

        result_record = {
            "record_id": str(record.get("record_id", f"record-{source_index}")),
            "description_norm": str(record.get("description_norm", "")),
            "feature_vector": vectors[source_index],
            "neighbors": neighbors,
        }
        results.append(result_record)
    return results
