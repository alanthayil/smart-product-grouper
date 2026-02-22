"""Clustering logic using cosine similarity and attribute matching."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

_DEFAULT_BLOCKING_SMALL_INPUT_CUTOFF = 300
_DEFAULT_RARE_TOKEN_MAX_FREQUENCY = 5


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


def _stock_code_match(record_a: dict, record_b: dict) -> bool:
    """Return True when both non-empty stock codes are equal."""
    stock_a = _normalized_optional(record_a.get("stock_code"))
    stock_b = _normalized_optional(record_b.get("stock_code"))
    return bool(stock_a and stock_b and stock_a == stock_b)


def _conflict_reasons(record_a: dict, record_b: dict) -> list[str]:
    """Return explicit attribute conflict reasons for a pair."""
    reasons: list[str] = []
    stock_a = _normalized_optional(record_a.get("stock_code"))
    stock_b = _normalized_optional(record_b.get("stock_code"))
    if stock_a and stock_b and stock_a != stock_b:
        reasons.append("stock_code_conflict")

    unit_name_a = _normalized_optional(record_a.get("unit_name"))
    unit_name_b = _normalized_optional(record_b.get("unit_name"))
    if unit_name_a and unit_name_b and unit_name_a != unit_name_b:
        reasons.append("unit_name_conflict")

    unit_system_a = _normalized_optional(record_a.get("unit_system"))
    unit_system_b = _normalized_optional(record_b.get("unit_system"))
    if unit_system_a and unit_system_b and unit_system_a != unit_system_b:
        reasons.append("unit_system_conflict")

    unit_value_a = record_a.get("unit_value")
    unit_value_b = record_b.get("unit_value")
    if unit_value_a is not None and unit_value_b is not None:
        if float(unit_value_a) != float(unit_value_b):
            reasons.append("unit_value_conflict")

    color_a = _normalized_optional(record_a.get("color"))
    color_b = _normalized_optional(record_b.get("color"))
    if color_a and color_b and color_a != color_b:
        reasons.append("color_conflict")

    quantity_total_a = record_a.get("quantity_total")
    quantity_total_b = record_b.get("quantity_total")
    if quantity_total_a is not None and quantity_total_b is not None:
        if float(quantity_total_a) != float(quantity_total_b):
            reasons.append("quantity_total_conflict")
    return reasons


def _compatible(record_a: dict, record_b: dict) -> bool:
    """Return True when no explicit attribute conflicts are detected."""
    return not _conflict_reasons(record_a, record_b)


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


def _all_pairs(count: int) -> list[tuple[int, int]]:
    """Return all unique index pairs for an input size."""
    pairs: list[tuple[int, int]] = []
    for source_index in range(count):
        for target_index in range(source_index + 1, count):
            pairs.append((source_index, target_index))
    return pairs


def _description_tokens(record: dict) -> list[str]:
    """Return normalized whitespace tokens from description text."""
    description = _normalized_optional(
        record.get("description_norm", record.get("description", ""))
    )
    return [token for token in description.split() if token]


def _add_block_pairs(
    blocks: dict[str, list[int]],
    candidate_pairs: set[tuple[int, int]],
) -> None:
    """Expand block membership lists into unique pair indices."""
    for members in blocks.values():
        if len(members) < 2:
            continue
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                source_index = members[left]
                target_index = members[right]
                candidate_pairs.add((min(source_index, target_index), max(source_index, target_index)))


def generate_candidate_pairs(
    records_or_features: list[dict],
    *,
    rare_token_max_frequency: int = _DEFAULT_RARE_TOKEN_MAX_FREQUENCY,
) -> list[tuple[int, int]]:
    """Generate candidate pairs from stock, prefix, and rare-token blocks."""
    if len(records_or_features) < 2:
        return []

    stock_blocks: dict[str, list[int]] = defaultdict(list)
    prefix_blocks: dict[str, list[int]] = defaultdict(list)
    rare_token_blocks: dict[str, list[int]] = defaultdict(list)
    tokens_per_record: list[set[str]] = []
    token_document_frequency: Counter[str] = Counter()

    for index, record in enumerate(records_or_features):
        stock_code = _normalized_optional(record.get("stock_code"))
        if stock_code:
            stock_blocks[stock_code].append(index)

        tokens = _description_tokens(record)
        if len(tokens) >= 2:
            prefix_blocks[f"p2:{' '.join(tokens[:2])}"].append(index)
        if len(tokens) >= 3:
            prefix_blocks[f"p3:{' '.join(tokens[:3])}"].append(index)

        unique_tokens = set(tokens)
        tokens_per_record.append(unique_tokens)
        token_document_frequency.update(unique_tokens)

    for index, unique_tokens in enumerate(tokens_per_record):
        for token in unique_tokens:
            if token_document_frequency[token] <= rare_token_max_frequency:
                rare_token_blocks[token].append(index)

    candidate_pairs: set[tuple[int, int]] = set()
    _add_block_pairs(stock_blocks, candidate_pairs)
    _add_block_pairs(prefix_blocks, candidate_pairs)
    _add_block_pairs(rare_token_blocks, candidate_pairs)
    return sorted(candidate_pairs)


def cluster(
    records_or_features: list[dict],
    *,
    similarity_threshold: float = 0.85,
    enable_blocking: bool = True,
    blocking_small_input_cutoff: int = _DEFAULT_BLOCKING_SMALL_INPUT_CUTOFF,
    rare_token_max_frequency: int = _DEFAULT_RARE_TOKEN_MAX_FREQUENCY,
    edge_debug: bool = False,
    edge_debug_top_k: int = 20,
    edge_debug_collector: dict[str, object] | None = None,
) -> list[dict]:
    """Assign cluster IDs from pairwise similarity and attribute gates."""
    if not records_or_features:
        return []

    vectors = [
        [float(value) for value in record.get("feature_vector", [])]
        for record in records_or_features
    ]

    adjacency: list[set[int]] = [set() for _ in records_or_features]

    use_blocking = enable_blocking and len(records_or_features) > blocking_small_input_cutoff
    if use_blocking:
        candidate_pairs = generate_candidate_pairs(
            records_or_features,
            rare_token_max_frequency=rare_token_max_frequency,
        )
        similarities = None
    else:
        candidate_pairs = _all_pairs(len(records_or_features))
        similarities = _pairwise_cosine_similarity(vectors)

    edge_debug_rows: list[dict[str, object]] = []
    collect_edge_debug = edge_debug and edge_debug_collector is not None
    for source_index, target_index in candidate_pairs:
        source_record = records_or_features[source_index]
        target_record = records_or_features[target_index]
        similarity = (
            _cosine_similarity(vectors[source_index], vectors[target_index])
            if similarities is None
            else similarities[source_index][target_index]
        )
        stock_match = _stock_code_match(source_record, target_record)
        conflict_reasons = _conflict_reasons(source_record, target_record)
        compatible = not conflict_reasons
        edge_decision = stock_match or (similarity >= similarity_threshold and compatible)
        if collect_edge_debug:
            edge_debug_rows.append(
                {
                    "record_id_i": str(source_record.get("record_id", f"record-{source_index}")),
                    "record_id_j": str(target_record.get("record_id", f"record-{target_index}")),
                    "description_i": str(
                        source_record.get("description_norm", source_record.get("description", ""))
                    ),
                    "description_j": str(
                        target_record.get("description_norm", target_record.get("description", ""))
                    ),
                    "similarity": float(similarity),
                    "stock_code_match": bool(stock_match),
                    "attrs_i": {
                        "stock_code": source_record.get("stock_code"),
                        "color": source_record.get("color"),
                        "quantity_total": source_record.get("quantity_total"),
                        "unit_name": source_record.get("unit_name"),
                        "unit_system": source_record.get("unit_system"),
                        "unit_value": source_record.get("unit_value"),
                    },
                    "attrs_j": {
                        "stock_code": target_record.get("stock_code"),
                        "color": target_record.get("color"),
                        "quantity_total": target_record.get("quantity_total"),
                        "unit_name": target_record.get("unit_name"),
                        "unit_system": target_record.get("unit_system"),
                        "unit_value": target_record.get("unit_value"),
                    },
                    "conflict_reasons": conflict_reasons,
                    "edge_decision": bool(edge_decision),
                }
            )
        if not edge_decision:
            continue
        adjacency[source_index].add(target_index)
        adjacency[target_index].add(source_index)

    if collect_edge_debug:
        safe_top_k = max(0, int(edge_debug_top_k))
        ranked_rows = sorted(edge_debug_rows, key=lambda row: float(row["similarity"]), reverse=True)
        edge_debug_collector["candidate_pairs_evaluated"] = len(edge_debug_rows)
        edge_debug_collector["top_k"] = safe_top_k
        edge_debug_collector["top_candidate_pairs"] = ranked_rows[:safe_top_k]

    cluster_ids = _build_connected_components(adjacency)
    clustered_records: list[dict] = []
    for index, record in enumerate(records_or_features):
        clustered_record: dict[str, object] = {
            "record_id": str(record.get("record_id", f"record-{index}")),
            "cluster_id": cluster_ids[index],
            "description_norm": str(record.get("description_norm", "")),
            "feature_vector": vectors[index],
        }
        stock_code = str(record.get("stock_code", "")).strip()
        if stock_code:
            clustered_record["stock_code"] = stock_code
        for field in ("color", "quantity_total", "unit_value", "unit_name", "unit_system"):
            if record.get(field) is not None:
                clustered_record[field] = record[field]
        clustered_records.append(clustered_record)
    return clustered_records
