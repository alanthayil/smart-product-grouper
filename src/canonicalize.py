"""Produce canonical labels for each cluster."""

from __future__ import annotations

from collections import Counter, defaultdict
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


def canonicalize(clusters: list[dict]) -> dict[int, str]:
    """Generate a deterministic canonical label for each cluster."""
    if not clusters:
        return {}

    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in clusters:
        grouped[int(record["cluster_id"])].append(record)

    labels: dict[int, str] = {}
    for cluster_id in sorted(grouped):
        records = grouped[cluster_id]
        base_description = _pick_base_description(records) or f"cluster {cluster_id}"
        unit_parts = _try_consistent_unit(records)
        labels[cluster_id] = (
            f"{base_description} {unit_parts[0]} {unit_parts[1]}"
            if unit_parts
            else base_description
        )
    return labels
