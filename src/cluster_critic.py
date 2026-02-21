"""Analyze cluster-level inconsistency risk and provide explanations."""

from __future__ import annotations

from src.canonicalize import _similarity_mean_score

_MAX_REASON_COUNT = 4


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

    if len(_distinct_non_empty_values(records, "stock_code")) > 1:
        reasons.append("stock_code_mixed")
    if len(_distinct_non_empty_values(records, "unit_name")) > 1:
        reasons.append("unit_name_mixed")
    if len(_distinct_non_empty_values(records, "unit_system")) > 1:
        reasons.append("unit_system_mixed")
    if len(_distinct_unit_values(records)) > 1:
        reasons.append("unit_value_mixed")

    return reasons


def _severity_label(risk_score: float) -> str:
    """Map score bands into a human-readable severity label."""
    if risk_score >= 0.67:
        return "high"
    if risk_score >= 0.34:
        return "medium"
    return "low"


def analyze_cluster(records: list[dict], label: str | None = None) -> dict:
    """Analyze one cluster and return risk score + explanation."""
    if not records:
        return {
            "risk_score": 0.0,
            "explanation": "Low inconsistency risk: cluster has no records to analyze.",
            "signals": [],
        }

    reasons = _suspect_reasons(records)
    reason_risk = len(reasons) / _MAX_REASON_COUNT
    similarity_score = _similarity_mean_score(records)
    similarity_risk = 1.0 - similarity_score
    risk_score = round(max(0.0, min(1.0, (0.7 * reason_risk) + (0.3 * similarity_risk))), 4)

    severity = _severity_label(risk_score)
    if reasons:
        explanation = (
            f"{severity.capitalize()} inconsistency risk ({risk_score:.4f})"
            f" for '{label or 'unlabeled cluster'}': "
            f"detected {', '.join(reasons)}."
        )
    else:
        explanation = (
            f"{severity.capitalize()} inconsistency risk ({risk_score:.4f})"
            f" for '{label or 'unlabeled cluster'}': "
            "no mixed attribute signals detected."
        )

    signals = [*reasons]
    if similarity_risk > 0.35:
        signals.append("semantic_similarity_low")

    return {
        "risk_score": risk_score,
        "explanation": explanation,
        "signals": signals,
    }
