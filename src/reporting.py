"""Render evaluation payloads into markdown reports."""

from __future__ import annotations

from datetime import datetime, timezone


def _generated_timestamp(generated_at: datetime | None) -> str:
    """Return a stable UTC timestamp string for report headers."""
    current = generated_at or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def render_evaluation_report(
    evaluation: dict,
    *,
    generated_at: datetime | None = None,
) -> str:
    """Build markdown from an evaluation JSON payload."""
    cluster_stats = evaluation.get("cluster_stats", {})
    suspect_clusters = evaluation.get("suspect_clusters", [])

    lines = [
        "# Evaluation Report",
        "",
        f"- Generated at: {_generated_timestamp(generated_at)}",
        "",
        "## Key Stats",
        "",
        f"- Total records: {evaluation.get('num_records', 0)}",
        f"- Total clusters: {cluster_stats.get('total_clusters', evaluation.get('num_clusters', 0))}",
        f"- Average cluster size: {cluster_stats.get('avg_cluster_size', 0.0)}",
        f"- Largest cluster: {cluster_stats.get('largest_cluster', 0)}",
        "",
        "## Suspect Clusters",
        "",
    ]

    if not suspect_clusters:
        lines.append("No suspect clusters detected.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Cluster ID | Size | Risk Score | Reasons | Explanation |",
            "|---|---:|---:|---|---|",
        ]
    )
    for suspect in suspect_clusters:
        cluster_id = str(suspect.get("cluster_id", ""))
        size = int(suspect.get("size", 0))
        risk_score = float(suspect.get("risk_score", 0.0))
        reasons = ", ".join(str(reason) for reason in suspect.get("reasons", []))
        explanation = str(suspect.get("explanation", ""))
        lines.append(f"| {cluster_id} | {size} | {risk_score:.4f} | {reasons} | {explanation} |")

    lines.append("")
    return "\n".join(lines)
