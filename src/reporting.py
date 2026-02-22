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
    edge_debug = evaluation.get("edge_debug")

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
    else:
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
            lines.append(
                f"| {cluster_id} | {size} | {risk_score:.4f} | {reasons} | {explanation} |"
            )

    lines.append("")
    if isinstance(edge_debug, dict):
        evaluated = int(edge_debug.get("candidate_pairs_evaluated", 0))
        top_k = int(edge_debug.get("top_k", 0))
        top_pairs = edge_debug.get("top_candidate_pairs", [])
        lines.extend(
            [
                "## Edge Debug Report",
                "",
                f"- Candidate pairs evaluated: {evaluated}",
                f"- Top pairs requested: {top_k}",
                "",
            ]
        )
        if not isinstance(top_pairs, list) or not top_pairs:
            lines.append("No edge debug rows available.")
            lines.append("")
            return "\n".join(lines)
        lines.extend(
            [
                "| record_i | record_j | similarity | stock_match | conflicts | edge_decision |",
                "|---|---|---:|---:|---|---:|",
            ]
        )
        for pair in top_pairs:
            if not isinstance(pair, dict):
                continue
            record_i = str(pair.get("record_id_i", ""))
            record_j = str(pair.get("record_id_j", ""))
            similarity = float(pair.get("similarity", 0.0))
            stock_match = bool(pair.get("stock_code_match", False))
            conflicts = ", ".join(str(reason) for reason in pair.get("conflict_reasons", []))
            edge_decision = bool(pair.get("edge_decision", False))
            lines.append(
                f"| {record_i} | {record_j} | {similarity:.4f} | {stock_match} | {conflicts} | {edge_decision} |"
            )
        lines.append("")
    return "\n".join(lines)
