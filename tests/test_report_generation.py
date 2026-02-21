"""Tests for markdown report rendering and report CLI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from generate_report import main as generate_report_main
from src.reporting import render_evaluation_report


def _sample_evaluation() -> dict:
    return {
        "num_records": 5,
        "num_clusters": 2,
        "cluster_sizes": {"0": 3, "1": 2},
        "labels": {"0": "item a", "1": "item b"},
        "cluster_stats": {
            "total_clusters": 2,
            "avg_cluster_size": 2.5,
            "largest_cluster": 3,
        },
        "suspect_clusters": [
            {
                "cluster_id": "1",
                "reasons": ["stock_code_mixed"],
                "size": 2,
                "risk_score": 0.175,
                "explanation": (
                    "Low inconsistency risk (0.1750) for 'item b': "
                    "detected stock_code_mixed."
                ),
            }
        ],
    }


def test_render_evaluation_report_includes_stats_and_suspects() -> None:
    report = render_evaluation_report(
        _sample_evaluation(),
        generated_at=datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "# Evaluation Report" in report
    assert "- Generated at: 2026-02-21T12:00:00Z" in report
    assert "- Total records: 5" in report
    assert "- Total clusters: 2" in report
    assert "- Average cluster size: 2.5" in report
    assert "- Largest cluster: 3" in report
    assert "| Cluster ID | Size | Risk Score | Reasons | Explanation |" in report
    assert "| 1 | 2 | 0.1750 | stock_code_mixed |" in report


def test_render_evaluation_report_shows_empty_state_for_no_suspects() -> None:
    payload = _sample_evaluation()
    payload["suspect_clusters"] = []

    report = render_evaluation_report(
        payload,
        generated_at=datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "## Suspect Clusters" in report
    assert "No suspect clusters detected." in report


def test_generate_report_cli_writes_default_output(monkeypatch) -> None:
    payload = json.dumps(_sample_evaluation())
    writes: dict[str, str] = {}

    def fake_read_text(self: Path, encoding: str = "utf-8") -> str:
        assert encoding == "utf-8"
        assert str(self) == "evaluation.json"
        return payload

    def fake_write_text(
        self: Path,
        text: str,
        encoding: str = "utf-8",
    ) -> int:
        assert encoding == "utf-8"
        writes[str(self)] = text
        return len(text)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(Path, "write_text", fake_write_text)

    exit_code = generate_report_main(["evaluation.json"])

    assert exit_code == 0
    assert "evaluation_report.md" in writes
    assert "## Key Stats" in writes["evaluation_report.md"]
    assert "## Suspect Clusters" in writes["evaluation_report.md"]


def test_generate_report_cli_writes_custom_output_path(monkeypatch) -> None:
    payload = json.dumps(_sample_evaluation())
    writes: dict[str, str] = {}

    def fake_read_text(self: Path, encoding: str = "utf-8") -> str:
        assert encoding == "utf-8"
        assert str(self) == "evaluation.json"
        return payload

    def fake_write_text(
        self: Path,
        text: str,
        encoding: str = "utf-8",
    ) -> int:
        assert encoding == "utf-8"
        writes[str(self)] = text
        return len(text)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(Path, "write_text", fake_write_text)

    exit_code = generate_report_main(["evaluation.json", "nested/my_report.md"])

    assert exit_code == 0
    output_key = "nested\\my_report.md" if "nested\\my_report.md" in writes else "nested/my_report.md"
    assert output_key in writes
    assert "# Evaluation Report" in writes[output_key]
