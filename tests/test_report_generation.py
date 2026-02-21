"""Tests for markdown report rendering and report CLI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile

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
            {"cluster_id": "1", "reasons": ["stock_code_mixed"], "size": 2}
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
    assert "| 1 | 2 | stock_code_mixed |" in report


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
    with tempfile.TemporaryDirectory(dir=".") as workdir:
        workdir_path = Path(workdir)
        input_path = workdir_path / "evaluation.json"
        input_path.write_text(json.dumps(_sample_evaluation()), encoding="utf-8")
        monkeypatch.chdir(workdir_path)

        exit_code = generate_report_main([str(input_path)])

        output_path = workdir_path / "evaluation_report.md"
        assert exit_code == 0
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "## Key Stats" in content
        assert "## Suspect Clusters" in content


def test_generate_report_cli_writes_custom_output_path() -> None:
    with tempfile.TemporaryDirectory(dir=".") as workdir:
        workdir_path = Path(workdir)
        input_path = workdir_path / "evaluation.json"
        output_path = workdir_path / "nested" / "my_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(json.dumps(_sample_evaluation()), encoding="utf-8")

        exit_code = generate_report_main([str(input_path), str(output_path)])

        assert exit_code == 0
        assert output_path.exists()
        assert "# Evaluation Report" in output_path.read_text(encoding="utf-8")
