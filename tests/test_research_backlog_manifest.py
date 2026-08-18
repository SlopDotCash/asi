from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "implementation-backlog.json"


def test_research_backlog_is_unique_and_issue_ready() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == "asi.research_implementation_backlog.v1"
    assert payload["repository"] == "elizaOS/asi"
    assert payload["published"] == {
        "date": "2026-08-17",
        "first_issue": 1559,
        "last_issue": 1586,
    }
    issues = payload["issues"]
    assert len(issues) == 28
    titles = [issue["title"] for issue in issues]
    assert len(titles) == len(set(titles))
    for issue in issues:
        assert issue["scope"].endswith(".")
        assert all(reference.startswith("https://") for reference in issue["references"])
