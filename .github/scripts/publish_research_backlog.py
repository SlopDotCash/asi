#!/usr/bin/env python3
"""Idempotently publish the research backlog as GitHub issues.

Dry-run is the default. Writing requires both ``--apply`` and a token with
Issues write permission in ``GITHUB_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs" / "research" / "implementation-backlog.json"


def _request(url: str, token: str, *, data: bytes | None = None) -> Any:
    method = "POST" if data is not None else "GET"
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "asi-research-backlog-publisher",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API returned {exc.code}: {detail}") from exc


def _issue_exists(repository: str, title: str, token: str) -> bool:
    query = urllib.parse.urlencode(
        {"q": f'repo:{repository} is:issue in:title "{title}"', "per_page": 100}
    )
    payload = _request(f"https://api.github.com/search/issues?{query}", token)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise RuntimeError("GitHub issue search did not return an item list")
    return any(
        isinstance(item, dict) and item.get("title") == title for item in payload["items"]
    )


def _body(issue: dict[str, Any], source: str) -> str:
    references = issue["references"]
    lines = [
        "### Objective",
        "",
        issue["scope"],
        "",
        "### Acceptance criteria",
        "",
        "- Pin the paper/code revision and record protocol differences before a long run.",
        "- Add failing-test-first unit/parity coverage and a mechanism-off reduction.",
        "- Match seeds, updates, observations, and allowed boundary/task information.",
        "- Report persistent bytes, environment/data steps, model queries, and timing telemetry.",
        "- Keep development results nonpromoting and retain negative outcomes.",
        "",
        f"Backlog source: `{source}`",
    ]
    if references:
        lines.extend(["", "### References", ""])
        lines.extend(f"- {reference}" for reference in references)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", default="elizaOS/asi")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    issues = manifest["issues"]
    if not args.apply:
        for issue in issues:
            print(f"CREATE {issue['title']}")
        print(f"dry-run: {len(issues)} issue(s); pass --apply to write", file=sys.stderr)
        return 0
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("--apply requires GITHUB_TOKEN with Issues write permission")
    created = 0
    skipped = 0
    for issue in issues:
        title = issue["title"]
        if _issue_exists(args.repo, title, token):
            print(f"SKIP {title}")
            skipped += 1
            continue
        data = json.dumps(
            {"title": title, "body": _body(issue, manifest["source"])},
            ensure_ascii=True,
        ).encode("utf-8")
        result = _request(f"https://api.github.com/repos/{args.repo}/issues", token, data=data)
        print(f"CREATE {result['html_url']}")
        created += 1
    print(f"created={created} skipped={skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
