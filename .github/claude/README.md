# Claude PR triage pipeline

`.github/workflows/claude-pr-triage.yml` runs Claude against every opened,
reopened, updated, or ready-for-review pull request (and on demand via
`workflow_dispatch` with a PR number).

## Outcomes

| Verdict | Action (deterministic workflow steps, not the model) |
|---|---|
| `malicious` | Close the PR, label `flagged-malicious`, comment, and block the author org-wide (requires `PR_TRIAGE_ADMIN_TOKEN`; otherwise warns for a manual ban). |
| `auto_merge` | Squash-merge the reviewed commit if the merge gates pass. Blocked merges receive `needs-human-review`. |
| `needs_research` | A deeper Claude pass researches the implementation against the base branch, posts a review; if it confirms correct + non-controversial it merges under the same gates, else labels `needs-human-review`. |

## Setup

1. Repo secret `CLAUDE_CODE_OAUTH_TOKEN` — required for both Claude steps.
   This uses your Claude subscription (Pro/Max) instead of API billing:
   run `claude setup-token` locally, then save the printed token as the
   secret. Tokens expire after roughly a year; re-run `claude setup-token`
   to rotate.
2. Repo secret `PR_TRIAGE_ADMIN_TOKEN` (optional) — an org-owner PAT with
   `admin:org`, used only for `PUT /orgs/{org}/blocks/{username}` when a PR
   is flagged malicious. Without it the PR is still closed and labeled.
3. Keep the required status checks on `main` so every merge passes CI.

## Safety properties

- PR head code is never checked out, built, or executed; only the base
  branch is checked out. Both the path list and diff come from the captured
  commit, and the review stops if the diff exceeds its size limit.
- The prompts treat PR title, body, and diff as untrusted. Claude's tools
  are limited to reading, searching, and writing files, and its full
  execution output stays out of the public workflow log.
- Both reviews require a successful action step and a successful Claude
  execution. An invalid classification verdict requests research. A
  failed research execution or invalid research verdict leaves the pull
  request for manual review.
- A gate independent of the model blocks automated merges for `.github/`,
  `outputs/`, `.claude/`, `.husky/`, `.mcp.json`, `.claude.json`,
  `.gitmodules`, `.gitattributes`, `.ripgreprc`, `CLAUDE.md`,
  `CLAUDE.local.md`, `AGENTS.md`, `VENDORING.md`, `pyproject.toml`, and
  `alberta_framework/evaluation/evidence_manifest.py`.
- The workflow requires a completed, successful GitHub Actions `ci-ok`
  check on the reviewed head. Immediately before the direct merge it
  confirms the PR still targets the reviewed branch and that branch still
  points to the reviewed base commit. GitHub atomically matches only the
  reviewed head SHA during the merge. The base verification is a preflight.
