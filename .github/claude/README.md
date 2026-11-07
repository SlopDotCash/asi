# Claude PR triage pipeline

`.github/workflows/claude-pr-triage.yml` runs Claude against every opened,
reopened, updated, or ready-for-review pull request (and on demand via
`workflow_dispatch` with a PR number).

## Outcomes

| Verdict | Action (deterministic workflow steps, not the model) |
|---|---|
| `malicious` | Close the PR, label `flagged-malicious`, comment, and block the author org-wide (requires `PR_TRIAGE_ADMIN_TOKEN`; otherwise warns for a manual ban). |
| `auto_merge` | Squash-merge the reviewed commit if required checks pass. Blocked merges receive `needs-human-review`. |
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
- The prompts treat PR title/body/diff as untrusted; verdicts are parsed
  from strict JSON, and a missing/invalid verdict fails safe to
  `needs-human-review`.
- Changes under `.github/`, `outputs/`, `CLAUDE.md`, `AGENTS.md`,
  `VENDORING.md`, `pyproject.toml`, or
  `alberta_framework/evaluation/evidence_manifest.py` are never auto-merged,
  enforced by a bash gate independent of the model.
- A direct merge request gives GitHub the reviewed head SHA. GitHub checks
  that SHA and the required branch checks at merge time.
- A failed or empty research step marks the workflow failed and leaves
  the pull request for manual review.
