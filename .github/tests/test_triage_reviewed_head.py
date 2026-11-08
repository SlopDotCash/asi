"""Exercise the privileged triage steps without credentials or network calls."""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

SHA = "a" * 40
OTHER_SHA = "b" * 40
BASE_SHA = "c" * 40
WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "claude-pr-triage.yml"
FETCH_STEP = "Fetch PR data (as data, not code)"
MERGE_STEP = "Merge the reviewed PR"
RESEARCH_STEP = "Post review and parse research verdict"
TRIAGE_STEP = "Parse verdict"
CHECK_PATH = (
    f"repos/fixture/repo/commits/{SHA}/check-runs"
    "?check_name=ci-ok&filter=latest&per_page=100"
)
CI_CHECK = {
    "id": 1, "name": "ci-ok", "head_sha": SHA,
    "status": "completed", "conclusion": "success", "app": {"slug": "github-actions"},
}

STUBS = r"""
git() {
  case "$1" in
    fetch) printf '%s\n' "$*" > "$WORKDIR/fetch-args" ;;
    rev-parse) printf '%s\n' "$BASE_SHA" ;;
    diff)
      case "$*" in
        *"$BASE_SHA...$METADATA_HEAD") ;;
        *) return 97 ;;
      esac
      case "$*" in
        *--name-only*) printf '%s\0' "$DIFF_PATH" ;;
        *) if [ "$OVERSIZED" = 1 ]; then
             head -c 400001 /dev/zero | tr '\000' 'x'
           else
             printf 'fixture diff\n'
           fi ;;
      esac ;;
    *) return 97 ;;
  esac
}
gh() {
  if [ "$1 $2" = "pr view" ]; then
    case "$*" in
      *--jq*) printf '%s\n' "$LIVE_HEAD" ;;
      *) cat "$WORKDIR/metadata-response.json" ;;
    esac
  elif [ "$1" = api ] && [ "$2 $3" = "--method PUT" ]; then
    printf '%s\n' "$*" >> "$WORKDIR/merge-calls"
    printf '%s\n' "$4" >> "$WORKDIR/api-paths"
    case "$*" in
      *"sha=$LIVE_HEAD -f merge_method=squash"*) ;;
      *) return 1 ;;
    esac
    if [ "$DIRECT_FAILURE" = 1 ]; then return 1; fi
    printf 'true\n'
  elif [ "$1" = api ]; then
    printf '%s\n' "$2" >> "$WORKDIR/api-paths"
    case "$2 $3 $4" in
      "$CHECK_PATH  ") cat "$WORKDIR/checks-response.json" ;;
      "repos/$REPO/pulls/$PR_NUMBER --jq .base.ref") printf '%s\n' "$LIVE_BASE_REF" ;;
      "repos/$REPO/branches/main --jq .commit.sha") printf '%s\n' "$LIVE_BASE_SHA" ;;
      *) return 97 ;;
    esac
  elif [ "$1 $2" = "pr edit" ]; then
    printf '%s\n' "$*" >> "$WORKDIR/edit-calls"
  elif [ "$1 $2" = "pr review" ]; then
    printf '%s\n' "$*" >> "$WORKDIR/review-calls"
  else
    return 97
  fi
}
"""


def shell_step(name):
    section = WORKFLOW.read_text(encoding="utf-8").split(f"      - name: {name}\n", 1)[1]
    section = section.split("\n      - name:", 1)[0]
    return textwrap.dedent(section.split("        run: |\n", 1)[1])


class TriageHeadTests(unittest.TestCase):
    def run_step(
        self, name, *, live=SHA, metadata=SHA, reviewed=SHA, direct_failure=False,
        protected=False, diff_path="README.md", oversized=False, base_branch="main",
        reviewed_base_ref="main", reviewed_base_sha=BASE_SHA, live_base_ref="main",
        live_base_sha=BASE_SHA, check_runs=(CI_CHECK,), triage_outcome="success",
        triage_conclusion="success", research_outcome="success", research_conclusion="success",
        triage_verdict='{"verdict":"auto_merge"}', research_verdict='{"verdict":"merge"}',
        review="fixture review\n",
    ):
        bash = os.environ.get("TRIAGE_TEST_BASH") or shutil.which("bash")
        jq = os.environ.get("TRIAGE_TEST_JQ") or shutil.which("jq")
        self.assertTrue(bash and jq, "bash and jq are required")
        with tempfile.TemporaryDirectory(prefix="triage-head-") as temporary:
            root = Path(temporary)
            fixtures = {
                "metadata-response.json": json.dumps({
                    "headRefOid": metadata, "baseRefName": base_branch,
                    "author": {"login": "fixture-author"},
                }),
                "checks-response.json": json.dumps({"check_runs": check_runs}),
                "verdict.json": triage_verdict,
                "research_verdict.json": research_verdict,
                "review.md": review,
                "changed_files.txt": "README.md\n",
            }
            for filename, contents in fixtures.items():
                (root / filename).write_text(contents, encoding="utf-8")
            script = root / "step.sh"
            script.write_text(STUBS + shell_step(name), encoding="utf-8", newline="\n")
            env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}
            env.update({
                "PATH": os.pathsep.join((str(Path(jq).parent), str(Path(bash).parent),
                                         "/usr/bin", "/bin")),
                "WORKDIR": root.as_posix(),
                "GITHUB_OUTPUT": (root / "github-output").as_posix(),
                "LIVE_HEAD": live, "METADATA_HEAD": metadata or "", "BASE_SHA": BASE_SHA,
                "DEFAULT_BRANCH": "main", "DIFF_PATH": diff_path,
                "CHECK_PATH": CHECK_PATH,
                "LIVE_BASE_REF": live_base_ref, "LIVE_BASE_SHA": live_base_sha,
                "OVERSIZED": "1" if oversized else "0",
                "DIRECT_FAILURE": "1" if direct_failure else "0",
                "TRIAGE_OUTCOME": triage_outcome, "TRIAGE_CONCLUSION": triage_conclusion,
                "RESEARCH_OUTCOME": research_outcome,
                "RESEARCH_CONCLUSION": research_conclusion,
                "PROTECTED_PATHS": "true" if protected else "false",
                "PR_NUMBER": "1", "REPO": "fixture/repo", "MSYS_NO_PATHCONV": "1",
            })
            if reviewed is not None:
                env["REVIEWED_HEAD"] = reviewed
            if reviewed_base_ref is not None:
                env["REVIEWED_BASE_REF"] = reviewed_base_ref
            if reviewed_base_sha is not None:
                env["REVIEWED_BASE_SHA"] = reviewed_base_sha
            run = subprocess.run(
                [bash, "--noprofile", "--norc", "-e", str(script)],
                env=env, capture_output=True, text=True, timeout=10,
            )
            files = {
                file.name: file.read_text(encoding="utf-8")
                for file in root.iterdir()
                if file.is_file() and file.name != "step.sh"
            }
            return run, files

    def test_fetch_pins_paths_and_patch_to_the_same_commit(self):
        run, files = self.run_step(FETCH_STEP)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(files["fetch-args"], f"fetch --no-tags origin {SHA}\n")
        self.assertIn(f"reviewed_head={SHA}\n", files["github-output"])
        self.assertIn("reviewed_base_ref=main\n", files["github-output"])
        self.assertIn(f"reviewed_base_sha={BASE_SHA}\n", files["github-output"])
        self.assertEqual(files["changed_files.txt"], "README.md\n")
        self.assertEqual(files["diff.patch"], "fixture diff\n")
        self.assertIn("protected=false", files["github-output"])

    def test_fetch_flags_protected_paths(self):
        for path in (
            ".github", ".github/workflows/fixture.yml", "outputs", "outputs/fixture.json",
            ".claude", ".claude/settings.json", ".husky", ".husky/pre-commit",
            ".mcp.json", ".claude.json", ".gitmodules", ".gitattributes", ".ripgreprc",
            "CLAUDE.md", "CLAUDE.local.md", "AGENTS.md", "VENDORING.md", "pyproject.toml",
            "alberta_framework/evaluation/evidence_manifest.py",
        ):
            with self.subTest(path=path):
                run, files = self.run_step(FETCH_STEP, diff_path=path)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("protected=true", files["github-output"])

    def test_fetch_fails_closed_on_unsafe_inputs(self):
        for opts in (
            {"live": OTHER_SHA}, {"metadata": None}, {"metadata": ""},
            {"metadata": f"{SHA}\nINJECTED=value"}, {"base_branch": "feature"},
            {"diff_path": ".github/workflows/with\nnewline"}, {"oversized": True},
        ):
            with self.subTest(opts=opts):
                run, files = self.run_step(FETCH_STEP, **opts)
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn("github-output", files)

    def test_same_head_uses_one_exact_direct_merge(self):
        run, files = self.run_step(MERGE_STEP)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(files["api-paths"].splitlines(), [
            CHECK_PATH, "repos/fixture/repo/pulls/1", "repos/fixture/repo/branches/main",
            "repos/fixture/repo/pulls/1/merge",
        ])
        self.assertIn(f"sha={SHA} -f merge_method=squash", files["merge-calls"])
        self.assertNotIn("edit-calls", files)

    def test_changed_head_or_blocked_merge_needs_manual_review(self):
        for opts in ({"live": OTHER_SHA}, {"direct_failure": True}):
            with self.subTest(opts=opts):
                run, files = self.run_step(MERGE_STEP, **opts)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(len(files["merge-calls"].splitlines()), 1)
                self.assertIn(f"sha={SHA}", files["merge-calls"])
                self.assertIn("--add-label needs-human-review", files["edit-calls"])

    def test_missing_reviewed_refs_disallow_merge(self):
        for field in ("reviewed", "reviewed_base_ref", "reviewed_base_sha"):
            with self.subTest(field=field):
                run, files = self.run_step(MERGE_STEP, **{field: None})
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn("merge-calls", files)

    def test_changed_base_needs_manual_review(self):
        for opts in ({"live_base_ref": "feature"}, {"live_base_sha": OTHER_SHA}):
            with self.subTest(opts=opts):
                run, files = self.run_step(MERGE_STEP, **opts)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertNotIn("merge-calls", files)
                self.assertIn("--add-label needs-human-review", files["edit-calls"])

    def test_missing_or_unsuccessful_head_ci_needs_manual_review(self):
        for checks in (
            [], [{**CI_CHECK, "name": "elsewhere"}], [{**CI_CHECK, "head_sha": OTHER_SHA}],
            [{**CI_CHECK, "app": {"slug": "other-app"}}],
            [{**CI_CHECK, "conclusion": "failure"}],
            [CI_CHECK, {**CI_CHECK, "id": 2, "status": "in_progress", "conclusion": None}],
        ):
            with self.subTest(checks=checks):
                run, files = self.run_step(MERGE_STEP, check_runs=checks)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertNotIn("merge-calls", files)
                self.assertIn("--add-label needs-human-review", files["edit-calls"])

    def test_classification_needs_both_success_signals_and_valid_verdict(self):
        for opts in (
            {}, {"triage_outcome": "failure"}, {"triage_conclusion": "failure"},
            {"triage_conclusion": ""}, {"triage_verdict": "[]"},
        ):
            with self.subTest(opts=opts):
                run, files = self.run_step(TRIAGE_STEP, **opts)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(files["github-output"],
                                 "verdict=needs_research\n" if opts else "verdict=auto_merge\n")

    def test_research_failure_or_empty_report_disallows_merge(self):
        for opts in (
            {"research_outcome": "failure"}, {"research_conclusion": "failure"},
            {"research_conclusion": ""}, {"review": ""},
        ):
            with self.subTest(opts=opts):
                run, files = self.run_step(RESEARCH_STEP, **opts)
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn("github-output", files)
                self.assertNotIn("review-calls", files)
                self.assertIn("--add-label needs-human-review", files["edit-calls"])

    def test_research_needs_valid_verdict_and_unprotected_paths(self):
        for opts in ({}, {"protected": True}, {"research_verdict": "[]"}):
            with self.subTest(opts=opts):
                run, files = self.run_step(RESEARCH_STEP, **opts)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("--comment", files["review-calls"])
                self.assertEqual(files["github-output"],
                                 "verdict=manual_review\n" if opts else "verdict=merge\n")
                self.assertEqual("edit-calls" in files, bool(opts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
