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
MERGE_STEPS = ("Merge obviously-correct PR", "Post review and act on research verdict")

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
    case "$*" in
      *"sha=$LIVE_HEAD -f merge_method=squash"*) ;;
      *) return 1 ;;
    esac
    if [ "$DIRECT_FAILURE" = 1 ]; then return 1; fi
    printf 'true\n'
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
        research_outcome="success", review="fixture review\n",
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
                "research_verdict.json": json.dumps({"verdict": "merge"}),
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
                "OVERSIZED": "1" if oversized else "0",
                "DIRECT_FAILURE": "1" if direct_failure else "0",
                "RESEARCH_OUTCOME": research_outcome,
                "PROTECTED_PATHS": "true" if protected else "false",
                "PR_NUMBER": "1", "REPO": "fixture/repo", "MSYS_NO_PATHCONV": "1",
            })
            if reviewed is not None:
                env["REVIEWED_HEAD"] = reviewed
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
        run, files = self.run_step("Fetch PR data (as data, not code)")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(files["fetch-args"], f"fetch --no-tags origin {SHA}\n")
        self.assertIn(f"reviewed_head={SHA}\n", files["github-output"])
        self.assertEqual(files["changed_files.txt"], "README.md\n")
        self.assertEqual(files["diff.patch"], "fixture diff\n")
        self.assertIn("protected=false", files["github-output"])
        run, files = self.run_step(
            "Fetch PR data (as data, not code)", diff_path=".github/workflows/fixture.yml"
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("protected=true", files["github-output"])

    def test_fetch_fails_closed_on_unsafe_inputs(self):
        for opts in (
            {"live": OTHER_SHA}, {"metadata": None}, {"metadata": ""},
            {"metadata": f"{SHA}\nINJECTED=value"}, {"base_branch": "feature"},
            {"diff_path": ".github/workflows/with\nnewline"}, {"oversized": True},
        ):
            with self.subTest(opts=opts):
                run, files = self.run_step("Fetch PR data (as data, not code)", **opts)
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn("github-output", files)

    def test_same_head_uses_one_exact_direct_merge(self):
        for name in MERGE_STEPS:
            with self.subTest(name=name):
                run, files = self.run_step(name)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(len(files["merge-calls"].splitlines()), 1)
                self.assertIn(f"sha={SHA} -f merge_method=squash", files["merge-calls"])
                self.assertNotIn("edit-calls", files)

    def test_changed_head_or_blocked_merge_needs_manual_review(self):
        for name in MERGE_STEPS:
            for opts in ({"live": OTHER_SHA}, {"direct_failure": True}):
                with self.subTest(name=name, opts=opts):
                    run, files = self.run_step(name, **opts)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(len(files["merge-calls"].splitlines()), 1)
                    self.assertIn(f"sha={SHA}", files["merge-calls"])
                    self.assertIn("--add-label needs-human-review", files["edit-calls"])

    def test_missing_reviewed_head_disallows_merge(self):
        for name in MERGE_STEPS:
            with self.subTest(name=name):
                run, files = self.run_step(name, reviewed=None)
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn("merge-calls", files)

    def test_research_failure_or_empty_report_disallows_merge(self):
        for opts in ({"research_outcome": "failure"}, {"review": ""}):
            with self.subTest(opts=opts):
                run, files = self.run_step(MERGE_STEPS[1], **opts)
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn("merge-calls", files)
                self.assertNotIn("review-calls", files)
                self.assertIn("--add-label needs-human-review", files["edit-calls"])

    def test_research_refuses_protected_paths(self):
        run, files = self.run_step(MERGE_STEPS[1], protected=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("merge-calls", files)
        self.assertIn("--add-label needs-human-review", files["edit-calls"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
