"""Exercise triage shell steps offline, including a head update during review.

Run directly with Python's standard library; no JAX, credentials, or API calls.
TRIAGE_TEST_BASH/TRIAGE_TEST_JQ can select native tools on Windows.
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

SHA = "a" * 40
NEW_SHA = "b" * 40
WORKFLOW = Path(os.environ.get(
    "TRIAGE_WORKFLOW_UNDER_TEST",
    str(Path(__file__).resolve().parents[1] / "workflows" / "claude-pr-triage.yml"),
))
BASH = os.environ.get("TRIAGE_TEST_BASH") or shutil.which("bash")
JQ = os.environ.get("TRIAGE_TEST_JQ") or shutil.which("jq")
MERGE_STEPS = ("Merge obviously-correct PR", "Post review and act on research verdict")


def step(name):
    section = WORKFLOW.read_text(encoding="utf-8").split(f"      - name: {name}\n", 1)[1]
    section = section.split("\n      - name:", 1)[0]
    return textwrap.dedent(section.split("        run: |\n", 1)[1])


class TriageHeadTests(unittest.TestCase):
    def run_step(self, name, *, live=SHA, metadata=SHA, auto_failure=False,
                 protected=False, reviewed=SHA):
        self.assertTrue(BASH and JQ, "bash and jq are required")
        with tempfile.TemporaryDirectory(prefix="triage-head-") as temporary:
            root = Path(temporary)
            (root / "metadata-response.json").write_text(json.dumps({
                "headRefOid": metadata, "author": {"login": "fixture-author"},
            }), encoding="utf-8")
            for file, content in {
                "verdict.json": json.dumps({"summary": "fixture", "reasons": "fixture"}),
                "research_verdict.json": json.dumps({"verdict": "merge"}),
                "review.md": "fixture review\n",
                "changed_files.txt": (
                    ".github/workflows/fixture.yml\n" if protected else "README.md\n"
                ),
                "files-response.txt": "README.md\n",
            }.items():
                (root / file).write_text(content, encoding="utf-8")
            prelude = r'''
jq() { "$JQ_EXECUTABLE" "$@"; }
gh() {
  if [ "$1 $2" = "pr view" ]; then
    case "$*" in
      *--jq*) printf '%s\n' "$LIVE_HEAD" ;;
      *) printf '%s\n' "$*" > "$WORKDIR/view-args"; cat "$WORKDIR/metadata-response.json" ;;
    esac
  elif [ "$1" = api ]; then
    cat "$WORKDIR/files-response.txt"
  elif [ "$1 $2" = "pr diff" ]; then
    printf 'fixture diff\n'
  elif [ "$1 $2" = "pr merge" ]; then
    printf '%s\n' "$*" >> "$WORKDIR/merge-calls"
    local expected="" previous="" arg
    for arg in "$@"; do
      if [ "$previous" = --match-head-commit ]; then expected="$arg"; fi
      previous="$arg"
    done
    if [ -n "$expected" ] && [ "$expected" != "$LIVE_HEAD" ]; then return 1; fi
    case "$*" in *--auto*) if [ "$AUTO_FAILURE" = 1 ]; then return 1; fi ;; esac
    printf '%s\n' "$LIVE_HEAD" > "$WORKDIR/merged-head"
  elif [ "$1 $2" = "pr edit" ] || [ "$1 $2" = "pr comment" ] || [ "$1 $2" = "pr review" ]; then
    return 0
  else
    printf 'unexpected gh invocation: %s\n' "$*" >&2
    return 97
  fi
}
'''
            script = root / "step.sh"
            script.write_text(prelude + step(name), encoding="utf-8", newline="\n")
            env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}
            env.update({
                "PATH": str(Path(BASH).parent), "WORKDIR": root.as_posix(),
                "GITHUB_ENV": (root / "github-env").as_posix(),
                "JQ_EXECUTABLE": Path(JQ).as_posix(),
                "LIVE_HEAD": live, "AUTO_FAILURE": "1" if auto_failure else "0",
                "PR_NUMBER": "1", "REPO": "fixture/repo", "MSYS_NO_PATHCONV": "1",
            })
            if reviewed is not None:
                env["REVIEWED_HEAD"] = reviewed
            syntax = subprocess.run([BASH, "--noprofile", "--norc", "-n", str(script)],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            run = subprocess.run([BASH, "--noprofile", "--norc", "-e", str(script)],
                                 env=env, capture_output=True, text=True, timeout=10)
            files = {file.name: file.read_text(encoding="utf-8") for file in root.iterdir()
                     if file.is_file() and file.name != "step.sh"}
            return run, files

    def test_fetch_captures_the_consistent_head(self):
        run, files = self.run_step("Fetch PR data (as data, not code)")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("headRefOid", files["view-args"])
        self.assertIn(f"REVIEWED_HEAD={SHA}\n", files["github-env"])
        self.assertEqual(files["changed_files.txt"], "README.md\n")

    def test_fetch_discards_a_changed_head(self):
        run, files = self.run_step("Fetch PR data (as data, not code)", live=NEW_SHA)
        self.assertNotEqual(run.returncode, 0)
        self.assertNotIn("github-env", files)

    def test_fetch_rejects_missing_or_malformed_head(self):
        for metadata in (None, "", "invalid", f"{SHA}\nINJECTED=value"):
            with self.subTest(metadata=metadata):
                run, files = self.run_step("Fetch PR data (as data, not code)", metadata=metadata)
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn("github-env", files)

    def test_same_head_can_merge(self):
        for name in MERGE_STEPS:
            with self.subTest(step=name):
                run, files = self.run_step(name)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(files["merged-head"], f"{SHA}\n")
                self.assertIn(f"--match-head-commit {SHA}", files["merge-calls"])

    def test_changed_head_cannot_merge(self):
        for name in MERGE_STEPS:
            with self.subTest(step=name):
                run, files = self.run_step(name, live=NEW_SHA)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertNotIn("merged-head", files)
                calls = files["merge-calls"].splitlines()
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(f"--match-head-commit {SHA}" in call for call in calls))

    def test_direct_fallback_keeps_the_same_head_guard(self):
        for name in MERGE_STEPS:
            with self.subTest(step=name):
                run, files = self.run_step(name, auto_failure=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(files["merged-head"], f"{SHA}\n")
                calls = files["merge-calls"].splitlines()
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(f"--match-head-commit {SHA}" in call for call in calls))

    def test_missing_reviewed_head_cannot_disable_the_guard(self):
        for name in MERGE_STEPS:
            for reviewed in (None, ""):
                with self.subTest(step=name, reviewed=reviewed):
                    run, files = self.run_step(name, reviewed=reviewed)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertNotIn("merge-calls", files)

    def test_research_still_refuses_protected_paths(self):
        run, files = self.run_step(MERGE_STEPS[1], protected=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("merge-calls", files)


if __name__ == "__main__":
    unittest.main(verbosity=2)
