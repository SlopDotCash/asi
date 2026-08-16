from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DRIVER = runpy.run_path(_ROOT / ".github" / "scripts" / "ipmnist_prereg.py")
_PROTOCOLS = cast(dict[str, Any], _DRIVER["PROTOCOLS"])
_authorization_line = cast(Any, _DRIVER["authorization_line"])
_registration_amendment_line = cast(Any, _DRIVER["registration_amendment_line"])
_classify_outcome = cast(Any, _DRIVER["classify_outcome"])
_strict_json = cast(Any, _DRIVER["_strict_json"])
_validate_runner_receipt = cast(Any, _DRIVER["_validate_runner_receipt"])
_validate_runtime = cast(Any, _DRIVER["_validate_runtime"])
_validate_summary_reconstruction = cast(Any, _DRIVER["_validate_summary_reconstruction"])
_validate_result_bundle = cast(Any, _DRIVER["validate_result_bundle"])
_verify_launch_authorization = cast(Any, _DRIVER["verify_launch_authorization"])
_workflow_runs = cast(Any, _DRIVER["_workflow_runs"])
_write_json = cast(Any, _DRIVER["_write_json"])
_DRIVER_GLOBALS = cast(dict[str, Any], _verify_launch_authorization.__globals__)
_WORKFLOW_RUN_GLOBALS = cast(dict[str, Any], _workflow_runs.__globals__)


def test_prereg_protocols_pin_exact_arms_and_seeds() -> None:
    issue51 = _PROTOCOLS["issue51"]
    assert issue51.issue == 51
    assert issue51.namespace == "replication_r1"
    assert issue51.control == "sigma0_shiftnorm_d099"
    assert issue51.candidate == "rls_head_resid_l1_preset005"
    assert issue51.seeds == (0, 1, 2)

    issue188 = _PROTOCOLS["issue188"]
    assert issue188.issue == 188
    assert issue188.namespace == "gate_ablation_r3"
    assert issue188.control == "rls_head_resid_l1_preset005"
    assert issue188.candidate == "rls_head_resid_l1_preset005_nogate"
    assert issue188.seeds == tuple(range(3, 13))


def test_issue184_protocol_pins_exact_arms_namespace_and_seeds() -> None:
    issue184 = _PROTOCOLS["issue184"]
    assert issue184.issue == 184
    assert issue184.namespace == "rls_preset_ablation_r1"
    assert issue184.control == "rls_head_resid_l1_preset005"
    assert issue184.candidate == "rls_head_resid_l1_noreset"
    assert issue184.seeds == (0, 1, 2)


def test_authorization_line_binds_every_launch_identity() -> None:
    line = _authorization_line(
        _PROTOCOLS["issue51"],
        source="1" * 40,
        tree="2" * 40,
        uv_lock_sha256="3" * 64,
        workflow_blob_sha1="4" * 40,
        driver_blob_sha1="5" * 40,
        ref_name="ipmnist-prereg-example",
    )
    assert line == (
        "ASI_PREREG_LAUNCH_V1 issue=51 protocol=issue51 "
        f"source={'1' * 40} tree={'2' * 40} uv_lock_sha256={'3' * 64} "
        f"workflow_blob_sha1={'4' * 40} driver_blob_sha1={'5' * 40} "
        "ref=ipmnist-prereg-example "
        "runner=github-hosted-macos-14-arm64-apple-m1 seeds=0,1,2 n=3 "
        "protocol_approval=approved seed_budget=approved compute=authorized-uncompensated"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"source": cast(Any, 1)},
        {"ref_name": cast(Any, 1)},
    ],
)
def test_authorization_line_rejects_nonstring_source_or_ref(
    overrides: dict[str, Any],
) -> None:
    arguments: dict[str, Any] = {
        "source": "1" * 40,
        "tree": "2" * 40,
        "uv_lock_sha256": "3" * 64,
        "workflow_blob_sha1": "4" * 40,
        "driver_blob_sha1": "5" * 40,
        "ref_name": "ipmnist-prereg-example",
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        _authorization_line(_PROTOCOLS["issue51"], **arguments)


def test_issue188_amendment_line_binds_the_complete_registered_change() -> None:
    line = _registration_amendment_line(
        _PROTOCOLS["issue188"],
        source="1" * 40,
        tree="2" * 40,
        uv_lock_sha256="3" * 64,
        workflow_blob_sha1="4" * 40,
        driver_blob_sha1="5" * 40,
        ref_name="ipmnist-prereg-example",
    )
    assert line == (
        "ASI_PREREG_AMENDMENT_V1 issue=188 protocol=issue188 "
        f"source={'1' * 40} tree={'2' * 40} uv_lock_sha256={'3' * 64} "
        f"workflow_blob_sha1={'4' * 40} driver_blob_sha1={'5' * 40} "
        "ref=ipmnist-prereg-example "
        "runner=github-hosted-macos-14-arm64-apple-m1 seeds=3,4,5,6,7,8,9,10,11,12 "
        "n=10 compute=uncompensated"
    )


@pytest.mark.parametrize(
    ("mean_diff", "diffs", "expected"),
    [
        (0.004882, (0.004, 0.005, 0.006), "replicated"),
        (0.005950, (0.004, 0.005, 0.006), "replicated"),
        (0.006, (0.004, 0.005, 0.006), "directionally_replicated"),
        (0.005, (0.004, 0.0, 0.006), "not_replicated"),
    ],
)
def test_issue51_outcomes_are_frozen(
    mean_diff: float, diffs: tuple[float, ...], expected: str
) -> None:
    assert (
        _classify_outcome("issue51", mean_diff=mean_diff, stderr_diff=0.0001, per_seed_diff=diffs)
        == expected
    )


@pytest.mark.parametrize(
    ("mean_diff", "stderr_diff", "expected"),
    [
        (-0.001, 0.0001, "not_load_bearing"),
        (-0.002, 0.0001, "load_bearing"),
        (-0.0015, 0.0001, "inconclusive"),
        (-0.0013, 0.0001, "inconclusive"),
    ],
)
def test_issue188_outcomes_are_frozen(mean_diff: float, stderr_diff: float, expected: str) -> None:
    assert (
        _classify_outcome(
            "issue188",
            mean_diff=mean_diff,
            stderr_diff=stderr_diff,
            per_seed_diff=(mean_diff,) * 10,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mean_diff", "diffs", "expected"),
    [
        (0.002000001, (0.001, 0.002, 0.003), "no_reset_win"),
        (0.002, (0.001, 0.002, 0.003), "inconclusive"),
        (0.003, (0.001, 0.0, 0.003), "inconclusive"),
        (-0.002000001, (-0.001, -0.002, -0.003), "reset_load_bearing"),
        (-0.002, (-0.001, -0.002, -0.003), "inconclusive"),
        (-0.003, (-0.001, 0.0, -0.003), "inconclusive"),
        (0.001, (0.0015, -0.0015, 0.0), "practical_equivalence"),
        (-0.001, (0.0015, -0.0015, 0.0), "practical_equivalence"),
        (0.0005, (0.001500001, -0.001, 0.0), "inconclusive"),
    ],
)
def test_issue184_outcomes_are_frozen(
    mean_diff: float, diffs: tuple[float, ...], expected: str
) -> None:
    assert (
        _classify_outcome(
            "issue184",
            mean_diff=mean_diff,
            stderr_diff=0.0,
            per_seed_diff=diffs,
        )
        == expected
    )


def test_outcome_validation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _classify_outcome(
            "issue188",
            mean_diff=float("nan"),
            stderr_diff=0.0,
            per_seed_diff=(0.0,) * 10,
        )


def _launch_api_payloads(comment: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = {
        "id": 123,
        "event": "workflow_dispatch",
        "head_sha": "1" * 40,
        "display_title": f"ipmnist-issue51-{'1' * 40}",
        "run_attempt": 1,
        "path": ".github/workflows/ipmnist-prereg.yml",
        "created_at": "2026-08-16T10:00:00Z",
        "html_url": "https://github.com/elizaOS/asi/actions/runs/123",
    }
    return current, [comment]


def _verify_with_comment(
    monkeypatch: pytest.MonkeyPatch,
    comment: dict[str, Any],
    *,
    current_overrides: dict[str, Any] | None = None,
    invocation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current, comments = _launch_api_payloads(comment)
    current.update(current_overrides or {})
    monkeypatch.setitem(_DRIVER_GLOBALS, "_github_json", lambda *_args, **_kwargs: current)
    monkeypatch.setitem(_DRIVER_GLOBALS, "_workflow_runs", lambda *_args, **_kwargs: [current])
    monkeypatch.setitem(_DRIVER_GLOBALS, "_github_pages", lambda *_args, **_kwargs: comments)
    arguments: dict[str, Any] = {
        "protocol_key": "issue51",
        "repository": "elizaOS/asi",
        "source": "1" * 40,
        "tree": "2" * 40,
        "uv_lock_sha256": "3" * 64,
        "workflow_blob_sha1": "4" * 40,
        "driver_blob_sha1": "5" * 40,
        "ref_name": "ipmnist-prereg-example",
        "run_id": 123,
        "run_attempt": 1,
        "token": "token",
    }
    arguments.update(invocation_overrides or {})
    return cast(
        dict[str, Any],
        _verify_launch_authorization(**arguments),
    )


def _authorization_comment(**overrides: Any) -> dict[str, Any]:
    body = _authorization_line(
        _PROTOCOLS["issue51"],
        source="1" * 40,
        tree="2" * 40,
        uv_lock_sha256="3" * 64,
        workflow_blob_sha1="4" * 40,
        driver_blob_sha1="5" * 40,
        ref_name="ipmnist-prereg-example",
    )
    comment: dict[str, Any] = {
        "id": 456,
        "body": body,
        "user": {"id": 18_633_264, "login": "lalalune"},
        "author_association": "MEMBER",
        "created_at": "2026-08-16T09:00:00Z",
        "updated_at": "2026-08-16T09:00:00Z",
        "html_url": "https://github.com/elizaOS/asi/issues/51#issuecomment-456",
    }
    comment.update(overrides)
    return comment


def test_launch_authorization_accepts_exact_unedited_project_owner_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _verify_with_comment(monkeypatch, _authorization_comment())
    assert payload["authorization_comment_id"] == 456
    assert payload["authorization_created_at"] == "2026-08-16T09:00:00Z"
    assert payload["authorization_updated_at"] == "2026-08-16T09:00:00Z"


@pytest.mark.parametrize(
    "overrides",
    [
        {"author_association": "OWNER"},
        {"user": {"id": 1, "login": "lalalune"}},
        {"user": {"id": 18_633_264.0, "login": "lalalune"}},
        {"updated_at": "2026-08-16T09:30:00Z"},
    ],
)
def test_launch_authorization_rejects_wrong_identity_or_edited_comment(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]
) -> None:
    with pytest.raises(RuntimeError, match="authorization"):
        _verify_with_comment(monkeypatch, _authorization_comment(**overrides))


def test_launch_authorization_requires_strictly_pre_dispatch_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="before workflow dispatch"):
        _verify_with_comment(
            monkeypatch,
            _authorization_comment(
                created_at="2026-08-16T10:00:00Z",
                updated_at="2026-08-16T10:00:00Z",
            ),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": None},
        {"id": -1},
        {"id": 456.0},
        {"html_url": None},
        {"html_url": "https://github.com/elizaOS/asi/issues/51#issuecomment-999"},
        {"html_url": "https://attacker.invalid/issues/51#issuecomment-456"},
    ],
)
def test_launch_authorization_rejects_noncanonical_comment_record(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]
) -> None:
    with pytest.raises(RuntimeError, match="authorization comment"):
        _verify_with_comment(monkeypatch, _authorization_comment(**overrides))


@pytest.mark.parametrize(
    ("current_overrides", "invocation_overrides"),
    [
        ({"id": 123.0}, {}),
        ({"run_attempt": 1.0}, {}),
        ({"html_url": None}, {}),
        ({"html_url": "https://attacker.invalid/actions/runs/123"}, {}),
        ({}, {"run_id": 123.0}),
        ({}, {"run_attempt": True}),
        ({}, {"repository": "attacker/asi"}),
    ],
)
def test_launch_authorization_rejects_noncanonical_run_record_or_invocation(
    monkeypatch: pytest.MonkeyPatch,
    current_overrides: dict[str, Any],
    invocation_overrides: dict[str, Any],
) -> None:
    with pytest.raises(RuntimeError, match="repository|run"):
        _verify_with_comment(
            monkeypatch,
            _authorization_comment(),
            current_overrides=current_overrides,
            invocation_overrides=invocation_overrides,
        )


@pytest.mark.parametrize(
    "timestamp",
    ["not-a-timestamp", "2026-08-16T09:00:00", "2026-08-16T10:00:00+01:00"],
)
def test_launch_authorization_requires_valid_utc_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    timestamp: str,
) -> None:
    with pytest.raises(RuntimeError, match="timestamp"):
        _verify_with_comment(
            monkeypatch,
            _authorization_comment(created_at=timestamp, updated_at=timestamp),
        )


def _issue188_authorization_comment(**overrides: Any) -> dict[str, Any]:
    body = _authorization_line(
        _PROTOCOLS["issue188"],
        source="1" * 40,
        tree="2" * 40,
        uv_lock_sha256="3" * 64,
        workflow_blob_sha1="4" * 40,
        driver_blob_sha1="5" * 40,
        ref_name="ipmnist-prereg-example",
    )
    comment: dict[str, Any] = {
        "id": 456,
        "body": body,
        "user": {"id": 18_633_264, "login": "lalalune"},
        "author_association": "MEMBER",
        "created_at": "2026-08-16T09:30:00Z",
        "updated_at": "2026-08-16T09:30:00Z",
        "html_url": "https://github.com/elizaOS/asi/issues/188#issuecomment-456",
    }
    comment.update(overrides)
    return comment


def _issue188_amendment_comment(**overrides: Any) -> dict[str, Any]:
    body = _registration_amendment_line(
        _PROTOCOLS["issue188"],
        source="1" * 40,
        tree="2" * 40,
        uv_lock_sha256="3" * 64,
        workflow_blob_sha1="4" * 40,
        driver_blob_sha1="5" * 40,
        ref_name="ipmnist-prereg-example",
    )
    comment: dict[str, Any] = {
        "id": 455,
        "body": body,
        "user": {"id": 18_633_264, "login": "lalalune"},
        "author_association": "MEMBER",
        "created_at": "2026-08-16T09:00:00Z",
        "updated_at": "2026-08-16T09:00:00Z",
        "html_url": "https://github.com/elizaOS/asi/issues/188#issuecomment-455",
    }
    comment.update(overrides)
    return comment


def _verify_issue188(
    monkeypatch: pytest.MonkeyPatch,
    comments: list[dict[str, Any]],
) -> dict[str, Any]:
    current = {
        "id": 123,
        "event": "workflow_dispatch",
        "head_sha": "1" * 40,
        "display_title": f"ipmnist-issue188-{'1' * 40}",
        "run_attempt": 1,
        "path": ".github/workflows/ipmnist-prereg.yml",
        "created_at": "2026-08-16T10:00:00Z",
        "html_url": "https://github.com/elizaOS/asi/actions/runs/123",
    }
    monkeypatch.setitem(_DRIVER_GLOBALS, "_github_json", lambda *_args, **_kwargs: current)
    monkeypatch.setitem(_DRIVER_GLOBALS, "_workflow_runs", lambda *_args, **_kwargs: [current])
    monkeypatch.setitem(_DRIVER_GLOBALS, "_github_pages", lambda *_args, **_kwargs: comments)
    return cast(
        dict[str, Any],
        _verify_launch_authorization(
            protocol_key="issue188",
            repository="elizaOS/asi",
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="3" * 64,
            workflow_blob_sha1="4" * 40,
            driver_blob_sha1="5" * 40,
            ref_name="ipmnist-prereg-example",
            run_id=123,
            run_attempt=1,
            token="token",
        ),
    )


def test_issue188_requires_one_exact_amendment_before_final_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment = _issue188_amendment_comment()
    authorization = _issue188_authorization_comment()
    payload = _verify_issue188(monkeypatch, [amendment, authorization])
    assert payload["registration_amendment_comment_id"] == 455
    assert payload["registration_amendment_created_at"] == "2026-08-16T09:00:00Z"

    with pytest.raises(RuntimeError, match="amendment"):
        _verify_issue188(monkeypatch, [authorization])


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": None},
        {"id": -1},
        {"id": 455.0},
        {"html_url": None},
        {"html_url": "https://github.com/elizaOS/asi/issues/188#issuecomment-999"},
        {"html_url": "https://attacker.invalid/issues/188#issuecomment-455"},
    ],
)
def test_issue188_rejects_noncanonical_amendment_record(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]
) -> None:
    with pytest.raises(RuntimeError, match="amendment comment"):
        _verify_issue188(
            monkeypatch,
            [_issue188_amendment_comment(**overrides), _issue188_authorization_comment()],
        )


def test_issue188_requires_distinct_amendment_and_authorization_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = _issue188_authorization_comment()
    amendment = _issue188_amendment_comment(
        id=authorization["id"],
        html_url=authorization["html_url"],
    )
    with pytest.raises(RuntimeError, match="distinct"):
        _verify_issue188(monkeypatch, [amendment, authorization])


@pytest.mark.parametrize("duplicate", ["amendment", "authorization"])
def test_issue188_rejects_duplicate_exact_owner_records(
    monkeypatch: pytest.MonkeyPatch, duplicate: str
) -> None:
    amendment = _issue188_amendment_comment()
    authorization = _issue188_authorization_comment()
    comments = [amendment, authorization]
    comments.append(dict(amendment if duplicate == "amendment" else authorization))
    with pytest.raises(RuntimeError, match="exactly one"):
        _verify_issue188(monkeypatch, comments)


@pytest.mark.parametrize(
    "overrides",
    [
        {"body": "ASI_PREREG_AMENDMENT_V1 source=wrong"},
        {"author_association": "OWNER"},
        {"user": {"id": 1, "login": "lalalune"}},
        {"updated_at": "2026-08-16T09:15:00Z"},
    ],
)
def test_issue188_rejects_inexact_edited_or_wrong_author_amendment(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(RuntimeError, match="amendment"):
        _verify_issue188(
            monkeypatch,
            [_issue188_amendment_comment(**overrides), _issue188_authorization_comment()],
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-a-timestamp",
        "2026-08-16T09:00:00",
        "2026-08-16T10:00:00+01:00",
        "2026-08-16T09:30:00Z",
        "2026-08-16T09:45:00Z",
    ],
)
def test_issue188_amendment_must_be_valid_utc_and_precede_authorization(
    monkeypatch: pytest.MonkeyPatch,
    timestamp: str,
) -> None:
    with pytest.raises(RuntimeError, match="amendment|timestamp"):
        _verify_issue188(
            monkeypatch,
            [
                _issue188_amendment_comment(created_at=timestamp, updated_at=timestamp),
                _issue188_authorization_comment(),
            ],
        )


def _searched_workflow_run(run_id: int) -> dict[str, Any]:
    return {
        "id": run_id,
        "event": "workflow_dispatch",
        "head_sha": "1" * 40,
        "display_title": f"unrelated-run-{run_id}",
        "path": ".github/workflows/ipmnist-prereg.yml",
    }


def test_workflow_run_search_binds_source_and_paginates_exact_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    responses = [
        {
            "total_count": 101,
            "workflow_runs": [_searched_workflow_run(value) for value in range(1, 101)],
        },
        {"total_count": 101, "workflow_runs": [_searched_workflow_run(101)]},
    ]

    def fake_github_json(path: str, *, token: str) -> dict[str, Any]:
        assert token == "token"
        calls.append(path)
        return responses.pop(0)

    monkeypatch.setitem(_WORKFLOW_RUN_GLOBALS, "_github_json", fake_github_json)
    runs = _workflow_runs("elizaOS/asi", source="1" * 40, token="token")

    assert [run["id"] for run in runs] == list(range(1, 102))
    assert len(calls) == 2
    assert all("event=workflow_dispatch" in path for path in calls)
    assert all(f"head_sha={'1' * 40}" in path for path in calls)


def test_workflow_run_search_rejects_unsearchable_or_changing_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        _WORKFLOW_RUN_GLOBALS,
        "_github_json",
        lambda *_args, **_kwargs: {"total_count": 1001, "workflow_runs": []},
    )
    with pytest.raises(RuntimeError, match="1,000"):
        _workflow_runs("elizaOS/asi", source="1" * 40, token="token")

    responses = [
        {
            "total_count": 101,
            "workflow_runs": [_searched_workflow_run(value) for value in range(1, 101)],
        },
        {"total_count": 100, "workflow_runs": []},
    ]
    monkeypatch.setitem(
        _WORKFLOW_RUN_GLOBALS,
        "_github_json",
        lambda *_args, **_kwargs: responses.pop(0),
    )
    with pytest.raises(RuntimeError, match="changed during pagination"):
        _workflow_runs("elizaOS/asi", source="1" * 40, token="token")


def test_workflow_run_search_never_skips_a_malformed_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    responses = [
        {
            "total_count": 101,
            "workflow_runs": [_searched_workflow_run(value) for value in range(1, 101)],
        },
        {"total_count": 101, "workflow_runs": ["not-a-workflow-run"]},
        {"total_count": 101, "workflow_runs": [_searched_workflow_run(101)]},
    ]

    def fake_github_json(path: str, *, token: str) -> dict[str, Any]:
        assert token == "token"
        calls.append(path)
        return responses.pop(0)

    monkeypatch.setitem(_WORKFLOW_RUN_GLOBALS, "_github_json", fake_github_json)
    with pytest.raises(RuntimeError, match="malformed result page"):
        _workflow_runs("elizaOS/asi", source="1" * 40, token="token")

    assert len(calls) == 2
    assert calls[-1].endswith("page=2")


def test_workflow_run_search_rejects_duplicate_ids_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {
            "total_count": 101,
            "workflow_runs": [_searched_workflow_run(value) for value in range(1, 101)],
        },
        {"total_count": 101, "workflow_runs": [_searched_workflow_run(100)]},
    ]
    monkeypatch.setitem(
        _WORKFLOW_RUN_GLOBALS,
        "_github_json",
        lambda *_args, **_kwargs: responses.pop(0),
    )
    with pytest.raises(RuntimeError, match="repeated a workflow run ID"):
        _workflow_runs("elizaOS/asi", source="1" * 40, token="token")


def test_workflow_installs_exact_uv_managed_python() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "ipmnist-prereg.yml").read_text(
        encoding="utf-8"
    )
    assert "actions/setup-python@" not in workflow
    assert "uv python install --managed-python 3.12.12" in workflow
    assert 'uv python find --managed-python --no-project 3.12.12' in workflow
    assert "CPython 3.12.12 arm64" in workflow
    assert '--python "$PYTHON_PATH"' in workflow
    assert '"jax_disable_jit": False' in workflow
    assert '"jax_random_seed_offset": 0' in workflow
    assert '"jax_default_prng_impl": "threefry2x32"' in workflow
    assert '"chex": "0.1.92"' in workflow
    assert 'PYTHONOPTIMIZE: "0"' in workflow
    assert 'RUNNER_ENVIRONMENT' in workflow
    assert 'sys.flags.optimize == 0' in workflow
    assert 'macos_version.startswith("14.")' in workflow
    assert 'for parent in outputs outputs/ipmnist_screening' in workflow
    assert '[[ -e "$target" || -L "$target" ]]' in workflow


def test_summary_reconstruction_rejects_resigned_derived_metrics_and_manifest() -> None:
    expected_manifest = [
        {
            "path": "outputs/ipmnist_screening/example/shards/control_seed0.json",
            "size_bytes": 10,
            "sha256": "a" * 64,
            "config_name": "control",
            "seed": 0,
        }
    ]
    summary = {
        "schema": "summary.v2",
        "created_unix": 1.0,
        "results": [{"config_name": "candidate", "mean_diff": 0.25}],
        "shard_manifest": expected_manifest,
    }
    recomputed = {
        **summary,
        "created_unix": 2.0,
        "shard_manifest": [{**expected_manifest[0], "path": "/absolute/input.json"}],
    }
    _validate_summary_reconstruction(
        summary=summary,
        recomputed=recomputed,
        expected_manifest=expected_manifest,
    )

    forged = {**summary, "results": [{"config_name": "candidate", "mean_diff": 0.5}]}
    with pytest.raises(ValueError, match="reconstruction"):
        _validate_summary_reconstruction(
            summary=forged,
            recomputed=recomputed,
            expected_manifest=expected_manifest,
        )

    forged_manifest = [{**expected_manifest[0], "sha256": "b" * 64}]
    with pytest.raises(ValueError, match="manifest"):
        _validate_summary_reconstruction(
            summary={**summary, "shard_manifest": forged_manifest},
            recomputed=recomputed,
            expected_manifest=expected_manifest,
        )


def test_strict_json_rejects_exponent_overflow(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"value": 1e999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        _strict_json(path)


def test_metadata_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    _write_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        _write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def _source_provenance() -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.source_provenance.v1",
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "git_object_format": "sha1",
        "relevant_source_scope": "tracked:alberta_framework/**,pyproject.toml,uv.lock",
        "relevant_source_file_count": 3,
        "relevant_source_sha256": "3" * 64,
        "uv_lock_sha256": "4" * 64,
        "worktree_clean": True,
    }


def _dataset_provenance() -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
        "source": {
            "provider": "openml",
            "name": "mnist_784",
            "version": 1,
            "row_start": 0,
            "row_stop_exclusive": 60_000,
        },
        "materialization": "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1",
        "x": {"dtype": "<f4", "shape": [60_000, 784], "sha256": "5" * 64},
        "y": {"dtype": "<i4", "shape": [60_000], "sha256": "6" * 64},
    }


def _runtime_environment() -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.runtime.v1",
        "python": {"implementation": "CPython", "version": "3.12.12"},
        "platform": {"system": "Darwin", "release": "23.6.0", "machine": "arm64"},
        "packages": {
            "chex": "0.1.92",
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
            "numpy": "2.5.1",
            "scikit-learn": "1.9.0",
        },
        "jax": {
            "backend": "cpu",
            "devices": [
                {"id": 0, "platform": "cpu", "device_kind": "Apple M1", "process_index": 0}
            ],
            "config": {
                "jax_enable_x64": False,
                "jax_default_matmul_precision": None,
                "jax_disable_jit": False,
                "jax_numpy_dtype_promotion": "standard",
                "jax_numpy_rank_promotion": "allow",
                "jax_random_seed_offset": 0,
                "jax_threefry_partitionable": True,
                "jax_default_prng_impl": "threefry2x32",
            },
        },
        "process_environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_DEFAULT_MATMUL_PRECISION": None,
            "JAX_ENABLE_X64": "false",
            "JAX_PLATFORM_NAME": "cpu",
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jax_enable_x64", True),
        ("jax_default_matmul_precision", "highest"),
        ("jax_disable_jit", True),
        ("jax_numpy_dtype_promotion", "strict"),
        ("jax_numpy_rank_promotion", "raise"),
        ("jax_random_seed_offset", 1),
        ("jax_threefry_partitionable", False),
        ("jax_default_prng_impl", "rbg"),
    ],
)
def test_runtime_rejects_any_jax_semantic_config_drift(field: str, value: object) -> None:
    environment = _runtime_environment()
    jax_binding = cast(dict[str, Any], environment["jax"])
    config = cast(dict[str, Any], jax_binding["config"])
    config[field] = value
    with pytest.raises(ValueError, match="JAX config"):
        _validate_runtime(environment)


def test_runtime_rejects_chex_version_drift() -> None:
    environment = _runtime_environment()
    packages = cast(dict[str, Any], environment["packages"])
    packages["chex"] = "9.9.9"
    with pytest.raises(ValueError, match="locked package"):
        _validate_runtime(environment)


@pytest.mark.parametrize("release", ["24.0.0", "22.6.0", "not-a-release", None])
def test_runtime_rejects_non_macos14_release(release: object) -> None:
    environment = _runtime_environment()
    platform_binding = cast(dict[str, Any], environment["platform"])
    platform_binding["release"] = release
    with pytest.raises(ValueError, match="macOS 14"):
        _validate_runtime(environment)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jax_enable_x64", 0),
        ("jax_disable_jit", 0.0),
        ("jax_random_seed_offset", False),
        ("jax_threefry_partitionable", 1),
    ],
)
def test_runtime_rejects_equal_but_wrong_type_jax_config(
    field: str, value: object
) -> None:
    environment = _runtime_environment()
    jax_binding = cast(dict[str, Any], environment["jax"])
    config = cast(dict[str, Any], jax_binding["config"])
    config[field] = value
    with pytest.raises(ValueError, match="JAX config"):
        _validate_runtime(environment)


def _write_issue51_bundle(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from alberta_framework.benchmarks.ipmnist_screening import (
        ScreeningRunResult,
        merge_shards,
        screening_spec,
        shard_payload,
    )
    from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

    namespace = root / "outputs" / "ipmnist_screening" / "replication_r1"
    shards_dir = namespace / "shards"
    shards_dir.mkdir(parents=True)
    config = IPMNISTConfig(
        n_tasks=60,
        task_length=5_000,
        input_dim=784,
        hidden1=300,
        hidden2=150,
        n_classes=10,
    )
    relative_paths: list[Path] = []
    for seed in (0, 1, 2):
        for arm, accuracy in (
            ("sigma0_shiftnorm_d099", 0.5),
            ("rls_head_resid_l1_preset005", 0.505),
        ):
            spec = screening_spec(arm)
            result = ScreeningRunResult(
                config_name=arm,
                base_learner=spec.base_learner,
                hyperparameters=dict(spec.hyperparameters),
                seed=seed,
                config=config,
                per_task_accuracy=np.full(60, accuracy, dtype=np.float64),
                per_task_loss=np.full(60, 0.5, dtype=np.float64),
                per_task_plasticity=np.full(60, 0.5, dtype=np.float64),
                wall_clock_seconds=1.0,
            )
            path = shards_dir / f"{arm}_seed{seed}.json"
            path.write_text(
                json.dumps(
                    shard_payload(
                        result,
                        source_provenance=_source_provenance(),
                        dataset_provenance=_dataset_provenance(),
                        environment=_runtime_environment(),
                    ),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            relative_paths.append(path.relative_to(root))
    monkeypatch.chdir(root)
    summary = merge_shards(
        relative_paths,
        control_name="sigma0_shiftnorm_d099",
        slope_window=15,
    )
    summary_path = namespace / "summary.json"
    summary_path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    return summary_path


def _write_runner_receipt(root: Path) -> Path:
    environment = _runtime_environment()
    jax_binding = cast(dict[str, Any], environment["jax"])
    receipt = {
        "schema": "asi.ipmnist_prereg.runner.v2",
        "runner_label": "macos-14",
        "runner_environment": "github-hosted",
        "runner_os": "macOS",
        "runner_arch": "ARM64",
        "cpu_brand": "Apple M1 (Virtual)",
        "platform": environment["platform"],
        "macos_version": "14.7.6",
        "machine": "arm64",
        "python": "3.12.12",
        "python_optimization_level": 0,
        "python_optimize_environment": "0",
        "packages": {
            name: cast(dict[str, Any], environment["packages"])[name]
            for name in ("chex", "jax", "jaxlib", "numpy", "scikit-learn")
        },
        "jax_backend": "cpu",
        "jax_devices": jax_binding["devices"],
        "jax_config": jax_binding["config"],
    }
    path = root / "runner.json"
    path.write_text(json.dumps(receipt, allow_nan=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runner_environment", "self-hosted"),
        ("runner_os", "Linux"),
        ("runner_arch", "X64"),
        ("macos_version", "15.0"),
        ("python_optimization_level", 1),
        ("python_optimization_level", False),
        ("python_optimize_environment", "1"),
    ],
)
def test_runner_receipt_rejects_host_or_optimized_runtime_forgery(
    field: str, value: object
) -> None:
    environment = _runtime_environment()
    receipt_path_payload = {
        "schema": "asi.ipmnist_prereg.runner.v2",
        "runner_label": "macos-14",
        "runner_environment": "github-hosted",
        "runner_os": "macOS",
        "runner_arch": "ARM64",
        "cpu_brand": "Apple M1 (Virtual)",
        "platform": environment["platform"],
        "macos_version": "14.7.6",
        "machine": "arm64",
        "python": "3.12.12",
        "python_optimization_level": 0,
        "python_optimize_environment": "0",
        "packages": environment["packages"],
        "jax_backend": "cpu",
        "jax_devices": cast(dict[str, Any], environment["jax"])["devices"],
        "jax_config": cast(dict[str, Any], environment["jax"])["config"],
    }
    receipt_path_payload[field] = value
    with pytest.raises(ValueError, match="runner receipt"):
        _validate_runner_receipt(receipt_path_payload, environment=environment)


def test_result_bundle_recomputes_summary_from_exact_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    result = _validate_result_bundle(
        protocol_key="issue51",
        root=tmp_path,
        runner_receipt=runner_receipt,
        source="1" * 40,
        tree="2" * 40,
        uv_lock_sha256="4" * 64,
    )
    assert result["outcome"] == "replicated"
    assert result["mean_diff"] == pytest.approx(0.005)


def test_result_bundle_rejects_consistently_resigned_jax_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alberta_framework.benchmarks.ipmnist_screening import merge_shards

    summary_path = _write_issue51_bundle(tmp_path, monkeypatch)
    shards_dir = summary_path.parent / "shards"
    for path in shards_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["environment"]["jax"]["config"]["jax_disable_jit"] = True
        path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    paths = sorted(path.relative_to(tmp_path) for path in shards_dir.glob("*.json"))
    summary_path.write_text(
        json.dumps(
            merge_shards(paths, control_name="sigma0_shiftnorm_d099", slope_window=15),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    runner_receipt = _write_runner_receipt(tmp_path)

    with pytest.raises(ValueError, match="JAX config"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


def test_result_bundle_rejects_runner_jax_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    payload = json.loads(runner_receipt.read_text(encoding="utf-8"))
    payload["jax_config"]["jax_random_seed_offset"] = 1
    runner_receipt.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="JAX config"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jax_enable_x64", 0),
        ("jax_disable_jit", 0.0),
        ("jax_random_seed_offset", False),
        ("jax_threefry_partitionable", 1),
    ],
)
def test_result_bundle_rejects_equal_but_wrong_type_runner_jax_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    payload = json.loads(runner_receipt.read_text(encoding="utf-8"))
    payload["jax_config"][field] = value
    runner_receipt.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="JAX config"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


def test_result_bundle_rejects_equal_but_wrong_type_runner_device_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    payload = json.loads(runner_receipt.read_text(encoding="utf-8"))
    payload["jax_devices"][0]["id"] = False
    runner_receipt.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="JAX devices"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


def test_result_bundle_rejects_runner_platform_not_bound_to_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    payload = json.loads(runner_receipt.read_text(encoding="utf-8"))
    payload["platform"]["release"] = "forged"
    runner_receipt.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="platform"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


def test_result_bundle_rejects_resigned_summary_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path = _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate = next(
        entry
        for entry in summary["results"]
        if entry["config_name"] == "rls_head_resid_l1_preset005"
    )
    candidate["paired_vs_control"]["mean_diff"] = 0.5
    summary_path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="reconstruction"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "outputs/ipmnist_screening/replication_r1/shards/forged.json"),
        ("size_bytes", 1),
        ("sha256", "f" * 64),
    ],
)
def test_result_bundle_rejects_summary_manifest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    summary_path = _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["shard_manifest"][0][field] = value
    summary_path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


def test_result_bundle_rejects_shard_filename_payload_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alberta_framework.benchmarks.ipmnist_screening import merge_shards

    summary_path = _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    shards_dir = summary_path.parent / "shards"
    control = shards_dir / "sigma0_shiftnorm_d099_seed0.json"
    candidate = shards_dir / "rls_head_resid_l1_preset005_seed1.json"
    control_raw = control.read_bytes()
    candidate_raw = candidate.read_bytes()
    control.write_bytes(candidate_raw)
    candidate.write_bytes(control_raw)
    paths = sorted(path.relative_to(tmp_path) for path in shards_dir.glob("*.json"))
    summary_path.write_text(
        json.dumps(
            merge_shards(paths, control_name="sigma0_shiftnorm_d099", slope_window=15),
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="filename/payload"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


def test_result_bundle_rejects_symlinked_shard_even_if_summary_is_resigned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alberta_framework.benchmarks.ipmnist_screening import merge_shards

    summary_path = _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    shards_dir = summary_path.parent / "shards"
    victim = shards_dir / "sigma0_shiftnorm_d099_seed0.json"
    escaped = tmp_path / "escaped-but-inside-root.json"
    escaped.write_bytes(victim.read_bytes())
    victim.unlink()
    victim.symlink_to(escaped)

    paths = sorted(path.relative_to(tmp_path) for path in shards_dir.glob("*.json"))
    summary = merge_shards(paths, control_name="sigma0_shiftnorm_d099", slope_window=15)
    for entry in summary["shard_manifest"]:
        raw_path = Path(entry["path"])
        entry["path"] = raw_path.resolve(strict=True).relative_to(tmp_path).as_posix()
    summary_path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )


def test_result_bundle_rejects_symlinked_protocol_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path = _write_issue51_bundle(tmp_path, monkeypatch)
    runner_receipt = _write_runner_receipt(tmp_path)
    namespace = summary_path.parent
    moved = tmp_path / "moved-namespace"
    namespace.rename(moved)
    namespace.symlink_to(moved, target_is_directory=True)

    with pytest.raises(ValueError, match="namespace.*symlink"):
        _validate_result_bundle(
            protocol_key="issue51",
            root=tmp_path,
            runner_receipt=runner_receipt,
            source="1" * 40,
            tree="2" * 40,
            uv_lock_sha256="4" * 64,
        )
