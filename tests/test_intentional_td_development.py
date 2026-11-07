"""Cheap plan/CLI contracts; campaign workloads are replaced with stubs."""

import json
import subprocess
import sys

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.benchmarks import intentional_td_development as campaign


@pytest.mark.parametrize("epsilon, expected", [(0.0, 1), (1.0, 0)])
def test_plan_and_policy_share_epsilon(monkeypatch, epsilon, expected):
    monkeypatch.setattr(campaign, "EPSILON", epsilon)
    monkeypatch.setattr(campaign.jr, "uniform", lambda key: jnp.array(0.5))
    monkeypatch.setattr(campaign.jr, "randint", lambda *args: jnp.array(0))
    assert campaign.development_plan()["epsilon"] == epsilon
    assert int(campaign._action(jnp.array([[0.0], [1.0]]), jnp.ones(1), jr.key(0))) == expected


def test_plan_and_environment_share_riverswim_size(monkeypatch):
    monkeypatch.setattr(campaign, "RIVERSWIM_STATES", 5)
    assert campaign.development_plan()["environments"]["riverswim_states"] == 5

    class ConstructorObservedError(Exception):
        pass

    def constructor(config):
        assert config.n_states == 5
        raise ConstructorObservedError

    monkeypatch.setattr(campaign, "RiverSwimMDP", constructor)
    # Stop at construction: no benchmark transitions execute in pytest.
    with pytest.raises(ConstructorObservedError):
        campaign.run_arm("riverswim", "intentional", 0)


@pytest.mark.parametrize("environment", ["bogus", "Switching", "", 0, None, ["switching"]])
def test_run_arm_rejects_unknown_environments_before_any_workload(monkeypatch, environment):
    def forbidden(config):
        pytest.fail(f"environment {environment!r} was run as an existing MDP")

    monkeypatch.setattr(campaign, "RiverSwimMDP", forbidden)
    monkeypatch.setattr(campaign, "SwitchingTwoStateMDP", forbidden)
    with pytest.raises(ValueError, match="unknown development environment"):
        campaign.run_arm(environment, "intentional", 0)


@pytest.fixture
def output_cli(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["intentional-td", "--output", str(output)])
    monkeypatch.setattr(campaign, "ENVIRONMENTS", ("switching",))
    monkeypatch.setattr(campaign, "SEEDS", (0,))
    monkeypatch.setattr(campaign, "ARMS", {
        name: campaign.ARMS[name] for name in ("intentional", "fixed_step")
    })
    monkeypatch.setattr(campaign, "_source_identity", lambda: {"test_stub": True})
    return output


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
def test_cli_retains_completed_rows_and_failure_context(monkeypatch, output_cli, exception_type):
    def run(environment, arm, seed):
        if arm == "fixed_step":
            raise exception_type("injected execution failure")
        return {"environment": environment, "arm": arm, "seed": seed}

    monkeypatch.setattr(campaign, "run_arm", run)
    with pytest.raises(SystemExit) as exit_info:
        campaign.main()
    assert exit_info.value.code == 1
    record = json.loads(output_cli.read_text())
    assert record["runs"] == [{"environment": "switching", "arm": "intentional", "seed": 0}]
    assert record["failures"] == [{
        "environment": "switching", "arm": "fixed_step", "seed": 0,
        "error": "injected execution failure", "error_type": exception_type.__name__,
    }]
    assert record["source"] == {"test_stub": True}


def test_source_identity_exception_is_retained_before_any_workload(monkeypatch, output_cli):
    def identity():
        raise subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"])

    def forbidden_run(*args):
        pytest.fail("source failure must prevent workload execution")

    monkeypatch.setattr(campaign, "_source_identity", identity)
    monkeypatch.setattr(campaign, "run_arm", forbidden_run)
    with pytest.raises(SystemExit) as exit_info:
        campaign.main()
    assert exit_info.value.code == 1
    record = json.loads(output_cli.read_text())
    assert record["source"] is None
    assert record["runs"] == []
    assert record["failures"][0]["phase"] == "source_identity"
    assert record["failures"][0]["error_type"] == "CalledProcessError"
    assert record["plan"] == campaign.development_plan()


def test_existing_output_is_refused_before_source_or_workload(monkeypatch, output_cli):
    output_cli.write_bytes(b"retained artifact\n")

    def forbidden(*args):
        pytest.fail("occupied output must be refused before source/workload calls")

    monkeypatch.setattr(campaign, "_source_identity", forbidden)
    monkeypatch.setattr(campaign, "run_arm", forbidden)
    with pytest.raises(FileExistsError):
        campaign.main()
    assert output_cli.read_bytes() == b"retained artifact\n"
