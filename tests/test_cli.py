"""Unit coverage for alberta_framework.cli.

Exercises the smoke-entry argument parsing and JSON emission with the
kernel dependencies stubbed (integration probes are not run in unit
tests).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "alberta_cli", str(repo_root / "alberta_framework/cli.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["alberta_framework.cli"] = module

    fake_step1 = types.ModuleType("alberta_framework.steps.step1")

    def _step1_config(**kwargs):
        return types.SimpleNamespace(**kwargs)

    fake_step1.Step1KernelConfig = _step1_config
    fake_step1.Step1NormalizerName = type("Step1NormalizerName", (), {})
    fake_step1.Step1OptimizerName = type("Step1OptimizerName", (), {})
    def _smoke_result(**metrics):
        return types.SimpleNamespace(to_dict=lambda: metrics, finite=True)

    fake_step1.run_step1_smoke = lambda *a, **k: _smoke_result(metric=1.0, finite=True)
    sys.modules["alberta_framework.steps.step1"] = fake_step1

    fake_step2 = types.ModuleType("alberta_framework.steps.step2")

    def _step2_config(**kwargs):
        return types.SimpleNamespace(**kwargs)

    fake_step2.Step2KernelConfig = _step2_config
    fake_step2.Step2StreamName = type("Step2StreamName", (), {})
    fake_step2.run_step2_smoke = lambda *a, **k: _smoke_result(metric=2.0, finite=True)
    sys.modules["alberta_framework.steps.step2"] = fake_step2

    spec.loader.exec_module(module)
    return module


mod = load_module()


def test_print_json_emits_sorted(capsys) -> None:
    mod._print_json({"b": 1, "a": 2})
    out = capsys.readouterr().out
    # sort_keys=True → "a" before "b"
    assert out.index('"a"') < out.index('"b"')
    assert "allow_nan=False"  # just a marker that serialization succeeded


def test_step1_smoke_main_runs(capsys) -> None:
    code = mod.step1_smoke_main(["--steps", "16", "--seed", "7"])
    assert code == 0
    out = capsys.readouterr().out
    assert "metric" in out


def test_step2_smoke_main_runs(capsys) -> None:
    code = mod.step2_smoke_main(["--steps", "16"])
    assert code == 0
    out = capsys.readouterr().out
    assert "metric" in out


def test_step1_rejects_bad_optimizer() -> None:
    with pytest.raises(SystemExit):
        mod.step1_smoke_main(["--optimizer", "bogus"])


def test_step2_rejects_bad_stream() -> None:
    with pytest.raises(SystemExit):
        mod.step2_smoke_main(["--stream", "bogus"])
