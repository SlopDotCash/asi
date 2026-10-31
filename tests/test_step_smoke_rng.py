"""Step smoke runners own their Threefry roots independently of the caller's RNG default.

Same contract as ``tests/test_step6_smoke_rng.py`` for the remaining runners
that build their root from a bare seed.
"""

from collections.abc import Callable
from functools import cache

import jax
import jax.random as jr
import pytest

from alberta_framework import pipeline
from alberta_framework.steps import step1, step2, step3, step4, step7, step8, step9, step12

SEED = 7

SMOKES: dict[str, Callable[[int], dict[str, object]]] = {
    "step1": lambda seed: step1.run_step1_smoke(steps=8, seed=seed, final_window=4).to_dict(),
    "step2": lambda seed: step2.run_step2_smoke(steps=8, seed=seed, final_window=4).to_dict(),
    "step3": lambda seed: step3.run_step3_smoke(steps=8, seed=seed, final_window=4).to_dict(),
    "step4": lambda seed: step4.run_step4_smoke(steps=4, seed=seed).to_dict(),
    "step7": lambda seed: step7.run_step7_smoke(steps=4, seed=seed).to_dict(),
    "step8": lambda seed: step8.run_step8_smoke(steps=4, seed=seed).to_dict(),
    "step9": lambda seed: step9.run_step9_smoke(steps=4, seed=seed).to_dict(),
    "step12": lambda seed: step12.run_step12_smoke(steps=4, seed=seed).to_dict(),
    "pipeline": lambda seed: pipeline.run_pipeline_smoke(steps=4, seed=seed).to_dict(),
}


@cache
def _threefry_result(name: str) -> dict[str, object]:
    with jax.default_prng_impl("threefry2x32"):
        return SMOKES[name](SEED)


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
@pytest.mark.parametrize("name", sorted(SMOKES))
def test_step_smoke_uses_threefry_independent_of_default(name: str, implementation: str) -> None:
    expected = _threefry_result(name)
    with jax.default_prng_impl(implementation):
        actual = SMOKES[name](SEED)
        assert str(jr.key_impl(jr.key(SEED))) == implementation
    assert actual == expected
