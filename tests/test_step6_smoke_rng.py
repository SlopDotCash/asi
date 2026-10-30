"""Step 6 owns its smoke keys independently of the caller's RNG default."""

import jax
import jax.random as jr
import pytest

from alberta_framework.steps.step6 import run_step6_smoke


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg", "unsafe_rbg"])
@pytest.mark.parametrize("seed", [7, 2**32 - 1])
def test_step6_smoke_uses_threefry_independent_of_default(
    implementation: str, seed: int
) -> None:
    with jax.default_prng_impl("threefry2x32"):
        expected = run_step6_smoke(steps=2, feature_dim=2, seed=seed).to_dict()
    with jax.default_prng_impl(implementation):
        actual = run_step6_smoke(steps=2, feature_dim=2, seed=seed).to_dict()
        assert str(jr.key_impl(jr.key(seed))) == implementation
    assert actual == expected
    assert actual["finite"] is True
