"""RNG-boundary checks for the Alberta Plan Step 1 streams."""

import jax.random as jr
import pytest

from alberta_framework.streams.alberta_plan_step1 import (
    AlbertaPlanStep1Stream,
    XDistShiftStream,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "stream",
    [AlbertaPlanStep1Stream(), XDistShiftStream(feature_dim=20, num_relevant=5)],
    ids=["target-drift", "input-distribution-shift"],
)
@pytest.mark.parametrize(
    "key",
    [jr.PRNGKey(0), jr.key(0, impl="rbg")],
    ids=["legacy-key", "non-threefry-key"],
)
def test_init_rejects_noncanonical_rng_roots(stream: object, key: object) -> None:
    """Public stream roots must not silently change the benchmark RNG contract."""
    with pytest.raises(TypeError, match="scalar typed Threefry JAX key"):
        stream.init(key)  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.parametrize(
    "stream",
    [AlbertaPlanStep1Stream(), XDistShiftStream(feature_dim=20, num_relevant=5)],
    ids=["target-drift", "input-distribution-shift"],
)
def test_init_accepts_explicit_threefry_root(
    stream: AlbertaPlanStep1Stream | XDistShiftStream,
) -> None:
    state = stream.init(jr.key(0, impl="threefry2x32"))

    assert str(jr.key_impl(state.key)) == "threefry2x32"

