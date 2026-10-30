"""Descriptor identities must be checked before JAX's optional int64 narrowing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.feature_bank_router import FeatureBankRouter, FeatureBankRouterConfig


@pytest.mark.unit
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("offset", [0, 2**32])
@pytest.mark.parametrize("boundary", ["init", "validate_descriptors", "route"])
def test_descriptor_source_dtype_is_checked(boundary: str, offset: int, x64: bool) -> None:
    with jax.experimental.enable_x64(x64):
        router = FeatureBankRouter(FeatureBankRouterConfig(base_dim=4, active_slots=1))
        descriptors = np.asarray([[offset, offset + 1]], dtype=np.int64)
        state = router.init(jnp.asarray([[0, 1]], dtype=jnp.int32))
        weights = jnp.arange(5, dtype=jnp.float32)
        with pytest.raises(TypeError, match="dtype int32"):
            if boundary == "route":
                router.route(state, weights, descriptors)
            else:
                getattr(router, boundary)(descriptors)
        result = router.route(state, weights, np.asarray([[0, 1]], dtype=np.int32))
        assert bool(result.diagnostics.route_applied)
        np.testing.assert_array_equal(result.consumers, weights)
        np.testing.assert_array_equal(result.state.descriptors, state.descriptors)
