"""Distribution scales must survive the float32 sampling kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.streams.alberta_plan_step1 import XDistShiftStream


@pytest.mark.unit
@pytest.mark.parametrize("bounds", [(1e-40, 1e-39), (1e-40, 1.0),
                                      (float(np.nextafter(np.float32(0), np.float32(1))), 1.0)])
def test_subnormal_scale_min_rejected(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="scale_min.*normal float32"):
        XDistShiftStream(4, 2, scale_min=bounds[0], scale_max=bounds[1])


@pytest.mark.unit
@pytest.mark.parametrize("minimum", [float(np.finfo(np.float32).tiny), 0.1])
def test_normal_scales_survive_init_and_resampling(minimum: float) -> None:
    stream = XDistShiftStream(4, 2, scale_min=minimum, scale_max=2 * minimum,
                              scale_change_interval=1, noise_in_target=False)
    state = jax.jit(stream.init)(jax.random.key(0))
    initial_scales = np.asarray(state.current_scales)
    for index in range(3):
        _, state = jax.jit(stream.step)(state, jnp.asarray(index, dtype=jnp.int32))
        scales = np.asarray(state.current_scales)
        assert np.all(np.isfinite(scales))
        assert np.all(scales >= minimum)
        assert np.all(scales < 2 * minimum)
    assert np.any(initial_scales != scales)
