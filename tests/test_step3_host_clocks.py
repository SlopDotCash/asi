"""The host Step 3 update must retain incoming lifecycle metadata across modes."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.horde import MixedHordeState
from alberta_framework.core.independent_demon_horde import IndependentDemonHordeState
from alberta_framework.steps.step3 import (
    MultiHeadMLPState,
    Step3HordeConfig,
    init_step3_state,
    make_step3_horde,
    step3_update,
)

_BIRTH = 1_700_000_000.125
_UPTIME = 16_777_216.25


@pytest.mark.parametrize(
    ("routing", "expected_cls"),
    [
        ("shared", MultiHeadMLPState),
        ("independent", IndependentDemonHordeState),
        ("mixed", MixedHordeState),
    ],
)
@pytest.mark.parametrize("cumulant_val", [1.0, float("nan")], ids=["active", "inactive"])
def test_step3_update_retains_host_metadata(
    routing: str,
    expected_cls: type,
    cumulant_val: float,
) -> None:
    cfg = Step3HordeConfig(
        gammas=(0.0, 0.9),
        lamdas=(0.0, 0.5),
        hidden_sizes=(4,),
        routing=routing,  # type: ignore[arg-type]
    )
    horde = make_step3_horde(cfg)
    state = init_step3_state(horde, feature_dim=2, key=jr.key(0))
    assert type(state) is expected_cls

    if isinstance(state, MultiHeadMLPState):
        state = state.replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)  # type: ignore[attr-defined]
    elif isinstance(state, IndependentDemonHordeState):
        state = state.replace(  # type: ignore[attr-defined]
            birth_timestamp=_BIRTH,
            uptime_s=_UPTIME,
            demon_states=tuple(
                d.replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)  # type: ignore[attr-defined]
                for d in state.demon_states
            ),
        )
    elif isinstance(state, MixedHordeState):
        assert state.shared_state is not None
        assert state.independent_state is not None
        state = state.replace(  # type: ignore[attr-defined]
            birth_timestamp=_BIRTH,
            uptime_s=_UPTIME,
            shared_state=state.shared_state.replace(  # type: ignore[attr-defined]
                birth_timestamp=_BIRTH,
                uptime_s=_UPTIME,
            ),
            independent_state=state.independent_state.replace(  # type: ignore[attr-defined]
                birth_timestamp=_BIRTH,
                uptime_s=_UPTIME,
                demon_states=tuple(
                    d.replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)  # type: ignore[attr-defined]
                    for d in state.independent_state.demon_states
                ),
            ),
        )

    result = step3_update(
        horde,
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.full(2, cumulant_val, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
    )
    jax.block_until_ready(result)
    new_state = result.state
    assert type(new_state.birth_timestamp) is float
    assert type(new_state.uptime_s) is float
    assert new_state.birth_timestamp == _BIRTH
    assert new_state.uptime_s == _UPTIME

    if isinstance(new_state, IndependentDemonHordeState):
        for demon in new_state.demon_states:
            assert type(demon.birth_timestamp) is float
            assert type(demon.uptime_s) is float
            assert demon.birth_timestamp == _BIRTH
            assert demon.uptime_s == _UPTIME
    elif isinstance(new_state, MixedHordeState):
        assert new_state.shared_state is not None
        assert new_state.independent_state is not None
        assert type(new_state.shared_state.birth_timestamp) is float
        assert type(new_state.shared_state.uptime_s) is float
        assert new_state.shared_state.birth_timestamp == _BIRTH
        assert new_state.shared_state.uptime_s == _UPTIME
        assert type(new_state.independent_state.birth_timestamp) is float
        assert type(new_state.independent_state.uptime_s) is float
        assert new_state.independent_state.birth_timestamp == _BIRTH
        assert new_state.independent_state.uptime_s == _UPTIME
        for demon in new_state.independent_state.demon_states:
            assert type(demon.birth_timestamp) is float
            assert type(demon.uptime_s) is float
            assert demon.birth_timestamp == _BIRTH
            assert demon.uptime_s == _UPTIME
