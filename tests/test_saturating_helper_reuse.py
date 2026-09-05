"""Pin canonical saturating-helper reuse for follow-up inline counter sites.

Seven more core modules used ``jnp.minimum(counter, MAX-1) + 1`` without the
canonical negative clamp from
``alberta_framework.core.normalizers._saturating_int32_counter_increment``
(``min(max(counter, 0), MAX-1) + 1``), so hostile negative state stayed
negative (``-5 -> -4``, ``-1 -> 0``) instead of clamping to ``1``.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest

from alberta_framework.core.normalizers import _saturating_int32_counter_increment

pytestmark = pytest.mark.unit

_FOLLOWUP_SOURCES = (
    "alberta_framework/core/off_policy_td.py",
    "alberta_framework/core/actor_critic.py",
    "alberta_framework/core/independent_demon_horde.py",
    "alberta_framework/core/resource_manager.py",
    "alberta_framework/core/sarsa.py",
    "alberta_framework/core/upgd_memory.py",
    "alberta_framework/core/cumulant_discovery.py",
)

_UNCLAMPED_CLONES = (
    "jnp.minimum(state.step_count, _INT32_MAX - 1) + 1",
    "jnp.minimum(demon_state.step_count, _INT32_MAX - 1) + 1",
    "jnp.minimum(state.ages, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)) + 1",
    "jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),\n                )\n                + 1",
)


def test_canonical_helper_clamps_negative_and_saturates() -> None:
    helper = _saturating_int32_counter_increment
    assert int(helper(jnp.asarray(-5, dtype=jnp.int32))) == 1
    assert int(helper(jnp.asarray(-1, dtype=jnp.int32))) == 1
    assert int(helper(jnp.asarray(0, dtype=jnp.int32))) == 1
    assert int(helper(jnp.asarray(5, dtype=jnp.int32))) == 6
    assert int(helper(jnp.asarray(2_147_483_647, dtype=jnp.int32))) == 2_147_483_647


def test_followup_sites_reuse_canonical_helper() -> None:
    for rel in _FOLLOWUP_SOURCES:
        text = Path(rel).read_text()
        assert "_saturating_int32_counter_increment" in text, rel
        assert "from alberta_framework.core.normalizers import" in text, rel
        for clone in _UNCLAMPED_CLONES:
            assert clone not in text, rel
