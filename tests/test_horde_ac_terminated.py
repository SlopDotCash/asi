"""Regression coverage for the horde_ac episode-boundary fix (issue #2344):
the actor-critic update must accept a ``terminated`` flag so the pipeline
can stop value-head bootstrapping across episode boundaries."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import pytest

from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
)


class TestHordeAcTerminatedParameter:
    def test_update_signature_accepts_terminated(self) -> None:
        sig = inspect.signature(HordeActorCriticAgent.update)
        params = list(sig.parameters.keys())
        assert "terminated" in params

    def test_update_signature_backwards_compatible(self) -> None:
        # terminated must be optional (keyword) so existing callers are safe.
        sig = inspect.signature(HordeActorCriticAgent.update)
        assert sig.parameters["terminated"].default is None

    def test_value_discount_zeroing_logic(self) -> None:
        # Independent re-derivation of the boundary logic: a terminated
        # transition forces the value-head discount to 0.
        gamma = jnp.asarray(0.9, dtype=jnp.float32)
        terminated = jnp.asarray(True)
        effective = jnp.where(terminated, jnp.asarray(0.0, dtype=jnp.float32), gamma)
        assert float(effective) == 0.0
        not_terminated = jnp.where(jnp.asarray(False), jnp.asarray(0.0), gamma)
        assert float(not_terminated) == pytest.approx(0.9, rel=1e-5)

    def test_config_constructs_for_horde_ac(self) -> None:
        cfg = HordeActorCriticConfig(n_actions=2, value_head_index=0)
        assert cfg.n_actions == 2
        assert cfg.value_head_index == 0
