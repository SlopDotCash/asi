"""Supplementary coverage for ftl_decision_fidelity and
recurring_multiagent helpers.

Covers previously untested helpers: fixed_action_sequence_menu (open-loop
menu with identical first action) and scripted_meet_avoid_partner_policy
(deterministic meet/avoid movement).
"""

import jax.numpy as jnp

from alberta_framework.evaluation.ftl_decision_fidelity import (
    DecisionFidelityConfig,
    fixed_action_sequence_menu,
)
from alberta_framework.streams.recurring_multiagent import (
    scripted_meet_avoid_partner_policy,
)


def test_fixed_action_sequence_menu_shape() -> None:
    config = DecisionFidelityConfig(horizon=50, menu_amplitudes=(0.5, 1.0, 1.5))
    menu = fixed_action_sequence_menu(config)
    assert menu.shape == (3, 50, 1)


def test_fixed_action_sequence_identical_first_action() -> None:
    config = DecisionFidelityConfig(horizon=20, menu_amplitudes=(0.5, 1.0, 1.5))
    menu = fixed_action_sequence_menu(config)
    first_actions = menu[:, 0, 0]
    assert abs(float(first_actions[0] - first_actions[1])) < 1e-6
    assert abs(float(first_actions[0] - first_actions[2])) < 1e-6


def test_fixed_action_sequence_zero_at_edges() -> None:
    config = DecisionFidelityConfig(horizon=30, menu_amplitudes=(1.0, 2.0, 3.0))
    menu = fixed_action_sequence_menu(config)
    # sin(pi * 0) = sin(pi * 1) = 0 → both edges zero.
    assert abs(float(menu[0, 0, 0])) < 1e-6
    assert abs(float(menu[0, -1, 0])) < 1e-6


def test_scripted_policy_deterministic() -> None:
    key = jnp.zeros((2,), dtype=jnp.uint32)
    obs = jnp.zeros(10)
    first = scripted_meet_avoid_partner_policy(obs, key)
    second = scripted_meet_avoid_partner_policy(obs, key)
    assert jnp.array_equal(first, second)


def test_scripted_policy_shape() -> None:
    obs = jnp.zeros(10)
    key = jnp.zeros((2,), dtype=jnp.uint32)
    action = scripted_meet_avoid_partner_policy(obs, key)
    assert action.ndim == 0  # scalar action
