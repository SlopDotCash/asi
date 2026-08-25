"""Bounded native smoke qualification for external world-model adapters.

The smoke uses one deterministic state-observation trace to exercise ASI's
observation-space, latent-space, sparse FTL, and latent mechanism-off surfaces.
Losses in different target spaces are deliberately not ranked.  This is a
development qualification record, never scientific or performance evidence.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.ftl_world_model import (
    SparseFTLWorldModel,
    SparseFTLWorldModelConfig,
)
from alberta_framework.core.latent_world_model import LatentWorldModel, LatentWorldModelConfig
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)

WORLD_MODEL_SMOKE_SCHEMA = "asi.external_world_model_smoke.v1"
_MAX_STEPS = 64


def _exact_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _tree_nbytes(tree: object) -> int:
    total = 0
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) > 4096:
        raise ValueError("model state contains too many array leaves")
    for leaf in leaves:
        array = np.asarray(leaf)
        total += int(array.nbytes)
        if total > 2**31 - 1:
            raise ValueError("model state exceeds signed-int32 byte capacity")
    return total


@dataclasses.dataclass(frozen=True, slots=True)
class WorldModelSmokeArm:
    arm_id: str
    metric_space: str
    prequential_losses: tuple[float, ...]
    persistent_bytes: int
    environment_steps: int
    model_updates: int
    model_queries: int

    def __post_init__(self) -> None:
        for name in ("arm_id", "metric_space"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode("utf-8")) > 96:
                raise ValueError(f"{name} must be a bounded exact string")
        if type(self.prequential_losses) is not tuple or not self.prequential_losses:
            raise ValueError("prequential_losses must be a non-empty exact tuple")
        if len(self.prequential_losses) > _MAX_STEPS:
            raise ValueError("prequential_losses contains too many steps")
        if any(type(value) is not float or not math.isfinite(value) or value < 0.0
               for value in self.prequential_losses):
            raise ValueError("prequential_losses must contain finite nonnegative exact floats")
        for name in ("persistent_bytes", "environment_steps", "model_updates", "model_queries"):
            value = getattr(self, name)
            _exact_int(value, name=name, minimum=1, maximum=2**31 - 1)
        if not self.environment_steps == self.model_updates == self.model_queries:
            raise ValueError(
                "smoke accounting must record one update and query per environment step"
            )
        if len(self.prequential_losses) != self.environment_steps:
            raise ValueError("one prequential loss is required per environment step")


@dataclasses.dataclass(frozen=True, slots=True)
class WorldModelSmokeResult:
    schema: str
    seed: int
    steps: int
    actions: tuple[int, ...]
    arms: tuple[WorldModelSmokeArm, ...]
    development_only: bool = True
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != WORLD_MODEL_SMOKE_SCHEMA:
            raise ValueError("unsupported world-model smoke schema")
        _exact_int(self.seed, name="seed", minimum=0, maximum=2**32 - 1)
        steps = _exact_int(self.steps, name="steps", minimum=1, maximum=_MAX_STEPS)
        if type(self.actions) is not tuple or len(self.actions) != steps:
            raise ValueError("actions must be an exact tuple with one item per step")
        if any(type(action) is not int or action not in (0, 1) for action in self.actions):
            raise ValueError("actions must contain exact binary integers")
        if type(self.arms) is not tuple or len(self.arms) != 4:
            raise ValueError("smoke result must contain the four frozen arms")
        if any(type(arm) is not WorldModelSmokeArm for arm in self.arms):
            raise ValueError("arms must contain exact WorldModelSmokeArm values")
        for arm in self.arms:
            WorldModelSmokeArm.__post_init__(arm)
        if tuple((arm.arm_id, arm.metric_space) for arm in self.arms) != (
            ("observation_space", "observation_mse"),
            ("latent_action_interactions", "latent_prediction_mse"),
            ("latent_no_interactions", "latent_prediction_mse"),
            ("sparse_ftl", "observation_delta_mse"),
        ):
            raise ValueError("smoke arms and metric spaces differ from the frozen roster")
        if any(arm.environment_steps != steps for arm in self.arms):
            raise ValueError("every smoke arm must match the outer step count")
        if type(self.development_only) is not bool or self.development_only is not True:
            raise ValueError("smoke result must remain development-only")
        if (type(self.scientific_promotion_allowed) is not bool
                or self.scientific_promotion_allowed is not False):
            raise ValueError("smoke result may not allow scientific promotion")


def _trace(steps: int) -> tuple[tuple[int, ...], tuple[tuple[np.ndarray, np.ndarray], ...]]:
    actions = tuple(index % 2 for index in range(steps))
    observation = np.asarray([0.25, -0.5], dtype=np.float32)
    transitions: list[tuple[np.ndarray, np.ndarray]] = []
    for action in actions:
        next_observation = np.asarray(
            [0.75 * observation[0] + float(action), -0.5 * observation[1] + 0.25 * action],
            dtype=np.float32,
        )
        transitions.append((observation.copy(), next_observation.copy()))
        observation = next_observation
    return actions, tuple(transitions)


def run_native_world_model_smoke(*, seed: object = 0, steps: object = 8) -> WorldModelSmokeResult:
    """Run the bounded matched trace without claiming cross-space loss comparability."""

    host_seed = _exact_int(seed, name="seed", minimum=0, maximum=2**32 - 1)
    host_steps = _exact_int(steps, name="steps", minimum=1, maximum=_MAX_STEPS)
    actions, transitions = _trace(host_steps)
    keys = jr.split(jr.key(host_seed), 3)
    direct = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
            use_layer_norm=False,
            include_action_interactions=True,
        )
    )
    latent = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=2,
            hidden_sizes=(),
            sparsity=0.0,
            use_layer_norm=False,
            include_action_interactions=True,
        )
    )
    latent_off = LatentWorldModel(
        dataclasses.replace(latent.config, include_action_interactions=False)
    )
    ftl = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(observation_dim=2, action_dim=2, projection_dim=2, bins=3)
    )
    direct_state = direct.init(keys[0])
    latent_state = latent.init(keys[1])
    latent_off_state = latent_off.init(keys[1])
    ftl_state = ftl.init(keys[2])
    initial_bytes = (
        _tree_nbytes(direct_state),
        _tree_nbytes(latent_state),
        _tree_nbytes(latent_off_state),
        _tree_nbytes(ftl_state),
    )
    losses: tuple[list[float], ...] = ([], [], [], [])
    for action, (observation, next_observation) in zip(actions, transitions, strict=True):
        action_array = jnp.asarray(action, dtype=jnp.int32)
        reward = jnp.asarray(float(next_observation[0]), dtype=jnp.float32)
        discount = jnp.asarray(0.99, dtype=jnp.float32)
        direct_result = direct.update(
            direct_state, observation, action_array, reward, discount, next_observation
        )
        latent_result = latent.update(
            latent_state, observation, action_array, reward, discount, next_observation
        )
        latent_off_result = latent_off.update(
            latent_off_state, observation, action_array, reward, discount, next_observation
        )
        one_hot = jax.nn.one_hot(action_array, 2, dtype=jnp.float32)
        ftl_result = ftl.update(ftl_state, observation, one_hot, next_observation)
        direct_state = direct_result.state
        latent_state = latent_result.state
        latent_off_state = latent_off_result.state
        ftl_state = ftl_result.state
        losses[0].append(float(direct_result.observation_mse))
        losses[1].append(float(latent_result.prediction_error))
        losses[2].append(float(latent_off_result.prediction_error))
        losses[3].append(float(ftl_result.squared_error))
    final_states = (direct_state, latent_state, latent_off_state, ftl_state)
    # Some learner host-only timestamp leaves canonicalize from float64 to
    # float32 on the first JIT update. Charge the larger observed state rather
    # than pretending that this representation transition costs no bytes.
    persistent_bytes = tuple(
        max(initial, _tree_nbytes(final))
        for initial, final in zip(initial_bytes, final_states, strict=True)
    )
    metric_spaces = (
        "observation_mse",
        "latent_prediction_mse",
        "latent_prediction_mse",
        "observation_delta_mse",
    )
    arm_ids = (
        "observation_space",
        "latent_action_interactions",
        "latent_no_interactions",
        "sparse_ftl",
    )
    arms = tuple(
        WorldModelSmokeArm(
            arm_id=arm_id,
            metric_space=metric_space,
            prequential_losses=tuple(float(value) for value in arm_losses),
            persistent_bytes=state_bytes,
            environment_steps=host_steps,
            model_updates=host_steps,
            model_queries=host_steps,
        )
        for arm_id, metric_space, arm_losses, state_bytes in zip(
            arm_ids, metric_spaces, losses, persistent_bytes, strict=True
        )
    )
    return WorldModelSmokeResult(
        schema=WORLD_MODEL_SMOKE_SCHEMA,
        seed=host_seed,
        steps=host_steps,
        actions=actions,
        arms=arms,
    )


__all__ = [
    "WORLD_MODEL_SMOKE_SCHEMA",
    "WorldModelSmokeArm",
    "WorldModelSmokeResult",
    "run_native_world_model_smoke",
]
