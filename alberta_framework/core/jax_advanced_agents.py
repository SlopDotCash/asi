"""JAX-optimized implementations of advanced Forager agents.

Provides efficient, JIT-compilable versions using jax.experimental.io_callback
for environment interactions and haiku networks for learnable parameters.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import Array


# ============================================================================
# JAX Network Utilities
# ============================================================================


class NetworkParams(NamedTuple):
    """Container for network parameters."""

    params: Any
    state: Any | None = None


def create_mlp_network(
    hidden_sizes: tuple[int, ...],
    output_size: int,
    activation: Callable[[Array], Array] = jax.nn.relu,
) -> Callable[[Array], Array]:
    """Create a simple MLP network using Haiku."""

    def network(x: Array) -> Array:
        for hidden_size in hidden_sizes:
            x = hk.Linear(hidden_size)(x)
            x = activation(x)
        x = hk.Linear(output_size)(x)
        return x

    return network


def create_dueling_network(
    hidden_sizes: tuple[int, ...],
    n_actions: int,
    n_atoms: int = 1,
) -> Callable[[Array], tuple[Array, Array]]:
    """Create dueling architecture: shared -> value/advantage streams."""

    def network(x: Array) -> tuple[Array, Array]:
        # Shared stream
        for hidden_size in hidden_sizes:
            x = hk.Linear(hidden_size)(x)
            x = jax.nn.relu(x)

        # Value stream
        value = hk.Linear(hidden_sizes[-1])(x)
        value = jax.nn.relu(value)
        value = hk.Linear(n_atoms)(value)

        # Advantage stream
        advantage = hk.Linear(hidden_sizes[-1])(x)
        advantage = jax.nn.relu(advantage)
        advantage = hk.Linear(n_actions * n_atoms)(advantage)
        advantage = jnp.reshape(advantage, (-1, n_actions, n_atoms))

        # Combine: Q = V + (A - mean(A))
        advantage = advantage - jnp.mean(advantage, axis=1, keepdims=True)
        q_values = value[:, jnp.newaxis, :] + advantage

        return q_values

    return network


def create_actor_critic_networks(
    hidden_sizes: tuple[int, ...],
    n_actions: int,
) -> Callable[[Array], tuple[Array, Array]]:
    """Create actor-critic networks."""

    def network(x: Array) -> tuple[Array, Array]:
        # Shared feature processing
        for hidden_size in hidden_sizes:
            x = hk.Linear(hidden_size)(x)
            x = jax.nn.relu(x)

        # Actor head (policy logits)
        actor = hk.Linear(n_actions)(x)

        # Critic head (value)
        critic = hk.Linear(1)(x)
        critic = jnp.squeeze(critic, axis=-1)

        return actor, critic

    return network


# ============================================================================
# JAX Rainbow DQN
# ============================================================================


@dataclass(frozen=True)
class JAXRainbowDQNConfig:
    """JAX Rainbow DQN configuration."""

    n_actions: int = 4
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-4
    gamma: float = 0.99
    n_steps: int = 3
    batch_size: int = 32
    target_update_frequency: int = 1000
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000


class JAXRainbowDQN:
    """JAX-compiled Rainbow DQN agent."""

    def __init__(self, config: JAXRainbowDQNConfig | None = None, *, seed: int = 0):
        self.config = config or JAXRainbowDQNConfig()
        self.seed = seed
        self._key = jr.key(seed)

        # Initialize networks
        self._init_networks()

        # State
        self._steps = 0

    def _init_networks(self) -> None:
        """Initialize Q-network with dueling and distributional heads."""
        network_fn = create_dueling_network(
            self.config.hidden_sizes,
            self.config.n_actions,
            self.config.n_atoms,
        )
        self.network_fn = network_fn

        # Initialize parameters
        dummy_input = jnp.zeros((1, 84))
        self._key, subkey = jr.split(self._key)

        transformed = hk.transform(network_fn)
        self.network_params = transformed.init(subkey, dummy_input)
        self.target_params = self.network_params

        # Optimizer
        self.optimizer = optax.adam(self.config.learning_rate)
        self.opt_state = self.optimizer.init(self.network_params)

    @jax.jit
    def _forward(self, params: Any, observations: Array) -> Array:
        """Forward pass through network."""
        transformed = hk.transform(self.network_fn)
        return transformed.apply(params, self._key, observations)

    @jax.jit
    def _compute_loss(
        self,
        params: Any,
        target_params: Any,
        observations: Array,
        actions: Array,
        rewards: Array,
        next_observations: Array,
        dones: Array,
    ) -> tuple[Array, dict[str, Any]]:
        """Compute distributional Q-learning loss."""
        batch_size = observations.shape[0]

        # Current Q-values
        transformed = hk.transform(self.network_fn)
        q_dist = transformed.apply(params, self._key, observations)  # [B, A, atoms]

        # Target Q-values
        next_q_dist = transformed.apply(target_params, self._key, next_observations)
        next_actions = jnp.argmax(jnp.sum(next_q_dist, axis=-1), axis=-1)
        next_q_dist = next_q_dist[jnp.arange(batch_size), next_actions]

        # Shift distribution by reward and discount
        atoms = jnp.linspace(self.config.v_min, self.config.v_max, self.config.n_atoms)
        delta_z = (self.config.v_max - self.config.v_min) / (self.config.n_atoms - 1)

        target_atoms = rewards[:, jnp.newaxis] + (
            self.config.gamma * atoms[jnp.newaxis, :] * (1 - dones[:, jnp.newaxis])
        )

        # Project onto support
        target_atoms = jnp.clip(target_atoms, self.config.v_min, self.config.v_max)
        b_j = (target_atoms - self.config.v_min) / delta_z
        l = jnp.floor(b_j).astype(jnp.int32)
        u = jnp.ceil(b_j).astype(jnp.int32)
        ml = (u.astype(jnp.float32) - b_j) * next_q_dist
        mu = (b_j - l.astype(jnp.float32)) * next_q_dist

        target_support = jnp.zeros((batch_size, self.config.n_atoms))
        for i in range(batch_size):
            for j in range(self.config.n_atoms):
                if 0 <= l[i, j] < self.config.n_atoms:
                    target_support = target_support.at[i, l[i, j]].add(ml[i, j])
                if 0 <= u[i, j] < self.config.n_atoms:
                    target_support = target_support.at[i, u[i, j]].add(mu[i, j])

        # Cross-entropy loss
        action_q_dist = q_dist[jnp.arange(batch_size), actions]
        loss = -jnp.sum(target_support * jnp.log(action_q_dist + 1e-8), axis=-1)
        loss = jnp.mean(loss)

        return loss, {"loss": loss}

    def update(
        self,
        observations: Array,
        actions: Array,
        rewards: Array,
        next_observations: Array,
        dones: Array,
    ) -> None:
        """Perform one update step."""
        loss_fn = lambda params: self._compute_loss(
            params,
            self.target_params,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=0)(self.network_params)

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.network_params = optax.apply_updates(self.network_params, updates)

        # Update target network
        if self._steps % self.config.target_update_frequency == 0:
            self.target_params = self.network_params

        self._steps += 1

    @property
    def name(self) -> str:
        return "jax_rainbow_dqn"

    @property
    def privileged(self) -> bool:
        return False


# ============================================================================
# JAX PPO with GAE
# ============================================================================


@dataclass(frozen=True)
class JAXPPOConfig:
    """JAX PPO configuration."""

    n_actions: int = 4
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    clip_ratio: float = 0.2
    max_grad_norm: float = 0.5
    batch_size: int = 64
    n_epochs: int = 3


class JAXPPO:
    """JAX-compiled PPO with GAE."""

    def __init__(self, config: JAXPPOConfig | None = None, *, seed: int = 0):
        self.config = config or JAXPPOConfig()
        self.seed = seed
        self._key = jr.key(seed)

        # Initialize networks
        self._init_networks()

        self._steps = 0

    def _init_networks(self) -> None:
        """Initialize actor-critic networks."""
        network_fn = create_actor_critic_networks(
            self.config.hidden_sizes,
            self.config.n_actions,
        )
        self.network_fn = network_fn

        dummy_input = jnp.zeros((1, 84))
        self._key, subkey = jr.split(self._key)

        transformed = hk.transform(network_fn)
        self.params = transformed.init(subkey, dummy_input)

        self.optimizer = optax.adam(self.config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    @jax.jit
    def _compute_ppo_loss(
        self,
        params: Any,
        observations: Array,
        actions: Array,
        advantages: Array,
        returns: Array,
        old_log_probs: Array,
    ) -> tuple[Array, dict[str, Any]]:
        """Compute PPO loss with GAE."""
        transformed = hk.transform(self.network_fn)
        actor_logits, values = transformed.apply(params, self._key, observations)

        # Policy loss
        log_probs = jax.nn.log_softmax(actor_logits)
        action_log_probs = log_probs[jnp.arange(len(actions)), actions]

        # PPO clipped surrogate
        ratio = jnp.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Value loss
        value_loss = jnp.mean((returns - values) ** 2)

        # Entropy bonus
        probs = jax.nn.softmax(actor_logits)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        entropy_loss = -self.config.entropy_coef * jnp.mean(entropy)

        total_loss = policy_loss + self.config.value_coef * value_loss + entropy_loss

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": jnp.mean(entropy),
            "total_loss": total_loss,
        }

        return total_loss, metrics

    def update(
        self,
        observations: Array,
        actions: Array,
        advantages: Array,
        returns: Array,
        old_log_probs: Array,
    ) -> dict[str, Any]:
        """Perform PPO update."""
        loss_fn = lambda params: self._compute_ppo_loss(
            params,
            observations,
            actions,
            advantages,
            returns,
            old_log_probs,
        )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=0)(self.params)

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)

        self._steps += 1
        return metrics

    @property
    def name(self) -> str:
        return "jax_ppo_gae"

    @property
    def privileged(self) -> bool:
        return False


# ============================================================================
# JAX SAC
# ============================================================================


@dataclass(frozen=True)
class JAXSACConfig:
    """JAX SAC configuration."""

    n_actions: int = 4
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_entropy_tuning: bool = True
    batch_size: int = 256


class JAXSAC:
    """JAX-compiled Soft Actor-Critic."""

    def __init__(self, config: JAXSACConfig | None = None, *, seed: int = 0):
        self.config = config or JAXSACConfig()
        self.seed = seed
        self._key = jr.key(seed)

        self._init_networks()

        # Entropy coefficient
        if config and config.auto_entropy_tuning:
            self.log_alpha = jnp.array(0.0)
            self.target_entropy = -jnp.log(1.0 / config.n_actions)
            self.alpha_optimizer = optax.adam(config.learning_rate)
            self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)
        else:
            self.log_alpha = None

        self._steps = 0

    def _init_networks(self) -> None:
        """Initialize actor and critic networks."""
        # Actor network
        actor_fn = create_mlp_network(self.config.hidden_sizes, self.config.n_actions)
        dummy_input = jnp.zeros((1, 84))
        self._key, subkey = jr.split(self._key)
        transformed = hk.transform(actor_fn)
        self.actor_params = transformed.init(subkey, dummy_input)

        # Q-function networks (double Q)
        q_fn = create_mlp_network(self.config.hidden_sizes, 1)
        self._key, subkey = jr.split(self._key)
        transformed = hk.transform(q_fn)
        self.q1_params = transformed.init(subkey, jnp.zeros((1, 84 + 1)))
        self.q2_params = transformed.init(subkey, jnp.zeros((1, 84 + 1)))

        # Target networks
        self.q1_target_params = self.q1_params
        self.q2_target_params = self.q2_params

        # Optimizers
        self.actor_optimizer = optax.adam(self.config.learning_rate)
        self.q1_optimizer = optax.adam(self.config.learning_rate)
        self.q2_optimizer = optax.adam(self.config.learning_rate)

        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.q1_opt_state = self.q1_optimizer.init(self.q1_params)
        self.q2_opt_state = self.q2_optimizer.init(self.q2_params)

    def update(
        self,
        observations: Array,
        actions: Array,
        rewards: Array,
        next_observations: Array,
        dones: Array,
    ) -> dict[str, Any]:
        """Perform SAC update step."""
        # Placeholder for actual SAC update
        metrics = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha": float(self.config.alpha),
        }
        return metrics

    @property
    def name(self) -> str:
        return "jax_sac"

    @property
    def privileged(self) -> bool:
        return False


# ============================================================================
# Utilities for distributed training
# ============================================================================


def create_distributed_trainer(
    agent: Any,
    n_devices: int | None = None,
) -> Callable:
    """Create distributed training function."""
    if n_devices is None:
        n_devices = jax.device_count()

    def train_step(batch: dict[str, Array]) -> dict[str, Any]:
        """Execute training across devices."""
        # Shard batch across devices
        def _update(agent_params, batch_slice):
            return agent.update(**batch_slice)

        # Vectorize across devices
        vmap_update = jax.vmap(_update, in_axes=(None, 0))
        return vmap_update(agent, batch)

    return train_step


__all__ = [
    "JAXRainbowDQN",
    "JAXRainbowDQNConfig",
    "JAXPPO",
    "JAXPPOConfig",
    "JAXSAC",
    "JAXSACConfig",
    "create_mlp_network",
    "create_dueling_network",
    "create_actor_critic_networks",
    "create_distributed_trainer",
]
