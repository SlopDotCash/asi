"""Forager open baselines: DQN, A3C, Horde, random.

This module implements baseline RL algorithms for the Forager gridworld domain.
Each baseline follows a standard interface: init() and act(state, training=bool).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "DQNAgent",
    "A3CAgent",
    "HordeAgent",
    "RandomAgent",
    "make_baseline",
]


# =============================================================================
# Random Baseline (trivial)
# =============================================================================


@dataclasses.dataclass
class RandomAgent:
    """Baseline: uniform random action sampling."""

    action_dim: int
    key: jax.random.PRNGKey = dataclasses.field(default_factory=lambda: jax.random.PRNGKey(0))

    def init(self, key: jax.random.PRNGKey, state_dim: int) -> None:
        """Initialize (no-op for random)."""
        self.key = key

    def act(self, state: Array, training: bool = True) -> int:
        """Sample random action."""
        self.key, subkey = jax.random.split(self.key)
        return int(jax.random.randint(subkey, (), 0, self.action_dim))

    def update(self, transition: dict[str, Any]) -> None:
        """Update (no-op for random)."""
        pass


# =============================================================================
# DQN Baseline
# =============================================================================


class DQNTransition(NamedTuple):
    """Experience tuple for replay buffer."""
    state: Array
    action: int
    reward: float
    next_state: Array
    done: bool


@dataclasses.dataclass
class DQNAgent:
    """Deep Q-Network: off-policy value-based control.

    Architecture: MLP Q-network for state->action values
    Mechanisms: experience replay, target network, epsilon-greedy exploration
    """

    action_dim: int
    state_dim: int
    hidden_dim: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 1000
    replay_buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 1000

    # State
    q_network_params: dict = dataclasses.field(default_factory=dict)
    target_network_params: dict = dataclasses.field(default_factory=dict)
    replay_buffer: list[DQNTransition] = dataclasses.field(default_factory=list)
    optimizer_state: dict = dataclasses.field(default_factory=dict)
    step_count: int = 0
    key: jax.random.PRNGKey = dataclasses.field(default_factory=lambda: jax.random.PRNGKey(0))

    def init(self, key: jax.random.PRNGKey, state_dim: int = None) -> None:
        """Initialize Q-networks and optimizer."""
        self.key = key
        self.state_dim = state_dim or self.state_dim

        # Initialize Q-network parameters
        key, subkey = jax.random.split(key)
        self.q_network_params = self._init_network(subkey)
        self.target_network_params = {k: v.copy() for k, v in self.q_network_params.items()}

        # Simple SGD optimizer state (learning rate)
        self.optimizer_state = {"learning_rate": self.learning_rate}
        self.step_count = 0

    def _init_network(self, key: jax.random.PRNGKey) -> dict[str, Array]:
        """Initialize MLP parameters: hidden_dim -> hidden_dim -> action_dim."""
        keys = jax.random.split(key, 4)
        return {
            "w1": jax.random.normal(keys[0], (self.state_dim, self.hidden_dim)) * 0.01,
            "b1": jnp.zeros(self.hidden_dim),
            "w2": jax.random.normal(keys[1], (self.hidden_dim, self.hidden_dim)) * 0.01,
            "b2": jnp.zeros(self.hidden_dim),
            "w_out": jax.random.normal(keys[2], (self.hidden_dim, self.action_dim)) * 0.01,
            "b_out": jnp.zeros(self.action_dim),
        }

    def _forward(self, params: dict[str, Array], state: Array) -> Array:
        """Forward pass: state -> action values."""
        x = state @ params["w1"] + params["b1"]
        x = jax.nn.relu(x)
        x = x @ params["w2"] + params["b2"]
        x = jax.nn.relu(x)
        q_values = x @ params["w_out"] + params["b_out"]
        return q_values

    def _epsilon(self) -> float:
        """Epsilon decay schedule."""
        progress = min(self.step_count / self.epsilon_decay, 1.0)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress)

    def act(self, state: Array, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        self.key, subkey = jax.random.split(self.key)

        if training and jax.random.uniform(subkey) < self._epsilon():
            # Explore: random action
            self.key, subkey = jax.random.split(self.key)
            return int(jax.random.randint(subkey, (), 0, self.action_dim))
        else:
            # Exploit: argmax Q
            q_values = self._forward(self.q_network_params, state)
            return int(jnp.argmax(q_values))

    def update(self, transition: DQNTransition) -> None:
        """Store transition and perform Q-learning update."""
        # Store in replay buffer
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

        self.step_count += 1

        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.target_network_params = {
                k: v.copy() for k, v in self.q_network_params.items()
            }

        # Training step if we have enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()

    def _train_step(self) -> None:
        """Perform one Q-learning update on a batch."""
        self.key, subkey = jax.random.split(self.key)
        indices = jax.random.choice(
            subkey, len(self.replay_buffer), shape=(self.batch_size,), replace=False
        )

        batch = [self.replay_buffer[i] for i in indices]
        states = jnp.stack([t.state for t in batch])
        actions = jnp.array([t.action for t in batch])
        rewards = jnp.array([t.reward for t in batch])
        next_states = jnp.stack([t.next_state for t in batch])
        dones = jnp.array([float(t.done) for t in batch])

        # Compute target Q-values
        next_q_values = jax.vmap(lambda s: self._forward(self.target_network_params, s))(
            next_states
        )
        next_max_q = jnp.max(next_q_values, axis=1)
        target_q = rewards + self.gamma * next_max_q * (1 - dones)

        # Compute current Q-values
        current_q_values = jax.vmap(lambda s: self._forward(self.q_network_params, s))(states)
        current_q = current_q_values[jnp.arange(self.batch_size), actions]

        # MSE loss
        loss = jnp.mean((current_q - target_q) ** 2)

        # Simple SGD update
        grad_w_out = 2 * jnp.mean((current_q - target_q)[:, None] *
                                   (current_q_values[jnp.arange(self.batch_size), :]), axis=0)
        self.q_network_params["w_out"] -= self.learning_rate * grad_w_out
        # (Simplified: full backprop would require jax.grad)


# =============================================================================
# A3C Baseline (simplified synchronous version)
# =============================================================================


@dataclasses.dataclass
class A3CAgent:
    """Actor-Critic (synchronous A3C): on-policy policy-gradient control."""

    action_dim: int
    state_dim: int
    hidden_dim: int = 128
    actor_learning_rate: float = 0.001
    critic_learning_rate: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # State
    actor_params: dict = dataclasses.field(default_factory=dict)
    critic_params: dict = dataclasses.field(default_factory=dict)
    key: jax.random.PRNGKey = dataclasses.field(default_factory=lambda: jax.random.PRNGKey(0))

    def init(self, key: jax.random.PRNGKey, state_dim: int = None) -> None:
        """Initialize actor and critic networks."""
        self.key = key
        self.state_dim = state_dim or self.state_dim

        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.actor_params = self._init_network(subkey1, output_dim=self.action_dim)
        self.critic_params = self._init_network(subkey2, output_dim=1)

    def _init_network(self, key: jax.random.PRNGKey, output_dim: int) -> dict[str, Array]:
        """Initialize MLP: state_dim -> hidden_dim -> output_dim."""
        keys = jax.random.split(key, 3)
        return {
            "w1": jax.random.normal(keys[0], (self.state_dim, self.hidden_dim)) * 0.01,
            "b1": jnp.zeros(self.hidden_dim),
            "w_out": jax.random.normal(keys[1], (self.hidden_dim, output_dim)) * 0.01,
            "b_out": jnp.zeros(output_dim),
        }

    def _forward_actor(self, state: Array) -> Array:
        """Actor forward: state -> logits for action distribution."""
        x = state @ self.actor_params["w1"] + self.actor_params["b1"]
        x = jax.nn.relu(x)
        logits = x @ self.actor_params["w_out"] + self.actor_params["b_out"]
        return logits  # softmax applied during sampling

    def _forward_critic(self, state: Array) -> Array:
        """Critic forward: state -> value estimate."""
        x = state @ self.critic_params["w1"] + self.critic_params["b1"]
        x = jax.nn.relu(x)
        value = x @ self.critic_params["w_out"] + self.critic_params["b_out"]
        return value.squeeze()

    def act(self, state: Array, training: bool = True) -> int:
        """Sample action from policy."""
        self.key, subkey = jax.random.split(self.key)
        logits = self._forward_actor(state)
        log_probs = jax.nn.log_softmax(logits)
        action = jax.random.categorical(subkey, log_probs)
        return int(action)

    def update(self, trajectory: list[dict] | dict) -> None:
        """Update actor and critic on a trajectory (episode or batch)."""
        # Handle single transition (convert to list)
        if isinstance(trajectory, dict):
            trajectory = [trajectory]

        if not trajectory:
            return

        # Compute returns and advantages
        states = jnp.stack([t["state"] for t in trajectory])
        actions = jnp.array([t["action"] for t in trajectory])
        rewards = jnp.array([t["reward"] for t in trajectory])
        values = jax.vmap(self._forward_critic)(states)

        # TD residuals (advantages)
        next_values = jnp.concatenate([values[1:], jnp.array([0.0])])
        td_residuals = rewards + self.gamma * next_values - values

        # Actor loss: -log(pi) * advantage
        logits = jax.vmap(self._forward_actor)(states)
        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = log_probs[jnp.arange(len(trajectory)), actions]
        actor_loss = -jnp.mean(action_log_probs * td_residuals)

        # Critic loss: MSE on value targets
        targets = rewards + self.gamma * next_values
        critic_loss = jnp.mean((values - targets) ** 2)

        # Simple gradient updates (would use jax.grad in production)
        self.actor_params["w_out"] -= self.actor_learning_rate * actor_loss
        self.critic_params["w_out"] -= self.critic_learning_rate * critic_loss


# =============================================================================
# Horde Baseline (wrapper around existing Horde)
# =============================================================================


@dataclasses.dataclass
class HordeAgent:
    """Horde: GVF-based option discovery and aggregation."""

    action_dim: int
    state_dim: int
    n_demons: int = 4

    def init(self, key: jax.random.PRNGKey, state_dim: int = None) -> None:
        """Initialize Horde demon pool (stub for now)."""
        self.state_dim = state_dim or self.state_dim
        # In production: initialize option demons from alberta_framework.core.horde

    def act(self, state: Array, training: bool = True) -> int:
        """Aggregate policies from demon pool."""
        # In production: query each demon for action, combine via voting or weighting
        # For now: return random action as placeholder
        return np.random.randint(0, self.action_dim)

    def update(self, transition: dict[str, Any]) -> None:
        """Update demon policies."""
        # In production: update each demon's value estimates
        pass


# =============================================================================
# Factory
# =============================================================================


def make_baseline(
    baseline_type: str,
    action_dim: int,
    state_dim: int = None,
    **kwargs,
) -> DQNAgent | A3CAgent | HordeAgent | RandomAgent:
    """Create a baseline agent."""
    if baseline_type == "dqn":
        return DQNAgent(action_dim=action_dim, state_dim=state_dim, **kwargs)
    elif baseline_type == "a3c":
        return A3CAgent(action_dim=action_dim, state_dim=state_dim, **kwargs)
    elif baseline_type == "horde":
        return HordeAgent(action_dim=action_dim, state_dim=state_dim, **kwargs)
    elif baseline_type == "random":
        return RandomAgent(action_dim=action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
