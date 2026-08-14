"""Real-world usage patterns and integration examples for advanced Forager agents."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# ============================================================================
# Example 1: Basic Usage with Dummy Environment
# ============================================================================


def example_basic_usage():
    """Simplest possible usage of an advanced agent."""
    from alberta_framework.core.advanced_forager_agents import RainbowDQNAgent

    # Create agent with defaults
    agent = RainbowDQNAgent(seed=42)

    # Simulate environment
    observation = np.random.randn(84).astype(np.float32)

    # Start episode
    action = agent.start(observation)
    print(f"Initial action: {action}")

    # Run steps
    for step in range(10):
        observation = np.random.randn(84).astype(np.float32)
        reward = float(np.random.randn())
        action = agent.step(reward, observation)

    # Get metadata
    metadata = agent.metadata()
    print(f"Agent: {metadata['name']}")
    print(f"Privileged: {metadata['privileged']}")


# ============================================================================
# Example 2: Custom Configuration
# ============================================================================


def example_custom_config():
    """Configure agents with custom hyperparameters."""
    from alberta_framework.core.advanced_forager_agents import (
        PPOAgent,
        PPOConfig,
        RainbowDQNConfig,
        RainbowDQNAgent,
    )

    # Rainbow DQN: exploration-heavy configuration
    rainbow_config = RainbowDQNConfig(
        hidden_sizes=(256, 256),  # Larger network
        learning_rate=5e-5,  # Slower learning
        n_steps=5,  # 5-step returns
        replay_buffer_size=500_000,  # Large buffer
        epsilon_start=1.0,
        epsilon_decay_steps=200_000,  # Slow exploration decay
    )
    rainbow_agent = RainbowDQNAgent(config=rainbow_config, seed=42)

    # PPO: stable training configuration
    ppo_config = PPOConfig(
        hidden_sizes=(128, 128),
        learning_rate=1e-3,
        lambda_=0.95,  # GAE tradeoff
        clip_ratio=0.1,  # Tighter trust region
        n_epochs=5,  # More training epochs
        rollout_length=4096,  # Longer rollouts
    )
    ppo_agent = PPOAgent(config=ppo_config, seed=42)

    print(f"Rainbow DQN: {rainbow_config.replay_buffer_size} buffer")
    print(f"PPO: {ppo_config.rollout_length} rollout length")


# ============================================================================
# Example 3: Multi-Seed Evaluation
# ============================================================================


def example_multi_seed_evaluation():
    """Evaluate agent across multiple seeds."""
    from alberta_framework.core.advanced_forager_agents import SACAgent, SACConfig

    config = SACConfig()
    results = []

    for seed in range(5):
        agent = SACAgent(config=config, seed=seed)

        # Run episode
        total_reward = 0.0
        observation = np.random.randn(84).astype(np.float32)
        action = agent.start(observation)

        for _ in range(100):
            observation = np.random.randn(84).astype(np.float32)
            reward = float(np.random.randn())
            total_reward += reward
            action = agent.step(reward, observation)

        results.append(total_reward)

    # Aggregate
    print(f"Mean: {np.mean(results):.2f}")
    print(f"Std: {np.std(results):.2f}")
    print(f"Min: {np.min(results):.2f}")
    print(f"Max: {np.max(results):.2f}")


# ============================================================================
# Example 4: Forager Integration
# ============================================================================


def example_forager_integration():
    """Integrate advanced agent with Forager benchmark."""
    from alberta_framework.benchmarks.forager import ForagerFeatureEncoder
    from alberta_framework.core.advanced_forager_agents import PPOAgent

    # Feature encoder for Forager observations
    encoder = ForagerFeatureEncoder()

    # Agent
    agent = PPOAgent(seed=42)

    # Simulate Forager-style observation (dict with 'image' and optional 'hint')
    forager_obs = {
        "image": np.random.randn(9, 9, 3).astype(np.float32),
        "hint": np.array([0.5, 0.5], dtype=np.float32),
    }

    # Encode and use
    feature_state = encoder.init()
    features = encoder.encode(forager_obs, feature_state)

    print(f"Original obs shape: (9, 9, 3)")
    print(f"Encoded features shape: {features.shape}")

    # Run with agent
    action = agent.start(features)

    for step in range(5):
        forager_obs = {
            "image": np.random.randn(9, 9, 3).astype(np.float32),
            "hint": np.array([0.5, 0.5], dtype=np.float32),
        }
        feature_state = encoder.advance(feature_state, action=action, reward=1.0)
        features = encoder.encode(forager_obs, feature_state)
        action = agent.step(1.0, features)


# ============================================================================
# Example 5: Benchmarking Multiple Agents
# ============================================================================


def example_benchmarking():
    """Benchmark and compare multiple agents."""
    from alberta_framework.core.advanced_forager_agents import (
        ModelBasedPlannerAgent,
        PlaNetConfig,
        PPOAgent,
        PPOConfig,
        RainbowDQNAgent,
        RainbowDQNConfig,
        SACAgent,
        SACConfig,
    )

    agents = {
        "Rainbow DQN": (RainbowDQNAgent, RainbowDQNConfig()),
        "PPO": (PPOAgent, PPOConfig()),
        "SAC": (SACAgent, SACConfig()),
        "Planning": (ModelBasedPlannerAgent, PlaNetConfig()),
    }

    results = {}

    for name, (AgentClass, config) in agents.items():
        print(f"\nEvaluating {name}...")

        seed_results = []
        for seed in range(3):
            agent = AgentClass(config=config, seed=seed)

            # Run episode
            total_reward = 0.0
            observation = np.random.randn(84).astype(np.float32)
            action = agent.start(observation)

            for _ in range(100):
                observation = np.random.randn(84).astype(np.float32)
                reward = float(np.random.randn())
                total_reward += reward
                action = agent.step(reward, observation)

            seed_results.append(total_reward)

        mean_reward = np.mean(seed_results)
        std_reward = np.std(seed_results)
        results[name] = (mean_reward, std_reward)

        print(f"  Mean ± Std: {mean_reward:.2f} ± {std_reward:.2f}")

    # Rank by performance
    print("\nRanking:")
    for rank, (name, (mean, std)) in enumerate(
        sorted(results.items(), key=lambda x: x[1][0], reverse=True),
        1,
    ):
        print(f"  {rank}. {name}: {mean:.2f} ± {std:.2f}")


# ============================================================================
# Example 6: Advanced Configuration Patterns
# ============================================================================


def example_advanced_configs():
    """Advanced configuration patterns for different scenarios."""
    from alberta_framework.core.advanced_forager_agents import (
        RainbowDQNAgent,
        RainbowDQNConfig,
    )

    # Configuration 1: Fast learning with exploration
    fast_config = RainbowDQNConfig(
        hidden_sizes=(64, 64),  # Smaller network
        learning_rate=1e-3,  # Faster learning
        n_steps=1,  # Single-step TD (fast updates)
        replay_buffer_size=50_000,  # Smaller buffer
        batch_size=64,  # Larger batches
    )

    # Configuration 2: Sample-efficient learning
    efficient_config = RainbowDQNConfig(
        hidden_sizes=(256, 256),  # Larger network
        learning_rate=1e-4,  # Slower learning
        n_steps=5,  # Multi-step returns
        replay_buffer_size=500_000,  # Large buffer
        batch_size=32,  # Smaller batches for stability
        prioritized_alpha=0.8,  # Strong prioritization
    )

    # Configuration 3: Conservative learning
    conservative_config = RainbowDQNConfig(
        hidden_sizes=(128, 128),
        learning_rate=5e-5,  # Very slow learning
        gamma=0.995,  # High discount for long-term view
        n_steps=10,  # Long n-step horizon
        target_update_frequency=5000,  # Slow target updates
        epsilon_end=0.1,  # Higher minimum exploration
    )

    configs = {
        "Fast Learning": fast_config,
        "Sample Efficient": efficient_config,
        "Conservative": conservative_config,
    }

    for name, config in configs.items():
        agent = RainbowDQNAgent(config=config)
        print(f"\n{name}:")
        print(f"  Network: {config.hidden_sizes}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  N-steps: {config.n_steps}")
        print(f"  Buffer size: {config.replay_buffer_size}")


# ============================================================================
# Example 7: Monitoring Agent Training
# ============================================================================


def example_monitoring():
    """Monitor agent training progress."""
    from alberta_framework.core.advanced_forager_agents import RainbowDQNAgent

    agent = RainbowDQNAgent(seed=42)
    observation = np.random.randn(84).astype(np.float32)
    action = agent.start(observation)

    print("Training progress:")
    print("Step | Epsilon | Buffer Size | Action")
    print("-" * 50)

    for step in range(100):
        observation = np.random.randn(84).astype(np.float32)
        reward = float(np.random.randn())
        action = agent.step(reward, observation)

        if step % 10 == 0:
            buffer_size = len(agent.replay_buffer.buffer)
            print(f"{step:4d} | {agent._epsilon:.4f}  | {buffer_size:11d} | {action}")


# ============================================================================
# Example 8: Custom Agent Wrapper for Forager
# ============================================================================


class ForagerAdvancedAgent:
    """Wrapper for advanced agents with Forager-specific integration."""

    def __init__(self, agent_class, agent_config, encoder_config=None):
        from alberta_framework.benchmarks.forager import ForagerFeatureEncoder

        self.agent = agent_class(config=agent_config, seed=0)
        self.encoder = ForagerFeatureEncoder(config=encoder_config)
        self.feature_state = self.encoder.init()

    @property
    def name(self) -> str:
        """Stable name for benchmarking."""
        return f"advanced_{self.agent.name}"

    @property
    def privileged(self) -> bool:
        """No privileged state."""
        return False

    def start(self, observation: Any, context: Any = None) -> int:
        """Initialize and select first action."""
        self.feature_state = self.encoder.init()
        features = self.encoder.encode(observation, self.feature_state)
        return self.agent.start(features)

    def step(self, reward: float, observation: Any, context: Any = None) -> int:
        """Learn and select next action."""
        self.feature_state = self.encoder.advance(
            self.feature_state,
            action=self._last_action,
            reward=reward,
        )
        features = self.encoder.encode(observation, self.feature_state)
        action = self.agent.step(reward, features)
        self._last_action = action
        return action

    def metadata(self) -> Mapping[str, Any]:
        """Return metadata."""
        base_metadata = self.agent.metadata()
        return {
            **base_metadata,
            "wrapped": True,
            "encoder": "ForagerFeatureEncoder",
        }


def example_forager_wrapper():
    """Use advanced agent with Forager wrapper."""
    from alberta_framework.core.advanced_forager_agents import PPOAgent, PPOConfig

    # Create wrapped agent
    agent = ForagerAdvancedAgent(PPOAgent, PPOConfig())

    print(f"Agent name: {agent.name}")
    print(f"Privileged: {agent.privileged}")
    print(f"Metadata: {agent.metadata()}")


# ============================================================================
# Example 9: Performance Analysis
# ============================================================================


def example_performance_analysis():
    """Analyze agent performance characteristics."""
    from alberta_framework.core.advanced_forager_agents import (
        ModelBasedPlannerAgent,
        PlaNetConfig,
        PPOAgent,
        PPOConfig,
        RainbowDQNAgent,
        RainbowDQNConfig,
        SACAgent,
        SACConfig,
    )

    agents = {
        "Rainbow DQN": (RainbowDQNAgent, RainbowDQNConfig()),
        "PPO": (PPOAgent, PPOConfig()),
        "SAC": (SACAgent, SACConfig()),
        "Planning": (ModelBasedPlannerAgent, PlaNetConfig()),
    }

    print("Agent Performance Analysis")
    print("=" * 80)
    print(f"{'Agent':<15} {'Sample Eff':<12} {'Stability':<12} {'Computation':<15}")
    print("-" * 80)

    characteristics = {
        "Rainbow DQN": ("Good", "Excellent", "Moderate"),
        "PPO": ("Moderate", "Excellent", "Low"),
        "SAC": ("Excellent", "Good", "High"),
        "Planning": ("Best", "Fair", "Very High"),
    }

    for name in agents:
        chars = characteristics[name]
        print(f"{name:<15} {chars[0]:<12} {chars[1]:<12} {chars[2]:<15}")

    print("\nRecommendations:")
    print("  - Exploration-heavy: Rainbow DQN with high epsilon_start")
    print("  - Stable training: PPO with λ=0.95, clip_ratio=0.2")
    print("  - Sample efficient: SAC with large replay_buffer_size")
    print("  - Long-horizon: Planning with planning_horizon=12+")


# ============================================================================
# Main: Run Examples
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED FORAGER AGENTS - USAGE EXAMPLES")
    print("=" * 80)

    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Config", example_custom_config),
        ("Multi-Seed Evaluation", example_multi_seed_evaluation),
        ("Forager Integration", example_forager_integration),
        ("Benchmarking", example_benchmarking),
        ("Advanced Configs", example_advanced_configs),
        ("Monitoring", example_monitoring),
        ("Forager Wrapper", example_forager_wrapper),
        ("Performance Analysis", example_performance_analysis),
    ]

    for example_name, example_func in examples:
        print(f"\n\n{'=' * 80}")
        print(f"EXAMPLE: {example_name}")
        print("=" * 80)
        try:
            example_func()
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
