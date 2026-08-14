"""Integration tests and benchmarking for advanced Forager agents.

Tests all four agents (Rainbow DQN, PPO, SAC, Model-Based Planning) with
the Forager environment protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alberta_framework.core.advanced_forager_agents import (
    ModelBasedPlannerAgent,
    PlaNetConfig,
    PPOAgent,
    PPOConfig,
    PrioritizedReplayBuffer,
    RainbowDQNAgent,
    RainbowDQNConfig,
    SACAgent,
    SACConfig,
)


class ForagerAgentBenchmark:
    """Benchmark suite for advanced agents on Forager."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./advanced_agents_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {
            "agents": {},
            "timestamps": {},
            "summaries": {},
        }

    def _run_agent_episode(
        self,
        agent: Any,
        env: Any,
        max_steps: int = 1000,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Run single episode with an agent."""
        env.seed(seed)
        observation = env.reset()

        action = agent.start(observation)
        total_reward = 0.0
        step_rewards = []

        for step in range(max_steps):
            observation, reward, done, info = env.step(action)
            total_reward += reward
            step_rewards.append(reward)

            action = agent.step(reward, observation)

            if done:
                break

        return {
            "total_reward": float(total_reward),
            "steps_taken": step + 1,
            "mean_reward": float(np.mean(step_rewards)),
            "max_reward": float(np.max(step_rewards)),
            "min_reward": float(np.min(step_rewards)),
            "std_reward": float(np.std(step_rewards)),
        }

    def benchmark_agent(
        self,
        agent_class: type,
        agent_config: Any,
        agent_name: str,
        env: Any,
        n_seeds: int = 5,
        max_steps: int = 1000,
    ) -> dict[str, Any]:
        """Benchmark agent across multiple seeds."""
        results = []

        for seed in range(n_seeds):
            agent = agent_class(config=agent_config, seed=seed)
            episode_result = self._run_agent_episode(
                agent,
                env,
                max_steps=max_steps,
                seed=seed,
            )
            results.append(episode_result)

        # Aggregate results
        rewards = np.array([r["total_reward"] for r in results])
        steps = np.array([r["steps_taken"] for r in results])

        summary = {
            "agent": agent_name,
            "n_seeds": n_seeds,
            "total_reward_mean": float(rewards.mean()),
            "total_reward_std": float(rewards.std()),
            "total_reward_min": float(rewards.min()),
            "total_reward_max": float(rewards.max()),
            "steps_mean": float(steps.mean()),
            "steps_std": float(steps.std()),
            "episodes": results,
        }

        self.results["agents"][agent_name] = summary
        return summary

    def save_results(self) -> Path:
        """Save benchmark results to JSON."""
        output_file = self.output_dir / "benchmark_results.json"

        # Convert numpy types to native Python types
        def convert_to_native(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(v) for v in obj]
            return obj

        results = convert_to_native(self.results)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        return output_file

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("ADVANCED FORAGER AGENTS BENCHMARK SUMMARY")
        print("=" * 80)

        for agent_name, summary in self.results["agents"].items():
            print(f"\n{agent_name}:")
            print(f"  Mean Reward: {summary['total_reward_mean']:.2f} ± {summary['total_reward_std']:.2f}")
            print(f"  Reward Range: [{summary['total_reward_min']:.2f}, {summary['total_reward_max']:.2f}]")
            print(f"  Mean Steps: {summary['steps_mean']:.1f} ± {summary['steps_std']:.1f}")


class AgentComparisonAnalysis:
    """Analyze and compare agent performance."""

    def __init__(self, results: dict[str, Any]):
        self.results = results

    def get_performance_ranking(self) -> list[tuple[str, float]]:
        """Rank agents by mean reward."""
        rankings = []
        for agent_name, summary in self.results["agents"].items():
            mean_reward = summary["total_reward_mean"]
            rankings.append((agent_name, mean_reward))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_consistency_ranking(self) -> list[tuple[str, float]]:
        """Rank agents by consistency (lower std is better)."""
        rankings = []
        for agent_name, summary in self.results["agents"].items():
            std_reward = summary["total_reward_std"]
            rankings.append((agent_name, std_reward))

        return sorted(rankings, key=lambda x: x[1])

    def generate_report(self) -> str:
        """Generate comparison report."""
        report = "\n" + "=" * 80 + "\n"
        report += "AGENT COMPARISON REPORT\n"
        report += "=" * 80 + "\n"

        # Performance ranking
        report += "\n1. PERFORMANCE RANKING (by mean reward):\n"
        performance = self.get_performance_ranking()
        for rank, (agent_name, reward) in enumerate(performance, 1):
            report += f"  {rank}. {agent_name}: {reward:.2f}\n"

        # Consistency ranking
        report += "\n2. CONSISTENCY RANKING (by std dev, lower is better):\n"
        consistency = self.get_consistency_ranking()
        for rank, (agent_name, std) in enumerate(consistency, 1):
            report += f"  {rank}. {agent_name}: {std:.2f}\n"

        # Detailed comparison
        report += "\n3. DETAILED COMPARISON:\n"
        report += "-" * 80 + "\n"
        for agent_name, summary in self.results["agents"].items():
            report += f"\n{agent_name}:\n"
            report += f"  Mean ± Std: {summary['total_reward_mean']:.2f} ± {summary['total_reward_std']:.2f}\n"
            report += f"  Range: [{summary['total_reward_min']:.2f}, {summary['total_reward_max']:.2f}]\n"
            report += f"  Episodes: {summary['n_seeds']}\n"

        return report


# ============================================================================
# Example Usage and Integration Tests
# ============================================================================


def test_prioritized_replay_buffer() -> None:
    """Test prioritized experience replay buffer."""
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
    rng = np.random.default_rng(42)

    # Add experiences
    for i in range(50):
        experience = (i, i, i, i)
        td_error = float(i % 10)
        buffer.add(experience, td_error)

    # Sample
    samples, indices, weights = buffer.sample(batch_size=32, beta=0.4, rng=rng)

    assert len(samples) == 32
    assert len(indices) == 32
    assert len(weights) == 32
    assert weights.min() > 0
    assert weights.max() <= 1.0

    print("✓ Prioritized replay buffer test passed")


def test_rainbow_dqn_agent() -> None:
    """Test Rainbow DQN agent initialization and basic operations."""
    config = RainbowDQNConfig(
        n_actions=4,
        hidden_sizes=(64, 64),
        replay_buffer_size=1000,
        batch_size=32,
    )
    agent = RainbowDQNAgent(config=config, seed=42)

    # Test metadata
    metadata = agent.metadata()
    assert metadata["name"] == "rainbow_dqn_all_6"
    assert not metadata["privileged"]
    assert len(metadata["improvements"]) == 6

    # Test action selection
    obs = np.random.randn(84).astype(np.float32)
    action = agent.start(obs)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 4

    print("✓ Rainbow DQN agent test passed")


def test_ppo_agent() -> None:
    """Test PPO agent with GAE."""
    config = PPOConfig(
        n_actions=4,
        hidden_sizes=(64, 64),
        batch_size=32,
        rollout_length=128,
    )
    agent = PPOAgent(config=config, seed=42)

    # Test metadata
    metadata = agent.metadata()
    assert metadata["name"] == "ppo_with_gae"
    assert not metadata["privileged"]

    # Test action selection
    obs = np.random.randn(84).astype(np.float32)
    action = agent.start(obs)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 4

    print("✓ PPO with GAE agent test passed")


def test_sac_agent() -> None:
    """Test SAC agent."""
    config = SACConfig(
        n_actions=4,
        hidden_sizes=(256, 256),
        replay_buffer_size=10000,
    )
    agent = SACAgent(config=config, seed=42)

    # Test metadata
    metadata = agent.metadata()
    assert metadata["name"] == "soft_actor_critic"
    assert not metadata["privileged"]

    # Test action selection
    obs = np.random.randn(84).astype(np.float32)
    action = agent.start(obs)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 4

    print("✓ SAC agent test passed")


def test_model_based_planner() -> None:
    """Test model-based planning agent."""
    config = PlaNetConfig(
        n_actions=4,
        obs_dim=64,
        latent_dim=200,
        planning_horizon=12,
    )
    agent = ModelBasedPlannerAgent(config=config, seed=42)

    # Test metadata
    metadata = agent.metadata()
    assert metadata["name"] == "model_based_planning"
    assert not metadata["privileged"]

    # Test action selection
    obs = np.random.randn(64).astype(np.float32)
    action = agent.start(obs)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 4

    print("✓ Model-based planner agent test passed")


def test_gae_computation() -> None:
    """Test Generalized Advantage Estimation."""
    config = PPOConfig()
    agent = PPOAgent(config=config)

    rewards = np.array([1.0, 2.0, 1.5, 0.5, 3.0], dtype=np.float32)
    values = np.array([2.0, 2.5, 2.0, 1.5, 1.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    advantages, returns = agent._compute_gae(rewards, values, dones)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert np.isfinite(advantages).all()
    assert np.isfinite(returns).all()

    print("✓ GAE computation test passed")


def run_all_tests() -> None:
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print("RUNNING ADVANCED AGENTS UNIT TESTS")
    print("=" * 80 + "\n")

    test_prioritized_replay_buffer()
    test_rainbow_dqn_agent()
    test_ppo_agent()
    test_sac_agent()
    test_model_based_planner()
    test_gae_computation()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
