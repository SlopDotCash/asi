"""CLI tool for running and benchmarking advanced Forager agents.

Usage:
    python -m alberta_framework.core.advanced_agents_cli benchmark --agent rainbow --seeds 5
    python -m alberta_framework.core.advanced_agents_cli compare --output results.json
    python -m alberta_framework.core.advanced_agents_cli test --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

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


AGENTS = {
    "rainbow": (RainbowDQNAgent, RainbowDQNConfig),
    "ppo": (PPOAgent, PPOConfig),
    "sac": (SACAgent, SACConfig),
    "planning": (ModelBasedPlannerAgent, PlaNetConfig),
}

AGENT_DESCRIPTIONS = {
    "rainbow": "Rainbow DQN with all 6 improvements",
    "ppo": "PPO with Generalized Advantage Estimation",
    "sac": "Soft Actor-Critic",
    "planning": "Model-Based Planning with World Model",
}


def create_dummy_env():
    """Create a simple dummy environment for testing."""

    class DummyEnv:
        def __init__(self):
            self.obs_dim = 84
            self.action_space_n = 4

        def reset(self):
            return np.random.randn(self.obs_dim).astype(np.float32)

        def step(self, action):
            obs = np.random.randn(self.obs_dim).astype(np.float32)
            reward = float(np.random.randn())
            done = bool(np.random.random() < 0.05)
            info = {}
            return obs, reward, done, info

    return DummyEnv()


def run_agent_episode(agent, env, max_steps: int = 1000):
    """Run a single episode with an agent."""
    obs = env.reset()
    action = agent.start(obs)
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        action = agent.step(reward, obs)
        steps += 1

        if done:
            break

    return {
        "total_reward": float(total_reward),
        "steps": steps,
        "avg_reward": float(total_reward / max(steps, 1)),
    }


def benchmark_agent(
    agent_name: str,
    n_seeds: int = 3,
    max_steps: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Benchmark a single agent."""
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")

    agent_class, config_class = AGENTS[agent_name]
    config = config_class()
    env = create_dummy_env()

    results = []
    if verbose:
        print(f"\nBenchmarking {agent_name} ({AGENT_DESCRIPTIONS[agent_name]})...")

    for seed in range(n_seeds):
        agent = agent_class(config=config, seed=seed)
        episode_result = run_agent_episode(agent, env, max_steps=max_steps)
        results.append(episode_result)

        if verbose:
            print(
                f"  Seed {seed}: reward={episode_result['total_reward']:.2f}, "
                f"steps={episode_result['steps']}"
            )

    # Aggregate
    rewards = np.array([r["total_reward"] for r in results])
    steps = np.array([r["steps"] for r in results])

    summary = {
        "agent": agent_name,
        "description": AGENT_DESCRIPTIONS[agent_name],
        "n_seeds": n_seeds,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "steps_mean": float(steps.mean()),
        "steps_std": float(steps.std()),
        "episodes": results,
    }

    if verbose:
        print(
            f"  Summary: {summary['reward_mean']:.2f} ± {summary['reward_std']:.2f} "
            f"(range: [{summary['reward_min']:.2f}, {summary['reward_max']:.2f}])"
        )

    return summary


def cmd_test(args: argparse.Namespace) -> int:
    """Run unit tests."""
    print("=" * 80)
    print("RUNNING UNIT TESTS")
    print("=" * 80)

    try:
        from alberta_framework.core.advanced_forager_agents_test import run_all_tests

        run_all_tests()
        return 0
    except Exception as e:
        print(f"Test failed: {e}")
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark a single agent."""
    try:
        result = benchmark_agent(
            args.agent,
            n_seeds=args.seeds,
            max_steps=args.steps,
            verbose=True,
        )

        if args.output:
            output_file = Path(args.output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {output_file}")

        return 0
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare all agents."""
    print("=" * 80)
    print("COMPARING ALL AGENTS")
    print("=" * 80)

    results = {
        "agents": {},
        "summary": {},
    }

    agent_names = args.agents if args.agents else list(AGENTS.keys())

    for agent_name in agent_names:
        try:
            result = benchmark_agent(
                agent_name,
                n_seeds=args.seeds,
                max_steps=args.steps,
                verbose=True,
            )
            results["agents"][agent_name] = result
        except Exception as e:
            print(f"Failed to benchmark {agent_name}: {e}")

    # Generate comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    sorted_agents = sorted(
        results["agents"].items(),
        key=lambda x: x[1]["reward_mean"],
        reverse=True,
    )

    print("\nRanking by mean reward:")
    for rank, (agent_name, result) in enumerate(sorted_agents, 1):
        print(
            f"  {rank}. {agent_name}: "
            f"{result['reward_mean']:.2f} ± {result['reward_std']:.2f}"
        )

    # Summary statistics
    print("\nDetailed comparison:")
    print("-" * 80)
    for agent_name, result in sorted_agents:
        print(f"\n{agent_name}:")
        print(f"  Mean ± Std: {result['reward_mean']:.2f} ± {result['reward_std']:.2f}")
        print(f"  Range: [{result['reward_min']:.2f}, {result['reward_max']:.2f}]")
        print(f"  Avg steps: {result['steps_mean']:.1f} ± {result['steps_std']:.1f}")

    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Display agent information."""
    if args.agent:
        if args.agent not in AGENTS:
            print(f"Unknown agent: {args.agent}")
            return 1

        agent_class, config_class = AGENTS[args.agent]
        config = config_class()

        print(f"\n{args.agent.upper()}")
        print("=" * 80)
        print(f"Description: {AGENT_DESCRIPTIONS[args.agent]}")
        print(f"\nDefault Configuration:")
        for key, value in vars(config).items():
            print(f"  {key}: {value}")

        # Create sample agent to get metadata
        agent = agent_class(config=config, seed=0)
        metadata = agent.metadata()
        print(f"\nMetadata:")
        print(f"  Name: {metadata['name']}")
        print(f"  Privileged: {metadata['privileged']}")

        if "improvements" in metadata:
            print(f"  Improvements: {', '.join(metadata['improvements'])}")
        if "components" in metadata:
            print(f"  Components: {', '.join(metadata['components'])}")
    else:
        print("\nAVAILABLE AGENTS")
        print("=" * 80)
        for agent_name, description in AGENT_DESCRIPTIONS.items():
            print(f"  {agent_name:15} - {description}")
        print("\nUse 'info --agent <name>' for details")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Forager Agents CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test --all
  %(prog)s benchmark --agent rainbow --seeds 5 --steps 100
  %(prog)s compare --agents rainbow ppo sac --output results.json
  %(prog)s info --agent rainbow
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run unit tests")
    test_parser.add_argument("--all", action="store_true", help="Run all tests")
    test_parser.set_defaults(func=cmd_test)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark a single agent")
    bench_parser.add_argument(
        "--agent",
        required=True,
        choices=list(AGENTS.keys()),
        help="Agent to benchmark",
    )
    bench_parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    bench_parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    bench_parser.add_argument("--output", help="Output JSON file")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare all agents")
    compare_parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(AGENTS.keys()),
        help="Agents to compare (default: all)",
    )
    compare_parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    compare_parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    compare_parser.add_argument("--output", help="Output JSON file")
    compare_parser.set_defaults(func=cmd_compare)

    # Info command
    info_parser = subparsers.add_parser("info", help="Display agent information")
    info_parser.add_argument("--agent", choices=list(AGENTS.keys()), help="Agent to describe")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
