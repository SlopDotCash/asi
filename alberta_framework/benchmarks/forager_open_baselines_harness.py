"""Forager open baselines harness: run and evaluate baseline agents.

This module provides the measurement harness for running baseline agents on
Forager tasks and collecting performance metrics.

Usage:
    python -m alberta_framework.benchmarks.forager_open_baselines run \\
        --baseline dqn --task-id 0 --num-episodes 100 --seed 0
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from alberta_framework.benchmarks.forager_open_baselines import make_baseline

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EpisodeResult:
    """Results for one episode."""
    episode: int
    steps: int
    return_: float
    success: bool


@dataclasses.dataclass
class MeasurementResult:
    """Results for a complete baseline measurement."""
    baseline: str
    task_id: int
    seed: int
    num_episodes: int
    episodes: list[EpisodeResult]

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics."""
        returns = [e.return_ for e in self.episodes]
        steps = [e.steps for e in self.episodes]
        successes = [e.success for e in self.episodes]

        return {
            "baseline": self.baseline,
            "task_id": self.task_id,
            "seed": self.seed,
            "num_episodes": self.num_episodes,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "mean_steps": float(np.mean(steps)),
            "success_rate": float(np.mean(successes)),
            "episodes": [dataclasses.asdict(e) for e in self.episodes],
        }


def run_baseline_on_task(
    baseline: str,
    task_id: int,
    num_episodes: int = 100,
    seed: int = 0,
    max_steps: int = 1000,
    render: bool = False,
) -> MeasurementResult:
    """Run a baseline on a Forager task and collect results.

    Args:
        baseline: Baseline name ('dqn', 'a3c', 'horde', 'random')
        task_id: Forager task ID (0-indexed)
        num_episodes: Number of episodes to run
        seed: Random seed
        max_steps: Maximum steps per episode
        render: Whether to render episodes (not implemented)

    Returns:
        MeasurementResult with episode-by-episode data and summary statistics
    """
    np.random.seed(seed)

    # Create environment (stub - would load actual Forager task)
    # For now: simple gridworld simulation
    state_dim = 16  # Placeholder
    action_dim = 4

    # Create and initialize agent
    agent = make_baseline(baseline, action_dim=action_dim, state_dim=state_dim)
    import jax
    agent.init(jax.random.PRNGKey(seed), state_dim=state_dim)

    results = []

    for episode_id in range(num_episodes):
        # Reset environment
        state = np.zeros(state_dim, dtype=np.float32)
        episode_return = 0.0
        success = False
        step_count = 0

        # Episode rollout
        for step in range(max_steps):
            # Agent action
            import jax.numpy as jnp
            action = agent.act(jnp.asarray(state), training=True)

            # Environment step (stub - random rewards)
            reward = np.random.normal(0, 0.1)
            done = np.random.rand() < 0.05  # 5% chance of episode end
            next_state = state + np.random.normal(0, 0.01, state_dim).astype(np.float32)

            episode_return += reward * (0.99 ** step)
            step_count += 1

            # Agent update (if applicable)
            transition = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
            agent.update(transition)

            state = next_state

            if done:
                success = True
                break

        results.append(EpisodeResult(
            episode=episode_id,
            steps=step_count,
            return_=float(episode_return),
            success=success,
        ))

        if (episode_id + 1) % 10 == 0:
            recent_returns = [r.return_ for r in results[-10:]]
            logger.info(
                f"Baseline {baseline} task {task_id}: episode {episode_id+1}/{num_episodes}, "
                f"recent avg return={np.mean(recent_returns):.4f}"
            )

    return MeasurementResult(
        baseline=baseline,
        task_id=task_id,
        seed=seed,
        num_episodes=num_episodes,
        episodes=results,
    )


def run_baseline_continual(
    baseline: str,
    num_tasks: int = 10,
    num_episodes_per_task: int = 100,
    seed: int = 0,
) -> list[MeasurementResult]:
    """Run a baseline on a sequence of tasks (continual learning).

    Args:
        baseline: Baseline name
        num_tasks: Number of consecutive tasks
        num_episodes_per_task: Episodes per task
        seed: Random seed

    Returns:
        List of MeasurementResult, one per task
    """
    results = []

    for task_id in range(num_tasks):
        logger.info(f"Running continual task {task_id+1}/{num_tasks}")
        result = run_baseline_on_task(
            baseline=baseline,
            task_id=task_id,
            num_episodes=num_episodes_per_task,
            seed=seed + task_id,
        )
        results.append(result)

    return results


def save_results(results: list[MeasurementResult] | MeasurementResult, output_path: Path) -> None:
    """Save measurement results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, MeasurementResult):
        results = [results]

    data = {
        "campaign": "forager_open_baselines",
        "num_measurements": len(results),
        "measurements": [r.summary() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Forager open baselines")
    parser.add_argument("--baseline", choices=["dqn", "a3c", "horde", "random"], default="dqn")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continual", action="store_true", help="Run continual task sequence")
    parser.add_argument("--output", type=Path, default=Path("outputs/forager_open_baselines/result.json"))

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.continual:
        results = run_baseline_continual(
            baseline=args.baseline,
            num_tasks=args.num_tasks,
            num_episodes_per_task=args.num_episodes,
            seed=args.seed,
        )
    else:
        result = run_baseline_on_task(
            baseline=args.baseline,
            task_id=args.task_id,
            num_episodes=args.num_episodes,
            seed=args.seed,
        )
        results = [result]

    save_results(results, args.output)
