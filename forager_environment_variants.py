"""Forager environment difficulty variants for progressive measurement.

Implements environment variants at different difficulty levels.
"""

from typing import Dict, Any
import jax.numpy as jnp


class ForagerEnvironmentVariants:
    """Generate Forager environment variants."""

    @staticmethod
    def create_easy_environment() -> Dict[str, Any]:
        """Easy environment - sparse rewards, simple dynamics."""
        return {
            "name": "easy",
            "reward_density": 0.5,  # 50% of states have rewards
            "max_reward": 1.0,
            "episode_length": 50,
            "action_noise": 0.0,
            "observation_noise": 0.01,
            "dynamics_complexity": "simple",
        }

    @staticmethod
    def create_medium_environment() -> Dict[str, Any]:
        """Medium environment - moderate difficulty."""
        return {
            "name": "medium",
            "reward_density": 0.3,
            "max_reward": 1.0,
            "episode_length": 100,
            "action_noise": 0.05,
            "observation_noise": 0.05,
            "dynamics_complexity": "moderate",
        }

    @staticmethod
    def create_hard_environment() -> Dict[str, Any]:
        """Hard environment - sparse rewards, complex dynamics."""
        return {
            "name": "hard",
            "reward_density": 0.1,
            "max_reward": 1.0,
            "episode_length": 200,
            "action_noise": 0.1,
            "observation_noise": 0.1,
            "dynamics_complexity": "complex",
        }

    @staticmethod
    def create_sparse_environment() -> Dict[str, Any]:
        """Sparse environment - only terminal rewards."""
        return {
            "name": "sparse",
            "reward_density": 0.01,
            "max_reward": 10.0,  # Higher terminal reward
            "episode_length": 300,
            "action_noise": 0.05,
            "observation_noise": 0.05,
            "dynamics_complexity": "complex",
        }

    @staticmethod
    def create_noisy_environment() -> Dict[str, Any]:
        """Noisy environment - high stochasticity."""
        return {
            "name": "noisy",
            "reward_density": 0.2,
            "max_reward": 1.0,
            "episode_length": 100,
            "action_noise": 0.3,
            "observation_noise": 0.3,
            "dynamics_complexity": "moderate",
        }

    @staticmethod
    def create_curriculum_environments() -> Dict[str, list]:
        """Curriculum of environments - easy to hard."""
        return {
            "easy_to_hard": [
                ForagerEnvironmentVariants.create_easy_environment(),
                ForagerEnvironmentVariants.create_medium_environment(),
                ForagerEnvironmentVariants.create_hard_environment(),
            ],
            "exploration_focus": [
                ForagerEnvironmentVariants.create_sparse_environment(),
                ForagerEnvironmentVariants.create_medium_environment(),
            ],
            "robustness": [
                ForagerEnvironmentVariants.create_noisy_environment(),
                ForagerEnvironmentVariants.create_hard_environment(),
            ],
        }


class ForagerTaskVariants:
    """Generate Forager task variants."""

    @staticmethod
    def create_gridworld_task() -> Dict[str, Any]:
        """Gridworld navigation task."""
        return {
            "type": "gridworld",
            "grid_size": 10,
            "n_goals": 3,
            "n_obstacles": 5,
        }

    @staticmethod
    def create_continuous_control_task() -> Dict[str, Any]:
        """Continuous control task."""
        return {
            "type": "continuous",
            "state_dim": 16,
            "action_dim": 4,
            "action_range": [-1, 1],
        }

    @staticmethod
    def create_discrete_action_task() -> Dict[str, Any]:
        """Discrete action task."""
        return {
            "type": "discrete",
            "state_dim": 16,
            "n_actions": 6,
        }

    @staticmethod
    def create_hierarchical_task() -> Dict[str, Any]:
        """Hierarchical task with options."""
        return {
            "type": "hierarchical",
            "n_options": 4,
            "option_length": 5,
            "state_dim": 16,
        }

    @staticmethod
    def create_multi_objective_task() -> Dict[str, Any]:
        """Multi-objective task with competing rewards."""
        return {
            "type": "multi_objective",
            "n_objectives": 3,
            "state_dim": 16,
            "action_dim": 4,
        }


def create_forager_measurement_curriculum() -> Dict[str, Any]:
    """Create complete curriculum for Forager measurements."""
    env_variant = ForagerEnvironmentVariants()
    task_variant = ForagerTaskVariants()

    return {
        "phase_1_smoke": {
            "environments": [env_variant.create_easy_environment()],
            "tasks": [task_variant.create_gridworld_task()],
            "episodes_per_task": 100,
            "n_seeds": 3,
        },
        "phase_2_exploration": {
            "environments": [
                env_variant.create_easy_environment(),
                env_variant.create_medium_environment(),
            ],
            "tasks": [
                task_variant.create_gridworld_task(),
                task_variant.create_discrete_action_task(),
            ],
            "episodes_per_task": 200,
            "n_seeds": 3,
        },
        "phase_3_robustness": {
            "environments": [
                env_variant.create_medium_environment(),
                env_variant.create_hard_environment(),
                env_variant.create_noisy_environment(),
            ],
            "tasks": [
                task_variant.create_continuous_control_task(),
                task_variant.create_hierarchical_task(),
                task_variant.create_multi_objective_task(),
            ],
            "episodes_per_task": 300,
            "n_seeds": 5,
        },
    }


FORAGER_ENVIRONMENT_VARIANTS = {
    "easy": ForagerEnvironmentVariants.create_easy_environment,
    "medium": ForagerEnvironmentVariants.create_medium_environment,
    "hard": ForagerEnvironmentVariants.create_hard_environment,
    "sparse": ForagerEnvironmentVariants.create_sparse_environment,
    "noisy": ForagerEnvironmentVariants.create_noisy_environment,
}

FORAGER_TASK_VARIANTS = {
    "gridworld": ForagerTaskVariants.create_gridworld_task,
    "continuous": ForagerTaskVariants.create_continuous_control_task,
    "discrete": ForagerTaskVariants.create_discrete_action_task,
    "hierarchical": ForagerTaskVariants.create_hierarchical_task,
    "multi_objective": ForagerTaskVariants.create_multi_objective_task,
}


def register_forager_variants():
    """Register all Forager environment and task variants."""
    print(f"[OK] Registered {len(FORAGER_ENVIRONMENT_VARIANTS)} environment + {len(FORAGER_TASK_VARIANTS)} task variants")
    return {
        "environments": FORAGER_ENVIRONMENT_VARIANTS,
        "tasks": FORAGER_TASK_VARIANTS,
    }
