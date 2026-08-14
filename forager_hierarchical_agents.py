"""Advanced Forager hierarchical agents - multi-level control.

Implements hierarchical agents for improved Forager performance.
"""

from typing import Callable, Dict, Any


def make_hierarchical_options_agent(hp: Dict[str, float]) -> Dict[str, Any]:
    """Hierarchical agent with learned options."""
    return {
        "name": "hierarchical_options",
        "type": "options_framework",
        "config": {
            "n_options": int(hp.get("n_options", 4)),
            "option_horizon": int(hp.get("option_horizon", 5)),
            "termination_prob": hp.get("termination_prob", 0.5),
            "base_agent": "dqn",
        },
        "description": "Options-framework with learned temporal abstractions"
    }


def make_goal_conditioned_agent(hp: Dict[str, float]) -> Dict[str, Any]:
    """Goal-conditioned hierarchical agent."""
    return {
        "name": "goal_conditioned",
        "type": "goal_conditioned",
        "config": {
            "goal_space_dim": int(hp.get("goal_dim", 8)),
            "hindsight_ratio": hp.get("hindsight_ratio", 0.8),
            "goal_conditioning": True,
        },
        "description": "Goal-conditioned HER with multi-level policies"
    }


def make_feudal_agent(hp: Dict[str, float]) -> Dict[str, Any]:
    """Feudal networks for hierarchical control."""
    return {
        "name": "feudal",
        "type": "feudal",
        "config": {
            "manager_horizon": int(hp.get("manager_horizon", 10)),
            "worker_horizon": int(hp.get("worker_horizon", 5)),
            "manager_action_dim": int(hp.get("manager_actions", 4)),
        },
        "description": "Feudal manager-worker hierarchy with goal communication"
    }


def make_skill_learning_agent(hp: Dict[str, float]) -> Dict[str, Any]:
    """Agent with unsupervised skill discovery."""
    return {
        "name": "skill_learning",
        "type": "skill_discovery",
        "config": {
            "n_skills": int(hp.get("n_skills", 8)),
            "skill_horizon": int(hp.get("skill_horizon", 5)),
            "discriminator_weight": hp.get("disc_weight", 1.0),
        },
        "description": "Unsupervised skill discovery with discriminator"
    }


def make_multi_task_agent(hp: Dict[str, float]) -> Dict[str, Any]:
    """Multi-task learning agent with shared representations."""
    return {
        "name": "multi_task",
        "type": "multi_task",
        "config": {
            "n_tasks": int(hp.get("n_tasks", 5)),
            "task_embedding_dim": int(hp.get("task_emb_dim", 16)),
            "shared_layers": int(hp.get("shared_layers", 3)),
        },
        "description": "Multi-task learning with task-conditioned policies"
    }


def make_meta_learning_agent(hp: Dict[str, float]) -> Dict[str, Any]:
    """Meta-learning agent for fast adaptation."""
    return {
        "name": "meta_learning",
        "type": "meta_learning",
        "config": {
            "inner_steps": int(hp.get("inner_steps", 3)),
            "inner_lr": hp.get("inner_lr", 0.01),
            "outer_lr": hp.get("outer_lr", 0.001),
        },
        "description": "MAML-style meta-learning for rapid task adaptation"
    }


HIERARCHICAL_AGENTS = {
    "hierarchical_options": make_hierarchical_options_agent,
    "goal_conditioned": make_goal_conditioned_agent,
    "feudal": make_feudal_agent,
    "skill_learning": make_skill_learning_agent,
    "multi_task": make_multi_task_agent,
    "meta_learning": make_meta_learning_agent,
}


def register_hierarchical_agents():
    """Register hierarchical Forager agents."""
    print(f"[OK] Registered {len(HIERARCHICAL_AGENTS)} hierarchical agents")
    return HIERARCHICAL_AGENTS
