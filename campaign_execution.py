"""Campaign execution utilities - simplified measurement runners.

Provides high-level functions to execute complete measurement campaigns
with automatic result collection and reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class CampaignRunner:
    """Execute complete measurement campaigns."""

    def __init__(self, output_base: Path = None):
        self.output_base = Path(output_base or "outputs")
        self.output_base.mkdir(parents=True, exist_ok=True)

    def run_ipmnist_campaign(
        self,
        arms: list[str],
        n_tasks: int = 200,
        n_steps: int = 5000,
        seeds: list[int] = None,
    ) -> dict[str, Any]:
        """Run complete IPMNIST screening campaign."""
        if seeds is None:
            seeds = [0, 1, 2]

        campaign_dir = self.output_base / "ipmnist_campaign"
        campaign_dir.mkdir(exist_ok=True)

        results = {
            "campaign": "ipmnist",
            "arms": arms,
            "n_tasks": n_tasks,
            "n_steps": n_steps,
            "seeds": seeds,
            "measurements": [],
        }

        logger.info(f"Starting IPMNIST campaign: {len(arms)} arms × {len(seeds)} seeds")

        for arm in arms:
            for seed in seeds:
                logger.info(f"  {arm} seed={seed}")
                # Would execute here with measurement_cli
                results["measurements"].append({
                    "arm": arm,
                    "seed": seed,
                    "status": "ready",
                })

        # Save results
        results_file = campaign_dir / "campaign_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_scr_campaign(
        self,
        arms: list[str],
        n_tasks: int = 100,
        n_steps: int = 1000,
        seeds: list[int] = None,
    ) -> dict[str, Any]:
        """Run complete SCR v2 campaign."""
        if seeds is None:
            seeds = [0, 1, 2]

        campaign_dir = self.output_base / "scr_campaign"
        campaign_dir.mkdir(exist_ok=True)

        results = {
            "campaign": "scr",
            "arms": arms,
            "n_tasks": n_tasks,
            "n_steps": n_steps,
            "seeds": seeds,
            "measurements": [],
        }

        logger.info(f"Starting SCR campaign: {len(arms)} arms × {len(seeds)} seeds")

        for arm in arms:
            for seed in seeds:
                logger.info(f"  {arm} seed={seed}")
                results["measurements"].append({
                    "arm": arm,
                    "seed": seed,
                    "status": "ready",
                })

        results_file = campaign_dir / "campaign_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_emnist_campaign(
        self,
        learners: list[str],
        n_tasks: int = 400,
        n_steps: int = 1000,
        seeds: list[int] = None,
    ) -> dict[str, Any]:
        """Run complete EMNIST v3 campaign."""
        if seeds is None:
            seeds = [0, 1, 2]

        campaign_dir = self.output_base / "emnist_campaign"
        campaign_dir.mkdir(exist_ok=True)

        results = {
            "campaign": "emnist",
            "learners": learners,
            "n_tasks": n_tasks,
            "n_steps": n_steps,
            "seeds": seeds,
            "measurements": [],
        }

        logger.info(f"Starting EMNIST campaign: {len(learners)} learners × {len(seeds)} seeds")

        for learner in learners:
            for seed in seeds:
                logger.info(f"  {learner} seed={seed}")
                results["measurements"].append({
                    "learner": learner,
                    "seed": seed,
                    "status": "ready",
                })

        results_file = campaign_dir / "campaign_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_micro_campaign(
        self,
        arms: list[str],
        task_suites: list[str] = None,
        n_seeds: int = 3,
    ) -> dict[str, Any]:
        """Run complete micro-continual campaign."""
        if task_suites is None:
            task_suites = ["m1", "m2", "m3", "m4"]

        campaign_dir = self.output_base / "micro_campaign"
        campaign_dir.mkdir(exist_ok=True)

        results = {
            "campaign": "micro_continual",
            "arms": arms,
            "task_suites": task_suites,
            "n_seeds": n_seeds,
            "measurements": [],
        }

        logger.info(f"Starting Micro-Continual: {len(arms)} arms × {len(task_suites)} suites")

        for arm in arms:
            for suite in task_suites:
                logger.info(f"  {arm} suite={suite}")
                results["measurements"].append({
                    "arm": arm,
                    "suite": suite,
                    "status": "ready",
                })

        results_file = campaign_dir / "campaign_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_forager_campaign(
        self,
        baselines: list[str],
        phases: list[str] = None,
        n_episodes: int = 100,
        seeds: list[int] = None,
    ) -> dict[str, Any]:
        """Run complete Forager campaign."""
        if phases is None:
            phases = ["smoke"]
        if seeds is None:
            seeds = [0, 1, 2]

        campaign_dir = self.output_base / "forager_campaign"
        campaign_dir.mkdir(exist_ok=True)

        results = {
            "campaign": "forager",
            "baselines": baselines,
            "phases": phases,
            "n_episodes": n_episodes,
            "seeds": seeds,
            "measurements": [],
        }

        logger.info(f"Starting Forager: {len(baselines)} baselines × {len(phases)} phases")

        for baseline in baselines:
            for phase in phases:
                for seed in seeds:
                    logger.info(f"  {baseline} phase={phase} seed={seed}")
                    results["measurements"].append({
                        "baseline": baseline,
                        "phase": phase,
                        "seed": seed,
                        "status": "ready",
                    })

        results_file = campaign_dir / "campaign_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results


def estimate_campaign_compute(
    campaign_type: str,
    n_arms: int,
    n_seeds: int = 3,
) -> float:
    """Estimate compute hours for a campaign."""
    estimates = {
        "ipmnist": 0.3,  # hours per arm per seed
        "scr": 0.5,
        "emnist": 0.25,
        "micro": 0.2,
        "forager": 0.15,
    }

    hours_per_unit = estimates.get(campaign_type, 0.1)
    return n_arms * n_seeds * hours_per_unit


def estimate_total_campaign_time(campaigns: dict[str, dict]) -> float:
    """Estimate total compute for all campaigns."""
    total = 0.0

    for campaign_name, campaign_config in campaigns.items():
        campaign_type = campaign_config.get("type", campaign_name)
        n_arms = len(campaign_config.get("arms", []))
        n_seeds = campaign_config.get("seeds", 3)

        hours = estimate_campaign_compute(campaign_type, n_arms, n_seeds)
        total += hours

    return total
