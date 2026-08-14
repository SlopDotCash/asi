"""Comprehensive measurement configuration generator - complete pipeline setup.

Generates end-to-end measurement configurations for all campaigns.
Integrates with unified arm registry for consistency and validation.
"""

from typing import Dict, List, Any
import json
from unified_arm_registry import (
    create_unified_registry,
    UnifiedArmRegistry,
)


class ComprehensiveMeasurementConfigGenerator:
    """Generate complete measurement configurations using unified registry."""

    def __init__(self):
        """Initialize with unified registry."""
        self.registry = create_unified_registry()

    def generate_ipmnist_config(self) -> Dict[str, Any]:
        """Generate IPMNIST measurement configuration from registry."""
        arms = self.registry.get_arms_by_campaign("ipmnist")
        total_hours = sum(arm.estimated_hours for arm in arms)

        return {
            "domain": "ipmnist",
            "n_arms": len(arms),
            "arms": [arm.name for arm in arms],
            "arm_details": [
                {
                    "name": arm.name,
                    "description": arm.description,
                    "tags": arm.tags,
                    "estimated_hours": arm.estimated_hours,
                }
                for arm in arms
            ],
            "n_tasks": 200,
            "n_steps": 5000,
            "n_seeds": 3,
            "estimated_hours": total_hours,
        }

    def generate_scr_config(self) -> Dict[str, Any]:
        """Generate SCR v2 measurement configuration from registry."""
        arms = self.registry.get_arms_by_campaign("scr")
        total_hours = sum(arm.estimated_hours for arm in arms)

        return {
            "domain": "scr",
            "n_arms": len(arms),
            "arms": [arm.name for arm in arms],
            "arm_details": [
                {
                    "name": arm.name,
                    "description": arm.description,
                    "tags": arm.tags,
                    "estimated_hours": arm.estimated_hours,
                }
                for arm in arms
            ],
            "n_tasks": 100,
            "n_steps": 1000,
            "n_seeds": 3,
            "estimated_hours": total_hours,
        }

    def generate_emnist_config(self) -> Dict[str, Any]:
        """Generate EMNIST v3 measurement configuration from registry."""
        learners = self.registry.get_arms_by_campaign("emnist")
        total_hours = sum(arm.estimated_hours for arm in learners)

        return {
            "domain": "emnist",
            "n_learners": len(learners),
            "learners": [arm.name for arm in learners],
            "learner_details": [
                {
                    "name": arm.name,
                    "description": arm.description,
                    "tags": arm.tags,
                    "estimated_hours": arm.estimated_hours,
                }
                for arm in learners
            ],
            "n_tasks": 400,
            "n_steps": 1000,
            "n_seeds": 3,
            "estimated_hours": total_hours,
        }

    def generate_micro_config(self) -> Dict[str, Any]:
        """Generate micro-continual measurement configuration from registry."""
        arms = self.registry.get_arms_by_campaign("micro_continual")
        total_hours = sum(arm.estimated_hours for arm in arms)

        return {
            "domain": "micro_continual",
            "n_arms": len(arms),
            "arms": [arm.name for arm in arms],
            "arm_details": [
                {
                    "name": arm.name,
                    "description": arm.description,
                    "tags": arm.tags,
                    "estimated_hours": arm.estimated_hours,
                }
                for arm in arms
            ],
            "task_suites": ["m1", "m2", "m3", "m4"],
            "n_seeds": 3,
            "estimated_hours": total_hours,
        }

    def generate_forager_config(self) -> Dict[str, Any]:
        """Generate Forager measurement configuration from registry."""
        baselines = self.registry.get_arms_by_campaign("forager")
        total_hours = sum(arm.estimated_hours for arm in baselines)

        return {
            "domain": "forager",
            "n_baselines": len(baselines),
            "baselines": [arm.name for arm in baselines],
            "baseline_details": [
                {
                    "name": arm.name,
                    "description": arm.description,
                    "tags": arm.tags,
                    "estimated_hours": arm.estimated_hours,
                }
                for arm in baselines
            ],
            "phases": ["smoke", "continual", "transfer"],
            "n_episodes": 100,
            "n_seeds": 3,
            "environments": ["easy", "medium", "hard", "sparse", "noisy"],
            "tasks": [
                "gridworld",
                "continuous",
                "discrete",
                "hierarchical",
                "multi_objective",
            ],
            "estimated_hours": total_hours,
        }

    def generate_rule_discovery_config(self) -> Dict[str, Any]:
        """Generate Rule Discovery V2 measurement configuration from registry."""
        genomes = self.registry.get_arms_by_campaign("rule_discovery")
        total_hours = sum(arm.estimated_hours for arm in genomes)

        return {
            "domain": "rule_discovery",
            "n_genomes": sum(arm.parameters.get("n_genomes", 0) for arm in genomes),
            "phases": {
                "phase_1a": {
                    "name": "Candidate Generation",
                    "n_genomes": 30,
                    "estimated_hours": 30,
                },
                "phase_1b": {
                    "name": "Ablation Studies",
                    "n_genomes": 30,
                    "estimated_hours": 20,
                },
                "phase_1c": {
                    "name": "Genetic Search",
                    "n_genomes": 50,
                    "estimated_hours": 40,
                },
                "phase_1d": {
                    "name": "Fine-tuning",
                    "n_genomes": 20,
                    "estimated_hours": 30,
                },
            },
            "estimated_hours": total_hours,
        }

    def generate_complete_manifest(self) -> Dict[str, Any]:
        """Generate complete measurement manifest for all campaigns."""
        return {
            "version": "2.0",
            "timestamp": "2026-08-15",
            "registry_version": self.registry.version,
            "campaigns": {
                "ipmnist": self.generate_ipmnist_config(),
                "scr": self.generate_scr_config(),
                "emnist": self.generate_emnist_config(),
                "micro_continual": self.generate_micro_config(),
                "forager": self.generate_forager_config(),
                "rule_discovery": self.generate_rule_discovery_config(),
            },
            "summary": {
                "total_campaigns": 6,
                "total_arms": self.registry.get_summary()["total_arms"],
                "total_estimated_hours": self.registry.get_summary()[
                    "total_estimated_hours"
                ],
                "measurement_status": "READY FOR EXECUTION",
            },
        }

    @staticmethod
    def export_manifest(manifest: Dict[str, Any], output_path: str) -> None:
        """Export measurement manifest to JSON file."""
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)


def generate_and_export_complete_measurement_manifest(
    output_dir: str = "configs",
) -> str:
    """Generate and export complete measurement manifest."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    generator = ComprehensiveMeasurementConfigGenerator()
    manifest = generator.generate_complete_manifest()

    output_path = os.path.join(output_dir, "measurement_manifest.json")
    generator.export_manifest(manifest, output_path)

    print(f"[OK] Exported measurement manifest to {output_path}")
    print(f"Total campaigns: {manifest['summary']['total_campaigns']}")
    print(f"Total arms: {manifest['summary']['total_arms']}")
    print(f"Total estimated hours: {manifest['summary']['total_estimated_hours']}")

    return output_path


if __name__ == "__main__":
    generate_and_export_complete_measurement_manifest()
