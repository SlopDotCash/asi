"""Comprehensive measurement configuration generator - complete pipeline setup.

Generates end-to-end measurement configurations for all campaigns.
"""

from typing import Dict, List, Any
import json


class ComprehensiveMeasurementConfigGenerator:
    """Generate complete measurement configurations."""

    @staticmethod
    def generate_ipmnist_config() -> Dict[str, Any]:
        """Generate IPMNIST measurement configuration."""
        return {
            "domain": "ipmnist",
            "arms": [
                "upgd_w_control", "adamw_control", "upgd_ema_norm",
                "upgd_w_norm_decay_09", "upgd_w_norm_decay_095", "upgd_w_norm_decay_999",
                "upgd_w_step_size_002", "upgd_w_step_size_005", "upgd_w_step_size_004",
                "upgd_w_weight_decay_01", "upgd_w_weight_decay_005", "upgd_w_weight_decay_0",
                "upgd_w_aggressive_combo", "upgd_w_conservative_combo", "upgd_w_balanced_combo",
                "upgd_w_ema_smoothing", "upgd_w_second_order_momentum", "upgd_w_adaptive_schedule",
                "upgd_w_gradient_clipping", "upgd_w_lookahead",
            ],
            "n_tasks": 200,
            "n_steps": 5000,
            "n_seeds": 3,
            "estimated_hours": 3.5,
        }

    @staticmethod
    def generate_scr_config() -> Dict[str, Any]:
        """Generate SCR v2 measurement configuration."""
        return {
            "domain": "scr",
            "arms": [
                "backprop_sgd_relu", "adamw_baseline", "upgd_w_baseline",
                "upgd_w_scr_step_002", "upgd_w_scr_step_005", "upgd_w_scr_step_004",
                "upgd_w_scr_wd_01", "upgd_w_scr_wd_005", "upgd_w_scr_wd_0",
                "upgd_w_scr_norm_09", "upgd_w_scr_norm_095", "upgd_w_scr_norm_999",
                "lion_optimizer", "adamw_warmup", "muon_optimizer", "normalized_sgd",
            ],
            "n_tasks": 100,
            "n_steps": 1000,
            "n_seeds": 3,
            "estimated_hours": 18,
        }

    @staticmethod
    def generate_emnist_config() -> Dict[str, Any]:
        """Generate EMNIST v3 measurement configuration."""
        return {
            "domain": "emnist",
            "learners": [
                "upgd_w", "adamw", "upgd_ema_norm", "sgd_ema_norm",
                "upgd_ema_norm_cbp", "sgd_norm_cbp", "upgd_l2init", "upgd_shiftnorm",
                "upgd_cbp_recycle_high", "upgd_cbp_recycle_mid", "upgd_cbp_recycle_low",
                "upgd_l2init_strong", "upgd_l2init_moderate", "upgd_l2init_weak",
                "upgd_shiftnorm_aggressive", "upgd_shiftnorm_balanced", "upgd_shiftnorm_conservative",
                "adamw_cbp", "adamw_l2init", "sgd_cbp", "sgd_shiftnorm",
                "mixup_augmented", "cutout_augmented", "randaugment", "adversarial_robust", "ensemble_augmented",
            ],
            "n_tasks": 400,
            "n_steps": 1000,
            "n_seeds": 3,
            "estimated_hours": 12,
        }

    @staticmethod
    def generate_micro_config() -> Dict[str, Any]:
        """Generate micro-continual measurement configuration."""
        return {
            "domain": "micro_continual",
            "arms": [
                "rls_head_resid", "alignment_first", "naive_bayes_extended",
                "dual_speed_rfs_rls", "actor_critic_micro",
                "replay_buffer_learner", "plasticity_modulated", "task_boundary_detector",
                "maml_inspired", "hypernetwork", "context_modulation", "episodic_memory",
                "loss_gated", "gradient_norm_gated", "variance_gated", "confidence_gated",
            ],
            "task_suites": ["m1", "m2", "m3", "m4"],
            "n_seeds": 3,
            "estimated_hours": 10,
        }

    @staticmethod
    def generate_forager_config() -> Dict[str, Any]:
        """Generate Forager measurement configuration."""
        return {
            "domain": "forager",
            "baselines": [
                "dqn", "a3c", "horde", "random",
                "dqn_smoke_opt", "dqn_continual_opt", "dqn_transfer_opt",
                "a3c_smoke_opt", "a3c_continual_opt", "a3c_transfer_opt",
                "dqn_curiosity", "a3c_entropy_reg", "dqn_a3c_ensemble",
            ],
            "phases": ["smoke", "continual", "transfer"],
            "n_episodes": 100,
            "n_seeds": 3,
            "environments": ["easy", "medium", "hard", "sparse", "noisy"],
            "tasks": ["gridworld", "continuous", "discrete", "hierarchical", "multi_objective"],
            "estimated_hours": 18,
        }

    @staticmethod
    def generate_rule_discovery_config() -> Dict[str, Any]:
        """Generate Rule Discovery V2 measurement configuration."""
        return {
            "domain": "rule_discovery",
            "phases": {
                "phase_1a": {
                    "name": "Candidate Generation",
                    "n_candidates": 50,
                    "estimated_hours": 30,
                },
                "phase_1b": {
                    "name": "Ablation Studies",
                    "n_ablations": 30,
                    "estimated_hours": 20,
                },
                "phase_1c": {
                    "name": "Genetic Search",
                    "n_generations": 5,
                    "estimated_hours": 40,
                },
                "phase_1d": {
                    "name": "Refinement",
                    "n_refinements": 20,
                    "estimated_hours": 30,
                },
            },
            "total_candidates": 150,
            "estimated_hours": 120,
        }

    @staticmethod
    def generate_complete_manifest() -> Dict[str, Any]:
        """Generate complete measurement manifest for all campaigns."""
        generator = ComprehensiveMeasurementConfigGenerator()

        return {
            "version": "1.0",
            "timestamp": "2024",
            "campaigns": {
                "ipmnist": generator.generate_ipmnist_config(),
                "scr": generator.generate_scr_config(),
                "emnist": generator.generate_emnist_config(),
                "micro_continual": generator.generate_micro_config(),
                "forager": generator.generate_forager_config(),
                "rule_discovery": generator.generate_rule_discovery_config(),
            },
            "summary": {
                "total_campaigns": 6,
                "total_arms": 100,
                "total_estimated_hours": 181.5,
            },
        }

    @staticmethod
    def export_manifest(manifest: Dict[str, Any], output_path: str) -> None:
        """Export measurement manifest to JSON file."""
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)


def generate_and_export_complete_measurement_manifest(output_dir: str = "configs") -> str:
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
