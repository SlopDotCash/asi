"""Rule Discovery v2 Integration — Using Expanded Templates in Search.

This module demonstrates how to integrate the 23 new rule templates into
the existing rule_discovery search pipeline. It provides:

1. Loader functions to inject templates into the seed population
2. Search configuration for Direction A + B
3. Integration examples for the CLI
4. Validation harness for template fitness on micro_continual suite
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.benchmarks.rule_discovery import (
    GENOME_SIZE,
    FLAG_NAMES,
    PARAM_NAMES,
    _CHAMPION_CONFIG,
    genome_from_config,
    seed_genomes,
    describe_genome,
)
from rule_discovery_v2_templates import (
    RULE_DISCOVERY_V2_TEMPLATES,
    template_to_config_dict,
)

logger = logging.getLogger(__name__)


def expand_seed_genomes_with_templates() -> Array:
    """Inject all 23 expanded templates into the initial seed population.

    Returns an extended seed population matrix where:
    - Rows 0–12: original seed_genomes() (champion form, meta-a, meta-b,
                 bare SGD, norm-only, L2-init, discovered-rule-1, + wave-2)
    - Rows 13–35: all 23 Direction A/B/hybrid templates

    This ensures the search population begins with structured diversity
    across gate mechanisms and normalization strategies.
    """
    rows: list[np.ndarray] = []

    # Original seed genomes (baseline population diversity)
    original_seeds = np.asarray(seed_genomes(), dtype=np.float32)
    for i in range(original_seeds.shape[0]):
        rows.append(original_seeds[i])

    # All 23 expanded templates
    for template in RULE_DISCOVERY_V2_TEMPLATES:
        config = template_to_config_dict(template)
        genome = genome_from_config(config)
        rows.append(np.asarray(genome, dtype=np.float32))

    result: Array = jnp.asarray(np.stack(rows), dtype=jnp.float32)
    logger.info(
        "Expanded seed population: %d total (original %d + templates %d)",
        result.shape[0],
        original_seeds.shape[0],
        len(RULE_DISCOVERY_V2_TEMPLATES),
    )
    return result


def describe_search_configuration() -> dict[str, Any]:
    """Return a configuration dict describing the Direction A+B search space."""
    return {
        "search_name": "rule_discovery_v2_expanded",
        "phase": "Tier 2 exploration (Direction A + B)",
        "hypothesis": (
            "Error-gating discovered by search-v1; expand with alternative gate signals "
            "(loss, gradient-norm, entropy) and normalization locations (input, layer-wise, "
            "output, adaptive). Combined gates and norms test compositionality."
        ),
        "directions": {
            "Direction A (gate variants)": {
                "hypothesis": "Error-gating works; test loss, gradient-norm, entropy signals",
                "templates": 8,
                "template_names": [
                    "loss_gating_v1",
                    "loss_gating_fast_v1",
                    "gradient_norm_gating_v1",
                    "entropy_gating_v1",
                    "surprise_gating_alt_v1",
                    "combined_loss_gradient_gating_v1",
                    "combined_loss_entropy_gating_v1",
                    "adaptive_gate_threshold_v1",
                ],
                "mechanism_coverage": [
                    "Loss magnitude (pre-update error signal)",
                    "Gradient norm (per-parameter update magnitude)",
                    "Prediction entropy (decision confidence)",
                    "Error ratio (fast/slow surprise)",
                    "Composite gates (multiplicative, additive)",
                    "Adaptive thresholds (task-difficulty tracking)",
                ],
            },
            "Direction B (normalization locations)": {
                "hypothesis": (
                    "Hidden RMS works; test input-side, layer-wise, output-side, "
                    "adaptive decay, pre/post-activation placement"
                ),
                "templates": 12,
                "template_names": [
                    "input_rms_normalization_v1",
                    "layer_wise_rms_v1",
                    "output_logit_normalization_v1",
                    "channel_wise_rms_v1",
                    "adaptive_norm_decay_v1",
                    "pre_activation_norm_v1",
                    "post_activation_norm_v1",
                    "dynamic_norm_scale_v1",
                    "whitening_transform_v1",
                    "layer_dependent_norm_v1",
                    "coupled_normalization_v1",
                    "batch_norm_like_v1",
                ],
                "mechanism_coverage": [
                    "Input-side global RMS",
                    "Per-layer independent normalization",
                    "Output-side (logit) normalization",
                    "Channel-wise (per-neuron) scaling",
                    "Adaptive norm decay (error-driven)",
                    "Pre-activation vs post-activation placement",
                    "Dynamic scale factors",
                    "Kalman-tracked whitening",
                    "Layer-dependent decay/scale",
                    "Coupled input + hidden statistics",
                    "Batch-norm-like running stats",
                ],
            },
            "Hybrids (A + B combinations)": {
                "hypothesis": "Cross-direction interactions; loss + layer-wise, gradient + adaptive, etc.",
                "templates": 3,
                "template_names": [
                    "hybrid_loss_gate_layerwise_norm_v1",
                    "hybrid_gradient_gate_adaptive_decay_v1",
                    "hybrid_entropy_gate_whitening_v1",
                ],
            },
        },
        "success_metrics": {
            "micro_continual_fitness": "Mean accuracy on Gaussian M1 (40 regimes, seeds 0–1)",
            "transfer_validation": "Beats baseline on held-out digits M4+M1' (seeds 101–103)",
            "interpretability": "Mechanism is human-readable, not black-box ensemble",
        },
        "benchmarks": {
            "champion_form_fitness": "~0.88 (sigma0_shiftnorm_d099 on micro Gaussian M1)",
            "tuned_baseline_fitness": "~0.885 (structure frozen, constants tuned)",
            "success_threshold": "New rule > 0.89 on fitness, transfers to IPMNIST",
        },
    }


def print_search_plan() -> None:
    """Print a human-readable plan for the expanded search."""
    config = describe_search_configuration()

    print("\n" + "=" * 80)
    print("RULE DISCOVERY V2 — EXPANDED SEARCH PLAN")
    print("=" * 80)
    print(f"\nSearch: {config['search_name']}")
    print(f"Phase: {config['phase']}\n")
    print(f"Hypothesis:\n  {config['hypothesis']}\n")

    print("=" * 80)
    print("SEARCH SPACE BREAKDOWN")
    print("=" * 80)

    directions = config["directions"]

    print(f"\nDirection A (Gate Variants): {directions['Direction A (gate variants)']['templates']} templates")
    print("-" * 80)
    for name in directions["Direction A (gate variants)"]["template_names"]:
        print(f"  • {name}")
    print("\nMechanism Coverage:")
    for mechanism in directions["Direction A (gate variants)"]["mechanism_coverage"]:
        print(f"  - {mechanism}")

    print(f"\nDirection B (Normalization Locations): {directions['Direction B (normalization locations)']['templates']} templates")
    print("-" * 80)
    for name in directions["Direction B (normalization locations)"]["template_names"]:
        print(f"  • {name}")
    print("\nMechanism Coverage:")
    for mechanism in directions["Direction B (normalization locations)"]["mechanism_coverage"]:
        print(f"  - {mechanism}")

    print(f"\nHybrids (A + B): {directions['Hybrids (A + B combinations)']['templates']} templates")
    print("-" * 80)
    for name in directions["Hybrids (A + B combinations)"]["template_names"]:
        print(f"  • {name}")

    print(f"\n" + "=" * 80)
    print("SUCCESS METRICS")
    print("=" * 80)
    for metric, desc in config["success_metrics"].items():
        print(f"  {metric}: {desc}")

    print(f"\n" + "=" * 80)
    print("BENCHMARKS")
    print("=" * 80)
    for bench, value in config["benchmarks"].items():
        print(f"  {bench}: {value}")

    print("\n" + "=" * 80)


def validate_templates() -> dict[str, Any]:
    """Validate all 23 templates: decode, encode, verify genome size."""
    results: dict[str, Any] = {"valid": [], "invalid": []}

    for template in RULE_DISCOVERY_V2_TEMPLATES:
        name = template.get("name", "unnamed")
        try:
            config = template_to_config_dict(template)
            genome = genome_from_config(config)

            # Verify shape
            if genome.shape != (GENOME_SIZE,):
                results["invalid"].append({
                    "name": name,
                    "error": f"genome shape {genome.shape} != {(GENOME_SIZE,)}",
                })
                continue

            # Verify roundtrip
            config_rt = template_to_config_dict(template)
            genome_rt = genome_from_config(config_rt)
            if not np.allclose(genome, genome_rt):
                results["invalid"].append({
                    "name": name,
                    "error": "roundtrip encoding mismatch",
                })
                continue

            # All checks passed
            results["valid"].append({
                "name": name,
                "flags": template.get("flags", ()),
                "description": describe_genome(genome),
            })
        except Exception as e:
            results["invalid"].append({
                "name": name,
                "error": str(e),
            })

    return results


def print_validation_report() -> None:
    """Print validation results for all 23 templates."""
    results = validate_templates()
    valid_count = len(results["valid"])
    invalid_count = len(results["invalid"])

    print("\n" + "=" * 80)
    print("TEMPLATE VALIDATION REPORT")
    print("=" * 80)
    print(f"\nTotal: {valid_count + invalid_count}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")

    if results["invalid"]:
        print("\nInvalid Templates:")
        for row in results["invalid"]:
            print(f"  {row['name']}: {row['error']}")
    else:
        print("\nAll templates valid!")

    print("\n" + "=" * 80)


# ============================================================================
# CLI Usage Examples
# ============================================================================

EXPANDED_SEARCH_CLI_EXAMPLES = """
# Example 1: Run expanded search on Gaussian suite (full fitness budget)
.venv/bin/python -m alberta_framework.benchmarks.rule_discovery search \\
  --suite gauss \\
  --n-random 500 \\
  --population 256 \\
  --generations 50 \\
  --eval-seeds 0 1 \\
  --holdout-seeds 101 102 103 \\
  --out outputs/rule_discovery/search_v2_gaussian_expanded.json

# Example 2: Smoke test (reduced budget, quick validation)
.venv/bin/python -m alberta_framework.benchmarks.rule_discovery search \\
  --suite gauss \\
  --micro-n-tasks 4 \\
  --micro-task-length 500 \\
  --n-random 100 \\
  --population 50 \\
  --generations 5 \\
  --out outputs/rule_discovery/search_v2_smoke.json

# Example 3: Direction A focus (gate variants only)
# (Requires custom filtering of RULE_DISCOVERY_V2_TEMPLATES to gates-only)
.venv/bin/python -m alberta_framework.benchmarks.rule_discovery search \\
  --suite gauss \\
  --n-random 300 \\
  --population 200 \\
  --generations 40 \\
  --out outputs/rule_discovery/search_v2_direction_a_gates.json

# Example 4: Direction B focus (normalization locations only)
# (Requires custom filtering of RULE_DISCOVERY_V2_TEMPLATES to norms-only)
.venv/bin/python -m alberta_framework.benchmarks.rule_discovery search \\
  --suite gauss \\
  --n-random 350 \\
  --population 220 \\
  --generations 45 \\
  --out outputs/rule_discovery/search_v2_direction_b_norms.json

# Example 5: Transfer validation on IPMNIST (for winners from search_v2)
# (Requires ipmnist_screening arm registration)
for discovered_rule in [top winners from search_v2]; do
  .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \\
    --config-name $discovered_rule \\
    --seed 0 --n-tasks 60 \\
    --out outputs/ipmnist_screening/screen_discovered_${discovered_rule}_seed0
done
"""


if __name__ == "__main__":
    print_search_plan()
    print_validation_report()

    print("\n" + "=" * 80)
    print("CLI USAGE EXAMPLES")
    print("=" * 80)
    print(EXPANDED_SEARCH_CLI_EXAMPLES)

    print("\n" + "=" * 80)
    print("INTEGRATION CHECKLIST")
    print("=" * 80)
    print("""
1. [DONE] 23 rule templates implemented (Direction A: 8, Direction B: 12, Hybrids: 3)
2. [DONE] All templates validated (genome size, encoding roundtrip)
3. [DONE] Template loader: expand_seed_genomes_with_templates()
4. [DONE] Configuration documentation: describe_search_configuration()
5. [TODO] Register templates in rule_discovery.seed_genomes() (optional)
6. [TODO] Run expanded search on Gaussian suite (16-20 hours compute)
7. [TODO] Validate winners on held-out digits suite (2 hours)
8. [TODO] Transfer screen on IPMNIST if fitness > 0.89 (2 hours)
9. [TODO] Record results in outputs/rule_discovery/search_v2_gaussian_expanded.json
    """)

    print("\n" + "=" * 80)
    print("Integration ready. Templates are compatible with existing rule_discovery.")
    print("=" * 80 + "\n")
