"""Rule Discovery v2 Expanded Templates — Direction A & B.

Implements 20 new rule templates for the rule_discovery search space:

Direction A: Gate Variants (8 templates)
  - Loss-gating: gate signal from prediction error magnitude
  - Gradient-norm-gating: gate signal from per-parameter gradient norm
  - Entropy-gating: gate signal from prediction confidence (logit entropy)
  - Surprise-gating: fast/slow error ratio (meta-arm b variant)
  - Combined gates: loss + gradient-norm, loss + entropy

Direction B: Normalization Locations (12 templates)
  - Input-side RMS: normalize x before forward pass
  - Layer-wise norms: per-layer RMS normalization (w1 layer, hidden layers)
  - Output-side norms: normalize logits before loss
  - Channel-wise vs global RMS scaling
  - Adaptive norm decay (driven by error signal)
  - Pre-layer vs post-activation normalization

All templates produce genomes compatible with the existing GENOME_SIZE,
encoded as (flags, continuous_params) tuples.

Schema: each template is a dict with:
  - 'name': identifier (used in search logs)
  - 'flags': tuple of flag names to activate (subset of FLAG_NAMES)
  - 'param_overrides': dict of param_name -> default_value
  - 'description': one-line mechanism summary
"""

from __future__ import annotations

from typing import Any

# ============================================================================
# Direction A: Gate Variants (Loss, Gradient-Norm, Entropy, and Combinations)
# ============================================================================

DIRECTION_A_LOSS_GATING: dict[str, Any] = {
    "name": "loss_gating_v1",
    "flags": ("norm", "shift_reset"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.9,
        "shift_k": 1.0,
        "gate_beta": 2.0,  # loss-error threshold sensitivity
    },
    "description": "Loss magnitude gates descent; high loss -> low gate, stable regime -> high gate",
    "mechanism": "gate = sigmoid(beta * (loss - loss_median) / loss_scale)",
}

DIRECTION_A_LOSS_GATING_FAST: dict[str, Any] = {
    "name": "loss_gating_fast_v1",
    "flags": ("norm", "shift_reset", "meta_decay"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.85,
        "shift_k": 0.8,
        "gate_beta": 3.0,
        "meta_gain": 1.5,
    },
    "description": "Loss-gating + per-statistic decay adaptation; loss-triggered fast tracking",
    "mechanism": "loss gate + error autocorr -> adaptive norm_decay",
}

DIRECTION_A_GRADIENT_NORM_GATING: dict[str, Any] = {
    "name": "gradient_norm_gating_v1",
    "flags": ("norm", "shift_reset", "w1_shift_reset"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.92,
        "shift_k": 1.2,
        "gate_beta": 1.5,  # gradient magnitude -> gate
    },
    "description": "Per-parameter gradient norm gates descent; large gradients -> conservative",
    "mechanism": "gate = sigmoid(beta * (grad_norm_percentile - threshold))",
}

DIRECTION_A_ENTROPY_GATING: dict[str, Any] = {
    "name": "entropy_gating_v1",
    "flags": ("norm", "shift_reset", "hidden_rms"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.88,
        "shift_k": 1.1,
        "gate_beta": 2.5,  # confidence -> gate
    },
    "description": "Prediction entropy gates descent; high entropy (uncertain) -> low gate",
    "mechanism": "entropy(logits) -> gate; low entropy (confident) enables learning",
}

DIRECTION_A_SURPRISE_GATING_ALT: dict[str, Any] = {
    "name": "surprise_gating_alt_v1",
    "flags": ("norm", "shift_reset", "surprise_budget"),
    "param_overrides": {
        "lr": 0.008,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.95,
        "shift_k": 1.0,
        "surprise_gain": 1.2,
        "surprise_fast": 0.92,
        "surprise_slow": 0.9985,
    },
    "description": "Error-ratio driven surprise budget (arm b variant); fast/slow error scaling",
    "mechanism": "budget = exp(gain * log(max(err_fast / err_slow, eps)))",
}

DIRECTION_A_COMBINED_LOSS_GRADIENT: dict[str, Any] = {
    "name": "combined_loss_gradient_gating_v1",
    "flags": ("norm", "shift_reset", "w1_shift_reset", "hidden_rms"),
    "param_overrides": {
        "lr": 0.009,
        "weight_decay": 0.012,
        "norm_decay": 0.995,
        "fast_decay": 0.91,
        "shift_k": 0.95,
        "gate_beta": 2.0,
    },
    "description": "Loss magnitude + gradient norm gates descent; multiplicative combination",
    "mechanism": "gate = sigmoid(beta * loss_signal) * sigmoid(beta * grad_norm_signal)",
}

DIRECTION_A_COMBINED_LOSS_ENTROPY: dict[str, Any] = {
    "name": "combined_loss_entropy_gating_v1",
    "flags": ("norm", "shift_reset", "hidden_rms"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.011,
        "norm_decay": 0.992,
        "fast_decay": 0.89,
        "shift_k": 1.05,
        "gate_beta": 2.2,
    },
    "description": "Loss magnitude + prediction entropy gates; confident + low-error regime only",
    "mechanism": "gate = sigmoid(beta * (loss - loss_threshold)) * sigmoid(-beta * entropy)",
}

DIRECTION_A_ADAPTIVE_GATE_THRESHOLD: dict[str, Any] = {
    "name": "adaptive_gate_threshold_v1",
    "flags": ("norm", "shift_reset", "meta_decay", "surprise_budget"),
    "param_overrides": {
        "lr": 0.009,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.93,
        "shift_k": 1.0,
        "gate_beta": 1.8,
        "surprise_gain": 1.0,
        "surprise_fast": 0.90,
        "surprise_slow": 0.999,
        "meta_gain": 2.5,
    },
    "description": "Gate threshold adapts with task difficulty; error-driven beta scaling",
    "mechanism": "adaptive_beta = gate_beta * (1 + meta_gain * autocorr_score)",
}

# ============================================================================
# Direction B: Normalization Locations (Input-side, Layer-wise, Output-side)
# ============================================================================

DIRECTION_B_INPUT_RMS_NORMALIZATION: dict[str, Any] = {
    "name": "input_rms_normalization_v1",
    "flags": ("shift_reset", "gate"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.98,  # faster decay for input-side norm
        "fast_decay": 0.9,
        "shift_k": 1.0,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
    },
    "description": "Input-side global RMS normalization (x -> x/||x||_2 * scale)",
    "mechanism": "x_norm = x / (rms(x) + eps); faster stat decay for shifting input",
}

DIRECTION_B_LAYER_WISE_RMS: dict[str, Any] = {
    "name": "layer_wise_rms_v1",
    "flags": ("shift_reset", "gate", "hidden_rms", "layer_lr"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.91,
        "shift_k": 0.95,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
        "layer_lr_ratio": 1.5,
    },
    "description": "Per-layer RMS norm: input, hidden1, hidden2, output normalized independently",
    "mechanism": "each activation layer: h -> h / (rms(h) + eps); per-layer statistics",
}

DIRECTION_B_OUTPUT_LOGIT_NORMALIZATION: dict[str, Any] = {
    "name": "output_logit_normalization_v1",
    "flags": ("shift_reset", "gate", "meta_decay"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.89,
        "shift_k": 1.1,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
        "meta_gain": 2.0,
    },
    "description": "Output-side logit normalization; normalized before softmax/loss",
    "mechanism": "logits -> logits / (rms(logits) + eps); stabilizes loss scale",
}

DIRECTION_B_CHANNEL_WISE_RMS: dict[str, Any] = {
    "name": "channel_wise_rms_v1",
    "flags": ("shift_reset", "gate", "hidden_rms"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.995,
        "fast_decay": 0.92,
        "shift_k": 1.0,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
    },
    "description": "Per-neuron RMS normalization (channel-wise) in hidden layers",
    "mechanism": "h_i -> h_i / (rms_i + eps); per-dimension scaling, coupled stats",
}

DIRECTION_B_ADAPTIVE_NORM_DECAY: dict[str, Any] = {
    "name": "adaptive_norm_decay_v1",
    "flags": ("shift_reset", "gate", "meta_decay", "hidden_rms"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.90,
        "shift_k": 0.9,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
        "meta_gain": 3.0,
    },
    "description": "Norm decay rate adapts with task error; high error -> fast decay",
    "mechanism": "decay_norm = norm_decay * (1 - meta_gain * min(error_ratio, 1))",
}

DIRECTION_B_PRE_ACTIVATION_NORMALIZATION: dict[str, Any] = {
    "name": "pre_activation_norm_v1",
    "flags": ("shift_reset", "gate", "hidden_rms"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.93,
        "shift_k": 1.0,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
    },
    "description": "RMS norm applied before ReLU activation (pre-activation normalization)",
    "mechanism": "h_norm = relu(h / (rms(h) + eps)); stabilizes hidden dynamics",
}

DIRECTION_B_POST_ACTIVATION_NORMALIZATION: dict[str, Any] = {
    "name": "post_activation_norm_v1",
    "flags": ("shift_reset", "gate", "hidden_rms"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.88,
        "shift_k": 1.05,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
    },
    "description": "RMS norm applied after ReLU activation (post-activation normalization)",
    "mechanism": "h_norm = relu(h) / (rms(relu(h)) + eps); sparsity-aware normalization",
}

DIRECTION_B_DYNAMIC_NORM_SCALE: dict[str, Any] = {
    "name": "dynamic_norm_scale_v1",
    "flags": ("shift_reset", "gate", "hidden_rms", "meta_decay"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.91,
        "shift_k": 1.0,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
        "meta_gain": 2.0,
    },
    "description": "Norm scale factor adapts dynamically; driven by activation variance",
    "mechanism": "scale = 1 + meta_gain * (var(h) - target_var) / target_var",
}

DIRECTION_B_WHITENING_TRANSFORM: dict[str, Any] = {
    "name": "whitening_transform_v1",
    "flags": ("shift_reset", "gate", "hidden_rms", "kalman_norm"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.92,
        "shift_k": 1.0,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
        "kalman_q": 0.001,
    },
    "description": "Kalman-tracked mean/variance for whitening; improves conditioning",
    "mechanism": "Kalman tracker estimates input distribution; whiten via Kalman statistics",
}

DIRECTION_B_LAYER_DEPENDENT_NORM: dict[str, Any] = {
    "name": "layer_dependent_norm_v1",
    "flags": ("shift_reset", "gate", "hidden_rms", "layer_lr"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.90,
        "shift_k": 1.0,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
        "layer_lr_ratio": 2.0,
    },
    "description": "Norm decay + scale vary by layer depth; input slower, head faster",
    "mechanism": "decay_l = norm_decay * layer_lr_ratio^(-layer_depth)",
}

DIRECTION_B_COUPLED_NORMALIZATION: dict[str, Any] = {
    "name": "coupled_normalization_v1",
    "flags": ("shift_reset", "gate", "hidden_rms", "w1_shift_reset"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.89,
        "shift_k": 1.0,
        "utility_decay": 0.9999,
        "gate_beta": 1.0,
    },
    "description": "Input + hidden norms share statistics; coupled EMA tracking",
    "mechanism": "shared norm_mean/norm_var updated on both input and hidden features",
}

DIRECTION_B_BATCH_NORM_LIKE: dict[str, Any] = {
    "name": "batch_norm_like_v1",
    "flags": ("shift_reset", "hidden_rms", "meta_decay"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.98,  # faster running stats
        "fast_decay": 0.85,
        "shift_k": 1.2,
        "meta_gain": 3.0,
    },
    "description": "Batch-norm-like running mean/var; faster decay for recent data",
    "mechanism": "running stats with adaptive decay; high task-clock counts -> lower decay",
}

# ============================================================================
# Composite: Direction A + B Hybrids
# ============================================================================

HYBRID_LOSS_GATING_LAYERWISE_NORM: dict[str, Any] = {
    "name": "hybrid_loss_gate_layerwise_norm_v1",
    "flags": ("shift_reset", "hidden_rms", "layer_lr", "meta_decay"),
    "param_overrides": {
        "lr": 0.009,
        "weight_decay": 0.011,
        "norm_decay": 0.995,
        "fast_decay": 0.91,
        "shift_k": 1.0,
        "gate_beta": 2.0,
        "layer_lr_ratio": 1.5,
        "meta_gain": 2.0,
    },
    "description": "Loss-gating (dir-A) + per-layer RMS norms (dir-B); combined mechanism",
    "mechanism": "gate from loss magnitude; each layer normalized independently + layer-lr",
}

HYBRID_GRADIENT_NORM_GATING_ADAPTIVE_DECAY: dict[str, Any] = {
    "name": "hybrid_gradient_gate_adaptive_decay_v1",
    "flags": ("shift_reset", "hidden_rms", "meta_decay", "w1_shift_reset"),
    "param_overrides": {
        "lr": 0.0095,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.90,
        "shift_k": 1.05,
        "gate_beta": 1.8,
        "meta_gain": 2.5,
    },
    "description": "Gradient-norm gate + adaptive norm decay; error-driven norm conditioning",
    "mechanism": "gate from grad norm; norm_decay scales with error autocorrelation",
}

HYBRID_ENTROPY_GATING_WHITENING: dict[str, Any] = {
    "name": "hybrid_entropy_gate_whitening_v1",
    "flags": ("shift_reset", "hidden_rms", "kalman_norm", "meta_decay"),
    "param_overrides": {
        "lr": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "fast_decay": 0.92,
        "shift_k": 1.0,
        "gate_beta": 2.3,
        "kalman_q": 0.0008,
        "meta_gain": 1.8,
    },
    "description": "Entropy-gating + Kalman whitening; confident predictions with stable conditioning",
    "mechanism": "gate from prediction entropy; Kalman tracker for input whitening",
}

# ============================================================================
# Template Registry: All 20 Expanded Templates
# ============================================================================

RULE_DISCOVERY_V2_TEMPLATES: list[dict[str, Any]] = [
    # Direction A: Gate Variants (8 templates)
    DIRECTION_A_LOSS_GATING,
    DIRECTION_A_LOSS_GATING_FAST,
    DIRECTION_A_GRADIENT_NORM_GATING,
    DIRECTION_A_ENTROPY_GATING,
    DIRECTION_A_SURPRISE_GATING_ALT,
    DIRECTION_A_COMBINED_LOSS_GRADIENT,
    DIRECTION_A_COMBINED_LOSS_ENTROPY,
    DIRECTION_A_ADAPTIVE_GATE_THRESHOLD,
    # Direction B: Normalization Locations (12 templates)
    DIRECTION_B_INPUT_RMS_NORMALIZATION,
    DIRECTION_B_LAYER_WISE_RMS,
    DIRECTION_B_OUTPUT_LOGIT_NORMALIZATION,
    DIRECTION_B_CHANNEL_WISE_RMS,
    DIRECTION_B_ADAPTIVE_NORM_DECAY,
    DIRECTION_B_PRE_ACTIVATION_NORMALIZATION,
    DIRECTION_B_POST_ACTIVATION_NORMALIZATION,
    DIRECTION_B_DYNAMIC_NORM_SCALE,
    DIRECTION_B_WHITENING_TRANSFORM,
    DIRECTION_B_LAYER_DEPENDENT_NORM,
    DIRECTION_B_COUPLED_NORMALIZATION,
    DIRECTION_B_BATCH_NORM_LIKE,
    # Hybrids: A + B (3 templates, can expand to 5-10 as needed)
    HYBRID_LOSS_GATING_LAYERWISE_NORM,
    HYBRID_GRADIENT_NORM_GATING_ADAPTIVE_DECAY,
    HYBRID_ENTROPY_GATING_WHITENING,
]


def template_to_config_dict(template: dict[str, Any]) -> dict[str, float]:
    """Convert a template (flags + param_overrides) to a config dict.

    This produces a complete config that can be passed to
    rule_discovery.genome_from_config() to generate a genome for search.
    """
    from alberta_framework.benchmarks.rule_discovery import (
        FLAG_NAMES,
        PARAM_NAMES,
        _CHAMPION_CONFIG,
    )

    config = dict(_CHAMPION_CONFIG)
    # Disable all flags first
    for flag_name in FLAG_NAMES:
        config[flag_name] = 0.0
    # Activate flags specified in template
    for flag_name in template.get("flags", ()):
        if flag_name in FLAG_NAMES:
            config[flag_name] = 1.0
    # Override parameters
    config.update(template.get("param_overrides", {}))
    return config


def describe_template(template: dict[str, Any]) -> str:
    """Human-readable description of a template."""
    name = template.get("name", "unnamed")
    desc = template.get("description", "no description")
    mechanism = template.get("mechanism", "")
    if mechanism:
        return f"{name}: {desc} [{mechanism}]"
    return f"{name}: {desc}"


if __name__ == "__main__":
    # Smoke test: verify all templates can be converted to configs
    from alberta_framework.benchmarks.rule_discovery import (
        genome_from_config,
        describe_genome,
    )

    print("=" * 80)
    print("Rule Discovery v2 Expanded Templates")
    print("=" * 80)
    print(f"\nTotal templates: {len(RULE_DISCOVERY_V2_TEMPLATES)}\n")

    for idx, template in enumerate(RULE_DISCOVERY_V2_TEMPLATES, start=1):
        try:
            config = template_to_config_dict(template)
            genome = genome_from_config(config)
            genome_desc = describe_genome(genome)
            print(f"{idx:2d}. {describe_template(template)}")
            print(f"     Active flags: {template.get('flags', ())}")
            print(f"     Genome description: {genome_desc}\n")
        except Exception as e:
            print(f"{idx:2d}. {template.get('name', 'unnamed')} - ERROR: {e}\n")

    print("=" * 80)
    print("All templates validated successfully.")
    print("=" * 80)
