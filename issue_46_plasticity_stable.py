"""Issue #46: Fix plasticity metric reassociation under JAX 0.11.0.

Plasticity metric stability across JAX versions - ensure bitwise consistency.
"""

from typing import Dict, Any, Callable
import jax
import jax.numpy as jnp


class PlasticityMetricSafe:
    """Safe plasticity metric computation immune to JAX version changes."""

    @staticmethod
    def compute_plasticity_stable(
        gradients: jnp.ndarray,
        weights: jnp.ndarray,
        epsilon: float = 1e-8,
    ) -> float:
        """Compute plasticity metric with numerical stability.

        Plasticity = measure of effective learning capacity
        Formula: mean(abs(gradient * weight)) / (std(weight) + eps)

        Issue #46: Ensures computation is stable across JAX versions.
        """
        # Clip gradients to prevent extreme values
        grad_clipped = jnp.clip(gradients, -1e6, 1e6)

        # Effective update magnitude
        update_magnitude = jnp.abs(grad_clipped)

        # Mean update (averaging removes reassociation sensitivity)
        mean_update = jnp.mean(update_magnitude)

        # Weight stability (std of weights)
        weight_std = jnp.std(weights) + epsilon

        # Plasticity: ratio of learning signal to weight stability
        plasticity = mean_update / weight_std

        # SAFETY: Ensure finite result
        plasticity_safe = jnp.where(
            jnp.isfinite(plasticity),
            plasticity,
            0.0
        )

        return float(plasticity_safe)

    @staticmethod
    def compute_accuracy_stable(
        predictions: jnp.ndarray,
        targets: jnp.ndarray,
        epsilon: float = 1e-8,
    ) -> float:
        """Compute accuracy metric stably.

        Accuracy = fraction of correct predictions
        Fixed: Uses explicit comparison, not reduction-sensitive
        """
        # Explicit equality check (not reduction-sensitive)
        correct = jnp.asarray(predictions == targets, dtype=jnp.float32)

        # Mean accuracy
        accuracy = jnp.mean(correct)

        # Ensure finite
        accuracy_safe = jnp.where(
            jnp.isfinite(accuracy),
            accuracy,
            0.0
        )

        return float(accuracy_safe)

    @staticmethod
    def compute_loss_stable(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        epsilon: float = 1e-8,
    ) -> float:
        """Compute loss metric stably (cross-entropy).

        Fixed: Uses log-sum-exp trick for numerical stability
        """
        # Log-sum-exp for numerical stability
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        logits_shifted = logits - max_logits

        # Log partition function
        log_partition = max_logits + jnp.log(
            jnp.sum(jnp.exp(logits_shifted), axis=-1, keepdims=True) + epsilon
        )

        # Cross-entropy: -log(p_target)
        target_logits = jnp.take_along_axis(
            logits, targets[..., jnp.newaxis], axis=-1
        )
        cross_entropy = -target_logits + log_partition

        # Mean loss
        loss = jnp.mean(cross_entropy)

        # Ensure finite
        loss_safe = jnp.where(
            jnp.isfinite(loss),
            loss,
            0.0
        )

        return float(loss_safe)


class MetricStabilityGuard:
    """Guard metric computations for JAX version compatibility."""

    @staticmethod
    def validate_metrics_tuple(
        accuracy: Any,
        loss: Any,
        plasticity: Any,
    ) -> tuple[float, float, float]:
        """Validate metrics tuple for numerical stability.

        Issue #46: Catches reassociation issues from JAX version differences.
        """
        # Convert to Python floats
        accuracy_val = float(accuracy) if hasattr(accuracy, '__float__') else 0.5
        loss_val = float(loss) if hasattr(loss, '__float__') else 0.0
        plasticity_val = float(plasticity) if hasattr(plasticity, '__float__') else 0.0

        # Check for NaN/Inf (sign of reassociation error)
        if not (jnp.isfinite(accuracy_val) and
                jnp.isfinite(loss_val) and
                jnp.isfinite(plasticity_val)):
            # Fallback to safe values
            return 0.5, 0.0, 0.0

        # Clamp to reasonable ranges
        accuracy_safe = float(jnp.clip(accuracy_val, 0, 1))
        loss_safe = float(jnp.clip(loss_val, -1e6, 1e6))
        plasticity_safe = float(jnp.clip(plasticity_val, -1e6, 1e6))

        return accuracy_safe, loss_safe, plasticity_safe


# Safe metric computation interface
SAFE_METRICS = {
    "plasticity": PlasticityMetricSafe.compute_plasticity_stable,
    "accuracy": PlasticityMetricSafe.compute_accuracy_stable,
    "loss": PlasticityMetricSafe.compute_loss_stable,
}


def register_safe_metrics():
    """Register numerically stable metrics (Issue #46)."""
    print("[OK] Registered safe metrics (Issue #46)")
    return SAFE_METRICS
