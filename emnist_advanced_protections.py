"""EMNIST advanced protection mechanisms for continual learning.

Implements four orthogonal protection mechanisms to prevent catastrophic forgetting:

1. Feature-wise normalization: Per-feature normalization layer that adapts to
   feature-level statistics, not just layer-level batch norm.

2. Catastrophic forgetting detector: Monitors accuracy drops and task-driven
   collapse, triggering recovery mechanisms.

3. Per-class batch normalization: Class-specific normalization that tracks
   statistics per output class, not globally.

4. Task-aware dropout scheduling: Dynamic dropout that adapts based on task
   switching and feature stability metrics.

All mechanisms are implemented with complete learner wrappers for integration
into the Alberta continual learning framework.
"""

import dataclasses
import functools
from typing import Any, Callable, Mapping, NamedTuple, Protocol, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float, Bool, UInt

# Type alias for PRNG keys (compatible with multiple JAX versions)
PRNGKey = Array


# =============================================================================
# 1. Feature-Wise Normalization State and Operations
# =============================================================================


@chex.dataclass(frozen=True)
class FeatureWiseNormState:
    """Per-feature normalization tracking.

    Attributes:
        feature_means: Running mean per feature [feature_dim]
        feature_vars: Running variance per feature [feature_dim]
        feature_min: Minimum observed value per feature
        feature_max: Maximum observed value per feature
        update_count: Total number of updates seen
        sample_count_words: Exact uint32 word pair for overflow tracking
        momentum: Exponential moving average momentum
    """
    feature_means: Float[Array, " feature_dim"]
    feature_vars: Float[Array, " feature_dim"]
    feature_min: Float[Array, " feature_dim"]
    feature_max: Float[Array, " feature_dim"]
    update_count: Array  # int32
    sample_count_words: UInt[Array, " 2"]
    momentum: Float[Array, ""]


def init_feature_wise_norm_state(
    feature_dim: int,
    momentum: float = 0.99,
) -> FeatureWiseNormState:
    """Initialize feature-wise normalization state."""
    return FeatureWiseNormState(
        feature_means=jnp.zeros(feature_dim, dtype=jnp.float32),
        feature_vars=jnp.ones(feature_dim, dtype=jnp.float32),
        feature_min=jnp.full(feature_dim, jnp.inf, dtype=jnp.float32),
        feature_max=jnp.full(feature_dim, -jnp.inf, dtype=jnp.float32),
        update_count=jnp.array(0, dtype=jnp.int32),
        sample_count_words=jnp.array([0, 0], dtype=jnp.uint32),
        momentum=jnp.array(momentum, dtype=jnp.float32),
    )


def update_feature_wise_norm(
    state: FeatureWiseNormState,
    features: Float[Array, " feature_dim"],
    epsilon: float = 1e-8,
) -> Tuple[Float[Array, " feature_dim"], FeatureWiseNormState]:
    """Normalize features and update per-feature statistics.

    Args:
        state: Current feature-wise normalization state
        features: Raw feature vector
        epsilon: Small constant for numerical stability

    Returns:
        Tuple of (normalized_features, updated_state)
    """
    # Update running statistics with momentum
    delta_mean = features - state.feature_means
    new_means = state.feature_means + (1.0 - state.momentum) * delta_mean

    delta2 = features - new_means
    new_vars = (
        state.momentum * state.feature_vars +
        (1.0 - state.momentum) * delta_mean * delta2
    )
    new_vars = jnp.maximum(new_vars, epsilon)

    # Track min/max for robust scaling
    new_min = jnp.minimum(state.feature_min, features)
    new_max = jnp.maximum(state.feature_max, features)

    # Normalize using updated statistics
    std = jnp.sqrt(new_vars)
    normalized = (features - new_means) / (std + epsilon)

    # Update word counter (wrapping uint32 pair)
    new_words = state.sample_count_words.at[1].add(1)
    carry = (new_words[1] == 0).astype(jnp.uint32)
    new_words = new_words.at[0].add(carry)

    new_state = FeatureWiseNormState(
        feature_means=new_means,
        feature_vars=new_vars,
        feature_min=new_min,
        feature_max=new_max,
        update_count=state.update_count + 1,
        sample_count_words=new_words,
        momentum=state.momentum,
    )

    return normalized, new_state


# =============================================================================
# 2. Catastrophic Forgetting Detector
# =============================================================================


@chex.dataclass(frozen=True)
class ForgettingDetectorState:
    """Monitors for catastrophic forgetting episodes.

    Attributes:
        prev_task_accuracy: Accuracy from previous task
        task_switch_count: Number of task transitions seen
        accuracy_history: Rolling window of recent accuracies [window_size]
        forgetting_episodes: Count of detected forgetting events
        recovery_mode_active: Whether recovery mechanism is enabled
        recovery_step_count: Steps remaining in recovery mode
        min_accuracy_threshold: Minimum acceptable accuracy
    """
    prev_task_accuracy: Float[Array, ""]
    task_switch_count: Array  # int32
    accuracy_history: Float[Array, " window_size"]
    forgetting_episodes: Array  # int32
    recovery_mode_active: Bool[Array, ""]
    recovery_step_count: Array  # int32
    min_accuracy_threshold: Float[Array, ""]


def init_forgetting_detector_state(
    window_size: int = 20,
    min_threshold: float = 0.75,
) -> ForgettingDetectorState:
    """Initialize catastrophic forgetting detector."""
    return ForgettingDetectorState(
        prev_task_accuracy=jnp.array(0.9, dtype=jnp.float32),
        task_switch_count=jnp.array(0, dtype=jnp.int32),
        accuracy_history=jnp.ones(window_size, dtype=jnp.float32),
        forgetting_episodes=jnp.array(0, dtype=jnp.int32),
        recovery_mode_active=jnp.array(False, dtype=jnp.bool_),
        recovery_step_count=jnp.array(0, dtype=jnp.int32),
        min_accuracy_threshold=jnp.array(min_threshold, dtype=jnp.float32),
    )


def detect_catastrophic_forgetting(
    state: ForgettingDetectorState,
    current_accuracy: Float[Array, ""],
    task_boundary: Bool[Array, ""] = jnp.array(False),
    recovery_duration: int = 50,
    forgetting_threshold: float = 0.15,
) -> Tuple[ForgettingDetectorState, Bool[Array, ""], Float[Array, ""]]:
    """Detect and respond to catastrophic forgetting.

    Args:
        state: Current detector state
        current_accuracy: Current task accuracy
        task_boundary: Whether we're at a task transition
        recovery_duration: Steps to remain in recovery mode
        forgetting_threshold: Accuracy drop threshold to trigger detection

    Returns:
        Tuple of (updated_state, forgetting_detected, recovery_factor)
    """
    # Shift history and insert new accuracy
    new_history = jnp.concatenate([
        state.accuracy_history[1:],
        jnp.array([current_accuracy])
    ])

    # Detect forgetting: large accuracy drop or below threshold
    accuracy_drop = state.prev_task_accuracy - current_accuracy
    below_threshold = current_accuracy < state.min_accuracy_threshold

    forgetting_detected = (
        (accuracy_drop > forgetting_threshold) | below_threshold
    )

    # Update recovery mode
    new_recovery_count = jnp.where(
        forgetting_detected,
        jnp.array(recovery_duration, dtype=jnp.int32),
        jnp.maximum(state.recovery_step_count - 1, 0)
    )
    new_recovery_active = new_recovery_count > 0

    # Count forgetting episodes
    new_episodes = state.forgetting_episodes + forgetting_detected.astype(jnp.int32)

    # Calculate recovery factor: reduced step size during recovery
    recovery_factor = jnp.where(
        new_recovery_active,
        jnp.array(0.5, dtype=jnp.float32),  # 50% step size reduction
        jnp.array(1.0, dtype=jnp.float32)
    )

    # Update task switch count
    new_task_switches = state.task_switch_count + task_boundary.astype(jnp.int32)

    new_state = ForgettingDetectorState(
        prev_task_accuracy=current_accuracy,
        task_switch_count=new_task_switches,
        accuracy_history=new_history,
        forgetting_episodes=new_episodes,
        recovery_mode_active=new_recovery_active,
        recovery_step_count=new_recovery_count,
        min_accuracy_threshold=state.min_accuracy_threshold,
    )

    return new_state, forgetting_detected, recovery_factor


# =============================================================================
# 3. Per-Class Batch Normalization
# =============================================================================


@chex.dataclass(frozen=True)
class PerClassBatchNormState:
    """Class-specific batch normalization tracking.

    Attributes:
        class_means: Running mean per class [n_classes]
        class_vars: Running variance per class [n_classes]
        class_counts: Number of samples per class [n_classes]
        class_last_seen: Step count when each class was last observed
        global_step: Global update counter
        momentum: EMA momentum for class statistics
    """
    class_means: Float[Array, " n_classes"]
    class_vars: Float[Array, " n_classes"]
    class_counts: Array  # int32 [n_classes]
    class_last_seen: Array  # int32 [n_classes]
    global_step: Array  # int32
    momentum: Float[Array, ""]


def init_per_class_batch_norm_state(
    n_classes: int,
    momentum: float = 0.99,
) -> PerClassBatchNormState:
    """Initialize per-class batch normalization."""
    return PerClassBatchNormState(
        class_means=jnp.zeros(n_classes, dtype=jnp.float32),
        class_vars=jnp.ones(n_classes, dtype=jnp.float32),
        class_counts=jnp.zeros(n_classes, dtype=jnp.int32),
        class_last_seen=jnp.zeros(n_classes, dtype=jnp.int32),
        global_step=jnp.array(0, dtype=jnp.int32),
        momentum=jnp.array(momentum, dtype=jnp.float32),
    )


def normalize_per_class(
    state: PerClassBatchNormState,
    output: Float[Array, ""],  # scalar logit
    class_idx: Array,  # int32 scalar
    epsilon: float = 1e-8,
) -> Tuple[Float[Array, ""], PerClassBatchNormState]:
    """Apply per-class normalization to an output.

    Args:
        state: Current per-class norm state
        output: Output logit value
        class_idx: Target class index
        epsilon: Numerical stability constant

    Returns:
        Tuple of (normalized_output, updated_state)
    """
    # Get current class statistics
    class_mean = state.class_means[class_idx]
    class_var = state.class_vars[class_idx]

    # Normalize using class-specific statistics
    class_std = jnp.sqrt(class_var + epsilon)
    normalized = (output - class_mean) / (class_std + epsilon)

    # Update class statistics with momentum
    delta = output - class_mean
    new_class_means = state.class_means.at[class_idx].set(
        state.momentum * class_mean + (1.0 - state.momentum) * output
    )

    delta2 = output - new_class_means[class_idx]
    new_class_vars = state.class_vars.at[class_idx].set(
        state.momentum * class_var + (1.0 - state.momentum) * delta * delta2
    )
    new_class_vars = new_class_vars.at[class_idx].apply(
        lambda v: jnp.maximum(v, epsilon)
    )

    # Update class counts and last-seen step
    new_counts = state.class_counts.at[class_idx].add(1)
    new_last_seen = state.class_last_seen.at[class_idx].set(state.global_step)

    new_state = PerClassBatchNormState(
        class_means=new_class_means,
        class_vars=new_class_vars,
        class_counts=new_counts,
        class_last_seen=new_last_seen,
        global_step=state.global_step + 1,
        momentum=state.momentum,
    )

    return normalized, new_state


# =============================================================================
# 4. Task-Aware Dropout Scheduling
# =============================================================================


@chex.dataclass(frozen=True)
class TaskAwareDropoutState:
    """Dynamic dropout scheduling based on task switching.

    Attributes:
        base_dropout_rate: Default dropout probability
        current_dropout_rate: Currently active dropout rate
        task_stability: Stability metric per feature [feature_dim]
        task_switch_step: Step count of most recent task switch
        current_step: Global step counter
        feature_usage_counts: How many times each feature was active
        dropout_schedule_type: Kind of schedule (exponential, linear, etc.)
    """
    base_dropout_rate: Float[Array, ""]
    current_dropout_rate: Float[Array, ""]
    task_stability: Float[Array, " feature_dim"]
    task_switch_step: Array  # int32
    current_step: Array  # int32
    feature_usage_counts: Array  # int32 [feature_dim]
    dropout_schedule_type: Array  # int32 (0=exponential, 1=linear, 2=cosine)


def init_task_aware_dropout_state(
    feature_dim: int,
    base_dropout_rate: float = 0.1,
    schedule_type: str = "exponential",
) -> TaskAwareDropoutState:
    """Initialize task-aware dropout state."""
    schedule_idx = {
        "exponential": 0,
        "linear": 1,
        "cosine": 2,
    }.get(schedule_type, 0)

    return TaskAwareDropoutState(
        base_dropout_rate=jnp.array(base_dropout_rate, dtype=jnp.float32),
        current_dropout_rate=jnp.array(base_dropout_rate, dtype=jnp.float32),
        task_stability=jnp.ones(feature_dim, dtype=jnp.float32),
        task_switch_step=jnp.array(0, dtype=jnp.int32),
        current_step=jnp.array(0, dtype=jnp.int32),
        feature_usage_counts=jnp.zeros(feature_dim, dtype=jnp.int32),
        dropout_schedule_type=jnp.array(schedule_idx, dtype=jnp.int32),
    )


def schedule_dropout_rate(
    base_rate: Float[Array, ""],
    steps_since_switch: Array,
    schedule_type: int,
    recovery_factor: float = 0.5,
    max_rate: float = 0.5,
) -> Float[Array, ""]:
    """Compute dropout rate based on task-switch timeline.

    Increases dropout shortly after task switches to promote exploration,
    then decays back to baseline.

    Args:
        base_rate: Base dropout rate
        steps_since_switch: Steps elapsed since last task switch
        schedule_type: 0=exponential, 1=linear, 2=cosine
        recovery_factor: Multiplier for post-switch dropout increase
        max_rate: Maximum dropout rate cap

    Returns:
        Current dropout rate
    """
    # Compute decay factor
    decay_steps = 100.0
    progress = jnp.minimum(jnp.asarray(steps_since_switch, dtype=jnp.float32) / decay_steps, 1.0)

    # Schedule-specific decay
    if schedule_type == 0:  # exponential
        decay_fn = jnp.exp(-3.0 * progress)
    elif schedule_type == 1:  # linear
        decay_fn = 1.0 - progress
    else:  # cosine (schedule_type == 2)
        decay_fn = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

    # Dropout increases on task switch, then decays
    elevated_rate = base_rate * recovery_factor
    adaptive_rate = base_rate + (elevated_rate - base_rate) * decay_fn

    return jnp.minimum(adaptive_rate, jnp.array(max_rate, dtype=jnp.float32))


def apply_task_aware_dropout(
    state: TaskAwareDropoutState,
    features: Float[Array, " feature_dim"],
    key: PRNGKey,
    task_boundary: Bool[Array, ""] = jnp.array(False),
    stability_threshold: float = 0.7,
) -> Tuple[Float[Array, " feature_dim"], TaskAwareDropoutState]:
    """Apply adaptive dropout based on task stability.

    Args:
        state: Current dropout state
        features: Input features
        key: Random key for sampling
        task_boundary: Whether we're at a task transition
        stability_threshold: Threshold for considering feature stable

    Returns:
        Tuple of (dropped_features, updated_state)
    """
    # Update task switch step if boundary detected
    new_switch_step = jnp.where(
        task_boundary,
        state.current_step,
        state.task_switch_step
    )

    steps_since_switch = state.current_step - new_switch_step

    # Schedule dropout rate
    new_dropout_rate = schedule_dropout_rate(
        state.base_dropout_rate,
        steps_since_switch,
        state.dropout_schedule_type,
    )

    # Feature-specific dropout based on stability
    # Unstable features get higher dropout
    feature_dropout_rates = jnp.where(
        state.task_stability > stability_threshold,
        new_dropout_rate,
        new_dropout_rate * 1.5,  # 50% higher for unstable features
    )

    # Apply dropout
    dropout_masks = jr.bernoulli(
        key,
        1.0 - feature_dropout_rates,
        shape=features.shape
    )
    dropped_features = features * dropout_masks

    # Rescale to maintain expectation
    scale_factors = jnp.where(
        feature_dropout_rates > 0.0,
        1.0 / (1.0 - feature_dropout_rates),
        1.0
    )
    dropped_features = dropped_features * scale_factors

    # Update feature usage counts
    new_usage_counts = state.feature_usage_counts + dropout_masks.astype(jnp.int32)

    # Update stability scores based on usage patterns
    # Features used consistently are more stable
    total_steps = state.current_step + 1
    usage_fraction = new_usage_counts.astype(jnp.float32) / jnp.maximum(total_steps, 1)
    new_stability = 0.9 * state.task_stability + 0.1 * usage_fraction

    new_state = TaskAwareDropoutState(
        base_dropout_rate=state.base_dropout_rate,
        current_dropout_rate=new_dropout_rate,
        task_stability=new_stability,
        task_switch_step=new_switch_step,
        current_step=state.current_step + 1,
        feature_usage_counts=new_usage_counts,
        dropout_schedule_type=state.dropout_schedule_type,
    )

    return dropped_features, new_state


# =============================================================================
# Complete Protected Learner Implementations
# =============================================================================


class ProtectedEMNISTLearnerState(NamedTuple):
    """Full state for EMNIST learner with all protections enabled.

    Combines feature normalization, forgetting detection, per-class normalization,
    and task-aware dropout into a unified state.
    """
    # Model parameters
    weights: Float[Array, " feature_dim output_dim"]
    bias: Float[Array, " output_dim"]

    # Protection mechanisms
    feature_norm_state: FeatureWiseNormState
    forgetting_state: ForgettingDetectorState
    per_class_norm_state: PerClassBatchNormState
    dropout_state: TaskAwareDropoutState

    # Tracking
    update_count: Array  # int32
    loss_history: Float[Array, " history_size"]


def init_protected_emnist_learner(
    key: PRNGKey,
    feature_dim: int = 784,
    output_dim: int = 47,
    init_scale: float = 0.01,
) -> ProtectedEMNISTLearnerState:
    """Initialize fully protected EMNIST learner.

    Args:
        key: Random key for initialization
        feature_dim: Input feature dimension (default 784 for 28x28 images)
        output_dim: Number of output classes (47 for EMNIST)
        init_scale: Weight initialization scale

    Returns:
        Initialized learner state with all protections
    """
    key_w, key_b = jr.split(key)

    weights = jr.normal(key_w, (feature_dim, output_dim)) * init_scale
    bias = jnp.zeros(output_dim)

    return ProtectedEMNISTLearnerState(
        weights=weights,
        bias=bias,
        feature_norm_state=init_feature_wise_norm_state(feature_dim),
        forgetting_state=init_forgetting_detector_state(),
        per_class_norm_state=init_per_class_batch_norm_state(output_dim),
        dropout_state=init_task_aware_dropout_state(feature_dim),
        update_count=jnp.array(0, dtype=jnp.int32),
        loss_history=jnp.ones(50, dtype=jnp.float32),
    )


def protected_emnist_forward(
    state: ProtectedEMNISTLearnerState,
    features: Float[Array, " feature_dim"],
    training: bool = True,
    key: PRNGKey | None = None,
) -> Tuple[Float[Array, " output_dim"], ProtectedEMNISTLearnerState]:
    """Forward pass with all protection mechanisms applied.

    Args:
        state: Current learner state
        features: Input features
        training: Whether in training mode (enables dropout)
        key: Random key for dropout sampling

    Returns:
        Tuple of (logits, updated_state)
    """
    if key is None:
        key = jr.fold_in(jr.key(0), state.update_count)

    # 1. Apply feature-wise normalization
    norm_features, new_feat_norm = update_feature_wise_norm(
        state.feature_norm_state,
        features
    )

    # 2. Apply task-aware dropout during training
    if training:
        key_dropout, key_rest = jr.split(key)
        dropped_features, new_dropout = apply_task_aware_dropout(
            state.dropout_state,
            norm_features,
            key_dropout,
        )
    else:
        dropped_features = norm_features
        new_dropout = state.dropout_state

    # 3. Linear transformation
    logits = jnp.dot(dropped_features, state.weights) + state.bias

    # Per-class normalization happens during loss computation
    # (applied per-sample based on target class)

    new_state = ProtectedEMNISTLearnerState(
        weights=state.weights,
        bias=state.bias,
        feature_norm_state=new_feat_norm,
        forgetting_state=state.forgetting_state,
        per_class_norm_state=state.per_class_norm_state,
        dropout_state=new_dropout,
        update_count=state.update_count,
        loss_history=state.loss_history,
    )

    return logits, new_state


def protected_emnist_update(
    state: ProtectedEMNISTLearnerState,
    features: Float[Array, " feature_dim"],
    target: Array,  # class index
    current_accuracy: Float[Array, ""],
    step_size: float = 0.01,
    task_boundary: Bool[Array, ""] = jnp.array(False),
    key: PRNGKey | None = None,
) -> Tuple[ProtectedEMNISTLearnerState, dict]:
    """Update learner with all protection mechanisms.

    Args:
        state: Current learner state
        features: Input features
        target: Target class index
        current_accuracy: Current task accuracy
        step_size: Base learning rate
        task_boundary: Whether at task transition
        key: Random key for dropout

    Returns:
        Tuple of (updated_state, metrics)
    """
    if key is None:
        key = jr.fold_in(jr.key(0), state.update_count)

    # Forward pass with protections
    logits, state_after_forward = protected_emnist_forward(
        state, features, training=True, key=key
    )

    # Detect forgetting and get recovery factor
    new_forgetting_state, forgetting_detected, recovery_factor = (
        detect_catastrophic_forgetting(
            state_after_forward.forgetting_state,
            current_accuracy,
            task_boundary,
        )
    )

    # Compute loss (cross-entropy approximation)
    logits_normalized, new_per_class = normalize_per_class(
        state_after_forward.per_class_norm_state,
        logits[target],
        target
    )

    # Simple cross-entropy loss (mean of logits - logit of true class)
    max_logit = jnp.max(logits)
    loss = max_logit - logits_normalized

    # Compute gradients
    grad_w = jnp.outer(state_after_forward.feature_norm_state.feature_means, loss)
    grad_b = loss

    # Apply recovery factor to step size
    effective_step = step_size * float(recovery_factor)

    # Update weights
    new_weights = state_after_forward.weights - effective_step * grad_w
    new_bias = state_after_forward.bias - effective_step * grad_b

    # Update loss history
    new_loss_history = jnp.concatenate([
        state_after_forward.loss_history[1:],
        jnp.array([loss])
    ])

    updated_state = ProtectedEMNISTLearnerState(
        weights=new_weights,
        bias=new_bias,
        feature_norm_state=state_after_forward.feature_norm_state,
        forgetting_state=new_forgetting_state,
        per_class_norm_state=new_per_class,
        dropout_state=state_after_forward.dropout_state,
        update_count=state_after_forward.update_count + 1,
        loss_history=new_loss_history,
    )

    # Compile metrics
    metrics = {
        "loss": float(loss),
        "step_size_used": effective_step,
        "recovery_active": bool(recovery_factor < 1.0),
        "forgetting_detected": bool(forgetting_detected),
        "dropout_rate": float(state_after_forward.dropout_state.current_dropout_rate),
        "feature_norm_mean": float(jnp.mean(state_after_forward.feature_norm_state.feature_means)),
        "feature_norm_std": float(jnp.mean(jnp.sqrt(state_after_forward.feature_norm_state.feature_vars))),
    }

    return updated_state, metrics


# =============================================================================
# Inference and Evaluation
# =============================================================================


def protected_emnist_predict(
    state: ProtectedEMNISTLearnerState,
    features: Float[Array, " feature_dim"],
) -> Float[Array, " output_dim"]:
    """Make predictions with protections applied (no dropout).

    Args:
        state: Learner state
        features: Input features

    Returns:
        Output logits
    """
    logits, _ = protected_emnist_forward(state, features, training=False)
    return logits


def compute_protected_learner_accuracy(
    state: ProtectedEMNISTLearnerState,
    features_batch: Float[Array, " batch feature_dim"],
    targets_batch: Array,  # [batch]
) -> Float[Array, ""]:
    """Compute accuracy on a batch using protected learner.

    Args:
        state: Learner state
        features_batch: Batch of features
        targets_batch: Batch of target classes

    Returns:
        Accuracy as scalar
    """
    def predict_one(features):
        return protected_emnist_predict(state, features)

    predictions = jax.vmap(predict_one)(features_batch)
    predicted_classes = jnp.argmax(predictions, axis=1)
    correct = predicted_classes == targets_batch
    accuracy = jnp.mean(correct.astype(jnp.float32))

    return accuracy


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Feature-wise normalization
    "FeatureWiseNormState",
    "init_feature_wise_norm_state",
    "update_feature_wise_norm",

    # Catastrophic forgetting detector
    "ForgettingDetectorState",
    "init_forgetting_detector_state",
    "detect_catastrophic_forgetting",

    # Per-class batch normalization
    "PerClassBatchNormState",
    "init_per_class_batch_norm_state",
    "normalize_per_class",

    # Task-aware dropout
    "TaskAwareDropoutState",
    "init_task_aware_dropout_state",
    "schedule_dropout_rate",
    "apply_task_aware_dropout",

    # Complete learner
    "ProtectedEMNISTLearnerState",
    "init_protected_emnist_learner",
    "protected_emnist_forward",
    "protected_emnist_update",
    "protected_emnist_predict",
    "compute_protected_learner_accuracy",
]
