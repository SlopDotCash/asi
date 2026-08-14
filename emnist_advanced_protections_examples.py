"""Example usage and integration guide for EMNIST advanced protections.

Demonstrates how to use the four protection mechanisms together
in a complete continual learning workflow.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from emnist_advanced_protections import (
    init_protected_emnist_learner,
    protected_emnist_forward,
    protected_emnist_update,
    protected_emnist_predict,
    compute_protected_learner_accuracy,
)


def example_single_task_training():
    """Example: Train on a single task with all protections."""
    print("\n=== Single Task Training ===")

    key = jr.key(0)
    key, subkey = jr.split(key)

    # Initialize learner with all protections
    learner = init_protected_emnist_learner(subkey, feature_dim=784, output_dim=47)

    # Simulate 100 training steps
    for step in range(100):
        key, subkey = jr.split(key)

        # Generate synthetic data
        features = jr.normal(subkey, (784,))
        target = jnp.array(step % 47)

        # Current accuracy (assuming it stays high on single task)
        accuracy = jnp.array(0.90)

        # Update learner
        learner, metrics = protected_emnist_update(
            learner,
            features,
            target,
            accuracy,
            step_size=0.01,
            key=subkey
        )

        if step % 20 == 0:
            print(f"Step {step}: loss={metrics['loss']:.4f}, "
                  f"dropout_rate={metrics['dropout_rate']:.4f}")

    print(f"Final update count: {learner.update_count}")


def example_multi_task_continual_learning():
    """Example: Continual learning across multiple tasks."""
    print("\n=== Multi-Task Continual Learning ===")

    key = jr.key(0)
    key, subkey = jr.split(key)

    learner = init_protected_emnist_learner(subkey)

    n_tasks = 3
    samples_per_task = 50

    for task_id in range(n_tasks):
        print(f"\nTask {task_id + 1}/{n_tasks}")

        for sample_idx in range(samples_per_task):
            key, subkey = jr.split(key)

            features = jr.normal(subkey, (784,))
            target = jnp.array((task_id * 15 + sample_idx) % 47)

            # Accuracy on current task starts high, may drop on new tasks
            base_accuracy = 0.90 - 0.1 * (0.5 if sample_idx < 10 else 0.0)
            accuracy = jnp.array(base_accuracy)

            # Mark task boundary at first sample
            is_task_boundary = jnp.array(sample_idx == 0 and task_id > 0)

            learner, metrics = protected_emnist_update(
                learner,
                features,
                target,
                accuracy,
                task_boundary=is_task_boundary,
                step_size=0.01,
                key=subkey
            )

            if sample_idx % 10 == 0:
                status = ""
                if metrics["recovery_active"]:
                    status += " [RECOVERY]"
                if metrics["forgetting_detected"]:
                    status += " [FORGETTING]"

                print(f"  Sample {sample_idx}: loss={metrics['loss']:.4f}{status}")


def example_batch_prediction():
    """Example: Make batch predictions with the protected learner."""
    print("\n=== Batch Prediction ===")

    key = jr.key(0)
    key, subkey = jr.split(key)

    learner = init_protected_emnist_learner(subkey)

    # Create test batch
    test_features = jr.normal(jr.key(1), (32, 784))
    test_targets = jr.randint(jr.key(2), (32,), 0, 47)

    # Compute accuracy
    accuracy = compute_protected_learner_accuracy(learner, test_features, test_targets)
    print(f"Batch accuracy: {accuracy:.4f}")

    # Make individual predictions
    for i in range(5):
        features = test_features[i]
        logits = protected_emnist_predict(learner, features)

        predicted_class = jnp.argmax(logits)
        true_class = test_targets[i]
        correct = predicted_class == true_class

        print(f"Sample {i}: predicted={predicted_class}, true={true_class}, "
              f"correct={correct}")


def example_catastrophic_forgetting_detection():
    """Example: Observe catastrophic forgetting detection in action."""
    print("\n=== Catastrophic Forgetting Detection ===")

    key = jr.key(0)
    key, subkey = jr.split(key)

    learner = init_protected_emnist_learner(subkey)

    # Phase 1: Good performance
    print("\nPhase 1: High accuracy (0.90)")
    for step in range(30):
        key, subkey = jr.split(key)

        features = jr.normal(subkey, (784,))
        target = jnp.array(step % 47)
        accuracy = jnp.array(0.90)

        learner, metrics = protected_emnist_update(
            learner, features, target, accuracy, key=subkey
        )

    # Phase 2: Sudden accuracy drop (task switch or distributional shift)
    print("\nPhase 2: Accuracy drops to 0.60 (catastrophic forgetting)")
    recovery_activations = 0

    for step in range(30):
        key, subkey = jr.split(key)

        features = jr.normal(subkey, (784,))
        target = jnp.array(step % 47)
        accuracy = jnp.array(0.60)  # Simulated accuracy drop

        learner, metrics = protected_emnist_update(
            learner, features, target, accuracy, key=subkey
        )

        if metrics["recovery_active"]:
            recovery_activations += 1

        if step == 0:
            print(f"  Step {step}: Forgetting detected={metrics['forgetting_detected']}, "
                  f"Recovery={metrics['recovery_active']}")

    print(f"\nTotal recovery activations: {recovery_activations}")
    print("Recovery mechanism reduces step size during this phase.")


def example_protection_mechanisms_interaction():
    """Example: Show how all four protections work together."""
    print("\n=== Protection Mechanisms Interaction ===")

    key = jr.key(0)
    key, subkey = jr.split(key)

    learner = init_protected_emnist_learner(subkey)

    metrics_over_time = {
        "step": [],
        "loss": [],
        "dropout_rate": [],
        "recovery_active": [],
        "forgetting_detected": [],
        "feature_norm_mean": [],
    }

    for step in range(100):
        key, subkey = jr.split(key)

        features = jr.normal(subkey, (784,))
        target = jnp.array(step % 47)

        # Varying accuracy to trigger different protection responses
        accuracy = jnp.array(0.85 + 0.1 * jnp.sin(step * 0.3))

        # Occasional task boundaries
        task_boundary = jnp.array(step % 25 == 0 and step > 0)

        learner, metrics = protected_emnist_update(
            learner,
            features,
            target,
            accuracy,
            task_boundary=task_boundary,
            key=subkey
        )

        # Record metrics
        metrics_over_time["step"].append(step)
        metrics_over_time["loss"].append(metrics["loss"])
        metrics_over_time["dropout_rate"].append(float(metrics["dropout_rate"]))
        metrics_over_time["recovery_active"].append(metrics["recovery_active"])
        metrics_over_time["forgetting_detected"].append(metrics["forgetting_detected"])
        metrics_over_time["feature_norm_mean"].append(metrics["feature_norm_mean"])

    # Report summary
    print("\nProtection Mechanisms Summary:")
    print(f"  Total steps: {len(metrics_over_time['step'])}")
    print(f"  Forgetting events: {sum(metrics_over_time['forgetting_detected'])}")
    print(f"  Recovery activations: {sum(metrics_over_time['recovery_active'])}")
    print(f"  Average dropout rate: {sum(metrics_over_time['dropout_rate']) / len(metrics_over_time['dropout_rate']):.4f}")
    print(f"  Final feature norm mean: {metrics_over_time['feature_norm_mean'][-1]:.4f}")


def main():
    """Run all examples."""
    print("="*60)
    print("EMNIST Advanced Protection Mechanisms - Examples")
    print("="*60)

    example_single_task_training()
    example_multi_task_continual_learning()
    example_batch_prediction()
    example_catastrophic_forgetting_detection()
    example_protection_mechanisms_interaction()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
