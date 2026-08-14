"""Example usage of SCR v2 advanced optimizers.

Demonstrates all four optimizer variants on a synthetic regression task.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from scr_v2_advanced_optimizers import (
    make_exponential_decay_lr_learner,
    make_nesterov_momentum_learner,
    make_dynamic_ensemble_learner,
    make_adaptive_rmsprop_learner,
)


def generate_task_data(
    key: jax.Array,
    num_samples: int = 100,
    feature_dim: int = 20,
) -> tuple[jax.Array, jax.Array]:
    """Generate synthetic regression data."""
    x = jr.normal(key, (num_samples, feature_dim))
    # True target: linear combination of features
    w_true = jr.normal(jr.fold_in(key, 1), (feature_dim,)) * 0.1
    y = jnp.dot(x, w_true) + 0.01 * jr.normal(jr.fold_in(key, 2), (num_samples,))
    return x, y


def compute_loss(params: dict, x: jax.Array, y: jax.Array) -> float:
    """Compute MSE loss."""
    hidden = jnp.maximum(jnp.dot(x, params["w"]) + params["b"], 0.0)
    pred = jnp.sum(hidden, axis=-1)
    return jnp.mean((pred - y) ** 2)


def run_optimizer(
    name: str,
    make_learner_fn,
    hp: dict,
    x: jax.Array,
    y: jax.Array,
    num_steps: int = 1000,
) -> dict:
    """Run an optimizer and track metrics."""
    init_fn, step_fn = make_learner_fn(hp)
    key = jr.key(42)
    params, state = init_fn(key, feature_dim=x.shape[1])

    losses = []
    lrs = []

    for step in range(num_steps):
        params, state, lr = step_fn(params, state, x, y)
        loss = compute_loss(params, x, y)
        losses.append(float(loss))
        lrs.append(lr)

        if step % 100 == 0:
            print(f"  {name:30s} Step {step:4d}: loss={loss:.6f}, lr={lr:.6f}")

    return {
        "name": name,
        "losses": losses,
        "lrs": lrs,
        "final_loss": losses[-1],
        "final_params": params,
        "final_state": state,
    }


def compare_optimizers():
    """Compare all four advanced optimizers on the same task."""
    print("=" * 80)
    print("SCR v2 Advanced Optimizers: Comparative Analysis")
    print("=" * 80)

    # Generate task data
    key = jr.key(0)
    x, y = generate_task_data(key, num_samples=100, feature_dim=20)
    print(f"\nTask: Synthetic regression with {x.shape[0]} samples, {x.shape[1]} features")
    print(f"Data: x shape {x.shape}, y shape {y.shape}")

    results = {}

    # 1. Exponential Decay LR
    print("\n" + "-" * 80)
    print("1. EXPONENTIAL ADAPTIVE LEARNING RATE DECAY")
    print("-" * 80)
    hp = {
        "base_lr": 0.01,
        "lr_decay_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.01,
    }
    results["exponential_decay"] = run_optimizer(
        "ExponentialDecayLR",
        make_exponential_decay_lr_learner,
        hp,
        x,
        y,
    )

    # 2. Nesterov Momentum
    print("\n" + "-" * 80)
    print("2. NESTEROV ACCELERATED GRADIENT")
    print("-" * 80)
    hp = {
        "learning_rate": 0.01,
        "momentum": 0.9,
        "nesterov_lookahead": 1.0,
        "weight_decay": 0.01,
    }
    results["nesterov"] = run_optimizer(
        "NesterovMomentum",
        make_nesterov_momentum_learner,
        hp,
        x,
        y,
    )

    # 3. Dynamic Ensemble
    print("\n" + "-" * 80)
    print("3. DYNAMIC ENSEMBLE OF 3 OPTIMIZERS")
    print("-" * 80)
    hp = {
        "learning_rate": 0.01,
        "momentum_sgd": 0.9,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "rmsprop_decay": 0.99,
        "weight_decay": 0.01,
    }
    results["ensemble"] = run_optimizer(
        "DynamicEnsemble",
        make_dynamic_ensemble_learner,
        hp,
        x,
        y,
    )

    # 4. Adaptive RMSprop
    print("\n" + "-" * 80)
    print("4. RMSPROP WITH ADAPTIVE EPSILON")
    print("-" * 80)
    hp = {
        "learning_rate": 0.01,
        "rmsprop_decay": 0.99,
        "base_epsilon": 1e-8,
        "epsilon_scale": 0.1,
        "weight_decay": 0.01,
    }
    results["adaptive_rmsprop"] = run_optimizer(
        "AdaptiveRMSprop",
        make_adaptive_rmsprop_learner,
        hp,
        x,
        y,
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: CONVERGENCE COMPARISON")
    print("=" * 80)
    print(f"{'Optimizer':<30s} {'Initial Loss':<15s} {'Final Loss':<15s} {'Improvement':<15s}")
    print("-" * 80)

    initial_losses = {}
    for name, result in results.items():
        initial_loss = result["losses"][0]
        final_loss = result["final_loss"]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        initial_losses[name] = initial_loss

        print(
            f"{result['name']:<30s} {initial_loss:<15.6f} {final_loss:<15.6f} "
            f"{improvement:<15.1f}%"
        )

    # Find best performer
    best_name = min(results.items(), key=lambda x: x[1]["final_loss"])[0]
    best_result = results[best_name]
    print(f"\n[BEST] Final loss: {best_result['name']} ({best_result['final_loss']:.6f})")

    # Learning rate analysis
    print("\n" + "-" * 80)
    print("LEARNING RATE STATISTICS")
    print("-" * 80)
    for name, result in results.items():
        lrs = result["lrs"]
        print(
            f"{result['name']:<30s} Initial LR: {lrs[0]:.6f}, "
            f"Final LR: {lrs[-1]:.6f}, Ratio: {lrs[-1]/lrs[0]:.4f}"
        )

    # Loss trajectory analysis
    print("\n" + "-" * 80)
    print("LOSS TRAJECTORY: EARLY VS LATE CONVERGENCE")
    print("-" * 80)
    for name, result in results.items():
        losses = result["losses"]
        early_avg = sum(losses[1:11]) / 10  # Average of steps 1-10
        mid_avg = sum(losses[400:410]) / 10  # Average of steps 400-410
        late_avg = sum(losses[990:1000]) / 10  # Average of steps 990-999

        print(f"\n{result['name']}:")
        print(f"  Steps 1-10 avg:     {early_avg:.6f}")
        print(f"  Steps 400-410 avg:  {mid_avg:.6f}")
        print(f"  Steps 990-999 avg:  {late_avg:.6f}")

    return results


if __name__ == "__main__":
    results = compare_optimizers()
    print("\n" + "=" * 80)
    print("Comparison complete. All optimizers successfully trained.")
    print("=" * 80)
