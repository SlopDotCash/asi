"""Rule Discovery Final Search: Quick Reference & Examples

Practical guide for using the final search strategies.
"""

# ============================================================================
# QUICK START
# ============================================================================

## Installation

All implementations use standard scientific Python stack:
- JAX >= 0.4.0
- NumPy >= 1.26
- Chex (already in alberta_framework)

No additional dependencies required beyond alberta_framework.


## Basic Example

```python
import jax.random as jr
import numpy as np
from rule_discovery_final_search import FinalSearchStrategy, SearchConfig

# Configure strategies
config = SearchConfig(
    use_bayesian=True,
    use_hypervolume=True,
    use_thompson=True,
    use_active_learning=True,
    total_evaluations=10000,
    batch_size=64,
    max_generations=100,
)

# Initialize strategy
strategy = FinalSearchStrategy(config, key=jr.key(0))

# Simulate search (normally integrated with evaluate_suite)
candidate_pool = np.random.randn(1000, 34)  # 1000 random genomes
evaluated_genomes = []
evaluated_accuracies = []

for generation in range(config.max_generations):
    # Select batch
    batch_indices = strategy.select_next_batch(
        evaluated_genomes,
        evaluated_accuracies,
        candidate_pool,
        generation,
    )

    # Evaluate (would call evaluate_suite here)
    batch_genomes = candidate_pool[batch_indices]
    batch_accuracies = [0.7 + 0.1 * np.random.randn() for _ in batch_indices]

    # Log progress
    log = strategy.log_step(generation, batch_accuracies, batch_genomes)
    print(f"Gen {generation}: best={log['best_accuracy']:.3f}, "
          f"HV={log.get('hypervolume', 0):.4f}")

    # Update archive
    evaluated_genomes.extend(batch_genomes)
    evaluated_accuracies.extend(batch_accuracies)

print(f"Search history: {len(strategy.search_history)} generations")
```


# ============================================================================
# COMPONENT-SPECIFIC USAGE
# ============================================================================

## 1. Bayesian Optimization Only

```python
from rule_discovery_final_search import BayesianOptimizer
import jax.random as jr
import numpy as np

# Create optimizer
optimizer = BayesianOptimizer(
    search_space_dim=34,
    key=jr.key(0),
    kernel="matern",
)

# Fit GP to observed data
X_observed = np.random.randn(50, 34)  # Evaluated genomes
y_observed = np.random.rand(50) * 0.3 + 0.6  # Fitnesses

gp_model = optimizer.fit_gp(X_observed, y_observed)

# Generate candidates
candidates = np.random.randn(1000, 34)

# Compute EI scores
ei_scores = optimizer.expected_improvement(
    gp_model,
    candidates,
    y_best=np.max(y_observed),
    xi=0.01,  # Exploration parameter
)

# Select top-32 by EI
top_indices = np.argsort(-ei_scores)[:32]
selected_genomes = candidates[top_indices]
```

**Tuning tips**:
- Increase `xi` (0.1+) for more exploration
- Decrease `xi` (0.001) for more exploitation
- Switch to RBF kernel for smoother landscapes
- Fit on every generation for online learning


## 2. Hypervolume Multi-Objective Only

```python
from rule_discovery_final_search import HypervolumOptimizer, ParetoPoint
import numpy as np

# Create optimizer with reference point
reference = np.array([-0.1, 1.0, -0.1])  # [min_acc, max_complexity, min_diversity]
optimizer = HypervolumOptimizer(reference_point=reference)

# Evaluate solutions
genomes = [np.random.randn(34) for _ in range(100)]
accuracies = np.random.rand(100) * 0.3 + 0.6

# Compute multi-objectives
objectives = []
for i, (genome, acc) in enumerate(zip(genomes, accuracies)):
    # Objective 1: Accuracy (maximize)
    obj_accuracy = acc

    # Objective 2: Complexity (minimize)
    n_flags = np.sum(genome[:16] > 0.5)
    obj_complexity = -float(n_flags) / 16.0

    # Objective 3: Diversity (maximize)
    active_frac = np.sum(genome > 0.5) / len(genome)
    entropy = -active_frac * np.log(active_frac + 1e-8)
    obj_diversity = entropy

    objectives.append(np.array([obj_accuracy, obj_complexity, obj_diversity]))

# Update Pareto front
optimizer.update_front(genomes, objectives)

# Get front statistics
print(f"Front size: {len(optimizer.pareto_front)}")
print(f"Hypervolume: {optimizer.hypervolume_indicator():.4f}")

# Select diverse solutions for evaluation
batch = optimizer.select_batch(batch_size=32)
for point in batch[:3]:
    print(f"  Accuracy={point.objectives[0]:.3f}, "
          f"Complexity={point.objectives[1]:.3f}, "
          f"Diversity={point.objectives[2]:.3f}, "
          f"Crowding distance={point.crowding_distance:.3f}")
```

**Tuning tips**:
- Adjust reference point to emphasize/de-emphasize regions
- Use first K fronts for more solutions
- Monitor hypervolume trend (should increase)
- Crowding distance identifies isolated solutions


## 3. Thompson Sampling Only

```python
from rule_discovery_final_search import ThompsonSamplerBandit
import jax.random as jr
import numpy as np

# Create bandit with mechanism families
bandit = ThompsonSamplerBandit(
    arm_names=[
        "baseline",
        "normalization",
        "gating",
        "surprise",
        "rls",
        "ensemble",
    ],
    key=jr.key(0),
)

# Simulate bandit loop
total_evaluations = 1000
for step in range(total_evaluations):
    # Thompson sample
    key = jr.fold_in(jr.key(0), step)
    selected_arm = bandit.sample_arm(key)

    # Evaluate mechanism family (simulate)
    reward = np.random.beta(2, 2)  # [0, 1]
    success = reward > 0.6

    # Update
    bandit.update_arm(selected_arm, success=success, reward=reward)

# Analyze results
print("Arm statistics:")
for name, arm in bandit.arms.items():
    total = arm.success_count + arm.failure_count
    win_rate = arm.success_count / (total + 1e-8)
    print(f"  {name:15} | pulls={total:3d} wins={arm.success_count:3d} "
          f"rate={win_rate:.2%} mean_reward={arm.mean_reward:.3f}")

# Get budget allocation
allocation = bandit.get_allocation()
print("\nBudget allocation:")
for arm, frac in sorted(allocation.items(), key=lambda x: -x[1]):
    print(f"  {arm:15} {frac:6.1%}")

# Access full history
print(f"\nTotal updates: {len(bandit.history)}")
for entry in bandit.history[-3:]:
    print(f"  Step: arm={entry['arm']}, success={entry['success']}, "
          f"reward={entry['reward']:.3f}")
```

**Tuning tips**:
- Define "success" threshold based on problem
- More arms = slower convergence, more exploration
- Success/failure ratio drives allocation
- Monitor entropy of allocation (should decrease over time)


## 4. Active Learning Curriculum Only

```python
from rule_discovery_final_search import ActiveLearningCurriculum
import numpy as np

# Create curriculum
curriculum = ActiveLearningCurriculum(
    task_names=["M1", "M2", "M3", "M4"],
    seed_pool=[0, 1, 2],  # 12 total (task, seed) pairs
)

# Simulate search with curriculum
total_generations = 100
for generation in range(total_generations):
    # Get curriculum phase (0=easy, 1=hard)
    phase = curriculum.get_curriculum_schedule(generation, total_generations)

    # Select informative (task, seed) pairs
    batch = curriculum.select_batch(batch_size=4, phase=phase)

    # Simulate evaluation of batch
    for task, seed in batch:
        # Simulate accuracies across different genomes
        accuracies = np.random.beta(2, 2, 20) + 0.3

        # Update curriculum informativeness
        curriculum.update_informativeness(task, seed, accuracies)

    if generation % 20 == 0:
        print(f"Gen {generation:3d} (phase={phase:.2f}): selected {batch}")

# Analyze which tasks were most informative
print("\nTask informativeness:")
for key, score in sorted(
    curriculum.informativeness.items(),
    key=lambda x: -x[1].information_gain
):
    print(f"  {key:8} | info_gain={score.information_gain:.4f} "
          f"difficulty={score.difficulty:.4f} "
          f"selected={score.selected_count:3d}")
```

**Tuning tips**:
- Information gain = variance (high = discriminative)
- Difficulty = 1 - accuracy (high = hard)
- Exploration bonus prevents over-sampling (1 / selected_count)
- Phase schedule drives early/late curriculum


# ============================================================================
# CLI EXAMPLES
# ============================================================================

## Example 1: Full Integration (All Strategies)

```bash
python -m rule_discovery_final_search_integration \
  --out results/full_integration.json \
  --total-evaluations 10000 \
  --batch-size 64 \
  --max-generations 100 \
  --use-bayesian \
  --use-hypervolume \
  --use-thompson \
  --use-active-learning \
  --gp-kernel matern \
  --ei-xi 0.01 \
  --eval-seeds 0 1 \
  --holdout-seeds 101 102 103
```

**Expected output**:
```
final search: bayesian=True hypervolume=True thompson=True curriculum=True
initial pool: 21 seeded + 491 random = 512 total
random+seeded phase: best fitness 0.82143 accuracy 0.82500 (...)
generation 0: best fitness 0.82563 (...), hypervolume 0.2341
generation 1: best fitness 0.83105 (...), hypervolume 0.2487
...
final search complete: 6400 evaluated, 2 promoted -> results/full_integration.json
```


## Example 2: Bayesian + Thompson (No Hypervolume)

```bash
python -m rule_discovery_final_search_integration \
  --out results/bayes_thompson.json \
  --total-evaluations 5000 \
  --batch-size 128 \
  --max-generations 50 \
  --use-bayesian \
  --use-thompson \
  --gp-kernel rbf
```


## Example 3: Hypervolume Only (Multi-Objective Baseline)

```bash
python -m rule_discovery_final_search_integration \
  --out results/hypervolume_baseline.json \
  --total-evaluations 10000 \
  --batch-size 64 \
  --max-generations 100 \
  --use-hypervolume
```


## Example 4: Custom Gauss Suite

```bash
python -m rule_discovery_final_search_integration \
  --out results/gauss_final_search.json \
  --total-evaluations 10000 \
  --batch-size 64 \
  --max-generations 100 \
  --use-bayesian \
  --use-hypervolume \
  --suite gauss \
  --micro-n-tasks 16 \
  --micro-task-length 5000
```


# ============================================================================
# OUTPUT ANALYSIS
# ============================================================================

## Result JSON Structure

```python
import json

with open("results/final_search_v1.json") as f:
    result = json.load(f)

# Top-level keys
print(f"Schema: {result['schema']}")
print(f"Evaluations: {result['n_evaluated']}")
print(f"Promoted solutions: {len(result['promoted'])}")
print(f"Wall time: {result['wall_clock_seconds']}s")

# Best candidates
for rank, candidate in enumerate(result['candidates'][:5]):
    print(f"\nRank {rank}:")
    print(f"  Accuracy: {candidate['search_accuracy']:.5f}")
    print(f"  Holdout: {candidate['holdout_accuracy']:.5f}")
    print(f"  Flags: {candidate['active_flags']}")
    print(f"  Description: {candidate['description']}")

# Generation history
for gen in result['generation_log'][-5:]:
    print(f"Gen {gen['generation']}: "
          f"best_acc={gen['best_accuracy']:.4f} "
          f"HV={gen.get('hypervolume', 0):.4f}")

# Strategy configuration used
print(f"\nStrategy config:")
for key, val in result['strategy'].items():
    print(f"  {key}: {val}")
```


# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

## Memory-Efficient Mode (Tight Budget)

```bash
# Smaller batch size, more generations
python -m rule_discovery_final_search_integration \
  --out results/memory_efficient.json \
  --total-evaluations 5000 \
  --batch-size 32 \
  --max-generations 160 \
  --batch-size-eval 64
```


## Speed-Optimized Mode (Large Budget)

```bash
# Larger batch size, fewer generations
python -m rule_discovery_final_search_integration \
  --out results/speed_optimized.json \
  --total-evaluations 20000 \
  --batch-size 256 \
  --max-generations 80 \
  --batch-size-eval 512
```


## Exploration-Heavy Mode (Uncertain Landscape)

```bash
# Higher EI exploration parameter
python -m rule_discovery_final_search_integration \
  --out results/exploration_heavy.json \
  --total-evaluations 10000 \
  --batch-size 64 \
  --ei-xi 0.1 \
  --gp-kernel rbf
```


## Exploitation-Heavy Mode (Good Initial Solutions)

```bash
# Lower EI exploration parameter
python -m rule_discovery_final_search_integration \
  --out results/exploitation_heavy.json \
  --total-evaluations 10000 \
  --batch-size 64 \
  --ei-xi 0.001 \
  --gp-kernel matern
```


# ============================================================================
# COMMON PATTERNS
# ============================================================================

## Pattern 1: Staged Search

```bash
# Stage 1: Exploration (all strategies enabled)
python -m rule_discovery_final_search_integration \
  --out results/stage1_exploration.json \
  --total-evaluations 5000 \
  --batch-size 64 \
  --use-bayesian --use-hypervolume --use-thompson --use-active-learning

# Stage 2: Refinement (Bayesian + Hypervolume)
python -m rule_discovery_final_search_integration \
  --out results/stage2_refinement.json \
  --total-evaluations 5000 \
  --batch-size 128 \
  --use-bayesian --use-hypervolume
```


## Pattern 2: Mechanism-Specific Search

```bash
# Focus on RLS mechanisms
python -m rule_discovery_final_search_integration \
  --out results/rls_focused.json \
  --total-evaluations 10000 \
  --batch-size 64 \
  --use-thompson  # Allocate budget to RLS arm
```


## Pattern 3: Holdout Validation

```bash
# Search on M1+M2+M3 (default)
python -m rule_discovery_final_search_integration \
  --out results/search_m123.json \
  --total-evaluations 10000 \
  --use-bayesian --use-hypervolume

# Validate on M4 + recurrence (already included in --holdout-names)
```


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

## Issue: "RuntimeError: Cholesky decomposition failed"

**Cause**: GP matrix K is ill-conditioned
**Fix**: Increase `noise_variance` or reduce `length_scale`
**Solution**: Use RBF kernel (more stable) or add ridge to K

```python
# Internal: already handled with 1e-4 * I ridge
```

## Issue: "Hypervolume not increasing"

**Cause**: Front is stagnating, reference point too pessimistic
**Fix**: Adjust reference point closer to feasible region
**Solution**: Monitor HV trend and adjust schedule

```python
# More permissive reference point
reference = np.array([-0.2, 1.5, -0.2])
```

## Issue: "Thompson arms not converging"

**Cause**: Reward threshold too high or arms too similar
**Fix**: Tune success threshold or use more diverse arms
**Solution**: Monitor arm history and allocation trend

```python
# Should see good arm convergence after ~100 pulls
```

## Issue: "Curriculum selecting same tasks repeatedly"

**Cause**: Exploration bonus too weak
**Fix**: Increase bonus weight (1 / selected_count) or randomize
**Solution**: Early phases should be more exploratory

```python
# Monitor selected_count for each (task, seed)
```


# ============================================================================
# NEXT STEPS & EXTENSIONS
# ============================================================================

## Extending Bayesian Optimization

```python
# Custom acquisition function
class CustomAcquisition:
    def __call__(self, model, candidates):
        mean, std = model.predict(candidates)
        # Custom formula here
        return acquisition_scores

optimizer.acquisition = CustomAcquisition()
```

## Extending Hypervolume

```python
# Use more fronts (not just first front)
fronts = optimizer.fast_non_dominated_sort(genomes, objectives)
for front in fronts[:3]:  # Keep first 3 fronts
    # Use for selection
```

## Extending Thompson Sampling

```python
# Custom reward model (e.g., Gaussian instead of Beta-Bernoulli)
class GaussianBandit(ThompsonSamplerBandit):
    def __init__(self, ...):
        # Maintain Gaussian posterior instead of Beta
        pass
```

## Extending Active Learning

```python
# Hierarchical curriculum (multiple phases)
class HierarchicalCurriculum(ActiveLearningCurriculum):
    def select_batch(self, batch_size, phase, **kwargs):
        # Multi-level scheduling
        pass
```


# ============================================================================
# REFERENCES
# ============================================================================

## Papers

1. Brochu et al. "A Tutorial on Bayesian Optimization" (2010)
2. Deb et al. "NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm" (2002)
3. Russo et al. "A Tutorial on Thompson Sampling" (2018)
4. Settles "Active Learning" (2009)

## Implementations

- Rule Discovery main: alberta_framework/benchmarks/rule_discovery.py
- Micro suite: alberta_framework/benchmarks/micro_continual.py
- IPMNIST protocol: alberta_framework/benchmarks/ipmnist_screening.py

## Documentation

- RULE_DISCOVERY_FINAL_SEARCH.md: Complete technical documentation
- RULE_DISCOVERY_FINAL_SEARCH_SUMMARY.py: Implementation overview
- This file: Quick reference and examples
"""

# Quick reference guide as module docstring
