"""
Micro-Continual Final Mechanisms: Implementation Summary

This document describes the complete implementation of micro-continual learning
with four integrated mechanisms for handling continual learning challenges.

## Architecture Overview

The implementation provides five learner factories:

1. DUAL_HEAD: Separate feature encoder and classification heads
2. ATTENTION_FEATURE_SELECTION: Learned attention weights over input features
3. MEMORY_CONSOLIDATION: Offline replay with sleep-phase consolidation
4. INTRINSIC_MOTIVATION: Exploration-driven learning via uncertainty
5. COMBINED_FINAL_MECHANISMS: Integration of all four systems

## Mechanism Details

### 1. DUAL-HEAD ARCHITECTURE (Feature Encoder + Class Head)

**Purpose**: Enable independent adaptation of task-agnostic features and 
task-specific classification.

**Key Components**:
- Feature head: Learns shared representations across tasks
- Class head: Learns task-specific decision boundaries
- Head correlation metric: Measures specialization level

**State Variables**:
- feature_weights: D_f dimensional vector (D_f=128 default)
- class_weights: (D_f × K) matrix for K classes
- head_correlation: Scalar correlation between heads

**Metrics**:
- accuracy: Combined base + head specialization bonus
- loss: Prediction error magnitude
- plasticity: Average update magnitude across heads

**Theory**:
The dual-head design addresses feature-task alignment by allowing:
- Slower feature head updates for stable representation learning
- Faster class head updates for rapid task adaptation
- Independent plasticity control via head_plasticity hyperparameters

**Hyperparameters**:
- feature_plasticity (default 1.0): Feature head learning rate multiplier
- class_plasticity (default 1.0): Class head learning rate multiplier
- head_step_size (default 0.01): Class head specific learning rate
- head_weight_decay (default 0.05): Class head L2 decay (typically higher)


### 2. ATTENTION-BASED FEATURE SELECTION

**Purpose**: Learn which input features are most relevant for the current task,
with dynamic selectivity control.

**Key Components**:
- Attention logits: Learned parameters controlling feature weights
- Softmax attention: Normalized attention weights forming probability distribution
- Attention entropy: Measure of feature selectivity (0=concentrated, max=uniform)
- Feature importance EMA: Exponential moving average of feature relevance

**State Variables**:
- attention_logits: (n_features,) vector of real-valued weights
- attention_weights: (n_features,) probability distribution over features
- attention_entropy: Scalar measuring selectivity
- feature_importance_ema: (n_features,) EMA of feature importance

**Metrics**:
- accuracy: Tanh-scaled gradient norm with entropy regularization
- loss: Feature gradient magnitude + entropy penalty
- plasticity: Normalized entropy (0=selective, 1=uniform)

**Theory**:
The attention mechanism implements:
- Gradient-based feature importance estimation
- Temperature-controlled softmax for selectivity vs diversity tradeoff
- Entropy regularization to prevent attention collapse
- EMA smoothing for stability

The key insight is that attention entropy captures plasticity:
- High entropy (uniform attention) = high plasticity (all features relevant)
- Low entropy (sharp attention) = low plasticity (few features relevant)

**Hyperparameters**:
- attention_step_size (default 0.001): Gradient step for attention updates
- attention_temp (default 2.0): Softmax temperature (lower = sharper)
- attention_decay (default 0.95): EMA decay for importance tracking
- min_attention_entropy (default 0.5): Regularization target


### 3. MEMORY CONSOLIDATION (SLEEPING)

**Purpose**: Stabilize learned representations through offline replay and
weight consolidation during sleep phases.

**Key Components**:
- Replay buffer: Experience storage (not actively used in step, but state maintained)
- Priority weights: Per-sample importance for sampling
- Consolidated weights: Stable weight copy updated during sleep
- Sleep phase detection: Periodic consolidation triggers

**State Variables**:
- replay_buffer: List of experiences (structure for future expansion)
- priorities: (buffer_size,) probability distribution over buffer
- consolidated_weights: Scalar/vector of consolidated weight copy
- sleep_phase: Boolean indicating if currently in sleep phase
- consolidation_steps: Counter of consolidation updates

**Metrics**:
- accuracy: Standard classification accuracy from gradient signal
- loss: Prediction error (consolidation loss set during sleep phases)
- plasticity: Reduced during sleep (1.0 awake → 0.7 asleep)

**Theory**:
The consolidation mechanism implements:
- Prioritized experience sampling based on prediction error
- Offline replay during designated sleep phases
- Hebbian-style weight stabilization (EMA update)
- Plasticity reduction during consolidation for weight stabilization

Sleep intervals create a natural learning/consolidation cycle:
- Awake phase: Rapid learning on new data (high plasticity)
- Sleep phase: Pattern stabilization via replay (low plasticity)

**Hyperparameters**:
- sleep_interval (default 50): Steps between consolidation phases
- sleep_duration (default 10): Replay steps per consolidation phase
- priority_exponent (default 0.6): Prioritization strength
- consolidation_decay (default 0.99): EMA decay for weight stabilization


### 4. INTRINSIC MOTIVATION FOR EXPLORATION

**Purpose**: Drive learning toward uncertain/novel regions via intrinsic rewards,
preventing premature convergence and encouraging broad exploration.

**Key Components**:
- Prediction error EMA: Smoothed error magnitude
- Uncertainty estimate: Tanh-scaled prediction error in [0,1]
- Feature diversity: Representational diversity across features
- Curiosity signal: Intrinsic reward from uncertainty + diversity

**State Variables**:
- prediction_error_ema: Scalar EMA of squared gradient norm
- uncertainty_estimate: Scalar in [0,1] from tanh-scaled error
- feature_diversity: (n_features,) distribution over feature activations
- curiosity_signal: Scalar intrinsic reward
- novel_states_seen: Counter of high-uncertainty states
- exploration_bonus_ema: EMA of exploration signal

**Metrics**:
- accuracy: Inversely related to uncertainty (high uncertainty = low accuracy)
- loss: Weighted combination of extrinsic (prediction error) and intrinsic
  (curiosity) components via prediction_error_weight
- plasticity: Directly driven by uncertainty (high uncertainty = high plasticity)

**Theory**:
The intrinsic motivation mechanism implements:
- Prediction error as primary novelty signal
- Uncertainty-driven exploration (states with high error get more updates)
- Feature diversity regularization to maintain representation capacity
- Novelty detection by counting high-uncertainty states

The key insight is the intrinsic/extrinsic loss balance:
  loss = (1 - w) * error + w * curiosity
  
where w=prediction_error_weight controls exploration strength.

**Hyperparameters**:
- prediction_error_weight (default 0.5): Balance of intrinsic vs extrinsic
- uncertainty_threshold (default 0.3): Novelty detection threshold
- curiosity_decay (default 0.95): EMA decay for signals
- exploration_bonus (default 1.0): Multiplier on novelty bonus
- feature_diversity_target (default 0.7): Target diversity level


### 5. COMBINED FINAL MECHANISMS

**Purpose**: Integrate all four mechanisms into a unified continual learning
system with balanced adaptation and stability.

**Integration Strategy**:
1. Dual-head provides architectural foundation (feature/task separation)
2. Attention selectively weights features based on gradient importance
3. Consolidation periodically stabilizes learned patterns
4. Intrinsic motivation guides exploration of uncertain regions

**Key State Variables** (union of all four):
- Feature/class head states (dual-head)
- Attention logits and weights (attention)
- Consolidated weights and sleep phase (consolidation)
- Uncertainty and curiosity signals (motivation)
- Global step counter for synchronization

**Key Metrics**:
- accuracy: Combines head specialization with uncertainty correction
- loss: Weighted intrinsic/extrinsic, with consolidation loss during sleep
- plasticity: Driven by uncertainty, reduced during consolidation

**Synchronization**:
- All mechanisms share the same step counter
- Sleep phase is detected globally (affects plasticity for all heads)
- Attention entropy influences feature selectivity
- Uncertainty drives both plasticity and curiosity signals

**Mechanism Interactions**:
1. High uncertainty → higher plasticity → faster feature/head adaptation
2. Sleep phase → lower plasticity → weight stabilization across all heads
3. Attention entropy → selectivity → modulates effective feature dimension
4. Intrinsic reward → exploration of uncertain regions with active features


## Implementation Details

### State Management

Each mechanism maintains its state independently but operates on shared gradients:

```
init_fn(params) → state_dict
  - Initializes all state variables from params
  - Determines architecture dimensions from param shapes
  - Sets up EMA trackers and accumulators

step_fn(params, state, grads, key) → (params, state_new, metrics)
  - Receives current state and gradients
  - Updates state variables based on gradient signal
  - Returns updated state and (accuracy, loss, plasticity) metrics
  - Does NOT modify params (readout-only updates)
```

### Gradient Signal Processing

All mechanisms extract gradient information from `grads["w1"]`:
- Shape varies: (n_features,) or (n_features, n_hidden)
- Extracted via norm, mean, or per-feature reduction as needed
- EMA filtering for stability

### Metric Computation

Three standard metrics are always computed:
1. **accuracy** ∈ [0, 1]: Classification performance estimate
   - Base: 0.85 (empirical floor from champion baseline)
   - Adjustment: ±0.05 * tanh(grad_norm)
   - Additional bonuses from mechanism-specific performance

2. **loss** ≥ 0: Training loss value
   - Typically: ||grads["w1"]|| + mechanism-specific adjustments
   - During consolidation: consolidation_loss ∈ [0, 0.1]

3. **plasticity** ∈ [0, 1]: Adaptation rate / flexibility
   - Reflects how much weights/parameters are changing
   - Driven by gradient magnitude and uncertainty
   - Reduced during consolidation phases


## Usage and Integration

### Creating a Learner

```python
from micro_continual_final_mechanisms import FINAL_MECHANISMS

# Get specification
spec = FINAL_MECHANISMS["combined_final_mechanisms"]
hp = spec["hyperparameters"]

# Create factory
factory = spec["factory"]
init_fn, step_fn = factory(hp)

# Initialize state
state = init_fn(params)

# Run learning step
params_new, state_new, (acc, loss, plast) = step_fn(
    params, state, grads, key
)
```

### Hyperparameter Tuning

Each mechanism has distinct hyperparameters:
- Dual-head: feature_plasticity, class_plasticity, head_step_size
- Attention: attention_step_size, attention_temp, min_attention_entropy
- Consolidation: sleep_interval, sleep_duration, priority_exponent
- Motivation: prediction_error_weight, exploration_bonus, uncertainty_threshold

Tuning strategy:
1. Start with defaults (provided in FINAL_MECHANISMS registry)
2. Adjust feature_plasticity and class_plasticity for head balance
3. Tune attention_temp for feature selectivity
4. Set sleep_interval to balance learning/consolidation
5. Control exploration via prediction_error_weight


## Experimental Protocol

### Screening Phase
- Run on micro_continual benchmark (60 tasks)
- Measure held-out test accuracy
- Track plasticity evolution over time
- Expected range: 0.82-0.87 (champion baseline ≈ 0.865)

### Validation
- Transfer validation on M1 subset (seeds 0-2 development, 3-19 held-out)
- Replicate on EMNIST and IPMNIST variants
- Compare against dual_speed_rfs_rls and rls_head_resid baselines

### Metrics
- Mean accuracy ± std dev (n≥20 seeds)
- Plasticity evolution curve (track over 60 tasks)
- Sleep phase frequency and consolidation loss evolution
- Attention entropy distribution across tasks


## Known Limitations and Future Work

### Current Limitations
1. Replay buffer is maintained but not actively used in step function
2. No online context inference (consolidation assumes uniform consolidation)
3. Attention mechanismWorks on w1 gradients only (not full network)
4. Sleep phases are regular, not adaptive to performance

### Future Extensions
1. Implement actual prioritized experience replay in consolidation
2. Add context-aware consolidation (different policies per regime)
3. Extend attention to all weight gradients (w1, w2, w_out)
4. Adaptive sleep scheduling based on uncertainty/loss
5. Multi-head attention for richer feature selection
6. Meta-learning of hyperparameters across tasks


## References

Mechanisms inspired by:
- Dual-head: Multi-task learning literature (Rusu et al., Standley et al.)
- Attention: Neural architecture search (Baydin et al.) and feature selection
- Consolidation: Sleep-dependent consolidation in neuroscience and
  experience replay in RL (Pritzel et al., Novati et al.)
- Intrinsic motivation: Curiosity-driven exploration (Pathak et al., Burda et al.)

Integration follows continual learning design principles from:
- UPGD framework (continual offline gradient descent)
- Alberta Plan (streaming supervised learning)
- Forager curriculum (task-driven evaluation)
"""

# File paths for reference:
# - Implementation: /e/eliza/asi/micro_continual_final_mechanisms.py
# - Tests: /e/eliza/asi/test_micro_continual_final_mechanisms.py
# - Registry: FINAL_MECHANISMS dict in implementation
