"""Reduced-horizon mechanism-combination screening for the UPGD IPMNIST lane.

Screens optimizer/mechanism combinations that might exceed the reproduced
UPGD-W SOTA on the ICLR-2024 online Input-permuted MNIST protocol
(:mod:`alberta_framework.benchmarks.upgd_ipmnist`). The screening proxy is
the *same* protocol at a reduced horizon (default 60 tasks x 5,000 steps
instead of 200 tasks): because :func:`~alberta_framework.benchmarks.
upgd_ipmnist.build_schedule` folds the task index into per-seed keys, a
60-task run is an exact prefix of the corresponding 200-task run for the
same seed, so the proxy can be validated bit-for-bit against the completed
full-horizon shards in ``outputs/upgd_ipmnist/partials/``.

Screened combinations (all paired against a ``upgd_w_control`` arm run on
identical seeds):

- ``upgd_idbd`` / ``upgd_autostep``: UPGD's utility gate combined with
  per-weight step-size adaptation (IDBD, Meyer error-free variant; Autostep,
  Mahmood et al. 2012). The meta signal is the *gated* loss gradient — the
  update direction UPGD actually applies (perturbation noise excluded).
- ``upgd_cbp`` / ``adamw_cbp``: dormant-unit recycling in the style of
  Continual Backprop (Dohare et al.), adapted to the protocol MLP: per-unit
  utility EMA of ``|activation * dL/d_activation|``, accumulator-driven
  replacement of the lowest-utility mature unit, incoming weights redrawn
  from the protocol's PyTorch-default uniform init (upstream CBP uses a
  sparse init), outgoing weights zeroed, optimizer state reset per unit.
- ``upgd_w_idbd_swift``: the IDBD arm plus SwiftTD's two supervised-mode
  stabilizers (Javed, Sharifnassab & Sutton, RLC 2024; equation forms from
  :mod:`alberta_framework.core.swift_td`): a network-global overshoot bound
  capping ``sum_i alpha_i * z_i^2`` at ``eta``, and persistent step-size
  decay ``log_alpha_i += ln(eps) * z_i^2`` plus meta-trace reset whenever
  the bound triggers.
- ``upgd_w_fade_head``: FADE-style meta-learned per-parameter weight decay
  (Ramesh, Lewandowski & Schmidhuber, arXiv 2604.27063) on the output layer
  only -- ``lambda_i = exp(gamma_i)`` adapted through a forward-mode
  sensitivity trace of the head weights w.r.t. their log decay rates;
  UPGD-W unchanged elsewhere.
- ``upgd_l2init``: decoupled decay pulls toward the *initial* weights
  (L2-Init, Kumar et al.) instead of toward zero.
- ``upgd_ema_norm``: EMA input normalization (equation-parity with
  :class:`~alberta_framework.core.normalizers.EMANormalizer`) in front of
  the protocol MLP.
- ``upgd_w_wclip_*``: UPGD-W followed by per-layer weight clipping to
  ``[-kappa * s_l, +kappa * s_l]`` where ``s_l = 1/sqrt(fan_in)`` is the
  protocol's uniform-init bound (Elsayed, Lan, Lyle & Mahmood, RLC 2024),
  with ``kappa`` in {1, 2} crossed with weight decay in {0.01, 0}.
- ``upgd_w_localgate``: the lean UPGD-W step with the sigmoid utility gate
  normalized by the *per-tensor* utility max instead of the network-global
  max (the paper's local/global distinction).
- ``upgd_w_*`` hyperparameter-neighborhood star around the published
  UPGD-W configuration (sigma, utility decay, weight decay).
- ``guarded_cbp_adam``: AdamW+CBP (the screening leader) plus UPGD-style
  utility *protection only* — Adam's applied per-weight delta is scaled by
  ``1 - guard_scale * gate`` with the gate from the UPGD ``-w*g`` utility
  EMA; no perturbation (CBP supplies regeneration). ``guard_scale=0``
  reduces bit-exactly to ``adamw_cbp`` (pinned).
- ``adamw_cbp_noreset``: ``adamw_cbp`` WITHOUT the per-unit Adam
  moment/count reset at CBP replacement (the leader resets by default) —
  dissects whether optimizer-state freshness at recycle is load-bearing.
  ``cbp_replacement_rate=0`` reduces to ``adamw_control`` (pinned).
- ``upgd_w_sigma0``: lean UPGD-W with ``sigma=0`` — pure utility-gated
  SGD + decoupled decay, no perturbation; the noise draw (~85-90% of the
  UPGD step cost) is skipped entirely. Bit-exact against the control
  factory run at ``noise_std=0`` (pinned).
- ``upgd_alpha_utility``: UPGD-W whose protection signal is per-weight
  step-size relevance — an IDBD ``log_alpha``/trace pair maintained as a
  *passive statistic* on the raw gradient (never applied as a step size);
  the gate is a scale-free squashing of each weight's log-step-size drift
  from init. ``meta_step_size=0`` reduces to the closed-form half-gated
  step (pinned).
- ``adamw_cbp_{r3e5,r3e4,m50,m200}``: axis-aligned mini-star around the
  untuned ``adamw_cbp`` leader (replacement rate 3e-5/3e-4, maturity
  50/200).
- ``adamw_cbp_ema_norm``: the exact ``adamw_cbp`` update behind the exact
  ``upgd_ema_norm`` EMA input normalizer (same decay/eps/state threading) —
  composition of the screening's two orthogonal wins (input conditioning +
  capacity regeneration). ``norm_enabled=0`` skips the normalizer entirely
  and reduces bit-exactly to ``adamw_cbp`` (pinned).
- ``sgd_ema_norm``: the gate ablation closing the ``upgd_ema_norm`` /
  ``upgd_ema_norm_sigma0`` dissection — plain SGD with decoupled weight decay
  (``w <- w * (1 - lr*wd) - lr * grad``) behind the exact ``upgd_ema_norm``
  EMA input normalizer (same decay/eps/state threading); no utility, no gate,
  no noise. Pinned against a hand-computed trajectory, and the normalizer
  states are pinned bitwise against ``upgd_ema_norm``'s on a shared stream.
- ``sigma0_*``: single-axis frontier extensions on the confirmed
  ``upgd_ema_norm_sigma0`` champion (normalize + utility-gated SGD + decay,
  no noise), all built by one factory whose defaults reduce bit-exactly to
  that champion (pinned): normalizer decay {0.99, 0.9999} and epsilon
  {1e-6, 1e-4} stars (``ema_normalize`` already centers with the EMA mean,
  so the statistics themselves are the unexplored axis), stateless
  per-example RMS normalization of both hidden ReLU layers
  (``sigma0_hidden_norm``; no learnable parameters), utility-gate
  temperature ``sigmoid(beta * scaled_utility)`` with beta {0.5, 2}, and
  the per-tensor (local) gate normalization retested under conditioning
  (``sigma0_localgate``; measured -0.0008 on raw inputs).
- ``colnorm_gate`` / ``muon_gate`` / ``lion_gate``: update-rule family swaps
  under the ``sigma0_ndecay099`` champion's conditioning (EMA input
  normalizer decay 0.99 + the exact UPGD utility gate, no perturbation).
  Only the descent direction changes: per-fan-in-dimension RMS-scaled gated
  SGD (activation conditioning at the weight level), Nesterov momentum +
  Newton-Schulz orthogonalized 2-D updates (Muon), and gated Lion
  (sign of the interpolated momentum at ~0.1x step size, decay 0.05).
- ``rff_rls``: the pre-registered existential control — no backprop, no MLP.
  The champion's EMA input normalizer feeds frozen random Fourier features
  (``sqrt(2/m) * cos(Omega x + b)``, m=1024) into a streaming one-vs-all
  recursive-least-squares readout with forgetting factor 0.999. If a fixed
  random projection + exponential-window RLS matches the deep arms, the
  benchmark measures tracking rather than learning. Sentinel probes fail
  closed (there is no trained protocol MLP to probe).
- ``sgd_ema_norm_d099`` / ``wclip_ema_norm`` / ``fade_head_ema_norm`` /
  ``snr_ema_norm`` / ``l2init_ema_norm``: the reviewer comparison rows —
  the strongest published plasticity mechanisms (per-layer weight clipping,
  Elsayed et al. RLC 2024; FADE meta-learned head decay, arXiv 2604.27063;
  SNR hypothesis-test neuron resets, Farias & Jozefiak arXiv 2410.20098;
  L2-Init, Kumar et al.) re-implemented from their papers and run behind
  the champion's EMA input conditioning (decay 0.99) on a plain-SGD base:
  our conditioning + THEIR mechanism vs our conditioning + our gate, with
  ``sgd_ema_norm_d099`` the mechanism-free floor. Each factory reduces to
  the shared normalized-SGD base when its mechanism constant is inert
  (pinned).
- ``intentional_updates_*``: a batch-size-one supervised protocol extension
  of Intentional Updates (Sharifnassab et al., arXiv:2604.19033v1).  It
  targets a fixed fractional decrease of current-example surprisal using the
  paper's Eq. 5 and RMSProp diagonal direction.  The registered family pins
  a fixed-step mechanism-off control, diagonal-normalization and clipping
  ablations, and a head-only feature-learning control.  This is not the
  paper's RL algorithm or a publication-equivalent reproduction.

Everything here is a development screening diagnostic — never promotable
scientific evidence. Benchmark executions happen through the CLI
(``run`` / ``merge`` / ``validate-proxy``), never inside pytest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import FunctionType, MappingProxyType
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework._strict_json import load_strict_json_object
from alberta_framework.benchmarks.cchain_ipmnist import (
    OFFICIAL_COMMIT as CCHAIN_OFFICIAL_COMMIT,
)
from alberta_framework.benchmarks.cchain_ipmnist import (
    PAPER_REVISION as CCHAIN_PAPER_REVISION,
)
from alberta_framework.benchmarks.cchain_ipmnist import (
    CChainState,
    cchain_host_diagnostics,
    cchain_hyperparameters,
    make_cchain_learner,
)
from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    PAPER_REVISION as NOISE_CURVATURE_PAPER_REVISION,
)
from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    NoiseCurvatureConfig,
    NoiseCurvatureState,
    init_noise_curvature_state,
    noise_curvature_persistent_bytes,
    noise_curvature_step,
)
from alberta_framework.benchmarks.replay_frozen_ipmnist import (
    PROL_COMMIT,
    PROL_PAPER_REVISION,
    RANDUMB_COMMIT,
    RANDUMB_PAPER_REVISION,
    RANPAC_COMMIT,
    RANPAC_PAPER_REVISION,
    REPLAY_OFFICIAL_CODE,
    REPLAY_PAPER_REVISION,
    frozen_hyperparameters,
    make_frozen_feature_learner,
    make_replay_context_learner,
    replay_hyperparameters,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    _PLASTICITY_LOSS_FLOOR,
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    PARTIAL_SCHEMA_V1,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    LeanUPGDState,
    LearnerInitFn,
    LearnerStepFn,
    _make_adamw_learner,
    _make_upgd_w_learner,
    _preflight_new_output,
    _sorted_param_shapes,
    _split_flat_noise,
    _validated_partial_payload,
    atomic_write_new,
    build_schedule,
    cross_entropy_loss,
    default_openml_data_home,
    init_mlp_params,
    lean_upgd_w_update,
    load_mnist_train,
    mlp_logits,
    validated_ipmnist_data,
)
from alberta_framework.core.adamo import AdamO, AdamOConfig, isometry_gradient
from alberta_framework.core.baseline_optimizers import Adam
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    select_transaction,
)
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    COMPARISON_ID as BOUNDED_ELASTIC_COMPARISON_ID,
)
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    PAPER_REVISION as BOUNDED_ELASTIC_PAPER_REVISION,
)
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    PAPER_SOURCE_SHA256 as BOUNDED_ELASTIC_PAPER_SOURCE_SHA256,
)
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    RESULT_SCHEMA as BOUNDED_ELASTIC_RESULT_SCHEMA,
)
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    bounded_elastic_resource_expectations,
    registered_bounded_elastic_hyperparameters,
    validate_bounded_elastic_development_result,
)
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    ADAPTATION_ID as CCHAIN_ADAPTATION_ID,
)
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    COMPARABILITY_GAPS as CCHAIN_COMPARABILITY_GAPS,
)
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    COMPARISON_ID as CCHAIN_COMPARISON_ID,
)
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    DEVELOPMENT_SEEDS as CCHAIN_DEVELOPMENT_SEEDS,
)
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    RESULT_SCHEMA as CCHAIN_RESULT_SCHEMA,
)
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    validate_cchain_development_result,
)
from alberta_framework.evaluation.l2er_ipmnist_nonpromoting import (
    COMPARISON_ID as L2ER_COMPARISON_ID,
)
from alberta_framework.evaluation.l2er_ipmnist_nonpromoting import (
    OFFICIAL_COMMIT as L2ER_OFFICIAL_COMMIT,
)
from alberta_framework.evaluation.l2er_ipmnist_nonpromoting import (
    PAPER_REVISION as L2ER_PAPER_REVISION,
)
from alberta_framework.evaluation.l2er_ipmnist_nonpromoting import (
    RESULT_SCHEMA as L2ER_RESULT_SCHEMA,
)
from alberta_framework.evaluation.l2er_ipmnist_nonpromoting import (
    validate_l2er_development_result,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    COMPARISON_ID as NOISE_CURVATURE_COMPARISON_ID,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    DEVELOPMENT_SEEDS as NOISE_CURVATURE_DEVELOPMENT_SEEDS,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    LIVE_CONTROL as NOISE_CURVATURE_LIVE_CONTROL,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    OFFICIAL_CODE_STATUS as NOISE_CURVATURE_OFFICIAL_CODE_STATUS,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    PROTOCOL_DIFFERENCES as NOISE_CURVATURE_PROTOCOL_DIFFERENCES,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    RESULT_SCHEMA as NOISE_CURVATURE_RESULT_SCHEMA,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    registered_arms as noise_curvature_registered_arms,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    registered_hyperparameters as noise_curvature_registered_hyperparameters,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    validate_noise_curvature_development_result,
)
from alberta_framework.evaluation.recurring_ipmnist_retention import (
    RecurringIPMNISTPhase,
    RecurringIPMNISTProtocol,
    RecurringIPMNISTRetentionReport,
    RecurringIPMNISTTrace,
    SentinelProbeBinding,
    SentinelProbeSnapshot,
    build_recurring_ipmnist_retention_report,
)
from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    COMPARISON_ID as REPLAY_FROZEN_COMPARISON_ID,
)
from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    PROTOCOL_GAPS as REPLAY_FROZEN_PROTOCOL_GAPS,
)
from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    RESULT_SCHEMA as REPLAY_FROZEN_RESULT_SCHEMA,
)
from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    expected_resources_for_result,
    validate_replay_frozen_result,
)

logger = logging.getLogger(__name__)

LEGACY_SHARD_SCHEMA = "alberta.ipmnist_screening.shard.v1"
SHARD_SCHEMA = "alberta.ipmnist_screening.shard.v2"
LEGACY_SUMMARY_SCHEMA = "alberta.ipmnist_screening.summary.v1"
SUMMARY_SCHEMA = "alberta.ipmnist_screening.summary.v2"
LEGACY_VALIDATION_SCHEMA = "alberta.ipmnist_screening.proxy_validation.v1"
VALIDATION_SCHEMA = "alberta.ipmnist_screening.proxy_validation.v2"
SOURCE_PROVENANCE_SCHEMA = "alberta.ipmnist_screening.source_provenance.v1"
DATASET_PROVENANCE_SCHEMA = "alberta.ipmnist_screening.dataset_provenance.v1"
RUNTIME_SCHEMA = "alberta.ipmnist_screening.runtime.v1"
PARTIAL_RESET_RECORD_SCHEMA = "asi.ipmnist.calibrated_partial_reset.development.v1"
CPR_PAPER_REVISION = "arXiv:2607.24996v1"
CPR_OFFICIAL_CODE_REVISION = (
    "LucMc/continual-learning@6fc2af34783159f5dda50c6915dda32c2d443604"
)
INTENTIONAL_UPDATES_RECORD_SCHEMA = "asi.ipmnist.intentional_updates.development.v1"
INTENTIONAL_UPDATES_PAPER_REVISION = "arXiv:2604.19033v1"
INTENTIONAL_UPDATES_CODE_REVISION = (
    "sharifnassab/Intentional_RL@e86e26fd8613ac212e9a52c3fed8a01d0a31f685"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_SCOPE = ("alberta_framework", "pyproject.toml", "uv.lock")
_SOURCE_SCOPE_LABEL = "tracked:alberta_framework/**,pyproject.toml,uv.lock"
_RUNTIME_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "XLA_FLAGS",
)
_DATASET_SOURCE = {
    "provider": "openml",
    "name": "mnist_784",
    "version": 1,
    "row_start": 0,
    "row_stop_exclusive": 60_000,
}
_DATASET_MATERIALIZATION = "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1"

#: Default reduced-horizon proxy: 60 tasks x 5,000 steps. At this horizon the
#: completed 10-seed full runs separate UPGD-W from AdamW by ~+0.022 average
#: online accuracy with every seed ordered correctly.
PROXY_N_TASKS = 60

#: Paired proxy improvement over the UPGD-W control above which a config is
#: flagged as a full-protocol confirmation candidate.
CONFIRMATION_THRESHOLD = 0.005

NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_screening_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
}

_IDBD_LOG_ALPHA_MIN = -10.0
_IDBD_LOG_ALPHA_MAX = 0.0  # alpha <= 1 keeps per-weight decay factors positive
_AUTOSTEP_ALPHA_MIN = 1e-8
_AUTOSTEP_ALPHA_MAX = 1.0
_MISSING_NOISE_POOL_STEPS = object()


def _validated_wall_clock_seconds(value: object, path: Path | str) -> float:
    """Return one shard wall clock as a finite, non-negative Python float."""
    message = f"{path}: wall_clock_seconds must be a finite, non-negative number"
    if type(value) is not int and type(value) is not float:
        raise ValueError(message)
    numeric_value = value
    try:
        wall_clock_seconds = float(numeric_value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(wall_clock_seconds) or wall_clock_seconds < 0.0:
        raise ValueError(message)
    return wall_clock_seconds


def _finite_wall_clock_total(values: Sequence[float], *, context: str) -> float:
    """Preserve the existing sum order while refusing float overflow."""
    try:
        total = float(sum(values))
    except OverflowError as exc:
        raise ValueError(f"{context}: wall_clock_seconds_total must be finite") from exc
    if not math.isfinite(total):
        raise ValueError(f"{context}: wall_clock_seconds_total must be finite")
    return total


def _clip_finite_log_alpha(log_alpha: Array, meta_delta: Array) -> Array:
    """Clip an IDBD log-step-size update, skipping non-finite channels."""
    return jnp.where(
        jnp.isfinite(meta_delta),
        jnp.clip(log_alpha + meta_delta, _IDBD_LOG_ALPHA_MIN, _IDBD_LOG_ALPHA_MAX),
        log_alpha,
    )


# Metrics returned by every screening step: (accuracy, loss, plasticity).
StepMetrics = tuple[Array, Array, Array]
ScreeningStepFn = Callable[
    [dict[str, Array], Any, Array, Array, Array],
    tuple[dict[str, Array], Any, StepMetrics],
]
#: Pure noise-consuming update ``(params, state, grads, noise, hp)`` used by
#: the pool-noise confirmation path (only lean-UPGD-family arms provide one).
NoiseUpdateFn = Callable[
    [dict[str, Array], Any, dict[str, Array], dict[str, Array], Mapping[str, float]],
    tuple[dict[str, Array], Any],
]
FrozenProbeInputFn = Callable[[Any, Array, Mapping[str, float]], Array]


def _lean_upgd_noise_update(
    params: dict[str, Array],
    state: Any,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hyperparameters: Mapping[str, float],
) -> tuple[dict[str, Array], Any]:
    """Adapt the concrete lean-UPGD update to the registry callable contract."""

    return lean_upgd_w_update(
        params,
        state,
        grads,
        noise,
        dict(hyperparameters),
    )


# =============================================================================
# Shared pieces
# =============================================================================


def _sorted_flat_noise(
    key: Array, params: dict[str, Array], noise_std: float
) -> dict[str, Array]:
    """Draw one flat N(0, sigma^2) vector and slice it per parameter.

    Identical construction (sorted names, one flat draw) to the lean UPGD-W
    learner in :mod:`alberta_framework.benchmarks.upgd_ipmnist`, so a combo
    that degenerates to plain UPGD-W consumes the same noise stream.

    ``noise_std`` is a trace-time Python float; at exactly ``0.0`` the draw
    is skipped and exact zeros are returned.  The values are identical either
    way (``0.0 * normal`` is exact zeros), but XLA cannot fold the draw away
    itself (a normal draw is not provably finite), so a factory that keeps
    the draw and one that shortcuts it compile to structurally different
    graphs whose fusions can reassociate derived float32 metrics by ~1 ulp
    on some backends even when every parameter update is bit-identical
    (issue #46).  The shortcut lives here, not in the callers, so every
    sigma=0 factory sharing this helper lowers to the same RNG-free graph
    while still skipping the draw's ~85-90% share of the UPGD step cost.
    """
    if noise_std == 0.0:
        return {name: jnp.zeros_like(value) for name, value in params.items()}
    names = sorted(params)
    shapes = [params[name].shape for name in names]
    counts = [int(np.prod(shape)) for shape in shapes]
    flat = jr.normal(key, (sum(counts),), jnp.float32) * noise_std
    chunks = jnp.split(flat, np.cumsum(counts)[:-1])
    return {
        name: chunk.reshape(shape)
        for name, chunk, shape in zip(names, chunks, shapes, strict=True)
    }


def _upgd_utility_and_gate(
    params: dict[str, Array],
    grads: dict[str, Array],
    utility: dict[str, Array],
    count: Array,
    utility_decay: float,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """UPGD utility EMA update + global-max sigmoid gate (lean-step equations)."""
    beta = utility_decay
    new_utility = {
        name: beta * utility[name] + (1.0 - beta) * (-grads[name] * params[name])
        for name in params
    }
    global_max = jnp.max(jnp.stack([jnp.max(new_utility[name]) for name in sorted(params)]))
    bias_correction = 1.0 - jnp.power(
        jnp.asarray(beta, dtype=jnp.float32), count.astype(jnp.float32)
    )
    gate = {
        name: jax.nn.sigmoid((new_utility[name] / bias_correction) / global_max)
        for name in params
    }
    return new_utility, gate


def _forward_with_activations(
    params: dict[str, Array], x: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """Forward pass returning ``(logits, z1, a1, z2, a2)`` for CBP bookkeeping."""
    z1 = x @ params["w1"] + params["b1"]
    a1 = jax.nn.relu(z1)
    z2 = a1 @ params["w2"] + params["b2"]
    a2 = jax.nn.relu(z2)
    logits = a2 @ params["w3"] + params["b3"]
    return logits, z1, a1, z2, a2


def _activation_loss_grads(
    params: dict[str, Array], logits: Array, y: Array, z2: Array
) -> tuple[Array, Array]:
    """Analytic ``(dL/da1, dL/da2)`` for softmax cross-entropy on one example."""
    dlogits = jax.nn.softmax(logits) - jax.nn.one_hot(y, logits.shape[0], dtype=jnp.float32)
    da2 = params["w3"] @ dlogits
    da1 = params["w2"] @ (da2 * (z2 > 0).astype(jnp.float32))
    return da1, da2


def _step_metrics(
    params_after: dict[str, Array], x: Array, y: Array, loss: Array, logits: Array
) -> StepMetrics:
    """Protocol metrics: pre-update accuracy, loss, post-update plasticity."""
    accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
    loss_after, _ = cross_entropy_loss(params_after, x, y)
    plasticity = jnp.clip(
        1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
    )
    return accuracy, loss, plasticity


# =============================================================================
# Adapted L2-ER comparator (Prakash et al., ICML 2026)
# =============================================================================

_L2ER_ARRAY_ELEMENTS = 1_000_000
_L2ER_MAX_WORKING_BYTES = 256 * 1024 * 1024
_L2ER_KEYS = frozenset(("w1", "b1", "w2", "b2", "w3", "b3"))
_L2ER_HP_KEYS = frozenset(
    (
        "step_size",
        "weight_decay",
        "er_step_size",
        "er_batch_size",
        "er_steps_per_batch",
        "er_epsilon",
        "er_enabled",
    )
)


def _l2er_array(value: object, *, name: str, ndim: int | None = None) -> Array:
    actual_type = type(value)
    if actual_type is not np.ndarray and not issubclass(
        actual_type, (jax.Array, jax.core.Tracer)
    ):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    array = jnp.asarray(value)
    if (
        array.size < 1
        or array.size > _L2ER_ARRAY_ELEMENTS
        or not jnp.issubdtype(array.dtype, jnp.floating)
        or (ndim is not None and array.ndim != ndim)
    ):
        raise ValueError(f"{name} must be a bounded float array with the required rank")
    return array


def _l2er_hp(hp: object) -> dict[str, float]:
    if type(hp) is not dict or frozenset(hp) != _L2ER_HP_KEYS:
        raise ValueError("L2-ER hyperparameters must be one exact registered object")
    checked: dict[str, float] = {}
    for name in _L2ER_HP_KEYS:
        value = hp[name]
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"L2-ER hyperparameter {name} must be a finite float")
        checked[name] = value
    if (
        checked["er_batch_size"] != 100.0
        or checked["er_steps_per_batch"] != 1.0
        or checked["er_epsilon"] != 1e-8
        or checked["er_enabled"] not in (0.0, 1.0)
    ):
        raise ValueError("L2-ER hyperparameters do not match the audited protocol")
    return checked


def _l2er_preflight_svd(features: Array) -> None:
    smaller = min(features.shape)
    if smaller and smaller > (2**31 - 1) // smaller:
        raise ValueError("L2-ER SVD working elements exceed signed int32")
    if smaller * smaller * features.dtype.itemsize > _L2ER_MAX_WORKING_BYTES:
        raise ValueError("L2-ER SVD working set exceeds 256 MiB")


def _l2er_params(params: object) -> dict[str, Array]:
    if type(params) is not dict or frozenset(params) != _L2ER_KEYS:
        raise ValueError("params must be one exact protocol MLP tree")
    checked = {
        name: _l2er_array(value, name=f"params.{name}")
        for name, value in params.items()
    }
    w1, b1 = checked["w1"], checked["b1"]
    w2, b2 = checked["w2"], checked["b2"]
    w3, b3 = checked["w3"], checked["b3"]
    if (
        w1.ndim != 2
        or b1.shape != (w1.shape[1],)
        or w2.ndim != 2
        or w2.shape[0] != w1.shape[1]
        or b2.shape != (w2.shape[1],)
        or w3.ndim != 2
        or w3.shape[0] != w2.shape[1]
        or b3.shape != (w3.shape[1],)
        or any(value.dtype != jnp.dtype(jnp.float32) for value in checked.values())
    ):
        raise ValueError("params must match the float32 protocol MLP shapes")
    return checked


@chex.dataclass(frozen=True, mappable_dataclass=False)
class L2ERState:
    """The charged 100-example ER buffer and its next insertion index."""

    example_buffer: Array
    buffer_count: Array
    transaction_valid: Array


def l2er_effective_rank_transaction(
    features: Array, epsilon: float = 1e-8
) -> tuple[Array, Array]:
    """Return the scale-stable entropy rank and caller-visible validity."""
    features = _l2er_array(features, name="features", ndim=2)
    if type(epsilon) is not float or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be an exact finite positive float")
    _l2er_preflight_svd(features)
    finite = jnp.all(jnp.isfinite(features))
    safe_features = jnp.where(finite, features, jnp.zeros_like(features))
    scale = jnp.max(jnp.abs(safe_features))
    _, exponent = jnp.frexp(scale)
    scaled_features = jnp.ldexp(safe_features, -exponent)
    singular_values = jnp.abs(jnp.linalg.svdvals(scaled_features.T))
    total = jnp.sum(singular_values)
    probabilities = singular_values / jnp.maximum(total, epsilon)
    entropy = -jnp.sum(probabilities * jnp.log(probabilities + epsilon))
    candidate = jnp.exp(entropy)
    valid = finite & jnp.all(jnp.isfinite(singular_values)) & jnp.isfinite(candidate)
    return jnp.where(valid, candidate, jnp.zeros_like(candidate)), valid


def l2er_effective_rank(features: Array, epsilon: float = 1e-8) -> Array:
    """Official entropy effective-rank estimator with explicit invalidity."""
    candidate, valid = l2er_effective_rank_transaction(features, epsilon)
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, candidate, jnp.full_like(candidate, jnp.nan))
    if not bool(valid):
        raise ValueError("effective rank must be finite")
    return candidate


def l2er_effective_rank_loss(
    params: dict[str, Array], examples: Array, epsilon: float = 1e-8
) -> Array:
    """Negative mean effective rank over the current MLP's hidden layers."""
    examples = _l2er_array(examples, name="examples", ndim=2)
    checked_params = _l2er_params(params)
    if examples.shape[1] != checked_params["w1"].shape[0]:
        raise ValueError("examples and first-layer parameter shapes must match")
    for width in (checked_params["w1"].shape[1], checked_params["w2"].shape[1]):
        if examples.shape[0] > _L2ER_ARRAY_ELEMENTS // width:
            raise ValueError("L2-ER activation allocation exceeds the element limit")
    finite = jnp.asarray(True, dtype=jnp.bool_)
    for value in checked_params.values():
        finite = finite & jnp.all(jnp.isfinite(value))
    if not isinstance(finite, jax.core.Tracer) and not bool(finite):
        raise ValueError("params must contain only finite values")
    checked_params = {
        name: jnp.where(finite, value, jnp.zeros_like(value))
        for name, value in checked_params.items()
    }
    hidden1 = jax.nn.relu(examples @ checked_params["w1"] + checked_params["b1"])
    hidden2 = jax.nn.relu(hidden1 @ checked_params["w2"] + checked_params["b2"])
    return -jnp.mean(
        jnp.stack(
            (
                l2er_effective_rank(hidden1, epsilon),
                l2er_effective_rank(hidden2, epsilon),
            )
        )
    )


def l2er_update(
    params: dict[str, Array],
    state: L2ERState,
    grads: dict[str, Array],
    example: Array,
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], L2ERState]:
    """One supervised SGD/L2 step and the scheduled separate ER step.

    This restates ``lop-jax``'s Permuted-MNIST ordering: each ordinary update
    first uses ``grad + weight_decay * parameter``; after each full ER block,
    one gradient step minimizes negative mean hidden effective rank.  The ER
    step does not advance or alter the ordinary optimizer state.
    """
    checked_hp = _l2er_hp(hp)
    if type(state) is not L2ERState:
        raise ValueError("state must be an exact L2ERState")
    if type(params) is not dict or type(grads) is not dict:
        raise ValueError("params and grads must be exact dicts")
    if frozenset(params) != _L2ER_KEYS or frozenset(grads) != _L2ER_KEYS:
        raise ValueError("params and grads must contain the exact protocol MLP keys")
    example = _l2er_array(example, name="example", ndim=1)
    checked_params = _l2er_params(params)
    checked_grads = {
        name: _l2er_array(value, name=f"grads.{name}") for name, value in grads.items()
    }
    if any(
        checked_grads[name].shape != value.shape
        or checked_grads[name].dtype != value.dtype
        for name, value in checked_params.items()
    ):
        raise ValueError("params and grads must have identical shapes and dtypes")
    w1 = checked_params["w1"]
    if w1.ndim != 2 or example.shape != (w1.shape[0],):
        raise ValueError("example and first-layer parameter shapes must match")
    batch_size = int(checked_hp["er_batch_size"])
    buffer = _l2er_array(state.example_buffer, name="state.example_buffer", ndim=2)
    if buffer.shape != (batch_size, example.shape[0]) or buffer.dtype != example.dtype:
        raise ValueError("L2-ER buffer shape and dtype must match the protocol")
    count = state.buffer_count
    if not isinstance(count, (jax.Array, jax.core.Tracer)) or (
        count.shape != () or count.dtype != jnp.dtype(jnp.int32)
    ):
        raise ValueError("buffer_count must be one scalar int32 JAX array")
    transaction_valid = state.transaction_valid
    if not isinstance(transaction_valid, (jax.Array, jax.core.Tracer)) or (
        transaction_valid.shape != () or transaction_valid.dtype != jnp.dtype(jnp.bool_)
    ):
        raise ValueError("transaction_valid must be one scalar bool JAX array")
    step_size = checked_hp["step_size"]
    weight_decay = checked_hp["weight_decay"]
    supervised_params = {
        name: value - step_size * (checked_grads[name] + weight_decay * value)
        for name, value in checked_params.items()
    }
    count_valid = (count >= 0) & (count < batch_size)
    safe_count = jnp.clip(count, 0, batch_size - 1)
    buffered = buffer.at[safe_count].set(example)
    next_count = safe_count + jnp.asarray(1, dtype=jnp.int32)
    complete = next_count == batch_size

    if checked_hp["er_enabled"] == 1.0:
        er_step_size = checked_hp["er_step_size"]
        epsilon = checked_hp["er_epsilon"]

        def apply_er(operand: tuple[dict[str, Array], Array]) -> dict[str, Array]:
            current_params, batch = operand
            er_grads = jax.grad(l2er_effective_rank_loss)(current_params, batch, epsilon)
            return {
                name: value - er_step_size * er_grads[name]
                for name, value in current_params.items()
            }

        new_params = jax.lax.cond(
            complete,
            apply_er,
            lambda operand: operand[0],
            (supervised_params, buffered),
        )
    else:
        new_params = supervised_params
    candidate_state = L2ERState(  # type: ignore[call-arg]
        example_buffer=jax.lax.cond(
            complete, jnp.zeros_like, lambda value: value, buffered
        ),
        buffer_count=jnp.where(complete, jnp.asarray(0, dtype=jnp.int32), next_count),
        transaction_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    valid = (
        transaction_valid
        & count_valid
        & floating_tree_is_finite(params)
        & floating_tree_is_finite(grads)
    )
    valid = valid & floating_tree_is_finite(new_params) & floating_tree_is_finite(candidate_state)
    safe_prior_params = jax.tree.map(
        lambda value: jnp.where(jnp.isfinite(value), value, jnp.zeros_like(value)),
        checked_params,
    )
    safe_prior_state = L2ERState(  # type: ignore[call-arg]
        example_buffer=jnp.where(jnp.isfinite(buffer), buffer, jnp.zeros_like(buffer)),
        buffer_count=jnp.where(count_valid, count, jnp.asarray(0, dtype=jnp.int32)),
        transaction_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    return (
        select_transaction(valid, new_params, safe_prior_params),
        select_transaction(valid, candidate_state, safe_prior_state),
    )


def _make_l2er_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Build one registered L2/ER reduction on the current IPMNIST MLP."""
    checked_hp = _l2er_hp(hp)

    def init_fn(params: dict[str, Array]) -> L2ERState:
        checked_params = _l2er_params(params)
        input_dim = checked_params["w1"].shape[0]
        if input_dim > _L2ER_ARRAY_ELEMENTS // int(checked_hp["er_batch_size"]):
            raise ValueError("L2-ER buffer exceeds the 1000000-element limit")
        return L2ERState(  # type: ignore[call-arg]
            example_buffer=jnp.zeros(
                (int(checked_hp["er_batch_size"]), input_dim), dtype=jnp.float32
            ),
            buffer_count=jnp.asarray(0, dtype=jnp.int32),
            transaction_valid=jnp.asarray(True, dtype=jnp.bool_),
        )

    def full_step(
        params: dict[str, Array], state: L2ERState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], L2ERState, StepMetrics]:
        del key
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        new_params, new_state = l2er_update(params, state, grads, x, checked_hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# Bounded growing/elastic adaptation (Kong & Sutton, arXiv:2608.01475v1)
# =============================================================================


@chex.dataclass(frozen=True, mappable_dataclass=False)
class BoundedStructureState:
    """Preallocated hidden-1 activity, online activation evidence, and clock."""

    active1: Array
    activation_sum1: Array
    step: Array


def bounded_masked_logits(
    params: dict[str, Array], x: Array, active1: Array
) -> tuple[Array, Array]:
    """Return logits and hidden-1 activations under the static capacity mask."""
    hidden1 = jax.nn.relu(x @ params["w1"] + params["b1"]) * active1
    hidden2 = jax.nn.relu(hidden1 @ params["w2"] + params["b2"])
    return hidden2 @ params["w3"] + params["b3"], hidden1


def bounded_masked_loss(
    params: dict[str, Array], x: Array, y: Array, active1: Array
) -> tuple[Array, tuple[Array, Array]]:
    """Cross-entropy with auxiliary masked logits and activations."""
    logits, hidden1 = bounded_masked_logits(params, x, active1)
    return -jax.nn.log_softmax(logits)[y], (logits, hidden1)


def _fresh_hidden1_slot(
    params: dict[str, Array], index: Array, key: Array, apply: Array
) -> dict[str, Array]:
    """Freshly initialize one hidden-1 slot using the runner's MLP convention."""
    key_in, key_bias, key_out = jr.split(key, 3)
    in_bound = 1.0 / math.sqrt(params["w1"].shape[0])
    out_bound = 1.0 / math.sqrt(params["w2"].shape[0])
    fresh_in = jr.uniform(key_in, (params["w1"].shape[0],), jnp.float32, -in_bound, in_bound)
    fresh_bias = jr.uniform(key_bias, (), jnp.float32, -in_bound, in_bound)
    fresh_out = jr.uniform(key_out, (params["w2"].shape[1],), jnp.float32, -out_bound, out_bound)
    result = dict(params)
    result["w1"] = params["w1"].at[:, index].set(jnp.where(apply, fresh_in, params["w1"][:, index]))
    result["b1"] = params["b1"].at[index].set(jnp.where(apply, fresh_bias, params["b1"][index]))
    result["w2"] = params["w2"].at[index, :].set(
        jnp.where(apply, fresh_out, params["w2"][index, :])
    )
    return result


def bounded_structure_event(
    params: dict[str, Array], state: BoundedStructureState, key: Array, hp: Mapping[str, float]
) -> tuple[dict[str, Array], BoundedStructureState]:
    """Apply one protocol boundary event: optional least-active prune, then growth."""
    active = state.active1
    if hp["pruning_enabled"] == 1.0:
        prune_index = jnp.argmin(jnp.where(active, state.activation_sum1, jnp.inf))
        can_prune = jnp.any(active)
        active = active.at[prune_index].set(
            jnp.where(can_prune, jnp.asarray(False), active[prune_index])
        )
    if hp["growth_enabled"] == 1.0:
        inactive = jnp.logical_not(active)
        grow_index = jnp.argmax(inactive).astype(jnp.int32)
        can_grow = jnp.any(inactive)
        params = _fresh_hidden1_slot(params, grow_index, key, can_grow)
        active = active.at[grow_index].set(
            jnp.where(can_grow, jnp.asarray(True), active[grow_index])
        )
    return params, BoundedStructureState(  # type: ignore[call-arg]
        active1=active,
        activation_sum1=jnp.zeros_like(state.activation_sum1),
        step=state.step,
    )


def bounded_structure_update(
    params: dict[str, Array],
    state: BoundedStructureState,
    grads: dict[str, Array],
    hidden1: Array,
    key: Array,
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], BoundedStructureState]:
    """One SGD update followed by the registered fixed-length boundary event."""
    candidate = {name: value - hp["step_size"] * grads[name] for name, value in params.items()}
    next_step = state.step + jnp.asarray(1, dtype=jnp.int32)
    accumulated = BoundedStructureState(  # type: ignore[call-arg]
        active1=state.active1,
        activation_sum1=state.activation_sum1 + jnp.abs(hidden1),
        step=next_step,
    )
    interval = int(hp["structure_interval"])
    due = next_step % interval == 0
    result = jax.lax.cond(
        due,
        lambda operand: bounded_structure_event(operand[0], operand[1], key, hp),
        lambda operand: operand,
        (candidate, accumulated),
    )
    return cast(tuple[dict[str, Array], BoundedStructureState], result)


def _make_bounded_structure_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Build the masked, preallocated adaptation used by all structure arms."""
    if dict(hp) not in (
        registered_bounded_elastic_hyperparameters("bounded_structure_off"),
        registered_bounded_elastic_hyperparameters("bounded_growth"),
        registered_bounded_elastic_hyperparameters("bounded_elastic"),
    ):
        raise ValueError("bounded structure hyperparameters are not a registered arm")

    def init_fn(params: dict[str, Array]) -> BoundedStructureState:
        width = params["w1"].shape[1]
        active_count = max(1, int(width * hp["initial_active_fraction"]))
        active = jnp.arange(width, dtype=jnp.int32) < active_count
        return BoundedStructureState(  # type: ignore[call-arg]
            active1=active,
            activation_sum1=jnp.zeros(width, dtype=jnp.float32),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array], state: BoundedStructureState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], BoundedStructureState, StepMetrics]:
        (loss, (logits, hidden1)), grads = jax.value_and_grad(bounded_masked_loss, has_aux=True)(
            params, x, y, state.active1
        )
        new_params, new_state = bounded_structure_update(params, state, grads, hidden1, key, hp)
        loss_after, _ = bounded_masked_loss(new_params, x, y, new_state.active1)
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
        )
        return new_params, new_state, (accuracy, loss, plasticity)

    return init_fn, full_step


_NOISE_CURVATURE_MODE_NAMES = {
    0.0: "fixed",
    1.0: "gradient_only",
    2.0: "volatility_only",
    3.0: "combined",
}


def _make_noise_curvature_learner(
    hp: Mapping[str, float], *, total_steps: int = 1_000_000
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Adapt arXiv:2509.19698v3 to the current online IPMNIST MLP."""

    if type(total_steps) is not int or not 1 <= total_steps <= (1 << 31) - 1:
        raise ValueError("total_steps must be an exact positive signed-int32 integer")
    if type(hp) is not dict:
        raise ValueError("noise-curvature hyperparameters must be an exact object")
    mode_value = hp.get("controller_mode")
    if type(mode_value) is not float or mode_value not in _NOISE_CURVATURE_MODE_NAMES:
        raise ValueError("controller_mode must identify one registered scheduler arm")
    mode = _NOISE_CURVATURE_MODE_NAMES[mode_value]
    arm = noise_curvature_registered_arms()[int(mode_value)]
    expected = noise_curvature_registered_hyperparameters(arm)
    if hp != expected or any(type(value) is not float for value in hp.values()):
        raise ValueError("hyperparameters do not match the registered scheduler arm")
    config = NoiseCurvatureConfig(
        mode=mode,  # type: ignore[arg-type]
        total_steps=total_steps,
        control_interval=int(hp["control_interval"]),
        power_iterations=int(hp["power_iterations"]),
        base_step_size=hp["step_size"],
        beta1=hp["beta1"],
        beta2=hp["beta2"],
        adam_epsilon=hp["eps"],
        weight_decay=hp["weight_decay"],
        ema_decay=hp["ema_decay"],
        volatility_epsilon=hp["volatility_epsilon"],
        volatility_inflation=hp["volatility_inflation"],
        volatility_kappa=hp["volatility_kappa"],
        safety_factor=hp["safety_factor"],
        cool_rate=hp["cool_rate"],
        warm_rate=hp["warm_rate"],
        warm_fraction=hp["warm_fraction"],
        timid_fraction=hp["timid_fraction"],
        effective_step_floor=hp["effective_step_floor"],
    )

    def init_fn(params: dict[str, Array]) -> Any:
        return init_noise_curvature_state(params, config)

    def full_step(
        params: dict[str, Array], state: Any, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], Any, StepMetrics]:
        del key
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        new_params, new_state = noise_curvature_step(
            params, state, grads, x, y, cross_entropy_loss, config
        )
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step

    return init_fn, full_step


def _wrap_grad_learner(
    init_fn: LearnerInitFn, step_fn: LearnerStepFn
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Adapt an ``upgd_ipmnist`` (grads, key)-driven learner to the screening API.

    Mirrors the ``run_ipmnist`` inner-step ordering exactly so control arms
    reproduce the full-horizon lane bit-for-bit.
    """

    def full_step(
        params: dict[str, Array], state: Any, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], Any, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        new_params, new_state = step_fn(params, state, grads, key)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (a) UPGD-W + per-weight step-sizes (IDBD / Autostep)
# =============================================================================


@chex.dataclass(frozen=True)
class UPGDIDBDState:
    """UPGD utility EMA/clock plus IDBD per-weight log step-sizes and traces."""

    utility: dict[str, Array]
    step: Array
    log_alpha: dict[str, Array]
    trace: dict[str, Array]


def upgd_idbd_update(
    params: dict[str, Array],
    state: UPGDIDBDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDIDBDState]:
    """UPGD-W step with IDBD-style per-weight step-sizes.

    The meta signal is the gated loss gradient ``z = grad * (1 - gate)`` —
    the direction UPGD actually descends (perturbation noise excluded from
    meta-learning). Following the Meyer error-free variant implemented by
    :class:`~alberta_framework.core.optimizers.IDBD`:
    ``log_alpha += meta * z * h`` (old trace), then
    ``h = h * max(0, 1 - alpha * z^2) + alpha * z`` with the new alpha.
    ``log_alpha`` is clipped to ``[-10, 0]`` (alpha <= 1) so the per-weight
    decoupled decay ``1 - alpha * wd`` stays positive.

    With ``meta_step_size = 0`` and ``initial_step_size`` equal to the
    published UPGD-W step size this reduces exactly to the lean UPGD-W step
    (pinned by a unit test).
    """
    wd = hp["weight_decay"]
    meta = hp["meta_step_size"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    new_params: dict[str, Array] = {}
    new_log_alpha: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        keep = 1.0 - gate[name]
        z = grads[name] * keep
        log_alpha = _clip_finite_log_alpha(
            state.log_alpha[name], meta * z * state.trace[name]
        )
        alpha = jnp.exp(log_alpha)
        new_params[name] = params[name] * (1.0 - alpha * wd) - alpha * (
            (grads[name] + noise[name]) * keep
        )
        trace_decay = jnp.maximum(0.0, 1.0 - alpha * z * z)
        new_log_alpha[name] = log_alpha
        new_trace[name] = state.trace[name] * trace_decay + alpha * z
    return new_params, UPGDIDBDState(  # type: ignore[call-arg]
        utility=utility, step=count, log_alpha=new_log_alpha, trace=new_trace
    )


def upgd_idbd_swift_update(
    params: dict[str, Array],
    state: UPGDIDBDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDIDBDState]:
    """:func:`upgd_idbd_update` plus SwiftTD's two supervised-mode stabilizers.

    Same state, same meta signal (``z = grad * (1 - gate)``, the direction
    UPGD actually descends), same IDBD meta/trace equations. In the role of
    SwiftTD's feature ``phi_i`` (:mod:`alberta_framework.core.swift_td`) this
    supervised per-weight arm uses that same ``z_i``:

    - **Overshoot bound**: the network-global correction ratio
      ``tau = sum_i alpha_i * z_i^2`` is capped at ``swift_eta``. When
      ``tau > swift_eta`` every per-weight step this update applies is scaled
      by ``swift_eta / tau`` (weight decay and trace extension included,
      exactly as SwiftTD's ``bound_scale`` scales its whole ``z_delta``).
    - **Persistent step-size decay on trigger**: when the bound fires the
      stored log step-sizes decay by ``ln(swift_eps) * z_i^2`` (proportional
      to each weight's contribution, then re-clipped to the IDBD bounds) and
      the meta-learning traces reset to zero, mirroring the reference decay
      block that zeroes SwiftTD's ``h`` traces.

    With ``swift_eta = inf`` and ``swift_eps = 1`` this reduces exactly to
    :func:`upgd_idbd_update` (pinned by a unit test).
    """
    wd = hp["weight_decay"]
    meta = hp["meta_step_size"]
    eta = hp["swift_eta"]
    log_eps = math.log(hp["swift_eps"])
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    z_all: dict[str, Array] = {}
    log_alpha_all: dict[str, Array] = {}
    for name in params:
        z_all[name] = grads[name] * (1.0 - gate[name])
        log_alpha_all[name] = _clip_finite_log_alpha(
            state.log_alpha[name], meta * z_all[name] * state.trace[name]
        )
    alpha_all = {name: jnp.exp(log_alpha_all[name]) for name in params}
    tau = jnp.sum(
        jnp.stack(
            [jnp.sum(alpha_all[name] * z_all[name] * z_all[name]) for name in sorted(params)]
        )
    )
    triggered = tau > eta
    bound_scale = jnp.where(triggered, eta / tau, 1.0)
    new_params: dict[str, Array] = {}
    new_log_alpha: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        keep = 1.0 - gate[name]
        z = z_all[name]
        alpha_eff = bound_scale * alpha_all[name]
        new_params[name] = params[name] * (1.0 - alpha_eff * wd) - alpha_eff * (
            (grads[name] + noise[name]) * keep
        )
        trace = state.trace[name] * jnp.maximum(0.0, 1.0 - alpha_eff * z * z) + alpha_eff * z
        new_trace[name] = jnp.where(triggered, 0.0, trace)
        new_log_alpha[name] = jnp.where(
            triggered,
            jnp.clip(
                log_alpha_all[name] + log_eps * z * z,
                _IDBD_LOG_ALPHA_MIN,
                _IDBD_LOG_ALPHA_MAX,
            ),
            log_alpha_all[name],
        )
    return new_params, UPGDIDBDState(  # type: ignore[call-arg]
        utility=utility, step=count, log_alpha=new_log_alpha, trace=new_trace
    )


#: Pure IDBD-family update ``(params, state, grads, noise, hp)``.
_IDBDUpdateFn = Callable[
    [dict[str, Array], UPGDIDBDState, dict[str, Array], dict[str, Array], Mapping[str, float]],
    tuple[dict[str, Array], UPGDIDBDState],
]


def _make_idbd_family_learner(
    hp: Mapping[str, float], update: _IDBDUpdateFn
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> UPGDIDBDState:
        log_init = math.log(hp["initial_step_size"])
        return UPGDIDBDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            log_alpha={
                name: jnp.full_like(value, log_init) for name, value in params.items()
            },
            trace={name: jnp.zeros_like(value) for name, value in params.items()},
        )

    def full_step(
        params: dict[str, Array], state: UPGDIDBDState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDIDBDState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


def _make_upgd_idbd_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_idbd_family_learner(hp, upgd_idbd_update)


def _make_upgd_idbd_swift_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_idbd_family_learner(hp, upgd_idbd_swift_update)


@chex.dataclass(frozen=True)
class UPGDAutostepState:
    """UPGD utility EMA/clock plus Autostep step-sizes, traces, normalizers."""

    utility: dict[str, Array]
    step: Array
    alpha: dict[str, Array]
    trace: dict[str, Array]
    normalizer: dict[str, Array]


def upgd_autostep_update(
    params: dict[str, Array],
    state: UPGDAutostepState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDAutostepState]:
    """UPGD-W step with Autostep per-weight step-size adaptation.

    Mahmood et al. 2012 Table 1 with the error-free meta gradient
    ``z * h`` (``z`` = gated loss gradient), the self-regulated normalizer
    ``v``, and a *network-global* effective-step normalizer
    ``M = max(sum(alpha * z^2), 1)`` across all parameters.
    """
    wd = hp["weight_decay"]
    mu = hp["meta_step_size"]
    tau = hp["tau"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    z_all = {name: grads[name] * (1.0 - gate[name]) for name in params}
    raw_alpha: dict[str, Array] = {}
    new_normalizer: dict[str, Array] = {}
    for name in params:
        z = z_all[name]
        meta_gradient = z * state.trace[name]
        abs_meta = jnp.abs(meta_gradient)
        v_update = state.normalizer[name] + (1.0 / tau) * state.alpha[name] * z * z * (
            abs_meta - state.normalizer[name]
        )
        v_candidate = jnp.maximum(abs_meta, v_update)
        valid_meta_update = jnp.logical_and(
            jnp.isfinite(meta_gradient), jnp.isfinite(v_candidate)
        )
        v_new = jnp.where(valid_meta_update, v_candidate, state.normalizer[name])
        safe_v = jnp.maximum(v_new, 1e-38)
        raw_alpha[name] = jnp.where(
            valid_meta_update & (v_new > 0.0),
            state.alpha[name] * jnp.exp(mu * meta_gradient / safe_v),
            state.alpha[name],
        )
        new_normalizer[name] = v_new
    effective = jnp.sum(
        jnp.stack(
            [
                jnp.sum(
                    jnp.where(
                        jnp.isfinite(z_all[name]),
                        raw_alpha[name] * z_all[name] * z_all[name],
                        0.0,
                    )
                )
                for name in sorted(params)
            ]
        )
    )
    m_factor = jnp.maximum(effective, 1.0)
    new_params: dict[str, Array] = {}
    new_alpha: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        keep = 1.0 - gate[name]
        z = z_all[name]
        alpha = jnp.clip(raw_alpha[name] / m_factor, _AUTOSTEP_ALPHA_MIN, _AUTOSTEP_ALPHA_MAX)
        new_params[name] = params[name] * (1.0 - alpha * wd) - alpha * (
            (grads[name] + noise[name]) * keep
        )
        new_alpha[name] = alpha
        trace_candidate = state.trace[name] * (1.0 - alpha * z * z) + alpha * z
        new_trace[name] = jnp.where(
            jnp.isfinite(trace_candidate),
            trace_candidate,
            state.trace[name],
        )
    return new_params, UPGDAutostepState(  # type: ignore[call-arg]
        utility=utility,
        step=count,
        alpha=new_alpha,
        trace=new_trace,
        normalizer=new_normalizer,
    )


def _make_upgd_autostep_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> UPGDAutostepState:
        return UPGDAutostepState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            alpha={
                name: jnp.full_like(value, hp["initial_step_size"])
                for name, value in params.items()
            },
            trace={name: jnp.zeros_like(value) for name, value in params.items()},
            normalizer={name: jnp.zeros_like(value) for name, value in params.items()},
        )

    def full_step(
        params: dict[str, Array], state: UPGDAutostepState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDAutostepState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_autostep_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (i) UPGD-W + FADE meta-learned per-parameter weight decay on the head
# =============================================================================

#: ``gamma <= 0`` keeps ``lambda = exp(gamma) <= 1`` so the head decay factor
#: ``1 - lambda`` stays in ``[0, 1]`` (no sign-flipping overshoot).
_FADE_GAMMA_MAX = 0.0
_FADE_HEAD_PARAMS = ("w3", "b3")


@chex.dataclass(frozen=True)
class UPGDFadeHeadState:
    """UPGD utility EMA/clock plus FADE log decay rates and sensitivity traces.

    ``gamma``/``fade_trace`` carry entries for the head parameters
    (``w3``/``b3``) only; hidden layers keep the protocol's fixed decay.
    """

    utility: dict[str, Array]
    step: Array
    gamma: dict[str, Array]
    fade_trace: dict[str, Array]


def upgd_w_fade_head_update(
    params: dict[str, Array],
    state: UPGDFadeHeadState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    head_input: Array,
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDFadeHeadState]:
    """Lean UPGD-W step with FADE meta-learned weight decay on the output layer.

    FADE-style per-parameter weight decay (Ramesh, Lewandowski & Schmidhuber,
    arXiv:2604.27063; equations re-derived, full text not locally cached) on
    ``w3``/``b3`` only. Hidden layers (``w1/b1/w2/b2``) take the unchanged
    lean UPGD-W step with the fixed decoupled decay
    ``1 - step_size * weight_decay``. The head replaces that fixed decay with
    a per-parameter factor ``1 - lambda_i``, ``lambda_i = exp(gamma_i)``,
    meta-learned online:

    - Meta update (old trace first, IDBD convention):
      ``gamma_i += theta_lambda * delta_t * x_i * g_i``, then capped at
      ``gamma_i <= 0`` so ``lambda_i <= 1``. For the softmax cross-entropy
      head the error-times-input product is exactly
      ``delta_t * x_i = -dL/dw_i`` elementwise (``delta_t`` = one-hot target
      minus softmax at the output; ``x_i`` = head input activation ``a2`` for
      ``w3`` and the constant 1 for ``b3``, SwiftTD's bias-feature
      convention), so it is implemented as ``gamma += theta * (-grad) * g``.
    - Sensitivity trace (forward-mode ``g_i ~ d w_i / d gamma_i`` through the
      head update, diagonal/IDBD approximation), with the *new* ``lambda_i``
      and the *pre-update* weight:
      ``g_i <- g_i * max(0, 1 - lambda_i - fade_alpha * x_i^2)
      - lambda_i * w_i``. Both subtractions inside the ``max`` shrink the
      trace, so the contraction factor lies in ``[0, 1]`` (``lambda_i <= 1``)
      and ``|g_i|`` stays bounded by a geometric sum of ``lambda_i * |w_i|``
      -- the stable orientation of the trace recursion.

    Sign-convention reading (chosen so lambda shrinks when decay hurts):
    ``g_i`` accumulates ``-lambda_i * w_i``, i.e. it opposes the sign of a
    decayed weight, while ``delta_t * x_i = -grad_i`` points where descent
    wants the weight to move. When decay hurts (descent wants the weight to
    grow away from zero, ``-grad_i`` aligned with ``w_i``) the product
    ``(-grad_i) * g_i`` is negative and ``gamma_i`` falls (lambda shrinks);
    when decay helps (stale weight the new task's gradient pushes toward
    zero) the product is positive and ``gamma_i`` rises.

    ``fade_alpha`` is FADE's base step-size inside the trace contraction only
    (published 0.005); the applied gradient step keeps the protocol
    ``step_size`` -- UPGD-W's gate, noise, and descent are unchanged on every
    layer. With ``fade_theta_lambda = 0`` and ``fade_gamma0 = -inf``
    (``lambda = 0``) the head reduces exactly to the control update with zero
    head weight decay (pinned by a unit test).
    """
    step_size = hp["step_size"]
    theta = hp["fade_theta_lambda"]
    fade_alpha = hp["fade_alpha"]
    hidden_decay = 1.0 - step_size * hp["weight_decay"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    head_sq = {
        "w3": (head_input * head_input)[:, None],
        "b3": jnp.ones_like(params["b3"]),
    }
    new_params: dict[str, Array] = {}
    new_gamma: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        descent = step_size * ((grads[name] + noise[name]) * (1.0 - gate[name]))
        if name in _FADE_HEAD_PARAMS:
            gamma = jnp.minimum(
                state.gamma[name] + theta * (-grads[name]) * state.fade_trace[name],
                _FADE_GAMMA_MAX,
            )
            lam = jnp.exp(gamma)
            new_params[name] = params[name] * (1.0 - lam) - descent
            contraction = jnp.maximum(0.0, 1.0 - lam - fade_alpha * head_sq[name])
            new_gamma[name] = gamma
            new_trace[name] = state.fade_trace[name] * contraction - lam * params[name]
        else:
            new_params[name] = params[name] * hidden_decay - descent
    return new_params, UPGDFadeHeadState(  # type: ignore[call-arg]
        utility=utility, step=count, gamma=new_gamma, fade_trace=new_trace
    )


def _make_upgd_w_fade_head_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]
    gamma0 = hp["fade_gamma0"]

    def init_fn(params: dict[str, Array]) -> UPGDFadeHeadState:
        return UPGDFadeHeadState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            gamma={name: jnp.full_like(params[name], gamma0) for name in _FADE_HEAD_PARAMS},
            fade_trace={name: jnp.zeros_like(params[name]) for name in _FADE_HEAD_PARAMS},
        )

    def full_step(
        params: dict[str, Array], state: UPGDFadeHeadState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDFadeHeadState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        _, _, _, _, a2 = _forward_with_activations(params, x)
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_w_fade_head_update(params, state, grads, noise, a2, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (c) UPGD-W + L2-Init (decay toward initial weights)
# =============================================================================


@chex.dataclass(frozen=True)
class UPGDL2InitState:
    """UPGD utility EMA/clock plus a frozen copy of the initial parameters."""

    utility: dict[str, Array]
    step: Array
    init_params: dict[str, Array]


def upgd_l2init_update(
    params: dict[str, Array],
    state: UPGDL2InitState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDL2InitState]:
    """Lean UPGD-W step whose decoupled decay pulls toward the initial weights."""
    step_size = hp["step_size"]
    wd = hp["weight_decay"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    new_params = {
        name: params[name]
        - step_size * wd * (params[name] - state.init_params[name])
        - step_size * ((grads[name] + noise[name]) * (1.0 - gate[name]))
        for name in params
    }
    return new_params, UPGDL2InitState(  # type: ignore[call-arg]
        utility=utility, step=count, init_params=state.init_params
    )


def _make_upgd_l2init_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> UPGDL2InitState:
        return UPGDL2InitState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            init_params={name: value for name, value in params.items()},
        )

    def full_step(
        params: dict[str, Array], state: UPGDL2InitState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDL2InitState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_l2init_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (e) UPGD-W + EMA input normalization
# =============================================================================


@chex.dataclass(frozen=True)
class EMANormState:
    """Inline EMA normalizer state (mean, var, sample count)."""

    mean: Array
    var: Array
    count: Array


def ema_normalize(
    state: EMANormState, observation: Array, decay: float, epsilon: float
) -> tuple[Array, EMANormState]:
    """Scan-friendly EMA with the core's zero-mean, unit-variance prior sample."""
    new_count = state.count + 1.0
    effective_decay = jnp.minimum(decay, 1.0 - 1.0 / (new_count + 1.0))
    delta = observation - state.mean
    new_mean = state.mean + (1.0 - effective_decay) * delta
    delta2 = observation - new_mean
    new_var = jnp.maximum(
        effective_decay * state.var + (1.0 - effective_decay) * delta * delta2, epsilon
    )
    normalized = (observation - new_mean) / (jnp.sqrt(new_var) + epsilon)
    return normalized, EMANormState(  # type: ignore[call-arg]
        mean=new_mean, var=new_var, count=new_count
    )


@chex.dataclass(frozen=True)
class UPGDNormState:
    """Lean UPGD state plus the EMA input-normalizer state."""

    utility: dict[str, Array]
    step: Array
    norm: EMANormState


def _make_upgd_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]
    decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]
    lean_hp = {
        name: hp[name] for name in ("step_size", "utility_decay", "noise_std", "weight_decay")
    }

    def init_fn(params: dict[str, Array]) -> UPGDNormState:
        input_dim = params["w1"].shape[0]
        return UPGDNormState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array], state: UPGDNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDNormState, StepMetrics]:
        x_norm, new_norm = ema_normalize(state.norm, x, decay, epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        lean_state = LeanUPGDState(  # type: ignore[call-arg]
            utility=state.utility, step=state.step
        )
        new_params, new_lean = lean_upgd_w_update(params, lean_state, grads, noise, lean_hp)
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, UPGDNormState(  # type: ignore[call-arg]
            utility=new_lean.utility, step=new_lean.step, norm=new_norm
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class SGDNormState:
    """Just the EMA input-normalizer state (the gate-ablation arm is stateless
    beyond the normalizer: no utility EMA, no step clock)."""

    norm: EMANormState


def _make_sgd_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Plain SGD + decoupled weight decay behind the exact ``upgd_ema_norm``
    EMA input normalizer (same decay/eps/state threading).

    The final dissection of the normalized-UPGD result: ``upgd_ema_norm_sigma0``
    showed the perturbation is not load-bearing under input conditioning, so
    the method there is normalize + utility-GATED SGD + decay. This arm drops
    the gate too — ``w <- w * (1 - lr*wd) - lr * grad`` — no utility, no gate,
    no noise (the RNG key is deliberately unused). Pinned by a hand-computed
    trajectory test; the normalizer path is pinned bitwise against
    ``upgd_ema_norm``'s on a shared stream.
    """
    step_size = hp["step_size"]
    decay_factor = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]

    def init_fn(params: dict[str, Array]) -> SGDNormState:
        input_dim = params["w1"].shape[0]
        return SGDNormState(  # type: ignore[call-arg]
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array], state: SGDNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], SGDNormState, StepMetrics]:
        del key  # no perturbation: the per-step noise key is unused
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        new_params = {
            name: params[name] * decay_factor - step_size * grads[name]
            for name in params
        }
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, SGDNormState(norm=new_norm), metrics  # type: ignore[call-arg]

    return init_fn, full_step


# =============================================================================
# Intentional Updates: supervised IPMNIST protocol extension
# =============================================================================


@chex.dataclass(frozen=True, mappable_dataclass=False)
class IntentionalUpdatesIPMNISTState:
    """RMS direction and adaptive-error scale owned by the extension.

    There is deliberately no eligibility trace: unlike the paper's RL
    trajectories, successive IPMNIST examples have no temporal-credit
    semantics.  This is the paper's lambda=0 construction applied to the
    correct-class log probability.
    """

    squared_gradient: dict[str, Array]
    step: Array
    clip_squared_error: Array
    clip_step: Array
    norm: EMANormState


_INTENTIONAL_UPDATES_HP_KEYS = frozenset(
    {
        "intentional_enabled",
        "intended_fraction",
        "fixed_step_size",
        "beta2",
        "optimizer_epsilon",
        "beta_clip",
        "clip_mult",
        "use_diagonal_normalization",
        "use_adaptive_clip",
        "update_features",
        "weight_decay",
        "norm_decay",
        "norm_epsilon",
    }
)


def _intentional_updates_hp(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise ValueError("Intentional Updates hyperparameters must be an exact dict")
    raw = cast(dict[object, object], value)
    if not all(type(key) is str for key in raw) or frozenset(raw) != _INTENTIONAL_UPDATES_HP_KEYS:
        raise ValueError("Intentional Updates hyperparameter keys do not match the protocol")
    hp = cast(dict[str, object], raw)
    if any(type(item) is not float or not math.isfinite(item) for item in hp.values()):
        raise ValueError("Intentional Updates hyperparameters must be finite exact floats")
    checked = cast(dict[str, float], hp)
    if (
        checked["intentional_enabled"] not in (0.0, 1.0)
        or checked["use_diagonal_normalization"] not in (0.0, 1.0)
        or checked["use_adaptive_clip"] not in (0.0, 1.0)
        or checked["update_features"] not in (0.0, 1.0)
        or checked["intended_fraction"] <= 0.0
        or checked["fixed_step_size"] < 0.0
        or not 0.0 <= checked["beta2"] < 1.0
        or checked["optimizer_epsilon"] <= 0.0
        or not 0.0 <= checked["beta_clip"] < 1.0
        or checked["clip_mult"] <= 0.0
        or checked["weight_decay"] < 0.0
        or not 0.0 <= checked["norm_decay"] < 1.0
        or checked["norm_epsilon"] <= 0.0
    ):
        raise ValueError("Intentional Updates hyperparameters violate the frozen bounds")
    if (
        checked["intended_fraction"] != 0.5
        or checked["fixed_step_size"] != 0.01
        or checked["beta2"] != 0.999
        or checked["optimizer_epsilon"] != 1e-8
        or checked["beta_clip"] != 0.9998
        or checked["clip_mult"] != 20.0
        or checked["weight_decay"] != 0.0
        or checked["norm_decay"] != 0.99
        or checked["norm_epsilon"] != 1e-8
    ):
        raise ValueError("Intentional Updates hyperparameters drift from the frozen protocol")
    return checked


def _intentional_updates_scalar_int_array(value: object, *, name: str) -> Array:
    actual_type = type(value)
    if not issubclass(actual_type, (jax.Array, jax.core.Tracer)):
        raise ValueError(f"{name} must be an exact JAX scalar integer array")
    result = jnp.asarray(value)
    if result.shape != () or result.dtype != jnp.dtype(jnp.int32):
        raise ValueError(f"{name} must be an exact JAX scalar integer array")
    return result


def _intentional_updates_invalid_result(valid: Array, value: Array) -> Array:
    if jnp.issubdtype(value.dtype, jnp.floating):
        fallback = jnp.full_like(value, jnp.nan)
    else:
        fallback = jnp.zeros_like(value)
    return jnp.where(valid, value, fallback)


def _make_intentional_updates_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Build the frozen supervised Intentional Updates screening slice.

    For surprisal ``L=-log p(y|x)``, the controlled scalar is ``log p(y|x)``.
    The intended change is ``eta * L`` and the direction is
    ``-rho * grad(L)``.  Paper Eq. 5 therefore gives

    ``alpha = eta * safe(L) / <grad(L), rho * grad(L)>``.

    This is an explicitly non-publication-equivalent supervised extension.
    ``intentional_enabled=0`` delegates to the exact normalized-SGD factory,
    which makes the mechanism-off path bit-for-bit identical rather than
    merely numerically close.
    """
    checked_hp = _intentional_updates_hp(hp)
    if checked_hp["intentional_enabled"] == 0.0:
        return _make_sgd_ema_norm_learner({
            "step_size": checked_hp["fixed_step_size"],
            "weight_decay": checked_hp["weight_decay"],
            "norm_decay": checked_hp["norm_decay"],
            "norm_epsilon": checked_hp["norm_epsilon"],
        })

    eta = checked_hp["intended_fraction"]
    beta2 = checked_hp["beta2"]
    beta_clip = checked_hp["beta_clip"]
    clip_mult = checked_hp["clip_mult"]
    epsilon = checked_hp["optimizer_epsilon"]
    norm_decay = checked_hp["norm_decay"]
    norm_epsilon = checked_hp["norm_epsilon"]
    use_diagonal = checked_hp["use_diagonal_normalization"] == 1.0
    use_adaptive_clip = checked_hp["use_adaptive_clip"] == 1.0
    update_features = checked_hp["update_features"] == 1.0

    def init_fn(params: dict[str, Array]) -> IntentionalUpdatesIPMNISTState:
        checked_params = _l2er_params(params)
        input_dim = checked_params["w1"].shape[0]
        return IntentionalUpdatesIPMNISTState(  # type: ignore[call-arg]
            squared_gradient={
                name: jnp.zeros_like(value) for name, value in checked_params.items()
            },
            step=jnp.asarray(0, dtype=jnp.int32),
            clip_squared_error=jnp.asarray(0.0, dtype=jnp.float32),
            clip_step=jnp.asarray(0, dtype=jnp.int32),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.asarray(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array],
        state: IntentionalUpdatesIPMNISTState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], IntentionalUpdatesIPMNISTState, StepMetrics]:
        del key
        checked_params = _l2er_params(params)
        checked_x = _l2er_array(x, name="x", ndim=1)
        checked_y = _intentional_updates_scalar_int_array(y, name="y")
        if checked_x.shape != (checked_params["w1"].shape[0],):
            raise ValueError("x must match the current IPMNIST input width")
        if type(state) is not IntentionalUpdatesIPMNISTState:
            raise ValueError("state must be an exact IntentionalUpdatesIPMNISTState")
        squared_gradient = _l2er_params(state.squared_gradient)
        if any(
            squared_gradient[name].shape != value.shape
            for name, value in checked_params.items()
        ):
            raise ValueError("squared-gradient state must match the parameter tree")
        step = _intentional_updates_scalar_int_array(state.step, name="state.step")
        clip_step = _intentional_updates_scalar_int_array(
            state.clip_step, name="state.clip_step"
        )
        clip_squared_error = _l2er_array(
            state.clip_squared_error, name="state.clip_squared_error", ndim=0
        )
        if type(state.norm) is not EMANormState:
            raise ValueError("state.norm must be an exact EMANormState")
        norm_mean = _l2er_array(state.norm.mean, name="state.norm.mean", ndim=1)
        norm_var = _l2er_array(state.norm.var, name="state.norm.var", ndim=1)
        norm_count = _l2er_array(state.norm.count, name="state.norm.count", ndim=0)
        if norm_mean.shape != checked_x.shape or norm_var.shape != checked_x.shape:
            raise ValueError("normalizer state must match the IPMNIST input width")
        if any(
            value.dtype != jnp.dtype(jnp.float32)
            for value in (checked_x, clip_squared_error, norm_mean, norm_var, norm_count)
        ):
            raise ValueError("Intentional Updates inputs and state must use float32")
        valid = (
            floating_tree_is_finite(checked_params)
            & floating_tree_is_finite(squared_gradient)
            & jnp.all(jnp.isfinite(checked_x))
            & jnp.isfinite(clip_squared_error)
            & jnp.all(jnp.isfinite(norm_mean))
            & jnp.all(jnp.isfinite(norm_var))
            & jnp.isfinite(norm_count)
            & (step >= 0)
            & (clip_step == step)
            & (step < jnp.asarray((1 << 31) - 1, dtype=jnp.int32))
            & (norm_count >= 0.0)
            & jnp.all(norm_var > 0.0)
            & (checked_y >= 0)
            & (checked_y < checked_params["b3"].shape[0])
        )
        if not isinstance(valid, jax.core.Tracer) and not bool(valid):
            raise ValueError("Intentional Updates inputs and state must be finite and valid")
        safe_params = jax.tree.map(
            lambda value: jnp.where(jnp.isfinite(value), value, jnp.zeros_like(value)),
            checked_params,
        )
        safe_x = jnp.where(jnp.isfinite(checked_x), checked_x, jnp.zeros_like(checked_x))
        safe_y = jnp.clip(checked_y, 0, checked_params["b3"].shape[0] - 1)
        safe_norm = EMANormState(  # type: ignore[call-arg]
            mean=jnp.where(jnp.isfinite(norm_mean), norm_mean, jnp.zeros_like(norm_mean)),
            var=jnp.where(jnp.isfinite(norm_var), norm_var, jnp.ones_like(norm_var)),
            count=jnp.where(
                jnp.isfinite(norm_count) & (norm_count >= 0.0),
                norm_count,
                jnp.zeros_like(norm_count),
            ),
        )
        x_norm, new_norm = ema_normalize(safe_norm, safe_x, norm_decay, norm_epsilon)
        (loss, logits), raw_grads = jax.value_and_grad(
            cross_entropy_loss, has_aux=True
        )(safe_params, x_norm, safe_y)
        grads = {
            name: (
                gradient
                if update_features or name in ("w3", "b3")
                else jnp.zeros_like(gradient)
            )
            for name, gradient in raw_grads.items()
        }
        new_step = step + jnp.asarray(1, dtype=jnp.int32)
        next_squared_gradient = {
            name: beta2 * squared_gradient[name]
            + (1.0 - beta2) * jnp.square(gradient)
            for name, gradient in grads.items()
        }
        bias_correction = 1.0 - beta2 ** new_step.astype(jnp.float32)
        scale = {
            name: (
                1.0 / (jnp.sqrt(value / bias_correction) + epsilon)
                if use_diagonal
                else jnp.ones_like(value)
            )
            for name, value in next_squared_gradient.items()
        }
        denominator = sum(
            (
                jnp.vdot(grads[name], scale[name] * grads[name]).real
                for name in sorted(grads)
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )

        new_clip_step = clip_step + jnp.asarray(1, dtype=jnp.int32)
        clip_squared_error = (
            beta_clip * clip_squared_error + (1.0 - beta_clip) * jnp.square(loss)
        )
        clip_bias_correction = 1.0 - beta_clip ** new_clip_step.astype(jnp.float32)
        adaptive_cap = clip_mult * jnp.sqrt(clip_squared_error / clip_bias_correction)
        safe_loss = jnp.minimum(loss, adaptive_cap) if use_adaptive_clip else loss
        multiplier = eta * safe_loss / jnp.maximum(denominator, epsilon)
        new_params = {
            name: safe_params[name] - multiplier * scale[name] * grads[name]
            for name in safe_params
        }
        metrics = _step_metrics(new_params, x_norm, safe_y, loss, logits)
        new_state = IntentionalUpdatesIPMNISTState(  # type: ignore[call-arg]
            squared_gradient=next_squared_gradient,
            step=new_step,
            clip_squared_error=clip_squared_error,
            clip_step=new_clip_step,
            norm=new_norm,
        )
        valid = valid & floating_tree_is_finite((new_params, new_state, metrics))
        if not isinstance(valid, jax.core.Tracer) and not bool(valid):
            raise ValueError("Intentional Updates produced a non-finite update")
        return (
            jax.tree.map(
                lambda value: _intentional_updates_invalid_result(valid, value), new_params
            ),
            jax.tree.map(
                lambda value: _intentional_updates_invalid_result(valid, value), new_state
            ),
            jax.tree.map(lambda value: _intentional_updates_invalid_result(valid, value), metrics),
        )

    return init_fn, full_step


# =============================================================================
# (n) sigma0_* frontier extensions on the normalized sigma0 champion
# =============================================================================


def _hidden_rms_normalize(activation: Array, epsilon: float) -> Array:
    """Stateless per-example RMS normalization of one hidden activation vector.

    ``a / sqrt(mean(a^2) + eps)`` — layer-norm-style conditioning with no
    learnable parameters and no running statistics (the stream-x recipe).
    The epsilon keeps an all-zero ReLU vector (fully dormant layer) exactly
    zero instead of NaN.
    """
    return activation / jnp.sqrt(jnp.mean(activation * activation) + epsilon)


#: Loss callable ``(params, x, y) -> (loss, logits)`` used by the extension
#: factory (protocol MLP or its hidden-RMS-normalized variant).
_ExtLossFn = Callable[[dict[str, Array], Array, Array], tuple[Array, Array]]


def _make_upgd_ema_norm_ext_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Frontier-extension factory on the ``upgd_ema_norm_sigma0`` champion.

    One factory, three orthogonal switches over the normalize + utility-gated
    SGD + decoupled-decay step (each inert at its default):

    - ``hidden_rms`` (default 0): RMS-normalize both hidden ReLU activation
      vectors per example (:func:`_hidden_rms_normalize`,
      ``hidden_rms_epsilon``) inside the forward pass — gradients, utilities,
      and metrics all see the normalized network.
    - ``gate_beta`` (default 1): utility-gate temperature — the sigmoid
      argument (bias-corrected utility over its max) is scaled by beta.
    - ``local_gate`` (default 0): normalize the gate by the per-tensor
      utility max (zero-guarded exactly as :func:`upgd_w_localgate_update`)
      instead of the network-global max.

    With every switch at its default the trajectory is bit-exact against
    ``upgd_ema_norm_sigma0`` (pinned by a unit test): the perturbation term
    comes from the same ``_sorted_flat_noise`` call as the champion's
    factory, whose sigma=0 short-circuit skips the 282,160-element normal
    draw and leaves the RNG key untouched for both factories, so the two
    lower to identical HLO (pinned; issue #46).  ``noise_std > 0`` keeps
    the champion's exact noise stream for completeness.
    """
    noise_std = hp["noise_std"]
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    gate_beta = hp.get("gate_beta", 1.0)
    local_gate = hp.get("local_gate", 0.0) != 0.0
    hidden_rms = hp.get("hidden_rms", 0.0) != 0.0
    rms_epsilon = hp.get("hidden_rms_epsilon", 1e-8)

    def _hidden_rms_loss(
        params: dict[str, Array], x: Array, y: Array
    ) -> tuple[Array, Array]:
        z1 = x @ params["w1"] + params["b1"]
        h1 = _hidden_rms_normalize(jax.nn.relu(z1), rms_epsilon)
        z2 = h1 @ params["w2"] + params["b2"]
        h2 = _hidden_rms_normalize(jax.nn.relu(z2), rms_epsilon)
        logits = h2 @ params["w3"] + params["b3"]
        return -jax.nn.log_softmax(logits)[y], logits

    loss_fn: _ExtLossFn = _hidden_rms_loss if hidden_rms else cross_entropy_loss

    def init_fn(params: dict[str, Array]) -> UPGDNormState:
        input_dim = params["w1"].shape[0]
        return UPGDNormState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array], state: UPGDNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDNormState, StepMetrics]:
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_norm, y
        )
        if not hidden_rms and not local_gate and gate_beta == 1.0:
            # Keep the inert extension on the authoritative lean-UPGD path.
            # Repeating the same equations here can produce low-bit parameter
            # differences under JAX even when the accuracy and loss traces
            # remain equal, which invalidates the reduction pin's plasticity
            # metric.  The shared path also preserves the exact noise stream
            # when a nonzero noise arm uses this factory.
            lean_state = LeanUPGDState(  # type: ignore[call-arg]
                utility=state.utility,
                step=state.step,
            )
            lean_hp = {
                name: hp[name]
                for name in ("step_size", "utility_decay", "noise_std", "weight_decay")
            }
            noise = _sorted_flat_noise(key, params, noise_std)
            reduced_params, reduced_lean = lean_upgd_w_update(
                params, lean_state, grads, noise, lean_hp
            )
            metrics = _step_metrics(reduced_params, x_norm, y, loss, logits)
            return reduced_params, UPGDNormState(  # type: ignore[call-arg]
                utility=reduced_lean.utility, step=reduced_lean.step, norm=new_norm
            ), metrics
        noise = _sorted_flat_noise(key, params, noise_std)
        count = state.step + jnp.array(1, dtype=jnp.int32)
        utility = {
            name: utility_decay * state.utility[name]
            + (1.0 - utility_decay) * (-grads[name] * params[name])
            for name in params
        }
        bias_correction = 1.0 - jnp.power(
            jnp.asarray(utility_decay, dtype=jnp.float32), count.astype(jnp.float32)
        )
        global_max = jnp.max(
            jnp.stack([jnp.max(utility[name]) for name in sorted(params)])
        )
        new_params: dict[str, Array] = {}
        for name in params:
            if local_gate:
                local_max = jnp.max(utility[name])
                divisor = jnp.where(local_max == 0.0, 1.0, local_max)
            else:
                divisor = global_max
            scaled = (utility[name] / bias_correction) / divisor
            if gate_beta != 1.0:
                scaled = gate_beta * scaled
            gate = jax.nn.sigmoid(scaled)
            new_params[name] = params[name] * param_decay - step_size * (
                (grads[name] + noise[name]) * (1.0 - gate)
            )
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        loss_after, _ = loss_fn(new_params, x_norm, y)
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
        )
        return new_params, UPGDNormState(  # type: ignore[call-arg]
            utility=utility, step=count, norm=new_norm
        ), (accuracy, loss, plasticity)

    return init_fn, full_step


# =============================================================================
# (o) Adaptive-decay normalizers: shift-triggered re-conditioning
# =============================================================================
#
# The wave-7/frontier-2 decomposition attributes the conditioning win to input
# -statistics *tracking speed*: decay 0.99 beats the 0.999 champion because it
# re-conditions faster after each pixel permutation, while slower decay is
# stabler within a task.  These normalizers try to get both: keep the slow
# champion statistics, but *detect* distribution shift online (never task
# boundaries — no oracle) and temporarily accelerate adaptation by resetting
# the count that drives ``ema_normalize``'s annealed effective decay
# ``min(decay, 1 - 1/(count + 1))``.


@chex.dataclass(frozen=True)
class UPGDAdaptiveNormState:
    """Lean UPGD state plus adaptive-decay normalizer state.

    ``norm.count`` is per-feature ``f32[d]`` for the shift-triggered
    normalizer and scalar for the warm-restart normalizer; ``fast_mean`` is
    the fast detection EMA in both.
    """

    utility: dict[str, Array]
    step: Array
    norm: EMANormState
    fast_mean: Array


def shift_adaptive_normalize(
    state: EMANormState,
    fast_mean: Array,
    observation: Array,
    *,
    decay: float,
    fast_decay: float,
    epsilon: float,
    shift_k: float,
    shift_delta: float,
    shift_refractory: float = 0.0,
) -> tuple[Array, EMANormState, Array, Array]:
    """Per-feature shift-triggered re-conditioning EMA normalizer.

    A fast per-feature mean EMA (``fast_decay``) runs beside the slow
    statistics.  When ``|fast_mean - mean| > shift_k * sqrt(var) +
    shift_delta`` for a feature, that feature's anneal count resets to zero,
    so its effective decay drops to 1/2 and re-anneals toward ``decay`` —
    fast re-conditioning exactly where the input distribution moved, slow
    stable statistics everywhere else.  ``shift_refractory`` rate-limits the
    detector per feature (mirroring :func:`warm_restart_normalize`'s clock
    guard): a feature may only trigger once its anneal count has matured past
    the refractory, so a just-reset feature anneals undisturbed instead of
    being pinned at effective decay 1/2 while diverged.  Counts are
    nonnegative, so the default ``0.0`` is bitwise the unguarded detector,
    and with an untriggerable threshold the equations are bitwise
    :func:`ema_normalize` (per-feature count).

    Returns ``(normalized, new_state, new_fast_mean, shifted_mask)``.
    """
    effective_fast = jnp.minimum(fast_decay, 1.0 - 1.0 / (state.count + 2.0))
    new_fast = effective_fast * fast_mean + (1.0 - effective_fast) * observation
    threshold = shift_k * jnp.sqrt(state.var) + shift_delta
    shifted = (jnp.abs(new_fast - state.mean) > threshold) & (
        state.count >= shift_refractory
    )
    new_count = jnp.where(shifted, 0.0, state.count) + 1.0
    effective_decay = jnp.minimum(decay, 1.0 - 1.0 / (new_count + 1.0))
    delta = observation - state.mean
    new_mean = state.mean + (1.0 - effective_decay) * delta
    delta2 = observation - new_mean
    new_var = jnp.maximum(
        effective_decay * state.var + (1.0 - effective_decay) * delta * delta2, epsilon
    )
    normalized = (observation - new_mean) / (jnp.sqrt(new_var) + epsilon)
    return normalized, EMANormState(  # type: ignore[call-arg]
        mean=new_mean, var=new_var, count=new_count
    ), new_fast, shifted


def warm_restart_normalize(
    state: EMANormState,
    fast_mean: Array,
    observation: Array,
    *,
    decay: float,
    fast_decay: float,
    epsilon: float,
    warm_threshold: float,
    warm_pad: float,
    warm_refractory: float,
) -> tuple[Array, EMANormState, Array, Array]:
    """Globally shift-reset annealed-decay EMA normalizer (batch-stats warmup).

    Divergence score = mean over features of ``|fast_mean - mean| /
    (sqrt(var) + warm_pad)``.  When the score exceeds ``warm_threshold`` and
    the scalar anneal clock has passed ``warm_refractory`` steps, the clock
    resets, so the effective decay warms up again from 1/2 toward ``decay``
    (``min(decay, 1 - 1/(t + 2))`` with ``t`` = steps since the last detected
    shift) exactly as at stream start.  Detection is purely observational —
    never a task-boundary oracle.  With an infinite threshold the equations
    are bitwise :func:`ema_normalize`.

    Returns ``(normalized, new_state, new_fast_mean, triggered)``.
    """
    effective_fast = jnp.minimum(fast_decay, 1.0 - 1.0 / (state.count + 2.0))
    new_fast = effective_fast * fast_mean + (1.0 - effective_fast) * observation
    score = jnp.mean(jnp.abs(new_fast - state.mean) / (jnp.sqrt(state.var) + warm_pad))
    triggered = (score > warm_threshold) & (state.count >= warm_refractory)
    new_count = jnp.where(triggered, 0.0, state.count) + 1.0
    effective_decay = jnp.minimum(decay, 1.0 - 1.0 / (new_count + 1.0))
    delta = observation - state.mean
    new_mean = state.mean + (1.0 - effective_decay) * delta
    delta2 = observation - new_mean
    new_var = jnp.maximum(
        effective_decay * state.var + (1.0 - effective_decay) * delta * delta2, epsilon
    )
    normalized = (observation - new_mean) / (jnp.sqrt(new_var) + epsilon)
    return normalized, EMANormState(  # type: ignore[call-arg]
        mean=new_mean, var=new_var, count=new_count
    ), new_fast, triggered


_AdaptiveNormalizeFn = Callable[
    [EMANormState, Array, Array], tuple[Array, EMANormState, Array, Array]
]


def _make_adaptive_norm_sigma0_learner(
    hp: Mapping[str, float],
    normalize: _AdaptiveNormalizeFn,
    init_count: Callable[[int], Array],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Shared sigma0 (normalize + utility-gated SGD + decay) learner over an
    adaptive normalizer.  The update equations are exactly the sigma0
    champion's (``_make_upgd_ema_norm_ext_learner`` defaults): explicit zero
    perturbation, bias-corrected utility EMA, global-max sigmoid gate,
    decoupled decay.  The RNG key is deliberately untouched."""
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]

    def init_fn(params: dict[str, Array]) -> UPGDAdaptiveNormState:
        input_dim = params["w1"].shape[0]
        return UPGDAdaptiveNormState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=init_count(input_dim),
            ),
            fast_mean=jnp.zeros(input_dim, dtype=jnp.float32),
        )

    def full_step(
        params: dict[str, Array],
        state: UPGDAdaptiveNormState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], UPGDAdaptiveNormState, StepMetrics]:
        del key  # sigma=0: no perturbation, the per-step noise key is unused
        x_norm, new_norm, new_fast, _ = normalize(state.norm, state.fast_mean, x)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        count = state.step + jnp.array(1, dtype=jnp.int32)
        utility = {
            name: utility_decay * state.utility[name]
            + (1.0 - utility_decay) * (-grads[name] * params[name])
            for name in params
        }
        bias_correction = 1.0 - jnp.power(
            jnp.asarray(utility_decay, dtype=jnp.float32), count.astype(jnp.float32)
        )
        global_max = jnp.max(
            jnp.stack([jnp.max(utility[name]) for name in sorted(params)])
        )
        new_params = {
            name: params[name] * param_decay
            - step_size
            * (grads[name] * (1.0 - jax.nn.sigmoid(
                (utility[name] / bias_correction) / global_max
            )))
            for name in params
        }
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, UPGDAdaptiveNormState(  # type: ignore[call-arg]
            utility=utility, step=count, norm=new_norm, fast_mean=new_fast
        ), metrics

    return init_fn, full_step


def _make_upgd_shiftnorm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """sigma0 champion update behind :func:`shift_adaptive_normalize`."""

    def normalize(
        state: EMANormState, fast_mean: Array, x: Array
    ) -> tuple[Array, EMANormState, Array, Array]:
        return shift_adaptive_normalize(
            state, fast_mean, x,
            decay=hp["norm_decay"],
            fast_decay=hp["fast_decay"],
            epsilon=hp["norm_epsilon"],
            shift_k=hp["shift_k"],
            shift_delta=hp["shift_delta"],
            shift_refractory=hp["shift_refractory"],
        )

    return _make_adaptive_norm_sigma0_learner(
        hp, normalize, lambda d: jnp.zeros(d, dtype=jnp.float32)
    )


def _make_upgd_warmnorm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """sigma0 champion update behind :func:`warm_restart_normalize`."""

    def normalize(
        state: EMANormState, fast_mean: Array, x: Array
    ) -> tuple[Array, EMANormState, Array, Array]:
        return warm_restart_normalize(
            state, fast_mean, x,
            decay=hp["norm_decay"],
            fast_decay=hp["fast_decay"],
            epsilon=hp["norm_epsilon"],
            warm_threshold=hp["warm_threshold"],
            warm_pad=hp["warm_pad"],
            warm_refractory=hp["warm_refractory"],
        )

    return _make_adaptive_norm_sigma0_learner(
        hp, normalize, lambda d: jnp.array(0.0, dtype=jnp.float32)
    )


# =============================================================================
# (o2) Discovered-rule translation factory (rule_discovery promotion lane)
# =============================================================================
#
# The automated update-rule discovery lane
# (:mod:`alberta_framework.benchmarks.rule_discovery`) searches a composable
# DSL over the campaign's primitive vocabulary on the micro continual suite.
# Candidates that beat the budget-matched champion-form baseline on the
# held-out micro tasks are translated verbatim into screening arms through
# this factory: one hyperparameter dict carries the mechanism flags
# (``flag_*`` as 0/1 floats) plus the discovered constants. With the
# champion-form flags (norm + shift_reset + gate, everything else off) the
# step is bit-exact against the registered ``sigma0_shiftnorm_d099``
# champion (pinned), so screened differences are attributable to the
# discovered composition alone.


@chex.dataclass(frozen=True)
class DiscoveredRuleState:
    """Discovered-rule carry: champion normalizer statistics + init snapshot
    + error-signal scalars (surprise-budget and meta-decay inputs).

    Wave-2 fields (rule-DSL expansion): per-feature Kalman posterior
    uncertainty, the closed-form RLS ensemble head over the last hidden
    layer, the streaming naive-Bayes ensemble member, and the vote-accuracy
    EMAs (net, rls, nb). All are always allocated (cheap at protocol scale);
    only enabled mechanisms touch them, so disabled flags leave the traced
    champion step bit-exact.
    """

    utility: dict[str, Array]
    step: Array
    norm: EMANormState
    fast_mean: Array
    init_params: dict[str, Array]
    err_fast: Array
    err_slow: Array
    err_autocorr: Array
    err_var: Array
    err_prev_delta: Array
    kalman_p: Array
    rls_p: Array
    rls_w: Array
    nb_mean: Array
    nb_var: Array
    nb_count: Array
    member_acc: Array


_DISCOVERED_RULE_DEFAULTS: dict[str, float] = {
    "flag_norm": 0.0,
    "flag_shift_reset": 0.0,
    "flag_gate": 0.0,
    "flag_decay_to_init": 0.0,
    "flag_surprise_budget": 0.0,
    "flag_meta_decay": 0.0,
    "flag_utility_shift_reset": 0.0,
    "flag_w1_shift_reset": 0.0,
    "flag_hidden_rms": 0.0,
    "flag_rls_head": 0.0,
    "flag_rls_reset_p": 0.0,
    "flag_nb_member": 0.0,
    "flag_lr_anneal": 0.0,
    "flag_layer_lr": 0.0,
    "flag_kalman_norm": 0.0,
    "step_size": 0.01,
    "weight_decay": 0.01,
    "norm_decay": 0.99,
    "fast_decay": 0.9,
    "shift_k": 1.0,
    "shift_delta": 0.02,
    "norm_epsilon": 1e-8,
    "utility_decay": 0.9999,
    "gate_beta": 1.0,
    "surprise_gain": 1.0,
    "surprise_fast": 0.95,
    "surprise_slow": 0.999,
    "meta_gain": 2.0,
    "hidden_rms_epsilon": 1e-8,
    "rls_lambda": 0.999,
    "nb_decay": 0.98,
    "vote_decay": 0.99,
    "anneal_lo": 0.5,
    "anneal_hi": 2.0,
    "layer_lr_ratio": 1.0,
    "kalman_q": 0.001,
    "noise_std": 0.0,
}

#: Fixed decay of the error autocorrelation/variance EMAs (meta-decay input);
#: mirrors ``rule_discovery._AUTOCORR_DECAY``.
_DISCOVERED_AUTOCORR_DECAY = 0.99
#: Wave-2 rule-DSL constants; mirror their ``rule_discovery`` twins exactly
#: so translated genomes reproduce the searched semantics.
_DISCOVERED_RLS_P0 = 10.0
_DISCOVERED_RLS_VOTE_TEMP = 4.0
_DISCOVERED_RLS_SCORE_CLIP = 25.0
_DISCOVERED_NB_VAR_FLOOR = 1e-3
_DISCOVERED_ANNEAL_R_HI = 2.0
_DISCOVERED_KALMAN_SHIFT_BOOST = 25.0
_DISCOVERED_LAYER_EXPONENT: dict[str, float] = {
    "w1": -1.0, "b1": -1.0, "w2": 0.0, "b2": 0.0, "w3": 1.0, "b3": 1.0,
}


def _discovered_rule_hp(**overrides: float) -> dict[str, float]:
    """Discovered-rule hyperparameters: champion-form constants + inert flags."""
    merged = dict(_DISCOVERED_RULE_DEFAULTS)
    merged.update(overrides)
    return merged


def _make_discovered_rule_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Materialize one discovered rule-DSL composition as a screening learner.

    Mechanism flags are read at build time (Python-level composition, the
    ``vreset_enabled`` precedent), so inactive mechanisms leave the traced
    step untouched: with ``flag_norm/flag_shift_reset/flag_gate`` set and
    everything else off, the arithmetic is exactly
    :func:`_make_upgd_shiftnorm_learner`'s champion step (bit-exact, pinned).
    Active extensions follow the rule-DSL semantics of
    :func:`alberta_framework.benchmarks.rule_discovery.rule_step`. The RNG
    key is deliberately unused (sigma-0 family).
    """
    f_norm = hp["flag_norm"] != 0.0
    f_shift_reset = hp["flag_shift_reset"] != 0.0
    f_gate = hp["flag_gate"] != 0.0
    f_init = hp["flag_decay_to_init"] != 0.0
    f_budget = hp["flag_surprise_budget"] != 0.0
    f_meta = hp["flag_meta_decay"] != 0.0
    f_ureset = hp["flag_utility_shift_reset"] != 0.0
    f_wreset = hp["flag_w1_shift_reset"] != 0.0
    f_rms = hp["flag_hidden_rms"] != 0.0
    f_rls = hp["flag_rls_head"] != 0.0
    f_rls_reset = hp["flag_rls_reset_p"] != 0.0
    f_nb = hp["flag_nb_member"] != 0.0
    f_anneal = hp["flag_lr_anneal"] != 0.0
    f_layer = hp["flag_layer_lr"] != 0.0
    f_kalman = hp["flag_kalman_norm"] != 0.0
    step_size = hp["step_size"]
    weight_decay = hp["weight_decay"]
    param_decay = 1.0 - step_size * weight_decay
    utility_decay = hp["utility_decay"]
    gate_beta = hp["gate_beta"]
    norm_decay = hp["norm_decay"]
    fast_decay = hp["fast_decay"]
    shift_k = hp["shift_k"]
    shift_delta = hp["shift_delta"]
    norm_epsilon = hp["norm_epsilon"]
    surprise_gain = hp["surprise_gain"]
    surprise_fast = hp["surprise_fast"]
    surprise_slow = hp["surprise_slow"]
    meta_gain = hp["meta_gain"]
    rms_epsilon = hp["hidden_rms_epsilon"]
    rls_lambda = hp["rls_lambda"]
    nb_decay = hp["nb_decay"]
    vote_decay = hp["vote_decay"]
    anneal_lo = hp["anneal_lo"]
    anneal_hi = hp["anneal_hi"]
    layer_lr_ratio = hp["layer_lr_ratio"]
    kalman_q = hp["kalman_q"]

    def _rms_loss(params: dict[str, Array], x: Array, y: Array) -> tuple[Array, Array]:
        z1 = x @ params["w1"] + params["b1"]
        h1 = _hidden_rms_normalize(jax.nn.relu(z1), rms_epsilon)
        z2 = h1 @ params["w2"] + params["b2"]
        h2 = _hidden_rms_normalize(jax.nn.relu(z2), rms_epsilon)
        logits = h2 @ params["w3"] + params["b3"]
        return -jax.nn.log_softmax(logits)[y], logits

    loss_fn = _rms_loss if f_rms else cross_entropy_loss

    def _loss_with_hidden(
        params: dict[str, Array], x: Array, y: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        """Loss + (logits, penultimate activation) — the RLS head's feature."""
        h1 = jax.nn.relu(x @ params["w1"] + params["b1"])
        if f_rms:
            h1 = _hidden_rms_normalize(h1, rms_epsilon)
        h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
        if f_rms:
            h2 = _hidden_rms_normalize(h2, rms_epsilon)
        logits = h2 @ params["w3"] + params["b3"]
        return -jax.nn.log_softmax(logits)[y], (logits, h2)

    def init_fn(params: dict[str, Array]) -> DiscoveredRuleState:
        input_dim = params["w1"].shape[0]
        n_classes = params["b3"].shape[0]
        rls_dim = params["b2"].shape[0] + 1
        chance = jnp.asarray(math.log(float(n_classes)), jnp.float32)
        return DiscoveredRuleState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.zeros(input_dim, dtype=jnp.float32),
            ),
            fast_mean=jnp.zeros(input_dim, dtype=jnp.float32),
            init_params={name: value for name, value in params.items()},
            err_fast=chance,
            err_slow=chance,
            err_autocorr=jnp.asarray(0.0, jnp.float32),
            err_var=jnp.asarray(0.0, jnp.float32),
            err_prev_delta=jnp.asarray(0.0, jnp.float32),
            kalman_p=jnp.ones(input_dim, dtype=jnp.float32),
            rls_p=_DISCOVERED_RLS_P0 * jnp.eye(rls_dim, dtype=jnp.float32),
            rls_w=jnp.zeros((rls_dim, n_classes), dtype=jnp.float32),
            nb_mean=jnp.zeros((n_classes, input_dim), dtype=jnp.float32),
            nb_var=jnp.ones((n_classes, input_dim), dtype=jnp.float32),
            nb_count=jnp.zeros(n_classes, dtype=jnp.float32),
            member_acc=jnp.full((3,), 1.0 / float(n_classes), dtype=jnp.float32),
        )

    def _general_normalize(
        norm: EMANormState, fast_mean: Array, x: Array,
        err_autocorr: Array, err_var: Array,
    ) -> tuple[Array, EMANormState, Array, Array]:
        """Detector + EMA statistics for the non-champion flag combinations
        (count reset and meta decay composable independently)."""
        effective_fast = jnp.minimum(fast_decay, 1.0 - 1.0 / (norm.count + 2.0))
        new_fast = effective_fast * fast_mean + (1.0 - effective_fast) * x
        threshold = shift_k * jnp.sqrt(norm.var) + shift_delta
        shifted = jnp.abs(new_fast - norm.mean) > threshold
        count_base = jnp.where(shifted, 0.0, norm.count) if f_shift_reset else norm.count
        new_count = count_base + 1.0
        if f_meta:
            autocorr_score = jnp.clip(err_autocorr / (err_var + 1e-8), 0.0, 1.0)
            decay_used = jnp.clip(
                1.0 - (1.0 - norm_decay) * (1.0 + meta_gain * autocorr_score),
                0.5,
                norm_decay,
            )
        else:
            decay_used = jnp.asarray(norm_decay, jnp.float32)
        effective_decay = jnp.minimum(decay_used, 1.0 - 1.0 / (new_count + 1.0))
        delta = x - norm.mean
        new_mean = norm.mean + (1.0 - effective_decay) * delta
        delta2 = x - new_mean
        new_var = jnp.maximum(
            effective_decay * norm.var + (1.0 - effective_decay) * delta * delta2,
            norm_epsilon,
        )
        normalized = (x - new_mean) / (jnp.sqrt(new_var) + norm_epsilon)
        return normalized, EMANormState(  # type: ignore[call-arg]
            mean=new_mean, var=new_var, count=new_count
        ), new_fast, shifted

    def _kalman_normalize(
        norm: EMANormState, fast_mean: Array, kalman_p: Array, x: Array,
        err_autocorr: Array, err_var: Array,
    ) -> tuple[Array, EMANormState, Array, Array, Array]:
        """Wave-2 conditioning alternative: per-feature predict-update Kalman
        mean tracking (process noise ``kalman_q`` scaled to the tracked
        variance; detected shifts reinflate the posterior uncertainty when
        ``flag_shift_reset`` composes), EMA variance as the observation-noise
        estimate. Mirrors ``rule_discovery.rule_step``'s kalman path."""
        effective_fast = jnp.minimum(fast_decay, 1.0 - 1.0 / (norm.count + 2.0))
        new_fast = effective_fast * fast_mean + (1.0 - effective_fast) * x
        threshold = shift_k * jnp.sqrt(norm.var) + shift_delta
        shifted = jnp.abs(new_fast - norm.mean) > threshold
        count_base = jnp.where(shifted, 0.0, norm.count) if f_shift_reset else norm.count
        new_count = count_base + 1.0
        if f_meta:
            autocorr_score = jnp.clip(err_autocorr / (err_var + 1e-8), 0.0, 1.0)
            decay_used = jnp.clip(
                1.0 - (1.0 - norm_decay) * (1.0 + meta_gain * autocorr_score),
                0.5,
                norm_decay,
            )
        else:
            decay_used = jnp.asarray(norm_decay, jnp.float32)
        effective_decay = jnp.minimum(decay_used, 1.0 - 1.0 / (new_count + 1.0))
        delta = x - norm.mean
        r_obs = jnp.maximum(norm.var, 1e-8)
        p_pred = kalman_p + kalman_q * r_obs
        if f_shift_reset:
            p_pred = p_pred + (
                shifted.astype(jnp.float32) * _DISCOVERED_KALMAN_SHIFT_BOOST * r_obs
            )
        gain = p_pred / (p_pred + r_obs)
        new_mean = norm.mean + gain * delta
        new_kalman_p = (1.0 - gain) * p_pred
        delta2 = x - new_mean
        new_var = jnp.maximum(
            effective_decay * norm.var + (1.0 - effective_decay) * delta * delta2,
            norm_epsilon,
        )
        normalized = (x - new_mean) / (jnp.sqrt(new_var) + norm_epsilon)
        return normalized, EMANormState(  # type: ignore[call-arg]
            mean=new_mean, var=new_var, count=new_count
        ), new_fast, shifted, new_kalman_p

    def full_step(
        params: dict[str, Array],
        state: DiscoveredRuleState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], DiscoveredRuleState, StepMetrics]:
        del key  # sigma-0 family: no perturbation, no randomness
        new_kalman_p = state.kalman_p
        if f_kalman:
            x_norm, new_norm, new_fast, shifted, new_kalman_p = _kalman_normalize(
                state.norm, state.fast_mean, state.kalman_p, x,
                state.err_autocorr, state.err_var,
            )
        elif f_shift_reset and not f_meta:
            # Exact champion normalizer call (bit-exact reduction path).
            x_norm, new_norm, new_fast, shifted = shift_adaptive_normalize(
                state.norm, state.fast_mean, x,
                decay=norm_decay,
                fast_decay=fast_decay,
                epsilon=norm_epsilon,
                shift_k=shift_k,
                shift_delta=shift_delta,
                shift_refractory=0.0,
            )
        else:
            x_norm, new_norm, new_fast, shifted = _general_normalize(
                state.norm, state.fast_mean, x, state.err_autocorr, state.err_var
            )
        x_used = x_norm if f_norm else x
        hidden2: Array | None = None
        if f_rls:
            (loss, (logits, hidden2)), grads = jax.value_and_grad(
                _loss_with_hidden, has_aux=True
            )(params, x_used, y)
        else:
            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, x_used, y
            )
        count = state.step + jnp.array(1, dtype=jnp.int32)
        prev_utility = state.utility
        if f_ureset:
            prev_utility = dict(prev_utility)
            prev_utility["w1"] = prev_utility["w1"] * (
                1.0 - shifted[:, None].astype(jnp.float32)
            )
        utility = {
            name: utility_decay * prev_utility[name]
            + (1.0 - utility_decay) * (-grads[name] * params[name])
            for name in params
        }
        bias_correction = 1.0 - jnp.power(
            jnp.asarray(utility_decay, dtype=jnp.float32), count.astype(jnp.float32)
        )
        global_max = jnp.max(
            jnp.stack([jnp.max(utility[name]) for name in sorted(params)])
        )
        ratio: Array | None = None
        if f_budget or f_anneal:
            ratio = (state.err_fast + 1e-8) / (state.err_slow + 1e-8)
        if f_budget:
            assert ratio is not None
            budget = jnp.clip(
                jnp.exp(surprise_gain * jnp.log(jnp.maximum(ratio, 1e-8))), 0.25, 4.0
            )
            lr_eff: Array | float = step_size * budget
        else:
            lr_eff = step_size
        if f_anneal:
            # Task-clock-free within-task annealing: error ratio 1 (converged)
            # runs at anneal_lo, ratio >= _DISCOVERED_ANNEAL_R_HI (fresh
            # surprise) at anneal_hi. Mirrors rule_discovery.rule_step.
            assert ratio is not None
            surprise_score = jnp.clip(
                (ratio - 1.0) / (_DISCOVERED_ANNEAL_R_HI - 1.0), 0.0, 1.0
            )
            lr_eff = lr_eff * (anneal_lo + (anneal_hi - anneal_lo) * surprise_score)
        if f_budget or f_anneal:
            decay_eff: Array | float = 1.0 - lr_eff * weight_decay
        else:
            decay_eff = param_decay

        # --- ensemble members (wave-2): pre-update readout state, the
        # protocol's predict-then-update convention. Net member always votes.
        rls_scores: Array | None = None
        nb_ll: Array | None = None
        ens_accuracy: Array | None = None
        new_member_acc = state.member_acc
        if f_rls or f_nb:
            n_classes = state.rls_w.shape[1]
            s_net = jax.nn.log_softmax(logits)
            members = [s_net]
            weights = [state.member_acc[0]]
            if f_rls:
                assert hidden2 is not None
                h_aug = jnp.concatenate(
                    [hidden2, jnp.ones((1,), dtype=jnp.float32)]
                )
                rls_scores = jnp.clip(
                    h_aug @ state.rls_w,
                    -_DISCOVERED_RLS_SCORE_CLIP,
                    _DISCOVERED_RLS_SCORE_CLIP,
                )
                members.append(
                    jax.nn.log_softmax(_DISCOVERED_RLS_VOTE_TEMP * rls_scores)
                )
                weights.append(state.member_acc[1])
            if f_nb:
                nb_var_safe = jnp.maximum(state.nb_var, _DISCOVERED_NB_VAR_FLOOR)
                nb_ll = -0.5 * jnp.sum(
                    jnp.log(nb_var_safe)
                    + (x_used[None, :] - state.nb_mean) ** 2 / nb_var_safe,
                    axis=1,
                )
                members.append(jax.nn.log_softmax(nb_ll / float(x_used.shape[0])))
                weights.append(state.member_acc[2])
            w_sum = sum(weights) + 1e-8
            combined = sum(w * s for w, s in zip(weights, members)) / w_sum
            ens_accuracy = (jnp.argmax(combined) == y).astype(jnp.float32)
            hit_net = (jnp.argmax(s_net) == y).astype(jnp.float32)
            hit_rls = (
                (jnp.argmax(rls_scores) == y).astype(jnp.float32)
                if rls_scores is not None
                else state.member_acc[1]
            )
            hit_nb = (
                (jnp.argmax(nb_ll) == y).astype(jnp.float32)
                if nb_ll is not None
                else state.member_acc[2]
            )
            hits = jnp.stack([hit_net, hit_rls, hit_nb])
            new_member_acc = vote_decay * state.member_acc + (1.0 - vote_decay) * hits
            del n_classes

        new_params: dict[str, Array] = {}
        for name in params:
            if f_layer:
                lr_name: Array | float = lr_eff * (
                    layer_lr_ratio ** _DISCOVERED_LAYER_EXPONENT[name]
                )
                decay_name: Array | float = 1.0 - lr_name * weight_decay
            else:
                lr_name, decay_name = lr_eff, decay_eff
            if f_gate:
                scaled = (utility[name] / bias_correction) / global_max
                if gate_beta != 1.0:
                    scaled = gate_beta * scaled
                descent = grads[name] * (1.0 - jax.nn.sigmoid(scaled))
            else:
                descent = grads[name]
            value = params[name] * decay_name - lr_name * descent
            if f_init:
                value = value + (lr_name * weight_decay) * state.init_params[name]
            new_params[name] = value
        if f_wreset:
            new_params["w1"] = jnp.where(
                shifted[:, None], state.init_params["w1"], new_params["w1"]
            )

        # --- wave-2 readout-state updates (post-prediction).
        new_rls_p, new_rls_w = state.rls_p, state.rls_w
        if f_rls:
            assert hidden2 is not None
            h_aug = jnp.concatenate([hidden2, jnp.ones((1,), dtype=jnp.float32)])
            ph = state.rls_p @ h_aug
            k_rls = ph / (rls_lambda + h_aug @ ph)
            rls_err = (
                jax.nn.one_hot(y, state.rls_w.shape[1], dtype=jnp.float32)
                - h_aug @ state.rls_w
            )
            new_rls_w = state.rls_w + jnp.outer(k_rls, rls_err)
            p_upd = (state.rls_p - jnp.outer(k_rls, ph)) / rls_lambda
            rls_eye = _DISCOVERED_RLS_P0 * jnp.eye(
                state.rls_p.shape[0], dtype=jnp.float32
            )
            leak = 2.0 * (1.0 - rls_lambda)
            p_upd = (1.0 - leak) * p_upd + leak * rls_eye
            p_upd = 0.5 * (p_upd + p_upd.T)
            if f_rls_reset:
                reset_p = jnp.max(shifted.astype(jnp.float32))
                p_upd = (1.0 - reset_p) * p_upd + reset_p * rls_eye
            new_rls_p = p_upd
        new_nb_mean, new_nb_var, new_nb_count = (
            state.nb_mean, state.nb_var, state.nb_count
        )
        if f_nb:
            n_classes_nb = state.nb_mean.shape[0]
            sel = jax.nn.one_hot(y, n_classes_nb, dtype=jnp.float32)
            eff_nb = jnp.minimum(
                nb_decay, 1.0 - 1.0 / (state.nb_count + 2.0)
            )[:, None]
            delta_nb = x_used[None, :] - state.nb_mean
            mean_cand = state.nb_mean + (1.0 - eff_nb) * delta_nb
            new_nb_mean = state.nb_mean + sel[:, None] * (mean_cand - state.nb_mean)
            delta2_nb = x_used[None, :] - new_nb_mean
            var_cand = jnp.maximum(
                eff_nb * state.nb_var + (1.0 - eff_nb) * delta_nb * delta2_nb,
                _DISCOVERED_NB_VAR_FLOOR,
            )
            new_nb_var = state.nb_var + sel[:, None] * (var_cand - state.nb_var)
            new_nb_count = state.nb_count + sel

        if f_rls or f_nb:
            assert ens_accuracy is not None
            loss_after, _ = loss_fn(new_params, x_used, y)
            plasticity = jnp.clip(
                1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
            )
            metrics: StepMetrics = (ens_accuracy, loss, plasticity)
        elif f_rms:
            accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
            loss_after, _ = loss_fn(new_params, x_used, y)
            plasticity = jnp.clip(
                1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
            )
            metrics = (accuracy, loss, plasticity)
        else:
            metrics = _step_metrics(new_params, x_used, y, loss, logits)
        delta_e = loss - state.err_slow
        return new_params, DiscoveredRuleState(  # type: ignore[call-arg]
            utility=utility,
            step=count,
            norm=new_norm,
            fast_mean=new_fast,
            init_params=state.init_params,
            err_fast=surprise_fast * state.err_fast + (1.0 - surprise_fast) * loss,
            err_slow=surprise_slow * state.err_slow + (1.0 - surprise_slow) * loss,
            err_autocorr=_DISCOVERED_AUTOCORR_DECAY * state.err_autocorr
            + (1.0 - _DISCOVERED_AUTOCORR_DECAY) * (delta_e * state.err_prev_delta),
            err_var=_DISCOVERED_AUTOCORR_DECAY * state.err_var
            + (1.0 - _DISCOVERED_AUTOCORR_DECAY) * (delta_e * delta_e),
            err_prev_delta=delta_e,
            kalman_p=new_kalman_p,
            rls_p=new_rls_p,
            rls_w=new_rls_w,
            nb_mean=new_nb_mean,
            nb_var=new_nb_var,
            nb_count=new_nb_count,
            member_acc=new_member_acc,
        ), metrics

    return init_fn, full_step


# =============================================================================
# (f) UPGD-W + per-layer weight clipping (Elsayed, Lan, Lyle & Mahmood, 2024)
# =============================================================================


def _wclip_bound(params: dict[str, Array], name: str, kappa: float) -> float:
    """Clipping bound ``kappa * s_l`` for parameter ``name``.

    ``s_l = 1/sqrt(fan_in)`` is the protocol's PyTorch-default uniform init
    bound (:func:`~alberta_framework.benchmarks.upgd_ipmnist.init_mlp_params`
    draws both ``w{l}`` and ``b{l}`` from ``U(-s_l, s_l)``); the paper clips
    weights and biases of layer ``l`` to ``[-kappa * s_l, +kappa * s_l]``.
    """
    fan_in = params[f"w{name[1:]}"].shape[0]
    return kappa / math.sqrt(fan_in)


def upgd_w_wclip_update(
    params: dict[str, Array],
    state: LeanUPGDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], LeanUPGDState]:
    """Lean UPGD-W step followed by per-layer weight clipping.

    Algorithm 1 of Elsayed et al. (RLC 2024): after the optimizer update,
    every weight and bias of layer ``l`` is clipped to
    ``[-kappa * s_l, +kappa * s_l]`` with ``s_l`` the uniform-init bound.
    With ``clip_kappa = inf`` the clip is a no-op and this reduces exactly
    to the lean UPGD-W step (pinned by a unit test).
    """
    kappa = hp["clip_kappa"]
    new_params, new_state = lean_upgd_w_update(params, state, grads, noise, dict(hp))
    clipped = {
        name: jnp.clip(
            new_params[name],
            -_wclip_bound(params, name, kappa),
            _wclip_bound(params, name, kappa),
        )
        for name in new_params
    }
    return clipped, new_state


def _make_upgd_w_wclip_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> LeanUPGDState:
        return LeanUPGDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array], state: LeanUPGDState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], LeanUPGDState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_w_wclip_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (h) UPGD-W with per-tensor (local) utility-gate normalization
# =============================================================================


def upgd_w_localgate_update(
    params: dict[str, Array],
    state: LeanUPGDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], LeanUPGDState]:
    """Lean UPGD-W step with the sigmoid gate normalized per parameter tensor.

    Identical to :func:`~alberta_framework.benchmarks.upgd_ipmnist.
    lean_upgd_w_update` except the utility scaling before the sigmoid divides
    by ``max(utility[name])`` of the *same tensor* rather than the
    network-global maximum. With a single parameter tensor the two coincide
    exactly (pinned by a unit test).

    Unlike the network-global maximum, a per-tensor utility max can be
    *exactly zero* (a tensor whose gradients are all zero, e.g. fully dead
    units); the global equation would then produce ``0/0``. In that case the
    divisor is replaced by 1, which yields the same ``sigmoid(0) = 0.5``
    gates the global rule assigns to zero utilities.
    """
    beta = hp["utility_decay"]
    step_size = hp["step_size"]
    decay = 1.0 - step_size * hp["weight_decay"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility = {
        name: beta * state.utility[name] + (1.0 - beta) * (-grads[name] * params[name])
        for name in params
    }
    bias_correction = 1.0 - jnp.power(
        jnp.asarray(beta, dtype=jnp.float32), count.astype(jnp.float32)
    )
    new_params = {}
    for name in params:
        local_max = jnp.max(utility[name])
        safe_max = jnp.where(local_max == 0.0, 1.0, local_max)
        gate = jax.nn.sigmoid((utility[name] / bias_correction) / safe_max)
        new_params[name] = params[name] * decay - step_size * (
            (grads[name] + noise[name]) * (1.0 - gate)
        )
    return new_params, LeanUPGDState(utility=utility, step=count)  # type: ignore[call-arg]


def _make_upgd_w_localgate_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> LeanUPGDState:
        return LeanUPGDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array], state: LeanUPGDState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], LeanUPGDState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_w_localgate_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (b)/(g) CBP-style dormant-unit recycling on UPGD-W and AdamW
# =============================================================================


@chex.dataclass(frozen=True)
class CBPState:
    """Per-unit recycling state for the two hidden layers of the protocol MLP."""

    util1: Array
    util2: Array
    age1: Array
    age2: Array
    accumulator: Array  # shape (2,)


def _init_cbp_state(config_hidden1: int, config_hidden2: int) -> CBPState:
    return CBPState(  # type: ignore[call-arg]
        util1=jnp.zeros(config_hidden1, dtype=jnp.float32),
        util2=jnp.zeros(config_hidden2, dtype=jnp.float32),
        age1=jnp.zeros(config_hidden1, dtype=jnp.int32),
        age2=jnp.zeros(config_hidden2, dtype=jnp.int32),
        accumulator=jnp.zeros(2, dtype=jnp.float32),
    )


@dataclass(frozen=True)
class _CBPLayerRefs:
    """Static wiring of one hidden layer inside the params dict."""

    in_weight: str
    in_bias: str
    out_weight: str

    def __post_init__(self) -> None:
        for attr in ("in_weight", "in_bias", "out_weight"):
            val = getattr(self, attr)
            if type(val) is not str or not val:
                raise ValueError(f"{attr} must be a non-empty string")


_CBP_LAYERS = (
    _CBPLayerRefs(in_weight="w1", in_bias="b1", out_weight="w2"),
    _CBPLayerRefs(in_weight="w2", in_bias="b2", out_weight="w3"),
)


def cbp_maybe_replace_layer(
    params: dict[str, Array],
    opt_arrays: dict[str, Array] | None,
    utility: Array,
    age: Array,
    accumulator: Array,
    layer: _CBPLayerRefs,
    key: Array,
    replacement_rate: float,
    maturity_threshold: int,
) -> tuple[dict[str, Array], dict[str, Array] | None, Array, Array, Array]:
    """Accumulate the replacement budget and recycle at most one unit.

    ``opt_arrays`` maps parameter names to *stacked* per-element optimizer
    state arrays of shape ``(k, *param_shape)``; the recycled unit's slices
    are reset to zero. Incoming weights are redrawn from the protocol's
    PyTorch-default uniform init; the incoming bias, outgoing weights,
    utility, and age reset to zero.
    """
    n_units = utility.shape[0]
    new_accumulator = accumulator + replacement_rate * n_units
    mature = age >= jnp.asarray(maturity_threshold, dtype=age.dtype)
    fire = jnp.logical_and(new_accumulator >= 1.0, jnp.any(mature))
    masked = jnp.where(mature, utility, jnp.inf)
    idx = jnp.argmin(masked).astype(jnp.int32)

    w_in = params[layer.in_weight]
    fan_in = w_in.shape[0]
    bound = 1.0 / math.sqrt(fan_in)
    fresh_col = jr.uniform(key, (fan_in,), jnp.float32, -bound, bound)
    new_params = dict(params)
    new_params[layer.in_weight] = w_in.at[:, idx].set(
        jnp.where(fire, fresh_col, w_in[:, idx])
    )
    b_in = params[layer.in_bias]
    new_params[layer.in_bias] = b_in.at[idx].set(jnp.where(fire, 0.0, b_in[idx]))
    w_out = params[layer.out_weight]
    new_params[layer.out_weight] = w_out.at[idx, :].set(
        jnp.where(fire, jnp.zeros(w_out.shape[1], dtype=w_out.dtype), w_out[idx, :])
    )

    new_opt_arrays = opt_arrays
    if opt_arrays is not None:
        new_opt_arrays = dict(opt_arrays)
        stack_in = opt_arrays[layer.in_weight]
        new_opt_arrays[layer.in_weight] = stack_in.at[:, :, idx].set(
            jnp.where(fire, jnp.zeros_like(stack_in[:, :, idx]), stack_in[:, :, idx])
        )
        stack_bias = opt_arrays[layer.in_bias]
        new_opt_arrays[layer.in_bias] = stack_bias.at[:, idx].set(
            jnp.where(fire, jnp.zeros_like(stack_bias[:, idx]), stack_bias[:, idx])
        )
        stack_out = opt_arrays[layer.out_weight]
        new_opt_arrays[layer.out_weight] = stack_out.at[:, idx, :].set(
            jnp.where(fire, jnp.zeros_like(stack_out[:, idx, :]), stack_out[:, idx, :])
        )

    new_utility = utility.at[idx].set(jnp.where(fire, 0.0, utility[idx]))
    new_age = age.at[idx].set(jnp.where(fire, jnp.int32(0), age[idx]))
    new_accumulator = jnp.where(fire, new_accumulator - 1.0, new_accumulator)
    return new_params, new_opt_arrays, new_utility, new_age, new_accumulator


def _cbp_update(
    params: dict[str, Array],
    opt_arrays: dict[str, Array] | None,
    cbp: CBPState,
    a1: Array,
    da1: Array,
    a2: Array,
    da2: Array,
    key: Array,
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], dict[str, Array] | None, CBPState]:
    """Utility EMA + age update, then at most one replacement per layer."""
    decay = hp["cbp_decay_rate"]
    util1 = decay * cbp.util1 + (1.0 - decay) * jnp.abs(a1 * da1)
    util2 = decay * cbp.util2 + (1.0 - decay) * jnp.abs(a2 * da2)
    age1 = cbp.age1 + 1
    age2 = cbp.age2 + 1
    key1, key2 = jr.split(key)
    maturity = int(hp["cbp_maturity_threshold"])
    rate = hp["cbp_replacement_rate"]
    params, opt_arrays, util1, age1, acc0 = cbp_maybe_replace_layer(
        params, opt_arrays, util1, age1, cbp.accumulator[0], _CBP_LAYERS[0], key1, rate, maturity
    )
    params, opt_arrays, util2, age2, acc1 = cbp_maybe_replace_layer(
        params, opt_arrays, util2, age2, cbp.accumulator[1], _CBP_LAYERS[1], key2, rate, maturity
    )
    new_cbp = CBPState(  # type: ignore[call-arg]
        util1=util1,
        util2=util2,
        age1=age1,
        age2=age2,
        accumulator=jnp.stack([acc0, acc1]),
    )
    return params, opt_arrays, new_cbp


def _make_sgd_cbp_budget_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Paper-step-size SGD with current fixed-capacity CBP recycling."""
    if dict(hp) != registered_bounded_elastic_hyperparameters("bounded_fixed_cbp"):
        raise ValueError("fixed CBP hyperparameters are not the registered budget arm")

    def init_fn(params: dict[str, Array]) -> CBPState:
        return _init_cbp_state(params["w1"].shape[1], params["w2"].shape[1])

    def full_step(
        params: dict[str, Array], state: CBPState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], CBPState, StepMetrics]:
        def loss_with_activations(
            current: dict[str, Array], observation: Array, label: Array
        ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
            logits, _, hidden1, z2, hidden2 = _forward_with_activations(
                current, observation
            )
            loss = -jax.nn.log_softmax(logits)[label]
            return loss, (logits, hidden1, z2, hidden2)

        (loss, (logits, hidden1, z2, hidden2)), grads = jax.value_and_grad(
            loss_with_activations, has_aux=True
        )(params, x, y)
        grad_hidden1, grad_hidden2 = _activation_loss_grads(params, logits, y, z2)
        candidate = {
            name: value - hp["step_size"] * grads[name] for name, value in params.items()
        }
        candidate, _, new_state = _cbp_update(
            candidate,
            None,
            state,
            hidden1,
            grad_hidden1,
            hidden2,
            grad_hidden2,
            key,
            hp,
        )
        return candidate, new_state, _step_metrics(candidate, x, y, loss, logits)

    return init_fn, full_step


@chex.dataclass(frozen=True)
class UPGDCBPState:
    """Lean UPGD state plus CBP recycling state."""

    utility: dict[str, Array]
    step: Array
    cbp: CBPState


def _make_upgd_cbp_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]
    lean_hp = {
        name: hp[name] for name in ("step_size", "utility_decay", "noise_std", "weight_decay")
    }

    def init_fn(params: dict[str, Array]) -> UPGDCBPState:
        return UPGDCBPState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            cbp=_init_cbp_state(params["w1"].shape[1], params["w2"].shape[1]),
        )

    def full_step(
        params: dict[str, Array], state: UPGDCBPState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDCBPState, StepMetrics]:
        key_noise, key_cbp = jr.split(key)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        _, _, a1, z2, a2 = _forward_with_activations(params, x)
        da1, da2 = _activation_loss_grads(params, logits, y, z2)
        noise = _sorted_flat_noise(key_noise, params, noise_std)
        lean_state = LeanUPGDState(  # type: ignore[call-arg]
            utility=state.utility, step=state.step
        )
        new_params, new_lean = lean_upgd_w_update(params, lean_state, grads, noise, lean_hp)
        opt_arrays = {name: new_lean.utility[name][None, ...] for name in new_params}
        new_params, updated_opt_arrays, new_cbp = _cbp_update(
            new_params, opt_arrays, state.cbp, a1, da1, a2, da2, key_cbp, hp
        )
        assert updated_opt_arrays is not None
        new_utility = {name: updated_opt_arrays[name][0] for name in new_params}
        metrics = _step_metrics(new_params, x, y, loss, logits)
        return new_params, UPGDCBPState(  # type: ignore[call-arg]
            utility=new_utility, step=new_lean.step, cbp=new_cbp
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class AdamCBPState:
    """Per-element Adam moments/counts plus CBP recycling state."""

    m: dict[str, Array]
    v: dict[str, Array]
    count: dict[str, Array]
    cbp: CBPState


def adam_elem_step(
    param: Array,
    m: Array,
    v: Array,
    count: Array,
    grad: Array,
    hp: Mapping[str, float],
) -> tuple[Array, Array, Array, Array]:
    """Adam *delta* with per-element bias-correction counts (not applied).

    Returns ``(step, new_m, new_v, new_count)`` so gated variants can scale
    the applied delta without touching the moment statistics
    (:func:`guarded_adam_update`); :func:`adam_elem_update` applies it as
    ``param - step``.
    """
    new_count = count + 1.0
    new_m = hp["beta1"] * m + (1.0 - hp["beta1"]) * grad
    new_v = hp["beta2"] * v + (1.0 - hp["beta2"]) * grad * grad
    m_hat = new_m / (1.0 - jnp.power(jnp.float32(hp["beta1"]), new_count))
    v_hat = new_v / (1.0 - jnp.power(jnp.float32(hp["beta2"]), new_count))
    step = hp["step_size"] * m_hat / (jnp.sqrt(v_hat) + hp["eps"])
    if hp["weight_decay"] != 0.0:
        step = step + hp["step_size"] * hp["weight_decay"] * param
    return step, new_m, new_v, new_count


def adam_elem_update(
    param: Array,
    m: Array,
    v: Array,
    count: Array,
    grad: Array,
    hp: Mapping[str, float],
) -> tuple[Array, Array, Array, Array]:
    """Adam step with per-element bias-correction counts.

    Matches ``baseline_optimizers.Adam.update_from_gradient`` exactly when
    every element shares the same count (pinned by a unit test); per-element
    counts let CBP restart bias correction for recycled units only.
    """
    step, new_m, new_v, new_count = adam_elem_step(param, m, v, count, grad, hp)
    return param - step, new_m, new_v, new_count


def _make_adamw_cbp_learner(
    hp: Mapping[str, float],
    *,
    reset_recycled_optimizer: bool = True,
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """AdamW+CBP; ``reset_recycled_optimizer=False`` keeps stale per-unit
    Adam moments/counts across CBP replacements (the ``adamw_cbp_noreset``
    dissection arm)."""

    def init_fn(params: dict[str, Array]) -> AdamCBPState:
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        return AdamCBPState(  # type: ignore[call-arg]
            m=dict(zeros),
            v=dict(zeros),
            count={name: jnp.zeros_like(value) for name, value in params.items()},
            cbp=_init_cbp_state(params["w1"].shape[1], params["w2"].shape[1]),
        )

    def full_step(
        params: dict[str, Array], state: AdamCBPState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], AdamCBPState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        _, _, a1, z2, a2 = _forward_with_activations(params, x)
        da1, da2 = _activation_loss_grads(params, logits, y, z2)
        new_params: dict[str, Array] = {}
        new_m: dict[str, Array] = {}
        new_v: dict[str, Array] = {}
        new_count: dict[str, Array] = {}
        for name, value in params.items():
            new_params[name], new_m[name], new_v[name], new_count[name] = adam_elem_update(
                value, state.m[name], state.v[name], state.count[name], grads[name], hp
            )
        if reset_recycled_optimizer:
            opt_arrays: dict[str, Array] | None = {
                name: jnp.stack([new_m[name], new_v[name], new_count[name]])
                for name in new_params
            }
            new_params, opt_arrays, new_cbp = _cbp_update(
                new_params, opt_arrays, state.cbp, a1, da1, a2, da2, key, hp
            )
            assert opt_arrays is not None
            new_m = {name: opt_arrays[name][0] for name in new_params}
            new_v = {name: opt_arrays[name][1] for name in new_params}
            new_count = {name: opt_arrays[name][2] for name in new_params}
        else:
            new_params, _, new_cbp = _cbp_update(
                new_params, None, state.cbp, a1, da1, a2, da2, key, hp
            )
        metrics = _step_metrics(new_params, x, y, loss, logits)
        return new_params, AdamCBPState(  # type: ignore[call-arg]
            m=new_m, v=new_v, count=new_count, cbp=new_cbp
        ), metrics

    return init_fn, full_step


def _make_adamw_cbp_noreset_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_adamw_cbp_learner(hp, reset_recycled_optimizer=False)


# =============================================================================
# (m) Composition: AdamW+CBP behind EMA input normalization
# =============================================================================


@chex.dataclass(frozen=True)
class AdamCBPNormState:
    """AdamW+CBP state plus the EMA input-normalizer state."""

    m: dict[str, Array]
    v: dict[str, Array]
    count: dict[str, Array]
    cbp: CBPState
    norm: EMANormState


def _make_adamw_cbp_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """The exact ``adamw_cbp`` update behind ``upgd_ema_norm``'s normalizer.

    The EMA input-normalization step (:func:`ema_normalize` equations,
    ``norm_decay``/``norm_epsilon``, per-step state threading) is identical
    to ``upgd_ema_norm``'s (pinned by unit tests); everything downstream —
    gradients, activations for CBP utility, the per-element AdamW step, and
    the recycling with optimizer-state reset — is the ``adamw_cbp`` step run
    on the normalized input. With ``norm_enabled = 0`` the normalizer is
    skipped entirely (state untouched) and the arm reduces bit-exactly to
    ``adamw_cbp`` (pinned by a unit test).
    """
    decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]
    normalize = hp.get("norm_enabled", 1.0) != 0.0

    def init_fn(params: dict[str, Array]) -> AdamCBPNormState:
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        input_dim = params["w1"].shape[0]
        return AdamCBPNormState(  # type: ignore[call-arg]
            m=dict(zeros),
            v=dict(zeros),
            count={name: jnp.zeros_like(value) for name, value in params.items()},
            cbp=_init_cbp_state(params["w1"].shape[1], params["w2"].shape[1]),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array], state: AdamCBPNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], AdamCBPNormState, StepMetrics]:
        if normalize:
            x_in, new_norm = ema_normalize(state.norm, x, decay, epsilon)
        else:
            x_in, new_norm = x, state.norm
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_in, y
        )
        _, _, a1, z2, a2 = _forward_with_activations(params, x_in)
        da1, da2 = _activation_loss_grads(params, logits, y, z2)
        new_params: dict[str, Array] = {}
        new_m: dict[str, Array] = {}
        new_v: dict[str, Array] = {}
        new_count: dict[str, Array] = {}
        for name, value in params.items():
            new_params[name], new_m[name], new_v[name], new_count[name] = adam_elem_update(
                value, state.m[name], state.v[name], state.count[name], grads[name], hp
            )
        opt_arrays: dict[str, Array] | None = {
            name: jnp.stack([new_m[name], new_v[name], new_count[name]])
            for name in new_params
        }
        new_params, opt_arrays, new_cbp = _cbp_update(
            new_params, opt_arrays, state.cbp, a1, da1, a2, da2, key, hp
        )
        assert opt_arrays is not None
        metrics = _step_metrics(new_params, x_in, y, loss, logits)
        return new_params, AdamCBPNormState(  # type: ignore[call-arg]
            m={name: opt_arrays[name][0] for name in new_params},
            v={name: opt_arrays[name][1] for name in new_params},
            count={name: opt_arrays[name][2] for name in new_params},
            cbp=new_cbp,
            norm=new_norm,
        ), metrics

    return init_fn, full_step


# =============================================================================
# (j) Guarded AdamW+CBP: utility protection on Adam's delta, CBP regeneration
# =============================================================================


@chex.dataclass(frozen=True)
class GuardedAdamCBPState:
    """Per-element Adam moments/counts, UPGD utility EMA + clock, CBP state."""

    m: dict[str, Array]
    v: dict[str, Array]
    count: dict[str, Array]
    utility: dict[str, Array]
    step: Array
    cbp: CBPState


def guarded_adam_update(
    params: dict[str, Array],
    m: dict[str, Array],
    v: dict[str, Array],
    count: dict[str, Array],
    grads: dict[str, Array],
    gate: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], dict[str, Array]]:
    """Adam step whose *applied* delta is scaled by ``1 - guard_scale * gate``.

    Protection only: the moment statistics see the raw gradients (exactly as
    UPGD's gate scales the applied update, not the utility bookkeeping), and
    there is no perturbation term. With ``guard_scale = 0`` the gating is
    skipped entirely and every parameter takes the plain
    :func:`adam_elem_step` delta, so the ``guarded_cbp_adam`` arm reduces
    bit-exactly to ``adamw_cbp`` (pinned by a unit test).
    """
    guard = hp["guard_scale"]
    new_params: dict[str, Array] = {}
    new_m: dict[str, Array] = {}
    new_v: dict[str, Array] = {}
    new_count: dict[str, Array] = {}
    for name in params:
        step, new_m[name], new_v[name], new_count[name] = adam_elem_step(
            params[name], m[name], v[name], count[name], grads[name], hp
        )
        if guard == 0.0:
            new_params[name] = params[name] - step
        else:
            new_params[name] = params[name] - step * (1.0 - guard * gate[name])
    return new_params, new_m, new_v, new_count


def _make_guarded_cbp_adam_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    def init_fn(params: dict[str, Array]) -> GuardedAdamCBPState:
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        return GuardedAdamCBPState(  # type: ignore[call-arg]
            m=dict(zeros),
            v=dict(zeros),
            count={name: jnp.zeros_like(value) for name, value in params.items()},
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            cbp=_init_cbp_state(params["w1"].shape[1], params["w2"].shape[1]),
        )

    def full_step(
        params: dict[str, Array],
        state: GuardedAdamCBPState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], GuardedAdamCBPState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        _, _, a1, z2, a2 = _forward_with_activations(params, x)
        da1, da2 = _activation_loss_grads(params, logits, y, z2)
        clock = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, clock, hp["utility_decay"]
        )
        new_params, new_m, new_v, new_count = guarded_adam_update(
            params, state.m, state.v, state.count, grads, gate, hp
        )
        # Recycled units also reset their guard utility (row 3): fresh units
        # restart at the neutral sigmoid(0) = 0.5 gate.
        opt_arrays: dict[str, Array] | None = {
            name: jnp.stack(
                [new_m[name], new_v[name], new_count[name], utility[name]]
            )
            for name in new_params
        }
        new_params, opt_arrays, new_cbp = _cbp_update(
            new_params, opt_arrays, state.cbp, a1, da1, a2, da2, key, hp
        )
        assert opt_arrays is not None
        metrics = _step_metrics(new_params, x, y, loss, logits)
        return new_params, GuardedAdamCBPState(  # type: ignore[call-arg]
            m={name: opt_arrays[name][0] for name in new_params},
            v={name: opt_arrays[name][1] for name in new_params},
            count={name: opt_arrays[name][2] for name in new_params},
            utility={name: opt_arrays[name][3] for name in new_params},
            step=clock,
            cbp=new_cbp,
        ), metrics

    return init_fn, full_step


# =============================================================================
# (k) Perturbation dissection: lean UPGD-W with sigma = 0
# =============================================================================


def _make_upgd_w_sigma0_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Zero-noise lean UPGD-W: utility gate + decoupled decay, no perturbation.

    Skips the per-step 282,160-element normal draw entirely (~85-90% of the
    UPGD-W step cost) instead of drawing and scaling by zero; the per-step
    RNG chain (``key, step_key = split(key)``) is untouched, so the
    trajectory is bit-exact against the control factory run with
    ``noise_std = 0`` (pinned by a unit test).
    """
    if hp["noise_std"] != 0.0:
        raise ValueError(
            f"upgd_w_sigma0 requires noise_std=0, got {hp['noise_std']!r}"
        )

    def init_fn(params: dict[str, Array]) -> LeanUPGDState:
        return LeanUPGDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array], state: LeanUPGDState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], LeanUPGDState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        new_params, new_state = lean_upgd_w_update(params, state, grads, zeros, dict(hp))
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (l) UPGD-W gated by passive IDBD step-size relevance instead of -w*g utility
# =============================================================================


@chex.dataclass(frozen=True)
class UPGDAlphaGateState:
    """Clock plus the passive IDBD statistics that drive the protection gate."""

    step: Array
    log_alpha: dict[str, Array]
    trace: dict[str, Array]


def upgd_alpha_utility_update(
    params: dict[str, Array],
    state: UPGDAlphaGateState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDAlphaGateState]:
    """UPGD-W step whose protection signal is per-weight step-size relevance.

    An IDBD ``log_alpha``/trace pair (Meyer error-free equations, exactly as
    :func:`upgd_idbd_update`) is maintained as a *passive statistic* on the
    raw loss gradient — it is never applied as a step size; the applied step
    keeps the protocol's fixed ``step_size``, decoupled decay, and
    perturbation. Protection instead reads each weight's log-step-size drift
    from its initial value ``la0 = ln(initial_step_size)``:

    - ``log_alpha_i += meta * g_i * h_i`` (old trace), clipped to
      ``[-10, 0]``; ``h_i = h_i * max(0, 1 - alpha_i * g_i^2) + alpha_i * g_i``.
    - ``s_i = log_alpha_i - la0``; ``gate_i = sigmoid(s_i / max_j |s_j|)``
      (network-global normalizer, mirroring UPGD's global-max gate; when all
      drifts are zero the gate is exactly 0.5). The normalization is
      scale-free — only the *ordering and relative size* of drifts matters,
      the rank-like reading of "alpha as relevance".
    - ``w_i' = w_i * (1 - lr*wd) - lr * (g_i + xi_i) * (1 - gate_i)``.

    Weights whose gradients correlate over time (consistent learners) grow
    ``log_alpha`` and are protected; weights whose gradients decorrelate
    (e.g. the input layer right after a permutation switch) *shed* protection
    because sign-alternating meta-gradients drive ``log_alpha`` down. With
    ``meta_step_size = 0`` every drift stays zero and the update reduces
    bit-exactly to the closed-form half-gated step (pinned by a unit test).
    """
    step_size = hp["step_size"]
    decay = 1.0 - step_size * hp["weight_decay"]
    meta = hp["meta_step_size"]
    la0 = math.log(hp["initial_step_size"])
    count = state.step + jnp.array(1, dtype=jnp.int32)
    new_log_alpha: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        g = grads[name]
        la = _clip_finite_log_alpha(
            state.log_alpha[name], meta * g * state.trace[name]
        )
        alpha = jnp.exp(la)
        new_log_alpha[name] = la
        new_trace[name] = state.trace[name] * jnp.maximum(0.0, 1.0 - alpha * g * g) + alpha * g
    drift = {name: new_log_alpha[name] - la0 for name in params}
    drift_max = jnp.max(
        jnp.stack([jnp.max(jnp.abs(drift[name])) for name in sorted(params)])
    )
    safe_max = jnp.where(drift_max > 0.0, drift_max, 1.0)
    new_params: dict[str, Array] = {}
    for name in params:
        gate = jax.nn.sigmoid(
            jnp.where(drift_max > 0.0, drift[name] / safe_max, 0.0)
        )
        new_params[name] = params[name] * decay - step_size * (
            (grads[name] + noise[name]) * (1.0 - gate)
        )
    return new_params, UPGDAlphaGateState(  # type: ignore[call-arg]
        step=count, log_alpha=new_log_alpha, trace=new_trace
    )


def _make_upgd_alpha_utility_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]
    la0 = math.log(hp["initial_step_size"])

    def init_fn(params: dict[str, Array]) -> UPGDAlphaGateState:
        return UPGDAlphaGateState(  # type: ignore[call-arg]
            step=jnp.array(0, dtype=jnp.int32),
            log_alpha={
                name: jnp.full_like(value, la0) for name, value in params.items()
            },
            trace={name: jnp.zeros_like(value) for name, value in params.items()},
        )

    def full_step(
        params: dict[str, Array], state: UPGDAlphaGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDAlphaGateState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_alpha_utility_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (o) Update-rule family swaps under the sigma0_ndecay099 champion conditioning
# =============================================================================

#: Quintic Newton-Schulz coefficients from the Muon reference implementation
#: (Jordan et al.); tuned for slope at zero, so the iteration oscillates the
#: singular values into a band around 1 rather than converging monotonically.
_MUON_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def _newton_schulz_orthogonalize(matrix: Array, n_steps: int) -> Array:
    """Approximately orthogonalize a 2-D matrix via quintic Newton-Schulz.

    The Muon reference procedure: normalize by the Frobenius norm (plus a
    1e-7 guard), run ``n_steps`` iterations of
    ``X <- a*X + (b*A + c*A@A) @ X`` with ``A = X @ X^T`` and the
    :data:`_MUON_NS_COEFFS` coefficients, operating on the transposed matrix
    whenever it has more rows than columns so ``A`` is the smaller Gram
    matrix. Scale-invariant by construction (the Frobenius normalization
    absorbs any positive scalar on the input).
    """
    a, b, c = _MUON_NS_COEFFS
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.T if transposed else matrix
    x = x / (jnp.linalg.norm(x) + 1e-7)
    for _ in range(n_steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return jnp.asarray(x.T if transposed else x)


def _init_input_norm_state(params: dict[str, Array]) -> EMANormState:
    """Fresh EMA input-normalizer state sized from the first-layer fan-in."""
    input_dim = params["w1"].shape[0]
    return EMANormState(  # type: ignore[call-arg]
        mean=jnp.zeros(input_dim, dtype=jnp.float32),
        var=jnp.ones(input_dim, dtype=jnp.float32),
        count=jnp.array(0.0, dtype=jnp.float32),
    )


@chex.dataclass(frozen=True)
class ColNormGateState:
    """UPGD utility EMA/clock, per-fan-in gradient mean-square EMAs, normalizer.

    ``vcol`` holds one entry per parameter: shape ``(fan_in,)`` for the 2-D
    weights (one EMA per input dimension, axis 0 of the ``(fan_in, fan_out)``
    protocol orientation) and the full parameter shape for biases.
    """

    utility: dict[str, Array]
    step: Array
    vcol: dict[str, Array]
    norm: EMANormState


def _make_colnorm_gate_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Column-wise RMS-scaled gated SGD behind the champion's conditioning.

    The ``sigma0_ndecay099`` champion's EMA input normalizer (decay
    ``norm_decay``) and exact UPGD utility gate
    (:func:`_upgd_utility_and_gate`) are kept; only the descent direction
    changes. Per 2-D weight ``W`` of shape ``(fan_in, fan_out)`` the state
    keeps ``v_col``, an EMA (``col_decay``) of the per-input-dimension
    mean-square gradient ``mean_j(G_ij^2)``; at batch size 1 the dense-layer
    gradient is the rank-1 outer product of the input activation and the
    backprop delta, so ``sqrt(v_col)`` is input/hidden-activation
    conditioning expressed at the weight level. The applied step is

    ``W <- W * (1 - lr*wd) - lr * (G * (1 - gate)) / (sqrt(v_col) + eps)``

    with ``v_col`` broadcast along the fan-out axis. Biases are scaled by an
    EMA of the per-element squared gradient (same decay/epsilon) — the exact
    1-D specialization of the column statistic. No perturbation and no bias
    correction on the EMA (RMSProp-style; documented, not an oversight); the
    RNG key is deliberately unused.
    """
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    col_decay = hp["col_decay"]
    col_epsilon = hp["col_epsilon"]

    def init_fn(params: dict[str, Array]) -> ColNormGateState:
        return ColNormGateState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            vcol={
                name: jnp.zeros(value.shape[0] if value.ndim == 2 else value.shape,
                                dtype=jnp.float32)
                for name, value in params.items()
            },
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: ColNormGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], ColNormGateState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        count = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, count, utility_decay
        )
        new_params: dict[str, Array] = {}
        new_vcol: dict[str, Array] = {}
        for name in params:
            g = grads[name]
            if params[name].ndim == 2:
                stat = jnp.mean(g * g, axis=1)
                v = col_decay * state.vcol[name] + (1.0 - col_decay) * stat
                denom = jnp.sqrt(v)[:, None] + col_epsilon
            else:
                v = col_decay * state.vcol[name] + (1.0 - col_decay) * (g * g)
                denom = jnp.sqrt(v) + col_epsilon
            new_params[name] = params[name] * param_decay - step_size * (
                g * (1.0 - gate[name]) / denom
            )
            new_vcol[name] = v
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, ColNormGateState(  # type: ignore[call-arg]
            utility=utility, step=count, vcol=new_vcol, norm=new_norm
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class MuonGateState:
    """UPGD utility EMA/clock, per-weight momentum buffers, input normalizer.

    ``momentum`` carries entries for the 2-D weight matrices only; biases
    take the plain gated-SGD step and keep no optimizer state.
    """

    utility: dict[str, Array]
    step: Array
    momentum: dict[str, Array]
    norm: EMANormState


def _make_muon_gate_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Muon-style orthogonalized gated update behind the champion's conditioning.

    EMA input normalizer and UPGD utility gate exactly as
    ``sigma0_ndecay099``; the 2-D weight step is the Muon recipe: momentum
    ``M <- mu * M + G`` (``mu = muon_momentum``), Nesterov update input
    ``G + mu * M`` (with the *new* ``M``, matching the reference
    ``buf.mul_(mu).add_(g); g.add_(buf, alpha=mu)``), then
    :func:`_newton_schulz_orthogonalize` (``muon_ns_steps`` iterations,
    :data:`_MUON_NS_COEFFS`), scaled by ``sqrt(max(m, n) / min(m, n))``:

    ``W <- W * (1 - lr*wd) - lr * (1 - gate) * scale * NS(G + mu*M)``

    The gate multiplies the orthogonalized direction elementwise (protection
    stays per-weight; orthogonalization sees raw momentum). Biases take the
    plain gated SGD step with the same decoupled decay. No perturbation; the
    RNG key is deliberately unused.
    """
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    mu = hp["muon_momentum"]
    ns_steps = int(hp["muon_ns_steps"])

    def init_fn(params: dict[str, Array]) -> MuonGateState:
        return MuonGateState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            momentum={
                name: jnp.zeros_like(value)
                for name, value in params.items()
                if value.ndim == 2
            },
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: MuonGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], MuonGateState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        count = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, count, utility_decay
        )
        new_params: dict[str, Array] = {}
        new_momentum: dict[str, Array] = {}
        for name in params:
            keep = 1.0 - gate[name]
            g = grads[name]
            if params[name].ndim == 2:
                momentum = mu * state.momentum[name] + g
                direction = _newton_schulz_orthogonalize(g + mu * momentum, ns_steps)
                m_dim, n_dim = params[name].shape
                scale = math.sqrt(max(m_dim, n_dim) / min(m_dim, n_dim))
                new_params[name] = params[name] * param_decay - step_size * (
                    keep * scale * direction
                )
                new_momentum[name] = momentum
            else:
                new_params[name] = params[name] * param_decay - step_size * (keep * g)
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, MuonGateState(  # type: ignore[call-arg]
            utility=utility, step=count, momentum=new_momentum, norm=new_norm
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class LionGateState:
    """UPGD utility EMA/clock, Lion momentum, and the input-normalizer state."""

    utility: dict[str, Array]
    step: Array
    momentum: dict[str, Array]
    norm: EMANormState


def _make_lion_gate_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Gated Lion behind the champion's conditioning.

    EMA input normalizer and UPGD utility gate exactly as
    ``sigma0_ndecay099``; the descent direction is Lion (Chen et al. 2023),
    with the published two-beta form — the sign direction interpolates the
    *pre-update* momentum with the fresh gradient:

    - ``c = lion_beta1 * m + (1 - lion_beta1) * g`` (old ``m``),
    - ``w <- w * (1 - lr*wd) - lr * (1 - gate) * sign(c)``,
    - ``m <- lion_beta2 * m + (1 - lion_beta2) * g``.

    Sign updates are scale-free, so the arm runs at ~0.1x the champion's
    step size with a correspondingly larger decoupled decay. Applied to every
    parameter (weights and biases). No perturbation; the RNG key is
    deliberately unused.
    """
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    beta1 = hp["lion_beta1"]
    beta2 = hp["lion_beta2"]

    def init_fn(params: dict[str, Array]) -> LionGateState:
        return LionGateState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            momentum={name: jnp.zeros_like(value) for name, value in params.items()},
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: LionGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], LionGateState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        count = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, count, utility_decay
        )
        new_params: dict[str, Array] = {}
        new_momentum: dict[str, Array] = {}
        for name in params:
            g = grads[name]
            interpolated = beta1 * state.momentum[name] + (1.0 - beta1) * g
            new_params[name] = params[name] * param_decay - step_size * (
                (1.0 - gate[name]) * jnp.sign(interpolated)
            )
            new_momentum[name] = beta2 * state.momentum[name] + (1.0 - beta2) * g
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, LionGateState(  # type: ignore[call-arg]
            utility=utility, step=count, momentum=new_momentum, norm=new_norm
        ), metrics

    return init_fn, full_step


# =============================================================================
# (r) rff_rls — no-backprop random-features + RLS tracking control
# =============================================================================


#: Fixed domain constant for deriving the frozen random-feature draw (the
#: screening init contract passes no RNG key, so the draw is folded from the
#: seed-dependent protocol init weights; see :func:`_make_rff_rls_learner`).
_RFF_KEY_DOMAIN = 0x52464601


@chex.dataclass(frozen=True)
class RFFRLSState:
    """Frozen random-Fourier projection plus streaming-RLS readout state.

    ``omega`` (m, input_dim) and ``phase`` (m,) are drawn once at init and
    never updated. ``p`` is the (m, m) inverse feature-correlation matrix and
    ``wout`` the (m, n_classes) one-vs-all readout — the only learned
    quantities. ``norm`` is the champion's EMA input normalizer.
    """

    omega: Array
    phase: Array
    p: Array
    wout: Array
    norm: EMANormState


def _make_rff_rls_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Random Fourier features + exponential-window recursive least squares.

    The pre-registered *existential control* for the screening lane: NO
    backprop, NO MLP — the protocol MLP parameters passed in by the harness
    are returned untouched every step (inert ballast kept only for harness
    shape compatibility). If a fixed random projection with a
    forgetting-factor linear readout matches the deep arms, the benchmark
    measures tracking rather than learning.

    Pipeline per step (predict-then-update, matching the harness ordering):

    1. ``x_norm = ema_normalize(x)`` with the champion's conditioning
       (``norm_decay`` 0.99, ``norm_epsilon`` 1e-8).
    2. ``z = clip(x_norm, -rff_clip, rff_clip)`` — near-zero-variance pixels
       produce extreme z-scores under the 1e-8-epsilon normalizer, which
       would randomize the cosine phases; clipping bounds the phase
       contribution of any single pixel.
    3. ``phi = sqrt(2/m) * cos(Omega @ z + b)`` with frozen
       ``Omega ~ N(0, rff_gamma * I)`` and ``b ~ Uniform[0, 2pi)``.
       ``rff_gamma`` sets the RBF kernel bandwidth: pre-activation variance
       is ``gamma * ||z||^2``; with 784-dim clipped z-scores the calibrated
       value is ~1e-3 (the original 0.05 put pre-activation std at ~6
       radians — pure phase noise — and scored 0.177, barely above chance;
       the 2-task diagnostic sweep recovered 0.79→0.81 at 5e-3→2e-4).
    3. Pre-update prediction ``argmax(Wout.T @ phi)`` scores the protocol's
       online accuracy.
    4. Sherman-Morrison RLS with forgetting factor ``rls_lambda``:
       ``Pp = P @ phi``; ``k = Pp / (lam + phi @ Pp)``;
       ``err = onehot(y) - Wout.T @ phi``; ``Wout += outer(k, err)``;
       ``P = (P - outer(k, Pp)) / lam``, then symmetrized
       (``P = (P + P.T)/2``) against float32 drift.

    The reported loss is the pre-update squared error ``0.5 * ||err||^2``
    (there is no cross-entropy here; per-task losses are NOT comparable with
    the MLP arms' CE — accuracy is the protocol metric). Plasticity is the
    protocol's own post-update one-step improvement ratio computed on the
    same squared-error loss. The RNG key threaded by the harness is
    deliberately unused (the step consumes no randomness).

    Init-key deviation (documented): the screening ``LearnerInitFn`` receives
    only the protocol init params, not the seed key, so the frozen
    ``Omega``/``b`` draw folds the raw float32 bits of the seed-dependent
    ``w1`` init into a fixed domain key — deterministic given the seed,
    distinct across seeds.
    """
    m = int(hp["rff_m"])
    gamma = hp["rff_gamma"]
    clip = hp["rff_clip"]
    rls_lambda = hp["rls_lambda"]
    ridge_init = hp["rls_ridge_init"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    feature_scale = math.sqrt(2.0 / m)

    def init_fn(params: dict[str, Array]) -> RFFRLSState:
        input_dim = params["w1"].shape[0]
        n_classes = params["w3"].shape[1]
        w1_bits = jax.lax.bitcast_convert_type(
            params["w1"].astype(jnp.float32).reshape(-1), jnp.uint32
        )
        key = jr.fold_in(jr.key(jnp.uint32(_RFF_KEY_DOMAIN)), w1_bits[0])
        key = jr.fold_in(key, w1_bits[-1])
        key_omega, key_phase = jr.split(key)
        omega = jr.normal(key_omega, (m, input_dim), jnp.float32) * math.sqrt(gamma)
        phase = jr.uniform(
            key_phase, (m,), jnp.float32, 0.0, 2.0 * math.pi
        )
        return RFFRLSState(  # type: ignore[call-arg]
            omega=omega,
            phase=phase,
            p=jnp.eye(m, dtype=jnp.float32) / ridge_init,
            wout=jnp.zeros((m, n_classes), dtype=jnp.float32),
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: RFFRLSState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], RFFRLSState, StepMetrics]:
        del key  # no randomness: the projection is frozen, RLS is closed-form
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        z = jnp.clip(x_norm, -clip, clip)
        phi = feature_scale * jnp.cos(state.omega @ z + state.phase)
        logits = state.wout.T @ phi
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        y_onehot = jax.nn.one_hot(y, state.wout.shape[1], dtype=jnp.float32)
        err = y_onehot - logits
        loss = 0.5 * jnp.sum(err * err)
        pp = state.p @ phi
        gain = pp / (rls_lambda + phi @ pp)
        new_wout = state.wout + jnp.outer(gain, err)
        new_p = (state.p - jnp.outer(gain, pp)) / rls_lambda
        new_p = 0.5 * (new_p + new_p.T)
        err_after = y_onehot - new_wout.T @ phi
        loss_after = 0.5 * jnp.sum(err_after * err_after)
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
        )
        return params, RFFRLSState(  # type: ignore[call-arg]
            omega=state.omega,
            phase=state.phase,
            p=new_p,
            wout=new_wout,
            norm=new_norm,
        ), (accuracy, loss, plasticity)

    return init_fn, full_step


def _make_lin_rls_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Linear floor of the tracking control: RLS on clipped z-scores directly.

    Identical pipeline to :func:`_make_rff_rls_learner` but with the feature
    map replaced by the normalized input itself, scaled by ``1/sqrt(d)`` and
    concatenated with a constant bias feature (``m = d + 1``). No random
    projection, no nonlinearity, no backprop — the cheapest possible measure
    of how far pure linear tracking goes on this protocol. In the 2-task
    calibration diagnostic this floor already reached 0.78, i.e. the
    published-SOTA neighborhood, with no representation at all.
    """
    clip = hp["rff_clip"]
    rls_lambda = hp["rls_lambda"]
    ridge_init = hp["rls_ridge_init"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]

    def init_fn(params: dict[str, Array]) -> RFFRLSState:
        input_dim = params["w1"].shape[0]
        n_classes = params["w3"].shape[1]
        m = input_dim + 1
        return RFFRLSState(  # type: ignore[call-arg]
            omega=jnp.zeros((1, input_dim), dtype=jnp.float32),
            phase=jnp.zeros((1,), dtype=jnp.float32),
            p=jnp.eye(m, dtype=jnp.float32) / ridge_init,
            wout=jnp.zeros((m, n_classes), dtype=jnp.float32),
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: RFFRLSState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], RFFRLSState, StepMetrics]:
        del key
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        z = jnp.clip(x_norm, -clip, clip)
        phi = jnp.concatenate(
            [z / jnp.sqrt(jnp.float32(z.shape[0])), jnp.ones((1,), jnp.float32)]
        )
        logits = state.wout.T @ phi
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        y_onehot = jax.nn.one_hot(y, state.wout.shape[1], dtype=jnp.float32)
        err = y_onehot - logits
        loss = 0.5 * jnp.sum(err * err)
        pp = state.p @ phi
        gain = pp / (rls_lambda + phi @ pp)
        new_wout = state.wout + jnp.outer(gain, err)
        new_p = (state.p - jnp.outer(gain, pp)) / rls_lambda
        new_p = 0.5 * (new_p + new_p.T)
        err_after = y_onehot - new_wout.T @ phi
        loss_after = 0.5 * jnp.sum(err_after * err_after)
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
        )
        return params, RFFRLSState(  # type: ignore[call-arg]
            omega=state.omega,
            phase=state.phase,
            p=new_p,
            wout=new_wout,
            norm=new_norm,
        ), (accuracy, loss, plasticity)

    return init_fn, full_step


# =============================================================================
# (v) Streaming naive Bayes: class-conditional diagonal Gaussians, no gradients
# =============================================================================


@chex.dataclass(frozen=True)
class NaiveBayesState:
    """Streaming class-conditional Gaussian statistics (no MLP is trained).

    ``cmean``/``cvar`` are ``(n_classes, input_dim)`` per-class feature means
    and variances under the annealed fast-EMA recurrence (equation parity
    with :func:`ema_normalize`, applied to the observed class's row only);
    ``ccount`` is each class's observed-example count (its anneal clock);
    ``prior`` is an annealed one-hot EMA class prior with scalar clock
    ``step``.
    """

    cmean: Array
    cvar: Array
    ccount: Array
    prior: Array
    step: Array


def naive_bayes_logits(state: NaiveBayesState, x: Array) -> Array:
    """Class log-posteriors (up to a shared constant) for one example.

    ``log prior_c - 0.5 * sum_i [log(2 pi var_ci) + (x_i - mu_ci)^2 / var_ci]``
    -- the diagonal-Gaussian class-conditional log-likelihood plus the log
    prior. Variances already carry the ``nb_var_epsilon`` floor from the
    update, so degenerate (constant) features contribute equally to every
    class and cancel in the argmax.
    """
    log_lik = -0.5 * jnp.sum(
        jnp.log(2.0 * math.pi * state.cvar)
        + (x[None, :] - state.cmean) ** 2 / state.cvar,
        axis=1,
    )
    return jnp.log(state.prior) + log_lik


def _make_naive_bayes_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """V3 development validation: streaming generative classifier, no gradients.

    Direction (B) made protocol-exact: online class-conditional diagonal
    Gaussians with the campaign's own annealed fast-EMA statistics. Per
    step (predict-then-update, the protocol ordering):

    1. Pre-update prediction ``argmax_c`` of :func:`naive_bayes_logits`
       scores the protocol's online accuracy; the reported loss is the
       cross-entropy of the softmax posterior at the true label (a proper
       probabilistic-classifier loss, but NOT comparable with the MLP arms'
       CE -- accuracy is the protocol metric).
    2. The observed label's row updates with the annealed fast-EMA
       (:func:`ema_normalize` equations conditioned on the class):
       effective decay ``min(nb_decay, 1 - 1/(count_c + 1))`` with count_c
       the class's own sample clock, Welford-style EMA variance floored at
       ``nb_var_epsilon``. Other class rows are bitwise untouched.
    3. The class prior is an annealed one-hot EMA on the total-step clock
       (it stays a probability vector exactly).

    Everything is closed-form and per-feature: no gradients, no backprop,
    no MLP (the protocol params pass through untouched, like ``rff_rls``),
    and the RNG key is deliberately unused. A pixel permutation permutes
    the stored per-class means/variances -- the statistics re-estimate at
    the fast-EMA timescale per class (~1/n_classes of the stream each).

    Plasticity is the protocol's post-update one-step improvement ratio on
    the same posterior cross-entropy.
    """
    decay = hp["nb_decay"]
    epsilon = hp["nb_var_epsilon"]

    def init_fn(params: dict[str, Array]) -> NaiveBayesState:
        input_dim = params["w1"].shape[0]
        n_classes = params["w3"].shape[1]
        return NaiveBayesState(  # type: ignore[call-arg]
            cmean=jnp.zeros((n_classes, input_dim), dtype=jnp.float32),
            cvar=jnp.ones((n_classes, input_dim), dtype=jnp.float32),
            ccount=jnp.zeros(n_classes, dtype=jnp.float32),
            prior=jnp.full((n_classes,), 1.0 / n_classes, dtype=jnp.float32),
            step=jnp.array(0.0, dtype=jnp.float32),
        )

    def full_step(
        params: dict[str, Array], state: NaiveBayesState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], NaiveBayesState, StepMetrics]:
        del key  # closed-form: the step consumes no randomness
        logits = naive_bayes_logits(state, x)
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        loss = -jax.nn.log_softmax(logits)[y]
        n_classes = state.prior.shape[0]
        onehot = jax.nn.one_hot(y, n_classes, dtype=jnp.float32)
        mask = onehot[:, None]
        new_ccount = state.ccount + onehot
        eff = jnp.minimum(decay, 1.0 - 1.0 / (new_ccount + 1.0))[:, None]
        delta = x[None, :] - state.cmean
        new_cmean = state.cmean + mask * (1.0 - eff) * delta
        delta2 = x[None, :] - new_cmean
        cand_var = jnp.maximum(
            eff * state.cvar + (1.0 - eff) * delta * delta2, epsilon
        )
        new_cvar = jnp.where(mask > 0.0, cand_var, state.cvar)
        new_step = state.step + 1.0
        prior_eff = jnp.minimum(decay, 1.0 - 1.0 / (new_step + 1.0))
        new_prior = prior_eff * state.prior + (1.0 - prior_eff) * onehot
        new_state = NaiveBayesState(  # type: ignore[call-arg]
            cmean=new_cmean,
            cvar=new_cvar,
            ccount=new_ccount,
            prior=new_prior,
            step=new_step,
        )
        loss_after = -jax.nn.log_softmax(naive_bayes_logits(new_state, x))[y]
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
        )
        return params, new_state, (accuracy, loss, plasticity)

    return init_fn, full_step


def _naive_bayes_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Refuse sentinel probes for the gradient-free naive-Bayes arm.

    Exactly the :func:`_rff_frozen_probe_input` situation: the deployed
    model is the streaming Gaussian statistics, not the (untouched,
    randomly initialized) protocol MLP, so probing ``mlp_logits`` would
    silently score a model that does not exist. Fail closed.
    """
    del state, observation, hyperparameters
    raise NotImplementedError(
        "sentinel probes are unsupported for the naive_bayes arm: there is "
        "no trained protocol MLP to probe (the deployed model is the "
        "streaming class-conditional Gaussian statistics)"
    )


# =============================================================================
# (w) rls_head — champion body + recursive-least-squares readout
# =============================================================================
#
# The convergence-shortfall attack (CEILING_ANALYSIS.md maps 0.029 of the
# champion's error to within-task convergence speed: plateau 0.9037 vs the
# 0.933 family asymptote).  The ``sigma0_shiftnorm_d099`` champion body
# (shift-adaptive EMA-norm decay 0.99 + utility-gated sigma-0 SGD + decoupled
# decay) is kept; the deployed readout becomes streaming recursive least
# squares on the 150-dim penultimate ReLU features (bias-augmented), which is
# exactly optimal for its squared-error objective at every step and converges
# in ~d samples instead of the SGD head's thousands (RFF+RLS precedent:
# 0.848 with a *random* 1024-dim body).
#
# Design decisions (probed by the arm family):
#
# (a) Targets: one-hot least-squares regression + argmax — the standard
#     streaming-classification practice (the rff_rls/lin_rls precedent).
#     Softmax/logistic targets admit no exact RLS recursion (IRLS-style
#     approximations forfeit the closed-form optimality that motivates the
#     arm), and distilling the parallel SGD head's softmax outputs cannot
#     beat its teacher; one-hot regression is therefore fixed by design.
# (b) Forgetting factor ``rls_lambda`` in {0.995, 0.999, 1.0} — RLS's own
#     staleness knob — plus an optional detector-driven P reset
#     (``rls_reset_frac`` <= 1 enables): when the champion's own per-feature
#     shift detector flags more than that fraction of input features in one
#     step, P resets to ``eye/ridge`` (readout weights are KEPT — the reset
#     restores estimation gain without discarding the mapping, mirroring the
#     normalizer's count reset).
# (c) Body error signal: ``head_resid=0`` trains body AND a parallel SGD
#     head with the exact champion update (safer; the six MLP tensors are
#     bit-exact the champion trajectory, pinned) — the RLS head is a pure
#     passenger readout.  ``head_resid=1`` backpropagates the RLS head's own
#     squared-error residual into the body (cleanest single error signal);
#     w3/b3 then pass through untouched, and the utility gate's global-max
#     normalization is guarded at zero utility (neutral gate 0.5) because a
#     zero readout yields exactly zero body gradients at stream start.
#
# Reduction pin: ``rls_ridge_init = inf`` gives P = 0 exactly, so the head
# is frozen at wout = 0 and every prediction is the degenerate constant
# argmax(0) = class 0 (measurable: accuracy = P[y == 0]).  The reported loss
# is the pre-update squared error 0.5*||onehot - logits||^2 (NOT comparable
# with the MLP arms' CE; accuracy is the protocol metric) and plasticity is
# the protocol's post-update one-step improvement on that same loss
# (head update only — the rff_rls precedent).


@chex.dataclass(frozen=True)
class RLSHeadState:
    """Champion-body carry plus the RLS readout on penultimate features.

    ``utility``/``step`` are the champion's utility EMA and clock (over all
    six tensors in parallel mode; the four body tensors in gated resid mode
    carry the signal, and both remain at their initial values in the no-gate
    resid ablation).  ``norm``/``fast_mean`` are the shift-adaptive
    normalizer.  ``p`` is the (h2+1, h2+1) inverse feature-correlation matrix
    and ``wout`` the (h2+1, n_classes) one-vs-all readout on bias-augmented
    penultimate features.
    """

    utility: dict[str, Array]
    step: Array
    norm: EMANormState
    fast_mean: Array
    p: Array
    wout: Array


_RLS_HEAD_BODY = ("w1", "b1", "w2", "b2")
_WHITEN_NORM_FLOOR = 1e-12
_NEWTON_RIDGE_REL = 1e-3
_NEWTON_RIDGE_ABS = 1e-6


@chex.dataclass(frozen=True)
class RLSHeadL2InitState:
    """Opt-in RLS carry with an immutable body-initialization snapshot.

    This distinct state keeps the incumbent :class:`RLSHeadState` schema
    unchanged.  ``init_params`` contains exactly the four residual-trained
    body tensors; the parallel protocol head and the RLS state are not part
    of the L2-to-initialization mechanism.
    """

    utility: dict[str, Array]
    step: Array
    norm: EMANormState
    fast_mean: Array
    p: Array
    wout: Array
    init_params: dict[str, Array]


def _rls_head_hp(**overrides: float) -> dict[str, float]:
    """Champion (``sigma0_shiftnorm_d099``) constants plus RLS-head defaults.

    ``rls_reset_frac`` defaults untriggerable (2.0 > any shifted fraction),
    which is bitwise the plain no-reset path (build-time composition,
    pinned).  ``head_resid`` selects the body error signal (0 = parallel
    champion SGD head, 1 = RLS residual).  ``gate_scale`` is a frozen
    endpoint switch, not a tuning knob: 1 keeps the incumbent utility-gated
    update and 0 selects plain decayed SGD for the residual body.
    """
    merged = {
        "step_size": 0.01,
        "weight_decay": 0.01,
        "utility_decay": 0.9999,
        "noise_std": 0.0,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
        "fast_decay": 0.9,
        "shift_k": 1.0,
        "shift_delta": 0.02,
        "shift_refractory": 0.0,
        "rls_lambda": 0.999,
        "rls_ridge_init": 1.0,
        "rls_reset_frac": 2.0,
        "rls_p_trace_cap": 0.0,
        "head_resid": 0.0,
        "gate_scale": 1.0,
    }
    merged.update(overrides)
    return merged


def _rls_head_l2init_hp() -> dict[str, float]:
    """Frozen issue-#14 endpoint: incumbent plus body L2-Init."""
    return {
        **_rls_head_hp(
            rls_lambda=1.0,
            rls_reset_frac=0.05,
            head_resid=1.0,
        ),
        "decay_to_init": 1.0,
    }


def _make_rls_head_l2init_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Build the one frozen body-only L2-Init arm, failing closed on drift."""
    expected = _rls_head_l2init_hp()
    if set(hp) != set(expected):
        raise ValueError(
            "frozen L2-Init configuration requires exactly the registered keys"
        )
    invalid = [
        name
        for name, expected_value in expected.items()
        if type(hp[name]) is not float
        or not math.isfinite(hp[name])
        or hp[name].hex() != expected_value.hex()
    ]
    if invalid:
        raise ValueError(
            "frozen L2-Init configuration differs at: " + ", ".join(invalid)
        )
    base_hp = dict(hp)
    del base_hp["decay_to_init"]
    return _make_rls_head_learner(base_hp, _decay_to_init=True)


def _make_rls_head_learner(
    hp: Mapping[str, float],
    *,
    _decay_to_init: bool = False,
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Champion body + streaming-RLS readout on the penultimate features.

    Per step (predict-then-update, the protocol ordering):

    1. ``x_norm`` via the champion's :func:`shift_adaptive_normalize`
       (identical constants and call).
    2. ``phi = concat(a2 / sqrt(h2 + 1), [1])`` from the PRE-update body
       (``a2`` = second hidden ReLU activation); pre-update prediction
       ``argmax(wout.T @ phi)`` scores the protocol's online accuracy.
    3. Body update — ``head_resid = 0``: exact champion arithmetic (the
       gated sigma-0 SGD step of ``_make_adaptive_norm_sigma0_learner``) on
       the cross-entropy gradient through the parallel SGD head w3/b3;
       ``head_resid = 1``: the same gated step on
       ``d(0.5*||onehot - wout.T @ phi||^2)/d(body)`` with ``wout`` held
       constant (w3/b3 untouched, zero-utility gate guarded to 0.5).
       The frozen ``gate_scale = 0`` endpoint instead applies plain decayed
       SGD to that residual gradient and skips all utility bookkeeping.  The
       private, opt-in L2-Init composition changes only the four residual
       body tensors' decay target; it has a distinct state and strict
       registered factory so the incumbent path and state schema remain
       unchanged.
    4. Sherman-Morrison RLS with forgetting ``rls_lambda`` (symmetrized P,
       the rff_rls equations), then the optional detector-driven P reset
       (``mean(shifted) >= rls_reset_frac`` => ``p = eye/ridge``, wout kept).

    The harness RNG key is deliberately unused (sigma-0 family, closed-form
    head).
    """
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    weight_decay = hp["weight_decay"]
    param_decay = 1.0 - step_size * weight_decay
    rls_lambda = hp["rls_lambda"]
    ridge_init = hp["rls_ridge_init"]
    reset_frac = hp["rls_reset_frac"]
    reset_enabled = reset_frac <= 1.0
    trace_cap = hp["rls_p_trace_cap"]
    cap_enabled = trace_cap > 0.0
    resid = hp["head_resid"] != 0.0
    resid_whiten = hp.get("resid_whiten", 0.0)
    if not 0.0 <= resid_whiten <= 1.0:
        raise ValueError("resid_whiten must lie in [0, 1]")
    whiten_enabled = resid_whiten > 0.0
    resid_newton = hp.get("resid_newton", 0.0)
    if resid_newton not in (0.0, 1.0):
        raise ValueError(
            "resid_newton selects a direction and must be 0.0 or 1.0"
        )
    newton = resid_newton == 1.0
    if whiten_enabled and not resid:
        raise ValueError(
            "resid_whiten is supported only for the residual body"
        )
    if whiten_enabled and _decay_to_init:
        raise ValueError("resid_whiten and L2-Init are not composed")
    gate_scale = hp.get("gate_scale", 1.0)
    if gate_scale not in (0.0, 1.0):
        raise ValueError(
            "gate_scale is a frozen ablation endpoint and must be 0.0 or 1.0"
        )
    gate_enabled = gate_scale == 1.0
    if not gate_enabled and not resid:
        raise ValueError("gate_scale=0.0 is supported only for the residual body")
    if _decay_to_init and (not resid or not gate_enabled):
        raise ValueError("L2-Init is supported only for the gated residual body")

    def normalize(
        state: EMANormState, fast_mean: Array, x: Array
    ) -> tuple[Array, EMANormState, Array, Array]:
        return shift_adaptive_normalize(
            state, fast_mean, x,
            decay=hp["norm_decay"],
            fast_decay=hp["fast_decay"],
            epsilon=hp["norm_epsilon"],
            shift_k=hp["shift_k"],
            shift_delta=hp["shift_delta"],
            shift_refractory=hp["shift_refractory"],
        )

    def init_fn(
        params: dict[str, Array],
    ) -> RLSHeadState | RLSHeadL2InitState:
        input_dim = params["w1"].shape[0]
        m = params["w2"].shape[1] + 1
        n_classes = params["w3"].shape[1]
        if _decay_to_init:
            return RLSHeadL2InitState(  # type: ignore[call-arg]
                utility={
                    name: jnp.zeros_like(value) for name, value in params.items()
                },
                step=jnp.array(0, dtype=jnp.int32),
                norm=EMANormState(  # type: ignore[call-arg]
                    mean=jnp.zeros(input_dim, dtype=jnp.float32),
                    var=jnp.ones(input_dim, dtype=jnp.float32),
                    count=jnp.zeros(input_dim, dtype=jnp.float32),
                ),
                fast_mean=jnp.zeros(input_dim, dtype=jnp.float32),
                p=jnp.eye(m, dtype=jnp.float32) / ridge_init,
                wout=jnp.zeros((m, n_classes), dtype=jnp.float32),
                init_params={name: params[name] for name in _RLS_HEAD_BODY},
            )
        return RLSHeadState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.zeros(input_dim, dtype=jnp.float32),
            ),
            fast_mean=jnp.zeros(input_dim, dtype=jnp.float32),
            p=jnp.eye(m, dtype=jnp.float32) / ridge_init,
            wout=jnp.zeros((m, n_classes), dtype=jnp.float32),
        )

    def _phi(params: dict[str, Array], x_norm: Array) -> Array:
        m = params["w2"].shape[1] + 1
        a1 = jax.nn.relu(x_norm @ params["w1"] + params["b1"])
        a2 = jax.nn.relu(a1 @ params["w2"] + params["b2"])
        scale = 1.0 / math.sqrt(m)
        return jnp.concatenate([a2 * scale, jnp.ones((1,), jnp.float32)])

    def _gated_sgd(
        params: dict[str, Array],
        grads: dict[str, Array],
        utility: dict[str, Array],
        count: Array,
        names: tuple[str, ...],
        guard_zero_max: bool,
        init_params: Mapping[str, Array] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Champion utility-EMA + global-max sigmoid gate + gated decayed SGD
        over ``names``; other tensors pass through with untouched utility."""
        new_utility = dict(utility)
        for name in names:
            new_utility[name] = utility_decay * utility[name] + (
                1.0 - utility_decay
            ) * (-grads[name] * params[name])
        bias_correction = 1.0 - jnp.power(
            jnp.asarray(utility_decay, dtype=jnp.float32), count.astype(jnp.float32)
        )
        global_max = jnp.max(
            jnp.stack([jnp.max(new_utility[name]) for name in sorted(names)])
        )
        if guard_zero_max:
            global_max = jnp.where(global_max == 0.0, 1.0, global_max)
        new_params = dict(params)
        for name in names:
            if init_params is None:
                # Keep the incumbent expression byte-for-byte unchanged.
                new_params[name] = params[name] * param_decay - step_size * (
                    grads[name]
                    * (
                        1.0
                        - jax.nn.sigmoid(
                            (new_utility[name] / bias_correction) / global_max
                        )
                    )
                )
            else:
                new_params[name] = (
                    params[name]
                    - step_size
                    * weight_decay
                    * (params[name] - init_params[name])
                    - step_size
                    * (
                        grads[name]
                        * (
                            1.0
                            - jax.nn.sigmoid(
                                (new_utility[name] / bias_correction) / global_max
                            )
                        )
                    )
                )
        return new_params, new_utility

    def full_step(
        params: dict[str, Array],
        state: RLSHeadState | RLSHeadL2InitState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[
        dict[str, Array], RLSHeadState | RLSHeadL2InitState, StepMetrics
    ]:
        del key  # sigma-0 body, closed-form head: no randomness consumed
        if _decay_to_init:
            if not isinstance(state, RLSHeadL2InitState):
                raise TypeError("L2-Init learner requires RLSHeadL2InitState")
            init_params: Mapping[str, Array] | None = state.init_params
        else:
            init_params = None
        x_norm, new_norm, new_fast, shifted = normalize(
            state.norm, state.fast_mean, x
        )
        count = (
            state.step + jnp.array(1, dtype=jnp.int32)
            if gate_enabled
            else state.step
        )
        n_classes = state.wout.shape[1]
        y_onehot = jax.nn.one_hot(y, n_classes, dtype=jnp.float32)
        if resid:
            body = {name: params[name] for name in _RLS_HEAD_BODY}

            if whiten_enabled:
                # Preconditioned residual signal.  ``delta`` is a constant of
                # the body (every factor below is detached), so the
                # surrogate's gradient is exactly ``-J_phi^T delta`` — the
                # incumbent's own gradient when ``delta = wout @ err``.  The
                # renormalization holds ``||delta|| = ||g||`` at every
                # ``resid_whiten``, so the direction rotates but the
                # magnitude, and hence the frozen step-size calibration, does
                # not move (negative result #1).

                def head_surrogate(
                    body_params: dict[str, Array],
                ) -> tuple[Array, tuple[Array, Array, Array]]:
                    merged = dict(params)
                    merged.update(body_params)
                    phi = _phi(merged, x_norm)
                    logits = state.wout.T @ phi
                    err = y_onehot - logits
                    surrogate_loss = 0.5 * jnp.sum(err * err)
                    err_c = jax.lax.stop_gradient(err)
                    g = state.wout @ err_c
                    if newton:
                        gram = state.wout.T @ state.wout
                        ridge = (
                            _NEWTON_RIDGE_REL * jnp.trace(gram) / n_classes
                            + _NEWTON_RIDGE_ABS
                        )
                        precond = state.wout @ jnp.linalg.solve(
                            gram
                            + ridge * jnp.eye(n_classes, dtype=jnp.float32),
                            err_c,
                        )
                    else:
                        precond = state.p @ g
                    d_dir = (1.0 - resid_whiten) * g + resid_whiten * precond
                    scale = jnp.linalg.norm(g) / jnp.maximum(
                        jnp.linalg.norm(d_dir), _WHITEN_NORM_FLOOR
                    )
                    delta = jax.lax.stop_gradient(d_dir * scale)
                    return -jnp.dot(phi, delta), (logits, phi, surrogate_loss)

                (_, (logits, phi, loss)), body_grads = jax.value_and_grad(
                    head_surrogate, has_aux=True
                )(body)
            else:

                def head_loss(
                    body_params: dict[str, Array],
                ) -> tuple[Array, tuple[Array, Array]]:
                    merged = dict(params)
                    merged.update(body_params)
                    phi = _phi(merged, x_norm)
                    logits = state.wout.T @ phi
                    err = y_onehot - logits
                    return 0.5 * jnp.sum(err * err), (logits, phi)

                (loss, (logits, phi)), body_grads = jax.value_and_grad(
                    head_loss, has_aux=True
                )(body)
            if gate_enabled:
                new_params, new_utility = _gated_sgd(
                    params,
                    body_grads,
                    state.utility,
                    count,
                    _RLS_HEAD_BODY,
                    guard_zero_max=True,
                    init_params=init_params,
                )
            else:
                # Issue #52's frozen ablation endpoint: do not compute or
                # carry utility EMA, bias correction, or sigmoid bookkeeping.
                # The Python closure flag makes this a build-time branch, so
                # the no-gate compiled graph contains only decayed SGD.
                new_params = dict(params)
                for name in _RLS_HEAD_BODY:
                    new_params[name] = (
                        params[name] * param_decay - step_size * body_grads[name]
                    )
                new_utility = state.utility
        else:
            _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                params, x_norm, y
            )
            new_params, new_utility = _gated_sgd(
                params, grads, state.utility, count,
                tuple(sorted(params)), guard_zero_max=False,
            )
            phi = _phi(params, x_norm)
            logits = state.wout.T @ phi
            err_pre = y_onehot - logits
            loss = 0.5 * jnp.sum(err_pre * err_pre)
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        err = y_onehot - logits
        pp = state.p @ phi
        gain = pp / (rls_lambda + phi @ pp)
        new_wout = state.wout + jnp.outer(gain, err)
        new_p = (state.p - jnp.outer(gain, pp)) / rls_lambda
        new_p = 0.5 * (new_p + new_p.T)
        if reset_enabled:
            m = new_p.shape[0]
            trigger = jnp.mean(shifted.astype(jnp.float32)) >= reset_frac
            new_p = jnp.where(
                trigger, jnp.eye(m, dtype=jnp.float32) / ridge_init, new_p
            )
        if cap_enabled:
            # Covariance wind-up guard: exponential forgetting grows P as
            # (1/lambda)^t along unexcited (dead-ReLU) feature directions —
            # float32 overflow and prediction collapse (wave-1 measurement:
            # onset ~ task 18 at lambda 0.999, exactly the e^88.7 overflow
            # horizon).  Rescale P to the trace cap whenever it exceeds it;
            # under the cap the step is bitwise untouched.
            new_p = new_p * jnp.minimum(1.0, trace_cap / jnp.trace(new_p))
        err_after = y_onehot - new_wout.T @ phi
        loss_after = 0.5 * jnp.sum(err_after * err_after)
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
        )
        if _decay_to_init:
            assert isinstance(state, RLSHeadL2InitState)
            return new_params, RLSHeadL2InitState(  # type: ignore[call-arg]
                utility=new_utility,
                step=count,
                norm=new_norm,
                fast_mean=new_fast,
                p=new_p,
                wout=new_wout,
                init_params=state.init_params,
            ), (accuracy, loss, plasticity)
        return new_params, RLSHeadState(  # type: ignore[call-arg]
            utility=new_utility,
            step=count,
            norm=new_norm,
            fast_mean=new_fast,
            p=new_p,
            wout=new_wout,
        ), (accuracy, loss, plasticity)

    return init_fn, full_step


@chex.dataclass(frozen=True)
class RLSHeadIdentState:
    """Incumbent state plus the online permutation-identification carry.

    ``inner`` is the unmodified incumbent state.  ``raw_norm``/``raw_fast``
    run the champion's shift detector on the RAW input stream (the incumbent's
    own detector sees the remapped stream, whose shifts include the remap
    landing).  ``ref_*`` accumulate task-0 class-conditional / marginal
    statistics until the first detected boundary freezes them; ``post_*``
    re-accumulate after every detected boundary.  ``remap`` is the current
    estimated inverse permutation composed with the reference layout
    (identity until the first match of each task).
    """

    inner: RLSHeadState
    raw_norm: EMANormState
    raw_fast: Array
    ref_class_sums: Array
    ref_class_count: Array
    ref_sq_sums: Array
    ref_frozen: Array
    post_class_sums: Array
    post_class_count: Array
    post_sq_sums: Array
    since_shift: Array
    remap: Array


def _identmap_assignment(
    ref_class_means: np.ndarray,
    ref_marg_mean: np.ndarray,
    ref_marg_std: np.ndarray,
    post_class_means: np.ndarray,
    post_marg_mean: np.ndarray,
    post_marg_std: np.ndarray,
) -> np.ndarray:
    """Hungarian assignment of post-shift positions to reference positions.

    V1's fingerprint: per-position class-conditional means plus marginal
    mean/std, each dimension z-scored across positions independently per
    side, Euclidean cost, ``linear_sum_assignment``.  Runs on the host via
    ``jax.pure_callback`` exactly once per matching step.
    """
    from scipy.optimize import linear_sum_assignment

    def fingerprint(cm: np.ndarray, mm: np.ndarray, ms: np.ndarray) -> np.ndarray:
        vec = np.concatenate([cm.T, mm[:, None], ms[:, None]], axis=1)
        mu = vec.mean(axis=0, keepdims=True)
        sd = vec.std(axis=0, keepdims=True)
        return (vec - mu) / np.maximum(sd, 1e-8)

    ref_vec = fingerprint(ref_class_means, ref_marg_mean, ref_marg_std)
    post_vec = fingerprint(post_class_means, post_marg_mean, post_marg_std)
    cost = (
        (ref_vec**2).sum(axis=1)[:, None]
        + (post_vec**2).sum(axis=1)[None, :]
        - 2.0 * ref_vec @ post_vec.T
    )
    _rows, cols = linear_sum_assignment(cost)
    return cols.astype(np.int32)


def _make_rls_head_identmap_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Incumbent + online permutation identification and input remap.

    The V7/V8 oracle chain measured that a partially correct input remap
    delivered at ~200 post-shift samples is worth ~+0.030 to the incumbent
    (V8: N=200 at V1's measured 0.62 identification accuracy scored
    0.8997), and that timing dominates accuracy.  This arm builds the real
    mechanism: V1's class-conditional fingerprint estimated online, the
    champion's own shift detector run on the raw stream, and a Hungarian
    assignment at ``ident_match_at`` samples after each detected boundary
    (optional re-matches at ``ident_match2``/``ident_match3`` refine the map
    as accuracy improves with N per V1's curve).  Prediction and learning
    then see ``x[remap]``.  Labels are consumed post-prediction, which the
    protocol permits.  ``ident_match_at = 0`` delegates verbatim to
    :func:`_make_rls_head_learner` (bit-exact reduction, pinned).
    """
    match1 = float(hp["ident_match_at"])
    if match1 == 0.0:
        return _make_rls_head_learner(hp)
    match2 = float(hp.get("ident_match2", 0.0))
    match3 = float(hp.get("ident_match3", 0.0))
    ident_frac = float(hp.get("ident_reset_frac", 0.05))
    if hp["head_resid"] == 0.0:
        raise ValueError("identmap is registered for the residual arm only")
    inner_init, inner_step = _make_rls_head_learner(hp)

    def raw_normalize(
        state: EMANormState, fast_mean: Array, x: Array
    ) -> tuple[Array, EMANormState, Array, Array]:
        return shift_adaptive_normalize(
            state, fast_mean, x,
            decay=hp["norm_decay"],
            fast_decay=hp["fast_decay"],
            epsilon=hp["norm_epsilon"],
            shift_k=hp["shift_k"],
            shift_delta=hp["shift_delta"],
            shift_refractory=hp["shift_refractory"],
        )

    def init_fn(params: dict[str, Array]) -> RLSHeadIdentState:
        input_dim = params["w1"].shape[0]
        n_classes = params["w3"].shape[1]
        return RLSHeadIdentState(  # type: ignore[call-arg]
            inner=inner_init(params),
            raw_norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.zeros(input_dim, dtype=jnp.float32),
            ),
            raw_fast=jnp.zeros(input_dim, dtype=jnp.float32),
            ref_class_sums=jnp.zeros((n_classes, input_dim), dtype=jnp.float32),
            ref_class_count=jnp.zeros((n_classes,), dtype=jnp.float32),
            ref_sq_sums=jnp.zeros((input_dim,), dtype=jnp.float32),
            ref_frozen=jnp.zeros((), dtype=jnp.bool_),
            post_class_sums=jnp.zeros((n_classes, input_dim), dtype=jnp.float32),
            post_class_count=jnp.zeros((n_classes,), dtype=jnp.float32),
            post_sq_sums=jnp.zeros((input_dim,), dtype=jnp.float32),
            since_shift=jnp.array(-(2**30), dtype=jnp.int32),
            remap=jnp.arange(input_dim, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array],
        state: RLSHeadIdentState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], RLSHeadIdentState, StepMetrics]:
        input_dim = x.shape[0]
        n_classes = state.ref_class_sums.shape[0]
        x_r = x[state.remap]
        new_params, new_inner, metrics = inner_step(
            params, state.inner, x_r, y, key
        )
        _x_norm, new_raw_norm, new_raw_fast, shifted = raw_normalize(
            state.raw_norm, state.raw_fast, x
        )
        trigger = jnp.mean(shifted.astype(jnp.float32)) >= ident_frac
        onehot = jax.nn.one_hot(y, n_classes, dtype=jnp.float32)
        class_incr = jnp.outer(onehot, x)
        keep_ref = jnp.logical_and(
            jnp.logical_not(state.ref_frozen), jnp.logical_not(trigger)
        )
        ref_class_sums = jnp.where(
            keep_ref, state.ref_class_sums + class_incr, state.ref_class_sums
        )
        ref_class_count = jnp.where(
            keep_ref, state.ref_class_count + onehot, state.ref_class_count
        )
        ref_sq_sums = jnp.where(
            keep_ref, state.ref_sq_sums + x * x, state.ref_sq_sums
        )
        ref_frozen = jnp.logical_or(state.ref_frozen, trigger)
        post_class_sums = jnp.where(
            trigger,
            class_incr,
            jnp.where(
                state.ref_frozen,
                state.post_class_sums + class_incr,
                state.post_class_sums,
            ),
        )
        post_class_count = jnp.where(
            trigger,
            onehot,
            jnp.where(
                state.ref_frozen,
                state.post_class_count + onehot,
                state.post_class_count,
            ),
        )
        post_sq_sums = jnp.where(
            trigger,
            x * x,
            jnp.where(
                state.ref_frozen, state.post_sq_sums + x * x, state.post_sq_sums
            ),
        )
        since = jnp.where(
            trigger, jnp.array(1, jnp.int32), state.since_shift + 1
        )
        remap = jnp.where(
            trigger, jnp.arange(input_dim, dtype=jnp.int32), state.remap
        )
        do_match = jnp.logical_and(
            ref_frozen,
            jnp.logical_or(
                since == int(match1),
                jnp.logical_or(
                    since == int(match2) if match2 else jnp.bool_(False),
                    since == int(match3) if match3 else jnp.bool_(False),
                ),
            ),
        )

        ref_n = jnp.sum(ref_class_count)
        post_n = jnp.sum(post_class_count)
        ref_class_means = ref_class_sums / jnp.maximum(
            ref_class_count[:, None], 1.0
        )
        post_class_means = post_class_sums / jnp.maximum(
            post_class_count[:, None], 1.0
        )
        ref_marg_mean = jnp.sum(ref_class_sums, axis=0) / jnp.maximum(ref_n, 1.0)
        post_marg_mean = jnp.sum(post_class_sums, axis=0) / jnp.maximum(
            post_n, 1.0
        )
        ref_marg_std = jnp.sqrt(
            jnp.maximum(
                ref_sq_sums / jnp.maximum(ref_n, 1.0) - ref_marg_mean**2, 0.0
            )
        )
        post_marg_std = jnp.sqrt(
            jnp.maximum(
                post_sq_sums / jnp.maximum(post_n, 1.0) - post_marg_mean**2, 0.0
            )
        )

        def matched(_: None) -> Array:
            return jax.pure_callback(
                _identmap_assignment,
                jax.ShapeDtypeStruct((input_dim,), jnp.int32),
                ref_class_means, ref_marg_mean, ref_marg_std,
                post_class_means, post_marg_mean, post_marg_std,
                vmap_method="sequential",
            )

        def unmatched(_: None) -> Array:
            return remap

        remap = jax.lax.cond(do_match, matched, unmatched, None)
        return new_params, RLSHeadIdentState(  # type: ignore[call-arg]
            inner=new_inner,
            raw_norm=new_raw_norm,
            raw_fast=new_raw_fast,
            ref_class_sums=ref_class_sums,
            ref_class_count=ref_class_count,
            ref_sq_sums=ref_sq_sums,
            ref_frozen=ref_frozen,
            post_class_sums=post_class_sums,
            post_class_count=post_class_count,
            post_sq_sums=post_sq_sums,
            since_shift=since,
            remap=remap,
        ), metrics

    return init_fn, full_step


def _rls_head_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Refuse sentinel probes for the RLS-readout arms.

    The deployed prediction is ``argmax(wout.T @ phi)``, not ``mlp_logits``:
    in resid mode w3/b3 are never trained, and in parallel mode probing the
    SGD head would silently score the passenger model instead of the
    deployed readout.  Fail closed, the rff_rls/naive_bayes precedent.
    """
    del state, observation, hyperparameters
    raise NotImplementedError(
        "sentinel probes are unsupported for the rls_head arms: the deployed "
        "model is the champion body + RLS readout, not the protocol MLP head "
        "that the probe harness would score"
    )


# =============================================================================
# (v2) Transient attack: champion/NB ensemble with online learned vote weights
# =============================================================================
#
# CEILING_ANALYSIS.md budget (i): the champion pays 0.041 of the metric in
# within-task re-adaptation transient (first-500-step accuracy 0.659 vs its
# 0.904 plateau), while the naive_bayes arm is FLAT from the first task
# (~0.785 per-task from t1, no transient) because its per-class statistics
# re-estimate at the fast-EMA timescale.  The ensemble deploys an
# accuracy-weighted probability mixture whose weights are learned ONLINE from
# the stream itself: per-member annealed EMAs of each member's own pre-update
# correctness (fast decay, no oracle, no task-boundary signal), squashed
# through a softmax with temperature ``ens_beta``.  Right after a permutation
# the champion's recent-accuracy EMA collapses within tens of steps while the
# NB member's holds, so the vote swings to NB; mid-task the champion
# re-converges above NB and takes the vote back.  Probes: (b) detector-driven
# NB anneal-clock reset (make the NB member itself shift-robust) and (c) a
# third closed-form fast-converging member, linear RLS over normalized pixels
# (the ``lin_rls`` pipeline verbatim).


@chex.dataclass(frozen=True)
class NBEnsembleState:
    """Member states + online vote-weight statistics for the NB ensemble.

    ``member_acc`` holds one annealed recent-accuracy EMA per member in the
    fixed order (champion MLP, naive Bayes[, linear RLS]); ``ens_step`` is
    its scalar anneal clock.  ``det_norm``/``det_fast`` are the raw-input
    shift detector for the NB reset probe (``None`` when the probe is off);
    ``reset_age`` counts steps since the last NB clock reset (refractory).
    ``rls`` is the optional third member's state (``None`` on 2-member arms).
    """

    net: UPGDAdaptiveNormState
    nb: NaiveBayesState
    rls: RFFRLSState | None
    member_acc: Array
    ens_step: Array
    det_norm: EMANormState | None
    det_fast: Array | None
    reset_age: Array


def _nb_ensemble_hp(**overrides: float) -> dict[str, float]:
    """Ensemble hyperparameters: champion + NB + lin_rls constants verbatim,
    plus the ensemble's own vote/probe knobs (inert flags default off)."""
    merged = _sigma0_ext_hp(
        # Champion member: verbatim sigma0_shiftnorm_d099.
        norm_decay=0.99,
        fast_decay=0.9,
        shift_k=1.0,
        shift_delta=0.02,
        shift_refractory=0.0,
        # NB member: verbatim naive_bayes.
        nb_decay=0.98,
        nb_var_epsilon=0.1,
        # Online vote weights, frozen by the 3-round 2-task seed-0
        # diagnostic (decay {0.95,0.98,0.99,0.995,0.999} x beta
        # {10,20,40,80,160}): mean rises monotonically to (0.995, 80) =
        # 0.8496 (t1 .8328/t2 .8664 vs champion .7848/.8652 in the same
        # loop); beta 160 and decay 0.999 both turn down.
        ens_decay=0.995,
        ens_beta=80.0,
        # Reduction pin + probe flags.
        ens_lock_network=0.0,
        ens_use_rls=0.0,
        ens_nb_reset=0.0,
        # Reset trigger, calibrated on the seed-0 raw-pixel detector trace:
        # boundary-step shifted-feature fraction .034-.061 vs mid-task p99
        # .0077 (max .0179) — 0.03 separates them with ~2x margin both ways.
        ens_reset_frac=0.03,
        ens_reset_refractory=500.0,
        # Linear-RLS member: verbatim lin_rls.
        rff_clip=3.0,
        rls_lambda=0.999,
        rls_ridge_init=1.0,
    )
    merged.update(overrides)
    return merged


def _make_nb_ensemble_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Adaptive ensemble of the shiftnorm champion and streaming naive Bayes.

    Per step (predict-then-update, the protocol ordering):

    1. Each member's pre-update class log-posterior is computed exactly as
       that member's own arm computes it (champion: ``mlp_logits`` on the
       shift-adaptive-normalized input; NB: :func:`naive_bayes_logits`;
       optional RLS: readout on clipped z-scores).  The deployed prediction
       is the log-domain accuracy-weighted probability mixture
       ``log sum_m w_m p_m(c)`` with ``w = softmax(ens_beta * member_acc)``;
       the reported loss is the mixture cross-entropy at the true label.
    2. Every member then runs its own arm's exact update on ``(x, y)``.
       Member correctness (each member's own pre-update accuracy metric)
       updates ``member_acc`` with the annealed recurrence
       ``min(ens_decay, 1 - 1/(t + 1))`` — running-average semantics early,
       fast fixed-decay tracking later.  No oracle anywhere: the weights are
       learned from the stream.
    3. Probe (b) (``ens_nb_reset``): a raw-input
       :func:`shift_adaptive_normalize` detector (champion constants)
       reports the per-feature shifted mask; when more than
       ``ens_reset_frac`` of features shift and the last reset is at least
       ``ens_reset_refractory`` steps old, the NB member's class anneal
       clocks reset to zero so its statistics re-estimate at effective decay
       1/2 (means/variances are not zeroed).
    4. Probe (c) (``ens_use_rls``): a third member, the ``lin_rls`` pipeline
       verbatim (bias-augmented clipped z-scores, Sherman-Morrison RLS).

    ``ens_lock_network=1`` is the reduction pin: the deployed prediction,
    loss, and plasticity are the champion member's own metrics bit-for-bit
    (the registered ``sigma0_shiftnorm_d099`` step), while the member EMAs
    still learn passively.  Plasticity in ensemble mode is the protocol's
    post-update one-step improvement ratio on the same mixture cross-entropy
    at unchanged vote weights.  The RNG key is deliberately unused (all
    members are closed-form or sigma-0).
    """
    lock = hp["ens_lock_network"] != 0.0
    use_rls = hp["ens_use_rls"] != 0.0
    nb_reset = hp["ens_nb_reset"] != 0.0
    ens_decay = hp["ens_decay"]
    ens_beta = hp["ens_beta"]
    reset_frac = hp["ens_reset_frac"]
    reset_refractory = hp["ens_reset_refractory"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    fast_decay = hp["fast_decay"]
    shift_k = hp["shift_k"]
    shift_delta = hp["shift_delta"]
    shift_refractory = hp["shift_refractory"]
    rls_clip = hp["rff_clip"]
    n_members = 3 if use_rls else 2

    net_init, net_step = _make_upgd_shiftnorm_learner(hp)
    nb_init, nb_step = _make_naive_bayes_learner(hp)
    lin_init, lin_step = _make_lin_rls_learner(hp)

    def init_fn(params: dict[str, Array]) -> NBEnsembleState:
        input_dim = params["w1"].shape[0]
        det_norm: EMANormState | None = None
        det_fast: Array | None = None
        if nb_reset:
            det_norm = EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.zeros(input_dim, dtype=jnp.float32),
            )
            det_fast = jnp.zeros(input_dim, dtype=jnp.float32)
        return NBEnsembleState(  # type: ignore[call-arg]
            net=net_init(params),
            nb=nb_init(params),
            rls=lin_init(params) if use_rls else None,
            member_acc=jnp.zeros(n_members, dtype=jnp.float32),
            ens_step=jnp.asarray(0.0, jnp.float32),
            det_norm=det_norm,
            det_fast=det_fast,
            # Armed at init: the first mature detected shift may reset.
            reset_age=jnp.asarray(reset_refractory, jnp.float32),
        )

    def _lin_features(state: RFFRLSState, x: Array) -> Array:
        """The lin_rls feature map (pre-update statistics, no state write)."""
        x_norm, _ = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        z = jnp.clip(x_norm, -rls_clip, rls_clip)
        return jnp.concatenate(
            [z / jnp.sqrt(jnp.float32(z.shape[0])), jnp.ones((1,), jnp.float32)]
        )

    def _members_update(
        params: dict[str, Array],
        state: NBEnsembleState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], NBEnsembleState, StepMetrics]:
        """Run every member's own update; learn the vote weights passively."""
        new_params, new_net, net_metrics = net_step(params, state.net, x, y, key)
        _, new_nb, nb_metrics = nb_step(params, state.nb, x, y, key)
        member_correct = [net_metrics[0], nb_metrics[0]]
        new_rls = state.rls
        if use_rls:
            assert state.rls is not None
            _, new_rls, rls_metrics = lin_step(params, state.rls, x, y, key)
            member_correct.append(rls_metrics[0])
        det_norm, det_fast, reset_age = state.det_norm, state.det_fast, state.reset_age
        if nb_reset:
            assert state.det_norm is not None and state.det_fast is not None
            _, det_norm, det_fast, shifted = shift_adaptive_normalize(
                state.det_norm,
                state.det_fast,
                x,
                decay=norm_decay,
                fast_decay=fast_decay,
                epsilon=norm_epsilon,
                shift_k=shift_k,
                shift_delta=shift_delta,
                shift_refractory=0.0,
            )
            frac = jnp.mean(shifted.astype(jnp.float32))
            trigger = (frac > reset_frac) & (state.reset_age >= reset_refractory)
            new_nb = NaiveBayesState(  # type: ignore[call-arg]
                cmean=new_nb.cmean,
                cvar=new_nb.cvar,
                ccount=jnp.where(trigger, 0.0, new_nb.ccount),
                prior=new_nb.prior,
                step=new_nb.step,
            )
            reset_age = jnp.where(trigger, 0.0, state.reset_age + 1.0)
        count = state.ens_step + 1.0
        eff = jnp.minimum(ens_decay, 1.0 - 1.0 / (count + 1.0))
        member_acc = eff * state.member_acc + (1.0 - eff) * jnp.stack(member_correct)
        new_state = NBEnsembleState(  # type: ignore[call-arg]
            net=new_net,
            nb=new_nb,
            rls=new_rls,
            member_acc=member_acc,
            ens_step=count,
            det_norm=det_norm,
            det_fast=det_fast,
            reset_age=reset_age,
        )
        return new_params, new_state, net_metrics

    if lock:

        def lock_step(
            params: dict[str, Array],
            state: NBEnsembleState,
            x: Array,
            y: Array,
            key: Array,
        ) -> tuple[dict[str, Array], NBEnsembleState, StepMetrics]:
            # Reduction pin: metrics AND params are the champion member's
            # bit-for-bit; the ensemble statistics still learn passively.
            return _members_update(params, state, x, y, key)

        return init_fn, lock_step

    def full_step(
        params: dict[str, Array],
        state: NBEnsembleState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], NBEnsembleState, StepMetrics]:
        # Pre-update member posteriors: each computed exactly as the member's
        # own arm computes them (the normalize call is re-derived, not
        # state-written; the member update below performs the real write).
        x_norm, _, _, _ = shift_adaptive_normalize(
            state.net.norm,
            state.net.fast_mean,
            x,
            decay=norm_decay,
            fast_decay=fast_decay,
            epsilon=norm_epsilon,
            shift_k=shift_k,
            shift_delta=shift_delta,
            shift_refractory=shift_refractory,
        )
        member_logp = [
            jax.nn.log_softmax(mlp_logits(params, x_norm)),
            jax.nn.log_softmax(naive_bayes_logits(state.nb, x)),
        ]
        phi: Array | None = None
        if use_rls:
            assert state.rls is not None
            phi = _lin_features(state.rls, x)
            member_logp.append(jax.nn.log_softmax(state.rls.wout.T @ phi))
        log_w = jax.nn.log_softmax(ens_beta * state.member_acc)
        mixture = jax.nn.log_softmax(
            jax.nn.logsumexp(jnp.stack(member_logp) + log_w[:, None], axis=0)
        )
        accuracy = (jnp.argmax(mixture) == y).astype(jnp.float32)
        loss = -mixture[y]
        new_params, new_state, _ = _members_update(params, state, x, y, key)
        # Plasticity: post-update mixture improvement at unchanged weights.
        member_logp_after = [
            jax.nn.log_softmax(mlp_logits(new_params, x_norm)),
            jax.nn.log_softmax(naive_bayes_logits(new_state.nb, x)),
        ]
        if use_rls:
            assert new_state.rls is not None and phi is not None
            member_logp_after.append(
                jax.nn.log_softmax(new_state.rls.wout.T @ phi)
            )
        mixture_after = jax.nn.log_softmax(
            jax.nn.logsumexp(jnp.stack(member_logp_after) + log_w[:, None], axis=0)
        )
        loss_after = -mixture_after[y]
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
        )
        return new_params, new_state, (accuracy, loss, plasticity)

    return init_fn, full_step


def _nb_ensemble_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Refuse sentinel probes for the nb_ensemble arms.

    The deployed predictor is the accuracy-weighted member mixture, not the
    protocol MLP alone: probing ``mlp_logits`` on the champion member would
    silently score a different model than the one the arm deploys.  Fail
    closed, exactly like the other non-MLP deployments.
    """
    del state, observation, hyperparameters
    raise NotImplementedError(
        "sentinel probes are unsupported for the nb_ensemble arms: the "
        "deployed predictor is the accuracy-weighted member mixture, not "
        "the protocol MLP alone"
    )


# =============================================================================
# (s) Optimizer-floor hybrids: Adam-class step adaptation + champion stability
# =============================================================================
#
# The ceiling analysis (outputs/ipmnist_screening/CEILING_ANALYSIS.md) maps a
# +0.041 "optimizer floor": the champion family asymptotes at ~0.933 on a
# stationary stream where protocol AdamW reaches 0.974 online — per-parameter
# step adaptation is the one capability the champion family lacks.  The naive
# composition (``adamw_cbp_ema_norm``) proved the convergence transfers: it
# scores 0.8425 on task 1 — the best first-task number of the campaign
# (champion 0.7863) — and then decays monotonically to 0.743 by task 200,
# because it ran the protocol AdamW hyperparameters: no utility gate,
# weight_decay 0, slow 0.999 normalizer.  Adam-class within-task convergence
# under conditioning is real; continual stability was simply never attached
# to it.  These arms attach the champion's full stability package (fast
# decay-0.99 conditioning, the exact UPGD utility gate, decoupled weight
# decay 0.01) to four adaptive descent directions; every arm is
# perturbation-free and consumes no per-step randomness.


@chex.dataclass(frozen=True)
class NormAdamGateState:
    """UPGD utility EMA/clock, per-element Adam moments, shift-normalizer.

    ``norm.count`` is per-feature ``f32[d]`` (shift-adaptive normalizer);
    ``fast_mean`` is its fast detection EMA.  ``m``/``v``/``count`` are the
    per-element Adam moments and bias-correction clocks.
    """

    utility: dict[str, Array]
    step: Array
    m: dict[str, Array]
    v: dict[str, Array]
    count: dict[str, Array]
    norm: EMANormState
    fast_mean: Array


def _make_norm_adam_fastv_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Gated AdamW behind the shift-adaptive normalizer, with shift-triggered
    second-moment resets on the input layer.

    Pipeline per step (champion-parity stability around an Adam core):

    1. :func:`shift_adaptive_normalize` conditions the input exactly as the
       ``sigma0_shiftnorm_d099`` arm (slow decay ``norm_decay``, fast
       detection EMA, per-feature anneal-count reset on detected shift).
    2. Gradients on the normalized input feed the exact UPGD utility gate
       (:func:`_upgd_utility_and_gate`, bias-corrected, global-max sigmoid).
    3. When ``vreset_enabled`` and feature ``i`` triggered the detector this
       step, the Adam moments and bias-correction counts of ``w1`` row ``i``
       (the weights consuming that feature) reset to zero *before* the moment
       update — the permutation that moved a feature's statistics also
       invalidated the curvature history of exactly those weights, so their
       next steps re-estimate scale from fresh gradients (per-element counts
       restart bias correction, the ``adamw_cbp`` recycling precedent).
       Deeper layers keep their moments (features carried across tasks).
    4. Per-element AdamW delta (:func:`adam_elem_step` with the weight decay
       held OUT of the moments), then the champion-form application: the gate
       scales only the descent term and the decoupled decay applies in full —
       ``w <- w * (1 - lr*wd) - delta * (1 - gate)``.

    ``beta2`` is the deliberate axis: 0.9 (10-step curvature memory) against
    the protocol's 0.99; ``beta1 = 0`` keeps the protocol's momentum-free
    form (the momentum axis is dissected separately by
    ``sgd_momentum_gate``).  With an untriggerable detector the step is
    bit-exact hand-composed normalize -> gated AdamW (pinned by a unit
    test); ``vreset_enabled = 0`` keeps the detector but carries the moments
    (the reset dissection arm).  The RNG key is deliberately unused.
    """
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    vreset = hp["vreset_enabled"] != 0.0
    adam_hp = {
        "beta1": hp["beta1"],
        "beta2": hp["beta2"],
        "step_size": step_size,
        "eps": hp["eps"],
        "weight_decay": 0.0,
    }

    def normalize(
        state: EMANormState, fast_mean: Array, x: Array
    ) -> tuple[Array, EMANormState, Array, Array]:
        return shift_adaptive_normalize(
            state, fast_mean, x,
            decay=hp["norm_decay"],
            fast_decay=hp["fast_decay"],
            epsilon=hp["norm_epsilon"],
            shift_k=hp["shift_k"],
            shift_delta=hp["shift_delta"],
            shift_refractory=hp["shift_refractory"],
        )

    def init_fn(params: dict[str, Array]) -> NormAdamGateState:
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        input_dim = params["w1"].shape[0]
        return NormAdamGateState(  # type: ignore[call-arg]
            utility=dict(zeros),
            step=jnp.array(0, dtype=jnp.int32),
            m=dict(zeros),
            v=dict(zeros),
            count={name: jnp.zeros_like(value) for name, value in params.items()},
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.zeros(input_dim, dtype=jnp.float32),
            ),
            fast_mean=jnp.zeros(input_dim, dtype=jnp.float32),
        )

    def full_step(
        params: dict[str, Array], state: NormAdamGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], NormAdamGateState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm, new_fast, shifted = normalize(state.norm, state.fast_mean, x)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        clock = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, clock, utility_decay
        )
        m, v, count = state.m, state.v, state.count
        if vreset:
            row_mask = shifted[:, None]
            m = dict(m)
            v = dict(v)
            count = dict(count)
            m["w1"] = jnp.where(row_mask, 0.0, m["w1"])
            v["w1"] = jnp.where(row_mask, 0.0, v["w1"])
            count["w1"] = jnp.where(row_mask, 0.0, count["w1"])
        new_params: dict[str, Array] = {}
        new_m: dict[str, Array] = {}
        new_v: dict[str, Array] = {}
        new_count: dict[str, Array] = {}
        for name in params:
            delta, new_m[name], new_v[name], new_count[name] = adam_elem_step(
                params[name], m[name], v[name], count[name], grads[name], adam_hp
            )
            new_params[name] = params[name] * param_decay - (
                delta * (1.0 - gate[name])
            )
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, NormAdamGateState(  # type: ignore[call-arg]
            utility=utility,
            step=clock,
            m=new_m,
            v=new_v,
            count=new_count,
            norm=new_norm,
            fast_mean=new_fast,
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class NormRMSGateState:
    """UPGD utility EMA/clock, per-element RMSprop accumulator, normalizer."""

    utility: dict[str, Array]
    step: Array
    v: dict[str, Array]
    norm: EMANormState


def _make_norm_rmsprop_gate_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Gated RMSprop behind the champion's EMA input normalizer.

    The minimal per-parameter second-moment tracker: no momentum, no bias
    correction (classic RMSprop; the uncorrected accumulator makes the first
    steps ~``lr/sqrt(1-rho)``-sized, a mild built-in warmup documented
    rather than corrected), fast ``rms_rho = 0.9`` (10-step curvature
    memory tracks batch-1 gradient scale closer than Adam's 100-step 0.99):

    ``v <- rho * v + (1-rho) * g^2``;
    ``w <- w * (1 - lr*wd) - lr * (g / (sqrt(v) + eps)) * (1 - gate)``

    with the exact champion conditioning (``ema_normalize`` decay 0.99) and
    the exact UPGD utility gate scaling only the descent term.  Pinned by a
    hand-computed trajectory test.  The RNG key is deliberately unused.
    """
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    rho = hp["rms_rho"]
    rms_epsilon = hp["rms_epsilon"]

    def init_fn(params: dict[str, Array]) -> NormRMSGateState:
        return NormRMSGateState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            v={name: jnp.zeros_like(value) for name, value in params.items()},
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: NormRMSGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], NormRMSGateState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        clock = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, clock, utility_decay
        )
        new_params: dict[str, Array] = {}
        new_v: dict[str, Array] = {}
        for name in params:
            g = grads[name]
            v = rho * state.v[name] + (1.0 - rho) * g * g
            direction = g / (jnp.sqrt(v) + rms_epsilon)
            new_params[name] = params[name] * param_decay - step_size * (
                direction * (1.0 - gate[name])
            )
            new_v[name] = v
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, NormRMSGateState(  # type: ignore[call-arg]
            utility=utility, step=clock, v=new_v, norm=new_norm
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class NormApolloGateState:
    """UPGD utility EMA/clock, per-channel second moments, normalizer.

    ``vchan`` holds one entry per parameter: shape ``(fan_out,)`` for the
    2-D weights (one accumulator per NEURON — axis 1 of the protocol's
    ``(fan_in, fan_out)`` orientation) and the full shape for biases (the
    exact 1-element-channel specialization).
    """

    utility: dict[str, Array]
    step: Array
    vchan: dict[str, Array]
    norm: EMANormState


def _make_norm_apollo_gate_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """APOLLO-style channel-wise gradient scaling, gated, behind the
    champion's conditioning.

    APOLLO (Zhu et al., arXiv 2412.05270, verified against the paper) shows
    AdamW's element-wise learning-rate adaptation can be coarsened to a
    STRUCTURED scaling: project ``R_t = P_t G_t`` with a random Gaussian
    ``P_t in R^{r x m}`` (resampled every T steps), keep AdamW moments in the
    low-rank space, and scale each channel of the raw gradient by the norm
    ratio ``s_j = ||R~_t[:, j]|| / ||R_t[:, j]||`` where
    ``R~_t = M_t^R / (sqrt(V_t^R) + eps)`` — SGD-like memory, AdamW-class
    quality (rank-1 tensor-wise in APOLLO-Mini with a ``sqrt``-heuristic
    scale ``alpha``).

    Batch-1 streaming adaptation (deviations documented):

    - **Exact channel statistics, no projection.** The random projection is
      a memory-saving JL estimator of exactly the per-channel norms the
      scaling consumes; at 300x150 scale with one example per step the exact
      statistics are affordable — the zero-variance limit of the estimator
      (``alpha = 1``, and no subspace-resample schedule applies).
    - **Channel-shared second moment.** ``v_j <- rho * v_j + (1 - rho) *
      mean_i(G_ij^2)`` per channel ``j`` — the fully-coarsened limit of the
      paper's structured-learning-rate thesis.  With the protocol's
      ``beta1 = 0`` (momentum dissected separately by ``sgd_momentum_gate``)
      the norm-ratio scaling collapses to ``1 / (sqrt(v_j) + eps)``, i.e.
      per-channel RMSprop: ``W[:, j]`` steps ``lr * G[:, j] / (sqrt(v_j) +
      eps)``.
    - **Channel = fan-OUT (per-neuron), not the paper's larger-dimension
      convention.**  Deliberate: (a) the fan-in axis is already measured on
      this protocol (``colnorm_gate``, -0.085 at horizon); (b) at batch 1
      the dense-layer gradient is the rank-1 outer product ``x delta^T``,
      so ``||G[:, j]|| = |delta_j| * ||x||`` — per-neuron scaling conditions
      the backprop delta, output-side conditioning that composes with (and
      does not duplicate) the input-side normalizer.
    - **No norm-growth limiter.** The paper's limiter guards early LLM
      gradient spikes; here the current-inclusive EMA normalizer is
      self-bounding (AUDIT.md F3), so the spike source it protects against
      is absent.
    - Biases keep per-element accumulators (1-element channels; pinned
      bitwise against ``norm_rmsprop_gate``'s bias path at matched decay).

    Stability package and application form exactly as the sibling arms:
    champion ``ema_normalize`` (decay 0.99), UPGD utility gate on the
    descent term only, decoupled decay.  The RNG key is deliberately unused.
    """
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    rho = hp["apollo_decay"]
    apollo_epsilon = hp["apollo_epsilon"]

    def init_fn(params: dict[str, Array]) -> NormApolloGateState:
        return NormApolloGateState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            vchan={
                name: jnp.zeros(
                    value.shape[1] if value.ndim == 2 else value.shape,
                    dtype=jnp.float32,
                )
                for name, value in params.items()
            },
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: NormApolloGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], NormApolloGateState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        clock = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, clock, utility_decay
        )
        new_params: dict[str, Array] = {}
        new_vchan: dict[str, Array] = {}
        for name in params:
            g = grads[name]
            if params[name].ndim == 2:
                stat = jnp.mean(g * g, axis=0)
                v = rho * state.vchan[name] + (1.0 - rho) * stat
                denom = jnp.sqrt(v)[None, :] + apollo_epsilon
            else:
                v = rho * state.vchan[name] + (1.0 - rho) * (g * g)
                denom = jnp.sqrt(v) + apollo_epsilon
            new_params[name] = params[name] * param_decay - step_size * (
                (g / denom) * (1.0 - gate[name])
            )
            new_vchan[name] = v
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, NormApolloGateState(  # type: ignore[call-arg]
            utility=utility, step=clock, vchan=new_vchan, norm=new_norm
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class NormMomentumGateState:
    """UPGD utility EMA/clock, EMA momentum buffers, normalizer."""

    utility: dict[str, Array]
    step: Array
    momentum: dict[str, Array]
    norm: EMANormState


def _make_sgd_momentum_gate_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """The champion's exact update with EMA-bias-corrected momentum.

    The cheapest convergence accelerator applied to ``sigma0_ndecay099``:
    the descent direction becomes the bias-corrected momentum EMA

    ``m <- mu * m + (1 - mu) * g``;  ``m_hat = m / (1 - mu^t)``;
    ``w <- w * (1 - lr*wd) - lr * m_hat * (1 - gate)``

    (the EMA form keeps ``m_hat`` at raw-gradient scale — the classical
    heavy-ball sum would multiply the effective step by ``1/(1-mu)``, which
    at batch 1 conflates the momentum question with a learning-rate change).
    The gate scales the applied momentum direction, the champion's
    conditioning and decay are untouched, and ``momentum = 0`` reduces
    bit-exactly to the ``sigma0_ndecay099`` champion (pinned by a unit
    test).  The RNG key is deliberately unused.
    """
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    mu = hp["momentum"]

    def init_fn(params: dict[str, Array]) -> NormMomentumGateState:
        return NormMomentumGateState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            momentum={name: jnp.zeros_like(value) for name, value in params.items()},
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: NormMomentumGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], NormMomentumGateState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        clock = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, clock, utility_decay
        )
        correction = 1.0 - jnp.power(
            jnp.asarray(mu, dtype=jnp.float32), clock.astype(jnp.float32)
        )
        new_params: dict[str, Array] = {}
        new_momentum: dict[str, Array] = {}
        for name in params:
            momentum = mu * state.momentum[name] + (1.0 - mu) * grads[name]
            m_hat = momentum / correction
            new_params[name] = params[name] * param_decay - step_size * (
                m_hat * (1.0 - gate[name])
            )
            new_momentum[name] = momentum
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, NormMomentumGateState(  # type: ignore[call-arg]
            utility=utility, step=clock, momentum=new_momentum, norm=new_norm
        ), metrics

    return init_fn, full_step


# =============================================================================
# (s) Reviewer comparison arms: published mechanisms behind the EMA normalizer
# =============================================================================
#
# The rows completing the comparison table "our conditioning + THEIR
# mechanism vs our conditioning + our gate": the strongest published
# plasticity mechanisms, re-implemented from their papers (rules verified
# against the sources 2026-08-02), run behind the exact champion input
# conditioning (EMA normalizer, decay 0.99) on a plain-SGD base — no utility
# gate, no perturbation.  Every factory reduces bit-exactly to the shared
# normalized-SGD base when its mechanism constant is inert (pinned).


def _make_wclip_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Weight Clipping (Elsayed, Lan, Lyle & Mahmood, RLC 2024) behind the
    champion's EMA input normalizer.

    Algorithm 1 of the paper on the comparison base: normalize the input
    (:func:`ema_normalize`, decay 0.99), take the plain SGD step
    ``w <- w * (1 - lr*wd) - lr * grad``, then clip every weight AND bias of
    layer ``l`` to ``[-kappa * s_l, +kappa * s_l]`` with
    ``s_l = 1/sqrt(fan_in)`` the protocol's uniform-init bound
    (:func:`_wclip_bound`; the paper clips biases too, and its example value
    is ``kappa = 2``).  Registered with ``wd = 0`` — the paper's Algorithm 1
    is standalone SGD + clip, positioning clipping as the *alternative* to
    decay-family regularizers (their Fig. 1 contrasts it with L2 and
    L2-Init).  No utility gate, no perturbation; the RNG key is deliberately
    unused.  With ``clip_kappa = inf`` the clip is a no-op and the
    trajectory is bit-exact against ``sgd_ema_norm``'s factory on the same
    hyperparameters (pinned by a unit test).
    """
    step_size = hp["step_size"]
    decay_factor = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]
    kappa = hp["clip_kappa"]

    def init_fn(params: dict[str, Array]) -> SGDNormState:
        return SGDNormState(norm=_init_input_norm_state(params))  # type: ignore[call-arg]

    def full_step(
        params: dict[str, Array], state: SGDNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], SGDNormState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        new_params = {
            name: jnp.clip(
                params[name] * decay_factor - step_size * grads[name],
                -_wclip_bound(params, name, kappa),
                _wclip_bound(params, name, kappa),
            )
            for name in params
        }
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, SGDNormState(norm=new_norm), metrics  # type: ignore[call-arg]

    return init_fn, full_step


@chex.dataclass(frozen=True)
class FadeHeadNormState:
    """FADE log decay rates + sensitivity traces (head only) and normalizer."""

    gamma: dict[str, Array]
    fade_trace: dict[str, Array]
    norm: EMANormState


def _make_fade_head_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """FADE meta-learned per-parameter head decay (Ramesh, Lewandowski &
    Schmidhuber, arXiv:2604.27063) behind the champion's EMA input
    normalizer, on the plain-SGD comparison base.

    The mechanism is exactly :func:`upgd_w_fade_head_update`'s head branch —
    same meta update (old trace first, capped at ``gamma <= 0``), same
    stable sensitivity-trace recursion, same published constants
    (``fade_alpha = 0.005``, ``gamma0 = -6.9``, ``theta_lambda = 0.1``) —
    with UPGD's gate and perturbation removed.  Hidden layers take the plain
    step ``w * (1 - lr*wd) - lr*grad`` (registered ``wd = 0`` — the paper
    applies its adaptive decay to the final layer only); the head takes
    ``w * (1 - lambda_i) - lr*grad`` with ``lambda_i = exp(gamma_i)``
    meta-learned online.  No utility gate, no perturbation; the RNG key is
    deliberately unused.  With ``fade_theta_lambda = 0`` and
    ``fade_gamma0 = -inf`` (``lambda = 0``) the trajectory is bit-exact
    against the plain normalized-SGD base at the same ``wd`` (pinned).
    """
    step_size = hp["step_size"]
    theta = hp["fade_theta_lambda"]
    fade_alpha = hp["fade_alpha"]
    gamma0 = hp["fade_gamma0"]
    hidden_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]

    def init_fn(params: dict[str, Array]) -> FadeHeadNormState:
        return FadeHeadNormState(  # type: ignore[call-arg]
            gamma={
                name: jnp.full_like(params[name], gamma0)
                for name in _FADE_HEAD_PARAMS
            },
            fade_trace={
                name: jnp.zeros_like(params[name]) for name in _FADE_HEAD_PARAMS
            },
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array],
        state: FadeHeadNormState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], FadeHeadNormState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        _, _, _, _, a2 = _forward_with_activations(params, x_norm)
        head_sq = {
            "w3": (a2 * a2)[:, None],
            "b3": jnp.ones_like(params["b3"]),
        }
        new_params: dict[str, Array] = {}
        new_gamma: dict[str, Array] = {}
        new_trace: dict[str, Array] = {}
        for name in params:
            descent = step_size * grads[name]
            if name in _FADE_HEAD_PARAMS:
                gamma = jnp.minimum(
                    state.gamma[name]
                    + theta * (-grads[name]) * state.fade_trace[name],
                    _FADE_GAMMA_MAX,
                )
                lam = jnp.exp(gamma)
                new_params[name] = params[name] * (1.0 - lam) - descent
                contraction = jnp.maximum(0.0, 1.0 - lam - fade_alpha * head_sq[name])
                new_gamma[name] = gamma
                new_trace[name] = state.fade_trace[name] * contraction - lam * params[name]
            else:
                new_params[name] = params[name] * hidden_decay - descent
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, FadeHeadNormState(  # type: ignore[call-arg]
            gamma=new_gamma, fade_trace=new_trace, norm=new_norm
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class SNRNormState:
    """Per-unit SNR statistics for both hidden layers plus the normalizer.

    ``silence*`` counts consecutive non-firing steps, ``rate*`` is the
    firing-indicator EMA accumulator, ``age*`` counts steps since the unit's
    last reset (bias correction for the EMA).
    """

    silence1: Array
    silence2: Array
    rate1: Array
    rate2: Array
    age1: Array
    age2: Array
    norm: EMANormState


def snr_reset_mask(
    silence: Array,
    rate: Array,
    age: Array,
    *,
    eta: float,
    rate_decay: float,
    rate_floor: float,
) -> Array:
    """SNR rejection test: reset iff ``P(A >= a) <= eta`` (geometric null).

    Farias & Jozefiak (arXiv:2410.20098), Algorithm 1 with their practical
    reduction: instead of the full inter-firing histogram, "track the mean
    inter-firing time and assume A is geometrically distributed with that
    mean".  Here the firing rate ``p`` is a bias-corrected EMA of the
    per-step firing indicator (the streaming stand-in for their fixed-length
    trailing window — one statistic per unit, exactly their reduction),
    floored at ``rate_floor``: a unit whose OBSERVED rate is ~0 keeps a
    long-tailed null and is not reset (the test detects rate *collapse*,
    not units that were always quiet).  The survival probability
    ``P(A >= a) = (1 - p)^(a - 1)`` is evaluated in log space; ``eta = 0``
    disables resets exactly (used by the reduction pin).
    """
    if eta <= 0.0:
        return jnp.zeros(silence.shape, dtype=bool)
    correction = 1.0 - jnp.power(
        jnp.asarray(rate_decay, dtype=jnp.float32), age.astype(jnp.float32)
    )
    p = jnp.clip(
        rate / jnp.maximum(correction, 1e-12), rate_floor, 1.0 - rate_floor
    )
    excess = jnp.maximum(silence.astype(jnp.float32) - 1.0, 0.0)
    return excess * jnp.log1p(-p) <= math.log(eta)


def snr_maybe_reset_layer(
    params: dict[str, Array],
    silence: Array,
    rate: Array,
    age: Array,
    layer: _CBPLayerRefs,
    key: Array,
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], Array, Array, Array, Array]:
    """Apply the SNR test to one hidden layer and reset every rejected unit.

    Paper reset rule: re-initialize the unit's incoming weights and bias by
    the network's own init rule (the protocol's PyTorch-default uniform
    ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` for both) and zero its outgoing
    weights.  Unlike the CBP arms (rate-budgeted argmin recycling of at most
    one unit), every unit failing the test resets in the same step, and the
    reset unit's SNR statistics restart from zero.

    Returns ``(params, silence, rate, age, reset_mask)``.
    """
    mask = snr_reset_mask(
        silence,
        rate,
        age,
        eta=hp["snr_eta"],
        rate_decay=hp["snr_rate_decay"],
        rate_floor=hp["snr_rate_floor"],
    )
    w_in = params[layer.in_weight]
    fan_in = w_in.shape[0]
    bound = 1.0 / math.sqrt(fan_in)
    key_w, key_b = jr.split(key)
    fresh_w = jr.uniform(key_w, w_in.shape, jnp.float32, -bound, bound)
    fresh_b = jr.uniform(
        key_b, params[layer.in_bias].shape, jnp.float32, -bound, bound
    )
    new_params = dict(params)
    new_params[layer.in_weight] = jnp.where(mask[None, :], fresh_w, w_in)
    new_params[layer.in_bias] = jnp.where(mask, fresh_b, params[layer.in_bias])
    new_params[layer.out_weight] = jnp.where(
        mask[:, None],
        jnp.zeros_like(params[layer.out_weight]),
        params[layer.out_weight],
    )
    new_silence = jnp.where(mask, 0, silence)
    new_rate = jnp.where(mask, 0.0, rate)
    new_age = jnp.where(mask, 0, age)
    return new_params, new_silence, new_rate, new_age, mask


def _make_snr_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Self-Normalized Resets (Farias & Jozefiak, arXiv:2410.20098) behind
    the champion's EMA input normalizer, on the plain-SGD comparison base.

    Paper Algorithm 1 ordering per step: forward pass (here on the
    normalized input), inter-firing-time update from the PRE-update
    activations, optimizer step (plain SGD; registered ``wd = 0`` — their
    standalone configuration), then the per-unit rejection test + resets
    (:func:`snr_maybe_reset_layer`) on both hidden ReLU layers.  Registered
    ``eta = 0.005`` sits inside the paper's Permuted-MNIST sweep grid
    {0.08 .. 0.00125}.  Documented deviation: the firing-rate estimate is a
    bias-corrected EMA (decay ``snr_rate_decay``) of the firing indicator
    rather than a fixed-length trailing-window mean — their own reduction
    (one statistic per unit) in streaming form.  Note their experiments
    batch 16 examples per step, so a healthy unit's per-step firing
    probability is ~1 there; under this protocol's one-example steps the
    same test at equal ``eta`` is necessarily more trigger-happy — that
    protocol difference is part of what this row measures.  With
    ``snr_eta = 0`` the test never rejects and the parameter trajectory is
    bit-exact against the plain normalized-SGD base (pinned).
    """
    step_size = hp["step_size"]
    decay_factor = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]
    rate_decay = hp["snr_rate_decay"]

    def init_fn(params: dict[str, Array]) -> SNRNormState:
        h1 = params["w1"].shape[1]
        h2 = params["w2"].shape[1]
        return SNRNormState(  # type: ignore[call-arg]
            silence1=jnp.zeros(h1, dtype=jnp.int32),
            silence2=jnp.zeros(h2, dtype=jnp.int32),
            rate1=jnp.zeros(h1, dtype=jnp.float32),
            rate2=jnp.zeros(h2, dtype=jnp.float32),
            age1=jnp.zeros(h1, dtype=jnp.int32),
            age2=jnp.zeros(h2, dtype=jnp.int32),
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array], state: SNRNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], SNRNormState, StepMetrics]:
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        _, _, a1, _, a2 = _forward_with_activations(params, x_norm)
        new_params = {
            name: params[name] * decay_factor - step_size * grads[name]
            for name in params
        }
        fired1 = a1 > 0.0
        fired2 = a2 > 0.0
        silence1 = jnp.where(fired1, 0, state.silence1 + 1)
        silence2 = jnp.where(fired2, 0, state.silence2 + 1)
        rate1 = rate_decay * state.rate1 + (1.0 - rate_decay) * fired1.astype(jnp.float32)
        rate2 = rate_decay * state.rate2 + (1.0 - rate_decay) * fired2.astype(jnp.float32)
        age1 = state.age1 + 1
        age2 = state.age2 + 1
        key1, key2 = jr.split(key)
        new_params, silence1, rate1, age1, _ = snr_maybe_reset_layer(
            new_params, silence1, rate1, age1, _CBP_LAYERS[0], key1, hp
        )
        new_params, silence2, rate2, age2, _ = snr_maybe_reset_layer(
            new_params, silence2, rate2, age2, _CBP_LAYERS[1], key2, hp
        )
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, SNRNormState(  # type: ignore[call-arg]
            silence1=silence1,
            silence2=silence2,
            rate1=rate1,
            rate2=rate2,
            age1=age1,
            age2=age2,
            norm=new_norm,
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class L2InitNormState:
    """Frozen initial parameters plus the EMA input-normalizer state."""

    init_params: dict[str, Array]
    norm: EMANormState


def _make_l2init_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """L2-Init (Kumar et al., regenerative regularization) behind the
    champion's EMA input normalizer, on the plain-SGD comparison base.

    The decoupled decay pulls toward the INITIAL weights instead of zero —
    ``w <- w - lr*wd*(w - w_0) - lr*grad`` (the same arithmetic form as the
    raw-input ``upgd_l2init`` arm, minus gate and perturbation), with the
    regularization strength carried by ``weight_decay`` (registered 0.01,
    matching the raw arm and the shared base's decay scale).  No utility
    gate, no perturbation; the RNG key is deliberately unused.  With
    ``weight_decay = 0`` the pull vanishes and the trajectory is bit-exact
    against the plain normalized-SGD base (pinned); a hand-computed
    trajectory pins the registered configuration.
    """
    step_size = hp["step_size"]
    wd = hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]

    def init_fn(params: dict[str, Array]) -> L2InitNormState:
        return L2InitNormState(  # type: ignore[call-arg]
            init_params={name: value for name, value in params.items()},
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array],
        state: L2InitNormState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], L2InitNormState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        new_params = {
            name: params[name]
            - step_size * wd * (params[name] - state.init_params[name])
            - step_size * grads[name]
            for name in params
        }
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, L2InitNormState(  # type: ignore[call-arg]
            init_params=state.init_params, norm=new_norm
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class UPGDGatedL2InitNormState:
    """Lean-UPGD utility EMA/clock, a frozen copy of the initial parameters,
    and the EMA input-normalizer state (see
    :func:`_make_sigma0_gated_l2init_learner`)."""

    utility: dict[str, Array]
    step: Array
    init_params: dict[str, Array]
    norm: EMANormState


def _make_sigma0_gated_l2init_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """``sigma0_ndecay099`` baseline plus an additive, utility-gated pull
    toward the initial weights.

    Ported idea: continuous utility-scaled soft resets (CCBP,
    OpenReview:UJqXhFFzKu; Calibrated Partial Resets, arXiv:2607.24996)
    report that a *continuous*, per-unit-utility-scaled partial pull of
    every hidden parameter toward its initial value dominates both
    decay-based (L2/Shrink-and-Perturb) and hard-reset (CBP/ReDO) methods at
    long horizons. This arm tests that mechanism's isolated marginal
    contribution on top of that historical baseline, reusing the baseline's own
    UPGD utility gate as the per-unit "graded reset" weight: ``1 - gate`` is
    large for low-utility (unprotected) units and near zero for high-utility
    (protected) ones, so the pull toward init is concentrated exactly where
    the source papers' utility-scaled reset is meant to act. ``sigma0_ndecay099``
    exposes no separate hidden-unit-firing utility signal of its own (unlike
    CBP-family arms), so the UPGD gate is the closest available substitute
    -- a documented deviation from the source papers' own utility statistic.

    Deviation from the source papers: they replace their baseline's
    decay/reset term outright; here the pull is an ADDITIVE new term next to
    the baseline's existing uniform decoupled weight decay (rather than a
    replacement), so this measures the isolated contribution of graded,
    utility-gated pull-toward-init rather than a full swap of the
    regularizer family.

    ``l2init_pull_scale = 0`` (the default) is inert: the new term is
    multiplied by exactly zero and the step routes through the identical
    ``lean_upgd_w_update`` call the baseline factory's own inert path uses,
    so the trajectory is bit-exact against ``sigma0_ndecay099`` (pinned by a
    unit test) rather than relying on floating-point cancellation of a zero
    term.
    """
    noise_std = hp["noise_std"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    pull_scale = hp.get("l2init_pull_scale", 0.0)
    lean_hp = {
        name: hp[name] for name in ("step_size", "utility_decay", "noise_std", "weight_decay")
    }

    def init_fn(params: dict[str, Array]) -> UPGDGatedL2InitNormState:
        return UPGDGatedL2InitNormState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            init_params={name: value for name, value in params.items()},
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array],
        state: UPGDGatedL2InitNormState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], UPGDGatedL2InitNormState, StepMetrics]:
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        if pull_scale == 0.0:
            # Exact champion path: call the identical function the champion
            # factory's own inert path calls, rather than relying on
            # floating-point cancellation of a zero-scaled extra term.
            lean_state = LeanUPGDState(  # type: ignore[call-arg]
                utility=state.utility, step=state.step
            )
            inert_new_params, new_lean = lean_upgd_w_update(
                params, lean_state, grads, noise, lean_hp
            )
            metrics = _step_metrics(inert_new_params, x_norm, y, loss, logits)
            return (
                inert_new_params,
                UPGDGatedL2InitNormState(  # type: ignore[call-arg]
                    utility=new_lean.utility,
                    step=new_lean.step,
                    init_params=state.init_params,
                    norm=new_norm,
                ),
                metrics,
            )
        beta = hp["utility_decay"]
        step_size = hp["step_size"]
        decay_factor = 1.0 - step_size * hp["weight_decay"]
        count = state.step + jnp.array(1, dtype=jnp.int32)
        utility = {
            name: beta * state.utility[name] + (1.0 - beta) * (-grads[name] * params[name])
            for name in params
        }
        global_max = jnp.max(
            jnp.stack([jnp.max(utility[name]) for name in sorted(params)])
        )
        bias_correction = 1.0 - jnp.power(
            jnp.asarray(beta, dtype=jnp.float32), count.astype(jnp.float32)
        )
        new_params: dict[str, Array] = {}
        for name in params:
            gate = jax.nn.sigmoid((utility[name] / bias_correction) / global_max)
            pull = pull_scale * (1.0 - gate) * (params[name] - state.init_params[name])
            new_params[name] = (
                params[name] * decay_factor
                - step_size * pull
                - step_size * ((grads[name] + noise[name]) * (1.0 - gate))
            )
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return (
            new_params,
            UPGDGatedL2InitNormState(  # type: ignore[call-arg]
                utility=utility, step=count, init_params=state.init_params, norm=new_norm
            ),
            metrics,
        )

    return init_fn, full_step


@chex.dataclass(frozen=True)
class CPRIPMNISTState:
    """Matched peak-state envelope for CPR and every registered control."""

    utility: dict[str, Array]
    init_params: dict[str, Array]
    step: Array
    norm: EMANormState


def _make_cpr_ipmnist_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Calibrated partial reset and matched reduction family.

    The paper defines per-neuron gradient utilities and periodic reset Eq. 7.
    This batch-size-one IPMNIST port uses per-parameter absolute-gradient EMA
    (a finer-grained utility) because the screening learner's state is
    parameter-keyed.  It retains the paper's layer/tensor mean normalization,
    Eq. 6 sigmoid shape (kappa=16), periodic pull, and reset-to-retained-init
    operator.  This protocol difference is bound in the result receipt.
    """
    mode_code = int(hp["mode_code"])
    if mode_code not in range(5):
        raise ValueError("mode_code must select utility/hard/L2/uniform/off")
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    reset_fraction = hp["reset_fraction"]
    reset_frequency = int(hp["reset_frequency"])
    if reset_frequency < 1:
        raise ValueError("reset_frequency must be positive")
    kappa = hp["utility_sharpness"]
    l2_strength = hp["l2_init_strength"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]

    def init_fn(params: dict[str, Array]) -> CPRIPMNISTState:
        return CPRIPMNISTState(  # type: ignore[call-arg]
            utility={name: jnp.ones_like(value) for name, value in params.items()},
            init_params={name: value for name, value in params.items()},
            step=jnp.asarray(0, dtype=jnp.int32),
            norm=_init_input_norm_state(params),
        )

    def full_step(
        params: dict[str, Array],
        state: CPRIPMNISTState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], CPRIPMNISTState, StepMetrics]:
        del key
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        new_step = state.step + jnp.asarray(1, dtype=jnp.int32)
        utility = {}
        for name in params:
            magnitude = jnp.abs(grads[name])
            score = magnitude / (jnp.mean(magnitude) + norm_epsilon)
            utility[name] = (
                utility_decay * state.utility[name] + (1.0 - utility_decay) * score
            )
        sgd_params = {
            name: params[name] - step_size * grads[name] for name in params
        }
        at_reset = jnp.equal(jnp.mod(new_step, reset_frequency), 0)
        new_params: dict[str, Array] = {}
        for name in params:
            mean_utility = jnp.mean(utility[name])
            normalized = utility[name] / jnp.maximum(mean_utility, norm_epsilon)
            calibrated = jnp.minimum(
                2.0 * jax.nn.sigmoid(-kappa * (normalized - 1.0)), 1.0
            )
            if mode_code == 0:  # CPR: periodic continuous utility-scaled pull
                rate = jnp.where(at_reset, reset_fraction * calibrated, 0.0)
            elif mode_code == 1:  # binary hard reset below tensor mean
                rate = jnp.where(at_reset & (normalized <= 1.0), 1.0, 0.0)
            elif mode_code == 2:  # L2-Init: uniform continuous regularization
                rate = jnp.full_like(params[name], step_size * l2_strength)
            elif mode_code == 3:  # utility-free periodic partial reset
                rate = jnp.where(at_reset, reset_fraction, 0.0)
            else:  # exact mechanism-off parameter path
                rate = jnp.zeros_like(params[name])
            pulled = sgd_params[name] + rate * (
                state.init_params[name] - sgd_params[name]
            )
            new_params[name] = sgd_params[name] if mode_code == 4 else pulled
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        recentered_utility = {
            name: (
                jnp.where(at_reset, jnp.ones_like(value), value)
                if mode_code in (0, 1, 3)
                else value
            )
            for name, value in utility.items()
        }
        return new_params, CPRIPMNISTState(  # type: ignore[call-arg]
            utility=recentered_utility,
            init_params=state.init_params,
            step=new_step,
            norm=new_norm,
        ), metrics

    return init_fn, full_step


# =============================================================================
# Config registry
# =============================================================================


def _raw_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Return the fixed protocol input for learners without preprocessing."""
    del state, hyperparameters
    return observation


def _ema_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Apply an EMA learner's current statistics without updating them.

    Online normalized arms update their EMA before predicting each training
    example.  A sentinel probe must be non-learning, so it uses the frozen
    checkpoint statistics.  The normalizer state is part of the checkpoint
    hash and the fixed pixel-permuted sentinel input is separately bound by
    :func:`ipmnist_sentinel_set_sha256`.
    """
    if hyperparameters.get("norm_enabled", 1.0) == 0.0:
        return observation
    norm = getattr(state, "norm", None)
    if not isinstance(norm, EMANormState):
        raise TypeError("an EMA frozen probe requires an EMANormState-backed learner")
    epsilon = hyperparameters.get("norm_epsilon")
    if epsilon is None or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("an EMA frozen probe requires finite positive norm_epsilon")
    return (observation - norm.mean) / (jnp.sqrt(norm.var) + epsilon)


def _hidden_rms_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Refuse sentinel probes for arms whose forward pass is not the plain MLP.

    ``sigma0_hidden_norm`` RMS-normalizes the hidden activations inside the
    forward pass; the probe harness computes logits with ``mlp_logits``, so
    any input-side transform would silently probe the wrong model.  Failing
    closed here is the honest option until the probe harness can accept a
    per-arm forward function.
    """
    del state, observation, hyperparameters
    raise NotImplementedError(
        "sentinel probes are unsupported for hidden-RMS-normalized arms: the "
        "deployed forward pass is not the plain protocol MLP"
    )


def _discovered_rule_frozen_probe_input(
    hyperparameters: Mapping[str, float],
) -> FrozenProbeInputFn:
    """Select the sentinel probe for a discovered-rule arm from its flags.

    ``flag_hidden_rms`` switches the discovered-rule forward pass to the same
    hidden-layer RMS normalization as ``sigma0_hidden_norm``, so those arms
    must fail closed exactly like it; the remaining discovered rules deploy
    the plain MLP behind an EMA input normalizer.
    """
    if float(hyperparameters.get("flag_hidden_rms", 0.0)) != 0.0:
        return _hidden_rms_frozen_probe_input
    return _ema_frozen_probe_input


def _rff_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Refuse sentinel probes for the no-backprop random-features arm.

    ``rff_rls`` never trains the protocol MLP — its deployed model is the
    frozen random projection plus the RLS readout.  The probe harness scores
    ``mlp_logits`` on the (untouched, randomly initialized) MLP params, so
    any input transform here would silently probe a model that does not
    exist.  Fail closed, exactly like :func:`_hidden_rms_frozen_probe_input`,
    so merge/reporting can never emit a meaningless plasticity/retention
    number for this arm.
    """
    del state, observation, hyperparameters
    raise NotImplementedError(
        "sentinel probes are unsupported for the rff_rls arm: there is no "
        "trained protocol MLP to probe (the deployed model is the frozen "
        "random-features + RLS readout)"
    )


@dataclass(frozen=True)
class ScreeningSpec:
    """One screening arm: a named learner configuration.

    Attributes:
        name: Registry key and shard identity.
        base_learner: ``"upgd_w"`` or ``"adamw"`` (cost/reporting bucket).
        mechanism: Short mechanism tag for reporting.
        hyperparameters: Full resolved hyperparameters (JSON-serializable).
        factory: Builds ``(init_fn, step_fn)`` from the hyperparameters.
        description: One-line description for the summary.
        noise_update: Pure noise-consuming update for the pool-noise
            confirmation path (``None`` = pool mode unsupported for this arm).
        frozen_probe_input: Applies the learner's current input preprocessing
            without updating its state.  Raw-input learners use the identity
            transform; adaptive normalizers must opt in explicitly.
    """

    name: str
    base_learner: str
    mechanism: str
    hyperparameters: dict[str, float]
    factory: Callable[[Mapping[str, float]], tuple[LearnerInitFn, ScreeningStepFn]]
    description: str = ""
    noise_update: NoiseUpdateFn | None = None
    frozen_probe_input: FrozenProbeInputFn = _raw_frozen_probe_input

    def __post_init__(self) -> None:
        for attr in ("name", "base_learner", "mechanism"):
            val = getattr(self, attr)
            if type(val) is not str or not val:
                raise ValueError(f"{attr} must be a non-empty string")
        if type(self.base_learner) is not str or self.base_learner not in ("upgd_w", "adamw"):
            raise ValueError("base_learner must name one supported screening learner")
        if type(self.description) is not str:
            raise TypeError("description must be an exact string")
        if type(self.hyperparameters) is not dict:
            raise TypeError("hyperparameters must be a dict")
        normalized: dict[str, float] = {}
        for key, value in self.hyperparameters.items():
            if type(key) is not str or not key:
                raise TypeError("hyperparameter keys must be exact non-empty strings")
            if (type(value) is not int and type(value) is not float) or not math.isfinite(value):
                raise ValueError("hyperparameter values must be finite built-in numbers")
            normalized[key] = float(value)
        object.__setattr__(self, "hyperparameters", normalized)
        if type(self.factory) is not FunctionType:
            raise TypeError("factory must be an exact Python function")
        for name in ("noise_update", "frozen_probe_input"):
            value = getattr(self, name)
            if value is not None and type(value) is not FunctionType:
                raise TypeError(f"{name} must be an exact Python function or None")


def _upgd_hp(**overrides: float) -> dict[str, float]:
    merged = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
    merged.update(overrides)
    return merged


def _sigma0_ext_hp(**overrides: float) -> dict[str, float]:
    """``upgd_ema_norm_sigma0``'s hyperparameters plus inert extension defaults."""
    merged = _upgd_hp(
        norm_decay=0.999,
        norm_epsilon=1e-8,
        noise_std=0.0,
        gate_beta=1.0,
        local_gate=0.0,
        hidden_rms=0.0,
    )
    merged.update(overrides)
    return merged


def _update_rule_hp(**overrides: float) -> dict[str, float]:
    """``sigma0_ndecay099``'s conditioning for the update-rule family swaps.

    Published UPGD-W hyperparameters plus the champion's EMA input-normalizer
    decay 0.99 and ``noise_std = 0`` (no perturbation); each arm adds only
    its own update-rule constants on top.
    """
    merged = _upgd_hp(norm_decay=0.99, norm_epsilon=1e-8, noise_std=0.0)
    merged.update(overrides)
    return merged


def _control_factory(
    make: Callable[[dict[str, float]], tuple[LearnerInitFn, LearnerStepFn]],
) -> Callable[[Mapping[str, float]], tuple[LearnerInitFn, ScreeningStepFn]]:
    def factory(hp: Mapping[str, float]) -> tuple[LearnerInitFn, ScreeningStepFn]:
        return _wrap_grad_learner(*make(dict(hp)))

    return factory


def _make_adamo_raw_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, LearnerStepFn]:
    """AdamO equations 16/19/20 adapted to the protocol MLP.

    Every matrix weight receives the rectangular Gram penalty; bias vectors
    remain unregularized. The task-gradient Adam moments never observe the
    isometry gradient. ``isometry_strength=0`` reduces bit-exactly to the
    protocol AdamW arm (whose selected weight decay is zero).
    """

    optimizer = AdamO(
        AdamOConfig(
            step_size=hp["step_size"],
            beta1=hp["beta1"],
            beta2=hp["beta2"],
            eps=hp["eps"],
            isometry_strength=hp["isometry_strength"],
            isometry_step_size=hp["isometry_step_size"],
        )
    )

    def init_fn(params: dict[str, Array]) -> dict[str, Any]:
        return {
            name: optimizer.init_for_shape(value.shape) for name, value in params.items()
        }

    def step_fn(
        params: dict[str, Array],
        state: dict[str, Any],
        grads: dict[str, Array],
        key: Array,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        del key
        candidate_params: dict[str, Array] = {}
        candidate_state: dict[str, Any] = {}
        update_applied = jnp.asarray(True, dtype=jnp.bool_)
        for name, value in params.items():
            update = optimizer.update_from_gradient_checked(
                state[name], grads[name], value, regularize=value.ndim == 2
            )
            candidate_params[name] = value - update.step
            candidate_state[name] = update.new_state
            update_applied = update_applied & update.update_applied
        update_applied = (
            update_applied
            & floating_tree_is_finite(params)
            & floating_tree_is_finite(state)
            & floating_tree_is_finite(grads)
            & floating_tree_is_finite(candidate_params)
            & floating_tree_is_finite(candidate_state)
        )
        return (
            select_transaction(update_applied, candidate_params, params),
            select_transaction(update_applied, candidate_state, state),
        )

    return init_fn, step_fn


def _make_adamo_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _wrap_grad_learner(*_make_adamo_raw_learner(hp))


def _make_joint_adam_isometry_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Naive equation-18 Adam control whose moments mix both gradients."""

    optimizer = Adam(
        step_size=hp["step_size"],
        beta1=hp["beta1"],
        beta2=hp["beta2"],
        eps=hp["eps"],
        weight_decay=0.0,
    )

    def init_fn(params: dict[str, Array]) -> dict[str, Any]:
        return {
            name: optimizer.init_for_shape(value.shape) for name, value in params.items()
        }

    def step_fn(
        params: dict[str, Array],
        state: dict[str, Any],
        grads: dict[str, Array],
        key: Array,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        del key
        candidate_params: dict[str, Array] = {}
        candidate_state: dict[str, Any] = {}
        update_applied = jnp.asarray(True, dtype=jnp.bool_)
        for name, value in params.items():
            combined_gradient = grads[name]
            if value.ndim == 2 and hp["isometry_strength"] != 0.0:
                combined_gradient = (
                    combined_gradient
                    + hp["isometry_strength"] * isometry_gradient(value)
                )
            update = optimizer.update_from_gradient_checked(
                state[name], combined_gradient, error=None, param=value
            )
            candidate_params[name] = value - update.step
            candidate_state[name] = update.new_state
            update_applied = update_applied & update.update_applied
        update_applied = (
            update_applied
            & floating_tree_is_finite(params)
            & floating_tree_is_finite(state)
            & floating_tree_is_finite(grads)
            & floating_tree_is_finite(candidate_params)
            & floating_tree_is_finite(candidate_state)
        )
        return (
            select_transaction(update_applied, candidate_params, params),
            select_transaction(update_applied, candidate_state, state),
        )

    return _wrap_grad_learner(init_fn, step_fn)


_CBP_DEFAULTS = {
    "cbp_decay_rate": 0.99,
    "cbp_replacement_rate": 1e-4,
    "cbp_maturity_threshold": 100.0,
}


def _build_registry() -> dict[str, ScreeningSpec]:
    specs = [
        ScreeningSpec(
            name="upgd_w_control",
            base_learner="upgd_w",
            mechanism="control",
            hyperparameters=_upgd_hp(),
            factory=_control_factory(_make_upgd_w_learner),
            description="Published UPGD-W (paired control arm; exact full-lane prefix).",
            noise_update=_lean_upgd_noise_update,
        ),
        ScreeningSpec(
            name="adamw_control",
            base_learner="adamw",
            mechanism="control",
            hyperparameters=dict(ADAMW_PROTOCOL_HYPERPARAMETERS),
            factory=_control_factory(_make_adamw_learner),
            description="Published AdamW baseline (proxy-ordering validation arm).",
        ),
        ScreeningSpec(
            name="adamo_inert",
            base_learner="adamw",
            mechanism="decoupled_isometry_inert",
            hyperparameters={
                **ADAMW_PROTOCOL_HYPERPARAMETERS,
                "isometry_strength": 0.0,
                "isometry_step_size": ADAMW_PROTOCOL_HYPERPARAMETERS["step_size"],
            },
            factory=_make_adamo_learner,
            description=(
                "AdamO implementation with lambda=0; exact AdamW mechanism-off reduction."
            ),
        ),
        ScreeningSpec(
            name="adamo_l1e3",
            base_learner="adamw",
            mechanism="decoupled_isometry",
            hyperparameters={
                **ADAMW_PROTOCOL_HYPERPARAMETERS,
                "isometry_strength": 1e-3,
                "isometry_step_size": ADAMW_PROTOCOL_HYPERPARAMETERS["step_size"],
            },
            factory=_make_adamo_learner,
            description=(
                "AdamO equation-20 decoupled Gram-isometry step; paper lambda=1e-3, "
                "adapted to ASI's matched AdamW hyperparameters and IPMNIST schedule."
            ),
        ),
        ScreeningSpec(
            name="adam_iso_joint_l1e3",
            base_learner="adamw",
            mechanism="joint_isometry",
            hyperparameters={
                **ADAMW_PROTOCOL_HYPERPARAMETERS,
                "isometry_strength": 1e-3,
            },
            factory=_make_joint_adam_isometry_learner,
            description=(
                "Naive equation-18 composite-loss Adam control: task and isometry "
                "gradients share moment statistics."
            ),
        ),
        ScreeningSpec(
            name="upgd_idbd",
            base_learner="upgd_w",
            mechanism="per_weight_step_sizes",
            hyperparameters=_upgd_hp(meta_step_size=1e-3, initial_step_size=0.01),
            factory=_make_upgd_idbd_learner,
            description="UPGD-W with IDBD per-weight step-sizes on the gated gradient.",
        ),
        ScreeningSpec(
            name="upgd_idbd_meta1e2",
            base_learner="upgd_w",
            mechanism="per_weight_step_sizes",
            hyperparameters=_upgd_hp(meta_step_size=1e-2, initial_step_size=0.01),
            factory=_make_upgd_idbd_learner,
            description="UPGD-W + IDBD, faster meta step-size.",
        ),
        ScreeningSpec(
            name="upgd_autostep",
            base_learner="upgd_w",
            mechanism="per_weight_step_sizes",
            hyperparameters=_upgd_hp(
                meta_step_size=1e-2, initial_step_size=0.01, tau=1e4
            ),
            factory=_make_upgd_autostep_learner,
            description="UPGD-W with Autostep per-weight step-sizes on the gated gradient.",
        ),
        ScreeningSpec(
            name="upgd_w_idbd_swift",
            base_learner="upgd_w",
            mechanism="per_weight_step_sizes",
            hyperparameters=_upgd_hp(
                meta_step_size=1e-3,
                initial_step_size=0.01,
                swift_eta=0.1,
                swift_eps=0.99,
            ),
            factory=_make_upgd_idbd_swift_learner,
            description=(
                "UPGD-W + IDBD with SwiftTD's overshoot bound (eta) and "
                "persistent step-size decay on trigger (eps)."
            ),
        ),
        ScreeningSpec(
            name="upgd_w_fade_head",
            base_learner="upgd_w",
            mechanism="meta_learned_weight_decay",
            hyperparameters=_upgd_hp(
                fade_alpha=0.005, fade_gamma0=-6.9, fade_theta_lambda=0.1
            ),
            factory=_make_upgd_w_fade_head_learner,
            description=(
                "UPGD-W with FADE meta-learned per-parameter weight decay on "
                "the output layer (w3/b3); hidden layers unchanged."
            ),
        ),
        ScreeningSpec(
            name="upgd_l2init",
            base_learner="upgd_w",
            mechanism="l2_init",
            hyperparameters=_upgd_hp(),
            factory=_make_upgd_l2init_learner,
            description="UPGD-W whose weight decay pulls toward the initial weights.",
        ),
        ScreeningSpec(
            name="upgd_ema_norm",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(norm_decay=0.999, norm_epsilon=1e-8),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description="UPGD-W behind an EMA input normalizer on the 784 pixels.",
        ),
        ScreeningSpec(
            name="upgd_cbp",
            base_learner="upgd_w",
            mechanism="dormant_unit_recycling",
            hyperparameters=_upgd_hp(**_CBP_DEFAULTS),
            factory=_make_upgd_cbp_learner,
            description="UPGD-W with CBP-style dormant-unit recycling.",
        ),
        # --- Wave 5: star around the confirmed upgd_ema_norm result (0.85357
        # at 200 tasks).  Its UPGD-W hyperparameters were tuned for RAW pixel
        # inputs; under EMA-normalized inputs the effective gradient scale,
        # the noise-to-gradient ratio, and the decay pressure all change, so
        # the published values are unlikely to remain optimal.  One axis per
        # arm, same factory.
        ScreeningSpec(
            name="upgd_ema_norm_wd0005",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(
                norm_decay=0.999, norm_epsilon=1e-8, weight_decay=0.005
            ),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "upgd_ema_norm with the independently confirmed better weight "
                "decay 0.005 (composition of the two confirmed wins)."
            ),
        ),
        ScreeningSpec(
            name="upgd_ema_norm_lr003",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(
                norm_decay=0.999, norm_epsilon=1e-8, step_size=0.03
            ),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description="upgd_ema_norm at 3x step size (normalized inputs change scale).",
        ),
        ScreeningSpec(
            name="upgd_ema_norm_lr0003",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(
                norm_decay=0.999, norm_epsilon=1e-8, step_size=0.003
            ),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description="upgd_ema_norm at 1/3 step size.",
        ),
        ScreeningSpec(
            name="upgd_ema_norm_sigma0",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(
                norm_decay=0.999, norm_epsilon=1e-8, noise_std=0.0
            ),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "upgd_ema_norm without the perturbation: is the noise "
                "load-bearing once inputs are conditioned?"
            ),
        ),
        # --- Wave 6: the final dissection of the normalized arm.  With
        # upgd_ema_norm_sigma0 tying upgd_ema_norm, the method reduces to
        # normalize + utility-gated SGD + decay; this arm drops the gate too.
        ScreeningSpec(
            name="sgd_ema_norm",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters={
                "step_size": 0.01,
                "weight_decay": 0.01,
                "norm_decay": 0.999,
                "norm_epsilon": 1e-8,
            },
            factory=_make_sgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "Gate ablation of upgd_ema_norm_sigma0: plain SGD + decoupled "
                "decay behind the exact EMA input normalizer — no utility, no "
                "gate, no noise."
            ),
        ),
        ScreeningSpec(
            name="adamw_cbp",
            base_learner="adamw",
            mechanism="dormant_unit_recycling",
            hyperparameters={**ADAMW_PROTOCOL_HYPERPARAMETERS, **_CBP_DEFAULTS},
            factory=_make_adamw_cbp_learner,
            description="AdamW with CBP-style recycling (Nature-combination reference arm).",
        ),
        # --- Wave 7: single-axis frontier extensions on the confirmed
        # upgd_ema_norm_sigma0 champion (0.85051 at 200 tasks).  The
        # decomposition attributes +0.061 to input conditioning and +0.011 to
        # the utility gate; these arms push the normalizer statistics
        # (ema_normalize already centers with the EMA mean, so decay/epsilon
        # are the unexplored axes), extend conditioning to the hidden layers,
        # and refine the gate under conditioning.  One axis per arm; the
        # shared factory's defaults reduce bit-exactly to the champion.
        ScreeningSpec(
            name="sigma0_hidden_norm",
            base_learner="upgd_w",
            mechanism="hidden_normalization",
            hyperparameters=_sigma0_ext_hp(hidden_rms=1.0, hidden_rms_epsilon=1e-8),
            factory=_make_upgd_ema_norm_ext_learner,
            frozen_probe_input=_hidden_rms_frozen_probe_input,
            description=(
                "upgd_ema_norm_sigma0 plus stateless per-example RMS "
                "normalization of both hidden ReLU layers (no learnable "
                "parameters — conditioning extended past the input)."
            ),
        ),
        ScreeningSpec(
            name="sigma0_localgate",
            base_learner="upgd_w",
            mechanism="local_gate_normalization",
            hyperparameters=_sigma0_ext_hp(local_gate=1.0),
            factory=_make_upgd_ema_norm_ext_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "upgd_ema_norm_sigma0 with the per-tensor gate normalization "
                "(-0.0008 on raw inputs; retested where conditioning rescales "
                "the utilities)."
            ),
        ),
        ScreeningSpec(
            name="guarded_cbp_adam",
            base_learner="adamw",
            mechanism="utility_guarded_recycling",
            hyperparameters={
                **ADAMW_PROTOCOL_HYPERPARAMETERS,
                **_CBP_DEFAULTS,
                "utility_decay": 0.9999,
                "guard_scale": 1.0,
            },
            factory=_make_guarded_cbp_adam_learner,
            description=(
                "AdamW+CBP with UPGD-style utility protection scaling Adam's "
                "applied delta by 1 - gate; no perturbation (CBP regenerates)."
            ),
        ),
        ScreeningSpec(
            name="adamw_cbp_noreset",
            base_learner="adamw",
            mechanism="dormant_unit_recycling",
            hyperparameters={**ADAMW_PROTOCOL_HYPERPARAMETERS, **_CBP_DEFAULTS},
            factory=_make_adamw_cbp_noreset_learner,
            description=(
                "adamw_cbp WITHOUT the per-unit Adam moment/count reset at "
                "replacement (moment-freshness dissection; the leader resets)."
            ),
        ),
        ScreeningSpec(
            name="adamw_cbp_ema_norm",
            base_learner="adamw",
            mechanism="input_normalization_recycling",
            hyperparameters={
                **ADAMW_PROTOCOL_HYPERPARAMETERS,
                **_CBP_DEFAULTS,
                "norm_decay": 0.999,
                "norm_epsilon": 1e-8,
                "norm_enabled": 1.0,
            },
            factory=_make_adamw_cbp_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "adamw_cbp behind the exact upgd_ema_norm EMA input "
                "normalizer (composition of the two orthogonal wins)."
            ),
        ),
        ScreeningSpec(
            name="upgd_w_sigma0",
            base_learner="upgd_w",
            mechanism="perturbation_dissection",
            hyperparameters=_upgd_hp(noise_std=0.0),
            factory=_make_upgd_w_sigma0_learner,
            description=(
                "Lean UPGD-W with sigma=0: pure utility-gated SGD + decoupled "
                "decay, no perturbation (noise draw skipped entirely)."
            ),
        ),
        ScreeningSpec(
            name="upgd_alpha_utility",
            base_learner="upgd_w",
            mechanism="alpha_protection_signal",
            hyperparameters=_upgd_hp(meta_step_size=1e-2, initial_step_size=0.01),
            factory=_make_upgd_alpha_utility_learner,
            description=(
                "UPGD-W whose protection gate reads passive IDBD per-weight "
                "step-size drift instead of the -w*g utility EMA."
            ),
        ),
    ]
    for cbp_overrides, tag in (
        ({"cbp_replacement_rate": 3e-5}, "r3e5"),
        ({"cbp_replacement_rate": 3e-4}, "r3e4"),
        ({"cbp_maturity_threshold": 50.0}, "m50"),
        ({"cbp_maturity_threshold": 200.0}, "m200"),
    ):
        specs.append(
            ScreeningSpec(
                name=f"adamw_cbp_{tag}",
                base_learner="adamw",
                mechanism="dormant_unit_recycling",
                hyperparameters={
                    **ADAMW_PROTOCOL_HYPERPARAMETERS,
                    **_CBP_DEFAULTS,
                    **cbp_overrides,
                },
                factory=_make_adamw_cbp_learner,
                description=(
                    "adamw_cbp leader mini-star: "
                    + ", ".join(f"{k}={v}" for k, v in cbp_overrides.items())
                    + "."
                ),
            )
        )
    for kappa, wd, tag in (
        (1.0, 0.01, "k1"),
        (2.0, 0.01, "k2"),
        (1.0, 0.0, "k1_wd0"),
        (2.0, 0.0, "k2_wd0"),
    ):
        specs.append(
            ScreeningSpec(
                name=f"upgd_w_wclip_{tag}",
                base_learner="upgd_w",
                mechanism="weight_clipping",
                hyperparameters=_upgd_hp(clip_kappa=kappa, weight_decay=wd),
                factory=_make_upgd_w_wclip_learner,
                description=(
                    f"UPGD-W + per-layer weight clipping to kappa={kappa} times the "
                    f"init bound (weight_decay={wd})."
                ),
                noise_update=upgd_w_wclip_update,
            )
        )
    specs.append(
        ScreeningSpec(
            name="upgd_w_localgate",
            base_learner="upgd_w",
            mechanism="local_gate_normalization",
            hyperparameters=_upgd_hp(),
            factory=_make_upgd_w_localgate_learner,
            description="UPGD-W with the utility gate normalized per parameter tensor.",
            noise_update=upgd_w_localgate_update,
        )
    )
    for value, tag in ((0.05, "sigma005"), (0.2, "sigma02")):
        specs.append(
            ScreeningSpec(
                name=f"upgd_w_{tag}",
                base_learner="upgd_w",
                mechanism="hyperparameter_neighborhood",
                hyperparameters=_upgd_hp(noise_std=value),
                factory=_control_factory(_make_upgd_w_learner),
                description=f"UPGD-W with sigma={value}.",
                noise_update=_lean_upgd_noise_update,
            )
        )
    for value, tag in ((0.999, "udecay0999"), (0.99999, "udecay099999")):
        specs.append(
            ScreeningSpec(
                name=f"upgd_w_{tag}",
                base_learner="upgd_w",
                mechanism="hyperparameter_neighborhood",
                hyperparameters=_upgd_hp(utility_decay=value),
                factory=_control_factory(_make_upgd_w_learner),
                description=f"UPGD-W with utility_decay={value}.",
                noise_update=_lean_upgd_noise_update,
            )
        )
    for value, tag in ((0.005, "wd0005"), (0.02, "wd002")):
        specs.append(
            ScreeningSpec(
                name=f"upgd_w_{tag}",
                base_learner="upgd_w",
                mechanism="hyperparameter_neighborhood",
                hyperparameters=_upgd_hp(weight_decay=value),
                factory=_control_factory(_make_upgd_w_learner),
                description=f"UPGD-W with weight_decay={value}.",
                noise_update=_lean_upgd_noise_update,
            )
        )
    for value, tag in (
        (0.99, "ndecay099"),
        (0.9999, "ndecay09999"),
        (0.9, "ndecay09"),
        (0.95, "ndecay095"),
        (0.98, "ndecay098"),
    ):
        specs.append(
            ScreeningSpec(
                name=f"sigma0_{tag}",
                base_learner="upgd_w",
                mechanism="input_normalization",
                hyperparameters=_sigma0_ext_hp(norm_decay=value),
                factory=_make_upgd_ema_norm_ext_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    f"upgd_ema_norm_sigma0 with normalizer decay {value} "
                    "(champion 0.999)."
                ),
            )
        )
    specs.append(
        ScreeningSpec(
            name="ema_norm_ndecay099",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_sigma0_ext_hp(norm_decay=0.99, noise_std=0.1),
            factory=_make_upgd_ema_norm_ext_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "upgd_ema_norm (full sigma=0.1 champion) with normalizer decay 0.99 "
                "(fast-decay winner transplanted onto the noisy champion)."
            ),
        )
    )
    for value, tag in ((1e-6, "eps1e6"), (1e-4, "eps1e4")):
        specs.append(
            ScreeningSpec(
                name=f"sigma0_{tag}",
                base_learner="upgd_w",
                mechanism="input_normalization",
                hyperparameters=_sigma0_ext_hp(norm_epsilon=value),
                factory=_make_upgd_ema_norm_ext_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    f"upgd_ema_norm_sigma0 with normalizer epsilon {value} "
                    "(champion 1e-8; floors the variance and pads the divisor)."
                ),
            )
        )
    for value, tag in ((0.5, "gate_beta05"), (2.0, "gate_beta2")):
        specs.append(
            ScreeningSpec(
                name=f"sigma0_{tag}",
                base_learner="upgd_w",
                mechanism="gate_temperature",
                hyperparameters=_sigma0_ext_hp(gate_beta=value),
                factory=_make_upgd_ema_norm_ext_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    f"upgd_ema_norm_sigma0 with utility-gate temperature beta={value} "
                    "(sigmoid of beta times the scaled utility)."
                ),
            )
        )
    specs.append(
        ScreeningSpec(
            name="sigma0_ndecay099_gated_l2init",
            base_learner="upgd_w",
            mechanism="gated_l2_init",
            hyperparameters=_sigma0_ext_hp(norm_decay=0.99, l2init_pull_scale=0.01),
            factory=_make_sigma0_gated_l2init_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "sigma0_ndecay099 historical baseline plus an additive utility-gated pull "
                "toward init (CCBP/Calibrated-Partial-Resets-style graded reset; "
                "l2init_pull_scale=0.01, matching the repo's established L2-Init "
                "regularization strength); l2init_pull_scale=0 reduces bit-exactly "
                "to the baseline."
            ),
        )
    )
    cpr_base = {
        "mode_code": 0.0,
        "step_size": 0.01,
        "utility_decay": 0.99,
        "reset_fraction": 0.01,
        "reset_frequency": 100.0,
        "utility_sharpness": 16.0,
        "l2_init_strength": 0.01,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }
    cpr_arms = (
        (
            "cpr_ipmnist",
            0.0,
            "CPR Eq. 6/7 supervised port: periodic utility-scaled partial pull to init.",
        ),
        (
            "cpr_hard_reset",
            1.0,
            "Binary below-mean hard-reset control in the matched CPR state envelope.",
        ),
        (
            "cpr_l2_init",
            2.0,
            "Continuous uniform L2-Init pull control in the matched CPR state envelope.",
        ),
        (
            "cpr_utility_free",
            3.0,
            "Periodic uniform partial-pull control without utility calibration.",
        ),
        (
            "cpr_off",
            4.0,
            "Exact mechanism-off normalized-SGD control with matched allocated state.",
        ),
    )
    for name, mode_code, description in cpr_arms:
        specs.append(ScreeningSpec(
            name=name,
            base_learner="upgd_w",
            mechanism="calibrated_partial_reset",
            hyperparameters={**cpr_base, "mode_code": mode_code},
            factory=_make_cpr_ipmnist_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=description,
        ))
    # --- Wave 8: update-rule family swaps under the sigma0_ndecay099 champion's
    # conditioning (EMA input normalizer decay 0.99 + the exact UPGD utility
    # gate, no perturbation).  Only the descent direction changes per arm.
    specs.extend(
        [
            ScreeningSpec(
                name="colnorm_gate",
                base_learner="upgd_w",
                mechanism="update_rule_family",
                hyperparameters=_update_rule_hp(
                    step_size=0.001, col_decay=0.99, col_epsilon=1e-8
                ),
                factory=_make_colnorm_gate_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    "Column-wise RMS-scaled gated SGD under the champion's "
                    "conditioning: per-fan-in-dimension EMA of the squared "
                    "gradient scales the gated step (activation conditioning "
                    "at the weight level); per-element EMA on biases. lr 1e-3 "
                    "(normalized updates: the champion's raw-gradient lr 0.01 "
                    "random-walked to chance; 2-task sweep 3e-4/1e-3/3e-3 -> "
                    ".786/.787/.679 vs champion .719 in the same loop)."
                ),
            ),
            ScreeningSpec(
                name="muon_gate",
                base_learner="upgd_w",
                mechanism="update_rule_family",
                hyperparameters=_update_rule_hp(
                    step_size=0.003, muon_momentum=0.95, muon_ns_steps=5.0
                ),
                factory=_make_muon_gate_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    "Muon-style gated update under the champion's conditioning: "
                    "Nesterov momentum + 5-step Newton-Schulz orthogonalization "
                    "of the 2-D weight updates, sqrt(max/min) shape scaling; "
                    "plain gated SGD on biases. lr 3e-3 (orthogonalized "
                    "updates: lr 0.01 was chance; 2-task sweep 1e-3/3e-3/6e-3/"
                    "1e-2 -> .764/.808/.775/.723 vs champion .719)."
                ),
            ),
            ScreeningSpec(
                name="lion_gate",
                base_learner="upgd_w",
                mechanism="update_rule_family",
                hyperparameters=_update_rule_hp(
                    step_size=0.0001,
                    weight_decay=0.05,
                    lion_beta1=0.9,
                    lion_beta2=0.99,
                ),
                factory=_make_lion_gate_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    "Gated Lion under the champion's conditioning: sign of the "
                    "beta1-interpolated momentum, decoupled decay 0.05. lr 1e-4 "
                    "(sign updates: lr 1e-3 diverged by task 2; 2-task sweep "
                    "1e-4/3e-4 -> .762/.681 vs champion .719)."
                ),
            ),
        ]
    )
    # --- Next-rung wave: shift-triggered re-conditioning normalizers +
    # composed gate refinement, informed by the tracking-speed mechanism
    # (decay 0.99 wins by re-conditioning faster after each permutation).
    shiftnorm_defaults = {
        "fast_decay": 0.9,
        "shift_k": 1.0,
        "shift_delta": 0.02,
        "shift_refractory": 0.0,
    }
    shiftnorm_variants: tuple[tuple[str, dict[str, float]], ...] = (
        ("sigma0_shiftnorm", {}),
        ("sigma0_shiftnorm_k05", {"shift_k": 0.5}),
        ("sigma0_shiftnorm_d099", {"norm_decay": 0.99}),
        # d099 detector mini-star (screen winner base): detector sensitivity
        # (shift_k), detector speed (fast_decay), per-feature trigger
        # rate-limiting (shift_refractory), and the d098 base — the frontier-2
        # decay star showed 0.98 statistically ties 0.99.
        ("sigma0_shiftnorm_d099_k05", {"norm_decay": 0.99, "shift_k": 0.5}),
        ("sigma0_shiftnorm_d099_k2", {"norm_decay": 0.99, "shift_k": 2.0}),
        ("sigma0_shiftnorm_d098", {"norm_decay": 0.98}),
        ("sigma0_shiftnorm_d099_f08", {"norm_decay": 0.99, "fast_decay": 0.8}),
        ("sigma0_shiftnorm_d099_f095", {"norm_decay": 0.99, "fast_decay": 0.95}),
        ("sigma0_shiftnorm_d099_r200", {"norm_decay": 0.99, "shift_refractory": 200.0}),
    )
    for name, shift_overrides in shiftnorm_variants:
        specs.append(
            ScreeningSpec(
                name=name,
                base_learner="upgd_w",
                mechanism="adaptive_input_normalization",
                hyperparameters=_sigma0_ext_hp(
                    **{**shiftnorm_defaults, **shift_overrides}
                ),
                factory=_make_upgd_shiftnorm_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    "upgd_ema_norm_sigma0 with per-feature shift-triggered "
                    "re-conditioning: a fast detection EMA resets a feature's "
                    "anneal count when it diverges from the slow statistics ("
                    + ", ".join(
                        f"{k}={v}"
                        for k, v in {**shiftnorm_defaults, **shift_overrides}.items()
                    )
                    + ")."
                ),
            )
        )
    specs.append(
        ScreeningSpec(
            name="sigma0_warmnorm",
            base_learner="upgd_w",
            mechanism="adaptive_input_normalization",
            hyperparameters=_sigma0_ext_hp(
                fast_decay=0.9,
                warm_threshold=1.0,
                warm_pad=0.01,
                warm_refractory=50.0,
            ),
            factory=_make_upgd_warmnorm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "upgd_ema_norm_sigma0 with batch-stats warmup: a global "
                "fast/slow divergence detector (no task-boundary oracle) "
                "resets the scalar anneal clock so the effective decay warms "
                "up from 1/2 toward 0.999 after each detected shift."
            ),
        )
    )
    specs.append(
        ScreeningSpec(
            name="sigma0_gateplus",
            base_learner="upgd_w",
            mechanism="gate_refinement_composition",
            hyperparameters=_sigma0_ext_hp(
                norm_decay=0.99, local_gate=1.0, gate_beta=2.0
            ),
            factory=_make_upgd_ema_norm_ext_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "sigma0_ndecay099 champion with the two near-flat gate "
                "refinements composed: per-tensor gate normalization AND "
                "temperature beta=2 on the conditioned-gradient utilities."
            ),
        )
    )
    # --- Automated update-rule discovery promotions (rule_discovery lane).
    # Top-3 discovered compositions from the micro-suite search
    # (outputs/rule_discovery/search_v1.json): every one beat the
    # budget-matched tuned champion-form baseline on the held-out micro
    # tasks (M4 + M1', disjoint seeds) AND on the canonical Gaussian
    # cross-suite (incl. the recurrence family). Constants are translated
    # VERBATIM from the discovered genomes (micro-scale-tuned; that
    # timescale mismatch is part of what this screen measures). All three
    # dropped the utility gate and adopted the error-gated plasticity
    # budget (surprise_budget) + hidden RMS normalization.
    discovered_rules: tuple[tuple[str, dict[str, float], str], ...] = (
        (
            "disc_r1",
            {
                "flag_norm": 1.0, "flag_shift_reset": 1.0, "flag_gate": 0.0,
                "flag_decay_to_init": 0.0, "flag_surprise_budget": 1.0,
                "flag_meta_decay": 0.0, "flag_utility_shift_reset": 0.0,
                "flag_w1_shift_reset": 0.0, "flag_hidden_rms": 1.0,
                "step_size": 0.0370901404621786,
                "weight_decay": 0.0001,
                "norm_decay": 0.9911066947977325,
                "fast_decay": 0.8549893343448638,
                "shift_k": 0.7131297990024876,
                "utility_decay": 0.9998283837753099,
                "gate_beta": 0.3374290896461889,
                "surprise_gain": 0.8360796272754669,
                "surprise_fast": 0.9642297768592835,
                "surprise_slow": 0.9996305719081341,
                "meta_gain": 2.142298936843872,
            },
            "Discovered rule 1 (micro holdout 0.6859 vs tuned baseline "
            "0.6165): shift-adaptive input norm + surprise-gated global "
            "step-size budget + hidden RMS; no utility gate.",
        ),
        (
            "disc_r2",
            {
                "flag_norm": 1.0, "flag_shift_reset": 1.0, "flag_gate": 0.0,
                "flag_decay_to_init": 1.0, "flag_surprise_budget": 1.0,
                "flag_meta_decay": 0.0, "flag_utility_shift_reset": 0.0,
                "flag_w1_shift_reset": 0.0, "flag_hidden_rms": 1.0,
                "step_size": 0.04385333652867646,
                "weight_decay": 0.008445640828094932,
                "norm_decay": 0.9645936290647181,
                "fast_decay": 0.9056834226846695,
                "shift_k": 0.6461113343143648,
                "utility_decay": 0.9999295508763486,
                "gate_beta": 0.4251058611416944,
                "surprise_gain": 0.5519864782691002,
                "surprise_fast": 0.9655785930156708,
                "surprise_slow": 0.9996083702106141,
                "meta_gain": 0.5,
            },
            "Discovered rule 2 (micro holdout 0.6763): shift-adaptive norm "
            "+ L2-Init pull (decay-to-init) + surprise budget + hidden RMS; "
            "no utility gate.",
        ),
        (
            "disc_r3",
            {
                "flag_norm": 1.0, "flag_shift_reset": 0.0, "flag_gate": 0.0,
                "flag_decay_to_init": 1.0, "flag_surprise_budget": 1.0,
                "flag_meta_decay": 0.0, "flag_utility_shift_reset": 0.0,
                "flag_w1_shift_reset": 1.0, "flag_hidden_rms": 1.0,
                "step_size": 0.04512338013332415,
                "weight_decay": 0.0004368518845358173,
                "norm_decay": 0.9405970575467439,
                "fast_decay": 0.8795402854681015,
                "shift_k": 1.6857961908692085,
                "utility_decay": 0.9,
                "gate_beta": 0.28608835742384475,
                "surprise_gain": 0.5698174573481083,
                "surprise_fast": 0.9263452136516571,
                "surprise_slow": 0.9996180875840047,
                "meta_gain": 2.1990984678268433,
            },
            "Discovered rule 3 (micro holdout 0.6752): fast EMA norm (no "
            "count reset) + detector-triggered w1 row reinit + L2-Init "
            "pull + surprise budget + hidden RMS; no utility gate.",
        ),
    )
    for disc_name, disc_hp, disc_description in discovered_rules:
        specs.append(
            ScreeningSpec(
                name=disc_name,
                base_learner="upgd_w",
                mechanism="discovered_rule",
                hyperparameters=_discovered_rule_hp(**disc_hp),
                factory=_make_discovered_rule_learner,
                frozen_probe_input=_discovered_rule_frozen_probe_input(disc_hp),
                description=disc_description,
            )
        )
    # Structure-vs-constants dissection of disc_r1 (the strongest discovered
    # rule, which lost -0.080 to the champion at its verbatim micro-tuned
    # constants while beating the published UPGD-W control +0.006): the same
    # discovered composition (surprise budget, no utility gate) at the
    # champion-scale constants, with hidden RMS isolated as its own axis
    # (hidden RMS measured -0.0186 on the champion in the sigma0 star).
    for diag_name, diag_rms, diag_axis in (
        ("disc_r1_pscale", 1.0, "with hidden RMS"),
        ("disc_r1_pscale_norms", 0.0, "without hidden RMS"),
    ):
        specs.append(
            ScreeningSpec(
                name=diag_name,
                base_learner="upgd_w",
                mechanism="discovered_rule_diagnostic",
                hyperparameters=_discovered_rule_hp(
                    flag_norm=1.0,
                    flag_shift_reset=1.0,
                    flag_surprise_budget=1.0,
                    flag_hidden_rms=diag_rms,
                    surprise_gain=0.8360796272754669,
                    surprise_fast=0.9642297768592835,
                    surprise_slow=0.9996305719081341,
                ),
                factory=_make_discovered_rule_learner,
                frozen_probe_input=_discovered_rule_frozen_probe_input(
                    {"flag_hidden_rms": diag_rms}
                ),
                description=(
                    "disc_r1 structure at champion-scale constants "
                    f"({diag_axis}): shift-adaptive norm + surprise-gated "
                    "step-size budget, no utility gate; lr/wd/decays from "
                    "sigma0_shiftnorm_d099."
                ),
            )
        )
    # --- Pre-registered existential control: no backprop, no MLP.
    specs.append(
        ScreeningSpec(
            name="rff_rls",
            # Reporting/cost bucket only (the schema allows upgd_w|adamw and
            # every candidate arm is paired against upgd_w_control); no
            # gradient plumbing is engaged — the arm ignores the MLP.
            base_learner="upgd_w",
            mechanism="random_features",
            hyperparameters={
                "rff_m": 1024.0,
                "rff_gamma": 0.001,
                "rff_clip": 3.0,
                "rls_lambda": 0.999,
                "rls_ridge_init": 1.0,
                "norm_decay": 0.99,
                "norm_epsilon": 1e-8,
                "noise_std": 0.0,
            },
            factory=_make_rff_rls_learner,
            frozen_probe_input=_rff_frozen_probe_input,
            description=(
                "No-backprop tracking control: champion EMA input normalizer "
                "(decay 0.99), z-scores clipped to +/-3, frozen random "
                "Fourier features (m=1024, Omega ~ N(0, 0.001*I) — bandwidth "
                "calibrated after the 0.05 draft scored chance-level phase "
                "noise), streaming one-vs-all RLS readout (forgetting 0.999, "
                "ridge init 1.0). If this matches the deep arms, the "
                "benchmark measures tracking rather than learning."
            ),
        )
    )
    specs.append(
        ScreeningSpec(
            name="lin_rls",
            base_learner="upgd_w",
            mechanism="random_features",
            hyperparameters={
                "rff_m": 785.0,
                "rff_gamma": 0.0,
                "rff_clip": 3.0,
                "rls_lambda": 0.999,
                "rls_ridge_init": 1.0,
                "norm_decay": 0.99,
                "norm_epsilon": 1e-8,
                "noise_std": 0.0,
            },
            factory=_make_lin_rls_learner,
            frozen_probe_input=_rff_frozen_probe_input,
            description=(
                "Linear floor of the tracking control: champion EMA input "
                "normalizer, z-scores clipped to +/-3 and scaled by "
                "1/sqrt(784) with a bias feature, streaming one-vs-all RLS "
                "readout (forgetting 0.999). No features at all — measures "
                "how far pure linear tracking goes on this protocol."
            ),
        )
    )
    # --- V3 development validation: streaming generative classifier (no network).
    specs.append(
        ScreeningSpec(
            name="naive_bayes",
            # Reporting/cost bucket only (schema allows upgd_w|adamw; every
            # candidate arm pairs against upgd_w_control); no gradient
            # plumbing is engaged — the arm ignores the MLP entirely.
            base_learner="upgd_w",
            mechanism="streaming_generative_classifier",
            hyperparameters={
                "nb_decay": 0.98,
                "nb_var_epsilon": 0.1,
                "noise_std": 0.0,
            },
            factory=_make_naive_bayes_learner,
            frozen_probe_input=_naive_bayes_frozen_probe_input,
            description=(
                "Streaming naive Bayes (V3 development validation): online "
                "class-conditional diagonal Gaussians with annealed "
                "fast-EMA statistics, prediction = argmax posterior; no "
                "gradients, no MLP. nb_decay 0.98 / var floor 0.1 from the "
                "2-task seed-0 diagnostic (decay {0.95..0.9999} x floor "
                "{0.001..0.5}: floor 0.1 dominates every decay; 0.98 edges "
                "0.99 .7939/.7915 with the best post-shift task-2 recovery "
                ".7892 — the same 0.98-0.99 plateau as the champion's "
                "conditioning-decay star)."
            ),
        )
    )
    # --- Transient attack (CEILING_ANALYSIS budget (i)): adaptive ensemble
    # of the sigma0_shiftnorm_d099 champion and the naive_bayes tracker.
    # Vote weights are learned ONLINE from per-member recent-accuracy EMAs
    # (softmax temperature ens_beta); no oracle, no task-boundary signal.
    for ens_name, ens_overrides, ens_extra in (
        (
            "nb_ensemble_champion",
            {},
            "the base two-member vote (champion MLP + naive Bayes)",
        ),
        (
            "nb_ensemble_nbreset",
            {"ens_nb_reset": 1.0},
            "plus detector-driven NB anneal-clock resets (probe b: "
            "shift-robust NB statistics; champion detector constants, "
            "trigger frac 0.03 from the seed-0 boundary/mid-task "
            "separation, refractory 500)",
        ),
        (
            "nb_ensemble_rls3",
            {"ens_use_rls": 1.0},
            "plus a third closed-form member, linear RLS over normalized "
            "pixels (probe c: lin_rls pipeline verbatim)",
        ),
    ):
        specs.append(
            ScreeningSpec(
                name=ens_name,
                base_learner="upgd_w",
                mechanism="transient_ensemble",
                hyperparameters=_nb_ensemble_hp(**ens_overrides),
                factory=_make_nb_ensemble_learner,
                frozen_probe_input=_nb_ensemble_frozen_probe_input,
                description=(
                    "Adaptive champion/NB ensemble (transient attack): "
                    "accuracy-weighted probability mixture with online vote "
                    "weights from annealed per-member correctness EMAs — "
                    + ens_extra
                    + ". Member constants verbatim from their arms; "
                    "ens_decay 0.995 / ens_beta 80 frozen at the 3-round "
                    "2-task seed-0 diagnostic argmax (0.8496 vs champion "
                    "0.8250 in the same loop) before the screen."
                ),
            )
        )
    # --- Convergence-shortfall attack (CEILING_ANALYSIS budget: 0.904
    # plateau vs 0.933 family asymptote): champion body + streaming-RLS
    # readout on the penultimate features (section (w) factory).  Constants
    # frozen by the 2-task seed-0 diagnostic (champion 0.825 in the same
    # loop): lambda star .8302/.8361/.8217 for 0.995/0.999/1.0 with the
    # task-1 AND task-2 gains the mechanism predicts; P-reset threshold
    # 0.05 calibrated from the detector's shifted fraction (within-task max
    # 0.018, boundary step 0.061 — 2.8x margin); residual-driven body kept
    # at lambda 0.999 only (its lambda-0.995 variant collapsed to 0.105 on
    # task 2 — a fast-forgetting head is unstable as the body's error
    # signal; ledgered, not registered).
    for rls_name, rls_overrides, rls_extra in (
        (
            "rls_head_l0999",
            {"rls_lambda": 0.999},
            "forgetting 0.999 (the diagnostic winner, +0.011 over the "
            "champion at 2 tasks)",
        ),
        (
            "rls_head_l0995",
            {"rls_lambda": 0.995},
            "forgetting 0.995 (fast staleness discount; best diagnostic "
            "task-1)",
        ),
        (
            "rls_head_l1",
            {"rls_lambda": 1.0},
            "no forgetting (growing-window exact least squares; the "
            "staleness control)",
        ),
        (
            "rls_head_l0999_preset005",
            {"rls_lambda": 0.999, "rls_reset_frac": 0.05},
            "forgetting 0.999 plus the detector-driven P reset (probe b: "
            "reuse the champion's shift detector; 2-task read -0.0066 on "
            "task 2, screened across 59 boundaries anyway)",
        ),
        (
            "rls_head_resid",
            {"rls_lambda": 0.999, "head_resid": 1.0},
            "body trained from the RLS head's own residual (probe c, "
            "cleanest error signal; best diagnostic task-2, 0.8774)",
        ),
        # Wave 2 — wind-up stabilized.  Wave-1 measurement: every lambda<1
        # arm collapsed to chance mid-screen (P grows as (1/lambda)^t along
        # unexcited dead-ReLU feature directions; float32 overflow ~ e^88.7
        # ~ 88k steps ~ task 18 at lambda 0.999 — the observed onset), a
        # failure mode the dense-feature rff_rls precedent could never see.
        # lambda=1 cannot wind up (P is nonincreasing PSD); staleness is
        # handled by the detector-driven P reset instead of by forgetting.
        (
            "rls_head_l1_preset005",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05},
            "no forgetting + detector-driven P reset at the calibrated 0.05 "
            "fraction (wind-up-immune staleness handling: exact LS within a "
            "task, fresh gain at detected shifts)",
        ),
        (
            "rls_head_l1_preset003",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.03},
            "no forgetting + P reset at the more sensitive 0.03 fraction "
            "(1.7x margin over the within-task detector maximum; catches "
            "weaker boundaries at 59-boundary scale)",
        ),
        (
            "rls_head_l0999_pcap",
            {"rls_lambda": 0.999, "rls_p_trace_cap": 1e4},
            "forgetting 0.999 with the P trace cap 1e4 (66x the init trace; "
            "salvages the forgetting mechanism as a bounded probe)",
        ),
        (
            "rls_head_resid_l1_preset005",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0},
            "residual-driven body on the wind-up-immune head (probe c "
            "rerun on the stable configuration)",
        ),
        (
            "rls_head_resid_l1_preset005_nogate",
            {
                "rls_lambda": 1.0,
                "rls_reset_frac": 0.05,
                "head_resid": 1.0,
                "gate_scale": 0.0,
            },
            "issue #52's preregistered gate ablation of the standing "
            "residual-trained incumbent: plain decayed SGD on the body, "
            "with no utility EMA, bias correction, or sigmoid gate",
        ),
        # Wave 3 — ridge star.  2-task seed-0 diagnostic: the initial/reset
        # ridge is the head's convergence-speed knob (P0 = I/ridge bounds
        # the earliest gains); means .8328/.8465/.8530/.8578/.8596 for
        # ridge 1.0/0.3/0.1/0.03/0.01 (champion 0.825), monotone toward
        # small ridge, with the gain on BOTH the from-scratch task (t1
        # .8426 at 0.01 vs champion .7848) and the post-shift task.  The
        # residual body reruns at small ridge (.8648 at 2 tasks, family
        # best: a fast-converging head makes its residual reliable early).
        (
            "rls_head_l0999_preset005_r01",
            {"rls_lambda": 0.999, "rls_reset_frac": 0.05,
             "rls_ridge_init": 0.1},
            "forgetting 0.999 + P reset 0.05 + ridge 0.1 (diag2 ridge-star "
            "winner)",
        ),
        (
            "rls_head_l0999_preset005_r003",
            {"rls_lambda": 0.999, "rls_reset_frac": 0.05,
             "rls_ridge_init": 0.03},
            "forgetting 0.999 + P reset 0.05 + ridge 0.03",
        ),
        (
            "rls_head_l0999_preset005_r001",
            {"rls_lambda": 0.999, "rls_reset_frac": 0.05,
             "rls_ridge_init": 0.01},
            "forgetting 0.999 + P reset 0.05 + ridge 0.01 (smallest probed "
            "ridge; frontier still rising at 2 tasks)",
        ),
        (
            "rls_head_resid_preset005_r01",
            {"rls_lambda": 0.999, "rls_reset_frac": 0.05,
             "rls_ridge_init": 0.1, "head_resid": 1.0},
            "residual-driven body at ridge 0.1 + P reset (family-best "
            "2-task diagnostic 0.8648; tests whether the small-ridge head "
            "stabilizes the body-chases-head feedback loop that collapsed "
            "at ridge 1.0 without resets)",
        ),
        (
            "rls_head_resid_preset005_r001",
            {"rls_lambda": 0.999, "rls_reset_frac": 0.05,
             "rls_ridge_init": 0.01, "head_resid": 1.0},
            "residual-driven body at ridge 0.01 + P reset (ridge direction "
            "probe on the residual loop)",
        ),
        # Preconditioned residual signal.  CEILING_ANALYSIS.md puts 0.029
        # of the champion's error on within-task convergence speed.  The
        # RLS head cashed in the READOUT's share of that (closed-form, ~d
        # samples); the BODY's share is untouched — it is still gated SGD
        # on an unconditioned feature-space error g = wout @ err.  Every
        # preconditioner screened against that share (IDBD, Autostep,
        # Adam+CBP) failed on continual stability, not on speed; both
        # directions below are built from state the incumbent already
        # carries and the shift detector already resets at boundaries, so
        # neither adds a cross-permutation carry of its own.  Each is
        # renormalized to ||g||, so the frozen step size still applies
        # (negative result #1).  resid_whiten=0 is the incumbent, bitwise.
        (
            "rls_head_resid_l1_preset005_gn",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0,
             "resid_whiten": 1.0},
            "body error signal whitened by the head's inverse feature "
            "correlation (delta = P g, renormalized to ||g||) — a "
            "heuristic, NOT Gauss-Newton: the curvature in phi is "
            "wout @ wout.T, not the activation second moment",
        ),
        (
            "rls_head_resid_l1_preset005_gn05",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0,
             "resid_whiten": 0.5},
            "half-whitened body error signal (the interpolation hedge)",
        ),
        (
            "rls_head_resid_l1_preset005_tp",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0,
             "resid_whiten": 1.0, "resid_newton": 1.0},
            "feature-space Newton body signal (delta = wout gram^-1 err, "
            "gram = wout.T wout ridged; the minimum-norm change in phi "
            "that would zero the head residual), renormalized to ||g||",
        ),
        # Composition cell.  The feature-space Newton direction (+0.001672,
        # precond_r1) and issue #1937's gate removal (+0.001712,
        # gate_ablation_r2) are independent modifications of the same body
        # update that each measured a real effect and each missed the +0.002
        # bar alone.  They have never been run together.  If the utility gate
        # is blunting the preconditioned direction, removing it should let
        # more of that direction through, so this cell is not merely the sum
        # of two main effects — it is where the interaction lives.
        (
            "rls_head_resid_l1_preset005_tp_nogate",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0,
             "resid_whiten": 1.0, "resid_newton": 1.0, "gate_scale": 0.0},
            "feature-space Newton body signal on the ungated body (plain "
            "decayed SGD): the untested cell of the gate x preconditioner "
            "2x2",
        ),
        (
            "rls_head_resid_l1_preset005_tp05",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0,
             "resid_whiten": 0.5, "resid_newton": 1.0},
            "half feature-space Newton body signal (the interpolation "
            "hedge)",
        ),
        # The untested cell of the head 2x2: the capped forgetting head is
        # the best PASSENGER-head arm measured (rls_head_l0999_pcap
        # 0.86608 vs rls_head_l1_preset005 0.86480) but has never carried
        # the residual body signal that moved the plateau.  Forgetting runs
        # here under BOTH wind-up guards at once (trace cap and P reset).
        (
            "rls_head_resid_l0999_pcap",
            {"rls_lambda": 0.999, "rls_reset_frac": 0.05, "head_resid": 1.0,
             "rls_p_trace_cap": 1e4},
            "residual-driven body on a forgetting-0.999 head held bounded "
            "by both wind-up guards (trace cap 1e4 and the detector-driven "
            "P reset at 0.05)",
        ),
    ):
        body_update = (
            "plain decayed residual SGD (utility bookkeeping removed)"
            if rls_overrides.get("gate_scale", 1.0) == 0.0
            else "utility-gated sigma-0 SGD"
        )
        specs.append(
            ScreeningSpec(
                name=rls_name,
                base_learner="upgd_w",
                mechanism="rls_readout",
                hyperparameters=_rls_head_hp(**rls_overrides),
                factory=_make_rls_head_learner,
                frozen_probe_input=_rls_head_frozen_probe_input,
                description=(
                    "Champion body (shift-adaptive EMA-norm d099 + "
                    + body_update
                    + ") with a streaming-RLS "
                    "one-hot readout on the bias-augmented 150-dim "
                    "penultimate features — " + rls_extra + ". One-hot LS "
                    "regression + argmax by design (softmax/logistic "
                    "targets admit no exact RLS recursion)."
                ),
            )
        )
    # Online permutation identification + input remap (V7/V8 chain).  V8
    # measured that a single-shot remap at N=200 post-shift samples and V1's
    # measured 0.62 identification accuracy lifts the incumbent to 0.8997;
    # a refining identifier rides the upper envelope.  These arms implement
    # the real mechanism: V1 class-conditional fingerprint accumulated
    # online, champion shift detector on the raw stream, Hungarian match at
    # the preregistered sample counts.  ident_match_at=0 delegates verbatim
    # to the incumbent factory (bit-exact reduction, pinned).
    for ident_name, ident_overrides, ident_extra in (
        (
            "rls_head_resid_identmap200",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0,
             "ident_match_at": 200.0},
            "single Hungarian match at 200 post-shift samples (V8's best "
            "single-shot cell)",
        ),
        (
            "rls_head_resid_identmap200_r",
            {"rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0,
             "ident_match_at": 200.0, "ident_match2": 500.0,
             "ident_match3": 2000.0},
            "matches at 200/500/2000 post-shift samples (rides V1's "
            "accuracy-vs-N curve: 0.62 -> 0.79 -> 0.84)",
        ),
    ):
        specs.append(
            ScreeningSpec(
                name=ident_name,
                base_learner="upgd_w",
                mechanism="rls_readout",
                hyperparameters=_rls_head_hp(**ident_overrides),
                factory=_make_rls_head_identmap_learner,
                frozen_probe_input=_rls_head_frozen_probe_input,
                description=(
                    "Residual RLS-head incumbent behind an online "
                    "permutation identifier: " + ident_extra + ". Labels "
                    "consumed post-prediction (protocol-legal)."
                ),
            )
        )
    specs.append(
        ScreeningSpec(
            name="rls_head_resid_l1_preset005_l2init",
            base_learner="upgd_w",
            mechanism="rls_readout",
            hyperparameters=_rls_head_l2init_hp(),
            factory=_make_rls_head_l2init_learner,
            frozen_probe_input=_rls_head_frozen_probe_input,
            description=(
                "Code-only issue-#14 endpoint: the exact "
                "rls_head_resid_l1_preset005 incumbent with decoupled "
                "L2-to-initialization on w1/b1/w2/b2 only; w3/b3 and the "
                "RLS recursion are unchanged. This registry entry has no "
                "result artifact and does not authorize execution."
            ),
        )
    )
    # --- Optimizer-floor hybrid wave (section (s) factories): Adam-class
    # step adaptation under the champion's full stability package.  The
    # naive composition adamw_cbp_ema_norm proved Adam-class task-1
    # convergence transfers under conditioning (0.8425, best measured) and
    # then decayed forever because it carried no gate, no weight decay, and
    # the slow normalizer; these arms attach exactly that missing package.
    hybrid_stability: dict[str, float] = {
        "weight_decay": 0.01,
        "utility_decay": 0.9999,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }
    hybrid_shift: dict[str, float] = {
        "fast_decay": 0.9,
        "shift_k": 1.0,
        "shift_delta": 0.02,
        "shift_refractory": 0.0,
    }
    # Per-arm lr from the 2-task diagnostic sweep (seed 0, 2 x 5000 steps,
    # champion 0.826 in the same loop): beta2 0.9 at 1e-4/3e-4/1e-3/3e-3 ->
    # .765/.834/.850/.812; beta2 0.99 -> .802/.856/.855/.805.
    for name, adam_lr, adam_overrides, adam_note in (
        (
            "norm_adam_fastv",
            0.001,
            {"beta2": 0.9, "vreset_enabled": 1.0},
            "fast v (beta2 0.9) + shift-triggered w1-row moment resets; "
            "lr 1e-3 (its sweep argmax .850)",
        ),
        (
            "norm_adam_fastv_b2099",
            0.0003,
            {"beta2": 0.99, "vreset_enabled": 1.0},
            "protocol v (beta2 0.99) + shift-triggered w1-row moment resets; "
            "lr 3e-4 (its sweep argmax .856)",
        ),
        (
            "norm_adam_gate",
            0.0003,
            {"beta2": 0.99, "vreset_enabled": 0.0},
            "protocol v (beta2 0.99), moments carried — the reset dissection "
            "at the b2099 arm's lr 3e-4 (single-axis pair)",
        ),
    ):
        specs.append(
            ScreeningSpec(
                name=name,
                base_learner="adamw",
                mechanism="optimizer_floor_hybrid",
                hyperparameters={
                    **hybrid_stability,
                    **hybrid_shift,
                    "step_size": adam_lr,
                    "beta1": 0.0,
                    "eps": 1e-8,
                    **adam_overrides,
                },
                factory=_make_norm_adam_fastv_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    "Optimizer-floor hybrid: gated AdamW behind the "
                    "shift-adaptive d099 normalizer with the champion's "
                    f"gate + wd stability package — {adam_note}."
                ),
            )
        )
    specs.append(
        ScreeningSpec(
            name="norm_rmsprop_gate",
            base_learner="adamw",
            mechanism="optimizer_floor_hybrid",
            hyperparameters={
                **hybrid_stability,
                "step_size": 0.001,
                "rms_rho": 0.9,
                "rms_epsilon": 1e-8,
            },
            factory=_make_norm_rmsprop_gate_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "Optimizer-floor hybrid: gated classic RMSprop (rho 0.9, no "
                "momentum, no bias correction) behind the champion's "
                "conditioning + gate + wd. lr 1e-3 (2-task sweep 1e-4/3e-4/"
                "1e-3/3e-3 -> .748/.823/.847/.807 vs champion .826)."
            ),
        )
    )
    specs.append(
        ScreeningSpec(
            name="norm_apollo_gate",
            base_learner="adamw",
            mechanism="optimizer_floor_hybrid",
            hyperparameters={
                **hybrid_stability,
                "step_size": 0.0003,
                "apollo_decay": 0.99,
                "apollo_epsilon": 1e-8,
            },
            factory=_make_norm_apollo_gate_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "Optimizer-floor hybrid: APOLLO-style per-neuron (fan-out) "
                "channel-wise gradient scaling (Zhu et al. 2024, exact "
                "channel statistics, beta1=0 limit) behind the champion's "
                "conditioning + gate + wd. lr 3e-4 (2-task sweep 1e-4/3e-4/"
                "1e-3/3e-3 -> .821/.861/.843/.755 vs champion .826 — the "
                "sweep's best arm)."
            ),
        )
    )
    for name, mu in (("sgd_momentum_gate", 0.9), ("sgd_momentum_gate_m099", 0.99)):
        specs.append(
            ScreeningSpec(
                name=name,
                base_learner="upgd_w",
                mechanism="optimizer_floor_hybrid",
                hyperparameters={
                    **hybrid_stability,
                    "step_size": 0.01,
                    "momentum": mu,
                },
                factory=_make_sgd_momentum_gate_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    "Optimizer-floor hybrid: the sigma0_ndecay099 champion "
                    f"update with EMA-bias-corrected momentum {mu} as the "
                    "descent direction (momentum=0 reduces bit-exactly to "
                    "the champion; pinned). lr 0.01, champion parity (2-task "
                    "sweep 3e-3/1e-2/3e-2 -> .730/.822/.826 at mu 0.9, "
                    ".717/.810/.821 at mu 0.99, champion .826 — momentum is "
                    "flat at 2 tasks; the screen decides the horizon)."
                ),
            )
        )
    # --- Reviewer comparison rows: the strongest published plasticity
    # mechanisms, re-implemented from their papers, behind the champion's
    # EMA input conditioning (decay 0.99) on a plain-SGD base — no utility
    # gate, no perturbation.  Together with sigma0_ndecay099 (conditioning +
    # OUR gate) and sgd_ema_norm_d099 (conditioning + nothing) they complete
    # the table "our conditioning + THEIR mechanism vs our conditioning +
    # our gate".
    comparison_base: dict[str, float] = {
        "step_size": 0.01,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }
    specs.append(
        ScreeningSpec(
            name="sgd_ema_norm_d099",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters={**comparison_base, "weight_decay": 0.01},
            factory=_make_sgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "Comparison-table base: plain SGD + decoupled decay behind "
                "the champion's decay-0.99 EMA input normalizer "
                "(sgd_ema_norm retimed to the champion's conditioning; no "
                "gate, no noise — the mechanism-free floor for the "
                "comparison rows)."
            ),
        )
    )
    specs.append(
        ScreeningSpec(
            name="wclip_ema_norm",
            base_learner="upgd_w",
            mechanism="weight_clipping",
            hyperparameters={
                **comparison_base, "weight_decay": 0.0, "clip_kappa": 2.0
            },
            factory=_make_wclip_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "Weight Clipping (Elsayed et al., RLC 2024) behind the "
                "champion's EMA normalizer: plain SGD then per-layer clip of "
                "weights AND biases to +/- 2/sqrt(fan_in) (their standalone "
                "Algorithm 1, wd 0). Does conditioning rescue clipping "
                "(raw-input upgd_w_wclip_k2 screened -0.0056)?"
            ),
        )
    )
    specs.append(
        ScreeningSpec(
            name="fade_head_ema_norm",
            base_learner="upgd_w",
            mechanism="meta_learned_weight_decay",
            hyperparameters={
                **comparison_base,
                "weight_decay": 0.0,
                "fade_alpha": 0.005,
                "fade_gamma0": -6.9,
                "fade_theta_lambda": 0.1,
            },
            factory=_make_fade_head_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "FADE meta-learned per-parameter weight decay on the output "
                "layer (arXiv:2604.27063 — the label-permuted EMNIST "
                "leader's mechanism) behind the champion's EMA normalizer on "
                "plain SGD; hidden layers undecayed (the paper adapts the "
                "final layer only). Published constants alpha=0.005, "
                "gamma0=-6.9, theta=0.1 (raw-input arm screened -0.019)."
            ),
        )
    )
    specs.append(
        ScreeningSpec(
            name="snr_ema_norm",
            base_learner="upgd_w",
            mechanism="neuron_reset_hypothesis_test",
            hyperparameters={
                **comparison_base,
                "weight_decay": 0.0,
                "snr_eta": 0.005,
                "snr_rate_decay": 0.999,
                "snr_rate_floor": 1e-4,
            },
            factory=_make_snr_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "Self-Normalized Resets (Farias & Jozefiak, "
                "arXiv:2410.20098) behind the champion's EMA normalizer on "
                "plain SGD: per-unit geometric-tail hypothesis test "
                "P(A >= a) <= eta on EMA-estimated firing rates; rejected "
                "units re-init incoming weights+bias (protocol uniform) and "
                "zero outgoing. eta=0.005 from the paper's PM sweep grid."
            ),
        )
    )
    specs.append(
        ScreeningSpec(
            name="l2init_ema_norm",
            base_learner="upgd_w",
            mechanism="l2_init",
            hyperparameters={**comparison_base, "weight_decay": 0.01},
            factory=_make_l2init_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "L2-Init (Kumar et al.) behind the champion's EMA "
                "normalizer on plain SGD: decoupled decay pulls toward the "
                "initial weights instead of zero (lambda = wd = 0.01, the "
                "raw-input upgd_l2init value; that arm screened +0.0014 and "
                "confirmed 0.78042 at 200 tasks)."
            ),
        )
    )
    # L2-ER is adapted from lop-jax commit 52ae3eb: exact estimator and
    # alternating update semantics on ASI's current, matched IPMNIST runner.
    # The center of each official sweep grid is predeclared here; this is not
    # represented as the paper's selected hyperparameter configuration.
    l2er_base: dict[str, float] = {
        "step_size": 1e-3,
        "er_batch_size": 100.0,
        "er_steps_per_batch": 1.0,
        "er_epsilon": 1e-8,
    }
    for arm_name, weight_decay, er_step_size, er_enabled, arm_description in (
        (
            "l2er_mechanism_off",
            0.0,
            0.0,
            0.0,
            "Matched plain-SGD mechanism-off control with the charged ER buffer.",
        ),
        (
            "l2er_l2_only",
            1e-4,
            0.0,
            0.0,
            "L2-only reduction at the center of the official weight-decay grid.",
        ),
        (
            "l2er_er_only",
            0.0,
            1e-3,
            1.0,
            "Effective-rank-only reduction at the center of the official ER grid.",
        ),
        (
            "l2er_combined",
            1e-4,
            1e-3,
            1.0,
            "Combined L2-ER at the centers of the official sweep grids.",
        ),
    ):
        specs.append(
            ScreeningSpec(
                name=arm_name,
                base_learner="upgd_w",
                mechanism="l2_effective_rank",
                hyperparameters={
                    **l2er_base,
                    "weight_decay": weight_decay,
                    "er_step_size": er_step_size,
                    "er_enabled": er_enabled,
                },
                factory=_make_l2er_learner,
                description=(
                    arm_description
                    + " Exact lop-jax entropy-rank estimator and update ordering; "
                    "adapted architecture/stream differences are protocol-bound."
                ),
            )
        )
    # Intentional Updates is published for streaming RL, not supervised
    # classification.  These arms freeze the smallest IPMNIST extension and
    # its required controls; they must never be described as a reproduction
    # of the paper's experiments or results.
    intentional_base = {
        "intentional_enabled": 1.0,
        "intended_fraction": 0.5,
        "fixed_step_size": 0.01,
        "beta2": 0.999,
        "optimizer_epsilon": 1e-8,
        "beta_clip": 0.9998,
        "clip_mult": 20.0,
        "use_diagonal_normalization": 1.0,
        "use_adaptive_clip": 1.0,
        "update_features": 1.0,
        "weight_decay": 0.0,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }
    intentional_arms = (
        (
            "intentional_updates_ipmnist",
            intentional_base,
            "Full supervised Eq. 5 extension: correct-class log-probability target, "
            "RMSProp diagonal direction, and official adaptive clipping constants.",
        ),
        (
            "intentional_updates_no_diag",
            {**intentional_base, "use_diagonal_normalization": 0.0},
            "Ablation removing the paper's RMSProp diagonal normalization.",
        ),
        (
            "intentional_updates_no_clip",
            {**intentional_base, "use_adaptive_clip": 0.0},
            "Ablation removing delta/surprisal clipping entirely.",
        ),
        (
            "intentional_updates_head_only",
            {**intentional_base, "update_features": 0.0},
            "Feature-control ablation: backpropagate normally but update only the head.",
        ),
        (
            "intentional_updates_off",
            {**intentional_base, "intentional_enabled": 0.0},
            "Matched mechanism-off control: fixed-step normalized SGD with no decay.",
        ),
    )
    for name, hyperparameters, description in intentional_arms:
        specs.append(ScreeningSpec(
            name=name,
            base_learner="upgd_w",
            mechanism="intentional_updates",
            hyperparameters=hyperparameters,
            factory=_make_intentional_updates_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=description,
        ))
    for arm, mechanism, description in (
        (
            "bounded_structure_off",
            "bounded_structure_off",
            "Mechanism-off masked SGD at the initial half-width; no growth or pruning.",
        ),
        (
            "bounded_growth",
            "bounded_growth",
            "Preallocated adaptive growth by one freshly initialized hidden-1 unit per task.",
        ),
        (
            "bounded_elastic",
            "bounded_elastic",
            "Boundary pruning of the least-active hidden-1 unit followed by fresh growth.",
        ),
    ):
        specs.append(
            ScreeningSpec(
                name=arm,
                base_learner="upgd_w",
                mechanism=mechanism,
                hyperparameters=registered_bounded_elastic_hyperparameters(arm),
                factory=_make_bounded_structure_learner,
                description=(
                    description
                    + " Bounded arXiv:2608.01475v1 adaptation; not paper/code parity."
                ),
            )
        )
    specs.append(
        ScreeningSpec(
            name="bounded_fixed_cbp",
            base_learner="upgd_w",
            mechanism="fixed_capacity_cbp",
            hyperparameters=registered_bounded_elastic_hyperparameters("bounded_fixed_cbp"),
            factory=_make_sgd_cbp_budget_learner,
            description=(
                "Fixed full-capacity SGD+CBP comparator under the same declared peak/final-size "
                "budgets as the bounded growing and elastic adaptations."
            ),
        )
    )
    # C-CHAIN is adapted from the official ICML-2025 implementation at
    # 2f8bedf: Adam plus a recent-policy cross-entropy on a prior-update reference
    # ring, adaptive relative-loss scaling, and the paper's two gradient
    # component ablations.  The online prior-example ring and all remaining
    # comparability gaps are frozen into its dedicated receipt validator.
    for (
        arm_name,
        churn_enabled,
        adaptive_coefficient,
        target_relative_loss_scale,
        gradient_component,
        arm_description,
    ) in (
        (
            "cchain_mechanism_off",
            0.0,
            0.0,
            0.0,
            0.0,
            "Matched Adam control with all C-CHAIN reference and diagnostic overhead charged.",
        ),
        (
            "cchain_full",
            1.0,
            1.0,
            10_000.0,
            0.0,
            "Full C-CHAIN churn gradient with the official target relative-loss scale.",
        ),
        (
            "cchain_orthogonal_only",
            1.0,
            1.0,
            10_000.0,
            1.0,
            "Orthogonal churn-gradient component: the paper's NTK-decorrelation ablation.",
        ),
        (
            "cchain_projective_only",
            1.0,
            1.0,
            10_000.0,
            2.0,
            "Projective churn-gradient component: the paper's step-size ablation.",
        ),
    ):
        specs.append(
            ScreeningSpec(
                name=arm_name,
                base_learner="adamw",
                mechanism="c_chain",
                hyperparameters=cchain_hyperparameters(
                    churn_enabled=churn_enabled,
                    adaptive_coefficient=adaptive_coefficient,
                    target_relative_loss_scale=target_relative_loss_scale,
                    gradient_component=gradient_component,
                ),
                factory=make_cchain_learner,
                description=(
                    arm_description
                    + " This is a permanently nonpromoting online-IPMNIST adaptation, "
                    "not a reproduction of the paper's continual-RL protocols."
                ),
            )
        )
    for name, replay_update, context, description in (
        (
            "replay_context_mechanism_off",
            0.0,
            0.0,
            "Charged Adam mechanism-off control for replay and label attention.",
        ),
        (
            "replay_gradient_only",
            1.0,
            0.0,
            "Prior-example replay-gradient ablation without contextual prediction.",
        ),
        (
            "replay_context_only",
            0.0,
            1.0,
            "Bounded label-attention prediction without replay-gradient influence.",
        ),
        (
            "replay_context_full",
            1.0,
            1.0,
            "Replay gradient plus bounded label-attention in-context proxy.",
        ),
    ):
        specs.append(
            ScreeningSpec(
                name=name,
                base_learner="adamw",
                mechanism="replay_in_context",
                hyperparameters=replay_hyperparameters(
                    replay_update=replay_update, context=context
                ),
                factory=make_replay_context_learner,
                description=(
                    description
                    + " Permanently nonpromoting; not a Transformer-paper reproduction."
                ),
            )
        )
    for name, method, mechanism_value, description in (
        (
            "randumb_random_features",
            0.0,
            1.0,
            "RanDumb-inspired random Fourier extractor and online linear head.",
        ),
        (
            "ranpac_random_projection",
            1.0,
            1.0,
            "RanPAC-inspired random ReLU projection and recursive ridge readout.",
        ),
        (
            "prol_prompt_mechanism_off",
            2.0,
            0.0,
            "Charged frozen-extractor/head control with PROL proxy prompts disabled.",
        ),
        (
            "prol_prompt_proxy",
            2.0,
            1.0,
            "PROL architecture proxy with prompt/affine and hard-soft updates.",
        ),
    ):
        specs.append(
            ScreeningSpec(
                name=name,
                base_learner="upgd_w",
                mechanism="frozen_feature_ceiling",
                hyperparameters=frozen_hyperparameters(
                    method=method, mechanism=mechanism_value
                ),
                factory=make_frozen_feature_learner,
                frozen_probe_input=_rff_frozen_probe_input,
                description=(
                    description
                    + " Zero imported pretraining is explicit; this is not a "
                    "paper-level reproduction."
                ),
            )
        )
    noise_curvature_descriptions = {
        "noise_curvature_fixed_adam_l2": (
            "Mechanism-off AdamW+L2 conditioning control; all diagnostics remain "
            "executed and charged but cannot change a layer learning rate."
        ),
        "noise_curvature_gradient_only": (
            "Gradient-noise-only Eq. 2 ablation (curvature inflation beta=0)."
        ),
        "noise_curvature_volatility_only": (
            "Curvature-volatility-only Eq. 1 inverse-bound ablation."
        ),
        "noise_curvature_combined": (
            "Joint layerwise gradient-noise and curvature-volatility Eq. 2 scheduler."
        ),
    }
    for arm_name in noise_curvature_registered_arms():
        specs.append(
            ScreeningSpec(
                name=arm_name,
                base_learner="adamw",
                mechanism="noise_curvature_scheduler",
                hyperparameters=noise_curvature_registered_hyperparameters(arm_name),
                factory=_make_noise_curvature_learner,
                description=(
                    noise_curvature_descriptions[arm_name]
                    + " Adapted from arXiv:2509.19698v3 to the current online "
                    "IPMNIST runner with a charged rolling diagnostic minibatch."
                ),
            )
        )
    return {spec.name: spec for spec in specs}


SCREENING_REGISTRY: Mapping[str, ScreeningSpec] = MappingProxyType(_build_registry())


def screening_spec(name: str) -> ScreeningSpec:
    """Look up a screening configuration by name."""
    if type(name) is not str:
        raise ValueError("screening config name must be an exact string")
    if name not in SCREENING_REGISTRY:
        raise ValueError(f"unknown screening config; expected one of {sorted(SCREENING_REGISTRY)}")
    return SCREENING_REGISTRY[name]


def _validated_screening_noise_mode(
    noise_mode: object,
    spec: ScreeningSpec,
    *,
    context: Path | str | None = None,
) -> str:
    """Validate one screening noise mode against the named arm's runner contract."""
    prefix = "" if context is None else f"{context}: "
    if type(noise_mode) is not str:
        raise ValueError(f"{prefix}noise_mode must be 'step' or 'pool'")
    if noise_mode not in ("step", "pool"):
        raise ValueError(
            f"{prefix}noise_mode must be 'step' or 'pool', got {noise_mode!r}"
        )
    if noise_mode == "pool" and spec.noise_update is None:
        raise ValueError(
            f"{prefix}noise_mode='pool' is unsupported for {spec.name!r}: the arm "
            "declares no noise-consuming update"
        )
    return noise_mode


def _validated_screening_noise_pool_steps(
    noise_mode: str,
    noise_pool_steps: object,
    *,
    context: Path | str | None = None,
    allow_unrecorded_pool: bool = False,
) -> int | None:
    """Return the effective pool-size contract for one screening run or shard.

    Exact step-mode runs do not consume a noise pool, so their canonical
    effective value is ``None``.  This also keeps pre-pool legacy shards,
    which omit both fields, readable as exact step-mode runs.  A loader may
    preserve an omitted pool size as unknown so historical files remain
    inspectable, but new results must record it and downstream merges refuse
    the unknown value.  Inferring the historical default would relabel any
    old custom-size run and recreate the ambiguity this provenance field
    closes.
    """
    prefix = "" if context is None else f"{context}: "
    if noise_mode == "step":
        if noise_pool_steps is not _MISSING_NOISE_POOL_STEPS and noise_pool_steps is not None:
            raise ValueError(
                f"{prefix}noise_pool_steps must be null or absent when "
                "noise_mode='step'"
            )
        return None
    if noise_pool_steps is _MISSING_NOISE_POOL_STEPS:
        if allow_unrecorded_pool:
            return None
        raise ValueError(
            f"{prefix}noise_pool_steps must be recorded when noise_mode='pool'"
        )
    if type(noise_pool_steps) is not int or noise_pool_steps < 2:
        raise ValueError(
            f"{prefix}noise_pool_steps must be recorded as a built-in integer >= 2 "
            "when noise_mode='pool'"
        )
    return noise_pool_steps


# =============================================================================
# Development-only recurring A/B/A retention adapter
# =============================================================================


RECURRING_IPMNIST_ADAPTER_SCHEMA = "alberta.ipmnist-screening.recurring-adapter.v1"


def _canonical_hash_array(array: object) -> np.ndarray:
    """Return a contiguous, little-endian, non-object array for hashing."""
    resolved = np.asarray(jax.device_get(array))
    if resolved.dtype.hasobject:
        raise TypeError("object arrays cannot enter a canonical SHA-256 binding")
    canonical_dtype = resolved.dtype.newbyteorder("<")
    return np.ascontiguousarray(resolved.astype(canonical_dtype, copy=False))


def _array_bundle_sha256(domain: str, arrays: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii") + b"\0")
    for name in sorted(arrays):
        encoded_name = name.encode("ascii")
        array = _canonical_hash_array(arrays[name])
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.nbytes.to_bytes(8, "little"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_capture(repo_root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repo_root), *args),
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError(
            "screening source provenance requires an available Git repository"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            "screening source provenance requires an available Git repository" + suffix
        )
    return completed.stdout


def _source_scope_status(repo_root: Path) -> bytes:
    return _git_capture(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *_SOURCE_SCOPE,
    )


def _tracked_worktree_status(repo_root: Path) -> bytes:
    return _git_capture(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=no",
    )


def _untracked_python_sources(repo_root: Path) -> bytes:
    return _git_capture(
        repo_root,
        "ls-files",
        "-z",
        "--others",
        "--exclude-standard",
        "--",
        "*.py",
        "*.pyc",
    )


def _ignored_python_sources(repo_root: Path) -> bytes:
    raw = _git_capture(
        repo_root,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "*.py",
        "*.pyc",
    )
    unsafe: list[bytes] = []
    for entry in raw.split(b"\0"):
        parts = entry.split(b"/")
        if not entry or parts[0] in {b".venv", b"outputs"} or b"__pycache__" in parts:
            continue
        unsafe.append(entry)
    return b"\0".join(unsafe)


def _tracked_index_flags(repo_root: Path) -> bytes:
    return _git_capture(repo_root, "ls-files", "-v", "-z")


def _head_source_blobs(raw_tree: bytes) -> dict[Path, str]:
    blobs: dict[Path, str] = {}
    try:
        for raw_entry in raw_tree.split(b"\0"):
            if not raw_entry:
                continue
            metadata, raw_path = raw_entry.split(b"\t", 1)
            _mode, object_type, object_id = metadata.split(b" ", 2)
            if object_type != b"blob":
                raise ValueError("non-blob source object")
            path = Path(os.fsdecode(raw_path))
            object_id_text = object_id.decode("ascii")
            if not _is_lower_hex(object_id_text, 40):
                raise ValueError("noncanonical source blob ID")
            blobs[path] = object_id_text
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("screening source provenance could not parse the HEAD tree") from exc
    return blobs


def _screening_source_provenance(repo_root: Path | None = None) -> dict[str, object]:
    """Bind a clean checkout and the actual package/lock bytes used by a run.

    The cleanliness scope intentionally excludes ``outputs/`` so sequential workers can
    append immutable shards. Any tracked change in the package/lock scope, including a
    staged change, is rejected. Untracked or ignored package files are found independently
    of Git status so an importable local module cannot evade the receipt.
    """
    requested_root = _REPO_ROOT if repo_root is None else Path(repo_root)
    try:
        root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "screening source provenance requires an available Git repository"
        ) from exc
    top_level_raw = _git_capture(root, "rev-parse", "--show-toplevel")
    try:
        top_level = Path(top_level_raw.decode("utf-8").strip()).resolve(strict=True)
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            "screening source provenance could not resolve the Git repository"
        ) from exc
    if top_level != root:
        raise RuntimeError(
            f"screening source provenance root {root} is not the Git top level {top_level}"
        )

    if _tracked_worktree_status(root):
        raise RuntimeError("screening source worktree is not clean: tracked files changed")
    if (
        _source_scope_status(root)
        or _untracked_python_sources(root)
        or _ignored_python_sources(root)
    ):
        raise RuntimeError(
            "screening source worktree is not clean: untracked Python/package source "
            "or source-scope file"
        )

    index_flags = _tracked_index_flags(root)
    if any(
        not entry.startswith(b"H ")
        for entry in index_flags.split(b"\0")
        if entry
    ):
        raise RuntimeError(
            "screening source worktree is not clean: tracked index flags can hide changes"
        )

    commit = _git_capture(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    tree = _git_capture(root, "rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    object_format = _git_capture(root, "rev-parse", "--show-object-format").decode(
        "ascii"
    ).strip()
    if object_format != "sha1" or not _is_lower_hex(commit, 40) or not _is_lower_hex(tree, 40):
        raise RuntimeError("screening source provenance requires canonical SHA-1 Git identities")

    tracked_raw = _git_capture(root, "ls-files", "-z", "--", *_SOURCE_SCOPE)
    head_tree_raw = _git_capture(root, "ls-tree", "-r", "-z", "HEAD", "--", *_SOURCE_SCOPE)
    try:
        tracked = tuple(
            sorted(
                Path(os.fsdecode(raw))
                for raw in tracked_raw.split(b"\0")
                if raw
            )
        )
    except UnicodeError as exc:
        raise RuntimeError("screening source provenance requires UTF-8 source paths") from exc
    head_blobs = _head_source_blobs(head_tree_raw)
    if set(tracked) != set(head_blobs):
        raise RuntimeError("screening source inventory does not exactly match HEAD")
    required = {
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("alberta_framework/benchmarks/ipmnist_screening.py"),
    }
    if not required.issubset(tracked):
        missing = sorted(path.as_posix() for path in required.difference(tracked))
        raise RuntimeError(
            f"screening source provenance is missing tracked source files: {missing}"
        )

    tracked_set = set(tracked)
    package_root = root / "alberta_framework"
    for candidate in package_root.rglob("*"):
        relative = candidate.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if (candidate.is_file() or candidate.is_symlink()) and relative not in tracked_set:
            raise RuntimeError(
                "screening source worktree is not clean: untracked package source "
                f"{relative.as_posix()}"
            )

    digest = hashlib.sha256()
    digest.update(b"alberta.ipmnist_screening.relevant_source.v1\0")
    uv_lock_sha256: str | None = None
    for relative in tracked:
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"invalid tracked source path {relative}")
        source_path = root / relative
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(f"tracked source is unavailable or not a regular file: {relative}")
        try:
            contents = source_path.read_bytes()
            encoded_path = relative.as_posix().encode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"could not read tracked source bytes: {relative}") from exc
        git_blob_digest = hashlib.sha1(
            b"blob " + str(len(contents)).encode("ascii") + b"\0" + contents,
            usedforsecurity=False,
        ).hexdigest()
        if git_blob_digest != head_blobs[relative]:
            raise RuntimeError(
                f"screening source worktree is not clean: {relative} differs from HEAD"
            )
        digest.update(len(encoded_path).to_bytes(4, "little"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
        if relative == Path("uv.lock"):
            uv_lock_sha256 = hashlib.sha256(contents).hexdigest()

    if uv_lock_sha256 is None:
        raise RuntimeError("screening source provenance could not bind uv.lock")
    final_identity = (
        _git_capture(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip(),
        _git_capture(root, "rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip(),
        _git_capture(root, "rev-parse", "--show-object-format").decode("ascii").strip(),
    )
    final_inventory = _git_capture(root, "ls-files", "-z", "--", *_SOURCE_SCOPE)
    final_head_tree = _git_capture(
        root, "ls-tree", "-r", "-z", "HEAD", "--", *_SOURCE_SCOPE
    )
    if final_identity != (commit, tree, object_format):
        raise RuntimeError("screening source identity changed while provenance was captured")
    if final_inventory != tracked_raw or final_head_tree != head_tree_raw:
        raise RuntimeError("screening source inventory changed while provenance was captured")
    if (
        _tracked_worktree_status(root)
        or _source_scope_status(root)
        or _untracked_python_sources(root)
        or _ignored_python_sources(root)
        or _tracked_index_flags(root) != index_flags
    ):
        raise RuntimeError("screening source files changed while provenance was captured")
    return {
        "schema": SOURCE_PROVENANCE_SCHEMA,
        "git_commit": commit,
        "git_tree": tree,
        "git_object_format": object_format,
        "relevant_source_scope": _SOURCE_SCOPE_LABEL,
        "relevant_source_file_count": len(tracked),
        "relevant_source_sha256": digest.hexdigest(),
        "uv_lock_sha256": uv_lock_sha256,
        "worktree_clean": True,
    }


def _screening_runtime_environment() -> dict[str, object]:
    """Return the runtime contract that must stay fixed through a screening run."""
    try:
        package_versions = {
            name: importlib.metadata.version(name)
            for name in ("chex", "jax", "jaxlib", "numpy", "scikit-learn")
        }
        backend = jax.default_backend()
        devices = [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
            for device in jax.devices()
        ]
    except Exception as exc:
        raise RuntimeError("screening runtime provenance is unavailable") from exc
    if not devices:
        raise RuntimeError("screening runtime provenance found no JAX devices")
    return {
        "schema": RUNTIME_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": package_versions,
        "jax": {
            "backend": backend,
            "devices": devices,
            "config": {
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "jax_default_matmul_precision": (
                    None
                    if jax.config.jax_default_matmul_precision is None
                    else str(jax.config.jax_default_matmul_precision)
                ),
                "jax_disable_jit": bool(jax.config.jax_disable_jit),
                "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
                "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
                "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
                "jax_threefry_partitionable": bool(
                    jax.config.jax_threefry_partitionable
                ),
                "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
            },
        },
        "process_environment": {
            name: os.environ.get(name) for name in _RUNTIME_ENVIRONMENT_KEYS
        },
    }


def _materialized_dataset_provenance(
    data_x: object, data_y: object
) -> dict[str, object]:
    """Hash the exact effective float32 features and int32 labels consumed by JAX."""
    raw_x = np.asarray(jax.device_get(data_x))
    raw_y = np.asarray(jax.device_get(data_y))
    if raw_x.dtype.kind not in {"f", "i", "u"} or raw_x.ndim != 2 or raw_x.size == 0:
        raise ValueError("dataset features must be a non-empty rank-two numeric array")
    if raw_y.dtype.kind not in {"i", "u"} or raw_y.ndim != 1:
        raise ValueError("dataset labels must be a rank-one integer array")
    if raw_x.shape[0] != raw_y.shape[0] or raw_y.size == 0:
        raise ValueError("dataset feature and label rows must be non-empty and aligned")
    int32_info = np.iinfo(np.int32)
    if np.any(raw_y < int32_info.min) or np.any(raw_y > int32_info.max):
        raise ValueError("dataset labels must be representable as int32")
    effective_x = _canonical_hash_array(np.asarray(raw_x, dtype=np.float32))
    effective_y = _canonical_hash_array(np.asarray(raw_y, dtype=np.int32))
    if not np.all(np.isfinite(effective_x)):
        raise ValueError("dataset features must remain finite after float32 materialization")
    return {
        "schema": DATASET_PROVENANCE_SCHEMA,
        "source": dict(_DATASET_SOURCE),
        "materialization": _DATASET_MATERIALIZATION,
        "x": {
            "dtype": effective_x.dtype.str,
            "shape": list(effective_x.shape),
            "sha256": _array_bundle_sha256(
                "alberta.ipmnist_screening.materialized_x.v1", {"x": effective_x}
            ),
        },
        "y": {
            "dtype": effective_y.dtype.str,
            "shape": list(effective_y.shape),
            "sha256": _array_bundle_sha256(
                "alberta.ipmnist_screening.materialized_y.v1", {"y": effective_y}
            ),
        },
    }


def _screening_dataset_provenance(
    data_x: object, data_y: object
) -> dict[str, object]:
    """Bind and validate the frozen OpenML MNIST training materialization."""
    provenance = _materialized_dataset_provenance(data_x, data_y)
    x = np.asarray(jax.device_get(data_x), dtype=np.float32)
    y = np.asarray(jax.device_get(data_y), dtype=np.int32)
    if x.shape != (60_000, 784) or y.shape != (60_000,):
        raise ValueError(
            "screening dataset must materialize OpenML mnist_784 v1 rows 0:60000 "
            "as x=(60000, 784) and y=(60000,)"
        )
    if np.any(x < -1.0) or np.any(x > 1.0):
        raise ValueError("screening dataset features must use the frozen [-1, 1] scaling")
    if np.any(y < 0) or np.any(y > 9):
        raise ValueError("screening dataset labels must be in [0, 9]")
    return provenance


def _validated_permutation(permutation: object, *, input_dim: int) -> np.ndarray:
    raw = np.asarray(jax.device_get(permutation))
    if raw.dtype.kind not in {"i", "u"} or raw.ndim != 1:
        raise ValueError("each permutation must be a one-dimensional integer array")
    if raw.shape != (input_dim,):
        raise ValueError(f"each permutation must have shape ({input_dim},)")
    resolved = np.asarray(raw, dtype=np.int64)
    if not np.array_equal(np.sort(resolved), np.arange(input_dim, dtype=np.int64)):
        raise ValueError("each permutation must contain every input index exactly once")
    return np.asarray(resolved, dtype=np.int32)


def _validated_sentinel_indices(
    sentinel_indices: Sequence[int] | np.ndarray | Array, *, n_examples: int
) -> np.ndarray:
    raw = np.asarray(jax.device_get(sentinel_indices))
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise TypeError("sentinel_indices must be a one-dimensional integer sequence")
    if raw.size == 0:
        raise ValueError("sentinel_indices must be non-empty")
    if np.any(raw < 0) or np.any(raw >= n_examples):
        raise ValueError("sentinel_indices must be in range for the supplied data")
    indices = np.asarray(raw, dtype=np.int64)
    if len(set(int(index) for index in indices)) != len(indices):
        raise ValueError("sentinel_indices must be unique and ordered explicitly")
    return indices


def _validated_recurring_phase_lengths(
    phase_lengths: Sequence[int],
) -> tuple[int, int, int]:
    lengths = tuple(phase_lengths)
    if len(lengths) != 3 or any(
        not isinstance(length, int) or isinstance(length, bool) or length <= 0
        for length in lengths
    ):
        raise ValueError("phase_lengths must contain exactly three positive integers")
    resolved = (int(lengths[0]), int(lengths[1]), int(lengths[2]))
    if resolved[0] != resolved[2]:
        raise ValueError("the two A phase lengths must be equal")
    return resolved


def _validated_recurring_seed(seed: int) -> int:
    return require_jax_seed(seed, name="seed")


def build_recurring_ipmnist_online_indices(
    *,
    seed: int,
    n_examples: int,
    phase_lengths: Sequence[int],
    sentinel_indices: Sequence[int] | np.ndarray | Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the evaluator-owned held-out A/B/A online example schedule.

    The first and recurring A exposures use the exact same ordered example
    indices (common random numbers); B uses an independent seed fold.  Every
    sentinel row is removed before any phase permutation is drawn.  This
    helper is intentionally public so those two properties can be checked
    without executing a learner.
    """
    resolved_seed = _validated_recurring_seed(seed)
    if (
        not isinstance(n_examples, int)
        or isinstance(n_examples, bool)
        or n_examples <= 0
    ):
        raise ValueError("n_examples must be a positive integer")
    lengths = _validated_recurring_phase_lengths(phase_lengths)
    indices = _validated_sentinel_indices(sentinel_indices, n_examples=n_examples)
    eligible_mask = np.ones(n_examples, dtype=np.bool_)
    eligible_mask[indices] = False
    eligible = np.flatnonzero(eligible_mask).astype(np.int32)
    if any(length > len(eligible) for length in lengths):
        raise ValueError(
            "each phase length must fit a without-replacement draw after holding out sentinels"
        )

    root = jr.key(jnp.uint32(resolved_seed))
    _, key_schedule, _ = jr.split(root, 3)
    _, key_sample = jr.split(key_schedule)

    def phase_order(fold_index: int, length: int) -> np.ndarray:
        offsets = np.asarray(
            jr.permutation(jr.fold_in(key_sample, fold_index), len(eligible))[:length]
        )
        return np.asarray(eligible[offsets], dtype=np.int32)

    a_order = phase_order(0, lengths[0])
    b_order = phase_order(1, lengths[1])
    return a_order, b_order, a_order.copy()


def _validated_ipmnist_data(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    *,
    input_dim: int | None,
    n_classes: int | None,
    min_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=input_dim,
        n_classes=n_classes,
        min_length=min_length,
    )


def ipmnist_permutation_sha256(permutation: np.ndarray | Array) -> str:
    """Bind one complete integer pixel permutation deterministically."""
    raw = np.asarray(jax.device_get(permutation))
    if raw.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    resolved = _validated_permutation(permutation, input_dim=int(raw.shape[0]))
    return _array_bundle_sha256(
        "alberta.ipmnist-screening.pixel-permutation.v1",
        {"permutation": resolved},
    )


def ipmnist_sentinel_set_sha256(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    permutation: np.ndarray | Array,
    sentinel_indices: Sequence[int] | np.ndarray | Array,
) -> str:
    """Bind ordered sentinel identities, labels, and pixel-permuted inputs.

    The digest covers the exact float32 source rows as well as the transformed
    rows.  Adaptive learner preprocessing (for example EMA normalization) is
    derived from the bound transformed rows and the separately hashed frozen
    learner state at each checkpoint.
    """
    resolved_x, resolved_y = _validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=None,
        n_classes=None,
    )
    resolved_permutation = _validated_permutation(
        permutation, input_dim=int(resolved_x.shape[1])
    )
    indices = _validated_sentinel_indices(
        sentinel_indices, n_examples=int(resolved_x.shape[0])
    )
    raw_examples = resolved_x[indices]
    return _array_bundle_sha256(
        "alberta.ipmnist-screening.ordered-sentinel-set.v1",
        {
            "example_indices": indices,
            "labels": resolved_y[indices],
            "permutation": resolved_permutation,
            "pixel_permuted_inputs": raw_examples[:, resolved_permutation],
            "raw_examples": raw_examples,
        },
    )


def _declared_learner_state_sha256(
    params: dict[str, Array], state: Any, learner_key: Array
) -> str:
    """Hash parameters, optimizer/mechanism state, and the next learner RNG key."""
    bundle = {
        "learner_key": jr.key_data(learner_key),
        "optimizer_and_mechanism_state": state,
        "params": params,
    }
    path_leaves, tree = jax.tree_util.tree_flatten_with_path(bundle)
    digest = hashlib.sha256()
    digest.update(b"alberta.ipmnist-screening.full-learner-state.v1\0")
    tree_bytes = str(tree).encode("utf-8")
    digest.update(len(tree_bytes).to_bytes(8, "little"))
    digest.update(tree_bytes)
    for path, leaf in path_leaves:
        path_bytes = repr(path).encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        array = _canonical_hash_array(leaf)
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _recurring_protocol_id(
    *,
    spec: ScreeningSpec,
    seed: int,
    config: IPMNISTConfig,
    phase_lengths: tuple[int, int, int],
    permutation_sha256: tuple[str, str, str],
    sentinel_indices_sha256: str,
    online_indices_sha256: tuple[str, str, str],
    relearning_window: int,
) -> str:
    manifest = {
        "schema": RECURRING_IPMNIST_ADAPTER_SCHEMA,
        "development_only": True,
        "config_name": spec.name,
        "base_learner": spec.base_learner,
        "hyperparameters": dict(spec.hyperparameters),
        "seed": seed,
        "config": config.to_config(),
        "phase_lengths": list(phase_lengths),
        "permutation_sha256": list(permutation_sha256),
        "sentinel_indices_sha256": sentinel_indices_sha256,
        "online_indices_sha256": list(online_indices_sha256),
        "relearning_window": relearning_window,
    }
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"ipmnist-screening-aba-{hashlib.sha256(encoded).hexdigest()}.v1"


def run_recurring_ipmnist_retention_development(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    spec: ScreeningSpec,
    *,
    seed: int,
    config: IPMNISTConfig,
    phase_lengths: Sequence[int],
    permutations: Sequence[np.ndarray | Array],
    sentinel_indices: Sequence[int] | np.ndarray | Array,
    relearning_window: int,
) -> RecurringIPMNISTRetentionReport:
    """Run an explicit, threshold-free A/B/A retention diagnostic.

    This adapter reuses the screening arm's initialization, online update,
    input preprocessing, and same-example plasticity equations.  The learner
    receives only ``(x, y, rng_key)`` in a continuous key chain: phase,
    permutation, exposure, and sentinel identities remain evaluator-only.

    ``spec`` must be the exact object returned by :func:`screening_spec` for
    its name.  Cloned or custom specs are rejected even when their visible
    fields match: otherwise a substituted factory or stateful probe callback
    could silently change the semantics committed by the protocol identity.

    Every argument governing the recurrence is mandatory.  Sentinel rows are
    held out from the online examples, and every phase samples the remaining
    rows without replacement using the caller's seed.  There is deliberately
    no default protocol, threshold, artifact writer, or evidence path.

    The evaluator report schema stores the SHA-256 commitment to the adapter
    manifest in ``protocol_id``, not the manifest preimage.  Callers must keep
    these explicit arguments with the in-memory development report if they
    need standalone reconstruction; this function does not create artifacts.
    """
    if not isinstance(spec, ScreeningSpec):
        raise TypeError("spec must be a ScreeningSpec")
    if SCREENING_REGISTRY.get(spec.name) is not spec:
        raise ValueError(
            "spec must be the exact registered object returned by screening_spec(name)"
        )
    if not isinstance(config, IPMNISTConfig):
        raise TypeError("config must be an IPMNISTConfig")
    resolved_seed = _validated_recurring_seed(seed)
    if (
        not isinstance(relearning_window, int)
        or isinstance(relearning_window, bool)
        or relearning_window <= 0
    ):
        raise ValueError("relearning_window must be a positive integer")

    typed_lengths = _validated_recurring_phase_lengths(phase_lengths)
    if config.n_tasks != 3 or typed_lengths[0] != config.task_length:
        raise ValueError(
            "config must describe three phases and bind task_length to both A exposures"
        )
    if relearning_window > typed_lengths[0]:
        raise ValueError("relearning_window cannot exceed an A phase length")

    resolved_x, resolved_y = _validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
    )
    raw_permutations = tuple(permutations)
    if len(raw_permutations) != 3:
        raise ValueError("permutations must contain the exact A/B/A phase tuple")
    resolved_permutations = tuple(
        _validated_permutation(permutation, input_dim=config.input_dim)
        for permutation in raw_permutations
    )
    if not np.array_equal(resolved_permutations[0], resolved_permutations[2]):
        raise ValueError("the first and third phase permutations must be exactly identical")
    if np.array_equal(resolved_permutations[0], resolved_permutations[1]):
        raise ValueError("the B permutation must be distinct from A")

    indices = _validated_sentinel_indices(
        sentinel_indices, n_examples=int(resolved_x.shape[0])
    )
    online_indices = build_recurring_ipmnist_online_indices(
        seed=resolved_seed,
        n_examples=int(resolved_x.shape[0]),
        phase_lengths=typed_lengths,
        sentinel_indices=indices,
    )

    root = jr.key(jnp.uint32(resolved_seed))
    key_init, _, key_noise = jr.split(root, 3)

    permutation_hashes = (
        ipmnist_permutation_sha256(resolved_permutations[0]),
        ipmnist_permutation_sha256(resolved_permutations[1]),
        ipmnist_permutation_sha256(resolved_permutations[2]),
    )
    sentinel_hashes = tuple(
        ipmnist_sentinel_set_sha256(resolved_x, resolved_y, permutation, indices)
        for permutation in resolved_permutations[:2]
    )
    online_hashes = tuple(
        _array_bundle_sha256(
            "alberta.ipmnist-screening.online-example-order.v1",
            {"example_indices": phase_indices},
        )
        for phase_indices in online_indices
    )
    typed_online_hashes = (online_hashes[0], online_hashes[1], online_hashes[2])
    sentinel_indices_hash = _array_bundle_sha256(
        "alberta.ipmnist-screening.sentinel-index-order.v1",
        {"sentinel_indices": indices},
    )
    protocol_id = _recurring_protocol_id(
        spec=spec,
        seed=resolved_seed,
        config=config,
        phase_lengths=typed_lengths,
        permutation_sha256=permutation_hashes,
        sentinel_indices_sha256=sentinel_indices_hash,
        online_indices_sha256=typed_online_hashes,
        relearning_window=relearning_window,
    )
    permutation_ids = (
        f"ipmnist-permutation-{permutation_hashes[0]}.v1",
        f"ipmnist-permutation-{permutation_hashes[1]}.v1",
    )
    sentinel_ids = (
        f"ipmnist-sentinel-{sentinel_hashes[0]}.v1",
        f"ipmnist-sentinel-{sentinel_hashes[1]}.v1",
    )
    phase_permutation_ids = (
        permutation_ids[0],
        permutation_ids[1],
        permutation_ids[0],
    )
    starts = (0, typed_lengths[0], typed_lengths[0] + typed_lengths[1])
    protocol = RecurringIPMNISTProtocol(
        protocol_id=protocol_id,
        phases=tuple(
            RecurringIPMNISTPhase(
                phase_index=index,
                start_step=starts[index],
                length=typed_lengths[index],
                permutation_id=phase_permutation_ids[index],
                exposure_index=0 if index < 2 else 1,
            )
            for index in range(3)
        ),
        sentinel_bindings=tuple(
            SentinelProbeBinding(
                permutation_id=permutation_ids[index],
                permutation_sha256=permutation_hashes[index],
                sentinel_set_id=sentinel_ids[index],
                sentinel_set_sha256=sentinel_hashes[index],
                sentinel_case_count=len(indices),
            )
            for index in range(2)
        ),
        relearning_window=relearning_window,
    )

    data_x_array = jnp.asarray(resolved_x, dtype=jnp.float32)
    data_y_array = jnp.asarray(resolved_y, dtype=jnp.int32)
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    params = init_mlp_params(key_init, config)
    state = init_fn(params)

    def run_phase(
        phase_params: dict[str, Array],
        phase_state: Any,
        learner_key: Array,
        permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array]:
        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], tuple[Array, Array]]:
            step_params, step_state, next_key = carry
            x = data_x_array[example][permutation]
            y = data_y_array[example]
            next_key, step_key = jr.split(next_key)
            new_params, new_state, metrics = step_fn(
                step_params, step_state, x, y, step_key
            )
            accuracy, _, plasticity = metrics
            return (new_params, new_state, next_key), (accuracy, plasticity)

        (new_params, new_state, new_key), (accuracies, plasticities) = jax.lax.scan(
            one_step,
            (phase_params, phase_state, learner_key),
            examples,
        )
        return new_params, new_state, new_key, accuracies, plasticities

    run_phase_jit = jax.jit(run_phase)
    accuracy_trace: list[float] = []
    plasticity_trace: list[float] = []
    snapshots: list[SentinelProbeSnapshot] = []
    permutation_by_id = {
        permutation_ids[0]: resolved_permutations[0],
        permutation_ids[1]: resolved_permutations[1],
    }
    sentinel_labels = resolved_y[indices]

    for phase_index in range(3):
        params, state, key_noise, accuracies, plasticities = run_phase_jit(
            params,
            state,
            key_noise,
            jnp.asarray(resolved_permutations[phase_index], dtype=jnp.int32),
            jnp.asarray(online_indices[phase_index], dtype=jnp.int32),
        )
        accuracy_trace.extend(
            float(value) for value in np.asarray(jax.device_get(accuracies)).reshape(-1)
        )
        plasticity_trace.extend(
            float(value) for value in np.asarray(jax.device_get(plasticities)).reshape(-1)
        )

        requirements = tuple(
            requirement
            for requirement in protocol.required_probe_snapshots
            if requirement.phase_index == phase_index
        )
        for requirement in requirements:
            state_hash_before = _declared_learner_state_sha256(params, state, key_noise)
            permutation = permutation_by_id[requirement.permutation_id]
            sentinel_inputs = jnp.asarray(
                resolved_x[indices][:, permutation], dtype=jnp.float32
            )
            model_inputs = spec.frozen_probe_input(
                state, sentinel_inputs, spec.hyperparameters
            )
            if model_inputs.shape != sentinel_inputs.shape:
                raise ValueError("frozen_probe_input must preserve sentinel input shape")
            logits = np.asarray(jax.device_get(mlp_logits(params, model_inputs)))
            if not np.all(np.isfinite(logits)):
                raise ValueError("a frozen sentinel probe produced non-finite logits")
            correctness = tuple(
                bool(value)
                for value in np.asarray(np.argmax(logits, axis=-1) == sentinel_labels)
            )
            state_hash_after = _declared_learner_state_sha256(params, state, key_noise)
            snapshots.append(
                SentinelProbeSnapshot.from_requirement(
                    requirement,
                    learner_state_sha256_before=state_hash_before,
                    learner_state_sha256_after=state_hash_after,
                    correctness=correctness,
                )
            )

    trace = RecurringIPMNISTTrace(
        pre_update_online_accuracy=tuple(accuracy_trace),
        post_update_one_step_plasticity=tuple(plasticity_trace),
    )
    return build_recurring_ipmnist_retention_report(
        protocol=protocol,
        trace=trace,
        sentinel_snapshots=tuple(snapshots),
    )


# =============================================================================
# Runner (single seed, one process per seed)
# =============================================================================


@dataclass(frozen=True)
class ScreeningRunResult:
    """Host-side per-task results of one (config, seed) screening run."""

    config_name: str
    base_learner: str
    hyperparameters: dict[str, float]
    seed: int
    config: IPMNISTConfig
    per_task_accuracy: np.ndarray
    per_task_loss: np.ndarray
    per_task_plasticity: np.ndarray
    wall_clock_seconds: float
    noise_mode: str = "step"
    noise_pool_steps: int | None = None
    mechanism_diagnostics: dict[str, float] | None = None

    def __post_init__(self) -> None:
        for attr in ("config_name", "base_learner", "noise_mode"):
            val = getattr(self, attr)
            if type(val) is not str or not val:
                raise ValueError(f"{attr} must be a non-empty string")
        if type(self.base_learner) is not str or self.base_learner not in ("upgd_w", "adamw"):
            raise ValueError("base_learner must name one supported screening learner")
        if type(self.noise_mode) is not str or self.noise_mode not in ("step", "pool"):
            raise ValueError("noise_mode must be 'step' or 'pool'")
        if type(self.hyperparameters) is not dict:
            raise TypeError("hyperparameters must be a dict")
        normalized: dict[str, float] = {}
        for key, value in self.hyperparameters.items():
            if type(key) is not str or not key:
                raise TypeError("hyperparameter keys must be exact non-empty strings")
            if (type(value) is not int and type(value) is not float) or not math.isfinite(value):
                raise ValueError("hyperparameter values must be finite built-in numbers")
            normalized[key] = float(value)
        object.__setattr__(self, "hyperparameters", normalized)
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        if type(self.config) is not IPMNISTConfig:
            raise TypeError("config must be an IPMNISTConfig")
        if (
            (
                type(self.wall_clock_seconds) is not int
                and type(self.wall_clock_seconds) is not float
            )
            or not math.isfinite(self.wall_clock_seconds)
            or self.wall_clock_seconds < 0.0
        ):
            raise ValueError("wall_clock_seconds must be a finite float")
        for arr_name in ("per_task_accuracy", "per_task_loss", "per_task_plasticity"):
            array = getattr(self, arr_name)
            if type(array) is not np.ndarray:
                raise TypeError(f"{arr_name} must be a numpy ndarray")
            if array.dtype != np.dtype(np.float64) or array.shape != (self.config.n_tasks,):
                raise ValueError(f"{arr_name} must be one float64 value per task")
            if not np.isfinite(array).all():
                raise ValueError(f"{arr_name} must contain only finite values")
        if np.any((self.per_task_accuracy < 0.0) | (self.per_task_accuracy > 1.0)):
            raise ValueError("per_task_accuracy must lie in [0, 1]")
        if np.any(self.per_task_loss < 0.0):
            raise ValueError("per_task_loss must be non-negative")
        if np.any((self.per_task_plasticity < 0.0) | (self.per_task_plasticity > 1.0)):
            raise ValueError("per_task_plasticity must lie in [0, 1]")
        object.__setattr__(
            self,
            "noise_pool_steps",
            _validated_screening_noise_pool_steps(self.noise_mode, self.noise_pool_steps),
        )
        diagnostics = self.mechanism_diagnostics
        if diagnostics is not None:
            if type(diagnostics) is not dict or not diagnostics:
                raise ValueError("mechanism_diagnostics must be a non-empty exact dict or None")
            normalized_diagnostics: dict[str, float] = {}
            for name, value in diagnostics.items():
                if type(name) is not str or not name or "\x00" in name:
                    raise ValueError("mechanism diagnostic names must be non-empty strings")
                if type(value) is not float or not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "mechanism diagnostic values must be finite nonnegative floats"
                    )
                normalized_diagnostics[name] = value
            object.__setattr__(self, "mechanism_diagnostics", normalized_diagnostics)


def l2er_development_result_payload(
    result: ScreeningRunResult, *, outcome: str
) -> dict[str, object]:
    """Build and strictly validate one nonpromoting L2-ER result receipt."""
    spec = screening_spec(result.config_name)
    if spec.mechanism != "l2_effective_rank" or result.noise_mode != "step":
        raise ValueError("an L2-ER receipt requires an exact-step registered L2-ER arm")
    if result.hyperparameters != spec.hyperparameters:
        raise ValueError("result hyperparameters drift from the registered L2-ER arm")
    config = result.config
    observations = config.n_tasks * config.task_length
    er_enabled = spec.hyperparameters["er_enabled"] == 1.0
    er_batch_size = int(spec.hyperparameters["er_batch_size"])
    er_updates = observations // er_batch_size if er_enabled else 0
    parameter_count = (
        config.input_dim * config.hidden1
        + config.hidden1
        + config.hidden1 * config.hidden2
        + config.hidden2
        + config.hidden2 * config.n_classes
        + config.n_classes
    )
    persistent_bytes = 4 * (parameter_count + er_batch_size * config.input_dim + 1) + 1
    payload: dict[str, object] = {
        "schema": L2ER_RESULT_SCHEMA,
        "comparison_id": L2ER_COMPARISON_ID,
        "paper_revision": L2ER_PAPER_REVISION,
        "official_commit": L2ER_OFFICIAL_COMMIT,
        "arm": result.config_name,
        "seed": result.seed,
        "n_tasks": config.n_tasks,
        "task_length": config.task_length,
        "input_dim": config.input_dim,
        "hidden1": config.hidden1,
        "hidden2": config.hidden2,
        "n_classes": config.n_classes,
        "observations": observations,
        "supervised_updates": observations,
        "effective_rank_updates": er_updates,
        "total_optimizer_updates": observations + er_updates,
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "hyperparameters": dict(spec.hyperparameters),
        "metrics": {
            "mean_online_accuracy": float(np.mean(result.per_task_accuracy)),
            "mean_loss": float(np.mean(result.per_task_loss)),
            "mean_plasticity": float(np.mean(result.per_task_plasticity)),
        },
        "resources": {
            "persistent_bytes": persistent_bytes,
            "environment_steps": 0,
            "data_steps": observations,
            "model_queries": 2 * observations + er_updates,
            "timing_seconds": float(result.wall_clock_seconds),
            "timing_is_telemetry_only": True,
        },
        "outcome": outcome,
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    return validate_l2er_development_result(payload)


def _partial_reset_peak_numeric_bytes(config: IPMNISTConfig) -> int:
    parameter_bytes = config.parameter_count * np.dtype(np.float32).itemsize
    normalizer_bytes = (2 * config.input_dim + 1) * np.dtype(np.float32).itemsize
    # Live params + retained init + utility EMA + int32 step + normalizer.
    return 3 * parameter_bytes + normalizer_bytes + np.dtype(np.int32).itemsize


def partial_reset_development_record(result: ScreeningRunResult) -> dict[str, Any]:
    """Create a strict in-memory receipt for one real CPR-family run."""
    if type(result) is not ScreeningRunResult:
        raise TypeError("result must be an exact ScreeningRunResult")
    spec = SCREENING_REGISTRY.get(result.config_name)
    if spec is None or spec.mechanism != "calibrated_partial_reset":
        raise ValueError("result must use a registered calibrated partial-reset arm")
    if result.hyperparameters != spec.hyperparameters:
        raise ValueError("result hyperparameters must match the registered arm")
    steps = result.config.n_steps
    peak_bytes = _partial_reset_peak_numeric_bytes(result.config)
    return {
        "schema": PARTIAL_RESET_RECORD_SCHEMA,
        "references": {
            "paper": CPR_PAPER_REVISION,
            "official_code": CPR_OFFICIAL_CODE_REVISION,
            "protocol_difference": (
                "batch-size-one IPMNIST; per-parameter rather than per-neuron gradient "
                "utility; retained initialization rather than fresh keyed draws; pulls all "
                "parameters rather than hidden incoming weights plus outgoing decay; no "
                "task-boundary information"
            ),
        },
        "arm": result.config_name,
        "seed": result.seed,
        "config": result.config.to_config(),
        "hyperparameters": dict(result.hyperparameters),
        "matched_axes": [
            "seed",
            "example_schedule",
            "observations",
            "updates",
            "allowed_boundary_information:none",
            "peak_state_envelope",
        ],
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "publication_equivalent": False,
            "retain_negative_outcome": True,
        },
        "resources": {
            "persistent_bytes": peak_bytes,
            "peak_numeric_bytes": peak_bytes,
            "environment_or_data_steps": steps,
            "observations": steps,
            "updates": steps,
            "model_queries": 2 * steps,
            "timing_telemetry_seconds": float(result.wall_clock_seconds),
            "timing_is_selection_metric": False,
        },
        "metrics": {
            "per_task_accuracy": result.per_task_accuracy.tolist(),
            "per_task_loss": result.per_task_loss.tolist(),
            "per_task_plasticity": result.per_task_plasticity.tolist(),
        },
    }


def _partial_reset_exact_object(
    value: object, *, keys: frozenset[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        raise ValueError(f"{context} must be an exact object with the frozen keys")
    return cast(dict[str, Any], value)


def _partial_reset_curve(value: object, *, length: int, context: str) -> list[float]:
    if (
        type(value) is not list
        or len(value) != length
        or any(type(item) is not float or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{context} must be a bounded finite float list")
    return cast(list[float], value)


def validate_partial_reset_development_record(record: object) -> dict[str, Any]:
    """Fail closed over CPR-family identity, counters, metrics, and policy."""
    payload = _partial_reset_exact_object(
        record,
        keys=frozenset(
            {
                "schema",
                "references",
                "arm",
                "seed",
                "config",
                "hyperparameters",
                "matched_axes",
                "policy",
                "resources",
                "metrics",
            }
        ),
        context="partial-reset record",
    )
    policy = _partial_reset_exact_object(
        payload["policy"],
        keys=frozenset(
            {
                "development_only",
                "scientific_promotion_allowed",
                "publication_equivalent",
                "retain_negative_outcome",
            }
        ),
        context="policy",
    )
    if policy != {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "publication_equivalent": False,
        "retain_negative_outcome": True,
    }:
        raise ValueError("partial-reset records are permanently nonpromoting")
    try:
        config_raw = _partial_reset_exact_object(
            payload["config"],
            keys=frozenset(
                {
                    "n_tasks",
                    "task_length",
                    "input_dim",
                    "hidden1",
                    "hidden2",
                    "n_classes",
                }
            ),
            context="config",
        )
        config = IPMNISTConfig(**config_raw)
        metrics = _partial_reset_exact_object(
            payload["metrics"],
            keys=frozenset(
                {"per_task_accuracy", "per_task_loss", "per_task_plasticity"}
            ),
            context="metrics",
        )
        resources = _partial_reset_exact_object(
            payload["resources"],
            keys=frozenset(
                {
                    "persistent_bytes",
                    "peak_numeric_bytes",
                    "environment_or_data_steps",
                    "observations",
                    "updates",
                    "model_queries",
                    "timing_telemetry_seconds",
                    "timing_is_selection_metric",
                }
            ),
            context="resources",
        )
        arm = payload["arm"]
        if type(arm) is not str:
            raise ValueError("arm must be an exact string")
        spec = screening_spec(arm)
        hyperparameters = _partial_reset_exact_object(
            payload["hyperparameters"],
            keys=frozenset(spec.hyperparameters),
            context="hyperparameters",
        )
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in hyperparameters.values()
        ):
            raise ValueError("hyperparameters must contain exact finite floats")
        references = _partial_reset_exact_object(
            payload["references"],
            keys=frozenset({"paper", "official_code", "protocol_difference"}),
            context="references",
        )
        if any(type(value) is not str for value in references.values()):
            raise ValueError("references must contain exact strings")
        matched_axes = payload["matched_axes"]
        if type(matched_axes) is not list or any(
            type(value) is not str for value in matched_axes
        ):
            raise ValueError("matched_axes must be an exact string list")
        timing = resources["timing_telemetry_seconds"]
        if type(timing) is not float or not math.isfinite(timing) or timing < 0.0:
            raise ValueError("timing telemetry must be one finite nonnegative float")
        result = ScreeningRunResult(
            config_name=arm,
            base_learner=spec.base_learner,
            hyperparameters=hyperparameters,
            seed=require_jax_seed(payload["seed"], name="record seed"),
            config=config,
            per_task_accuracy=np.asarray(
                _partial_reset_curve(
                    metrics["per_task_accuracy"],
                    length=config.n_tasks,
                    context="per_task_accuracy",
                ),
                dtype=np.float64,
            ),
            per_task_loss=np.asarray(
                _partial_reset_curve(
                    metrics["per_task_loss"],
                    length=config.n_tasks,
                    context="per_task_loss",
                ),
                dtype=np.float64,
            ),
            per_task_plasticity=np.asarray(
                _partial_reset_curve(
                    metrics["per_task_plasticity"],
                    length=config.n_tasks,
                    context="per_task_plasticity",
                ),
                dtype=np.float64,
            ),
            wall_clock_seconds=timing,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid partial-reset result fields") from error
    expected = partial_reset_development_record(result)
    if payload != expected:
        if payload.get("resources") != expected["resources"]:
            raise ValueError("partial-reset resource receipt does not match the run")
        raise ValueError("partial-reset record does not match the frozen protocol")
    return expected


def _intentional_updates_persistent_numeric_bytes(
    config: IPMNISTConfig, *, mechanism_enabled: bool
) -> int:
    """Exact params + learner-state numeric payload, excluding runtime buffers."""
    parameter_bytes = config.parameter_count * np.dtype(np.float32).itemsize
    normalizer_bytes = (2 * config.input_dim + 1) * np.dtype(np.float32).itemsize
    if not mechanism_enabled:
        return parameter_bytes + normalizer_bytes
    # One RMS bank plus int32 step, float32 clip EMA, and int32 clip step.
    mechanism_bytes = parameter_bytes + 3 * np.dtype(np.float32).itemsize
    return parameter_bytes + normalizer_bytes + mechanism_bytes


_INTENTIONAL_MAX_RECORD_TASKS = 1_000_000
_INTENTIONAL_MAX_PERSISTENT_BYTES = 256 * 1024 * 1024


def _canonical_intentional_screening_result(result: object) -> ScreeningRunResult:
    """Re-run host dataclass gates after possible frozen-instance mutation."""
    if type(result) is not ScreeningRunResult:
        raise TypeError("result must be an exact ScreeningRunResult")
    if type(result.config) is not IPMNISTConfig:
        raise TypeError("result.config must be an exact IPMNISTConfig")
    config = IPMNISTConfig(**result.config.to_config())
    return ScreeningRunResult(
        config_name=result.config_name,
        base_learner=result.base_learner,
        hyperparameters=result.hyperparameters,
        seed=result.seed,
        config=config,
        per_task_accuracy=result.per_task_accuracy,
        per_task_loss=result.per_task_loss,
        per_task_plasticity=result.per_task_plasticity,
        wall_clock_seconds=result.wall_clock_seconds,
        noise_mode=result.noise_mode,
        noise_pool_steps=result.noise_pool_steps,
    )


def intentional_updates_development_record(
    result: ScreeningRunResult,
) -> dict[str, Any]:
    """Project one real screening run into a strict, nonpromoting record.

    The existing screening shard remains the canonical campaign artifact.
    This in-memory projection adds the mechanism-specific gates and exact
    resource counters requested by issue #1561 without creating or mutating
    anything under ``outputs/``.
    """
    result = _canonical_intentional_screening_result(result)
    if result.noise_mode != "step" or result.noise_pool_steps is not None:
        raise ValueError("Intentional Updates records require exact-step execution")
    spec = SCREENING_REGISTRY.get(result.config_name)
    if spec is None or spec.mechanism != "intentional_updates":
        raise ValueError("result must use a registered Intentional Updates arm")
    if result.base_learner != spec.base_learner:
        raise ValueError("result base learner must match the registered arm")
    if _intentional_updates_hp(result.hyperparameters) != spec.hyperparameters:
        raise ValueError("result hyperparameters must match the registered arm")
    steps = result.config.n_steps
    mechanism_enabled = spec.hyperparameters["intentional_enabled"] == 1.0
    feature_updates = spec.hyperparameters["update_features"] == 1.0
    persistent_bytes = _intentional_updates_persistent_numeric_bytes(
        result.config, mechanism_enabled=mechanism_enabled
    )
    if steps > ((1 << 31) - 1) // 2:
        raise ValueError("Intentional Updates model-query budget exceeds signed int32")
    if persistent_bytes > _INTENTIONAL_MAX_PERSISTENT_BYTES:
        raise ValueError("Intentional Updates persistent state exceeds 256 MiB")
    return {
        "schema": INTENTIONAL_UPDATES_RECORD_SCHEMA,
        "references": {
            "paper": INTENTIONAL_UPDATES_PAPER_REVISION,
            "official_code": INTENTIONAL_UPDATES_CODE_REVISION,
            "equation": "Eq. 5 with y=log p(label|x), Delta=eta*surprisal",
            "protocol_difference": (
                "supervised IPMNIST, lambda=0, no RL transition/trace; this is a "
                "protocol extension, not the publication algorithm"
            ),
        },
        "arm": result.config_name,
        "seed": result.seed,
        "config": result.config.to_config(),
        "hyperparameters": dict(result.hyperparameters),
        "matched_axes": [
            "seed",
            "example_schedule",
            "observations",
            "updates",
            "allowed_boundary_information:none",
        ],
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "publication_equivalent": False,
        },
        "gates": {
            "mechanism_enabled": mechanism_enabled,
            "backpropagation": True,
            "feature_updates": feature_updates,
            "head_updates": True,
        },
        "resources": {
            "observations": steps,
            "updates": steps,
            "backward_passes": steps,
            # value_and_grad performs one pre-update model evaluation and
            # _step_metrics performs the post-update same-example query.
            "model_queries": 2 * steps,
            "persistent_numeric_bytes": persistent_bytes,
            "timing_telemetry_seconds": float(result.wall_clock_seconds),
            "timing_is_selection_metric": False,
        },
        "metrics": {
            "per_task_accuracy": result.per_task_accuracy.tolist(),
            "per_task_loss": result.per_task_loss.tolist(),
            "per_task_plasticity": result.per_task_plasticity.tolist(),
        },
    }


def _trusted_intentional_json(value: object, *, context: str) -> object:
    value_type = type(value)
    if value_type is dict:
        mapping = cast(dict[object, object], value)
        if not all(type(key) is str for key in mapping):
            raise ValueError(f"{context} keys must be exact strings")
        return {
            cast(str, key): _trusted_intentional_json(item, context=f"{context}.{key}")
            for key, item in mapping.items()
        }
    if value_type is list:
        sequence = cast(list[object], value)
        return [
            _trusted_intentional_json(item, context=f"{context}[{index}]")
            for index, item in enumerate(sequence)
        ]
    if value_type is str or value_type is bool or value_type is int or value is None:
        return value
    if value_type is float and math.isfinite(cast(float, value)):
        return value
    raise ValueError(f"{context} must contain only finite exact JSON values")


def _intentional_exact_object(
    value: object, keys: frozenset[str], *, context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    resolved = cast(dict[str, Any], value)
    if frozenset(resolved) != keys:
        raise ValueError(f"{context} keys do not match the frozen schema")
    return resolved


def _intentional_exact_int(value: object, *, context: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= (1 << 31) - 1:
        raise ValueError(f"{context} must be a bounded exact integer")
    return value


def validate_intentional_updates_development_record(
    record: object,
) -> dict[str, Any]:
    """Fail closed over an Intentional Updates development record."""
    if type(record) is not dict:
        raise ValueError("Intentional Updates record must be an exact object")
    payload = _intentional_exact_object(
        _trusted_intentional_json(record, context="Intentional Updates record"),
        frozenset(
            {
                "schema",
                "references",
                "arm",
                "seed",
                "config",
                "hyperparameters",
                "matched_axes",
                "policy",
                "gates",
                "resources",
                "metrics",
            }
        ),
        context="Intentional Updates record",
    )
    if payload["schema"] != INTENTIONAL_UPDATES_RECORD_SCHEMA or type(payload["schema"]) is not str:
        raise ValueError("Intentional Updates schema does not match the frozen protocol")
    references = _intentional_exact_object(
        payload["references"],
        frozenset({"paper", "official_code", "equation", "protocol_difference"}),
        context="references",
    )
    if any(type(value) is not str for value in references.values()):
        raise ValueError("references must contain exact strings")
    if type(payload["arm"]) is not str:
        raise ValueError("arm must be an exact string")
    spec = screening_spec(payload["arm"])
    if spec.mechanism != "intentional_updates":
        raise ValueError("arm must be a registered Intentional Updates arm")
    _intentional_exact_int(payload["seed"], context="seed")
    config_raw = _intentional_exact_object(
        payload["config"],
        frozenset({"n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes"}),
        context="config",
    )
    for key, value in config_raw.items():
        _intentional_exact_int(value, context=f"config.{key}", positive=True)
    if config_raw["n_tasks"] > _INTENTIONAL_MAX_RECORD_TASKS:
        raise ValueError("config.n_tasks exceeds the bounded record limit")
    hyperparameters = _intentional_exact_object(
        payload["hyperparameters"], _INTENTIONAL_UPDATES_HP_KEYS, context="hyperparameters"
    )
    if _intentional_updates_hp(hyperparameters) != spec.hyperparameters:
        raise ValueError("hyperparameters do not match the registered arm")
    matched_axes = payload["matched_axes"]
    expected_axes = [
        "seed",
        "example_schedule",
        "observations",
        "updates",
        "allowed_boundary_information:none",
    ]
    if (
        type(matched_axes) is not list
        or len(matched_axes) != len(expected_axes)
        or any(type(value) is not str for value in matched_axes)
        or matched_axes != expected_axes
    ):
        raise ValueError("matched_axes do not match the frozen protocol")
    policy = _intentional_exact_object(
        payload["policy"],
        frozenset({"development_only", "scientific_promotion_allowed", "publication_equivalent"}),
        context="policy",
    )
    if policy != {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "publication_equivalent": False,
    } or any(type(value) is not bool for value in policy.values()):
        raise ValueError("Intentional Updates records are permanently nonpromoting")
    gates = _intentional_exact_object(
        payload["gates"],
        frozenset({"mechanism_enabled", "backpropagation", "feature_updates", "head_updates"}),
        context="gates",
    )
    if any(type(value) is not bool for value in gates.values()):
        raise ValueError("gates must contain exact booleans")
    resources = _intentional_exact_object(
        payload["resources"],
        frozenset(
            {
                "observations",
                "updates",
                "backward_passes",
                "model_queries",
                "persistent_numeric_bytes",
                "timing_telemetry_seconds",
                "timing_is_selection_metric",
            }
        ),
        context="resources",
    )
    for key in (
        "observations",
        "updates",
        "backward_passes",
        "model_queries",
        "persistent_numeric_bytes",
    ):
        _intentional_exact_int(resources[key], context=f"resources.{key}")
    if (
        type(resources["timing_telemetry_seconds"]) is not float
        or not 0.0 <= resources["timing_telemetry_seconds"] <= 604_800.0
        or resources["timing_is_selection_metric"] is not False
        or resources["persistent_numeric_bytes"] > _INTENTIONAL_MAX_PERSISTENT_BYTES
    ):
        raise ValueError("resources violate the bounded telemetry-only protocol")
    metrics = _intentional_exact_object(
        payload["metrics"],
        frozenset(
            {
                "per_task_accuracy",
                "per_task_loss",
                "per_task_plasticity",
            }
        ),
        context="metrics",
    )
    n_tasks = config_raw["n_tasks"]
    for key in ("per_task_accuracy", "per_task_loss", "per_task_plasticity"):
        values = metrics[key]
        if (
            type(values) is not list
            or len(values) != n_tasks
            or any(type(value) is not float or not math.isfinite(value) for value in values)
        ):
            raise ValueError(f"{key} must contain one exact finite float per task")
    try:
        config = IPMNISTConfig(**config_raw)
        arm = payload["arm"]
        seed = require_jax_seed(payload["seed"], name="record seed")
        result = ScreeningRunResult(
            config_name=arm,
            base_learner=screening_spec(arm).base_learner,
            hyperparameters=hyperparameters,
            seed=seed,
            config=config,
            per_task_accuracy=np.asarray(metrics["per_task_accuracy"], dtype=np.float64),
            per_task_loss=np.asarray(metrics["per_task_loss"], dtype=np.float64),
            per_task_plasticity=np.asarray(
                metrics["per_task_plasticity"], dtype=np.float64
            ),
            wall_clock_seconds=resources["timing_telemetry_seconds"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Intentional Updates result fields") from error
    expected = intentional_updates_development_record(result)
    # Canonical JSON distinguishes hostile booleans from integers and also
    # guarantees the record remains finite/serializable.
    try:
        actual_json = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        expected_json = json.dumps(
            expected, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Intentional Updates record must be finite strict JSON") from error
    if actual_json != expected_json:
        if payload.get("resources") != expected["resources"]:
            raise ValueError("Intentional Updates resource counters do not match the run")
        raise ValueError("Intentional Updates record does not match the frozen protocol")
    return expected
def bounded_elastic_development_result_payload(
    result: ScreeningRunResult, *, outcome: str
) -> dict[str, object]:
    """Build and validate one exact development-only bounded-structure receipt."""
    spec = screening_spec(result.config_name)
    registered_arms = {
        "bounded_structure_off",
        "bounded_growth",
        "bounded_elastic",
        "bounded_fixed_cbp",
    }
    if result.config_name not in registered_arms or result.noise_mode != "step":
        raise ValueError("receipt requires one exact-step registered bounded-structure arm")
    if result.hyperparameters != spec.hyperparameters:
        raise ValueError("result hyperparameters drift from the registered arm")
    config = result.config
    observations = config.n_tasks * config.task_length
    structure_resources = bounded_elastic_resource_expectations(
        arm=result.config_name,
        n_tasks=config.n_tasks,
        input_dim=config.input_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        n_classes=config.n_classes,
    )
    payload: dict[str, object] = {
        "schema": BOUNDED_ELASTIC_RESULT_SCHEMA,
        "comparison_id": BOUNDED_ELASTIC_COMPARISON_ID,
        "paper_revision": BOUNDED_ELASTIC_PAPER_REVISION,
        "paper_source_sha256": BOUNDED_ELASTIC_PAPER_SOURCE_SHA256,
        "arm": result.config_name,
        "seed": result.seed,
        "n_tasks": config.n_tasks,
        "task_length": config.task_length,
        "input_dim": config.input_dim,
        "hidden1": config.hidden1,
        "hidden2": config.hidden2,
        "n_classes": config.n_classes,
        "observations": observations,
        "updates": observations,
        "allowed_boundary_information": ["known_fixed_length_task_boundary"],
        "allowed_task_information": ["current_example_label"],
        "hyperparameters": dict(spec.hyperparameters),
        "metrics": {
            "mean_online_accuracy": float(np.mean(result.per_task_accuracy)),
            "mean_loss": float(np.mean(result.per_task_loss)),
            "mean_plasticity": float(np.mean(result.per_task_plasticity)),
        },
        "resources": {
            **structure_resources,
            "environment_steps": 0,
            "data_steps": observations,
            "model_queries": 2 * observations,
            "timing_seconds": float(result.wall_clock_seconds),
            "timing_is_telemetry_only": True,
        },
        "outcome": outcome,
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    return validate_bounded_elastic_development_result(payload)


def cchain_development_result_payload(
    result: ScreeningRunResult, *, outcome: str
) -> dict[str, object]:
    """Build and strictly validate one nonpromoting C-CHAIN result receipt."""
    spec = screening_spec(result.config_name)
    if spec.mechanism != "c_chain" or result.noise_mode != "step":
        raise ValueError("a C-CHAIN receipt requires an exact-step registered C-CHAIN arm")
    if result.hyperparameters != spec.hyperparameters:
        raise ValueError("result hyperparameters drift from the registered C-CHAIN arm")
    diagnostics = result.mechanism_diagnostics
    if type(diagnostics) is not dict:
        raise ValueError("a C-CHAIN receipt requires measured mechanism diagnostics")
    required_diagnostics = {
        "mean_probability_kl",
        "mean_logit_mse",
        "final_coefficient",
        "diagnostic_updates",
        "ntk_threshold_rank",
        "ntk_off_diagonal_abs_mean",
        "ntk_diagonal_mean",
        "ntk_examples",
    }
    if set(diagnostics) != required_diagnostics:
        raise ValueError("C-CHAIN mechanism diagnostics have unexpected or missing fields")
    config = result.config
    observations = config.n_tasks * config.task_length
    active_updates = max(observations - int(spec.hyperparameters["snapshot_warmup_updates"]), 0)
    ntk_examples = int(diagnostics["ntk_examples"])
    parameter_count = (
        config.input_dim * config.hidden1
        + config.hidden1
        + config.hidden1 * config.hidden2
        + config.hidden2
        + config.hidden2 * config.n_classes
        + config.n_classes
    )
    persistent_scalars = (
        4 * parameter_count
        + 5 * 6
        + int(spec.hyperparameters["reference_capacity"]) * config.input_dim
        + 2 * int(spec.hyperparameters["coefficient_window"])
        + 9
    )
    ntk_rows = ntk_examples * config.n_classes
    task_model_queries = 2 * observations
    churn_model_queries = 2 * active_updates
    ntk_model_queries = ntk_examples
    payload: dict[str, object] = {
        "schema": CCHAIN_RESULT_SCHEMA,
        "comparison_id": CCHAIN_COMPARISON_ID,
        "paper_revision": CCHAIN_PAPER_REVISION,
        "official_commit": CCHAIN_OFFICIAL_COMMIT,
        "adaptation_id": CCHAIN_ADAPTATION_ID,
        "comparability_gaps": list(CCHAIN_COMPARABILITY_GAPS),
        "arm": result.config_name,
        "seed": result.seed,
        "development_seed_protocol": list(CCHAIN_DEVELOPMENT_SEEDS),
        "n_tasks": config.n_tasks,
        "task_length": config.task_length,
        "input_dim": config.input_dim,
        "hidden1": config.hidden1,
        "hidden2": config.hidden2,
        "n_classes": config.n_classes,
        "observations": observations,
        "updates": observations,
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "hyperparameters": dict(spec.hyperparameters),
        "metrics": {
            "mean_online_accuracy": float(np.mean(result.per_task_accuracy)),
            "mean_loss": float(np.mean(result.per_task_loss)),
            "mean_plasticity": float(np.mean(result.per_task_plasticity)),
            **diagnostics,
        },
        "resources": {
            "persistent_bytes": 4 * persistent_scalars,
            "ntk_jacobian_envelope_bytes": 2 * ntk_rows * parameter_count * 4,
            "environment_steps": 0,
            "data_steps": observations,
            "optimizer_updates": observations,
            "task_model_queries": task_model_queries,
            "churn_reference_updates": active_updates,
            "churn_model_queries": churn_model_queries,
            "ntk_model_queries": ntk_model_queries,
            "model_queries": (
                task_model_queries + churn_model_queries + ntk_model_queries
            ),
            "timing_seconds": float(result.wall_clock_seconds),
            "timing_is_telemetry_only": True,
        },
        "outcome": outcome,
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    return validate_cchain_development_result(payload)


def replay_frozen_development_result_payload(
    result: ScreeningRunResult, *, outcome: str
) -> dict[str, object]:
    """Build one strict replay/frozen-feature development receipt."""
    spec = screening_spec(result.config_name)
    if spec.mechanism not in {"replay_in_context", "frozen_feature_ceiling"}:
        raise ValueError("result is not a registered replay/frozen-feature arm")
    if result.noise_mode != "step" or result.hyperparameters != spec.hyperparameters:
        raise ValueError("replay/frozen receipt requires exact-step registered settings")
    family = (
        "replay"
        if spec.mechanism == "replay_in_context"
        else {
            "randumb_random_features": "randumb",
            "ranpac_random_projection": "ranpac",
            "prol_prompt_mechanism_off": "prol",
            "prol_prompt_proxy": "prol",
        }[result.config_name]
    )
    config = result.config
    observations = config.n_tasks * config.task_length
    resources = expected_resources_for_result(
        family,
        observations,
        config.input_dim,
        config.hidden1,
        config.hidden2,
        config.n_classes,
    )
    payload: dict[str, object] = {
        "schema": REPLAY_FROZEN_RESULT_SCHEMA,
        "comparison_id": REPLAY_FROZEN_COMPARISON_ID,
        "paper_revisions": [
            REPLAY_PAPER_REVISION,
            RANDUMB_PAPER_REVISION,
            RANPAC_PAPER_REVISION,
            PROL_PAPER_REVISION,
        ],
        "official_commits": [
            REPLAY_OFFICIAL_CODE,
            RANDUMB_COMMIT,
            RANPAC_COMMIT,
            PROL_COMMIT,
        ],
        "protocol_gaps": list(REPLAY_FROZEN_PROTOCOL_GAPS),
        "arm": result.config_name,
        "family": family,
        "seed": result.seed,
        "n_tasks": config.n_tasks,
        "task_length": config.task_length,
        "input_dim": config.input_dim,
        "hidden1": config.hidden1,
        "hidden2": config.hidden2,
        "n_classes": config.n_classes,
        "observations": observations,
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "hyperparameters": dict(spec.hyperparameters),
        "metrics": {
            "mean_online_accuracy": float(np.mean(result.per_task_accuracy)),
            "mean_loss": float(np.mean(result.per_task_loss)),
            "mean_plasticity": float(np.mean(result.per_task_plasticity)),
        },
        "resources": {
            **resources,
            "timing_seconds": float(result.wall_clock_seconds),
            "timing_is_telemetry_only": True,
        },
        "outcome": outcome,
        "negative_outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    return validate_replay_frozen_result(payload)


def noise_curvature_development_result_payload(
    result: ScreeningRunResult, *, outcome: str
) -> dict[str, object]:
    """Build and strictly validate one nonpromoting scheduler receipt."""

    spec = screening_spec(result.config_name)
    if spec.mechanism != "noise_curvature_scheduler" or result.noise_mode != "step":
        raise ValueError(
            "a noise-curvature receipt requires an exact-step registered scheduler arm"
        )
    if result.hyperparameters != spec.hyperparameters:
        raise ValueError("result hyperparameters drift from the registered scheduler arm")
    config = result.config
    observations = config.n_tasks * config.task_length
    interval = int(spec.hyperparameters["control_interval"])
    power_iterations = int(spec.hyperparameters["power_iterations"])
    if config.task_length % interval:
        raise ValueError("task_length must be divisible by control_interval")
    controller_events = observations // interval
    first_order_queries = observations + controller_events * interval
    loss_queries = observations
    hvp_queries = controller_events * 3 * power_iterations
    persistent_bytes = noise_curvature_persistent_bytes(
        parameter_count=config.parameter_count,
        input_dim=config.input_dim,
        control_interval=interval,
    )
    payload: dict[str, object] = {
        "schema": NOISE_CURVATURE_RESULT_SCHEMA,
        "comparison_id": NOISE_CURVATURE_COMPARISON_ID,
        "paper_revision": NOISE_CURVATURE_PAPER_REVISION,
        "official_code_status": NOISE_CURVATURE_OFFICIAL_CODE_STATUS,
        "protocol_differences": list(NOISE_CURVATURE_PROTOCOL_DIFFERENCES),
        "live_control": NOISE_CURVATURE_LIVE_CONTROL,
        "arm": result.config_name,
        "seed": result.seed,
        "development_seed_protocol": list(NOISE_CURVATURE_DEVELOPMENT_SEEDS),
        "n_tasks": config.n_tasks,
        "task_length": config.task_length,
        "input_dim": config.input_dim,
        "hidden1": config.hidden1,
        "hidden2": config.hidden2,
        "n_classes": config.n_classes,
        "observations": observations,
        "updates": observations,
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "hyperparameters": dict(spec.hyperparameters),
        "metrics": {
            "mean_online_accuracy": float(np.mean(result.per_task_accuracy)),
            "mean_loss": float(np.mean(result.per_task_loss)),
            "mean_plasticity": float(np.mean(result.per_task_plasticity)),
        },
        "resources": {
            "persistent_bytes": persistent_bytes,
            "environment_steps": 0,
            "data_steps": observations,
            "model_queries": first_order_queries + loss_queries + hvp_queries,
            "first_order_gradient_queries": first_order_queries,
            "loss_only_queries": loss_queries,
            "hessian_vector_product_queries": hvp_queries,
            "controller_events": controller_events,
            "timing_seconds": float(result.wall_clock_seconds),
            "timing_is_telemetry_only": True,
        },
        "outcome": outcome,
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    return validate_noise_curvature_development_result(payload)


def run_screening_config(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    spec: ScreeningSpec,
    seed: int,
    config: IPMNISTConfig,
    progress_every: int | None = None,
    noise_mode: str = "step",
    noise_pool_steps: int = 64,
    *,
    _task_observer: Callable[[int, Mapping[str, Array], Any], None] | None = None,
) -> ScreeningRunResult:
    """Run one screening configuration for one seed.

    Seed derivation, schedules, init, and the per-step RNG chain mirror
    :func:`~alberta_framework.benchmarks.upgd_ipmnist.run_ipmnist` exactly,
    so control arms reproduce the full-horizon lane and every arm shares the
    control's task/example schedule for paired comparison.

    ``noise_mode="pool"`` mirrors ``run_ipmnist(noise_mode="pool")`` --
    including its per-task pool-key split and per-step offset draw, so the
    control arm reproduces the full lane's pool trajectories bit-for-bit
    (pinned by a unit test) -- but consumes ``spec.noise_update`` instead of
    the fixed UPGD-W equations. Pool shards are a screening-only
    approximation: they record ``noise_mode`` and their effective
    ``noise_pool_steps`` and never merge with exact shards nor pass proxy
    validation.
    """
    if progress_every is not None and (
        type(progress_every) is not int or progress_every <= 0
    ):
        raise ValueError("progress_every must be a positive integer or None")
    if _task_observer is not None and type(_task_observer) is not FunctionType:
        raise TypeError("_task_observer must be an exact Python function or None")
    resolved_seed = require_jax_seed(seed, name="seed")
    noise_mode = _validated_screening_noise_mode(noise_mode, spec)
    if spec.mechanism == "l2_effective_rank":
        er_batch_size = int(spec.hyperparameters["er_batch_size"])
        if config.task_length % er_batch_size != 0:
            raise ValueError("L2-ER requires task_length divisible by er_batch_size")
    if spec.mechanism == "intentional_updates":
        enabled = spec.hyperparameters["intentional_enabled"] == 1.0
        persistent_bytes = _intentional_updates_persistent_numeric_bytes(
            config, mechanism_enabled=enabled
        )
        if config.n_steps > ((1 << 31) - 1) // 2:
            raise ValueError("Intentional Updates model-query budget exceeds signed int32")
        if persistent_bytes > _INTENTIONAL_MAX_PERSISTENT_BYTES:
            raise ValueError("Intentional Updates persistent state exceeds 256 MiB")
    if spec.name.startswith("bounded_") and config.task_length != 5000:
        raise ValueError("bounded structure arms require the registered task_length=5000")
    if spec.mechanism == "noise_curvature_scheduler":
        interval = int(spec.hyperparameters["control_interval"])
        if config.task_length % interval:
            raise ValueError(
                "noise-curvature scheduling requires task_length divisible by "
                "control_interval"
            )
    effective_noise_pool_steps = _validated_screening_noise_pool_steps(
        noise_mode,
        noise_pool_steps if noise_mode == "pool" else None,
    )
    resolved_x, resolved_y = _validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        min_length=config.task_length,
    )
    data_x = jnp.asarray(resolved_x, dtype=jnp.float32)
    data_y = jnp.asarray(resolved_y, dtype=jnp.int32)
    n_train = int(data_x.shape[0])

    if spec.mechanism == "noise_curvature_scheduler":
        init_fn, step_fn = _make_noise_curvature_learner(
            spec.hyperparameters, total_steps=config.n_steps
        )
    else:
        init_fn, step_fn = spec.factory(spec.hyperparameters)

    root = jr.key(jnp.uint32(resolved_seed), impl="threefry2x32")
    key_init, key_schedule, key_noise = jr.split(root, 3)
    params = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, n_train)
    state = init_fn(params)

    def run_task(
        params: dict[str, Array],
        state: Any,
        key: Array,
        permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array, Array]:
        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], StepMetrics]:
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            key, step_key = jr.split(key)
            new_params, new_state, metrics = step_fn(step_params, step_state, x, y, step_key)
            return (new_params, new_state, key), metrics

        (params, state, key), (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, state, key), examples
        )
        return params, state, key, accuracies, losses, plasticities

    shapes = _sorted_param_shapes(config)
    n_flat = int(sum(np.prod(shape) for shape in shapes.values()))
    pool_len = (
        effective_noise_pool_steps * n_flat
        if effective_noise_pool_steps is not None
        else 0
    )
    pool_noise_std = float(spec.hyperparameters.get("noise_std", 0.0))
    noise_update = spec.noise_update
    hp = spec.hyperparameters

    def run_task_pool(
        params: dict[str, Array],
        state: Any,
        key: Array,
        permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array, Array]:
        key, pool_key = jr.split(key)
        pool = jr.normal(pool_key, (pool_len,), jnp.float32) * pool_noise_std

        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], StepMetrics]:
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                step_params, x, y
            )
            key, step_key = jr.split(key)
            offset = jr.randint(step_key, (), 0, pool_len - n_flat + 1)
            flat_noise = jax.lax.dynamic_slice(pool, (offset,), (n_flat,))
            noise = _split_flat_noise(flat_noise, shapes)
            assert noise_update is not None
            new_params, new_state = noise_update(step_params, step_state, grads, noise, hp)
            return (new_params, new_state, key), _step_metrics(
                new_params, x, y, loss, logits
            )

        (params, state, key), (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, state, key), examples
        )
        return params, state, key, accuracies, losses, plasticities

    run_task_jit = jax.jit(run_task_pool if noise_mode == "pool" else run_task)

    task_accuracy: list[float] = []
    task_loss: list[float] = []
    task_plasticity: list[float] = []
    started = time.monotonic()
    for task in range(config.n_tasks):
        params, state, key_noise, accuracies, losses, plasticities = run_task_jit(
            params,
            state,
            key_noise,
            schedule.permutations[task],
            schedule.example_indices[task],
        )
        if spec.mechanism == "l2_effective_rank":
            if type(state) is not L2ERState or not bool(state.transaction_valid):
                raise RuntimeError("L2-ER update transaction became invalid")
        if spec.mechanism == "noise_curvature_scheduler":
            if not isinstance(state, NoiseCurvatureState):
                raise RuntimeError("noise-curvature learner returned an invalid state")
            failures = int(jax.device_get(state.diagnostic_failures))
            if failures:
                raise RuntimeError(
                    "noise-curvature diagnostics became non-finite; refusing a result"
                )
        task_accuracy.append(float(jnp.mean(accuracies)))
        task_loss.append(float(jnp.mean(losses)))
        task_plasticity.append(float(jnp.mean(plasticities)))
        if _task_observer is not None:
            # Diagnostics are deliberately downstream of the complete task update.  The
            # learner never receives this task boundary or anything returned by the observer.
            _task_observer(task, MappingProxyType(dict(params)), state)
        if progress_every is not None and (task + 1) % progress_every == 0:
            elapsed = time.monotonic() - started
            logger.info(
                "%s seed=%d task %d/%d online_acc=%.4f elapsed=%.1fs",
                spec.name,
                resolved_seed,
                task + 1,
                config.n_tasks,
                task_accuracy[-1],
                elapsed,
            )
    mechanism_diagnostics = (
        cchain_host_diagnostics(params, state)
        if type(state) is CChainState
        else None
    )
    return ScreeningRunResult(
        config_name=spec.name,
        base_learner=spec.base_learner,
        hyperparameters=dict(spec.hyperparameters),
        seed=resolved_seed,
        config=config,
        per_task_accuracy=np.asarray(task_accuracy, dtype=np.float64),
        per_task_loss=np.asarray(task_loss, dtype=np.float64),
        per_task_plasticity=np.asarray(task_plasticity, dtype=np.float64),
        wall_clock_seconds=time.monotonic() - started,
        noise_mode=noise_mode,
        noise_pool_steps=effective_noise_pool_steps,
        mechanism_diagnostics=mechanism_diagnostics,
    )


# =============================================================================
# Shards, merge, summary
# =============================================================================


# Derived from the dataclass itself so a new protocol field cannot silently
# reopen the gap this closes: every IPMNISTConfig field has a published
# default, so a shard whose config omits one reconstructs at that default
# instead of what actually ran.
_IPMNIST_CONFIG_FIELDS = frozenset(field.name for field in fields(IPMNISTConfig))

_V2_SHARD_FIELDS = frozenset(
    {
        "schema",
        "evidence_policy",
        "config_name",
        "base_learner",
        "hyperparameters",
        "seed",
        "noise_mode",
        "noise_pool_steps",
        "config",
        "per_task_accuracy",
        "per_task_loss",
        "per_task_plasticity",
        "wall_clock_seconds",
        "created_unix",
        "source_provenance",
        "dataset_provenance",
        "environment",
    }
)
_V2_MECHANISM_SHARD_FIELDS = _V2_SHARD_FIELDS | frozenset({"mechanism_receipt"})


def _require_exact_keys(
    value: object, expected: frozenset[str] | set[str], *, context: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    missing = sorted(expected.difference(actual))
    unexpected = sorted(actual.difference(expected))
    if missing or unexpected:
        parts: list[str] = []
        if missing:
            parts.append(f"missing field(s) {missing}")
        if unexpected:
            parts.append(f"unexpected field(s) {unexpected}")
        raise ValueError(f"{context}: {'; '.join(parts)}")
    return cast(Mapping[str, Any], value)


def _required_nonempty_string(value: object, *, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _is_finite_json_number(value: object) -> bool:
    if type(value) is not int and type(value) is not float:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _require_screening_curve_domain(
    values: np.ndarray, field: str, *, context: str
) -> None:
    if type(field) is not str:
        raise ValueError(f"{context}: field must be an exact string")
    if field in {"per_task_accuracy", "per_task_plasticity"}:
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{context}: {field} values must be in [0, 1]")
    elif field == "per_task_loss" and np.any(values < 0.0):
        raise ValueError(f"{context}: {field} values must be non-negative")


def _validated_nonpromoting_policy(value: object, *, context: str) -> dict[str, object]:
    policy = _require_exact_keys(
        value,
        set(NONPROMOTING_POLICY),
        context=f"{context} evidence_policy",
    )
    if any(
        type(policy[name]) is not type(expected) or policy[name] != expected
        for name, expected in NONPROMOTING_POLICY.items()
    ):
        raise ValueError(
            f"{context}: evidence_policy must be the frozen nonpromoting policy"
        )
    return dict(policy)


def _validated_registered_hyperparameters(
    value: object, spec: ScreeningSpec, *, context: str
) -> dict[str, float]:
    hyperparameters = _require_exact_keys(
        value,
        set(spec.hyperparameters),
        context=f"{context} hyperparameters",
    )
    invalid = [
        name
        for name, expected in spec.hyperparameters.items()
        if type(hyperparameters[name]) is not float
        or not math.isfinite(cast(float, hyperparameters[name]))
        or cast(float, hyperparameters[name]).hex() != expected.hex()
    ]
    if invalid:
        raise ValueError(
            f"{context}: hyperparameters must exactly match the registered arm; "
            f"invalid field(s): {sorted(invalid)}"
        )
    return {name: cast(float, hyperparameters[name]) for name in spec.hyperparameters}


def _validated_source_provenance(value: object, *, context: str) -> dict[str, Any]:
    provenance = _require_exact_keys(
        value,
        {
            "schema",
            "git_commit",
            "git_tree",
            "git_object_format",
            "relevant_source_scope",
            "relevant_source_file_count",
            "relevant_source_sha256",
            "uv_lock_sha256",
            "worktree_clean",
        },
        context=f"{context} source provenance",
    )
    if provenance["schema"] != SOURCE_PROVENANCE_SCHEMA:
        raise ValueError(f"{context} source provenance has an unsupported schema")
    if provenance["git_object_format"] != "sha1":
        raise ValueError(f"{context} source provenance git_object_format must be 'sha1'")
    if not _is_lower_hex(provenance["git_commit"], 40) or not _is_lower_hex(
        provenance["git_tree"], 40
    ):
        raise ValueError(f"{context} source provenance must record lowercase Git SHA-1 IDs")
    if provenance["relevant_source_scope"] != _SOURCE_SCOPE_LABEL:
        raise ValueError(f"{context} source provenance has an unsupported source scope")
    file_count = provenance["relevant_source_file_count"]
    if type(file_count) is not int or file_count <= 0:
        raise ValueError(f"{context} source provenance file count must be positive")
    if not _is_lower_hex(provenance["relevant_source_sha256"], 64) or not _is_lower_hex(
        provenance["uv_lock_sha256"], 64
    ):
        raise ValueError(f"{context} source provenance must record lowercase SHA-256 digests")
    if provenance["worktree_clean"] is not True:
        raise ValueError(f"{context} source provenance must record a clean source worktree")
    return dict(provenance)


def _validated_dataset_provenance(value: object, *, context: str) -> dict[str, Any]:
    provenance = _require_exact_keys(
        value,
        {"schema", "source", "materialization", "x", "y"},
        context=f"{context} dataset provenance",
    )
    if provenance["schema"] != DATASET_PROVENANCE_SCHEMA:
        raise ValueError(f"{context} dataset provenance has an unsupported schema")
    source = _require_exact_keys(
        provenance["source"],
        {"provider", "name", "version", "row_start", "row_stop_exclusive"},
        context=f"{context} dataset provenance source",
    )
    if (
        type(source["provider"]) is not str
        or type(source["name"]) is not str
        or type(source["version"]) is not int
        or type(source["row_start"]) is not int
        or type(source["row_stop_exclusive"]) is not int
        or dict(source) != _DATASET_SOURCE
    ):
        raise ValueError(f"{context} dataset provenance has an unsupported source selection")
    if provenance["materialization"] != _DATASET_MATERIALIZATION:
        raise ValueError(f"{context} dataset provenance has an unsupported materialization")

    arrays: dict[str, Mapping[str, Any]] = {}
    expected_ranks = {"x": 2, "y": 1}
    expected_dtypes = {"x": "<f4", "y": "<i4"}
    for name in ("x", "y"):
        binding = _require_exact_keys(
            provenance[name],
            {"dtype", "shape", "sha256"},
            context=f"{context} dataset provenance {name}",
        )
        shape = binding["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != expected_ranks[name]
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError(
                f"{context} dataset provenance {name} shape must be a positive "
                f"rank-{expected_ranks[name]} integer list"
            )
        if binding["dtype"] != expected_dtypes[name]:
            raise ValueError(
                f"{context} dataset provenance {name} dtype must be {expected_dtypes[name]!r}"
            )
        if not _is_lower_hex(binding["sha256"], 64):
            raise ValueError(
                f"{context} dataset provenance {name} must record a lowercase SHA-256"
            )
        arrays[name] = binding
    if arrays["x"]["shape"][0] != arrays["y"]["shape"][0]:
        raise ValueError(f"{context} dataset provenance x/y row counts must match")
    if arrays["x"]["shape"] != [60_000, 784] or arrays["y"]["shape"] != [60_000]:
        raise ValueError(
            f"{context} dataset provenance must bind canonical x=[60000, 784] and "
            "y=[60000] materializations"
        )
    return dict(provenance)


def _validate_dataset_config_binding(
    provenance: Mapping[str, Any], config: IPMNISTConfig, *, context: str
) -> None:
    x_binding = cast(Mapping[str, Any], provenance["x"])
    x_shape = cast(list[int], x_binding["shape"])
    if config.input_dim != x_shape[1]:
        raise ValueError(
            f"{context}: protocol input_dim={config.input_dim} does not match the "
            f"dataset width {x_shape[1]}"
        )
    if config.n_classes != 10:
        raise ValueError(f"{context}: canonical MNIST protocol n_classes must be 10")


def _validated_runtime_environment(value: object, *, context: str) -> dict[str, Any]:
    environment = _require_exact_keys(
        value,
        {"schema", "python", "platform", "packages", "jax", "process_environment"},
        context=f"{context} runtime environment",
    )
    if environment["schema"] != RUNTIME_SCHEMA:
        raise ValueError(f"{context} runtime environment has an unsupported schema")
    python = _require_exact_keys(
        environment["python"],
        {"implementation", "version"},
        context=f"{context} runtime environment python",
    )
    platform_binding = _require_exact_keys(
        environment["platform"],
        {"system", "release", "machine"},
        context=f"{context} runtime environment platform",
    )
    packages = _require_exact_keys(
        environment["packages"],
        {"chex", "jax", "jaxlib", "numpy", "scikit-learn"},
        context=f"{context} runtime environment packages",
    )
    jax_binding = _require_exact_keys(
        environment["jax"],
        {"backend", "devices", "config"},
        context=f"{context} runtime environment jax",
    )
    jax_config = _require_exact_keys(
        jax_binding["config"],
        {
            "jax_enable_x64",
            "jax_default_matmul_precision",
            "jax_disable_jit",
            "jax_numpy_dtype_promotion",
            "jax_numpy_rank_promotion",
            "jax_random_seed_offset",
            "jax_threefry_partitionable",
            "jax_default_prng_impl",
        },
        context=f"{context} runtime environment jax.config",
    )
    process_environment = _require_exact_keys(
        environment["process_environment"],
        set(_RUNTIME_ENVIRONMENT_KEYS),
        context=f"{context} runtime environment process_environment",
    )
    for field, item in python.items():
        _required_nonempty_string(item, context=f"{context} runtime environment python.{field}")
    for field, item in platform_binding.items():
        _required_nonempty_string(
            item, context=f"{context} runtime environment platform.{field}"
        )
    for field, item in packages.items():
        _required_nonempty_string(
            item, context=f"{context} runtime environment packages.{field}"
        )
    _required_nonempty_string(
        jax_binding["backend"], context=f"{context} runtime environment jax.backend"
    )
    devices = jax_binding["devices"]
    if not isinstance(devices, list) or not devices:
        raise ValueError(f"{context} runtime environment JAX devices must be non-empty")
    device_identities: set[tuple[str, int, int]] = set()
    for index, device in enumerate(devices):
        binding = _require_exact_keys(
            device,
            {"id", "platform", "device_kind", "process_index"},
            context=f"{context} runtime environment JAX device {index}",
        )
        device_id = binding["id"]
        process_index = binding["process_index"]
        if (
            type(device_id) is not int
            or type(process_index) is not int
            or device_id < 0
            or process_index < 0
        ):
            raise ValueError(
                f"{context} runtime environment JAX device IDs must be "
                "non-negative integers"
            )
        device_platform = _required_nonempty_string(
            binding["platform"],
            context=f"{context} runtime environment JAX device {index}.platform",
        )
        identity = (device_platform, process_index, device_id)
        if identity in device_identities:
            raise ValueError(
                f"{context} runtime environment JAX device identities must be unique"
            )
        device_identities.add(identity)
        _required_nonempty_string(
            binding["device_kind"],
            context=f"{context} runtime environment JAX device {index}.device_kind",
        )
    for field in (
        "jax_enable_x64",
        "jax_disable_jit",
        "jax_threefry_partitionable",
    ):
        if type(jax_config[field]) is not bool:
            raise ValueError(f"{context} runtime environment {field} must be boolean")
    precision = jax_config["jax_default_matmul_precision"]
    if precision is not None and (type(precision) is not str or not precision):
        raise ValueError(
            f"{context} runtime environment jax_default_matmul_precision must be a "
            "string or null"
        )
    for field in (
        "jax_numpy_dtype_promotion",
        "jax_numpy_rank_promotion",
        "jax_default_prng_impl",
    ):
        _required_nonempty_string(
            jax_config[field], context=f"{context} runtime environment {field}"
        )
    if type(jax_config["jax_random_seed_offset"]) is not int:
        raise ValueError(
            f"{context} runtime environment jax_random_seed_offset must be an integer"
        )
    if any(
        value is not None and type(value) is not str
        for value in process_environment.values()
    ):
        raise ValueError(
            f"{context} runtime environment process values must be strings or null"
        )
    return dict(environment)


def shard_payload(
    result: ScreeningRunResult,
    *,
    source_provenance: Mapping[str, object],
    dataset_provenance: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, Any]:
    """Serialize one bound (config, seed) screening run as a strict v2 shard."""
    seed = require_jax_seed(result.seed, name="result seed")
    if type(result.base_learner) is not str or not result.base_learner:
        raise ValueError("new shard base_learner must be a non-empty string")
    spec = screening_spec(result.config_name)
    if result.base_learner != spec.base_learner:
        raise ValueError(
            f"new shard base_learner must match registered arm {spec.base_learner!r}"
        )
    hyperparameters = _validated_registered_hyperparameters(
        result.hyperparameters, spec, context="new shard"
    )
    noise_mode = _validated_screening_noise_mode(result.noise_mode, spec)
    noise_pool_steps = _validated_screening_noise_pool_steps(
        noise_mode, result.noise_pool_steps
    )
    source_binding = _validated_source_provenance(source_provenance, context="new shard")
    dataset_binding = _validated_dataset_provenance(dataset_provenance, context="new shard")
    runtime_binding = _validated_runtime_environment(environment, context="new shard")
    _validate_dataset_config_binding(dataset_binding, result.config, context="new shard")
    curves: dict[str, np.ndarray] = {}
    for field in ("per_task_accuracy", "per_task_loss", "per_task_plasticity"):
        try:
            raw_values = np.asarray(jax.device_get(getattr(result, field)))
            if raw_values.dtype.kind not in {"i", "u", "f"}:
                raise TypeError("non-numeric curve")
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"new shard {field} must be finite with shape ({result.config.n_tasks},)"
            ) from exc
        if values.shape != (result.config.n_tasks,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"new shard {field} must be finite with shape ({result.config.n_tasks},)"
            )
        _require_screening_curve_domain(values, field, context="new shard")
        curves[field] = values
    wall_clock_seconds = _validated_wall_clock_seconds(
        result.wall_clock_seconds, "new shard"
    )
    payload: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "config_name": result.config_name,
        "base_learner": result.base_learner,
        "hyperparameters": hyperparameters,
        "seed": seed,
        "noise_mode": noise_mode,
        "noise_pool_steps": noise_pool_steps,
        "config": result.config.to_config(),
        "per_task_accuracy": [round(float(v), 8) for v in curves["per_task_accuracy"]],
        "per_task_loss": [round(float(v), 8) for v in curves["per_task_loss"]],
        "per_task_plasticity": [
            round(float(v), 8) for v in curves["per_task_plasticity"]
        ],
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "created_unix": time.time(),
        "source_provenance": source_binding,
        "dataset_provenance": dataset_binding,
        "environment": runtime_binding,
    }
    if spec.mechanism in {
        "c_chain",
        "replay_in_context",
        "frozen_feature_ceiling",
    }:
        persisted_result = replace(
            result,
            per_task_accuracy=np.asarray(payload["per_task_accuracy"], dtype=np.float64),
            per_task_loss=np.asarray(payload["per_task_loss"], dtype=np.float64),
            per_task_plasticity=np.asarray(
                payload["per_task_plasticity"], dtype=np.float64
            ),
            wall_clock_seconds=cast(float, payload["wall_clock_seconds"]),
        )
        payload["mechanism_receipt"] = (
            cchain_development_result_payload(
                persisted_result, outcome="inconclusive"
            )
            if spec.mechanism == "c_chain"
            else replay_frozen_development_result_payload(
                persisted_result, outcome="inconclusive"
            )
        )
    elif result.mechanism_diagnostics is not None:
        raise ValueError(
            "only a registered mechanism lane may persist mechanism diagnostics"
        )
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("new shard payload must be strict JSON") from exc
    return payload


def load_shard(
    path: Path,
    *,
    spec_registry: Mapping[str, ScreeningSpec] | None = None,
) -> dict[str, Any]:
    """Load legacy v1 or strictly validate a source-bound v2 screening shard."""
    registry = SCREENING_REGISTRY if spec_registry is None else spec_registry
    payload = load_strict_json_object(path)
    schema = payload.get("schema")
    if schema not in {LEGACY_SHARD_SCHEMA, SHARD_SCHEMA}:
        raise ValueError(
            f"{path}: not a supported {LEGACY_SHARD_SCHEMA} or {SHARD_SCHEMA} shard"
        )
    is_v2 = schema == SHARD_SCHEMA
    if is_v2:
        config_name_value = payload.get("config_name")
        expected_fields = (
            _V2_MECHANISM_SHARD_FIELDS
            if type(config_name_value) is str
            and config_name_value in SCREENING_REGISTRY
            and SCREENING_REGISTRY[config_name_value].mechanism
            in {"c_chain", "replay_in_context", "frozen_feature_ceiling"}
            else _V2_SHARD_FIELDS
        )
        _require_exact_keys(payload, expected_fields, context=str(path))
        payload["evidence_policy"] = _validated_nonpromoting_policy(
            payload["evidence_policy"], context=str(path)
        )
        created_unix = payload["created_unix"]
        if type(created_unix) is not int and type(created_unix) is not float:
            raise ValueError(f"{path}: created_unix must be a finite, non-negative number")
        try:
            created_value = float(created_unix)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"{path}: created_unix must be a finite, non-negative number"
            ) from exc
        if not math.isfinite(created_value) or created_value < 0.0:
            raise ValueError(f"{path}: created_unix must be a finite, non-negative number")
        payload["created_unix"] = created_value
        payload["source_provenance"] = _validated_source_provenance(
            payload["source_provenance"], context=str(path)
        )
        payload["dataset_provenance"] = _validated_dataset_provenance(
            payload["dataset_provenance"], context=str(path)
        )
        payload["environment"] = _validated_runtime_environment(
            payload["environment"], context=str(path)
        )
    _require_exact_keys(
        payload["config"], _IPMNIST_CONFIG_FIELDS, context=f"{path}: config"
    )
    config = IPMNISTConfig(**payload["config"])
    if is_v2:
        _validate_dataset_config_binding(
            payload["dataset_provenance"], config, context=str(path)
        )
    for fieldname in ("per_task_accuracy", "per_task_loss", "per_task_plasticity"):
        # Curve typing and metric-domain checks apply to every schema: the
        # legacy v1 shards are the campaign's live format, and a numeric
        # string, boolean, or out-of-domain value would otherwise rank.
        raw_values = payload.get(fieldname)
        if (
            not isinstance(raw_values, list)
            or len(raw_values) != config.n_tasks
            or any(not _is_finite_json_number(value) for value in raw_values)
        ):
            raise ValueError(
                f"{path}: {fieldname} must be a list of finite JSON numbers "
                f"with length {config.n_tasks}"
            )
        values = np.asarray(raw_values, dtype=np.float64)
        if values.shape != (config.n_tasks,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {fieldname} must be finite with shape ({config.n_tasks},)")
        _require_screening_curve_domain(values, fieldname, context=str(path))
        payload[fieldname] = [float(value) for value in values]
    payload["wall_clock_seconds"] = _validated_wall_clock_seconds(
        payload.get("wall_clock_seconds"), path
    )
    payload["seed"] = require_jax_seed(payload.get("seed"), name=f"{path}: seed")
    config_name = _required_nonempty_string(
        payload.get("config_name"), context=f"{path}: config_name"
    )
    if config_name not in registry:
        raise ValueError(f"{path}: unknown config_name")
    spec = registry[config_name]
    base_learner = _required_nonempty_string(
        payload.get("base_learner"), context=f"{path}: base_learner"
    )
    if is_v2 and base_learner != spec.base_learner:
        raise ValueError(
            f"{path}: base_learner must match registered arm {spec.base_learner!r}"
        )
    if is_v2:
        payload["hyperparameters"] = _validated_registered_hyperparameters(
            payload.get("hyperparameters"), spec, context=str(path)
        )
        if spec.mechanism == "c_chain":
            receipt = validate_cchain_development_result(payload["mechanism_receipt"])
            expected_axes = {
                "arm": config_name,
                "seed": payload["seed"],
                "n_tasks": config.n_tasks,
                "task_length": config.task_length,
                "input_dim": config.input_dim,
                "hidden1": config.hidden1,
                "hidden2": config.hidden2,
                "n_classes": config.n_classes,
                "hyperparameters": payload["hyperparameters"],
            }
            if any(receipt[name] != expected for name, expected in expected_axes.items()):
                raise ValueError(f"{path}: C-CHAIN receipt drifts from its enclosing shard")
            if receipt["outcome"] != "inconclusive":
                raise ValueError(
                    f"{path}: a single C-CHAIN shard must remain outcome-inconclusive"
                )
            receipt_metrics = cast(Mapping[str, float], receipt["metrics"])
            expected_metrics = {
                "mean_online_accuracy": float(np.mean(payload["per_task_accuracy"])),
                "mean_loss": float(np.mean(payload["per_task_loss"])),
                "mean_plasticity": float(np.mean(payload["per_task_plasticity"])),
            }
            if any(
                receipt_metrics[name] != expected
                for name, expected in expected_metrics.items()
            ):
                raise ValueError(
                    f"{path}: C-CHAIN receipt metrics drift from persisted shard curves"
                )
            receipt_resources = cast(Mapping[str, object], receipt["resources"])
            if receipt_resources["timing_seconds"] != payload["wall_clock_seconds"]:
                raise ValueError(
                    f"{path}: C-CHAIN receipt timing drifts from its enclosing shard"
                )
            payload["mechanism_receipt"] = receipt
        elif spec.mechanism in {"replay_in_context", "frozen_feature_ceiling"}:
            receipt = validate_replay_frozen_result(payload["mechanism_receipt"])
            expected_axes = {
                "arm": config_name,
                "seed": payload["seed"],
                "n_tasks": config.n_tasks,
                "task_length": config.task_length,
                "input_dim": config.input_dim,
                "hidden1": config.hidden1,
                "hidden2": config.hidden2,
                "n_classes": config.n_classes,
                "hyperparameters": payload["hyperparameters"],
            }
            if any(receipt[name] != expected for name, expected in expected_axes.items()):
                raise ValueError(
                    f"{path}: replay/frozen receipt drifts from its enclosing shard"
                )
            if receipt["outcome"] != "inconclusive":
                raise ValueError(
                    f"{path}: a single replay/frozen shard must remain outcome-inconclusive"
                )
            receipt_metrics = cast(Mapping[str, float], receipt["metrics"])
            expected_metrics = {
                "mean_online_accuracy": float(np.mean(payload["per_task_accuracy"])),
                "mean_loss": float(np.mean(payload["per_task_loss"])),
                "mean_plasticity": float(np.mean(payload["per_task_plasticity"])),
            }
            if receipt_metrics != expected_metrics:
                raise ValueError(
                    f"{path}: replay/frozen receipt metrics drift from shard curves"
                )
            receipt_resources = cast(Mapping[str, object], receipt["resources"])
            if receipt_resources["timing_seconds"] != payload["wall_clock_seconds"]:
                raise ValueError(
                    f"{path}: replay/frozen receipt timing drifts from its shard"
                )
            payload["mechanism_receipt"] = receipt
    noise_mode = _validated_screening_noise_mode(
        payload.get("noise_mode", "step"), spec, context=path
    )
    noise_pool_steps = _validated_screening_noise_pool_steps(
        noise_mode,
        payload.get("noise_pool_steps", _MISSING_NOISE_POOL_STEPS),
        context=path,
        allow_unrecorded_pool=not is_v2,
    )
    payload["noise_mode"] = noise_mode
    payload["noise_pool_steps"] = noise_pool_steps
    payload["config_name"] = config_name
    payload["base_learner"] = base_learner
    if not isinstance(payload.get("hyperparameters"), dict):
        raise ValueError(f"{path}: hyperparameters must be an object")
    if not is_v2:
        environment = payload.get("environment")
        required_environment_fields = ("jax", "numpy", "python", "platform")
        if not isinstance(environment, dict) or any(
            type(environment.get(field)) is not str or not environment[field]
            for field in required_environment_fields
        ):
            raise ValueError(
                f"{path}: environment must record non-empty jax, numpy, python, "
                "and platform strings"
            )
    return payload


def _artifact_file_binding(path: Path, *, context: str) -> dict[str, object]:
    artifact_path = Path(path)
    try:
        raw = artifact_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{context}: could not read artifact {artifact_path}") from exc
    return {
        "path": artifact_path.as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _artifact_file_bindings(
    paths: Sequence[Path], *, context: str
) -> list[dict[str, object]]:
    return [
        _artifact_file_binding(Path(path), context=context)
        for path in paths
    ]


def _require_artifact_bindings_unchanged(
    paths: Sequence[Path],
    expected: Sequence[Mapping[str, object]],
    *,
    context: str,
) -> None:
    current = _artifact_file_bindings(paths, context=context)
    expected_base = [
        {
            "path": binding["path"],
            "size_bytes": binding["size_bytes"],
            "sha256": binding["sha256"],
        }
        for binding in expected
    ]
    if current != expected_base:
        raise RuntimeError(f"{context} changed while the derived receipt was built")


def _require_embedded_artifact_manifest_unchanged(
    manifest: object, *, context: str
) -> None:
    if not isinstance(manifest, list) or not manifest:
        raise RuntimeError(f"{context} is missing an input artifact manifest")
    paths: list[Path] = []
    bindings: list[Mapping[str, object]] = []
    for item in manifest:
        if not isinstance(item, Mapping):
            raise RuntimeError(f"{context} has an invalid input artifact manifest")
        path = item.get("path")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if (
            type(path) is not str
            or not path
            or type(size_bytes) is not int
            or size_bytes < 0
            or not _is_lower_hex(sha256, 64)
        ):
            raise RuntimeError(f"{context} has an invalid input artifact manifest")
        paths.append(Path(path))
        bindings.append(cast(Mapping[str, object], item))
    _require_artifact_bindings_unchanged(paths, bindings, context=context)


def _screening_batch_environment(
    shards: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return the exact runtime contract shared by one derived artifact."""
    if not shards:
        raise ValueError("no shards given")
    reference_environment = shards[0]["environment"]
    environment_mismatches = [
        f"{shard['config_name']}/seed={shard['seed']}"
        for shard in shards
        if shard["environment"] != reference_environment
    ]
    if environment_mismatches:
        raise ValueError(
            "shards span multiple runtime environments; process same-environment runs "
            f"separately (mismatched: {environment_mismatches})"
        )
    return dict(reference_environment)


def _screening_batch_binding(
    shards: Sequence[dict[str, Any]], *, field: str, label: str
) -> dict[str, Any]:
    """Return an exact provenance binding shared by every v2 shard."""
    if not shards:
        raise ValueError("no shards given")
    reference = shards[0][field]
    mismatches = [
        f"{shard['config_name']}/seed={shard['seed']}"
        for shard in shards
        if shard[field] != reference
    ]
    if mismatches:
        raise ValueError(
            f"shards span multiple {label} bindings; process matching runs separately "
            f"(mismatched: {mismatches})"
        )
    return dict(reference)


def _validate_screening_arm_contract(
    config_name: str,
    per_seed: Mapping[int, dict[str, Any]],
) -> None:
    """Reject learner or hyperparameter drift within one named arm."""
    seeds = sorted(per_seed)
    reference_base_learner = per_seed[seeds[0]]["base_learner"]
    mismatched_base_learners = [
        seed
        for seed in seeds
        if per_seed[seed]["base_learner"] != reference_base_learner
    ]
    if mismatched_base_learners:
        raise ValueError(
            f"config {config_name!r} has inconsistent base_learner across seeds: "
            f"seed {seeds[0]} used {reference_base_learner!r}, seed(s) "
            f"{mismatched_base_learners} used different values"
        )
    reference_hp = per_seed[seeds[0]]["hyperparameters"]
    mismatched_hp = [
        seed for seed in seeds if per_seed[seed]["hyperparameters"] != reference_hp
    ]
    if mismatched_hp:
        raise ValueError(
            f"config {config_name!r} has inconsistent hyperparameters across seeds: "
            f"seed {seeds[0]} used {reference_hp!r}, seed(s) {mismatched_hp} used "
            "different values; refusing to merge runs of different mechanisms "
            "under one config_name"
        )


def _late_window_slope(per_task_accuracy: np.ndarray, window: int) -> float:
    """OLS slope (accuracy per task) over the final ``window`` tasks."""
    tail = per_task_accuracy[-window:]
    x = np.arange(tail.shape[0], dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x * (tail - tail.mean())) / denom)


def _validated_proxy_atol(value: object) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or value > 1e-6
    ):
        raise ValueError("proxy validation atol must be a finite float in [0, 1e-6]")
    return value


def merge_shards(
    paths: Sequence[Path],
    control_name: str = "upgd_w_control",
    slope_window: int = 15,
    *,
    spec_registry: Mapping[str, ScreeningSpec] | None = None,
) -> dict[str, Any]:
    """Merge shards into a ranked screening summary with paired comparisons.

    Every config must carry exactly the same seed set. The headline ranking
    and every comparison against the control are therefore computed over the
    same paired runs; an incomplete worker batch is rejected before ranking.
    """
    if type(control_name) is not str:
        raise ValueError("control is invalid")
    normalized_paths = [Path(path) for path in paths]
    input_bindings = _artifact_file_bindings(
        normalized_paths, context="screening shard input"
    )
    shards = [
        load_shard(path, spec_registry=spec_registry) for path in normalized_paths
    ]
    if not shards:
        raise ValueError("no shards given")
    shard_schemas = {shard["schema"] for shard in shards}
    if len(shard_schemas) != 1:
        raise ValueError("shards span multiple shard schemas; merge v1 and v2 separately")
    shard_schema = shard_schemas.pop()
    source_provenance = (
        _screening_batch_binding(
            shards, field="source_provenance", label="source provenance"
        )
        if shard_schema == SHARD_SCHEMA
        else None
    )
    dataset_provenance = (
        _screening_batch_binding(shards, field="dataset_provenance", label="dataset")
        if shard_schema == SHARD_SCHEMA
        else None
    )
    configs = {tuple(sorted(s["config"].items())) for s in shards}
    if len(configs) != 1:
        raise ValueError("shards span multiple protocol configs; merge them separately")
    if type(slope_window) is not int or slope_window < 2:
        raise ValueError("slope_window must be a built-in integer of at least 2")
    noise_modes = {s["noise_mode"] for s in shards}
    if len(noise_modes) != 1:
        raise ValueError(
            "shards span multiple noise modes (pool results are a screening-only "
            "approximation); merge them separately"
        )
    noise_mode = noise_modes.pop()
    unrecorded_pool_shards = [
        f"{s['config_name']}/seed={s['seed']}"
        for s in shards
        if noise_mode == "pool" and s["noise_pool_steps"] is None
    ]
    if unrecorded_pool_shards:
        raise ValueError(
            "pool-mode shard(s) do not record noise_pool_steps; rerun them to a new path "
            f"before merging (unrecorded: {unrecorded_pool_shards})"
        )
    noise_pool_sizes = {s["noise_pool_steps"] for s in shards}
    if len(noise_pool_sizes) != 1:
        raise ValueError(
            "pool-mode shards span multiple noise_pool_steps values; merge them separately"
        )
    noise_pool_steps = noise_pool_sizes.pop()
    reference_environment = _screening_batch_environment(shards)
    by_config: dict[str, dict[int, dict[str, Any]]] = {}
    for shard in shards:
        per_seed = by_config.setdefault(shard["config_name"], {})
        if shard["seed"] in per_seed:
            raise ValueError(
                f"duplicate shard for config={shard['config_name']} seed={shard['seed']}"
            )
        per_seed[shard["seed"]] = shard

    if control_name not in by_config:
        raise ValueError(
            f"control {control_name!r} is not among the merged shards "
            f"(present: {sorted(by_config)}); a summary without its control "
            "would silently carry no paired_vs_control blocks"
        )
    for name, per_seed in sorted(by_config.items()):
        _validate_screening_arm_contract(name, per_seed)
    control = by_config[control_name]
    control_seeds = sorted(control)
    for name, per_seed in sorted(by_config.items()):
        seeds = sorted(per_seed)
        if name != control_name and not any(seed in control for seed in seeds):
            raise ValueError(
                f"config {name!r} shares no seeds with control {control_name!r} "
                f"(seeds {seeds} vs {control_seeds}); refusing to rank an "
                "unpaired entry in the summary"
            )
    seed_sets = {
        name: tuple(sorted(per_seed))
        for name, per_seed in sorted(by_config.items())
    }
    if len(set(seed_sets.values())) != 1:
        raise ValueError(
            f"seed sets differ across configs: {seed_sets}; "
            "merge_shards ranks configs on paired seeds only"
        )

    entries: list[dict[str, Any]] = []
    for name, per_seed in sorted(by_config.items()):
        seeds = sorted(per_seed)
        wall_clock_total = _finite_wall_clock_total(
            [per_seed[s]["wall_clock_seconds"] for s in seeds],
            context=f"config {name!r}",
        )
        acc = np.stack(
            [np.asarray(per_seed[s]["per_task_accuracy"], dtype=np.float64) for s in seeds]
        )
        per_seed_avg = acc.mean(axis=1)
        slopes = np.asarray([_late_window_slope(acc[i], slope_window) for i in range(len(seeds))])
        entry: dict[str, Any] = {
            "config_name": name,
            "base_learner": per_seed[seeds[0]]["base_learner"],
            "hyperparameters": per_seed[seeds[0]]["hyperparameters"],
            "seeds": seeds,
            "n_seeds": len(seeds),
            "average_online_accuracy_mean": float(per_seed_avg.mean()),
            "average_online_accuracy_stderr": (
                float(per_seed_avg.std(ddof=1) / math.sqrt(len(seeds)))
                if len(seeds) > 1
                else 0.0
            ),
            "per_seed_average_online_accuracy": [round(float(v), 6) for v in per_seed_avg],
            "late_window_slope_mean": float(slopes.mean()),
            "per_seed_late_window_slope": [round(float(v), 8) for v in slopes],
            "average_plasticity_mean": float(
                np.mean(
                    [
                        np.mean(per_seed[s]["per_task_plasticity"])
                        for s in seeds
                    ]
                )
            ),
            "wall_clock_seconds_total": round(wall_clock_total, 2),
        }
        common = [s for s in seeds if s in control]
        if name != control_name and common:
            control_avg = np.asarray(
                [
                    np.mean(np.asarray(control[s]["per_task_accuracy"], dtype=np.float64))
                    for s in common
                ]
            )
            ours_avg = np.asarray(
                [
                    np.mean(np.asarray(per_seed[s]["per_task_accuracy"], dtype=np.float64))
                    for s in common
                ]
            )
            diff = ours_avg - control_avg
            entry["paired_vs_control"] = {
                "control": control_name,
                "seeds": common,
                "per_seed_diff": [round(float(v), 6) for v in diff],
                "mean_diff": float(diff.mean()),
                "stderr_diff": (
                    float(diff.std(ddof=1) / math.sqrt(len(common)))
                    if len(common) > 1
                    else 0.0
                ),
                "all_seeds_improve": bool(np.all(diff > 0.0)),
                "beats_control": bool(diff.mean() > 0.0),
                # A one-seed paired mean has no spread, so it cannot authorize
                # the expensive confirmation wave. Keep the paired summary
                # available for mid-wave inspection, but require two shared
                # seeds that all improve before setting the compute-spending
                # flag.
                "confirmation_candidate": bool(
                    len(common) >= 2
                    and np.all(diff > 0.0)
                    and diff.mean() > CONFIRMATION_THRESHOLD
                ),
            }
        entries.append(entry)

    entries.sort(key=lambda e: e["average_online_accuracy_mean"], reverse=True)
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA if shard_schema == SHARD_SCHEMA else LEGACY_SUMMARY_SCHEMA,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "created_unix": time.time(),
        "protocol_config": dict(shards[0]["config"]),
        "environment": dict(reference_environment),
        "noise_mode": noise_mode,
        "noise_pool_steps": noise_pool_steps,
        "control_name": control_name,
        "confirmation_threshold": CONFIRMATION_THRESHOLD,
        "slope_window": slope_window,
        "n_shards": len(shards),
        "results": entries,
    }
    if source_provenance is not None and dataset_provenance is not None:
        summary["source_provenance"] = source_provenance
        summary["dataset_provenance"] = dataset_provenance
        summary["shard_manifest"] = sorted(
            (
                {
                    **binding,
                    "config_name": shard["config_name"],
                    "seed": shard["seed"],
                }
                for binding, shard in zip(input_bindings, shards, strict=True)
            ),
            key=lambda entry: (
                cast(str, entry["config_name"]),
                cast(int, entry["seed"]),
                cast(str, entry["path"]),
            ),
        )
    _require_artifact_bindings_unchanged(
        normalized_paths,
        input_bindings,
        context="screening shard input",
    )
    return summary


def validate_proxy(
    shard_paths: Sequence[Path],
    partials_dir: Path,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Validate control shards against the completed full-horizon partials.

    Checks, per control shard, that the proxy per-task accuracy equals the
    first ``n_tasks`` entries of the corresponding 200-task shard (exact
    prefix property), and that the proxy horizon preserves the known
    UPGD-W > AdamW ordering both in the proxy runs and in the full-run
    prefixes at the same task index.
    """
    atol = _validated_proxy_atol(atol)
    partials_dir = Path(partials_dir)
    normalized_shard_paths = [Path(path) for path in shard_paths]
    shard_bindings = _artifact_file_bindings(
        normalized_shard_paths, context="proxy-validation shard input"
    )
    shards = [load_shard(path) for path in normalized_shard_paths]
    if not shards:
        raise ValueError("no shards given")
    shard_schemas = {shard["schema"] for shard in shards}
    if len(shard_schemas) != 1:
        raise ValueError(
            "proxy-validation shards span multiple shard schemas; validate v1 and v2 "
            "separately"
        )
    shard_schema = shard_schemas.pop()
    source_provenance = (
        _screening_batch_binding(
            shards, field="source_provenance", label="source provenance"
        )
        if shard_schema == SHARD_SCHEMA
        else None
    )
    dataset_provenance = (
        _screening_batch_binding(shards, field="dataset_provenance", label="dataset")
        if shard_schema == SHARD_SCHEMA
        else None
    )
    environment = _screening_batch_environment(shards)
    configs = {tuple(sorted(shard["config"].items())) for shard in shards}
    if len(configs) != 1:
        raise ValueError(
            "proxy-validation shards span multiple protocol configs or horizons; "
            "validate them separately"
        )
    learner_by_control = {
        "upgd_w_control": "upgd_w",
        "adamw_control": "adamw",
    }
    by_control: dict[str, dict[int, dict[str, Any]]] = {}
    for path, shard in zip(shard_paths, shards, strict=True):
        if shard.get("noise_mode", "step") != "step":
            raise ValueError(
                f"{path}: proxy validation requires noise_mode='step' shards "
                f"(got {shard.get('noise_mode')!r})"
            )
        learner = learner_by_control.get(shard["config_name"])
        if learner is None:
            raise ValueError(f"{path}: proxy validation accepts only control shards")
        if shard["base_learner"] != learner:
            raise ValueError(
                f"{path}: control {shard['config_name']!r} must record "
                f"base_learner={learner!r}"
            )
        expected_hp = SCREENING_REGISTRY[shard["config_name"]].hyperparameters
        if shard["hyperparameters"] != expected_hp:
            raise ValueError(
                f"{path}: control {shard['config_name']!r} must record its frozen "
                f"hyperparameters {expected_hp!r}"
            )
        per_seed = by_control.setdefault(shard["config_name"], {})
        if shard["seed"] in per_seed:
            raise ValueError(
                f"duplicate proxy-validation shard for control={shard['config_name']} "
                f"seed={shard['seed']}"
            )
        per_seed[shard["seed"]] = shard
    missing_controls = sorted(set(learner_by_control) - set(by_control))
    if missing_controls:
        raise ValueError(f"proxy validation is missing control shard(s): {missing_controls}")
    for config_name, per_seed in by_control.items():
        _validate_screening_arm_contract(config_name, per_seed)
    control_seed_sets = {
        config_name: tuple(sorted(per_seed))
        for config_name, per_seed in by_control.items()
    }
    if len(set(control_seed_sets.values())) != 1:
        raise ValueError(
            f"proxy-validation control seed sets differ: {control_seed_sets}; "
            "paired controls are required"
        )
    checks: list[dict[str, Any]] = []
    proxy_avg: dict[str, list[float]] = {"upgd_w": [], "adamw": []}
    full_avg: dict[str, list[float]] = {"upgd_w": [], "adamw": []}
    n_tasks_seen: set[int] = set()
    reference_paths: list[Path] = []
    reference_bindings: list[dict[str, object]] = []
    reference_manifest: list[dict[str, object]] = []
    for path, shard in zip(shard_paths, shards, strict=True):
        learner = learner_by_control[shard["config_name"]]
        seed = shard["seed"]
        n_tasks = int(shard["config"]["n_tasks"])
        n_tasks_seen.add(n_tasks)
        partial_path = partials_dir / f"{learner}_seed{seed}.json"
        reference_binding = _artifact_file_binding(
            partial_path, context="proxy-validation reference partial"
        )
        reference = _validated_partial_payload(
            partial_path,
            schema=PARTIAL_SCHEMA_V1,
            seed_field="seeds",
        )
        reference_paths.append(partial_path)
        reference_bindings.append(reference_binding)
        reference_manifest.append(
            {
                **reference_binding,
                "learner": learner,
                "seed": seed,
            }
        )
        if reference["learner"] != learner:
            raise ValueError(
                f"{partial_path}: reference learner {reference['learner']!r} does not "
                f"match expected learner {learner!r}"
            )
        if reference["seeds"] != [seed]:
            raise ValueError(
                f"{partial_path}: reference seeds must equal the shard seed [{seed}]"
            )
        expected_reference_hp = {
            "upgd_w": UPGD_W_PROTOCOL_HYPERPARAMETERS,
            "adamw": ADAMW_PROTOCOL_HYPERPARAMETERS,
        }[learner]
        if reference["hyperparameters"] != expected_reference_hp:
            raise ValueError(
                f"{partial_path}: reference hyperparameters do not match the frozen "
                f"{learner} protocol"
            )
        reference_config = dict(reference["config"])
        proxy_config = dict(shard["config"])
        reference_shape = {
            key: value for key, value in reference_config.items() if key != "n_tasks"
        }
        proxy_shape = {
            key: value for key, value in proxy_config.items() if key != "n_tasks"
        }
        if (
            reference_shape != proxy_shape
            or int(reference_config["n_tasks"]) < int(proxy_config["n_tasks"])
        ):
            raise ValueError(
                f"{partial_path}: reference config is incompatible with the proxy prefix"
            )
        full_curve = np.asarray(reference["per_task_accuracy"][0], dtype=np.float64)
        proxy_curve = np.asarray(shard["per_task_accuracy"], dtype=np.float64)
        max_abs_diff = float(np.max(np.abs(proxy_curve - full_curve[:n_tasks])))
        proxy_avg[learner].append(float(proxy_curve.mean()))
        full_avg[learner].append(float(full_curve[:n_tasks].mean()))
        checks.append(
            {
                "config_name": shard["config_name"],
                "seed": seed,
                "reference_partial": partial_path.as_posix(),
                "max_abs_per_task_diff": max_abs_diff,
                "prefix_match": bool(max_abs_diff <= atol),
            }
        )
    ordering_proxy = (
        bool(np.mean(proxy_avg["upgd_w"]) > np.mean(proxy_avg["adamw"]))
        if proxy_avg["upgd_w"] and proxy_avg["adamw"]
        else None
    )
    ordering_full_prefix = (
        bool(np.mean(full_avg["upgd_w"]) > np.mean(full_avg["adamw"]))
        if full_avg["upgd_w"] and full_avg["adamw"]
        else None
    )
    report: dict[str, Any] = {
        "schema": (
            VALIDATION_SCHEMA if shard_schema == SHARD_SCHEMA else LEGACY_VALIDATION_SCHEMA
        ),
        "created_unix": time.time(),
        "atol": atol,
        "environment": environment,
        "n_tasks": sorted(n_tasks_seen),
        "checks": checks,
        "all_prefixes_match": bool(all(c["prefix_match"] for c in checks)),
        "proxy_mean_average_online_accuracy": {
            k: (float(np.mean(v)) if v else None) for k, v in proxy_avg.items()
        },
        "full_prefix_mean_average_online_accuracy": {
            k: (float(np.mean(v)) if v else None) for k, v in full_avg.items()
        },
        "proxy_preserves_upgd_over_adamw": ordering_proxy,
        "full_prefix_preserves_upgd_over_adamw": ordering_full_prefix,
        "proxy_validated": bool(
            all(c["prefix_match"] for c in checks)
            and ordering_proxy is True
            and ordering_full_prefix is True
        ),
    }
    if source_provenance is not None and dataset_provenance is not None:
        report["evidence_policy"] = dict(NONPROMOTING_POLICY)
        report["source_provenance"] = source_provenance
        report["dataset_provenance"] = dataset_provenance
        report["shard_manifest"] = sorted(
            (
                {
                    **binding,
                    "config_name": shard["config_name"],
                    "seed": shard["seed"],
                }
                for binding, shard in zip(shard_bindings, shards, strict=True)
            ),
            key=lambda entry: (
                cast(str, entry["config_name"]),
                cast(int, entry["seed"]),
                cast(str, entry["path"]),
            ),
        )
        report["reference_partial_manifest"] = sorted(
            reference_manifest,
            key=lambda entry: (
                cast(str, entry["learner"]),
                cast(int, entry["seed"]),
                cast(str, entry["path"]),
            ),
        )
    _require_artifact_bindings_unchanged(
        normalized_shard_paths,
        shard_bindings,
        context="proxy-validation shard input",
    )
    _require_artifact_bindings_unchanged(
        reference_paths,
        reference_bindings,
        context="proxy-validation reference partial",
    )
    return report


# =============================================================================
# CLI
# =============================================================================


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable JSON result without a shared-temp race.

    Screening workers may run concurrently, so both the temporary name and
    final publication must be exclusive.  Refusing an occupied destination
    keeps a duplicate launch from silently replacing another worker's shard.
    """

    encoded = (
        json.dumps(payload, allow_nan=False, indent=1, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write_new(Path(path), encoded)


def _screening_derivation_bindings(
    shard_paths: Sequence[Path],
) -> tuple[dict[str, object], dict[str, object]] | None:
    """Capture the current derivation context only for strict v2 inputs."""
    if not shard_paths:
        return None
    first = load_strict_json_object(Path(shard_paths[0]))
    if first.get("schema") != SHARD_SCHEMA:
        return None
    return _screening_source_provenance(), _screening_runtime_environment()


def _require_v2_derivation_context(
    payload: Mapping[str, Any],
    bindings: tuple[dict[str, object], dict[str, object]] | None,
) -> None:
    schema = payload.get("schema")
    if schema not in {SUMMARY_SCHEMA, VALIDATION_SCHEMA}:
        return
    if bindings is None:
        raise RuntimeError("v2 derivation did not capture a source/runtime context")
    source_provenance, runtime_environment = bindings
    if payload.get("source_provenance") != source_provenance:
        raise RuntimeError(
            "v2 derivation source does not match the source recorded by its shards"
        )
    if payload.get("environment") != runtime_environment:
        raise RuntimeError(
            "v2 derivation runtime does not match the runtime recorded by its shards"
        )
    if _screening_source_provenance() != source_provenance:
        raise RuntimeError("screening source provenance changed during derivation")
    if _screening_runtime_environment() != runtime_environment:
        raise RuntimeError("screening runtime environment changed during derivation")
    _require_embedded_artifact_manifest_unchanged(
        payload.get("shard_manifest"), context="v2 shard inputs"
    )
    if schema == VALIDATION_SCHEMA:
        _require_embedded_artifact_manifest_unchanged(
            payload.get("reference_partial_manifest"),
            context="v2 proxy reference partials",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Screening CLI: ``run`` one (config, seed); ``merge``; ``validate-proxy``."""
    parser = argparse.ArgumentParser(description="IPMNIST mechanism-combination screening")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one (config, seed) shard")
    run_p.add_argument("--config-name", required=True, choices=sorted(SCREENING_REGISTRY))
    run_p.add_argument("--seed", type=int, required=True)
    run_p.add_argument("--n-tasks", type=int, default=PROXY_N_TASKS)
    run_p.add_argument("--task-length", type=int, default=5000)
    run_p.add_argument("--data-home", type=Path, default=None)
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--progress-every", type=int, default=10)
    run_p.add_argument(
        "--noise-mode", choices=("step", "pool"), default="step",
        help="'pool' = screening-only pool-noise approximation "
             "(lean-UPGD-family arms only; never mergeable with exact shards)",
    )
    run_p.add_argument(
        "--noise-pool-steps",
        type=int,
        default=64,
        help="effective pool size recorded in pool-mode shards (must be >= 2)",
    )

    merge_p = sub.add_parser("merge", help="merge shards into a ranked summary")
    merge_p.add_argument("--shards", type=Path, nargs="+", required=True)
    merge_p.add_argument("--control-name", default="upgd_w_control")
    merge_p.add_argument("--slope-window", type=int, default=15)
    merge_p.add_argument("--output", type=Path, required=True)

    val_p = sub.add_parser("validate-proxy", help="validate control shards vs full partials")
    val_p.add_argument("--shards", type=Path, nargs="+", required=True)
    val_p.add_argument("--partials-dir", type=Path,
                       default=Path("outputs/upgd_ipmnist/partials"))
    val_p.add_argument("--atol", type=float, default=1e-6)
    val_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    run_seed = (
        require_jax_seed(args.seed, name="seed")
        if args.command == "run"
        else None
    )

    # Refuse an already-published destination before loading data or processing
    # shards.  This is intentionally only a preflight: a claim file or advisory
    # lock could strand the output after a crashed worker.  The exclusive
    # publication in ``_atomic_write_json`` remains authoritative when two
    # workers pass this check concurrently.
    output_path = _preflight_new_output(
        args.out if args.command == "run" else args.output
    )
    derivation_bindings = (
        _screening_derivation_bindings(args.shards)
        if args.command in {"merge", "validate-proxy"}
        else None
    )

    if args.command == "run":
        assert run_seed is not None
        seed = run_seed
        spec = screening_spec(args.config_name)
        config = IPMNISTConfig(n_tasks=args.n_tasks, task_length=args.task_length)
        source_provenance = _screening_source_provenance()
        runtime_environment = _screening_runtime_environment()
        data_home = args.data_home if args.data_home is not None else default_openml_data_home()
        logger.info("loading MNIST from data_home=%s", data_home)
        data_x, data_y = load_mnist_train(data_home)
        dataset_provenance = _screening_dataset_provenance(data_x, data_y)
        logger.info(
            "running %s seed=%d for %d tasks x %d steps "
            "(noise_mode=%s, noise_pool_steps=%s)",
            spec.name,
            seed,
            config.n_tasks,
            config.task_length,
            args.noise_mode,
            args.noise_pool_steps if args.noise_mode == "pool" else None,
        )
        result = run_screening_config(
            data_x, data_y, spec, seed, config,
            progress_every=args.progress_every,
            noise_mode=args.noise_mode,
            noise_pool_steps=args.noise_pool_steps,
        )
        if _screening_source_provenance() != source_provenance:
            raise RuntimeError("screening source provenance changed during execution")
        if _screening_runtime_environment() != runtime_environment:
            raise RuntimeError("screening runtime environment changed during execution")
        if _screening_dataset_provenance(data_x, data_y) != dataset_provenance:
            raise RuntimeError("screening dataset provenance changed during execution")
        payload = shard_payload(
            result,
            source_provenance=source_provenance,
            dataset_provenance=dataset_provenance,
            environment=runtime_environment,
        )
        _atomic_write_json(output_path, payload)
        logger.info(
            "%s seed=%d done: avg online acc %.4f (wall %.1fs) -> %s",
            spec.name,
            seed,
            float(result.per_task_accuracy.mean()),
            result.wall_clock_seconds,
            args.out,
        )
    elif args.command == "merge":
        summary = merge_shards(
            args.shards, control_name=args.control_name, slope_window=args.slope_window
        )
        _require_v2_derivation_context(summary, derivation_bindings)
        _atomic_write_json(output_path, summary)
        logger.info("merged %d shards -> %s", summary["n_shards"], args.output)
    elif args.command == "validate-proxy":
        report = validate_proxy(args.shards, args.partials_dir, atol=args.atol)
        _require_v2_derivation_context(report, derivation_bindings)
        _atomic_write_json(output_path, report)
        logger.info(
            "proxy_validated=%s (prefix_match=%s ordering=%s) -> %s",
            report["proxy_validated"],
            report["all_prefixes_match"],
            report["proxy_preserves_upgd_over_adamw"],
            args.output,
        )
        if not report["proxy_validated"]:
            logger.error("proxy validation rejected; receipt was preserved at %s", args.output)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
