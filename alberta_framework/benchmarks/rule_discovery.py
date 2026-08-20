"""Automated update-rule discovery over the campaign's primitive vocabulary.

The IPMNIST campaign hand-designed ~60 arms; its theory
(``docs/research/ipmnist-theory.md``) identifies a small primitive vocabulary —
per-feature EMA statistics at multiple decays, shift detectors, gates
(utility, L2-init pull), decays, resets, normalizations, and error signals.
This module makes that vocabulary a **composable DSL**: one branchless
JAX-jittable update step parameterized by a flat genome vector (discrete
mechanism flags + bounded continuous constants), so an entire candidate
population evaluates under a single ``vmap`` on the micro continual suite
(:mod:`alberta_framework.benchmarks.micro_continual`), and a random +
evolutionary search can screen thousands of update rules in minutes.

Genome layout (``GENOME_SIZE`` raw floats in ``[0, 1]``):

- ``FLAG_NAMES`` (thresholded at 0.5): ``norm`` (EMA input z-scoring),
  ``shift_reset`` (per-feature shift-triggered anneal-count reset — the
  champion's re-conditioning), ``gate`` (UPGD utility gate on the descent
  term), ``decay_to_init`` (L2-Init pull instead of decay-to-zero),
  ``surprise_budget`` (meta-arm b: global step size scaled by the
  fast/slow error-EMA ratio), ``meta_decay`` (meta-arm a: statistic
  tracking speed adapted from prediction-error autocorrelation),
  ``utility_shift_reset`` (stale-utility cleanup on detected feature
  shifts), ``w1_shift_reset`` (input-layer rows of detected-shift features
  reset to init), ``hidden_rms`` (stateless hidden-layer RMS
  normalization). Wave-2 mechanism classes: ``rls_head`` (closed-form
  exponentially-forgetting RLS readout over the last hidden layer, ensemble
  member), ``rls_reset_p`` (detected-shift reset of the RLS precision to
  its ridge prior), ``nb_member`` (streaming naive-Bayes ensemble member
  over the conditioned input, recent-accuracy vote weighting),
  ``lr_anneal`` (task-clock-free within-task lr annealing driven by the
  fast/slow error ratio: fast early, low late), ``layer_lr`` (per-layer lr
  ratio: input layer at ``ratio**-1``, head at ``ratio**+1``), and
  ``kalman_norm`` (per-feature predict-update Kalman mean tracking as the
  conditioning alternative to the EMA, detected shifts reinflating its
  posterior uncertainty).
- ``PARAM_NAMES``: bounded transforms of the raw genes (log/linear/
  one-minus-power-of-ten decays), documented in ``_PARAM_BOUNDS``.

The champion-form genome (``norm + shift_reset + gate`` with the
``sigma0_shiftnorm_d099`` constants) reproduces the registered champion arm
step-for-step (pinned in ``tests/test_rule_discovery.py``), so the search
space *contains* the record holder and every fitness gain is a genuine
composition discovery, not a reparameterization artifact.

Overfitting guard: search fitness reads only the search tasks (digits lane
M1+M2+M3; gauss lane G1+G3); the recurrence family and a
differently-parameterized twin of the permutation family are **held out**
for selection validation, and promotion to the real 60-task protocol goes
through ``ipmnist_screening`` arms (failing-test-first) against the
champion bar. The wave-2 fitness (``--suite gauss``) runs on the
transfer-validated gauss-v1 streams of
:mod:`alberta_framework.benchmarks.micro_continual`.

Everything here is a development diagnostic — never promotable evidence.
Search executions happen through the CLI, never inside pytest.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed, require_unique_jax_seeds
from alberta_framework.benchmarks.ipmnist_screening import _atomic_write_json
from alberta_framework.benchmarks.micro_continual import (
    _INIT_DOMAIN,
    HOLDOUT_TASKS,
    MICRO_GAUSS_SUITE_VERSION,
    MICRO_SUITE,
    MICRO_SUITE_VERSION,
    SEARCH_TASKS,
    MicroStreamConfig,
    MicroTaskConfig,
    build_micro_stream,
    generate_stream,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    init_mlp_params,
)

logger = logging.getLogger(__name__)

RESULT_SCHEMA = "alberta.rule_discovery.search.v1"

NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_screening_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
}

FLAG_NAMES: tuple[str, ...] = (
    "norm",
    "shift_reset",
    "gate",
    "decay_to_init",
    "surprise_budget",
    "meta_decay",
    "utility_shift_reset",
    "w1_shift_reset",
    "hidden_rms",
    # --- wave-2 mechanism classes (expanded search).
    "rls_head",
    "rls_reset_p",
    "nb_member",
    "lr_anneal",
    "layer_lr",
    "kalman_norm",
)

PARAM_NAMES: tuple[str, ...] = (
    "lr",
    "weight_decay",
    "norm_decay",
    "fast_decay",
    "shift_k",
    "utility_decay",
    "gate_beta",
    "surprise_gain",
    "surprise_fast",
    "surprise_slow",
    "meta_gain",
    # --- wave-2 constants.
    "rls_lambda",
    "nb_decay",
    "vote_decay",
    "anneal_lo",
    "anneal_hi",
    "layer_lr_ratio",
    "kalman_q",
)

#: value = mode(lo + raw * (hi - lo)); modes: 0 = 10**u, 1 = linear, 2 = 1 - 10**-u.
_MODE_LOG10 = 0
_MODE_LINEAR = 1
_MODE_OMP10 = 2
_PARAM_BOUNDS: dict[str, tuple[int, float, float]] = {
    "lr": (_MODE_LOG10, -3.0, math.log10(0.5)),  # 1e-3 .. 0.5
    "weight_decay": (_MODE_LOG10, -4.0, math.log10(0.05)),  # 1e-4 .. 5e-2
    "norm_decay": (_MODE_OMP10, 0.5, 4.0),  # ~0.684 .. 0.9999
    "fast_decay": (_MODE_LINEAR, 0.7, 0.97),
    "shift_k": (_MODE_LOG10, math.log10(0.25), math.log10(4.0)),
    "utility_decay": (_MODE_OMP10, 1.0, 5.0),  # 0.9 .. 0.99999
    "gate_beta": (_MODE_LOG10, math.log10(0.25), math.log10(4.0)),
    "surprise_gain": (_MODE_LINEAR, 0.25, 2.0),
    "surprise_fast": (_MODE_LINEAR, 0.8, 0.99),
    "surprise_slow": (_MODE_OMP10, 2.0, 4.0),  # 0.99 .. 0.9999
    "meta_gain": (_MODE_LINEAR, 0.5, 4.0),
    "rls_lambda": (_MODE_OMP10, 2.0, 4.0),  # 0.99 .. 0.9999
    "nb_decay": (_MODE_OMP10, 1.0, 3.0),  # 0.9 .. 0.999
    "vote_decay": (_MODE_OMP10, 1.0, 3.0),  # 0.9 .. 0.999
    "anneal_lo": (_MODE_LINEAR, 0.05, 1.0),
    "anneal_hi": (_MODE_LINEAR, 1.0, 4.0),
    "layer_lr_ratio": (_MODE_LOG10, math.log10(0.25), math.log10(4.0)),
    "kalman_q": (_MODE_LOG10, -5.0, -1.0),
}

_N_FLAGS = len(FLAG_NAMES)
GENOME_SIZE = _N_FLAGS + len(PARAM_NAMES)

#: Fitness penalty per active mechanism flag (parsimony pressure).
FLAG_PENALTY = 0.0015

#: Detector margin, matching the champion arm's frozen ``shift_delta``.
SHIFT_DELTA = 0.02
_EPS = 1e-8
#: Fixed decay of the error autocorrelation/variance EMAs (meta-decay input).
_AUTOCORR_DECAY = 0.99
#: Surprise-budget clip range (multiplier on the global step size).
_BUDGET_LO = 0.25
_BUDGET_HI = 4.0
#: RLS-head constants: ridge prior scale, vote temperature on the regression
#: scores, score clip (NaN guard while the flag is off), and the shift-boost
#: reinflation applied to the Kalman normalizer's uncertainty on detection.
_RLS_P0 = 10.0
_RLS_VOTE_TEMP = 4.0
_RLS_SCORE_CLIP = 25.0
_NB_VAR_FLOOR = 1e-3
#: lr-anneal surprise mapping: error ratio 1 -> anneal_lo, ratio >= this ->
#: anneal_hi (task-clock-free within-task annealing).
_ANNEAL_R_HI = 2.0
_KALMAN_SHIFT_BOOST = 25.0
#: Per-tensor exponent of the layer-lr ratio (input layer slow, head fast
#: when ratio > 1; the single gene spans both directions).
_LAYER_EXPONENT: dict[str, float] = {
    "w1": -1.0, "b1": -1.0, "w2": 0.0, "b2": 0.0, "w3": 1.0, "b3": 1.0,
}

_MODES = np.asarray([_PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
_LOS = np.asarray([_PARAM_BOUNDS[name][1] for name in PARAM_NAMES])
_HIS = np.asarray([_PARAM_BOUNDS[name][2] for name in PARAM_NAMES])


def _param_values(raw: Array) -> Array:
    """Map raw continuous genes in ``[0, 1]`` to their bounded values."""
    u = _LOS + jnp.clip(raw, 0.0, 1.0) * (_HIS - _LOS)
    log10v = jnp.power(10.0, u)
    omp10v = 1.0 - jnp.power(10.0, -u)
    modes = jnp.asarray(_MODES)
    return jnp.where(
        modes == _MODE_LOG10, log10v, jnp.where(modes == _MODE_LINEAR, u, omp10v)
    ).astype(jnp.float32)


def decode_genome(genome: np.ndarray | Array) -> dict[str, float]:
    """Decode a raw genome into named flags (0/1) and continuous constants."""
    raw = np.asarray(genome, dtype=np.float64)
    if raw.shape != (GENOME_SIZE,):
        raise ValueError(f"genome must have shape ({GENOME_SIZE},), got {raw.shape}")
    config: dict[str, float] = {}
    for index, name in enumerate(FLAG_NAMES):
        config[name] = 1.0 if raw[index] > 0.5 else 0.0
    for index, name in enumerate(PARAM_NAMES):
        mode, lo, hi = _PARAM_BOUNDS[name]
        u = lo + float(np.clip(raw[_N_FLAGS + index], 0.0, 1.0)) * (hi - lo)
        if mode == _MODE_LOG10:
            config[name] = float(10.0**u)
        elif mode == _MODE_LINEAR:
            config[name] = float(u)
        else:
            config[name] = float(1.0 - 10.0**-u)
    return config


def genome_from_config(config: Mapping[str, float]) -> np.ndarray:
    """Inverse of :func:`decode_genome` (flags become exactly 0.0/1.0)."""
    raw = np.zeros((GENOME_SIZE,), dtype=np.float64)
    for index, name in enumerate(FLAG_NAMES):
        raw[index] = 1.0 if float(config[name]) > 0.5 else 0.0
    for index, name in enumerate(PARAM_NAMES):
        mode, lo, hi = _PARAM_BOUNDS[name]
        value = float(config[name])
        if mode == _MODE_LOG10:
            u = math.log10(value)
        elif mode == _MODE_LINEAR:
            u = value
        else:
            u = -math.log10(1.0 - value)
        raw[_N_FLAGS + index] = np.clip((u - lo) / (hi - lo), 0.0, 1.0)
    return raw.astype(np.float32)


_CHAMPION_CONFIG: dict[str, float] = {
    "norm": 1.0,
    "shift_reset": 1.0,
    "gate": 1.0,
    "decay_to_init": 0.0,
    "surprise_budget": 0.0,
    "meta_decay": 0.0,
    "utility_shift_reset": 0.0,
    "w1_shift_reset": 0.0,
    "hidden_rms": 0.0,
    # sigma0_shiftnorm_d099 constants (frozen champion form).
    "lr": 0.01,
    "weight_decay": 0.01,
    "norm_decay": 0.99,
    "fast_decay": 0.9,
    "shift_k": 1.0,
    "utility_decay": 0.9999,
    "gate_beta": 1.0,
    # Inactive-mechanism defaults (only read when their flag is on).
    "surprise_gain": 1.0,
    "surprise_fast": 0.95,
    "surprise_slow": 0.999,
    "meta_gain": 2.0,
    # Wave-2 flags (all off in the champion form) and their defaults.
    "rls_head": 0.0,
    "rls_reset_p": 0.0,
    "nb_member": 0.0,
    "lr_anneal": 0.0,
    "layer_lr": 0.0,
    "kalman_norm": 0.0,
    "rls_lambda": 0.999,
    "nb_decay": 0.98,
    "vote_decay": 0.99,
    "anneal_lo": 0.5,
    "anneal_hi": 2.0,
    "layer_lr_ratio": 1.0,
    "kalman_q": 0.001,
}

#: The strongest wave-1 discovery at champion-scale constants
#: (``disc_r1_pscale_norms``): surprise budget replaces the utility gate.
_DISC_SURPRISE_CONFIG: dict[str, float] = {
    **_CHAMPION_CONFIG,
    "gate": 0.0,
    "surprise_budget": 1.0,
    "surprise_gain": 0.8360796272754669,
    "surprise_fast": 0.9642297768592835,
    "surprise_slow": 0.9996305719081341,
}


def champion_form_genome() -> np.ndarray:
    """Genome encoding the ``sigma0_shiftnorm_d099`` champion form."""
    return genome_from_config(_CHAMPION_CONFIG)


def seed_genomes() -> Array:
    """Hand-designed seeds injected into the initial search population.

    Rows: champion form; meta-arm (a) surprise-driven per-statistic decay;
    meta-arm (b) error-gated plasticity budget; bare SGD+decay; norm-only
    SGD; champion with L2-init pull.
    """
    rows: list[np.ndarray] = []
    rows.append(champion_form_genome())
    meta_a = dict(_CHAMPION_CONFIG)
    meta_a["meta_decay"] = 1.0
    rows.append(genome_from_config(meta_a))
    meta_b = dict(_CHAMPION_CONFIG)
    meta_b["surprise_budget"] = 1.0
    rows.append(genome_from_config(meta_b))
    bare = {name: 0.0 for name in FLAG_NAMES}
    bare.update({name: _CHAMPION_CONFIG[name] for name in PARAM_NAMES})
    rows.append(genome_from_config(bare))
    norm_only = dict(bare)
    norm_only["norm"] = 1.0
    rows.append(genome_from_config(norm_only))
    l2init = dict(_CHAMPION_CONFIG)
    l2init["decay_to_init"] = 1.0
    rows.append(genome_from_config(l2init))
    # Wave-1 discovery (champion-scale constants) + wave-2 mechanism seeds:
    # each new mechanism enters the initial population on both the champion
    # form and the discovered gate-free form.
    rows.append(genome_from_config(_DISC_SURPRISE_CONFIG))
    for base in (_CHAMPION_CONFIG, _DISC_SURPRISE_CONFIG):
        for extra in (
            {"rls_head": 1.0},
            {"rls_head": 1.0, "rls_reset_p": 1.0},
            {"nb_member": 1.0},
            {"lr_anneal": 1.0},
            {"layer_lr": 1.0, "layer_lr_ratio": 2.0},
            {"kalman_norm": 1.0},
        ):
            variant = dict(base)
            variant.update(extra)
            rows.append(genome_from_config(variant))
    return jnp.asarray(np.stack(rows), dtype=jnp.float32)


def describe_genome(genome: np.ndarray | Array) -> str:
    """One-line interpretable description: active primitives + their constants."""
    config = decode_genome(np.asarray(genome))
    active = [name for name in FLAG_NAMES if config[name] == 1.0]
    relevant: list[str] = ["lr", "weight_decay"]
    if config["norm"] or config["meta_decay"]:
        relevant.append("norm_decay")
    if config["shift_reset"] or config["utility_shift_reset"] or config["w1_shift_reset"]:
        relevant.extend(["fast_decay", "shift_k"])
    if config["gate"]:
        relevant.extend(["utility_decay", "gate_beta"])
    if config["surprise_budget"]:
        relevant.extend(["surprise_gain", "surprise_fast", "surprise_slow"])
    if config["meta_decay"]:
        relevant.append("meta_gain")
    if config["rls_head"]:
        relevant.append("rls_lambda")
    if config["nb_member"]:
        relevant.append("nb_decay")
    if config["rls_head"] or config["nb_member"]:
        relevant.append("vote_decay")
    if config["lr_anneal"]:
        relevant.extend(["anneal_lo", "anneal_hi"])
    if config["layer_lr"]:
        relevant.append("layer_lr_ratio")
    if config["kalman_norm"]:
        relevant.append("kalman_q")
    flags_text = "+".join(active) if active else "(bare sgd)"
    params_text = " ".join(f"{name}={config[name]:.4g}" for name in relevant)
    return f"{flags_text} | {params_text}"


@chex.dataclass(frozen=True)
class RuleState:
    """Carry state of the composable rule step.

    ``init_params`` is the frozen init snapshot (L2-init target and reset
    source); the error scalars drive the surprise-budget and meta-decay
    mechanisms; the normalizer statistics mirror the champion's
    shift-adaptive EMA normalizer. Wave-2 state: per-feature Kalman
    uncertainty (``kalman_p``), the closed-form RLS head over the last
    hidden layer (``rls_p``/``rls_w``), the streaming naive-Bayes member
    (``nb_mean``/``nb_var``/``nb_count``), and the ensemble vote-accuracy
    EMAs (``member_acc``: net, rls, nb).
    """

    utility: dict[str, Array]
    step: Array
    init_params: dict[str, Array]
    norm_mean: Array
    norm_var: Array
    norm_count: Array
    fast_mean: Array
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


def init_rule_state(params: dict[str, Array]) -> RuleState:
    """Fresh rule state for an initialized parameter tree."""
    input_dim = params["w1"].shape[0]
    n_classes = params["b3"].shape[0]
    rls_dim = params["b2"].shape[0] + 1  # last hidden layer + bias feature
    chance_loss = jnp.asarray(math.log(float(n_classes)), jnp.float32)
    return RuleState(  # type: ignore[call-arg]
        utility={name: jnp.zeros_like(value) for name, value in params.items()},
        step=jnp.array(0, dtype=jnp.int32),
        init_params={name: value for name, value in params.items()},
        norm_mean=jnp.zeros(input_dim, dtype=jnp.float32),
        norm_var=jnp.ones(input_dim, dtype=jnp.float32),
        norm_count=jnp.zeros(input_dim, dtype=jnp.float32),
        fast_mean=jnp.zeros(input_dim, dtype=jnp.float32),
        err_fast=chance_loss,
        err_slow=chance_loss,
        err_autocorr=jnp.asarray(0.0, jnp.float32),
        err_var=jnp.asarray(0.0, jnp.float32),
        err_prev_delta=jnp.asarray(0.0, jnp.float32),
        kalman_p=jnp.ones(input_dim, dtype=jnp.float32),
        rls_p=_RLS_P0 * jnp.eye(rls_dim, dtype=jnp.float32),
        rls_w=jnp.zeros((rls_dim, n_classes), dtype=jnp.float32),
        nb_mean=jnp.zeros((n_classes, input_dim), dtype=jnp.float32),
        nb_var=jnp.ones((n_classes, input_dim), dtype=jnp.float32),
        nb_count=jnp.zeros(n_classes, dtype=jnp.float32),
        member_acc=jnp.full((3,), 1.0 / float(n_classes), dtype=jnp.float32),
    )


def _rms_mix(hidden: Array, f_rms: Array) -> Array:
    rms = jnp.sqrt(jnp.mean(hidden * hidden) + _EPS)
    return f_rms * (hidden / rms) + (1.0 - f_rms) * hidden


def _loss_logits(
    params: dict[str, Array], x: Array, y: Array, f_rms: Array
) -> tuple[Array, tuple[Array, Array]]:
    hidden = jax.nn.relu(x @ params["w1"] + params["b1"])
    hidden = _rms_mix(hidden, f_rms)
    hidden = jax.nn.relu(hidden @ params["w2"] + params["b2"])
    hidden = _rms_mix(hidden, f_rms)
    logits = hidden @ params["w3"] + params["b3"]
    return -jax.nn.log_softmax(logits)[y], (logits, hidden)


def rule_step(
    genome: Array,
    params: dict[str, Array],
    state: RuleState,
    x: Array,
    y: Array,
) -> tuple[dict[str, Array], RuleState, Array, Array]:
    """One branchless online step of the genome's update rule.

    Returns ``(new_params, new_state, correct, loss)`` with ``correct`` the
    pre-update prediction hit (the protocol's online-accuracy convention).
    Champion-form genomes reproduce the ``sigma0_shiftnorm_d099`` equations
    (:mod:`alberta_framework.benchmarks.ipmnist_screening`) step-for-step.
    """
    flags = (genome[:_N_FLAGS] > 0.5).astype(jnp.float32)
    values = _param_values(genome[_N_FLAGS:])
    f_norm, f_shift_reset, f_gate, f_init, f_budget, f_meta, f_ureset, f_wreset, f_rms = (
        flags[0], flags[1], flags[2], flags[3], flags[4], flags[5], flags[6], flags[7],
        flags[8],
    )
    f_rls, f_rls_reset, f_nb, f_anneal, f_layer, f_kalman = (
        flags[9], flags[10], flags[11], flags[12], flags[13], flags[14],
    )
    p_lr, p_wd, p_ndecay, p_fast, p_shift_k, p_udecay, p_beta = (
        values[0], values[1], values[2], values[3], values[4], values[5], values[6],
    )
    p_sgain, p_sfast, p_sslow, p_mgain = values[7], values[8], values[9], values[10]
    p_rlam, p_nbdecay, p_vote, p_alo, p_ahi, p_lratio, p_kq = (
        values[11], values[12], values[13], values[14], values[15], values[16],
        values[17],
    )

    # --- shift-adaptive per-feature statistics (champion normalizer parity).
    effective_fast = jnp.minimum(p_fast, 1.0 - 1.0 / (state.norm_count + 2.0))
    new_fast = effective_fast * state.fast_mean + (1.0 - effective_fast) * x
    threshold = p_shift_k * jnp.sqrt(state.norm_var) + SHIFT_DELTA
    shifted = jnp.abs(new_fast - state.norm_mean) > threshold
    shifted_f = shifted.astype(jnp.float32)
    count_reset = jnp.where(shifted, 0.0, state.norm_count)
    new_count = (
        f_shift_reset * count_reset + (1.0 - f_shift_reset) * state.norm_count
    ) + 1.0
    # meta-decay (arm a): error autocorrelation speeds statistic tracking.
    autocorr_score = jnp.clip(state.err_autocorr / (state.err_var + _EPS), 0.0, 1.0)
    meta_target = jnp.clip(
        1.0 - (1.0 - p_ndecay) * (1.0 + p_mgain * autocorr_score), 0.5, p_ndecay
    )
    decay_used = f_meta * meta_target + (1.0 - f_meta) * p_ndecay
    effective_decay = jnp.minimum(decay_used, 1.0 - 1.0 / (new_count + 1.0))
    delta = x - state.norm_mean
    ema_mean = state.norm_mean + (1.0 - effective_decay) * delta
    # Kalman-style predict-update mean tracking (wave-2 conditioning
    # alternative): process noise scaled to the tracked variance, detected
    # shifts reinflate the posterior uncertainty (composes with shift_reset).
    r_obs = jnp.maximum(state.norm_var, _EPS)
    p_pred = (
        state.kalman_p
        + p_kq * r_obs
        + f_shift_reset * shifted_f * _KALMAN_SHIFT_BOOST * r_obs
    )
    kalman_gain = p_pred / (p_pred + r_obs)
    kalman_mean = state.norm_mean + kalman_gain * delta
    new_kalman_p = (1.0 - kalman_gain) * p_pred
    new_mean = f_kalman * kalman_mean + (1.0 - f_kalman) * ema_mean
    delta2 = x - new_mean
    new_var = jnp.maximum(
        effective_decay * state.norm_var + (1.0 - effective_decay) * delta * delta2,
        _EPS,
    )
    x_norm = (x - new_mean) / (jnp.sqrt(new_var) + _EPS)
    x_used = f_norm * x_norm + (1.0 - f_norm) * x

    # --- forward + gradients (pre-update prediction = online accuracy).
    (loss, (logits, hidden2)), grads = jax.value_and_grad(_loss_logits, has_aux=True)(
        params, x_used, y, f_rms
    )

    # --- ensemble members (pre-update readout state; the protocol's
    # predict-then-update convention). The net member is always active;
    # rls/nb join the vote when their flags are on, weighted by their
    # recent-accuracy EMAs.
    n_classes = state.rls_w.shape[1]
    h_aug = jnp.concatenate([hidden2, jnp.ones((1,), dtype=jnp.float32)])
    rls_scores = jnp.clip(h_aug @ state.rls_w, -_RLS_SCORE_CLIP, _RLS_SCORE_CLIP)
    input_dim = x_used.shape[0]
    nb_var_safe = jnp.maximum(state.nb_var, _NB_VAR_FLOOR)
    nb_ll = -0.5 * jnp.sum(
        jnp.log(nb_var_safe)
        + (x_used[None, :] - state.nb_mean) ** 2 / nb_var_safe,
        axis=1,
    )
    s_net = jax.nn.log_softmax(logits)
    s_rls = jax.nn.log_softmax(_RLS_VOTE_TEMP * rls_scores)
    s_nb = jax.nn.log_softmax(nb_ll / float(input_dim))
    w_net = state.member_acc[0]
    w_rls = f_rls * state.member_acc[1]
    w_nb = f_nb * state.member_acc[2]
    w_sum = w_net + w_rls + w_nb + _EPS
    combined = (w_net * s_net + w_rls * s_rls + w_nb * s_nb) / w_sum
    correct = (jnp.argmax(combined) == y).astype(jnp.float32)
    member_hits = jnp.stack(
        [
            (jnp.argmax(s_net) == y).astype(jnp.float32),
            (jnp.argmax(s_rls) == y).astype(jnp.float32),
            (jnp.argmax(s_nb) == y).astype(jnp.float32),
        ]
    )
    new_member_acc = p_vote * state.member_acc + (1.0 - p_vote) * member_hits

    # --- utility gate (champion equations; optional stale-utility cleanup).
    clock = state.step + jnp.array(1, dtype=jnp.int32)
    utility_prev = dict(state.utility)
    utility_prev["w1"] = utility_prev["w1"] * (1.0 - f_ureset * shifted[:, None])
    utility = {
        name: p_udecay * utility_prev[name] + (1.0 - p_udecay) * (-grads[name] * params[name])
        for name in params
    }
    bias_correction = 1.0 - jnp.power(p_udecay, clock.astype(jnp.float32))
    global_max = jnp.max(jnp.stack([jnp.max(utility[name]) for name in sorted(params)]))
    # --- surprise budget (arm b): global step size scales with error ratio.
    ratio = (state.err_fast + _EPS) / (state.err_slow + _EPS)
    budget = jnp.clip(
        jnp.exp(p_sgain * jnp.log(jnp.maximum(ratio, _EPS))), _BUDGET_LO, _BUDGET_HI
    )
    # --- within-task lr annealing (wave-2, task-clock-free): the fast/slow
    # error ratio maps [1, _ANNEAL_R_HI] onto [anneal_lo, anneal_hi] —
    # fast right after a surprise spike, low once the task converges.
    surprise_score = jnp.clip((ratio - 1.0) / (_ANNEAL_R_HI - 1.0), 0.0, 1.0)
    anneal_value = p_alo + (p_ahi - p_alo) * surprise_score
    anneal_mult = 1.0 + f_anneal * (anneal_value - 1.0)
    lr_eff = p_lr * (1.0 + f_budget * (budget - 1.0)) * anneal_mult
    new_params: dict[str, Array] = {}
    for name in params:
        # Per-layer lr ratio (wave-2): input layer at ratio**-1, head at
        # ratio**+1; exponent zero (flag off) is exactly 1.0.
        lr_name = lr_eff * jnp.power(p_lratio, f_layer * _LAYER_EXPONENT[name])
        decay_factor = 1.0 - lr_name * p_wd
        gate = jax.nn.sigmoid(
            p_beta * (utility[name] / bias_correction) / global_max
        )
        stepped = (
            params[name] * decay_factor
            - lr_name * grads[name] * (1.0 - f_gate * gate)
            + lr_name * p_wd * f_init * state.init_params[name]
        )
        new_params[name] = stepped
    new_params["w1"] = jnp.where(
        jnp.logical_and(f_wreset > 0.5, shifted[:, None]),
        state.init_params["w1"],
        new_params["w1"],
    )

    # --- RLS head update (wave-2): exponentially-forgetting recursive least
    # squares on one-hot targets over the last hidden layer, with a leak
    # toward the ridge prior (bounds P under weak excitation) and an
    # optional detected-shift reset of P (uncertainty reinflation).
    ph = state.rls_p @ h_aug
    rls_denom = p_rlam + h_aug @ ph
    k_rls = ph / rls_denom
    rls_err = jax.nn.one_hot(y, n_classes, dtype=jnp.float32) - h_aug @ state.rls_w
    new_rls_w = state.rls_w + jnp.outer(k_rls, rls_err)
    p_upd = (state.rls_p - jnp.outer(k_rls, ph)) / p_rlam
    rls_eye = _RLS_P0 * jnp.eye(state.rls_p.shape[0], dtype=jnp.float32)
    leak = 2.0 * (1.0 - p_rlam)
    p_upd = (1.0 - leak) * p_upd + leak * rls_eye
    p_upd = 0.5 * (p_upd + p_upd.T)
    reset_p = f_rls_reset * jnp.max(shifted_f)
    new_rls_p = (1.0 - reset_p) * p_upd + reset_p * rls_eye

    # --- streaming naive-Bayes member update (wave-2): count-annealed EMA
    # class-conditional diagonal Gaussians over the conditioned input.
    sel = jax.nn.one_hot(y, n_classes, dtype=jnp.float32)
    eff_nb = jnp.minimum(p_nbdecay, 1.0 - 1.0 / (state.nb_count + 2.0))[:, None]
    delta_nb = x_used[None, :] - state.nb_mean
    mean_cand = state.nb_mean + (1.0 - eff_nb) * delta_nb
    new_nb_mean = state.nb_mean + sel[:, None] * (mean_cand - state.nb_mean)
    delta2_nb = x_used[None, :] - new_nb_mean
    var_cand = jnp.maximum(
        eff_nb * state.nb_var + (1.0 - eff_nb) * delta_nb * delta2_nb,
        _NB_VAR_FLOOR,
    )
    new_nb_var = state.nb_var + sel[:, None] * (var_cand - state.nb_var)
    new_nb_count = state.nb_count + sel

    # --- error-signal statistics (surprise + autocorrelation trackers).
    delta_e = loss - state.err_slow
    new_state = RuleState(  # type: ignore[call-arg]
        utility=utility,
        step=clock,
        init_params=state.init_params,
        norm_mean=new_mean,
        norm_var=new_var,
        norm_count=new_count,
        fast_mean=new_fast,
        err_fast=p_sfast * state.err_fast + (1.0 - p_sfast) * loss,
        err_slow=p_sslow * state.err_slow + (1.0 - p_sslow) * loss,
        err_autocorr=_AUTOCORR_DECAY * state.err_autocorr
        + (1.0 - _AUTOCORR_DECAY) * (delta_e * state.err_prev_delta),
        err_var=_AUTOCORR_DECAY * state.err_var
        + (1.0 - _AUTOCORR_DECAY) * (delta_e * delta_e),
        err_prev_delta=delta_e,
        kalman_p=new_kalman_p,
        rls_p=new_rls_p,
        rls_w=new_rls_w,
        nb_mean=new_nb_mean,
        nb_var=new_nb_var,
        nb_count=new_nb_count,
        member_acc=new_member_acc,
    )
    return new_params, new_state, correct, loss


def run_stream(
    genome: Array,
    init_params: dict[str, Array],
    xs: Array,
    ys: Array,
    task_length: int,
) -> tuple[Array, Array]:
    """Run one genome over one materialized stream.

    Returns ``(mean_online_accuracy, per_task_accuracy)`` where tasks are
    contiguous ``task_length`` blocks of the stream.
    """
    state = init_rule_state(init_params)

    def one_step(
        carry: tuple[dict[str, Array], RuleState], step: Array
    ) -> tuple[tuple[dict[str, Array], RuleState], Array]:
        params, state = carry
        new_params, new_state, correct, _ = rule_step(
            genome, params, state, xs[step], ys[step]
        )
        return (new_params, new_state), correct

    n_steps = xs.shape[0]
    (_, _), corrects = jax.lax.scan(
        one_step, (init_params, state), jnp.arange(n_steps)
    )
    per_task = corrects.reshape(n_steps // task_length, task_length).mean(axis=1)
    return corrects.mean(), per_task


_batched_run = jax.jit(
    jax.vmap(run_stream, in_axes=(0, None, None, None, None)), static_argnums=(4,)
)


def _net_config(config: MicroTaskConfig) -> IPMNISTConfig:
    return IPMNISTConfig(
        n_tasks=config.n_tasks,
        task_length=config.task_length,
        input_dim=config.input_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        n_classes=config.n_classes,
    )


#: Micro-MLP widths on the gauss suite (the transfer-validated operating
#: point of ``outputs/micro_continual/SUITE.md``).
_GAUSS_HIDDEN1 = 75
_GAUSS_HIDDEN2 = 38

#: Either micro-suite family: the provisional digits tasks or the canonical
#: transfer-validated Gaussian streams.
EvalConfig = MicroTaskConfig | MicroStreamConfig


def _materialize_eval(
    config: EvalConfig, seed: int
) -> tuple[Array, Array, dict[str, Array], int]:
    """Stream + paired network init for one ``(config, seed)``.

    Digits tasks keep the provisional-suite RNG chain (``key(seed)``); gauss
    streams use the canonical suite's ``_INIT_DOMAIN`` chain so genome runs
    are init-paired with the ladder anchors of ``run_micro_arm``.
    """
    if isinstance(config, MicroStreamConfig):
        stream = generate_stream(config, int(seed))
        net = IPMNISTConfig(
            n_tasks=config.n_regimes,
            task_length=config.regime_length,
            input_dim=config.dim,
            hidden1=_GAUSS_HIDDEN1,
            hidden2=_GAUSS_HIDDEN2,
            n_classes=config.n_classes,
        )
        key_init = jr.fold_in(jr.key(jnp.uint32(seed)), _INIT_DOMAIN)
        params = init_mlp_params(key_init, net)
        return jnp.asarray(stream.x), jnp.asarray(stream.y), params, config.regime_length
    stream_digits = build_micro_stream(config, int(seed))
    params = init_mlp_params(jr.key(np.uint32(seed)), _net_config(config))
    return (
        jnp.asarray(stream_digits.xs),
        jnp.asarray(stream_digits.ys),
        params,
        config.task_length,
    )


def evaluate_population(
    genomes: Array,
    config: EvalConfig,
    *,
    seeds: Sequence[int],
    batch_size: int = 256,
) -> np.ndarray:
    """Mean online accuracy of every genome on one micro task across seeds.

    Paired evaluation: every genome sees the identical stream and identical
    network init per seed (the screening convention).

    Raises:
        ValueError: If ``seeds`` are not unique valid seeds, ``genomes`` is not
            a ``(n_genomes, GENOME_SIZE)`` matrix, or ``batch_size`` is not a
            positive built-in ``int``.
    """
    seeds = require_unique_jax_seeds(seeds, name="seeds")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive built-in int")
    genomes = jnp.asarray(genomes, dtype=jnp.float32)
    if genomes.ndim != 2 or genomes.shape[1] != GENOME_SIZE:
        raise ValueError(
            f"genomes must have shape (n_genomes, {GENOME_SIZE}), got {tuple(genomes.shape)}"
        )
    if not bool(jnp.all(jnp.isfinite(genomes))):
        raise ValueError("genomes must contain only finite values")
    n_genomes = int(genomes.shape[0])
    total = np.zeros((n_genomes,), dtype=np.float64)
    for seed in seeds:
        xs, ys, params, task_length = _materialize_eval(config, int(seed))
        start = 0
        while start < n_genomes:
            stop = min(start + batch_size, n_genomes)
            chunk = genomes[start:stop]
            pad = 0
            if stop - start < batch_size and n_genomes > batch_size:
                pad = batch_size - (stop - start)
                chunk = jnp.concatenate([chunk, chunk[-1:].repeat(pad, axis=0)])
            mean_accuracy, _ = _batched_run(chunk, params, xs, ys, task_length)
            block = np.asarray(mean_accuracy, dtype=np.float64)
            if pad:
                block = block[: stop - start]
            total[start:stop] += block
            start = stop
    result: np.ndarray = total / float(len(seeds))
    return result


def flag_count(genome: np.ndarray | Array) -> int:
    """Number of active mechanism flags."""
    raw = np.asarray(genome)
    return int(np.sum(raw[:_N_FLAGS] > 0.5))


def penalized_fitness(accuracy: np.ndarray, genomes: np.ndarray | Array) -> np.ndarray:
    """Search fitness: mean accuracy minus the per-active-flag parsimony tax."""
    raw = np.asarray(genomes)
    counts = np.sum(raw[:, :_N_FLAGS] > 0.5, axis=1)
    result: np.ndarray = np.asarray(accuracy, dtype=np.float64) - FLAG_PENALTY * counts
    return result


def random_genomes(key: Array, n: int) -> Array:
    """Uniform random genomes (flags ~ Bernoulli(0.5) via thresholding)."""
    return jr.uniform(key, (n, GENOME_SIZE), jnp.float32, 0.0, 1.0)


def mutate(
    key: Array,
    genome: Array,
    *,
    flag_flip_prob: float = 0.15,
    sigma: float = 0.15,
) -> Array:
    """Flip flags with probability ``flag_flip_prob``; jitter continuous genes."""
    key_flip, key_noise = jr.split(key)
    flags = (genome[:_N_FLAGS] > 0.5).astype(jnp.float32)
    flips = jr.bernoulli(key_flip, flag_flip_prob, (_N_FLAGS,))
    new_flags = jnp.where(flips, 1.0 - flags, flags)
    cont = genome[_N_FLAGS:]
    noise = jr.normal(key_noise, cont.shape, jnp.float32) * sigma
    new_cont = jnp.clip(cont + noise, 0.0, 1.0)
    return jnp.concatenate([new_flags, new_cont])


def crossover(key: Array, first: Array, second: Array) -> Array:
    """Uniform crossover: each gene from one parent."""
    mask = jr.bernoulli(key, 0.5, (GENOME_SIZE,))
    return jnp.where(mask, first, second)


def _require_unique_task_names(task_names: Sequence[str], *, name: str) -> tuple[str, ...]:
    names = tuple(task_names)
    if not names:
        raise ValueError(f"{name} must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique task names; got {list(names)}")
    return names


# CLI last-fit is `--n-random 3072`. `10**12` still reaches
# `random_genomes(key, n)` as shape `(n, GENOME_SIZE)` float32.
_MAX_SEARCH_INT = 1_000_000


def _require_search_int(name: str, value: object, *, minimum: int) -> int:
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value > _MAX_SEARCH_INT:
        raise ValueError(f"{name} exceeds the search integer limit")
    return value


def evaluate_suite(
    genomes: Array,
    task_names: Sequence[str],
    *,
    seeds: Sequence[int],
    batch_size: int = 256,
    suite: Mapping[str, EvalConfig] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mean accuracy across the named micro tasks; also per-task vectors.

    Raises:
        ValueError: If ``task_names`` is empty or repeats a task, which would
            silently turn the equal-weight task mean into a weighted one.
    """
    task_names = _require_unique_task_names(task_names, name="task_names")
    registry = MICRO_SUITE if suite is None else suite
    per_task: dict[str, np.ndarray] = {}
    for name in task_names:
        per_task[name] = evaluate_population(
            genomes, registry[name], seeds=seeds, batch_size=batch_size
        )
    mean = np.mean(np.stack([per_task[name] for name in task_names]), axis=0)
    return mean, per_task


#: Gauss-suite fitness lanes (wave-2 search): fitness reads G1+G3 only;
#: G4 (recurrence) and the perturbed-geometry G1p are selection holdouts.
GAUSS_SEARCH_TASKS: tuple[str, ...] = ("G1", "G3")
GAUSS_HOLDOUT_TASKS: tuple[str, ...] = ("G4", "G1p")

#: Search-lane regime count on the frozen gauss-v1 geometry. SUITE.md's
#: guidance: the validated structure lives in the geometry, not the
#: horizon, so fitness runs shrink ``n_regimes`` (16 x 5000 steps here)
#: while keeping the protocol regime length (Adam-class operating regime).
GAUSS_SEARCH_REGIMES = 16


def gauss_suite(n_regimes: int = GAUSS_SEARCH_REGIMES) -> dict[str, MicroStreamConfig]:
    """The wave-2 fitness registry on the transfer-validated gauss-v1 suite.

    ``G1``/``G3``/``G4`` freeze the validated M1 geometry
    (``MicroStreamConfig`` defaults) on their families; ``G1p`` perturbs the
    geometry itself (dim 192, 4 components, 1.5-decade spectrum, wider
    separation) so holdout survival requires transfer across operating
    points, not seed luck.
    """
    return {
        "G1": MicroStreamConfig(
            family="input_permutation", n_regimes=n_regimes, regime_length=5000
        ),
        "G3": MicroStreamConfig(
            family="scale_shift", n_regimes=n_regimes, regime_length=5000
        ),
        "G4": MicroStreamConfig(
            family="recurrence", n_regimes=n_regimes, regime_length=5000,
            recurrence_pool=max(2, min(4, n_regimes)),
        ),
        "G1p": MicroStreamConfig(
            family="input_permutation", n_regimes=n_regimes, regime_length=5000,
            dim=192, n_components=4, spectrum_decades=1.5, mean_separation=0.5,
            component_sparsity=8,
        ),
    }


def _suite_geometry(config: EvalConfig) -> dict[str, Any]:
    """JSON geometry row for either suite family."""
    if isinstance(config, MicroStreamConfig):
        return {
            "suite": "gauss",
            "family": config.family,
            "n_tasks": config.n_regimes,
            "task_length": config.regime_length,
            "input_dim": config.dim,
        }
    return {
        "suite": "digits",
        "family": config.kind,
        "n_tasks": config.n_tasks,
        "task_length": config.task_length,
        "input_dim": config.input_dim,
    }


def _resolved_suite(
    n_tasks: int | None, task_length: int | None, suite_kind: str = "digits"
) -> dict[str, EvalConfig]:
    if n_tasks is not None:
        n_tasks = _require_search_int("n_tasks", n_tasks, minimum=1)
    if task_length is not None:
        task_length = _require_search_int("task_length", task_length, minimum=1)
    suite: dict[str, EvalConfig]
    if suite_kind == "gauss":
        suite = dict(gauss_suite(n_tasks if n_tasks is not None else GAUSS_SEARCH_REGIMES))
        if task_length is not None:
            for name, config in suite.items():
                assert isinstance(config, MicroStreamConfig)
                pool = min(config.recurrence_pool, max(2, config.n_regimes))
                suite[name] = dataclasses.replace(
                    config, regime_length=task_length, recurrence_pool=pool
                )
        return suite
    suite = dict(MICRO_SUITE)
    if n_tasks is not None or task_length is not None:
        for name, config in suite.items():
            assert isinstance(config, MicroTaskConfig)
            suite[name] = dataclasses.replace(
                config,
                n_tasks=n_tasks if n_tasks is not None else config.n_tasks,
                task_length=(
                    task_length if task_length is not None else config.task_length
                ),
            )
    return suite


def tune_champion_baseline(
    key: Array,
    *,
    task_names: Sequence[str],
    eval_seeds: Sequence[int],
    batch_size: int,
    suite: Mapping[str, EvalConfig],
    n_random: int = 256,
    generations: int = 4,
    children: int = 64,
) -> tuple[np.ndarray, float, list[tuple[np.ndarray, float]]]:
    """Tune the champion STRUCTURE's continuous constants at micro scale.

    The promotion bar must be the champion *form* (norm + shift_reset +
    gate, decay-to-zero) with constants given the same search budget class —
    otherwise "beats the champion" would only mean "retuned the learning
    rate for 500-step tasks", the exact micro-overfitting trap. Flags stay
    frozen to the champion pattern; only continuous genes move.

    Returns ``(best_genome, best_accuracy, evaluated)`` where ``evaluated``
    holds every (genome, accuracy) pair for the archive.
    """
    n_random = _require_search_int("n_random", n_random, minimum=0)
    generations = _require_search_int("generations", generations, minimum=0)
    children = _require_search_int("children", children, minimum=1)
    champion = champion_form_genome()
    flags = jnp.asarray(champion[:_N_FLAGS])

    def _project(block: Array) -> Array:
        return jnp.concatenate(
            [jnp.broadcast_to(flags, (block.shape[0], _N_FLAGS)), block[:, _N_FLAGS:]],
            axis=1,
        )

    key_random, key_climb = jr.split(key)
    pool = _project(random_genomes(key_random, n_random))
    pool = jnp.concatenate([jnp.asarray(champion)[None, :], pool], axis=0)
    accuracy, _ = evaluate_suite(
        pool, task_names, seeds=eval_seeds, batch_size=batch_size, suite=suite
    )
    evaluated = [(np.asarray(pool[i]), float(accuracy[i])) for i in range(pool.shape[0])]
    best_genome = np.asarray(pool[int(np.argmax(accuracy))])
    best_accuracy = float(np.max(accuracy))
    for _ in range(generations):
        key_climb, key_gen = jr.split(key_climb)
        child_keys = jr.split(key_gen, children)
        block = jnp.stack(
            [
                mutate(k, jnp.asarray(best_genome), flag_flip_prob=0.0, sigma=0.1)
                for k in child_keys
            ]
        )
        block = _project(block)
        child_accuracy, _ = evaluate_suite(
            block, task_names, seeds=eval_seeds, batch_size=batch_size, suite=suite
        )
        evaluated.extend(
            (np.asarray(block[i]), float(child_accuracy[i])) for i in range(children)
        )
        if float(np.max(child_accuracy)) > best_accuracy:
            best_accuracy = float(np.max(child_accuracy))
            best_genome = np.asarray(block[int(np.argmax(child_accuracy))])
    return best_genome, best_accuracy, evaluated


def run_search(
    *,
    n_random: int = 3072,
    population: int = 256,
    generations: int = 12,
    elite: int = 32,
    eval_seeds: Sequence[int] = (0, 1),
    holdout_seeds: Sequence[int] = (101, 102, 103),
    top_k: int = 12,
    batch_size: int = 256,
    search_seed: int = 0,
    task_names: Sequence[str] = SEARCH_TASKS,
    holdout_names: Sequence[str] = HOLDOUT_TASKS,
    suite: Mapping[str, EvalConfig] | None = None,
) -> dict[str, Any]:
    """Random + evolutionary search over the rule DSL with holdout validation.

    Search fitness reads only ``task_names`` (default M1+M2+M3) on
    ``eval_seeds``; the final candidate ranking is validated on
    ``holdout_names`` (default M4+M1') with disjoint ``holdout_seeds``
    against the **budget-matched tuned champion-form baseline** (see
    :func:`tune_champion_baseline`). Never promotes anything by itself —
    promotion to the real protocol goes through ``ipmnist_screening`` arms.
    """
    n_random = _require_search_int("n_random", n_random, minimum=0)
    population = _require_search_int("population", population, minimum=1)
    generations = _require_search_int("generations", generations, minimum=0)
    elite = _require_search_int("elite", elite, minimum=1)
    top_k = _require_search_int("top_k", top_k, minimum=1)
    batch_size = _require_search_int("batch_size", batch_size, minimum=1)
    eval_seeds = require_unique_jax_seeds(eval_seeds, name="eval_seeds")
    holdout_seeds = require_unique_jax_seeds(holdout_seeds, name="holdout_seeds")
    search_seed = require_jax_seed(search_seed, name="search_seed")
    task_names = _require_unique_task_names(task_names, name="task_names")
    holdout_names = _require_unique_task_names(holdout_names, name="holdout_names")
    if set(task_names) & set(holdout_names):
        raise ValueError("search tasks and holdout tasks must be disjoint")
    if set(eval_seeds) & set(holdout_seeds):
        raise ValueError("search seeds and holdout seeds must be disjoint")
    registry = dict(MICRO_SUITE if suite is None else suite)
    started = time.monotonic()
    root = jr.key(np.uint32(search_seed))
    key_random, key_evolve, key_baseline = jr.split(root, 3)

    seeds_block = seed_genomes()
    n_seeded = int(seeds_block.shape[0])
    randoms = random_genomes(key_random, max(n_random - n_seeded, 0))
    pool = jnp.concatenate([seeds_block, randoms], axis=0)
    logger.info(
        "rule-discovery search: %d initial candidates (%d seeded), tasks=%s seeds=%s",
        int(pool.shape[0]), n_seeded, list(task_names), list(eval_seeds),
    )
    accuracy, per_task = evaluate_suite(
        pool, task_names, seeds=eval_seeds, batch_size=batch_size, suite=registry
    )
    fitness = penalized_fitness(accuracy, pool)

    archive_genomes: list[np.ndarray] = [np.asarray(g) for g in pool]
    archive_accuracy: list[float] = [float(a) for a in accuracy]
    archive_fitness: list[float] = [float(f) for f in fitness]
    archive_origin: list[str] = ["seeded"] * n_seeded + ["random"] * (
        int(pool.shape[0]) - n_seeded
    )

    order = np.argsort(-fitness)
    pop_idx = order[:population]
    pop = jnp.asarray(np.stack([np.asarray(archive_genomes[i]) for i in pop_idx]))
    pop_fitness = np.asarray([archive_fitness[i] for i in pop_idx])
    logger.info(
        "random phase done: best fitness %.5f accuracy %.5f (%s)",
        float(pop_fitness[0]),
        float(archive_accuracy[int(order[0])]),
        describe_genome(np.asarray(pop[0])),
    )

    generation_log: list[dict[str, Any]] = []
    for generation in range(generations):
        key_evolve, key_gen = jr.split(key_evolve)
        elite_order = np.argsort(-pop_fitness)
        elite_idx = elite_order[:elite]
        elites = pop[jnp.asarray(elite_idx)]
        n_children = population - elite
        child_keys = jr.split(key_gen, n_children)
        children: list[Array] = []
        for child_key in child_keys:
            k_pick, k_cross, k_mut = jr.split(child_key, 3)
            picks = jr.randint(k_pick, (2,), 0, elite)
            child = crossover(k_cross, elites[picks[0]], elites[picks[1]])
            children.append(mutate(k_mut, child))
        children_block = jnp.stack(children)
        child_accuracy, _ = evaluate_suite(
            children_block, task_names, seeds=eval_seeds,
            batch_size=batch_size, suite=registry,
        )
        child_fitness = penalized_fitness(child_accuracy, children_block)
        for index in range(n_children):
            archive_genomes.append(np.asarray(children_block[index]))
            archive_accuracy.append(float(child_accuracy[index]))
            archive_fitness.append(float(child_fitness[index]))
            archive_origin.append(f"generation_{generation}")
        pop = jnp.concatenate([elites, children_block], axis=0)
        pop_fitness = np.concatenate(
            [pop_fitness[elite_idx], child_fitness]
        )
        best = int(np.argmax(pop_fitness))
        generation_log.append(
            {
                "generation": generation,
                "best_fitness": float(pop_fitness[best]),
                "best_description": describe_genome(np.asarray(pop[best])),
                "mean_child_fitness": float(np.mean(child_fitness)),
            }
        )
        logger.info(
            "generation %d: best fitness %.5f (%s)",
            generation,
            float(pop_fitness[best]),
            generation_log[-1]["best_description"],
        )

    # --- budget-matched champion-form baseline (structure frozen, constants tuned).
    tuned_baseline, tuned_baseline_search_accuracy, baseline_evals = (
        tune_champion_baseline(
            key_baseline,
            task_names=task_names,
            eval_seeds=eval_seeds,
            batch_size=batch_size,
            suite=registry,
        )
    )
    for genome_row, accuracy_value in baseline_evals:
        archive_genomes.append(genome_row)
        archive_accuracy.append(accuracy_value)
        archive_fitness.append(
            float(penalized_fitness(np.asarray([accuracy_value]), genome_row[None, :])[0])
        )
        archive_origin.append("baseline_tuning")
    logger.info(
        "tuned champion-form baseline: search accuracy %.5f (%s)",
        tuned_baseline_search_accuracy,
        describe_genome(tuned_baseline),
    )

    # --- selection validation on held-out tasks and seeds.
    archive_matrix = np.stack(archive_genomes)
    archive_fit = np.asarray(archive_fitness)
    unique: dict[bytes, int] = {}
    for raw_index in np.argsort(-archive_fit):
        index = int(raw_index)
        digest = archive_matrix[index].tobytes()
        if digest not in unique:
            unique[digest] = index
        if len(unique) >= top_k:
            break
    top_indices = list(unique.values())
    candidates = jnp.asarray(np.stack([archive_matrix[i] for i in top_indices]))
    reference = jnp.asarray(champion_form_genome())[None, :]
    baseline = jnp.asarray(tuned_baseline)[None, :]
    holdout_pool = jnp.concatenate([baseline, reference, candidates], axis=0)
    holdout_mean, holdout_per_task = evaluate_suite(
        holdout_pool, holdout_names, seeds=holdout_seeds,
        batch_size=batch_size, suite=registry,
    )
    baseline_holdout = float(holdout_mean[0])
    reference_holdout = float(holdout_mean[1])
    n_base_rows = 2

    candidate_rows: list[dict[str, Any]] = []
    for rank, archive_index in enumerate(top_indices):
        row = {
            "rank_by_search_fitness": rank,
            "genome": [float(v) for v in archive_matrix[archive_index]],
            "config": decode_genome(archive_matrix[archive_index]),
            "description": describe_genome(archive_matrix[archive_index]),
            "origin": archive_origin[archive_index],
            "active_flags": flag_count(archive_matrix[archive_index]),
            "search_fitness": float(archive_fit[archive_index]),
            "search_accuracy": float(archive_accuracy[archive_index]),
            "holdout_accuracy": float(holdout_mean[n_base_rows + rank]),
            "holdout_per_task": {
                name: float(values[n_base_rows + rank])
                for name, values in holdout_per_task.items()
            },
            "beats_baseline_on_holdout": bool(
                float(holdout_mean[n_base_rows + rank]) > baseline_holdout
            ),
        }
        candidate_rows.append(row)
    promoted = sorted(
        (row for row in candidate_rows if row["beats_baseline_on_holdout"]),
        key=lambda row: -float(row["holdout_accuracy"]),
    )[:3]

    gauss_lane = any(
        isinstance(registry[name], MicroStreamConfig)
        for name in list(task_names) + list(holdout_names)
    )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "micro_suite_version": (
            MICRO_GAUSS_SUITE_VERSION if gauss_lane else MICRO_SUITE_VERSION
        ),
        "settings": {
            "n_random": n_random,
            "population": population,
            "generations": generations,
            "elite": elite,
            "eval_seeds": list(eval_seeds),
            "holdout_seeds": list(holdout_seeds),
            "top_k": top_k,
            "batch_size": batch_size,
            "search_seed": search_seed,
            "task_names": list(task_names),
            "holdout_names": list(holdout_names),
            "flag_penalty": FLAG_PENALTY,
            "suite_geometry": {
                name: _suite_geometry(registry[name])
                for name in list(task_names) + list(holdout_names)
            },
        },
        "n_evaluated": len(archive_genomes),
        "baseline": {
            "kind": "tuned_champion_form (structure frozen, constants budget-matched)",
            "genome": [float(v) for v in tuned_baseline],
            "config": decode_genome(tuned_baseline),
            "description": describe_genome(tuned_baseline),
            "search_accuracy": tuned_baseline_search_accuracy,
            "holdout_accuracy": baseline_holdout,
            "holdout_per_task": {
                name: float(values[0]) for name, values in holdout_per_task.items()
            },
        },
        "champion_constants_reference": {
            "description": describe_genome(champion_form_genome()),
            "holdout_accuracy": reference_holdout,
            "holdout_per_task": {
                name: float(values[1]) for name, values in holdout_per_task.items()
            },
        },
        "generation_log": generation_log,
        "candidates": candidate_rows,
        "promoted": promoted,
        "environment": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "wall_clock_seconds": round(time.monotonic() - started, 2),
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``search`` runs the discovery loop and writes one result JSON."""
    parser = argparse.ArgumentParser(description="update-rule discovery search")
    sub = parser.add_subparsers(dest="command", required=True)
    search_p = sub.add_parser("search", help="run random + evolutionary search")
    search_p.add_argument("--out", type=Path, required=True)
    search_p.add_argument("--n-random", type=int, default=3072)
    search_p.add_argument("--population", type=int, default=256)
    search_p.add_argument("--generations", type=int, default=12)
    search_p.add_argument("--elite", type=int, default=32)
    search_p.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1])
    search_p.add_argument(
        "--holdout-seeds", type=int, nargs="+", default=[101, 102, 103]
    )
    search_p.add_argument("--top-k", type=int, default=12)
    search_p.add_argument("--batch-size", type=int, default=256)
    search_p.add_argument("--search-seed", type=int, default=0)
    search_p.add_argument(
        "--suite", choices=("digits", "gauss"), default="digits",
        help=(
            "fitness suite: the provisional digits tasks (M1..M1p) or the "
            "transfer-validated gauss-v1 streams (G1/G3 search, G4/G1p holdout)"
        ),
    )
    search_p.add_argument("--tasks", nargs="+", default=None)
    search_p.add_argument("--holdout-tasks", nargs="+", default=None)
    search_p.add_argument(
        "--micro-n-tasks", type=int, default=None,
        help=(
            "override every micro task's n_tasks/n_regimes (gauss-lane "
            "budget knob per SUITE.md; smoke-only on the digits lane)"
        ),
    )
    search_p.add_argument(
        "--micro-task-length", type=int, default=None,
        help="override every micro task's task/regime length (smoke runs only)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    if args.command == "search":
        suite = _resolved_suite(args.micro_n_tasks, args.micro_task_length, args.suite)
        if args.suite == "gauss":
            default_tasks, default_holdout = GAUSS_SEARCH_TASKS, GAUSS_HOLDOUT_TASKS
        else:
            default_tasks, default_holdout = SEARCH_TASKS, HOLDOUT_TASKS
        tasks = tuple(args.tasks) if args.tasks else default_tasks
        holdout_tasks = (
            tuple(args.holdout_tasks) if args.holdout_tasks else default_holdout
        )
        payload = run_search(
            n_random=args.n_random,
            population=args.population,
            generations=args.generations,
            elite=args.elite,
            eval_seeds=tuple(args.eval_seeds),
            holdout_seeds=tuple(args.holdout_seeds),
            top_k=args.top_k,
            batch_size=args.batch_size,
            search_seed=args.search_seed,
            task_names=tasks,
            holdout_names=holdout_tasks,
            suite=suite,
        )
        _atomic_write_json(args.out, payload)
        logger.info(
            "search complete: %d evaluated, %d promoted -> %s",
            payload["n_evaluated"], len(payload["promoted"]), args.out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
