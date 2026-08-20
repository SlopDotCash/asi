"""Calibrated Partial Reset arms for the prospective issue #1563 IPMNIST lane."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, Literal

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array

from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningSpec,
    ScreeningStepFn,
    StepMetrics,
    _step_metrics,
)
from alberta_framework.benchmarks.upgd_ipmnist import LearnerInitFn, cross_entropy_loss

PAPER_REVISION: Final = "arXiv:2607.24996v1"
OFFICIAL_CODE_REVISION: Final = "LucMc/continual-learning@6fc2af34783159f5dda50c6915dda32c2d443604"
ARMS: Final = (
    "cpr_off",
    "cpr_utility",
    "cpr_utility_free",
    "cpr_l2_init",
    "cpr_hard_reset",
)
_MODES: Final = ("off", "utility", "utility_free", "l2_init", "hard_reset")


@chex.dataclass(frozen=True)
class CPRState:
    """Matched Adam state, retained initialization, and per-neuron utility."""

    first_moment: dict[str, Array]
    second_moment: dict[str, Array]
    initial_params: dict[str, Array]
    utility1: Array
    utility2: Array
    step: Array


def _checked_hyperparameters(value: Mapping[str, float]) -> dict[str, float]:
    expected = {
        "step_size",
        "beta1",
        "beta2",
        "epsilon",
        "utility_decay",
        "replacement_rate",
        "sharpness",
        "update_frequency",
        "l2_init_strength",
    }
    if (
        type(value) is not dict
        or len(value) != len(expected)
        or any(type(name) is not str for name in value)
        or set(value) != expected
    ):
        raise ValueError("CPR hyperparameters differ from the exact contract")
    result: dict[str, float] = {}
    for name in sorted(expected):
        item = value[name]
        if type(item) is not float or not math.isfinite(item):
            raise ValueError("CPR hyperparameters must be exact finite floats")
        result[name] = item
    if not 0.0 < result["step_size"] <= 1.0:
        raise ValueError("CPR step size is out of bounds")
    if not 0.0 <= result["beta1"] < 1.0 or not 0.0 <= result["beta2"] < 1.0:
        raise ValueError("CPR Adam decays are out of bounds")
    if not 0.0 <= result["utility_decay"] < 1.0:
        raise ValueError("CPR utility decay is out of bounds")
    if not 0.0 <= result["replacement_rate"] <= 1.0:
        raise ValueError("CPR replacement rate is out of bounds")
    if not 0.0 < result["epsilon"] <= 1.0 or not 0.0 < result["sharpness"] <= 64.0:
        raise ValueError("CPR numeric stabilizers are out of bounds")
    frequency = result["update_frequency"]
    if not frequency.is_integer() or not 1 <= frequency <= 1_000_000:
        raise ValueError("CPR update frequency is out of bounds")
    if not 0.0 <= result["l2_init_strength"] <= 1.0:
        raise ValueError("CPR L2-init strength is out of bounds")
    return result


def _normalized_utility(gradient: Array, previous: Array, decay: float) -> Array:
    score = jnp.mean(jnp.abs(gradient), axis=0)
    normalized = score / (jnp.mean(score) + jnp.asarray(1e-8, dtype=score.dtype))
    return decay * previous + (1.0 - decay) * normalized


def _partial_reset(
    params: dict[str, Array],
    initial: dict[str, Array],
    utility1: Array,
    utility2: Array,
    *,
    replacement_rate: float,
    sharpness: float,
    mode: Literal["utility", "utility_free", "hard_reset"],
) -> dict[str, Array]:
    def rate(utility: Array) -> Array:
        if mode == "utility":
            shape = jnp.minimum(2.0 * jax.nn.sigmoid(-sharpness * (utility - 1.0)), 1.0)
            return replacement_rate * shape
        if mode == "utility_free":
            return jnp.full_like(utility, replacement_rate)
        return jnp.where(utility <= 1.0, 1.0, 0.0)

    rate1 = rate(utility1)
    rate2 = rate(utility2)
    result = dict(params)
    result["w1"] = params["w1"] + rate1[None, :] * (initial["w1"] - params["w1"])
    result["w2"] = params["w2"] * (1.0 - rate1[:, None])
    result["w2"] = result["w2"] + rate2[None, :] * (initial["w2"] - result["w2"])
    result["w3"] = params["w3"] * (1.0 - rate2[:, None])
    return result


def _make_cpr_learner(
    hp: Mapping[str, float],
    *,
    mode: Literal["off", "utility", "utility_free", "l2_init", "hard_reset"],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    if type(mode) is not str or mode not in _MODES:
        raise ValueError("unknown CPR reduction")
    checked = _checked_hyperparameters(hp)

    def init_fn(params: dict[str, Array]) -> CPRState:
        return CPRState(
            first_moment={name: jnp.zeros_like(value) for name, value in params.items()},
            second_moment={name: jnp.zeros_like(value) for name, value in params.items()},
            initial_params=dict(params),
            utility1=jnp.ones(params["w1"].shape[1], dtype=jnp.float32),
            utility2=jnp.ones(params["w2"].shape[1], dtype=jnp.float32),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def step_fn(
        params: dict[str, Array], state: CPRState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], CPRState, StepMetrics]:
        del key
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(params, x, y)
        step = state.step + jnp.asarray(1, dtype=jnp.int32)
        first = {
            name: checked["beta1"] * state.first_moment[name]
            + (1.0 - checked["beta1"]) * grads[name]
            for name in params
        }
        second = {
            name: checked["beta2"] * state.second_moment[name]
            + (1.0 - checked["beta2"]) * jnp.square(grads[name])
            for name in params
        }
        correction1 = 1.0 - jnp.power(checked["beta1"], step.astype(jnp.float32))
        correction2 = 1.0 - jnp.power(checked["beta2"], step.astype(jnp.float32))
        updated = {
            name: params[name]
            - checked["step_size"]
            * (first[name] / correction1)
            / (jnp.sqrt(second[name] / correction2) + checked["epsilon"])
            for name in params
        }
        utility1 = _normalized_utility(grads["w1"], state.utility1, checked["utility_decay"])
        utility2 = _normalized_utility(grads["w2"], state.utility2, checked["utility_decay"])
        if mode == "l2_init":
            updated = {
                name: value + checked["l2_init_strength"] * (state.initial_params[name] - value)
                for name, value in updated.items()
            }
        elif mode != "off":
            should_reset = step % int(checked["update_frequency"]) == 0
            reset = _partial_reset(
                updated,
                state.initial_params,
                utility1,
                utility2,
                replacement_rate=checked["replacement_rate"],
                sharpness=checked["sharpness"],
                mode=mode,
            )
            updated = {
                name: jnp.where(should_reset, reset[name], value) for name, value in updated.items()
            }
            utility1 = jnp.where(should_reset, jnp.ones_like(utility1), utility1)
            utility2 = jnp.where(should_reset, jnp.ones_like(utility2), utility2)
        new_state = CPRState(
            first_moment=first,
            second_moment=second,
            initial_params=state.initial_params,
            utility1=utility1,
            utility2=utility2,
            step=step,
        )
        return updated, new_state, _step_metrics(updated, x, y, loss, logits)

    return init_fn, step_fn


def _make_off(hp: Mapping[str, float]) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_cpr_learner(hp, mode="off")


def _make_utility(hp: Mapping[str, float]) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_cpr_learner(hp, mode="utility")


def _make_utility_free(hp: Mapping[str, float]) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_cpr_learner(hp, mode="utility_free")


def _make_l2_init(hp: Mapping[str, float]) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_cpr_learner(hp, mode="l2_init")


def _make_hard_reset(hp: Mapping[str, float]) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_cpr_learner(hp, mode="hard_reset")


def _hyperparameters() -> dict[str, float]:
    return {
        "step_size": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "utility_decay": 0.99,
        "replacement_rate": 0.01,
        "sharpness": 16.0,
        "update_frequency": 1000.0,
        "l2_init_strength": 1e-5,
    }


_FACTORIES: Final = {
    "cpr_off": _make_off,
    "cpr_utility": _make_utility,
    "cpr_utility_free": _make_utility_free,
    "cpr_l2_init": _make_l2_init,
    "cpr_hard_reset": _make_hard_reset,
}


def calibrated_partial_reset_spec(name: str) -> ScreeningSpec:
    if type(name) is not str or name not in ARMS:
        raise ValueError("unknown calibrated partial reset arm")
    return ScreeningSpec(
        name=name,
        base_learner="adamw",
        mechanism="calibrated_partial_reset",
        hyperparameters=_hyperparameters(),
        factory=_FACTORIES[name],
        description=f"CPR matched-development reduction: {name}",
    )


def persistent_numeric_bytes(*, input_dim: int, hidden1: int, hidden2: int, n_classes: int) -> int:
    values = input_dim * hidden1 + hidden1 + hidden1 * hidden2 + hidden2
    values += hidden2 * n_classes + n_classes
    # Parameters + retained initialization + Adam first/second moments.
    values *= 4
    # Two per-neuron utility vectors plus one int32 step.
    values += hidden1 + hidden2 + 1
    return values * 4
