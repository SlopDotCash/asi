"""Deterministic gradual-transition primitives for development IPMNIST lanes.

This module adapts the transition definitions in Liu and Mou,
``arXiv:2602.09234v2``.  It does not define or execute an evidence protocol.
The transition coefficient and task identity are evaluator-owned: learners see
only the resulting example/target, exactly as in the abrupt IPMNIST lane.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed

TransitionMode = Literal[
    "abrupt", "input_interpolation", "output_interpolation", "task_sampling"
]
_MODES = frozenset(
    {"abrupt", "input_interpolation", "output_interpolation", "task_sampling"}
)
_INT32_MAX = 2**31 - 1

GRADUAL_IPMNIST_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.ipmnist.gradual-transition.protocol.v1",
        "paper_revision": "arXiv:2602.09234v2",
        "paper_revision_date": "2026-06-16",
        "adaptation_difference": (
            "single-pass ASI IPMNIST uses evaluator-owned per-example transitions; "
            "the paper iterates mini-batches within tasks without revealing boundaries"
        ),
        "matched_axes": ("seed", "updates", "observations", "example_order"),
        "learner_observes_transition_alpha": False,
        "learner_observes_task_boundary": False,
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)


def _exact_index(name: str, value: object, *, minimum: int) -> int:
    if type(value) is not int and not isinstance(value, np.integer):
        raise ValueError(f"{name} must be an integer")
    try:
        resolved = operator.index(cast(SupportsIndex, value))
    except Exception as error:
        raise ValueError(f"{name} must be an integer") from error
    if resolved < minimum or resolved > _INT32_MAX:
        raise ValueError(f"{name} must be in [{minimum}, {_INT32_MAX}]")
    return resolved


def _alpha(value: object) -> float:
    if type(value) is not float and type(value) is not int:
        raise ValueError("alpha must be a finite real number in [0, 1]")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError("alpha must be a finite real number in [0, 1]")
    return resolved


@dataclass(frozen=True)
class GradualTransitionConfig:
    """One prospectively selectable transition rule.

    ``transition_steps`` counts update opportunities from the old task at
    alpha zero to the new task at alpha one.  Abrupt mode requires one step
    and is the mechanism-off reduction.
    """

    mode: TransitionMode
    transition_steps: int

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        resolved = _exact_index("transition_steps", self.transition_steps, minimum=1)
        if self.mode == "abrupt" and resolved != 1:
            raise ValueError("abrupt mode requires transition_steps=1")
        object.__setattr__(self, "transition_steps", resolved)


def transition_alpha(step: int, config: GradualTransitionConfig) -> float:
    """Return a deterministic uniform interpolation coefficient."""
    resolved_step = _exact_index("step", step, minimum=0)
    if config.mode == "abrupt":
        return 1.0
    return min(resolved_step / config.transition_steps, 1.0)


def input_interpolation(old: Array, new: Array, alpha: float) -> Array:
    """Apply paper equation ``x_alpha = (1-alpha)x_old + alpha*x_new``."""
    resolved = _alpha(alpha)
    old_array = jnp.asarray(old)
    new_array = jnp.asarray(new)
    if old_array.shape != new_array.shape:
        raise ValueError("old and new inputs must have identical shapes")
    if old_array.dtype != new_array.dtype or not jnp.issubdtype(old_array.dtype, jnp.floating):
        raise ValueError("old and new inputs must share a floating dtype")
    return (1.0 - resolved) * old_array + resolved * new_array


def output_interpolation(
    old_label: int, new_label: int, alpha: float, *, n_classes: int
) -> Array:
    """Interpolate old one-hot -> uniform -> new one-hot as paper section 4."""
    resolved_alpha = _alpha(alpha)
    classes = _exact_index("n_classes", n_classes, minimum=2)
    old = _exact_index("old_label", old_label, minimum=0)
    new = _exact_index("new_label", new_label, minimum=0)
    if old >= classes or new >= classes:
        raise ValueError("labels must be smaller than n_classes")
    uniform = jnp.full((classes,), 1.0 / classes, dtype=jnp.float32)
    if resolved_alpha <= 0.5:
        old_target = jax.nn.one_hot(old, classes, dtype=jnp.float32)
        return (1.0 - 2.0 * resolved_alpha) * old_target + 2.0 * resolved_alpha * uniform
    new_target = jax.nn.one_hot(new, classes, dtype=jnp.float32)
    return (2.0 * resolved_alpha - 1.0) * new_target + (2.0 - 2.0 * resolved_alpha) * uniform


def task_sampling_mask(
    *, seed: int, transition_id: int, count: int, alpha: float
) -> np.ndarray:
    """Select exactly ``floor(alpha * count)`` positions from the new task.

    A Threefry root and transition fold make selection independent across
    transitions and reproducible across processes.  The mask is evaluator
    state, never learner-visible boundary information.
    """
    resolved_seed = require_jax_seed(seed, name="seed")
    resolved_transition = _exact_index("transition_id", transition_id, minimum=0)
    resolved_count = _exact_index("count", count, minimum=1)
    new_count = math.floor(_alpha(alpha) * resolved_count)
    order = np.asarray(
        jax.device_get(
            jr.permutation(jr.fold_in(jr.key(resolved_seed), resolved_transition), resolved_count)
        )
    )
    mask = np.zeros(resolved_count, dtype=np.bool_)
    mask[order[:new_count]] = True
    return mask
