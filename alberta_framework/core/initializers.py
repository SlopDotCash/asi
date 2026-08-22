"""Weight initializers for neural networks.

Implements sparse initialization following Elsayed et al. 2024
("Streaming Deep Reinforcement Learning Finally Works").
"""

import math
import operator
from fractions import Fraction
from typing import SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar_with_ratio,
)

_INT32_MAX = 2_147_483_647
_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    }
)
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, Fraction, *(np.dtype(c).type for c in ("e", "f", "d", "g"))}
)
_ALLOWED_REAL_TYPES = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _require_int(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _require_float32_in_unit(name: str, value: object) -> float:
    if type(value) not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{name} must be a real number")
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name, value
    )
    if not math.isfinite(stored):
        raise ValueError(f"{name} must be finite")
    if stored < 0.0 or stored > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    if numerator != 0 and abs(numerator) * (1 << 149) <= denominator:
        raise ValueError(f"{name} is subnormal and would flush to zero")
    return stored


def _require_init_type(value: object) -> str:
    if type(value) is not str:
        raise ValueError("init_type must be 'uniform' or 'normal'")
    if value not in ("uniform", "normal"):
        raise ValueError("init_type must be 'uniform' or 'normal'")
    return value


def sparse_init(
    key: Array,
    shape: tuple[int, int],
    sparsity: float = 0.9,
    init_type: str = "uniform",
) -> Float[Array, "fan_out fan_in"]:
    """Create a sparsely initialized weight matrix.

    Applies the paper-specified SparseInit distribution and then zeros out
    ``ceil(sparsity * fan_in)`` inputs per output neuron. This creates sparser
    gradient flows that improve stability in streaming learning settings.

    Reference: Elsayed, Vasan, and Mahmood 2024, "Streaming Deep Reinforcement
    Learning Finally Works" (arXiv:2410.14606v2), Algorithm 1 (SparseInit), and
    the authors' reference ``sparse_init.py`` at commit ``40bd4a61``
    (https://github.com/mohmdelsayed/streaming-drl/blob/40bd4a61/sparse_init.py).

    Args:
        key: JAX random key
        shape: Weight matrix shape (fan_out, fan_in)
        sparsity: Fraction of input connections to zero out per output neuron.
            The zeroed count is ``ceil(sparsity * fan_in)``, exactly as in the
            reference implementation (default: 0.9 means 90% sparse)
        init_type: Initialization distribution, "uniform" or "normal"
            (default: the uniform bound specified by SparseInit Algorithm 1)

    Returns:
        Weight matrix of given shape with specified sparsity

    Raises:
        ValueError: If shape is not a two-element tuple or list, dimensions are not
            positive integers, sparsity is not a finite real number in [0.0, 1.0], or
            init_type is not 'uniform' or 'normal'.

    Examples:
    ```python
    import jax.random as jr
    from alberta_framework.core.initializers import sparse_init

    key = jr.key(42)
    weights = sparse_init(key, (128, 10), sparsity=0.9)
    # weights has shape (128, 10), ~90% zeros per row
    ```
    """
    if type(shape) not in (tuple, list) or len(shape) != 2:
        raise ValueError("shape must be a two-element tuple or list (fan_out, fan_in)")
    # Extract without invoking hostile __iter__ hooks beyond the exact tuple/list check
    try:
        fan_out_raw, fan_in_raw = shape
    except Exception as error:
        raise ValueError("shape must be a two-element tuple or list (fan_out, fan_in)") from error
    fan_out = _require_int("shape fan_out", fan_out_raw, minimum=1)
    fan_in = _require_int("shape fan_in", fan_in_raw, minimum=1)
    if fan_out > _INT32_MAX or fan_in > _INT32_MAX:
        raise ValueError("shape dimensions must fit signed int32")
    # Allocation-free preflight: fan_out * fan_in * 4 bytes must fit int32
    if fan_out > _INT32_MAX // max(1, fan_in) or 4 * fan_out * fan_in > _INT32_MAX:
        raise ValueError("shape byte count must fit signed int32")
    sparsity_validated = _require_float32_in_unit("sparsity", sparsity)
    # SparseInit zeros ``ceil(sparsity * fan_in)`` inputs per output neuron:
    # Elsayed et al. 2024 (arXiv:2410.14606v2) Algorithm 1 takes ``n <- s x
    # fan_in`` as a count, and the authors' reference implementation resolves it
    # as ``num_zeros = int(math.ceil(sparsity * fan_in))``.  Evaluated in the same
    # binary64 arithmetic as the reference so a decimal literal such as 0.9 gives
    # the published count rather than one more.
    num_zeros = math.ceil(sparsity_validated * fan_in)

    # Split key for init and sparsity mask
    init_key, mask_key = jr.split(key)

    # SparseInit Algorithm 1 uses this exact uniform bound; its variance is
    # deliberately not JAX's canonical ``lecun_uniform`` variance.
    init_type_checked = _require_init_type(init_type)
    scale = 1.0 / fan_in**0.5
    if init_type_checked == "uniform":
        weights = jr.uniform(
            init_key, (fan_out, fan_in), dtype=jnp.float32, minval=-scale, maxval=scale
        )
    elif init_type_checked == "normal":
        weights = jr.normal(init_key, (fan_out, fan_in), dtype=jnp.float32) * scale
    else:
        raise ValueError("init_type must be 'uniform' or 'normal'")

    if num_zeros <= 0:
        return weights
    if num_zeros >= fan_in:
        return jnp.zeros_like(weights)

    # Exact per-row sparsity without per-row random permutations.  `jr.permutation`
    # lowers to a shuffle kernel that can be very slow to compile in large suites.
    scores = jr.uniform(mask_key, (fan_out, fan_in), dtype=jnp.float32)
    zero_idx = jax.lax.top_k(scores, num_zeros)[1]
    row_idx = jnp.arange(fan_out)[:, None]
    masks = jnp.ones((fan_out, fan_in), dtype=jnp.float32).at[row_idx, zero_idx].set(0.0)

    return weights * masks
