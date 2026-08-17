"""Constructor scalar contracts for Step 1 baseline optimizers.

AdaGain, Adam, RMSprop, and NADALINE persist constructor scalars into
``init`` / ``init_for_shape`` state. A NaN, Inf, or bool must not construct:
IEEE-blind assignment stores the raw value, and ``jnp.array`` then writes
NaN / Inf / ``1.0`` into every later update.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.baseline_optimizers import NADALINE, AdaGain, Adam, RMSprop


class _HostileFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
        raise RuntimeError("untrusted ratio hook executed")


class _ClassSpoof:
    @property
    def __class__(self):  # type: ignore[no-untyped-def]
        return float

    def __float__(self) -> float:  # pragma: no cover
        return 0.1


@pytest.mark.parametrize(
    ("ctor", "field", "bad"),
    [
        (lambda v: AdaGain(initial_step_size=v), "initial_step_size", float("nan")),
        (lambda v: AdaGain(initial_step_size=v), "initial_step_size", float("inf")),
        (lambda v: AdaGain(initial_step_size=v), "initial_step_size", 0.0),
        (lambda v: AdaGain(initial_step_size=v), "initial_step_size", -0.1),
        (lambda v: AdaGain(initial_step_size=v), "initial_step_size", True),
        (lambda v: AdaGain(initial_step_size=v), "initial_step_size", _HostileFloat(0.05)),
        (lambda v: AdaGain(initial_step_size=v), "initial_step_size", _ClassSpoof()),
        (lambda v: AdaGain(meta_step_size=v), "meta_step_size", float("nan")),
        (lambda v: AdaGain(meta_step_size=v), "meta_step_size", float("-inf")),
        (lambda v: AdaGain(meta_step_size=v), "meta_step_size", -0.1),
        (lambda v: AdaGain(meta_step_size=v), "meta_step_size", True),
        (lambda v: AdaGain(forgetting_rate=v), "forgetting_rate", float("nan")),
        (lambda v: AdaGain(forgetting_rate=v), "forgetting_rate", -0.1),
        (lambda v: AdaGain(forgetting_rate=v), "forgetting_rate", 1.1),
        (lambda v: AdaGain(forgetting_rate=v), "forgetting_rate", True),
        (lambda v: Adam(step_size=v), "step_size", float("nan")),
        (lambda v: Adam(step_size=v), "step_size", 0.0),
        (lambda v: Adam(step_size=v), "step_size", True),
        (lambda v: Adam(beta1=v), "beta1", 1.0),
        (lambda v: Adam(beta1=v), "beta1", -0.1),
        (lambda v: Adam(beta1=v), "beta1", True),
        (lambda v: Adam(beta2=v), "beta2", 1.0),
        (lambda v: Adam(beta2=v), "beta2", float("inf")),
        (lambda v: Adam(eps=v), "eps", 0.0),
        (lambda v: Adam(eps=v), "eps", float("inf")),
        (lambda v: Adam(eps=v), "eps", True),
        (lambda v: Adam(weight_decay=v), "weight_decay", -0.1),
        (lambda v: Adam(weight_decay=v), "weight_decay", float("nan")),
        (lambda v: Adam(weight_decay=v), "weight_decay", True),
        (lambda v: RMSprop(step_size=v), "step_size", float("nan")),
        (lambda v: RMSprop(step_size=v), "step_size", 0.0),
        (lambda v: RMSprop(decay=v), "decay", 1.1),
        (lambda v: RMSprop(decay=v), "decay", True),
        (lambda v: RMSprop(eps=v), "eps", float("inf")),
        (lambda v: NADALINE(step_size=v), "step_size", float("nan")),
        (lambda v: NADALINE(step_size=v), "step_size", True),
        (lambda v: NADALINE(decay=v), "decay", -0.1),
        (lambda v: NADALINE(eps=v), "eps", 0.0),
    ],
)
def test_baseline_optimizer_ctors_reject_illegal_scalars(ctor, field: str, bad: object) -> None:
    with pytest.raises(ValueError, match=field):
        ctor(bad)


def test_legal_zero_and_unit_scalars_remain_constructible() -> None:
    adagain = AdaGain(initial_step_size=0.05, meta_step_size=0.0, forgetting_rate=1.0)
    assert adagain.to_config()["meta_step_size"] == 0.0
    assert adagain.to_config()["forgetting_rate"] == 1.0

    adam = Adam(step_size=0.01, beta1=0.0, beta2=0.0, weight_decay=0.0)
    assert adam.to_config()["beta1"] == 0.0
    assert adam.to_config()["weight_decay"] == 0.0

    rms = RMSprop(step_size=0.01, decay=0.0)
    assert rms.to_config()["decay"] == 0.0

    nadal = NADALINE(step_size=0.01, decay=0.0)
    assert nadal.to_config()["decay"] == 0.0


@pytest.mark.parametrize("np_type", [np.float32, np.float64])
def test_baseline_optimizer_ctors_canonicalize_numpy_floats(np_type: type) -> None:
    opt = AdaGain(
        initial_step_size=np_type(0.05),
        meta_step_size=np_type(0.0),
        forgetting_rate=np_type(1.0),
    )
    assert opt.to_config()["initial_step_size"] == pytest.approx(0.05)
    assert type(opt.to_config()["initial_step_size"]) is float
