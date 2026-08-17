"""Hostile validation for streams/feature_discovery sink gates."""

import numpy as np
import pytest

from alberta_framework.streams.feature_discovery import (
    InteractionFeatureDiscoveryStream,
    NonlinearFeatureDiscoveryStream,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat hook must not leak via !r")

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat hook must not leak via !r")


class _HostileInt(int):
    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("HostileInt.__int__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


def test_rejects_string_subclass_for_feature_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        NonlinearFeatureDiscoveryStream(feature_dim=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_feature_dim_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        NonlinearFeatureDiscoveryStream(feature_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)


def test_rejects_bool_and_hostile_int_for_feature_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        NonlinearFeatureDiscoveryStream(feature_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        NonlinearFeatureDiscoveryStream(feature_dim=np.bool_(True))  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer"):
        NonlinearFeatureDiscoveryStream(feature_dim=_HostileInt(4))
    assert _HostileInt.calls == 0


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must narrow to a finite float32") as exc:
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=_HostileFloat(1.0))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)


def test_rejects_plain_string_for_feature_std() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std="1.0")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_positive_real() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=_StringSubclass("1.0"))  # type: ignore[arg-type]


def test_rejects_nonpositive_feature_std_without_repr() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=0.0)
    with pytest.raises(ValueError, match="must be a real number") as exc:
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=_StringSubclass("bad"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc.value)


def test_rejects_include_squares_non_bool_without_repr() -> None:
    with pytest.raises(TypeError, match="must be a boolean") as exc:
        InteractionFeatureDiscoveryStream(feature_dim=4, include_squares=_StringSubclass("true"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil = _EvilStr("true")
    with pytest.raises(TypeError, match="must be a boolean"):
        InteractionFeatureDiscoveryStream(feature_dim=4, include_squares=evil)  # type: ignore[arg-type]


def test_valid_configs_still_pass() -> None:
    s = NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=1.0, linear_scale=0.05)
    assert s.feature_dim == 4
    s2 = InteractionFeatureDiscoveryStream(feature_dim=4, include_squares=True)
    assert s2.include_squares is True
    s3 = InteractionFeatureDiscoveryStream(feature_dim=4, include_squares=np.bool_(False))
    assert s3.include_squares is False


def test_numpy_scalars_pass() -> None:
    s = NonlinearFeatureDiscoveryStream(feature_dim=np.int32(4), feature_std=np.float32(1.0))
    assert s.feature_dim == 4
    assert s.feature_std == pytest.approx(1.0)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (3, 4)

    with pytest.raises(ValueError, match="must narrow to a finite float32"):
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=RatioFloat(0.5))  # type: ignore[arg-type]
