# mypy: disable-error-code="call-arg,misc,no-untyped-def,unused-ignore"
"""Tests for GVF types: DemonType, GVFSpec, HordeSpec, create_horde_spec."""

from fractions import Fraction

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework import (
    DemonType,
    GVFSpec,
    HordeSpec,
    create_horde_spec,
)


class TestDemonType:
    """Tests for DemonType enum."""

    def test_prediction_value(self):
        assert DemonType.PREDICTION.value == "prediction"

    def test_control_value(self):
        assert DemonType.CONTROL.value == "control"

    def test_from_string(self):
        assert DemonType("prediction") is DemonType.PREDICTION
        assert DemonType("control") is DemonType.CONTROL


class TestGVFSpec:
    """Tests for GVFSpec construction and serialization."""

    def test_basic_construction(self):
        spec = GVFSpec(
            name="is_malicious",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=0,
        )
        assert spec.name == "is_malicious"
        assert spec.demon_type is DemonType.PREDICTION
        assert spec.gamma == 0.0
        assert spec.lamda == 0.0
        assert spec.cumulant_index == 0
        assert spec.terminal_reward == 0.0  # default

    def test_temporal_demon(self):
        spec = GVFSpec(
            name="future_attacks",
            demon_type=DemonType.PREDICTION,
            gamma=0.9,
            lamda=0.8,
            cumulant_index=1,
            terminal_reward=0.0,
        )
        assert spec.gamma == 0.9
        assert spec.lamda == 0.8

    def test_config_roundtrip(self):
        original = GVFSpec(
            name="attack_type",
            demon_type=DemonType.PREDICTION,
            gamma=0.95,
            lamda=0.5,
            cumulant_index=2,
            terminal_reward=1.0,
        )
        config = original.to_config()
        restored = GVFSpec.from_config(config)

        assert restored.name == original.name
        assert restored.demon_type is original.demon_type
        assert restored.gamma == original.gamma
        assert restored.lamda == original.lamda
        assert restored.cumulant_index == original.cumulant_index
        assert restored.terminal_reward == original.terminal_reward

    def test_config_format(self):
        spec = GVFSpec(
            name="test",
            demon_type=DemonType.CONTROL,
            gamma=0.99,
            lamda=0.0,
            cumulant_index=-1,
        )
        config = spec.to_config()
        assert config["demon_type"] == "control"
        assert config["name"] == "test"
        assert config["gamma"] == 0.99

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("gamma", -0.01),
            ("gamma", 1.01),
            ("gamma", float("nan")),
            ("gamma", float("inf")),
            ("gamma", float("-inf")),
            ("lamda", -0.01),
            ("lamda", 1.01),
            ("lamda", float("nan")),
            ("lamda", float("inf")),
            ("lamda", float("-inf")),
        ],
    )
    def test_invalid_discount_or_trace_decay_is_rejected(self, field, value):
        kwargs = {
            "name": "invalid",
            "demon_type": DemonType.PREDICTION,
            "gamma": 0.9,
            "lamda": 0.8,
            "cumulant_index": 0,
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=field):
            GVFSpec(**kwargs)

        config = {
            "name": "invalid",
            "demon_type": "prediction",
            "gamma": 0.9,
            "lamda": 0.8,
            "cumulant_index": 0,
            "terminal_reward": 0.0,
        }
        config[field] = value
        with pytest.raises(ValueError, match=field):
            GVFSpec.from_config(config)

    @pytest.mark.parametrize(
        "value",
        [
            True,
            False,
            "0.5",
            None,
            10**400,
            1.0e-50,
            Fraction(1, 10**1000),
            Fraction((2**100) + 1, 2**100),
            np.nextafter(np.longdouble(1.0), np.longdouble(2.0)),
            jnp.asarray(0.5),
            jnp.asarray([0.5]),
        ],
    )
    @pytest.mark.parametrize("field", ["gamma", "lamda"])
    def test_discount_and_trace_decay_require_concrete_float32_reals(
        self,
        field,
        value,
    ):
        kwargs = {
            "name": "invalid",
            "demon_type": DemonType.PREDICTION,
            "gamma": 0.9,
            "lamda": 0.8,
            "cumulant_index": 0,
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=field):
            GVFSpec(**kwargs)

        config = {
            **kwargs,
            "demon_type": "prediction",
            "terminal_reward": 0.0,
        }
        with pytest.raises(ValueError, match=field):
            GVFSpec.from_config(config)

    @pytest.mark.parametrize("field", ["gamma", "lamda"])
    def test_discount_and_trace_decay_reject_class_spoofed_float(self, field):
        class _SpoofedFloat:
            """Mimics ``float`` via ``__class__`` to defeat ``isinstance``."""

            @property
            def __class__(self) -> type:  # type: ignore[override]
                return float

            def __float__(self) -> float:
                return 0.5

            def __lt__(self, other: object) -> bool:
                return 0.5 < other  # type: ignore[operator]

            def __gt__(self, other: object) -> bool:
                return 0.5 > other  # type: ignore[operator]

            def __eq__(self, other: object) -> bool:
                return 0.5 == other

            def __ne__(self, other: object) -> bool:
                return 0.5 != other

            def __hash__(self) -> int:
                return hash(0.5)

        kwargs = {
            "name": "invalid",
            "demon_type": DemonType.PREDICTION,
            "gamma": 0.9,
            "lamda": 0.8,
            "cumulant_index": 0,
        }
        kwargs[field] = _SpoofedFloat()

        with pytest.raises(ValueError, match=field):
            GVFSpec(**kwargs)

    def test_discount_and_trace_decay_normalize_supported_real_scalars(self):
        spec = GVFSpec(
            name="boundary",
            demon_type=DemonType.PREDICTION,
            gamma=np.float64(0.0),
            lamda=np.int64(1),
            cumulant_index=0,
        )

        assert type(spec.gamma) is float
        assert type(spec.lamda) is float
        horde = create_horde_spec([spec])
        chex.assert_trees_all_equal(horde.gammas, jnp.asarray([0.0], dtype=jnp.float32))
        chex.assert_trees_all_equal(horde.lamdas, jnp.asarray([1.0], dtype=jnp.float32))


class TestHordeSpec:
    """Tests for HordeSpec construction and serialization."""

    def test_create_horde_spec(self):
        demons = [
            GVFSpec(
                name="d0", demon_type=DemonType.PREDICTION, gamma=0.0, lamda=0.0, cumulant_index=0
            ),
            GVFSpec(
                name="d1", demon_type=DemonType.PREDICTION, gamma=0.9, lamda=0.8, cumulant_index=1
            ),
            GVFSpec(
                name="d2", demon_type=DemonType.PREDICTION, gamma=0.0, lamda=0.0, cumulant_index=2
            ),
        ]
        spec = create_horde_spec(demons)

        assert len(spec.demons) == 3
        chex.assert_shape(spec.gammas, (3,))
        chex.assert_shape(spec.lamdas, (3,))

        # Check pre-computed arrays
        chex.assert_trees_all_close(spec.gammas, jnp.array([0.0, 0.9, 0.0]))
        chex.assert_trees_all_close(spec.lamdas, jnp.array([0.0, 0.8, 0.0]))

    def test_config_roundtrip(self):
        demons = [
            GVFSpec(
                name="d0", demon_type=DemonType.PREDICTION, gamma=0.0, lamda=0.0, cumulant_index=0
            ),
            GVFSpec(
                name="d1", demon_type=DemonType.PREDICTION, gamma=0.95, lamda=0.5, cumulant_index=1
            ),
        ]
        original = create_horde_spec(demons)
        config = original.to_config()
        restored = HordeSpec.from_config(config)

        assert len(restored.demons) == 2
        assert restored.demons[0].name == "d0"
        assert restored.demons[1].gamma == 0.95
        chex.assert_trees_all_close(restored.gammas, original.gammas)
        chex.assert_trees_all_close(restored.lamdas, original.lamdas)

    def test_rlsecd_5_head_spec(self):
        """Validate rlsecd's 5-head configuration as GVF demons.

        All heads are single-step prediction demons (gamma=0, pi=behavior).
        """
        rlsecd_demons = [
            GVFSpec(
                name="is_malicious",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),
            GVFSpec(
                name="attack_type",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=1,
            ),
            GVFSpec(
                name="severity",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=2,
            ),
            GVFSpec(
                name="confidence",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=3,
            ),
            GVFSpec(
                name="action_quality",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=4,
            ),
        ]
        spec = create_horde_spec(rlsecd_demons)

        assert len(spec.demons) == 5
        # All gammas should be 0 (single-step prediction)
        chex.assert_trees_all_close(spec.gammas, jnp.zeros(5))
        chex.assert_trees_all_close(spec.lamdas, jnp.zeros(5))

        # All should be prediction demons
        for d in spec.demons:
            assert d.demon_type is DemonType.PREDICTION

    def test_single_demon(self):
        spec = create_horde_spec(
            [
                GVFSpec(
                    name="only",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.99,
                    lamda=0.9,
                    cumulant_index=0,
                ),
            ]
        )
        assert len(spec.demons) == 1
        chex.assert_shape(spec.gammas, (1,))

    def test_demons_are_tuple(self):
        """Demons should be stored as tuple for immutability."""
        demons = [
            GVFSpec(
                name="d0", demon_type=DemonType.PREDICTION, gamma=0.0, lamda=0.0, cumulant_index=0
            ),
        ]
        spec = create_horde_spec(demons)
        assert isinstance(spec.demons, tuple)


class TestGVFSpecRemainingFields:
    """Name, cumulant index, terminal reward, and empty hordes must fail closed."""

    def test_legal_defaults_stay_bit_identical(self):
        spec = GVFSpec(
            name="d0",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=0,
        )
        assert spec.name == "d0"
        assert spec.cumulant_index == 0
        assert type(spec.cumulant_index) is int
        assert spec.terminal_reward == 0.0
        assert type(spec.terminal_reward) is float

    def test_negative_terminal_reward_is_a_legal_pseudo_reward(self):
        spec = GVFSpec(
            name="z",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=-1,
            terminal_reward=-1.0,
        )
        assert spec.cumulant_index == -1
        assert spec.terminal_reward == -1.0
        assert spec.to_config()["terminal_reward"] == -1.0

    @pytest.mark.parametrize("value", [2.0**-150, -(2.0**-150), 5e-324, -5e-324])
    def test_terminal_reward_rejects_nonzero_float32_underflow(self, value):
        with pytest.raises(ValueError, match="terminal_reward must remain nonzero"):
            GVFSpec(
                name="d0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
                terminal_reward=value,
            )

    @pytest.mark.parametrize("value", [0.0, 2.0**-149, -(2.0**-149)])
    def test_terminal_reward_preserves_zero_and_float32_minsubnormal(self, value):
        spec = GVFSpec(
            name="d0",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=0,
            terminal_reward=value,
        )
        assert spec.terminal_reward == value

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), True, False, "0.0", None],
    )
    def test_terminal_reward_rejects_non_finite_and_non_real_identities(self, value):
        with pytest.raises(ValueError, match="terminal_reward"):
            GVFSpec(
                name="d0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
                terminal_reward=value,
            )
        with pytest.raises(ValueError, match="terminal_reward"):
            GVFSpec.from_config(
                {
                    "name": "d0",
                    "demon_type": "prediction",
                    "gamma": 0.0,
                    "lamda": 0.0,
                    "cumulant_index": 0,
                    "terminal_reward": value,
                }
            )

    def test_terminal_reward_rejects_class_spoofed_float(self):
        class _SpoofedFloat:
            @property
            def __class__(self) -> type:  # type: ignore[override]
                return float

            def __float__(self) -> float:
                raise RuntimeError("must not run")

        with pytest.raises(ValueError, match="terminal_reward"):
            GVFSpec(
                name="d0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
                terminal_reward=_SpoofedFloat(),
            )

    @pytest.mark.parametrize(
        "value",
        [True, False, 1.5, float("nan"), "0", None, 2**31],
    )
    def test_cumulant_index_rejects_bool_float_and_out_of_range(self, value):
        with pytest.raises(ValueError, match="cumulant_index"):
            GVFSpec(
                name="d0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=value,
            )

    def test_true_cumulant_index_does_not_silently_select_channel_one(self):
        with pytest.raises(ValueError, match="cumulant_index"):
            GVFSpec(
                name="d0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=True,
            )

    def test_numpy_cumulant_index_canonicalizes_to_int(self):
        spec = GVFSpec(
            name="d0",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=np.int64(2),
        )
        assert spec.cumulant_index == 2
        assert type(spec.cumulant_index) is int

    @pytest.mark.parametrize("value", ["", None, True, 0])
    def test_name_must_be_a_nonempty_string(self, value):
        with pytest.raises(ValueError, match="name"):
            GVFSpec(
                name=value,
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            )

    def test_demon_type_must_be_the_enum(self):
        with pytest.raises(ValueError, match="demon_type"):
            GVFSpec(
                name="d0",
                demon_type="prediction",
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            )

    def test_create_horde_spec_rejects_empty_and_non_gvf_items(self):
        with pytest.raises(ValueError, match="nonempty"):
            create_horde_spec([])
        with pytest.raises(ValueError, match="GVFSpec"):
            create_horde_spec(["d0"])  # type: ignore[list-item]

    def test_from_config_rejects_empty_horde(self):
        with pytest.raises(ValueError, match="nonempty"):
            HordeSpec.from_config({"demons": []})

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "not a dict",
            123,
            [
                ("name", "d0"),
                ("demon_type", "prediction"),
                ("gamma", 0.0),
                ("lamda", 0.0),
                ("cumulant_index", 0),
                ("terminal_reward", 0.0),
            ],
            (1, 2, 3),
        ],
    )
    def test_gvf_spec_from_config_rejects_non_dict(self, payload):
        with pytest.raises(ValueError, match="GVFSpec config must be an exact dict"):
            GVFSpec.from_config(payload)

    def test_gvf_spec_from_config_rejects_non_string_keys(self):
        payload = {
            1: "d0",
            "demon_type": "prediction",
            "gamma": 0.0,
            "lamda": 0.0,
            "cumulant_index": 0,
            "terminal_reward": 0.0,
        }
        with pytest.raises(ValueError, match="keys must be exact strings"):
            GVFSpec.from_config(payload)

    @pytest.mark.parametrize(
        "missing_key",
        ["name", "demon_type", "gamma", "lamda", "cumulant_index", "terminal_reward"],
    )
    def test_gvf_spec_from_config_rejects_missing_fields(self, missing_key):
        payload = {
            "name": "d0",
            "demon_type": "prediction",
            "gamma": 0.0,
            "lamda": 0.0,
            "cumulant_index": 0,
            "terminal_reward": 0.0,
        }
        del payload[missing_key]
        with pytest.raises(ValueError, match="fields do not match the serialized schema"):
            GVFSpec.from_config(payload)

    def test_gvf_spec_from_config_rejects_extra_fields(self):
        payload = {
            "name": "d0",
            "demon_type": "prediction",
            "gamma": 0.0,
            "lamda": 0.0,
            "cumulant_index": 0,
            "terminal_reward": 0.0,
            "extra_field": 42,
        }
        with pytest.raises(ValueError, match="fields do not match the serialized schema"):
            GVFSpec.from_config(payload)

    @pytest.mark.parametrize(
        "invalid_demon_type", [123, True, None, "unsupported_type", []]
    )
    def test_gvf_spec_from_config_rejects_invalid_demon_type(
        self, invalid_demon_type
    ):
        payload = {
            "name": "d0",
            "demon_type": invalid_demon_type,
            "gamma": 0.0,
            "lamda": 0.0,
            "cumulant_index": 0,
            "terminal_reward": 0.0,
        }
        with pytest.raises(
            ValueError, match="demon_type must be a valid DemonType string"
        ):
            GVFSpec.from_config(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "not a dict",
            123,
            [{"demons": []}],
        ],
    )
    def test_horde_spec_from_config_rejects_non_dict(self, payload):
        with pytest.raises(ValueError, match="HordeSpec config must be an exact dict"):
            HordeSpec.from_config(payload)

    def test_horde_spec_from_config_rejects_non_string_keys(self):
        with pytest.raises(ValueError, match="keys must be exact strings"):
            HordeSpec.from_config({42: []})

    def test_horde_spec_from_config_rejects_extra_fields(self):
        spec = GVFSpec(
            name="d0",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=0,
        )
        payload = {"demons": [spec.to_config()], "extra": 1}
        with pytest.raises(ValueError, match="fields do not match the serialized schema"):
            HordeSpec.from_config(payload)

    @pytest.mark.parametrize("invalid_demons", ["not a list", 123, None, {"a": 1}])
    def test_horde_spec_from_config_rejects_non_list_demons(self, invalid_demons):
        with pytest.raises(ValueError, match="HordeSpec demons must be an exact list"):
            HordeSpec.from_config({"demons": invalid_demons})


@pytest.mark.parametrize("dtype_code", ["e", "f", "d", "g"])
def test_gvf_spec_accepts_all_numpy_floating_types_for_terminal_reward(dtype_code: str) -> None:
    val = np.dtype(dtype_code).type(1.5)
    spec = GVFSpec(
        name="d",
        demon_type=DemonType.PREDICTION,
        gamma=0.0,
        lamda=0.0,
        cumulant_index=0,
        terminal_reward=val,
    )
    assert spec.terminal_reward == 1.5
