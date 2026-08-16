"""Mechanism tests for the experimental PrototypeAgent composition surface.

These tests establish routing, shape, update, and isolation invariants.  They
do not establish an integrated Alberta Plan completion result.
"""

from __future__ import annotations

import warnings
from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.intelligence_amplification import IAConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeArrayResult,
    PrototypeExperientialMemoryInput,
    PrototypeTransition,
    PrototypeUpdateResult,
    _increment_decision_id,
    feature_to_subtask_specs,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SPEC0 = SubtaskSpec(
    feature_index=0,
    threshold=0.5,
    pseudo_reward_scale=1.0,
    max_option_steps=8,
)
_SPEC1 = SubtaskSpec(
    feature_index=1,
    threshold=0.3,
    pseudo_reward_scale=2.0,
    max_option_steps=4,
)

OBS_DIM = 4
N_PRIM = 2


def _materialize_typed_keys(tree):
    """Convert typed PRNG leaves so Chex can compare complete agent states."""

    def convert(value):
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(
            dtype,
            jax.dtypes.prng_key,
        ):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


def _oak_cfg(
    specs: tuple[SubtaskSpec, ...] = (_SPEC0,),
    obs_dim: int = OBS_DIM,
    n_prim: int = N_PRIM,
) -> OaKConfig:
    stomp = STOMPConfig(
        subtask_specs=specs,
        observation_dim=obs_dim,
        n_primitive_actions=n_prim,
    )
    return OaKConfig(stomp=stomp)


def _wm_cfg(
    obs_dim: int = OBS_DIM,
    n_actions: int = N_PRIM,
    gamma: float = 0.99,
) -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=obs_dim,
        n_actions=n_actions,
        gamma=gamma,
        hidden_sizes=(),  # linear for speed
        step_size=0.1,
        error_decay=0.99,
    )


def _minimal_config() -> PrototypeAgentConfig:
    """OaK-only, no world model, no horde, no IA."""
    return PrototypeAgentConfig(oak=_oak_cfg())


def _full_config(n_dreams: int = 2) -> PrototypeAgentConfig:
    """All components enabled."""
    horde_spec = create_horde_spec(
        [
            GVFSpec(
                name="v0.9",
                demon_type=DemonType.PREDICTION,
                cumulant_index=0,
                gamma=0.9,
                lamda=0.0,
            ),
            GVFSpec(
                name="r",
                demon_type=DemonType.PREDICTION,
                cumulant_index=0,
                gamma=0.0,
                lamda=0.0,
            ),
        ]
    )
    from alberta_framework.core.intelligence_amplification import ExoCerebellumConfig

    ia_cortex = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(_SPEC0,),
            observation_dim=OBS_DIM,
            n_primitive_actions=N_PRIM,
        )
    )
    ia_cfg = IAConfig(
        cerebellum=ExoCerebellumConfig(n_demons=2, obs_dim=OBS_DIM, step_size=0.05),
        cortex=ia_cortex,
    )
    return PrototypeAgentConfig(
        oak=_oak_cfg(),
        world_model=_wm_cfg(),
        dreaming=DreamingConfig(warmup_steps=1, max_model_error_ema=1e6),
        buffer_capacity=20,
        n_dreams_per_step=n_dreams,
        horde_spec=horde_spec,
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        ia=ia_cfg,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestPrototypeAgentConfigValidation:
    def test_buffer_capacity_positive(self) -> None:
        with pytest.raises(ValueError, match="buffer_capacity"):
            PrototypeAgentConfig(oak=_oak_cfg(), buffer_capacity=0)

    def test_n_dreams_non_negative(self) -> None:
        with pytest.raises(ValueError, match="n_dreams_per_step"):
            PrototypeAgentConfig(oak=_oak_cfg(), n_dreams_per_step=-1)

    def test_dreams_require_world_model(self) -> None:
        with pytest.raises(ValueError, match="world_model"):
            PrototypeAgentConfig(oak=_oak_cfg(), n_dreams_per_step=2, world_model=None)

    def test_unknown_dream_next_observation_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="dream_next_observation_mode"):
            PrototypeAgentConfig(
                oak=_oak_cfg(),
                dream_next_observation_mode="unknown",  # type: ignore[arg-type]
            )

    def test_sample_one_hot_dreams_reject_gru_observations(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        with pytest.raises(ValueError, match="GRU-augmented"):
            PrototypeAgentConfig(
                oak=_oak_cfg(obs_dim=2),
                gru_perception=GRUPerceptionConfig(
                    observation_dim=1,
                    hidden_dim=1,
                ),
                dream_next_observation_mode="sample_one_hot",
            )

    def test_world_model_observation_dim_must_match_oak(self) -> None:
        with pytest.raises(ValueError, match="world_model.observation_dim"):
            PrototypeAgentConfig(
                oak=_oak_cfg(obs_dim=2),
                world_model=_wm_cfg(obs_dim=3),
                dream_next_observation_mode="sample_one_hot",
            )

    def test_world_model_action_count_must_match_oak(self) -> None:
        with pytest.raises(ValueError, match="world_model.n_actions"):
            PrototypeAgentConfig(
                oak=_oak_cfg(n_prim=2),
                world_model=_wm_cfg(n_actions=3),
                dream_next_observation_mode="sample_one_hot",
            )

    def test_ia_obs_dim_must_match_oak(self) -> None:
        from alberta_framework.core.intelligence_amplification import ExoCerebellumConfig

        bad_cortex = OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(_SPEC0,),
                observation_dim=OBS_DIM + 1,  # mismatched
                n_primitive_actions=N_PRIM,
            )
        )
        ia_bad = IAConfig(
            cerebellum=ExoCerebellumConfig(n_demons=2, obs_dim=OBS_DIM + 1, step_size=0.05),
            cortex=bad_cortex,
        )
        with pytest.raises(ValueError, match="observation_dim"):
            PrototypeAgentConfig(oak=_oak_cfg(obs_dim=OBS_DIM), ia=ia_bad)

    def test_horde_step_size_positive(self) -> None:
        with pytest.raises(ValueError, match="horde_step_size"):
            PrototypeAgentConfig(oak=_oak_cfg(), horde_step_size=0.0)

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (float("nan"), "horde_step_size must be a finite real number"),
            (float("inf"), "horde_step_size must be a finite real number"),
            (1e100, "horde_step_size must remain finite once narrowed to float32"),
            (5e-324, "horde_step_size must remain positive once narrowed to float32"),
            (1e-50, "horde_step_size must remain positive once narrowed to float32"),
            (Fraction(1, 10**400), "horde_step_size must remain positive once narrowed"),
        ],
    )
    def test_horde_step_size_must_be_positive_in_float32(self, value: object, message: str) -> None:
        """The Horde consumes float32: host-finite values narrowing to inf or 0 are refused."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=message):
                PrototypeAgentConfig(oak=_oak_cfg(), horde_step_size=value)  # type: ignore[arg-type]

    def test_horde_step_size_canonicalizes_reals_and_round_trips(self) -> None:
        config = PrototypeAgentConfig(oak=_oak_cfg(), horde_step_size=Fraction(1, 8))
        assert type(config.horde_step_size) is float and config.horde_step_size == 0.125
        big = PrototypeAgentConfig(oak=_oak_cfg(), horde_step_size=(2**25 - 1) * 2**103 - 1)
        assert big.horde_step_size == float(np.finfo(np.float32).max)
        restored = PrototypeAgentConfig.from_config(big.to_config())
        assert restored.horde_step_size == big.horde_step_size
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        assert int(agent.update(state, 0.1, jnp.ones(OBS_DIM)).state.step_count) == 1

    def test_world_model_gamma_zero_requires_explicit_transition_api(self) -> None:
        """A zero model horizon is valid, but the legacy API has no terminal input."""
        config = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(gamma=0.0),
        )
        restored = PrototypeAgentConfig.from_config(config.to_config())
        assert restored.world_model is not None
        assert restored.world_model.gamma == 0.0

        agent = PrototypeAgent(config)
        state = agent.start(agent.init(jr.key(41)), jnp.zeros(OBS_DIM))
        next_observation = jnp.ones(OBS_DIM, dtype=jnp.float32)
        with pytest.raises(ValueError, match="use update_transition"):
            agent.update(state, jnp.asarray(1.0), next_observation)

        result = agent.update_transition(
            state,
            PrototypeTransition(
                observation=state.current_raw_observation,
                action=state.current_action,
                decision_id=state.current_decision_id,
                reward=jnp.asarray(1.0, dtype=jnp.float32),
                discount=jnp.asarray(0.0, dtype=jnp.float32),
                terminated=jnp.asarray(True),
                truncated=jnp.asarray(False),
                next_observation=next_observation,
                next_decision_observation=next_observation,
            ),
        )
        assert bool(result.transition_diagnostics.valid)
        assert int(result.state.step_count) == 1


# ---------------------------------------------------------------------------
# Config roundtrip
# ---------------------------------------------------------------------------


class TestPrototypeAgentConfigRoundtrip:
    def test_minimal_roundtrip(self) -> None:
        cfg = _minimal_config()
        restored = PrototypeAgentConfig.from_config(cfg.to_config())
        assert restored.oak.observation_dim == cfg.oak.observation_dim
        assert restored.world_model is None
        assert restored.horde_spec is None
        assert restored.ia is None

    def test_full_roundtrip(self) -> None:
        cfg = _full_config()
        restored = PrototypeAgentConfig.from_config(cfg.to_config())
        assert restored.oak.observation_dim == cfg.oak.observation_dim
        assert restored.world_model is not None
        assert restored.horde_spec is not None
        assert restored.ia is not None
        assert restored.n_dreams_per_step == cfg.n_dreams_per_step
        assert restored.buffer_capacity == cfg.buffer_capacity

    def test_legacy_dream_mode_preserves_serialized_config(self) -> None:
        cfg = _full_config()
        payload = cfg.to_config()
        assert cfg.dream_next_observation_mode == "model_prediction"
        assert "dream_next_observation_mode" not in payload
        assert (
            PrototypeAgentConfig.from_config(payload).dream_next_observation_mode
            == "model_prediction"
        )

    def test_sample_one_hot_dream_mode_roundtrip(self) -> None:
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=0),
            n_dreams_per_step=1,
            dream_next_observation_mode="sample_one_hot",
        )
        payload = cfg.to_config()
        assert payload["dream_next_observation_mode"] == "sample_one_hot"
        restored = PrototypeAgentConfig.from_config(payload)
        assert restored.dream_next_observation_mode == "sample_one_hot"

    def test_from_config_rejects_wrong_type_and_unknown_fields(self) -> None:
        payload = _minimal_config().to_config()
        wrong_type = dict(payload, type="NotPrototypeAgentConfig")
        with pytest.raises(ValueError, match="payload type"):
            PrototypeAgentConfig.from_config(wrong_type)
        unknown = dict(payload, unexpected=1)
        with pytest.raises(ValueError, match="unknown fields: unexpected"):
            PrototypeAgentConfig.from_config(unknown)


# ---------------------------------------------------------------------------
# Init and start
# ---------------------------------------------------------------------------


class TestPrototypeAgentInit:
    def test_init_minimal_state_shapes(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        assert isinstance(state, PrototypeAgentState)
        assert state.world_model_state is None
        assert state.buffer_state is None
        assert state.horde_state is None
        assert state.ia_state is None
        assert state.step_count == 0

    def test_init_oak_state_present(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        n_total = N_PRIM + 1  # 1 option
        bls = state.oak_state.stomp_state.base_learner_state
        assert len(bls.head_params.weights) == n_total

    def test_init_full_state_shapes(self) -> None:
        agent = PrototypeAgent(_full_config())
        state = agent.init(jr.key(0))
        assert state.world_model_state is not None
        assert state.buffer_state is not None
        assert state.horde_state is not None
        assert state.ia_state is not None

    def test_start_primes_oak(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        obs = jnp.ones(OBS_DIM)
        primed = agent.start(state, obs)
        chex.assert_trees_all_close(primed.oak_state.stomp_state.base_last_obs, obs, atol=1e-6)

    def test_start_primes_ia(self) -> None:
        agent = PrototypeAgent(_full_config())
        state = agent.init(jr.key(0))
        obs = jnp.ones(OBS_DIM)
        primed = agent.start(state, obs)
        assert primed.ia_state is not None
        chex.assert_trees_all_close(
            primed.ia_state.cortex_state.stomp_state.base_last_obs, obs, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Act
# ---------------------------------------------------------------------------


class TestPrototypeAgentAct:
    def test_act_returns_valid_action(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        obs = jnp.zeros(OBS_DIM)
        action = agent.act(state, obs)
        chex.assert_shape(action, ())
        assert int(action) < N_PRIM


# ---------------------------------------------------------------------------
# Update: minimal (OaK only)
# ---------------------------------------------------------------------------


class TestPrototypeAgentUpdateMinimal:
    def test_update_contract(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))

        assert isinstance(result, PrototypeUpdateResult)
        assert int(result.state.step_count) == 1
        chex.assert_shape(result.action, ())
        assert jnp.isfinite(result.oak_td_error)
        assert result.world_model_error is None
        assert result.dream_td_errors is None
        assert result.horde_td_errors is None
        assert result.ia_augmented_obs is None
        assert result.ia_recommendation is None

        state = result.state
        for _ in range(9):
            result = agent.update(state, jnp.array(0.5), jnp.ones(OBS_DIM))
            state = result.state
        assert int(state.step_count) == 10


# ---------------------------------------------------------------------------
# Update: full agent (world model + dreaming + horde + IA)
# ---------------------------------------------------------------------------


class TestPrototypeAgentUpdateFull:
    def test_full_update_contract(self) -> None:
        agent = PrototypeAgent(_full_config(n_dreams=2))
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        cumulants = jnp.array([0.5, 0.3], dtype=jnp.float32)
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM), cumulants)

        assert result.world_model_error is not None
        assert jnp.isfinite(result.world_model_error)
        assert result.dream_td_errors is not None
        chex.assert_shape(result.dream_td_errors, (2,))
        assert result.horde_td_errors is not None
        chex.assert_shape(result.horde_td_errors, (2,))  # 2 demons
        assert result.ia_augmented_obs is not None
        chex.assert_shape(result.ia_augmented_obs, (OBS_DIM + 2,))  # obs + 2 cerebellum demons
        assert result.ia_recommendation is not None
        chex.assert_shape(result.ia_recommendation, ())
        assert int(result.state.buffer_state.size) == 1
        assert int(result.state.world_model_state.step_count) == 1


def test_experiential_memory_uses_one_accounted_policy_step() -> None:
    """Prototype proposal and write must share one recorded pre-state query."""
    memory_config = ExperientialMemoryConfig(
        capacity=3,
        observation_dim=OBS_DIM,
        key_dim=OBS_DIM,
        action_dim=N_PRIM,
        outcome_dim=OBS_DIM + 1,
        top_k=1,
        min_neighbors=1,
    )
    agent = PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak_cfg(),
            experiential_memory=memory_config,
        )
    )
    resources = agent.experiential_memory_resource_declaration
    assert resources is not None
    assert resources.categorical_policy_queries == 1
    assert resources.causal_step_queries == 0
    assert resources.total_deterministic_prestate_queries == 1

    state = agent.start(agent.init(jr.key(31)), jnp.zeros(OBS_DIM, dtype=jnp.float32))
    transition = PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        next_observation=jnp.ones(OBS_DIM, dtype=jnp.float32),
        next_decision_observation=jnp.ones(OBS_DIM, dtype=jnp.float32),
    )
    memory_input = PrototypeExperientialMemoryInput(
        available=jnp.asarray(True),
        current_prototype_decision_id=state.current_decision_id,
        next_prototype_decision_id=_increment_decision_id(state.current_decision_id),
        query_representation_version=jnp.asarray(0, dtype=jnp.int32),
        entry_representation_version=jnp.asarray(0, dtype=jnp.int32),
        query_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True),
        entry_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
        entry_uncertainty_available=jnp.asarray(True),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(1.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True),
        provenance_id=jnp.asarray(1, dtype=jnp.int32),
        source_id=jnp.asarray(1, dtype=jnp.int32),
        next_action_safety_mask=jnp.ones(N_PRIM, dtype=jnp.bool_),
    )

    result = agent.update_transition(
        state,
        transition,
        experiential_memory_input=memory_input,
    )
    diagnostics = result.experiential_memory_diagnostics
    memory_state = agent._experiential_memory_component_state(result.state.ia_state)

    assert bool(diagnostics.transaction_applied)
    assert int(diagnostics.deterministic_prestate_query_count) == 1
    assert int(memory_state.query_count) == 1
    assert int(memory_state.write_count) == 1


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


class TestPrototypeAgentScan:
    def test_scan_minimal_contract(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        n_steps = 8
        rewards = jr.normal(jr.key(42), (n_steps,))
        next_obs = jr.normal(jr.key(43), (n_steps, OBS_DIM))
        result = agent.scan(state, rewards, next_obs)

        assert isinstance(result, PrototypeArrayResult)
        chex.assert_shape(result.actions, (n_steps,))
        chex.assert_shape(result.oak_td_errors, (n_steps,))
        chex.assert_shape(result.oak_average_rewards, (n_steps,))
        assert int(result.state.step_count) == n_steps
        chex.assert_tree_all_finite(result.oak_td_errors)

    @pytest.mark.parametrize(
        ("field", "shape"),
        [
            ("rewards", (8, 1)),
            ("rewards", (2, 4)),
            ("next_observations", (OBS_DIM, 8)),
            ("next_observations", (2, 4 * OBS_DIM)),
            ("discounts", (2, 4)),
        ],
    )
    def test_scan_rejects_wrong_shaped_arrays_instead_of_reshaping(
        self, field: str, shape: tuple[int, ...]
    ) -> None:
        """A transposed or wrongly stacked buffer must not be silently reinterpreted row-major."""
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        n_steps = 8
        arrays: dict[str, object] = {
            "rewards": jr.normal(jr.key(42), (n_steps,)),
            "next_observations": jr.normal(jr.key(43), (n_steps, OBS_DIM)),
            "discounts": jnp.full((n_steps,), 0.9, dtype=jnp.float32),
        }
        arrays[field] = jnp.reshape(arrays[field], shape)
        with pytest.raises(ValueError, match=f"{field} must have shape"):
            agent.scan(
                state,
                arrays["rewards"],
                arrays["next_observations"],
                discounts=arrays["discounts"],
            )

    def test_scan_matches_sequential(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        init_state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        n_steps = 5
        rewards = jnp.array([0.1, 0.2, 0.0, -0.1, 0.5])
        next_obs = jnp.ones((n_steps, OBS_DIM)) * jnp.arange(
            n_steps,
            dtype=jnp.float32,
        )[:, None]

        # Sequential
        state = init_state
        seq_actions = []
        for i in range(n_steps):
            result = agent.update(state, rewards[i], next_obs[i])
            seq_actions.append(int(result.action))
            state = result.state
        seq_final_step = int(state.step_count)

        # Scan
        scan_result = agent.scan(init_state, rewards, next_obs)
        scan_final_step = int(scan_result.state.step_count)

        assert seq_final_step == scan_final_step == n_steps

    def test_scan_world_model_config_update(self) -> None:
        """Scan with world model enabled runs without error."""
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=100),  # warmup prevents dreaming
            n_dreams_per_step=0,
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        n_steps = 6
        result = agent.scan(state, jnp.zeros(n_steps), jnp.zeros((n_steps, OBS_DIM)))
        assert int(result.state.world_model_state.step_count) == n_steps


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


class TestPrototypeAgentCurate:
    def test_curate_returns_new_agent_and_state(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        # Run a few steps to build utility EMA
        for _ in range(20):
            result = agent.update(state, jnp.array(0.0), jnp.ones(OBS_DIM))
            state = result.state
        new_agent, new_state = agent.curate(state, jr.key(1))
        assert isinstance(new_agent, PrototypeAgent)
        assert isinstance(new_state, PrototypeAgentState)

    def test_curate_preserves_non_oak_states(self) -> None:
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=1000),
            n_dreams_per_step=0,
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        for _ in range(5):
            result = agent.update(state, jnp.array(0.0), jnp.ones(OBS_DIM))
            state = result.state
        new_agent, new_state = agent.curate(state, jr.key(2))
        # World model state preserved
        assert (
            int(new_state.world_model_state.step_count)
            == int(state.world_model_state.step_count)
        )

    def test_curate_preserves_dream_next_observation_mode(self) -> None:
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=1000),
            n_dreams_per_step=0,
            dream_next_observation_mode="sample_one_hot",
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jax.nn.one_hot(0, OBS_DIM))
        for _ in range(20):
            state = agent.update(
                state,
                jnp.array(0.0),
                jax.nn.one_hot(1, OBS_DIM),
            ).state
        new_agent, _ = agent.curate(state, jr.key(2))
        assert new_agent.config.dream_next_observation_mode == "sample_one_hot"

    def test_curated_agent_can_continue_learning(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        for _ in range(10):
            result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))
            state = result.state
        new_agent, new_state = agent.curate(state, jr.key(5))
        result = new_agent.update(new_state, jnp.array(0.5), jnp.ones(OBS_DIM))
        assert jnp.isfinite(result.oak_td_error)


# ---------------------------------------------------------------------------
# Auto subtask specs
# ---------------------------------------------------------------------------


class TestAutoSubtaskSpecs:
    @pytest.mark.parametrize(
        "n_subtasks",
        [np.int64(2), np.array(2, dtype=np.int64), jnp.int32(2)],
        ids=["numpy-scalar", "numpy-array", "jax-scalar"],
    )
    def test_auto_subtask_specs_accepts_integer_scalars(self, n_subtasks) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))

        specs = agent.auto_subtask_specs(state, n_subtasks=n_subtasks)

        assert len(specs) == 2

    @pytest.mark.parametrize("n_subtasks", [-1, True, False, 1.5, "2"])
    def test_auto_subtask_specs_rejects_invalid_counts(self, n_subtasks) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))

        with pytest.raises(
            ValueError,
            match="^n_subtasks must be a non-negative integer$",
        ):
            agent.auto_subtask_specs(state, n_subtasks=n_subtasks)

    def test_auto_subtask_specs_count(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        for _ in range(5):
            result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))
            state = result.state
        specs = agent.auto_subtask_specs(state, n_subtasks=3)
        assert len(specs) == 3

    def test_auto_subtask_specs_valid_indices(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jr.normal(jr.key(7), (OBS_DIM,)))
        for _ in range(10):
            result = agent.update(state, jr.normal(jr.key(8), ()), jr.normal(jr.key(9), (OBS_DIM,)))
            state = result.state
        specs = agent.auto_subtask_specs(state, n_subtasks=4)
        for spec in specs:
            assert 0 <= spec.feature_index < OBS_DIM

    def test_auto_subtask_specs_unique_indices(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jr.normal(jr.key(10), (OBS_DIM,)))
        for _ in range(10):
            result = agent.update(state, jnp.array(1.0), jr.normal(jr.key(11), (OBS_DIM,)))
            state = result.state
        specs = agent.auto_subtask_specs(state, n_subtasks=OBS_DIM)
        indices = [s.feature_index for s in specs]
        assert len(indices) == len(set(indices))


# ---------------------------------------------------------------------------
# feature_to_subtask_specs standalone
# ---------------------------------------------------------------------------


class TestFeatureToSubtaskSpecs:
    @pytest.mark.parametrize("n_subtasks", [-1, True, False, 1.5, "2"])
    def test_rejects_invalid_subtask_counts(self, n_subtasks) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))

        with pytest.raises(
            ValueError,
            match="^n_subtasks must be a non-negative integer$",
        ):
            feature_to_subtask_specs(state.oak_state, n_subtasks=n_subtasks)

    @pytest.mark.parametrize(
        "n_subtasks",
        [np.int64(2), np.array(2, dtype=np.int64), jnp.int32(2)],
        ids=["numpy-scalar", "numpy-array", "jax-scalar"],
    )
    def test_accepts_integer_scalar_counts(self, n_subtasks) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))

        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=n_subtasks)

        assert len(specs) == 2

    @pytest.mark.parametrize("n_subtasks", [0, np.int64(0), jnp.int32(0)])
    def test_zero_subtask_count_returns_empty(self, n_subtasks) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))

        assert feature_to_subtask_specs(state.oak_state, n_subtasks=n_subtasks) == ()

    def test_huge_subtask_count_caps_at_observation_dim(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))

        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=10**100)

        assert len(specs) == OBS_DIM

    def test_returns_correct_count(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=2)
        assert len(specs) == 2

    def test_caps_at_obs_dim(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=100)
        assert len(specs) <= OBS_DIM

    def test_respects_threshold(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=2, threshold=0.7)
        for spec in specs:
            assert spec.threshold == pytest.approx(0.7)

    def test_respects_max_option_steps(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=2, max_option_steps=15)
        for spec in specs:
            assert spec.max_option_steps == 15

    def test_valid_feature_indices(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=OBS_DIM)
        for spec in specs:
            assert 0 <= spec.feature_index < OBS_DIM

    def test_ranks_by_max_across_sources_not_sum(self) -> None:
        """Pins the documented contract (issue #412): ranking is the maximum
        absolute Q-weight across all base and option policies, i.e.
        ``jnp.maximum(feature_importance, opt_importance)`` per feature — not
        the sum of the two per-source maxima. A feature that is mid-weight in
        both sources must not outrank a feature holding the single largest
        weight anywhere.

        Reproduction values from the issue: per-feature base max
        ``[1.0, 0.6, 0.1]``, per-feature option max ``[0.0, 0.6, 0.1]``.
        Documented (max) ranking is ``[0, 1, 2]``; the pre-fix summed
        ranking was ``[1, 0, 2]``.
        """
        obs_dim = 3
        config = PrototypeAgentConfig(oak=_oak_cfg(obs_dim=obs_dim, n_prim=N_PRIM))
        agent = PrototypeAgent(config)
        state = agent.init(jr.key(0))
        oak_state = state.oak_state
        stomp_state = oak_state.stomp_state

        base_row = jnp.array([1.0, 0.6, 0.1], dtype=jnp.float32)
        head_params = stomp_state.base_learner_state.head_params
        new_head_params = head_params.replace(
            weights=tuple(base_row[None, :] for _ in head_params.weights)
        )
        new_base_learner_state = stomp_state.base_learner_state.replace(
            head_params=new_head_params
        )

        opt_row = jnp.array([0.0, 0.6, 0.1], dtype=jnp.float32)
        new_option_policies = stomp_state.option_policies.replace(
            q_weights=jnp.broadcast_to(opt_row, stomp_state.option_policies.q_weights.shape)
        )

        patched_oak_state = oak_state.replace(
            stomp_state=stomp_state.replace(
                base_learner_state=new_base_learner_state,
                option_policies=new_option_policies,
            )
        )

        specs = feature_to_subtask_specs(patched_oak_state, n_subtasks=3)

        assert [s.feature_index for s in specs] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Config serialization agent roundtrip
# ---------------------------------------------------------------------------


class TestPrototypeAgentSerializationRoundtrip:
    def test_from_config_to_config_minimal(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        restored = PrototypeAgent.from_config(agent.to_config())
        assert restored.config.oak.observation_dim == OBS_DIM
        assert restored.config.world_model is None

    def test_from_config_to_config_full(self) -> None:
        agent = PrototypeAgent(_full_config())
        restored = PrototypeAgent.from_config(agent.to_config())
        assert restored.config.n_dreams_per_step == agent.config.n_dreams_per_step
        assert restored.config.horde_spec is not None
        assert restored.config.ia is not None

    def test_sample_one_hot_checkpoint_resume_replays_identical_dream_draws(
        self,
        tmp_path,
    ) -> None:
        base = _full_config(n_dreams=1)
        config = PrototypeAgentConfig.from_config(
            {
                **base.to_config(),
                "dream_next_observation_mode": "sample_one_hot",
            }
        )
        agent = PrototypeAgent(config)
        state = agent.start(agent.init(jr.key(41)), jax.nn.one_hot(0, OBS_DIM))
        transitions = (
            (0.25, jax.nn.one_hot(1, OBS_DIM)),
            (-0.5, jax.nn.one_hot(2, OBS_DIM)),
            (1.0, jax.nn.one_hot(3, OBS_DIM)),
        )
        for reward, observation in transitions:
            state = agent.update(state, reward, observation).state

        checkpoint_path = tmp_path / "prototype-sample-one-hot"
        save_prototype_checkpoint(agent, state, checkpoint_path)
        restored_agent, restored_state = load_prototype_checkpoint(checkpoint_path)

        assert (
            restored_agent.config.dream_next_observation_mode
            == "sample_one_hot"
        )
        chex.assert_trees_all_close(
            _materialize_typed_keys(restored_state),
            _materialize_typed_keys(state),
        )
        reward = jnp.array(0.75, dtype=jnp.float32)
        observation = jax.nn.one_hot(1, OBS_DIM)
        uninterrupted = agent.update(state, reward, observation)
        resumed = restored_agent.update(restored_state, reward, observation)
        chex.assert_trees_all_close(
            _materialize_typed_keys(resumed),
            _materialize_typed_keys(uninterrupted),
        )

    def test_checkpoint_loader_rejects_wrong_schema_or_config_digest(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        import alberta_framework.core.prototype_agent as prototype_module

        config = PrototypeAgent(_minimal_config()).to_config()
        monkeypatch.setattr(
            prototype_module,
            "load_checkpoint_metadata",
            lambda _path: {
                "schema": "alberta.prototype_agent.v0",
                "agent_config": config,
                "config_sha256": "unused",
            },
        )
        with pytest.raises(ValueError, match="PrototypeAgent v1"):
            load_prototype_checkpoint(tmp_path / "not-read")

        monkeypatch.setattr(
            prototype_module,
            "load_checkpoint_metadata",
            lambda _path: {
                "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
                "agent_config": config,
                "config_sha256": "tampered",
            },
        )
        with pytest.raises(ValueError, match="digest"):
            load_prototype_checkpoint(tmp_path / "not-read")

        noncanonical = dict(config)
        noncanonical["dream_next_observation_mode"] = "model_prediction"
        monkeypatch.setattr(
            prototype_module,
            "load_checkpoint_metadata",
            lambda _path: {
                "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
                "agent_config": noncanonical,
                "config_sha256": prototype_module._prototype_config_digest(
                    noncanonical
                ),
            },
        )
        with pytest.raises(ValueError, match="not canonical"):
            load_prototype_checkpoint(tmp_path / "not-read")


# ---------------------------------------------------------------------------
# Dreaming mechanics
# ---------------------------------------------------------------------------


class TestPrototypeAgentDreaming:
    def test_dreams_accepted_after_warmup(self) -> None:
        """After warmup, at least some dream TD errors should be nonzero."""
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=1, max_model_error_ema=1e6),
            buffer_capacity=50,
            n_dreams_per_step=4,
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        # Warm up the world model and buffer
        for _ in range(10):
            result = agent.update(state, jnp.array(0.5), jr.normal(jr.key(13), (OBS_DIM,)))
            state = result.state
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert result.dream_td_errors is not None
        chex.assert_shape(result.dream_td_errors, (4,))
        chex.assert_tree_all_finite(result.dream_td_errors)

    def test_dreams_zero_before_warmup(self) -> None:
        """During warmup, dream TD errors should all be zero (gated)."""
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=10000, max_model_error_ema=1e6),
            buffer_capacity=50,
            n_dreams_per_step=3,
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert result.dream_td_errors is not None
        # All gated dreams produce 0.0
        chex.assert_trees_all_close(result.dream_td_errors, jnp.zeros(3), atol=1e-6)


# ---------------------------------------------------------------------------
# GRU Perception (Step 8 sub-component a)
# ---------------------------------------------------------------------------

GRU_OBS_DIM = 4
GRU_HIDDEN = 8
GRU_AUG_DIM = GRU_OBS_DIM + GRU_HIDDEN


def _gru_config() -> PrototypeAgentConfig:
    from alberta_framework.core.prototype_agent import GRUPerceptionConfig

    return PrototypeAgentConfig(
        oak=_oak_cfg(obs_dim=GRU_AUG_DIM),
        gru_perception=GRUPerceptionConfig(
            observation_dim=GRU_OBS_DIM,
            hidden_dim=GRU_HIDDEN,
        ),
    )


class TestGRUPerceptionConfig:
    def test_augmented_dim(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        cfg = GRUPerceptionConfig(observation_dim=4, hidden_dim=16)
        assert cfg.augmented_dim() == 20

    def test_config_roundtrip(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        cfg = GRUPerceptionConfig(observation_dim=6, hidden_dim=32)
        restored = GRUPerceptionConfig.from_config(cfg.to_config())
        assert restored.observation_dim == 6
        assert restored.hidden_dim == 32

    def test_oak_dim_mismatch_raises(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        with pytest.raises(ValueError, match="oak.observation_dim"):
            PrototypeAgentConfig(
                oak=_oak_cfg(obs_dim=4),  # wrong — should be 4+8=12
                gru_perception=GRUPerceptionConfig(observation_dim=4, hidden_dim=8),
            )

    def test_world_model_dim_mismatch_raises(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        with pytest.raises(ValueError, match="world_model.observation_dim"):
            PrototypeAgentConfig(
                oak=_oak_cfg(obs_dim=12),  # correct: 4+8
                world_model=ActionConditionedWorldModelConfig(
                    observation_dim=4,  # wrong — should be 12
                    n_actions=2,
                ),
                gru_perception=GRUPerceptionConfig(observation_dim=4, hidden_dim=8),
            )

    def test_prototype_config_roundtrip_with_gru(self) -> None:
        cfg = _gru_config()
        restored = PrototypeAgentConfig.from_config(cfg.to_config())
        assert restored.gru_perception is not None
        assert restored.gru_perception.observation_dim == GRU_OBS_DIM
        assert restored.gru_perception.hidden_dim == GRU_HIDDEN


class TestGRUPerceptionStateInit:
    def test_hidden_zeros_at_init(self) -> None:
        agent = PrototypeAgent(_gru_config())
        state = agent.init(jr.key(0))
        assert state.gru_state is not None
        chex.assert_shape(state.gru_state.hidden, (GRU_HIDDEN,))
        assert float(jnp.max(jnp.abs(state.gru_state.hidden))) == pytest.approx(0.0)

    def test_weight_shapes_correct(self) -> None:
        agent = PrototypeAgent(_gru_config())
        state = agent.init(jr.key(1))
        gru = state.gru_state
        chex.assert_shape(gru.W_z, (GRU_HIDDEN, GRU_OBS_DIM))
        chex.assert_shape(gru.U_z, (GRU_HIDDEN, GRU_HIDDEN))
        chex.assert_shape(gru.b_z, (GRU_HIDDEN,))

    def test_no_gru_state_when_disabled(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        assert state.gru_state is None


class TestGRUPerceptionUpdate:
    def test_hidden_updates_after_start(self) -> None:
        agent = PrototypeAgent(_gru_config())
        state0 = agent.init(jr.key(0))
        obs = jr.normal(jr.key(1), (GRU_OBS_DIM,))
        state1 = agent.start(state0, obs)
        assert float(jnp.max(jnp.abs(state1.gru_state.hidden))) > 0.0

    def test_oak_receives_augmented_obs(self) -> None:
        """OaK last_obs should have augmented dimension after start."""
        agent = PrototypeAgent(_gru_config())
        state = agent.start(agent.init(jr.key(0)), jr.normal(jr.key(1), (GRU_OBS_DIM,)))
        stored = state.oak_state.stomp_state.base_last_obs
        chex.assert_shape(stored, (GRU_AUG_DIM,))

    def test_update_changes_hidden(self) -> None:
        agent = PrototypeAgent(_gru_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(GRU_OBS_DIM))
        h0 = state.gru_state.hidden
        obs = jr.normal(jr.key(2), (GRU_OBS_DIM,))
        result = agent.update(state, jnp.array(1.0), obs)
        h1 = result.state.gru_state.hidden
        assert not jnp.allclose(h0, h1)

    def test_update_finite(self) -> None:
        agent = PrototypeAgent(_gru_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(GRU_OBS_DIM))
        for _ in range(10):
            obs = jr.normal(jr.key(42), (GRU_OBS_DIM,))
            result = agent.update(state, jnp.array(1.0), obs)
            state = result.state
        assert jnp.isfinite(result.oak_td_error)
        assert jnp.all(jnp.isfinite(state.gru_state.hidden))

    def test_curate_preserves_gru_config(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        agent = PrototypeAgent(
            PrototypeAgentConfig(
                oak=_oak_cfg(specs=(_SPEC0, _SPEC1), obs_dim=GRU_AUG_DIM),
                gru_perception=GRUPerceptionConfig(
                    observation_dim=GRU_OBS_DIM, hidden_dim=GRU_HIDDEN
                ),
            )
        )
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(GRU_OBS_DIM))
        new_agent, new_state = agent.curate(state, jr.key(99))
        assert new_agent.config.gru_perception is not None
        assert new_agent.config.gru_perception.hidden_dim == GRU_HIDDEN


class TestAutoCurate:
    """Tests for auto_curate_every config field and maybe_curate() method."""

    def _agent(self, auto_curate_every: int = 0) -> tuple[PrototypeAgent, PrototypeAgentState]:
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(specs=(_SPEC0, _SPEC1)),
            auto_curate_every=auto_curate_every,
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        return agent, state

    def test_config_roundtrip_with_auto_curate(self) -> None:
        cfg = PrototypeAgentConfig(oak=_oak_cfg(), auto_curate_every=50)
        cfg2 = PrototypeAgentConfig.from_config(cfg.to_config())
        assert cfg2.auto_curate_every == 50

    def test_negative_auto_curate_raises(self) -> None:
        with pytest.raises(ValueError, match="auto_curate_every"):
            PrototypeAgentConfig(oak=_oak_cfg(), auto_curate_every=-1)

    def test_maybe_curate_disabled_returns_same(self) -> None:
        agent, state = self._agent(auto_curate_every=0)
        new_agent, new_state = agent.maybe_curate(state, jr.key(1))
        assert new_agent is agent
        assert new_state is state

    def test_maybe_curate_defers_at_zero_step(self) -> None:
        """At step 0 the schedule aligns (0 % 10 == 0) but curation defers.

        Evicting at birth would act on untrained utility estimates (and can
        land mid-option right after ``start``), so ``curate`` returns the
        agent unchanged; scheduled firing is covered by
        ``test_maybe_curate_fires_every_n_steps``.
        """
        agent, state = self._agent(auto_curate_every=10)
        new_agent, new_state = agent.maybe_curate(state, jr.key(2))
        assert new_agent is agent
        assert new_state is state

    def test_maybe_curate_does_not_fire_at_non_aligned_step(self) -> None:
        agent, state = self._agent(auto_curate_every=10)
        # Advance step_count to 1 via an update
        obs = jr.normal(jr.key(7), (OBS_DIM,))
        result = agent.update(state, jnp.array(0.0), obs)
        state1 = result.state
        assert int(state1.step_count) == 1
        new_agent, new_state = agent.maybe_curate(state1, jr.key(3))
        assert new_agent is agent
        assert new_state is state1

    def test_maybe_curate_preserves_auto_curate_every(self) -> None:
        agent, state = self._agent(auto_curate_every=5)
        new_agent, _ = agent.maybe_curate(state, jr.key(4))
        assert new_agent.config.auto_curate_every == 5

    def test_maybe_curate_fires_every_n_steps(self) -> None:
        agent, state = self._agent(auto_curate_every=5)
        curations = 0
        obs = jr.normal(jr.key(0), (OBS_DIM,))
        for i in range(15):
            if int(state.step_count) % 5 == 0:
                agent, state = agent.maybe_curate(state, jr.key(i + 100))
                curations += 1
            result = agent.update(state, jnp.array(0.0), obs)
            state = result.state
        # Fires at step_count 0, 5, 10 → exactly 3
        assert curations == 3
