"""Unit tests verifying optimizer fallback truthiness safety across core learners."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.horde_actor_critic import (
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
)
from alberta_framework.core.independent_demon_horde import IndependentDemonHorde
from alberta_framework.core.learners import LinearLearner, TDLinearLearner
from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner
from alberta_framework.core.off_policy_horde import OffPolicyHordeLearner
from alberta_framework.core.optimizers import LMS, TDIDBD, Autostep
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec


class _FalsyLMS(LMS):
    def __bool__(self) -> bool:
        return False


class _FalsyTD(TDIDBD):
    def __bool__(self) -> bool:
        return False


class _HostileLMS(LMS):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("optimizer truth hook executed")


class _HostileTD(TDIDBD):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("td optimizer truth hook executed")


class _HostileOptimizerConfig(dict[str, object]):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("optimizer config truth hook executed")


class _MockUnsupportedOptimizer(Autostep):
    def supported_for_mlp(self) -> object:  # type: ignore[override]
        return "truthy-non-bool"


class _MockFalsySupportedOptimizer(Autostep):
    def supported_for_mlp(self) -> object:  # type: ignore[override]
        return 0


def _sample_horde_spec() -> HordeSpec:
    demon = GVFSpec(
        name="d0",
        demon_type=DemonType.PREDICTION,
        gamma=0.9,
        lamda=0.8,
        cumulant_index=0,
    )
    return HordeSpec(
        demons=(demon,),
        gammas=jnp.array([0.9]),
        lamdas=jnp.array([0.8]),
    )


def _sample_control_horde_spec() -> HordeSpec:
    demon = GVFSpec(
        name="control_d0",
        demon_type=DemonType.CONTROL,
        gamma=0.0,
        lamda=0.8,
        cumulant_index=0,
    )
    return HordeSpec(
        demons=(demon,),
        gammas=jnp.array([0.0]),
        lamdas=jnp.array([0.8]),
    )


class TestMultiHeadMLPLearnerOptimizerTruthiness:
    def test_multi_head_mlp_preserves_custom_falsy_optimizer(self) -> None:
        opt = _FalsyLMS(step_size=0.0123)
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(8,), optimizer=opt)
        assert learner._optimizer is opt
        assert getattr(learner._optimizer, "_step_size", None) == 0.0123

    def test_multi_head_mlp_preserves_custom_falsy_head_optimizer(self) -> None:
        trunk_opt = LMS(step_size=0.1)
        head_opt = _FalsyLMS(step_size=0.0456)
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(8,),
            optimizer=trunk_opt,
            head_optimizer=head_opt,
        )
        assert learner._optimizer is trunk_opt
        assert learner._head_optimizer is head_opt
        assert getattr(learner._head_optimizer, "_step_size", None) == 0.0456

    def test_multi_head_mlp_optimizer_does_not_invoke_truthiness(self) -> None:
        _HostileLMS.calls = 0
        opt = _HostileLMS(step_size=0.1)
        head_opt = _HostileLMS(step_size=0.05)
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(8,),
            optimizer=opt,
            head_optimizer=head_opt,
        )
        assert learner._optimizer is opt
        assert learner._head_optimizer is head_opt
        assert _HostileLMS.calls == 0

    def test_multi_head_mlp_supported_for_mlp_strict_check(self) -> None:
        bad_opt = _MockUnsupportedOptimizer(initial_step_size=0.01)
        with pytest.raises(ValueError, match="does not support the MLP shape-generic"):
            MultiHeadMLPLearner(
                n_heads=2,
                hidden_sizes=(8,),
                optimizer=bad_opt,  # type: ignore[arg-type]
            )

        good_opt = LMS(step_size=0.1)
        with pytest.raises(ValueError, match="head_optimizer.*does not support the MLP"):
            MultiHeadMLPLearner(
                n_heads=2,
                hidden_sizes=(8,),
                optimizer=good_opt,
                head_optimizer=bad_opt,  # type: ignore[arg-type]
            )


class TestLinearLearnerOptimizerTruthiness:
    def test_linear_learner_preserves_custom_falsy_optimizer(self) -> None:
        opt = _FalsyLMS(step_size=0.00789)
        learner = LinearLearner(optimizer=opt)
        assert learner._optimizer is opt
        assert getattr(learner._optimizer, "_step_size", None) == 0.00789

    def test_linear_learner_does_not_invoke_truthiness(self) -> None:
        _HostileLMS.calls = 0
        opt = _HostileLMS(step_size=0.01)
        learner = LinearLearner(optimizer=opt)
        assert learner._optimizer is opt
        assert _HostileLMS.calls == 0


class TestTDLinearLearnerOptimizerTruthiness:
    def test_td_linear_learner_preserves_custom_falsy_optimizer(self) -> None:
        opt = _FalsyTD(initial_step_size=0.0321)
        learner = TDLinearLearner(opt)
        assert learner._optimizer is opt

    def test_td_linear_learner_does_not_invoke_truthiness(self) -> None:
        _HostileTD.calls = 0
        opt = _HostileTD(initial_step_size=0.01)
        learner = TDLinearLearner(opt)
        assert learner._optimizer is opt
        assert _HostileTD.calls == 0


class TestIndependentDemonHordeOptimizerTruthiness:
    def test_independent_demon_horde_preserves_custom_falsy_optimizer(self) -> None:
        spec = _sample_horde_spec()
        opt = _FalsyLMS(step_size=0.0432)
        horde = IndependentDemonHorde(horde_spec=spec, optimizer=opt)
        assert horde._optimizer is opt

    def test_independent_demon_horde_does_not_invoke_truthiness(self) -> None:
        _HostileLMS.calls = 0
        spec = _sample_horde_spec()
        opt = _HostileLMS(step_size=0.01)
        horde = IndependentDemonHorde(horde_spec=spec, optimizer=opt)
        assert horde._optimizer is opt
        assert _HostileLMS.calls == 0

    def test_independent_demon_horde_supported_for_mlp_strict_check(self) -> None:
        spec = _sample_horde_spec()
        bad_opt = _MockUnsupportedOptimizer(initial_step_size=0.01)
        with pytest.raises(ValueError, match="does not support the MLP shape-generic"):
            IndependentDemonHorde(
                horde_spec=spec,
                optimizer=bad_opt,  # type: ignore[arg-type]
            )

        good_opt = LMS(step_size=0.1)
        with pytest.raises(ValueError, match="head_optimizer.*does not support the MLP"):
            IndependentDemonHorde(
                horde_spec=spec,
                optimizer=good_opt,
                head_optimizer=bad_opt,  # type: ignore[arg-type]
            )


class TestOffPolicyHordeLearnerOptimizerTruthiness:
    def test_off_policy_horde_preserves_custom_falsy_optimizer(self) -> None:
        spec = _sample_horde_spec()
        opt = _FalsyLMS(step_size=0.0654)
        horde = OffPolicyHordeLearner(horde_spec=spec, optimizer=opt)
        assert horde._optimizer is opt

    def test_off_policy_horde_does_not_invoke_truthiness(self) -> None:
        _HostileLMS.calls = 0
        spec = _sample_horde_spec()
        opt = _HostileLMS(step_size=0.01)
        head_opt = _HostileLMS(step_size=0.02)
        horde = OffPolicyHordeLearner(
            horde_spec=spec,
            optimizer=opt,
            head_optimizer=head_opt,
        )
        assert horde._optimizer is opt
        assert horde._head_optimizer is head_opt
        assert horde.to_config()["head_optimizer"] == head_opt.to_config()
        assert _HostileLMS.calls == 0

    def test_off_policy_horde_config_adoption_does_not_invoke_truthiness(self) -> None:
        payload = OffPolicyHordeLearner(_sample_horde_spec()).to_config()
        payload["head_optimizer"] = _HostileOptimizerConfig(LMS(step_size=0.02).to_config())
        _HostileOptimizerConfig.calls = 0

        restored = OffPolicyHordeLearner.from_config(payload)

        assert restored._head_optimizer is not None
        assert _HostileOptimizerConfig.calls == 0


class TestHordeActorCriticOptimizerSupportedCheck:
    def test_horde_actor_critic_rejects_non_true_supported_for_mlp(self) -> None:
        from alberta_framework.core.horde import HordeLearner

        spec = _sample_horde_spec()
        critic = HordeLearner(horde_spec=spec)
        config = NonlinearHordeActorCriticConfig(
            n_actions=2,
            value_head_index=0,
        )
        bad_opt = _MockUnsupportedOptimizer(initial_step_size=0.01)
        with pytest.raises(ValueError, match="does not support the MLP shape-generic"):
            NonlinearHordeActorCriticAgent(
                config=config,
                critic=critic,
                actor_optimizer=bad_opt,  # type: ignore[arg-type]
            )

    def test_q_horde_actor_critic_rejects_non_true_supported_for_mlp(self) -> None:
        from alberta_framework.core.horde import HordeLearner

        spec = _sample_control_horde_spec()
        critic = HordeLearner(horde_spec=spec)
        config = NonlinearQHordeActorCriticConfig(
            n_actions=1,
        )
        bad_opt = _MockFalsySupportedOptimizer(initial_step_size=0.01)
        with pytest.raises(ValueError, match="does not support the MLP shape-generic"):
            NonlinearQHordeActorCriticAgent(
                config=config,
                critic=critic,
                actor_optimizer=bad_opt,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("q_control", (False, True))
    def test_actor_optimizer_config_adoption_does_not_invoke_truthiness(
        self, q_control: bool
    ) -> None:
        from alberta_framework.core.horde import HordeLearner

        if q_control:
            agent = NonlinearQHordeActorCriticAgent(
                NonlinearQHordeActorCriticConfig(n_actions=1),
                HordeLearner(_sample_control_horde_spec()),
            )
        else:
            agent = NonlinearHordeActorCriticAgent(
                NonlinearHordeActorCriticConfig(n_actions=2, value_head_index=0),
                HordeLearner(_sample_horde_spec()),
            )
        payload = agent.to_config()
        payload["actor_optimizer"] = _HostileOptimizerConfig(payload["actor_optimizer"])
        _HostileOptimizerConfig.calls = 0

        restored = type(agent).from_config(payload)

        assert restored.actor_optimizer.to_config() == agent.actor_optimizer.to_config()
        assert _HostileOptimizerConfig.calls == 0
