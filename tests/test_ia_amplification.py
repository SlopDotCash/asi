"""Development diagnostic for closed-loop Intelligence Amplification.

Step 12 asks whether an IA agent *increases a partner agent's decision-making
capacity*.  This module probes that outcome in a narrow closed loop, with the
partner in control of execution and the exo-cortex credited on the partner's
EXECUTED action (``IAAgent.update(..., partner_action=effective_action)``).

The 12 seeds, intervention probabilities, hyperparameters, and assertion
margins below are development/calibration evidence.  They are not a promoted
Step 12 result: there is no held-out seed schedule, confidence interval,
versioned artifact, resource match, or independent environment family.  In
particular, the intervention-probability sweep was inspected before the
``p=0.5`` condition was selected.  A fail-closed held-out evaluation must
freeze that choice and compare paired interventions before this result can
support an L2 claim.

Protocol
--------
Partner: linear :class:`DifferentialSARSAAgent` (q_step_size=0.02, epsilon=0.1)
on :class:`SwitchingTwoStateMDP` (phase_length=200, payoffs invert each phase,
optimal average reward 1.0 in both phases, uniform-random policy earns 0.5).
The phase is hidden, so re-adaptation speed after each switch dominates the
average reward over 1200 steps (6 phases).  Arms are paired over 12 seeds
(``jr.key(0..11)``); every arm consumes an identical RNG-split pattern, and the
environment is deterministic, so ignore-always reproduces the partner-alone
trajectory bitwise.

Arms:

* ``alone``      — partner by itself.
* ``rec(p)``     — partner + IAAgent; each step the partner proposes its own
  epsilon-greedy action, the exo-cortex (OaK, base_step_size=0.3, greedy
  recommendation) recommends, and :func:`update_recommendation_protocol`
  selects the executed action with acceptance probability ``p``.  The cortex is
  credited on the executed action.
* ``aug``        — partner + IAAgent; the partner's features are
  ``[obs, exo-cerebellum predictions]``.  Under a policy the cerebellum's
  next-obs predictions encode the recent action->state structure — a proxy for
  the hidden phase — so they are genuinely informative features.
* ``aug-noise``  — dimensionality control: predictions replaced by uniform
  noise of the same shape.

Measured calibration (2026-07-30, 12 seeds, CPU JAX, mean reward per step)
--------------------------------------------------------------------------
==========================  ======  ======  ======  =========================
arm                          mean    std     min    pairdiff vs alone
                                                    (mean / min / max)
==========================  ======  ======  ======  =========================
alone                       0.5572  0.0359  0.4792  —
rec p=0.00 (ignore-always)  0.5572  0.0359  0.4792  0 / 0 / 0 (bitwise equal)
rec p=0.25                  0.7381  0.0272  0.6942  +0.1809 / +0.1042 / +0.2367
rec p=0.50                  0.8035  0.0256  0.7583  +0.2464 / +0.1633 / +0.3092
rec p=0.75                  0.8215  0.0582  0.6742  +0.2644 / +0.1358 / +0.3575
rec p=0.90                  0.6415  0.1597  0.3117  +0.0844 / -0.2375 / +0.3708
rec p=1.00 (accept-always)  0.4880  0.0365  0.3675  -0.0692 / -0.1167 / -0.0217
aug (cerebellum preds)      0.7111  0.0196  0.6850  +0.1540 / +0.1100 / +0.2242
aug (uniform noise)         0.5913  0.0315  0.5292  +0.0341 / -0.0092 / +0.0817
==========================  ======  ======  ======  =========================

Findings (all asserted below with >=2x margins):

1. **Recommendation amplification** — shared agency at p=0.5 lifts the
   partner's mean reward by +0.2464 (worst seed +0.1633) over partner-alone.
2. **Observation amplification** — cerebellum-prediction features lift mean
   reward by +0.1540 (worst seed +0.1100); the noise control shows the gain
   comes from prediction content (+0.0341 only), not extra dimensions.
   Both wirings improve this development score; neither is yet a held-out
   acceptance result.
3. **Value correlates with following** — mean reward rises monotonically over
   acceptance p in {0, 0.25, 0.5}; ignoring the channel (p=0) leaves the
   partner exactly at baseline.
4. **Full delegation collapses** — at p=1.0 the executed stream contains no
   exploration (the recommendation is a greedy argmax), the cortex self-loops,
   and reward drops to the random-policy level (0.4880 ~= 0.5).  The channel's
   value requires retained partner exploration: amplification is a property of
   the *pair*, not of substituting the cortex for the partner.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)
from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAAgent,
    IAConfig,
    RecommendationProtocolConfig,
    RecommendationProtocolState,
    init_recommendation_protocol_state,
    update_recommendation_protocol,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.streams.closed_loop import (
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

pytestmark = pytest.mark.slow

_OBS_DIM = 2
_N_ACTIONS = 2
_N_DEMONS = 2
_NUM_STEPS = 1200
_PHASE_LENGTH = 200
_N_SEEDS = 12

# Assertion margins, all >=2x away from the measured values in the module
# docstring (measured -> threshold):
_REC_MEAN_GAIN = 0.12  # measured +0.2464
_REC_MIN_GAIN = 0.08  # measured worst-seed +0.1633
_AUG_MEAN_GAIN = 0.07  # measured +0.1540
_AUG_MIN_GAIN = 0.04  # measured worst-seed +0.1100
_DOSE_LOW_GAP = 0.09  # rec(0.25) - rec(0): measured +0.1809
_DOSE_HIGH_GAP = 0.02  # rec(0.5) - rec(0.25): measured +0.0654
_NOISE_CONTROL_GAP = 0.06  # aug - aug_noise: measured +0.1198
_ACCEPT_ALWAYS_MEAN_FLOOR = 0.40  # measured 0.4880 (random policy earns 0.5)
_ACCEPT_ALWAYS_SEED_FLOOR = 0.25  # measured worst-seed 0.3675
_ACCEPT_ALWAYS_SHARED_GAP = 0.15  # rec(0.5) - rec(1.0): measured +0.3155


def _make_partner() -> DifferentialSARSAAgent:
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=_N_ACTIONS,
            q_step_size=0.02,
            average_reward_step_size=0.01,
            epsilon_start=0.1,
            epsilon_end=0.1,
        )
    )


def _make_ia() -> IAAgent:
    stomp = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=_OBS_DIM,
        n_primitive_actions=_N_ACTIONS,
        base_step_size=0.3,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )
    cerebellum = ExoCerebellumConfig(n_demons=_N_DEMONS, obs_dim=_OBS_DIM)
    return IAAgent(IAConfig(cerebellum=cerebellum, cortex=OaKConfig(stomp=stomp)))


def _make_env() -> SwitchingTwoStateMDP:
    return SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=_PHASE_LENGTH))


def _run_alone(env: SwitchingTwoStateMDP, partner: DifferentialSARSAAgent, key: Array) -> Array:
    """Partner-alone closed loop; mirrors the rec arm's RNG-split pattern."""
    k_env, k_agent, _k_ia, k_steps = jr.split(key, 4)
    env_state = env.init(k_env)
    obs0 = env.observe(env_state)
    p_state = partner.init(_OBS_DIM, k_agent)
    p_state, _ = partner.start(p_state, obs0)

    def step(carry: tuple, k: Array) -> tuple[tuple, Array]:
        env_state, p_state = carry
        k_env_step, _k_accept = jr.split(k)
        next_obs, reward, env_state = env.step(env_state, p_state.last_action, k_env_step)
        proposal, p_key = partner.select_action(p_state, next_obs)
        p_state = p_state.replace(rng_key=p_key)
        p_result = partner.update(p_state, reward, next_obs, next_action=proposal)
        return (env_state, p_result.state), reward

    _, rewards = jax.lax.scan(step, (env_state, p_state), jr.split(k_steps, _NUM_STEPS))
    return rewards


def _run_recommendation(
    env: SwitchingTwoStateMDP,
    partner: DifferentialSARSAAgent,
    ia: IAAgent,
    key: Array,
    p_accept: float,
) -> tuple[Array, RecommendationProtocolState]:
    """Partner + IA recommendation channel with acceptance probability ``p_accept``.

    The partner proposes its own action, the recommendation protocol picks the
    executed action, and the exo-cortex is credited on that executed action.
    """
    proto_config = RecommendationProtocolConfig()
    k_env, k_agent, k_ia, k_steps = jr.split(key, 4)
    env_state = env.init(k_env)
    obs0 = env.observe(env_state)
    p_state = partner.init(_OBS_DIM, k_agent)
    p_state, _ = partner.start(p_state, obs0)
    ia_state = ia.start(ia.init(k_ia), obs0)
    proto_state = init_recommendation_protocol_state()

    def step(carry: tuple, k: Array) -> tuple[tuple, tuple[Array, Array]]:
        env_state, p_state, ia_state, proto_state = carry
        k_env_step, k_accept = jr.split(k)
        obs = p_state.last_observation
        action = p_state.last_action  # effective action decided on the previous step
        next_obs, reward, env_state = env.step(env_state, action, k_env_step)
        ia_result = ia.update(ia_state, obs, reward, next_obs, partner_action=action)
        proposal, p_key = partner.select_action(p_state, next_obs)
        p_state = p_state.replace(rng_key=p_key)
        accept = jr.uniform(k_accept) < p_accept
        proto_result = update_recommendation_protocol(
            proto_config, proto_state, ia_result.recommendation, proposal, accept
        )
        p_result = partner.update(
            p_state, reward, next_obs, next_action=proto_result.effective_action
        )
        carry = (env_state, p_result.state, ia_result.state, proto_result.state)
        return carry, (reward, proto_result.accepted)

    (_, _, _, proto_final), (rewards, _) = jax.lax.scan(
        step,
        (env_state, p_state, ia_state, proto_state),
        jr.split(k_steps, _NUM_STEPS),
    )
    return rewards, proto_final


def _run_augmented(
    env: SwitchingTwoStateMDP,
    partner: DifferentialSARSAAgent,
    ia: IAAgent,
    key: Array,
    noise_features: bool,
) -> Array:
    """Partner acting on ``[obs, cerebellum predictions]`` features.

    With ``noise_features=True`` the predictions are replaced by uniform noise
    of the same shape — the dimensionality control.
    """
    k_env, k_agent, k_ia, k_steps = jr.split(key, 4)
    env_state = env.init(k_env)
    obs0 = env.observe(env_state)
    p_state = partner.init(_OBS_DIM + _N_DEMONS, k_agent)
    ia_state = ia.start(ia.init(k_ia), obs0)
    preds0 = ia.cerebellum.predict(ia_state.cerebellum_state, obs0)
    p_state, _ = partner.start(p_state, jnp.concatenate([obs0, preds0]))

    def step(carry: tuple, k: Array) -> tuple[tuple, Array]:
        env_state, p_state, ia_state, raw_obs = carry
        k_env_step, k_noise = jr.split(k)
        action = p_state.last_action
        next_obs, reward, env_state = env.step(env_state, action, k_env_step)
        ia_result = ia.update(ia_state, raw_obs, reward, next_obs, partner_action=action)
        preds = ia.cerebellum.predict(ia_result.state.cerebellum_state, next_obs)
        if noise_features:
            preds = jr.uniform(k_noise, (_N_DEMONS,))
        p_result = partner.update(p_state, reward, jnp.concatenate([next_obs, preds]))
        return (env_state, p_result.state, ia_result.state, next_obs), reward

    _, rewards = jax.lax.scan(
        step, (env_state, p_state, ia_state, obs0), jr.split(k_steps, _NUM_STEPS)
    )
    return rewards


@functools.lru_cache(maxsize=1)
def _experiment() -> dict:
    """Run all arms over paired seeds once and cache the per-seed results."""
    env = _make_env()
    partner = _make_partner()
    partner_aug = _make_partner()
    ia = _make_ia()
    seeds = [jr.key(s) for s in range(_N_SEEDS)]

    alone_fn = jax.jit(functools.partial(_run_alone, env, partner))
    alone_rewards = jnp.stack([alone_fn(k) for k in seeds])

    rec_rewards: dict[float, Array] = {}
    rec_protocols: dict[float, list[RecommendationProtocolState]] = {}
    for p_accept in (0.0, 0.25, 0.5, 1.0):
        rec_fn = jax.jit(
            functools.partial(_run_recommendation, env, partner, ia, p_accept=p_accept)
        )
        outputs = [rec_fn(k) for k in seeds]
        rec_rewards[p_accept] = jnp.stack([rewards for rewards, _ in outputs])
        rec_protocols[p_accept] = [proto for _, proto in outputs]

    aug_fn = jax.jit(functools.partial(_run_augmented, env, partner_aug, ia, noise_features=False))
    noise_fn = jax.jit(functools.partial(_run_augmented, env, partner_aug, ia, noise_features=True))
    aug_rewards = jnp.stack([aug_fn(k) for k in seeds])
    noise_rewards = jnp.stack([noise_fn(k) for k in seeds])

    return {
        "alone": alone_rewards,
        "rec": rec_rewards,
        "protocols": rec_protocols,
        "aug": aug_rewards,
        "aug_noise": noise_rewards,
    }


def _per_seed_means(rewards: Array) -> Array:
    return rewards.mean(axis=1)


# ---------------------------------------------------------------------------
# Development probe: the IA agent raises the partner's average reward
# ---------------------------------------------------------------------------


def test_recommendation_channel_amplifies_partner() -> None:
    """Shared agency (p=0.5) beats partner-alone on every paired seed.

    Measured: pairdiff mean +0.2464, worst seed +0.1633; thresholds hold >=2x
    margins (0.12 / 0.08).
    """
    data = _experiment()
    alone = _per_seed_means(data["alone"])
    rec = _per_seed_means(data["rec"][0.5])
    assert bool(jnp.all(jnp.isfinite(alone))) and bool(jnp.all(jnp.isfinite(rec)))

    diff = rec - alone
    assert float(diff.mean()) > _REC_MEAN_GAIN, (
        f"recommendation amplification too small: mean pairdiff {float(diff.mean()):.4f}"
    )
    assert float(diff.min()) > _REC_MIN_GAIN, (
        f"recommendation amplification not seed-robust: worst pairdiff {float(diff.min()):.4f}"
    )


def test_observation_augmentation_amplifies_partner() -> None:
    """Cerebellum-prediction features beat partner-alone on every paired seed.

    Measured: pairdiff mean +0.1540, worst seed +0.1100; thresholds hold >=2x
    margins (0.07 / 0.04).  On this calibrated seed set the second wiring is
    not merely non-degrading; it also improves the measured score.
    """
    data = _experiment()
    alone = _per_seed_means(data["alone"])
    aug = _per_seed_means(data["aug"])
    assert bool(jnp.all(jnp.isfinite(aug)))

    diff = aug - alone
    assert float(diff.mean()) > _AUG_MEAN_GAIN, (
        f"augmentation amplification too small: mean pairdiff {float(diff.mean()):.4f}"
    )
    assert float(diff.min()) > _AUG_MIN_GAIN, (
        f"augmentation amplification not seed-robust: worst pairdiff {float(diff.min()):.4f}"
    )


def test_augmentation_gain_is_prediction_content_not_dimensionality() -> None:
    """Uniform-noise features of the same shape do not reproduce the gain.

    Measured: aug - aug_noise mean +0.1198 (threshold 0.06, 2x); noise arm's
    own gain over alone is only +0.0341.
    """
    data = _experiment()
    aug = _per_seed_means(data["aug"])
    noise = _per_seed_means(data["aug_noise"])
    assert bool(jnp.all(jnp.isfinite(noise)))
    assert float((aug - noise).mean()) > _NOISE_CONTROL_GAP, (
        f"prediction features do not beat noise features: gap {float((aug - noise).mean()):.4f}"
    )


# ---------------------------------------------------------------------------
# The channel's value correlates with following it
# ---------------------------------------------------------------------------


def test_ignore_always_is_exactly_partner_alone() -> None:
    """With p=0 the IA agent observes but never intervenes.

    The reward trajectory must be bitwise identical to partner-alone on every
    seed (deterministic env, identical RNG-split pattern): attaching the IA
    agent in observe-only mode does not perturb the partner.
    """
    data = _experiment()
    assert jnp.array_equal(data["rec"][0.0], data["alone"])


def test_value_correlates_with_following_the_recommendations() -> None:
    """Mean reward rises monotonically over acceptance p in {0, 0.25, 0.5}.

    Measured means: 0.5572 (p=0) -> 0.7381 (p=0.25) -> 0.8035 (p=0.5); the
    asserted gaps (0.09, 0.02) hold 2x/3x margins.
    """
    data = _experiment()
    mean_at = {p: float(_per_seed_means(rews).mean()) for p, rews in data["rec"].items()}
    assert mean_at[0.25] - mean_at[0.0] > _DOSE_LOW_GAP, (
        f"following 25% of recommendations did not help: {mean_at}"
    )
    assert mean_at[0.5] - mean_at[0.25] > _DOSE_HIGH_GAP, (
        f"following more recommendations did not help more: {mean_at}"
    )


def test_accept_always_remains_finite_but_underperforms_shared_agency() -> None:
    """Full delegation (p=1.0) is measurably worse than shared agency.

    The recommendation is a greedy argmax, so an accept-always partner
    contributes no exploration to the executed stream and the cortex
    self-loops at the random-policy level (measured mean 0.4880, worst seed
    0.3675 — finite over this 1200-step horizon).  Shared agency at p=0.5 beats
    it by +0.3155 (threshold 0.15, 2x): the development result comes from the
    pair, not from replacing the partner with the cortex.
    """
    data = _experiment()
    accept_always = _per_seed_means(data["rec"][1.0])
    shared = _per_seed_means(data["rec"][0.5])
    assert bool(jnp.all(jnp.isfinite(accept_always)))

    assert float(accept_always.mean()) > _ACCEPT_ALWAYS_MEAN_FLOOR
    assert float(accept_always.min()) > _ACCEPT_ALWAYS_SEED_FLOOR
    assert float(shared.mean() - accept_always.mean()) > _ACCEPT_ALWAYS_SHARED_GAP, (
        "shared agency should beat full delegation"
    )


# ---------------------------------------------------------------------------
# Protocol accounting
# ---------------------------------------------------------------------------


def test_recommendation_protocol_accounting() -> None:
    """Acceptance counters match the configured acceptance probabilities."""
    data = _experiment()
    for proto in data["protocols"][0.5]:
        accepted = int(proto.accepted_count)
        assert accepted + int(proto.rejected_count) == _NUM_STEPS
        assert int(proto.step_count) == _NUM_STEPS
        # Binomial(1200, 0.5) std is ~17 accepts (~0.014 as a rate); the
        # +-0.1 rate band is a ~7x margin.
        assert 0.4 * _NUM_STEPS < accepted < 0.6 * _NUM_STEPS
    for proto in data["protocols"][1.0]:
        assert int(proto.accepted_count) == _NUM_STEPS
        assert int(proto.rejected_count) == 0
    for proto in data["protocols"][0.0]:
        assert int(proto.accepted_count) == 0
        assert int(proto.rejected_count) == _NUM_STEPS


def test_ia_configs_reject_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="n_demons"):
        ExoCerebellumConfig(n_demons=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="obs_dim"):
        ExoCerebellumConfig(obs_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_demons"):
        ExoCerebellumConfig(n_demons=4.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="obs_dim"):
        ExoCerebellumConfig(obs_dim=4.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="acceptance_ema_decay"):
        RecommendationProtocolConfig(acceptance_ema_decay=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="acceptance_ema_decay"):
        RecommendationProtocolConfig(acceptance_ema_decay=1.0 - 1e-10)


def test_ia_configs_accept_and_canonicalize_numpy_integers() -> None:
    cfg = ExoCerebellumConfig(
        n_demons=np.int32(8),
        obs_dim=np.uint16(16),
    )
    assert type(cfg.n_demons) is int
    assert type(cfg.obs_dim) is int
    assert cfg.n_demons == 8
    assert cfg.obs_dim == 16

    protocol = RecommendationProtocolConfig(acceptance_ema_decay=np.float32(0.5))
    assert type(protocol.acceptance_ema_decay) is float


@pytest.mark.parametrize("field", ["n_demons", "obs_dim"])
@pytest.mark.parametrize("value", [0, -1, 2**31, np.uint64(2**63)])
def test_exo_cerebellum_dimensions_use_signed_int32_domain(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        ExoCerebellumConfig(**{field: value})


def test_ia_configs_reject_class_spoofed_scalars() -> None:
    class Spoof:
        @property
        def __class__(self):
            return int

        def __index__(self):
            return 4

    with pytest.raises(ValueError, match="n_demons"):
        ExoCerebellumConfig(n_demons=Spoof())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="acceptance_ema_decay"):
        RecommendationProtocolConfig(  # type: ignore[arg-type]
            acceptance_ema_decay=Spoof()
        )
