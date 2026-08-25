# mypy: disable-error-code="call-arg"
"""Development-only action-conditioned latent control lane.

The lane makes :class:`LatentWorldModel` an actual online controller: after a
short matched warm-up, predicted immediate rewards choose actions in the
recurring ``SwitchingTwoStateMDP``.  It is intentionally much smaller than the
visual, replay-based protocols in Dreamer-CDP, JEDI, and JEPA-WM.  Records are
permanently nonpromoting and retain negative outcomes.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import operator
import sys
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import alberta_framework.core.latent_world_model as latent_world_model_module
import alberta_framework.core.sarsa as sarsa_module
import alberta_framework.streams.closed_loop as closed_loop_module
from alberta_framework.benchmarks.development_provenance import (
    DevelopmentIdentity,
    collect_development_identity,
    identity_from_payload,
    require_current_identity,
)
from alberta_framework.core.latent_world_model import LatentWorldModel, LatentWorldModelConfig
from alberta_framework.core.sarsa import SARSAAgent, SARSAConfig
from alberta_framework.streams.closed_loop import SwitchingTwoStateConfig, SwitchingTwoStateMDP

ACTION_LATENT_SCHEMA = "asi.action_conditioned_latent.development.v1"
FROZEN_DEVELOPMENT_SEEDS = (1_575_000, 1_575_001, 1_575_002, 1_575_003)
FROZEN_ARM_IDS = (
    "latent_action_interactions",
    "latent_no_interactions",
    "latent_action_masked",
    "latent_decision_off",
    "mechanism_off",
    "sarsa_control",
)
PINNED_RESEARCH = MappingProxyType({
    "dreamer_cdp_paper": "arXiv:2603.07083v2",
    "dreamer_cdp_code": "a851fa3e3d70b624b094ee1810ad4bb602346092",
    "jedi_paper": "arXiv:2605.13013v1",
    "jepa_wm_paper": "arXiv:2512.24497v3",
    "jepa_wm_code": "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0",
})
PINNED_RESEARCH_ITEMS = tuple(sorted(PINNED_RESEARCH.items()))
_MAX_STEPS = 4096
_MAX_SEEDS = 16
_INT32_MAX = 2**31 - 1
WORKLOAD_REGISTRY = (
    ("arm_ids", FROZEN_ARM_IDS),
    ("development_seeds", FROZEN_DEVELOPMENT_SEEDS),
    ("max_steps", _MAX_STEPS),
    ("max_seeds", _MAX_SEEDS),
)


def _current_identity() -> DevelopmentIdentity:
    return collect_development_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(latent_world_model_module, sarsa_module, closed_loop_module),
        workload_registry=WORKLOAD_REGISTRY,
        paper_registry=PINNED_RESEARCH,
    )


def _exact_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _tree_nbytes(tree: object) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) > 4096:
        raise ValueError("state contains too many leaves")
    total = 0
    for leaf in leaves:
        try:
            array = np.asarray(leaf)
        except TypeError:
            # Typed Threefry keys deliberately reject NumPy conversion; their
            # exact persistent representation is the two underlying uint32s.
            array = np.asarray(jr.key_data(leaf))
        total += int(array.nbytes)
    if not 0 <= total <= _INT32_MAX:
        raise ValueError("state bytes exceed signed-int32 capacity")
    return total


def _trace_hash(values: Sequence[int | float], *, float_trace: bool = False) -> str:
    dtype = np.dtype("<f4") if float_trace else np.dtype("<i4")
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class ActionLatentProtocol:
    """Bounded development configuration; benchmark seeds are frozen."""

    steps: int = 256
    phase_length: int = 32
    warmup_steps: int = 16
    exploration_period: int = 4
    seeds: tuple[int, ...] = FROZEN_DEVELOPMENT_SEEDS

    def __post_init__(self) -> None:
        steps = _exact_int(self.steps, name="steps", minimum=4, maximum=_MAX_STEPS)
        phase_length = _exact_int(
            self.phase_length, name="phase_length", minimum=2, maximum=steps
        )
        _exact_int(self.warmup_steps, name="warmup_steps", minimum=1, maximum=steps)
        _exact_int(
            self.exploration_period, name="exploration_period", minimum=2, maximum=steps
        )
        if type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= _MAX_SEEDS:
            raise ValueError("seeds must be a bounded exact tuple")
        if self.seeds != FROZEN_DEVELOPMENT_SEEDS:
            raise ValueError("development seeds are frozen")
        if steps < 2 * phase_length:
            raise ValueError("steps must cover at least one A/B recurrence")
        for index, seed in enumerate(self.seeds):
            _exact_int(seed, name=f"seeds[{index}]", minimum=0, maximum=2**32 - 1)


@dataclasses.dataclass(frozen=True, slots=True)
class ActionLatentArmReceipt:
    """Strict compact receipt for one arm and seed."""

    arm_id: str
    seed: int
    return_sum: float
    late_return_sum: float
    mean_prequential_loss: float | None
    action_sha256: str
    reward_sha256: str
    environment_steps: int
    model_updates: int
    training_queries: int
    decision_queries: int
    persistent_mechanism_bytes: int
    persistent_environment_bytes: int
    negative_outcome_retained: bool

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in FROZEN_ARM_IDS:
            raise ValueError("unsupported arm_id")
        _exact_int(self.seed, name="seed", minimum=0, maximum=2**32 - 1)
        for name in ("return_sum", "late_return_sum"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be an exact finite float")
        loss = self.mean_prequential_loss
        if loss is not None and (type(loss) is not float or not math.isfinite(loss) or loss < 0):
            raise ValueError("mean_prequential_loss must be None or a finite nonnegative float")
        for name in ("action_sha256", "reward_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in (
            "environment_steps",
            "model_updates",
            "training_queries",
            "decision_queries",
            "persistent_mechanism_bytes",
            "persistent_environment_bytes",
        ):
            _exact_int(getattr(self, name), name=name, minimum=0, maximum=_INT32_MAX)
        if self.environment_steps < 1 or self.persistent_environment_bytes < 1:
            raise ValueError("environment resources must be positive")
        if self.model_updates != self.training_queries:
            raise ValueError("each model update must record its prequential training query")
        mechanism_enabled = self.arm_id != "mechanism_off"
        if mechanism_enabled != (self.persistent_mechanism_bytes > 0):
            raise ValueError("persistent resources disagree with arm mechanism")
        model_arm = self.arm_id.startswith("latent_")
        if model_arm != (self.model_updates > 0):
            raise ValueError("model updates disagree with arm mechanism")
        if self.negative_outcome_retained is not True:
            raise ValueError("negative outcomes must remain retained")


@dataclasses.dataclass(frozen=True, slots=True)
class ActionLatentResult:
    """Nonpromoting matched multi-arm development result."""

    schema: str
    protocol: ActionLatentProtocol
    research_pins: tuple[tuple[str, str], ...]
    arms: tuple[ActionLatentArmReceipt, ...]
    identity: DevelopmentIdentity
    development_only: bool = True
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != ACTION_LATENT_SCHEMA:
            raise ValueError("unsupported action-latent schema")
        if type(self.protocol) is not ActionLatentProtocol:
            raise ValueError("protocol must be an exact ActionLatentProtocol")
        if self.research_pins != PINNED_RESEARCH_ITEMS:
            raise ValueError("research pins differ from the audited roster")
        if type(self.identity) is not DevelopmentIdentity:
            raise ValueError("identity must be exact")
        expected = tuple(
            (arm_id, seed) for seed in self.protocol.seeds for arm_id in FROZEN_ARM_IDS
        )
        if type(self.arms) is not tuple or any(
            type(arm) is not ActionLatentArmReceipt for arm in self.arms
        ):
            raise ValueError("arms must be an exact receipt tuple")
        if tuple((arm.arm_id, arm.seed) for arm in self.arms) != expected:
            raise ValueError("arm/seed roster differs from the frozen matched order")
        if any(arm.environment_steps != self.protocol.steps for arm in self.arms):
            raise ValueError("all arms must consume the matched environment horizon")
        for arm in self.arms:
            if (
                not 0.0 <= arm.return_sum <= float(self.protocol.steps)
                or not arm.return_sum.is_integer()
                or not 0.0 <= arm.late_return_sum <= float(self.protocol.phase_length)
                or not arm.late_return_sum.is_integer()
                or arm.late_return_sum > arm.return_sum
            ):
                raise ValueError("arm returns differ from the exact environment reward lattice")
        by_seed = {
            seed: [arm for arm in self.arms if arm.seed == seed]
            for seed in self.protocol.seeds
        }
        for seed, receipts in by_seed.items():
            decision_off = receipts[3]
            mechanism_off = receipts[4]
            if (
                decision_off.action_sha256 != mechanism_off.action_sha256
                or decision_off.reward_sha256 != mechanism_off.reward_sha256
                or decision_off.return_sum != mechanism_off.return_sum
            ):
                raise ValueError("decision-off must have exact mechanism-off control parity")
            eligible_decisions = sum(
                step >= self.protocol.warmup_steps
                and step % self.protocol.exploration_period != 0
                for step in range(self.protocol.steps)
            )
            expected_decision_queries = (
                2 * eligible_decisions,
                2 * eligible_decisions,
                2 * eligible_decisions,
                0,
                0,
                self.protocol.steps + 1,
            )
            if tuple(arm.decision_queries for arm in receipts) != expected_decision_queries:
                raise ValueError("decision-query receipts differ from the frozen schedule")
            expected_updates = (
                self.protocol.steps,
                self.protocol.steps,
                self.protocol.steps,
                self.protocol.steps,
                0,
                0,
            )
            if tuple(arm.model_updates for arm in receipts) != expected_updates:
                raise ValueError("model-update receipts differ from the frozen schedule")
            expected_environment_bytes = _expected_environment_bytes(
                seed, phase_length=self.protocol.phase_length
            )
            for arm in receipts:
                if arm.persistent_environment_bytes != expected_environment_bytes:
                    raise ValueError("environment state bytes differ from the derived state")
                if arm.persistent_mechanism_bytes != _expected_mechanism_bytes(
                    arm.arm_id, seed=seed
                ):
                    raise ValueError("mechanism bytes differ from the derived state")
        if self.development_only is not True or self.scientific_promotion_allowed is not False:
            raise ValueError("action-latent results are permanently nonpromoting")

    def to_payload(self) -> dict[str, object]:
        """Return the canonical JSON-compatible receipt."""
        # Canonicalize tuples to JSON arrays now, rather than accepting two
        # subtly different in-memory encodings in the strict validator.
        raw = dataclasses.asdict(self)
        raw["research_pins"] = dict(self.research_pins)
        return cast(dict[str, object], json.loads(json.dumps(raw)))


def _research_pins_from_payload(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not dict:
        raise ValueError("research pins must be an exact object")
    pins = cast(dict[object, object], value)
    if any(type(key) is not str or type(item) is not str for key, item in pins.items()):
        raise ValueError("research pins must contain exact strings")
    return tuple(sorted(cast(dict[str, str], pins).items()))


def validate_action_latent_payload(payload: object) -> ActionLatentResult:
    """Fail closed on hostile or expanded result payloads."""
    if type(payload) is not dict:
        raise ValueError("payload must be an exact dict")
    root = cast(dict[object, object], payload)
    fields = {field.name for field in dataclasses.fields(ActionLatentResult)}
    if any(type(key) is not str for key in root) or set(root) != fields:
        raise ValueError("payload fields differ from the schema")
    protocol_raw = root["protocol"]
    if type(protocol_raw) is not dict or set(protocol_raw) != {
        "steps", "phase_length", "warmup_steps", "exploration_period", "seeds"
    }:
        raise ValueError("protocol payload differs from the schema")
    protocol_dict = cast(dict[str, object], protocol_raw)
    seeds = protocol_dict["seeds"]
    if type(seeds) is not list:
        raise ValueError("serialized seeds must be an exact list")
    protocol = ActionLatentProtocol(
        steps=protocol_dict["steps"],  # type: ignore[arg-type]
        phase_length=protocol_dict["phase_length"],  # type: ignore[arg-type]
        warmup_steps=protocol_dict["warmup_steps"],  # type: ignore[arg-type]
        exploration_period=protocol_dict["exploration_period"],  # type: ignore[arg-type]
        seeds=tuple(seeds),
    )
    arms_raw = root["arms"]
    if type(arms_raw) is not list or len(arms_raw) != len(FROZEN_ARM_IDS) * len(protocol.seeds):
        raise ValueError("serialized arms have the wrong bounded length")
    arm_fields = {field.name for field in dataclasses.fields(ActionLatentArmReceipt)}
    arms: list[ActionLatentArmReceipt] = []
    for item in arms_raw:
        if (
            type(item) is not dict
            or any(type(key) is not str for key in item)
            or set(item) != arm_fields
        ):
            raise ValueError("serialized arm differs from the schema")
        arms.append(ActionLatentArmReceipt(**cast(dict[str, Any], item)))
    result = ActionLatentResult(
        schema=root["schema"],  # type: ignore[arg-type]
        protocol=protocol,
        research_pins=_research_pins_from_payload(root["research_pins"]),
        arms=tuple(arms),
        identity=identity_from_payload(root["identity"]),
        development_only=root["development_only"],  # type: ignore[arg-type]
        scientific_promotion_allowed=root["scientific_promotion_allowed"],  # type: ignore[arg-type]
    )
    require_current_identity(result.identity, _current_identity())
    return result


def _latent_model(*, interactions: bool) -> LatentWorldModel:
    return LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=4,
            hidden_sizes=(),
            step_size=0.03,
            sparsity=0.0,
            use_layer_norm=False,
            include_action_interactions=interactions,
        )
    )


def _sarsa_agent() -> SARSAAgent:
    return SARSAAgent(
        SARSAConfig(
            n_actions=2,
            gamma=0.99,
            epsilon_start=0.1,
            epsilon_end=0.1,
            epsilon_decay_steps=1,
        ),
        hidden_sizes=(),
        sparsity=0.0,
        use_layer_norm=False,
    )


def _expected_environment_bytes(seed: int, *, phase_length: int) -> int:
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=phase_length))
    env_key, _mechanism_key = jr.split(jr.key(seed))
    return _tree_nbytes(env.init(env_key))


def _expected_mechanism_bytes(arm_id: str, *, seed: int) -> int:
    if arm_id == "mechanism_off":
        return 0
    _env_key, mechanism_key = jr.split(jr.key(seed))
    if arm_id.startswith("latent_"):
        model = _latent_model(interactions=arm_id != "latent_no_interactions")
        return _tree_nbytes(model.init(mechanism_key))
    if arm_id == "sarsa_control":
        return _tree_nbytes(_sarsa_agent().init(2, mechanism_key))
    raise ValueError("unsupported arm_id")


def _base_actions(seed: int, steps: int) -> tuple[int, ...]:
    values = jr.randint(jr.fold_in(jr.key(seed), 91), (steps,), 0, 2, dtype=jnp.int32)
    return tuple(int(value) for value in np.asarray(values))


@functools.partial(jax.jit, static_argnums=(0, 3))
def select_latent_action(
    model: LatentWorldModel,
    state: object,
    observation: jax.Array,
    action_masked: bool,
) -> jax.Array:
    """Choose a live action from two immediate-reward model queries."""
    model_state = cast(Any, state)
    second_action = 0 if action_masked else 1
    rewards = jnp.stack(
        (
            model.predict(model_state, observation, jnp.asarray(0, dtype=jnp.int32)).reward,
            model.predict(
                model_state, observation, jnp.asarray(second_action, dtype=jnp.int32)
            ).reward,
        )
    )
    return jnp.argmax(rewards).astype(jnp.int32)


def _run_latent_arm(
    protocol: ActionLatentProtocol,
    *,
    seed: int,
    arm_id: str,
) -> ActionLatentArmReceipt:
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=protocol.phase_length))
    env_key, model_key = jr.split(jr.key(seed))
    env_state = env.init(env_key)
    observation = env.observe(env_state)
    base_actions = _base_actions(seed, protocol.steps)
    interactions = arm_id != "latent_no_interactions"
    model = _latent_model(interactions=interactions)
    model_state = model.init(model_key)
    initial_bytes = _tree_nbytes(model_state)
    actions: list[int] = []
    rewards: list[float] = []
    losses: list[float] = []
    decision_queries = 0
    for step in range(protocol.steps):
        use_base = (
            arm_id == "latent_decision_off"
            or step < protocol.warmup_steps
            or step % protocol.exploration_period == 0
        )
        if use_base:
            action = base_actions[step]
        else:
            decision_queries += 2
            action = int(
                select_latent_action(
                    model, model_state, observation, arm_id == "latent_action_masked"
                )
            )
        next_observation, reward, env_state = env.step(
            env_state, jnp.asarray(action, dtype=jnp.int32), jr.fold_in(env_key, step)
        )
        model_action = 0 if arm_id == "latent_action_masked" else action
        update = model.update(
            model_state,
            observation,
            jnp.asarray(model_action, dtype=jnp.int32),
            reward,
            jnp.asarray(0.99, dtype=jnp.float32),
            next_observation,
        )
        model_state = update.state
        observation = next_observation
        actions.append(action)
        rewards.append(float(reward))
        losses.append(float(update.prediction_error))
    return ActionLatentArmReceipt(
        arm_id=arm_id,
        seed=seed,
        return_sum=float(sum(rewards)),
        late_return_sum=float(sum(rewards[-protocol.phase_length :])),
        mean_prequential_loss=float(np.mean(np.asarray(losses, dtype=np.float64))),
        action_sha256=_trace_hash(actions),
        reward_sha256=_trace_hash(rewards, float_trace=True),
        environment_steps=protocol.steps,
        model_updates=protocol.steps,
        training_queries=protocol.steps,
        decision_queries=decision_queries,
        persistent_mechanism_bytes=max(initial_bytes, _tree_nbytes(model_state)),
        persistent_environment_bytes=_tree_nbytes(env_state),
        negative_outcome_retained=True,
    )


def _run_mechanism_off(
    protocol: ActionLatentProtocol, *, seed: int
) -> ActionLatentArmReceipt:
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=protocol.phase_length))
    env_key, _unused_model_key = jr.split(jr.key(seed))
    state = env.init(env_key)
    actions = _base_actions(seed, protocol.steps)
    rewards: list[float] = []
    for step, action in enumerate(actions):
        _observation, reward, state = env.step(
            state, jnp.asarray(action, dtype=jnp.int32), jr.fold_in(env_key, step)
        )
        rewards.append(float(reward))
    return ActionLatentArmReceipt(
        arm_id="mechanism_off",
        seed=seed,
        return_sum=float(sum(rewards)),
        late_return_sum=float(sum(rewards[-protocol.phase_length :])),
        mean_prequential_loss=None,
        action_sha256=_trace_hash(actions),
        reward_sha256=_trace_hash(rewards, float_trace=True),
        environment_steps=protocol.steps,
        model_updates=0,
        training_queries=0,
        decision_queries=0,
        persistent_mechanism_bytes=0,
        persistent_environment_bytes=_tree_nbytes(state),
        negative_outcome_retained=True,
    )


def _run_sarsa(protocol: ActionLatentProtocol, *, seed: int) -> ActionLatentArmReceipt:
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=protocol.phase_length))
    env_key, agent_key = jr.split(jr.key(seed))
    env_state = env.init(env_key)
    observation = env.observe(env_state)
    agent = _sarsa_agent()
    state = agent.init(2, agent_key)
    initial_bytes = _tree_nbytes(state)
    action, next_key = agent.select_action(state, observation)
    state = state.replace(last_action=action, last_observation=observation, rng_key=next_key)  # type: ignore[attr-defined]
    actions: list[int] = []
    rewards: list[float] = []
    for step in range(protocol.steps):
        next_observation, reward, env_state = env.step(
            env_state, action, jr.fold_in(env_key, step)
        )
        next_action, next_key = agent.select_action(state, next_observation)
        update = agent.update(
            state,
            reward=reward,
            observation=next_observation,
            terminated=jnp.asarray(False),
            next_action=next_action,
        )
        actions.append(int(action))
        rewards.append(float(reward))
        state = update.state.replace(
            last_action=next_action,
            last_observation=next_observation,
            rng_key=next_key,
        )
        observation = next_observation
        action = next_action
    del observation
    return ActionLatentArmReceipt(
        arm_id="sarsa_control",
        seed=seed,
        return_sum=float(sum(rewards)),
        late_return_sum=float(sum(rewards[-protocol.phase_length :])),
        mean_prequential_loss=None,
        action_sha256=_trace_hash(actions),
        reward_sha256=_trace_hash(rewards, float_trace=True),
        environment_steps=protocol.steps,
        model_updates=0,
        training_queries=0,
        decision_queries=protocol.steps + 1,
        persistent_mechanism_bytes=max(initial_bytes, _tree_nbytes(state)),
        persistent_environment_bytes=_tree_nbytes(env_state),
        negative_outcome_retained=True,
    )


def run_action_conditioned_latent_lane(
    protocol: ActionLatentProtocol | None = None,
) -> ActionLatentResult:
    """Execute all frozen arms/seeds without promoting or writing artifacts."""
    protocol = ActionLatentProtocol() if protocol is None else protocol
    if type(protocol) is not ActionLatentProtocol:
        raise ValueError("protocol must be an exact ActionLatentProtocol")
    receipts: list[ActionLatentArmReceipt] = []
    for seed in protocol.seeds:
        latent_receipts = {
            arm_id: _run_latent_arm(protocol, seed=seed, arm_id=arm_id)
            for arm_id in FROZEN_ARM_IDS[:4]
        }
        off = _run_mechanism_off(protocol, seed=seed)
        decision_off = latent_receipts["latent_decision_off"]
        # The model trains behind an explicitly disabled decision interface;
        # its environment transcript must reduce exactly to mechanism-off.
        if (
            decision_off.action_sha256 != off.action_sha256
            or decision_off.reward_sha256 != off.reward_sha256
        ):
            raise RuntimeError("mechanism-off transcript parity failed")
        receipts.extend((*latent_receipts.values(), off, _run_sarsa(protocol, seed=seed)))
    return ActionLatentResult(
        schema=ACTION_LATENT_SCHEMA,
        protocol=protocol,
        research_pins=PINNED_RESEARCH_ITEMS,
        arms=tuple(receipts),
        identity=_current_identity(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded frozen development lane and emit a validated receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--phase-length", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--exploration-period", type=int, default=4)
    args = parser.parse_args(argv)
    result = run_action_conditioned_latent_lane(
        ActionLatentProtocol(
            steps=args.steps,
            phase_length=args.phase_length,
            warmup_steps=args.warmup_steps,
            exploration_period=args.exploration_period,
        )
    )
    payload = result.to_payload()
    validate_action_latent_payload(payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTION_LATENT_SCHEMA",
    "FROZEN_ARM_IDS",
    "FROZEN_DEVELOPMENT_SEEDS",
    "PINNED_RESEARCH",
    "ActionLatentArmReceipt",
    "ActionLatentProtocol",
    "ActionLatentResult",
    "run_action_conditioned_latent_lane",
    "select_latent_action",
    "validate_action_latent_payload",
]
