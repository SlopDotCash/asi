# mypy: disable-error-code="call-arg"
"""Bounded JEPA encoder-transfer feasibility lane.

This native lane does not execute V-JEPA 2 or JEPA-WM.  It isolates the
architectural question their results raise: can a predictor receive a frozen
encoder learned on earlier ASI-only transitions and improve live control?  All
imported-pretraining counters are fixed to zero and every result is permanently
nonpromoting.
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
import time
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

JEPA_TRANSFER_SCHEMA = "asi.jepa_transfer_feasibility.development.v1"
FROZEN_DEVELOPMENT_SEEDS = (1_577_000, 1_577_001, 1_577_002, 1_577_003)
FROZEN_ARM_IDS = (
    "asi_encoder_transfer",
    "no_pretraining",
    "encoder_permuted",
    "full_warm_start_ceiling",
    "transfer_decision_off",
    "mechanism_off",
    "sarsa_control",
)
PINNED_RESEARCH = MappingProxyType({
    "jepa_wm_paper": "arXiv:2512.24497v3",
    "jepa_wm_code": "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0",
    "vjepa2_paper": "arXiv:2506.09985v1",
    "vjepa2_code": "204698b45b3712590f06245fbfba32d3be539812",
})
PINNED_RESEARCH_ITEMS = tuple(sorted(PINNED_RESEARCH.items()))
_MAX_STEPS = 4096
_INT32_MAX = 2**31 - 1
_LATENT_DIM = 4
WORKLOAD_REGISTRY = (
    ("arm_ids", FROZEN_ARM_IDS),
    ("development_seeds", FROZEN_DEVELOPMENT_SEEDS),
    ("latent_dim", _LATENT_DIM),
    ("max_steps", _MAX_STEPS),
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
            array = np.asarray(jr.key_data(leaf))
        total += int(array.nbytes)
    if not 0 <= total <= _INT32_MAX:
        raise ValueError("state bytes exceed signed-int32 capacity")
    return total


def _sha(values: Sequence[int | float], *, floats: bool = False) -> str:
    dtype = np.dtype("<f4") if floats else np.dtype("<i4")
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class JEPATransferProtocol:
    """Frozen matched development axes."""

    steps: int = 256
    phase_length: int = 32
    pretraining_steps: int = 32
    warmup_steps: int = 16
    exploration_period: int = 4
    seeds: tuple[int, ...] = FROZEN_DEVELOPMENT_SEEDS

    def __post_init__(self) -> None:
        steps = _exact_int(self.steps, name="steps", minimum=4, maximum=_MAX_STEPS)
        phase = _exact_int(self.phase_length, name="phase_length", minimum=2, maximum=steps)
        _exact_int(
            self.pretraining_steps,
            name="pretraining_steps",
            minimum=1,
            maximum=_MAX_STEPS,
        )
        _exact_int(self.warmup_steps, name="warmup_steps", minimum=1, maximum=steps)
        _exact_int(
            self.exploration_period, name="exploration_period", minimum=2, maximum=steps
        )
        if steps < 2 * phase:
            raise ValueError("steps must cover the matched A/B recurrence")
        if type(self.seeds) is not tuple or self.seeds != FROZEN_DEVELOPMENT_SEEDS:
            raise ValueError("development seeds are frozen")


@dataclasses.dataclass(frozen=True, slots=True)
class TransferResources:
    """Exact semantic counters and persistent payload sizes."""

    environment_steps: int
    pretraining_examples: int
    pretraining_updates: int
    pretraining_encoder_updates: int
    imported_pretraining_bytes: int
    pretraining_replay_bytes: int
    online_replay_bytes: int
    encoder_queries: int
    model_queries: int
    control_queries: int
    encoder_bytes: int
    persistent_mechanism_bytes: int
    persistent_environment_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _exact_int(getattr(self, field.name), name=field.name, minimum=0, maximum=_INT32_MAX)
        if self.environment_steps < 1 or self.persistent_environment_bytes < 1:
            raise ValueError("environment resources must be positive")
        if self.imported_pretraining_bytes != 0 or self.online_replay_bytes != 0:
            raise ValueError("this lane forbids imported pretraining and online replay")
        if self.pretraining_examples != self.pretraining_updates:
            raise ValueError("pretraining consumes each stored example exactly once")
        if self.pretraining_encoder_updates > self.pretraining_updates:
            raise ValueError("encoder updates cannot exceed pretraining updates")


@dataclasses.dataclass(frozen=True, slots=True)
class TimingTelemetry:
    """Process-local telemetry; never an acceptance or comparison metric."""

    pretraining_ns: int
    environment_ns: int
    decision_query_ns: int
    online_update_ns: int
    control_ns: int
    clock: str = "perf_counter_ns_process_local_telemetry_only"

    def __post_init__(self) -> None:
        for name in (
            "pretraining_ns",
            "environment_ns",
            "decision_query_ns",
            "online_update_ns",
            "control_ns",
        ):
            _exact_int(getattr(self, name), name=name, minimum=0, maximum=2**63 - 1)
        if (
            type(self.clock) is not str
            or self.clock != "perf_counter_ns_process_local_telemetry_only"
        ):
            raise ValueError("unsupported timing scope")


@dataclasses.dataclass(frozen=True, slots=True)
class TransferArmReceipt:
    arm_id: str
    seed: int
    return_sum: float
    late_return_sum: float
    mean_prediction_loss: float | None
    action_sha256: str
    reward_sha256: str
    resources: TransferResources
    timing: TimingTelemetry
    negative_outcome_retained: bool = True

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in FROZEN_ARM_IDS:
            raise ValueError("unsupported arm_id")
        _exact_int(self.seed, name="seed", minimum=0, maximum=2**32 - 1)
        for name in ("return_sum", "late_return_sum"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be an exact finite float")
        loss = self.mean_prediction_loss
        if loss is not None and (type(loss) is not float or not math.isfinite(loss) or loss < 0):
            raise ValueError("mean_prediction_loss must be finite and nonnegative or None")
        for name in ("action_sha256", "reward_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if (
            type(self.resources) is not TransferResources
            or type(self.timing) is not TimingTelemetry
        ):
            raise ValueError("nested receipts must use exact schema types")
        if self.negative_outcome_retained is not True:
            raise ValueError("negative outcomes must remain retained")


@dataclasses.dataclass(frozen=True, slots=True)
class JEPATransferResult:
    schema: str
    protocol: JEPATransferProtocol
    research_pins: tuple[tuple[str, str], ...]
    arms: tuple[TransferArmReceipt, ...]
    identity: DevelopmentIdentity
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    visual_robotics_parity_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != JEPA_TRANSFER_SCHEMA:
            raise ValueError("unsupported JEPA-transfer schema")
        if type(self.protocol) is not JEPATransferProtocol:
            raise ValueError("protocol must be exact")
        if self.research_pins != PINNED_RESEARCH_ITEMS:
            raise ValueError("research pins differ from the independently audited roster")
        if type(self.identity) is not DevelopmentIdentity:
            raise ValueError("identity must be exact")
        expected = tuple(
            (arm_id, seed) for seed in self.protocol.seeds for arm_id in FROZEN_ARM_IDS
        )
        if type(self.arms) is not tuple or any(
            type(arm) is not TransferArmReceipt for arm in self.arms
        ):
            raise ValueError("arms must be exact receipts")
        if tuple((arm.arm_id, arm.seed) for arm in self.arms) != expected:
            raise ValueError("arm/seed roster differs from the frozen order")
        for seed_index in range(len(self.protocol.seeds)):
            group = self.arms[
                seed_index * len(FROZEN_ARM_IDS) : (seed_index + 1) * len(FROZEN_ARM_IDS)
            ]
            if any(arm.resources.environment_steps != self.protocol.steps for arm in group):
                raise ValueError("environment horizons must match")
            if len({arm.resources.persistent_environment_bytes for arm in group}) != 1:
                raise ValueError("environment state bytes must match")
            decision_off, mechanism_off = group[4], group[5]
            if (
                decision_off.action_sha256 != mechanism_off.action_sha256
                or decision_off.reward_sha256 != mechanism_off.reward_sha256
            ):
                raise ValueError("decision-off must reduce exactly to mechanism-off")
            self._validate_resource_schedule(group)
        if (
            self.development_only is not True
            or self.scientific_promotion_allowed is not False
            or self.visual_robotics_parity_claimed is not False
        ):
            raise ValueError("results are permanently nonpromoting and non-parity")

    def _validate_resource_schedule(self, group: tuple[TransferArmReceipt, ...]) -> None:
        pretraining_arms = {0, 2, 3, 4}
        eligible = sum(
            step >= self.protocol.warmup_steps
            and step % self.protocol.exploration_period != 0
            for step in range(self.protocol.steps)
        )
        for index, arm in enumerate(group):
            resources = arm.resources
            expected_pretraining = (
                self.protocol.pretraining_steps if index in pretraining_arms else 0
            )
            expected_decisions = 0 if index in (4, 5) else (2 * eligible if index < 5 else 0)
            expected_model_updates = self.protocol.steps if index < 5 else 0
            expected_model_queries = (
                2 * expected_pretraining + expected_model_updates + expected_decisions
            )
            expected_encoder_queries = (
                6 * expected_pretraining
                + 4 * expected_model_updates
                + expected_decisions
            )
            if (
                resources.pretraining_examples != expected_pretraining
                or resources.pretraining_updates != expected_pretraining
                or resources.pretraining_encoder_updates != expected_pretraining
                or resources.pretraining_replay_bytes != 24 * expected_pretraining
                or resources.model_queries != expected_model_queries
                or resources.encoder_queries != expected_encoder_queries
            ):
                raise ValueError("resource counters differ from the frozen semantic schedule")
            if resources.control_queries != (self.protocol.steps + 1 if index == 6 else 0):
                raise ValueError("control-query counters differ from the frozen schedule")
            expected_encoder_bytes = 4 * (2 * _LATENT_DIM + _LATENT_DIM) if index < 5 else 0
            if resources.encoder_bytes != expected_encoder_bytes:
                raise ValueError("encoder byte receipts differ from the frozen architecture")
            env = SwitchingTwoStateMDP(
                SwitchingTwoStateConfig(phase_length=self.protocol.phase_length)
            )
            env_key, agent_key = jr.split(jr.key(arm.seed))
            expected_environment_bytes = _tree_nbytes(env.init(env_key))
            if resources.persistent_environment_bytes != expected_environment_bytes:
                raise ValueError("environment bytes differ from the derived state")
            if index < 5:
                # Model and optimizer array shapes are fixed across updates.
                # JIT canonicalizes the learner's two process-local Python
                # float telemetry leaves to scalar float32 arrays on its first
                # update. The full-warm-start arm arrives in that structural
                # state; other arms retain a larger fresh state at their peak.
                structural_state = _model(encoder_learning=False).init(
                    jr.fold_in(jr.key(arm.seed), 2)
                )
                if index == 3:
                    learner_state = cast(Any, structural_state).learner_state
                    structural_state = cast(Any, structural_state).replace(
                        learner_state=learner_state.replace(
                            birth_timestamp=jnp.asarray(
                                learner_state.birth_timestamp, dtype=jnp.float32
                            ),
                            uptime_s=jnp.asarray(learner_state.uptime_s, dtype=jnp.float32),
                        )
                    )
                expected_mechanism_bytes = _tree_nbytes(structural_state)
            elif index == 5:
                expected_mechanism_bytes = 0
            else:
                agent = SARSAAgent(
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
                expected_mechanism_bytes = _tree_nbytes(agent.init(2, agent_key))
            if resources.persistent_mechanism_bytes != expected_mechanism_bytes:
                raise ValueError("persistent mechanism bytes differ from the derived state")

    def to_payload(self) -> dict[str, object]:
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


def validate_jepa_transfer_payload(payload: object) -> JEPATransferResult:
    """Decode only the bounded exact JSON schema."""
    if type(payload) is not dict:
        raise ValueError("payload must be an exact dict")
    root = cast(dict[str, Any], payload)
    if set(root) != {field.name for field in dataclasses.fields(JEPATransferResult)}:
        raise ValueError("payload fields differ from the schema")
    protocol_raw = root["protocol"]
    if type(protocol_raw) is not dict or set(protocol_raw) != {
        "steps", "phase_length", "pretraining_steps", "warmup_steps",
        "exploration_period", "seeds",
    }:
        raise ValueError("protocol payload differs from the schema")
    protocol_values = dict(cast(dict[str, Any], protocol_raw))
    seeds = protocol_values.pop("seeds")
    if type(seeds) is not list:
        raise ValueError("serialized seeds must be an exact list")
    protocol = JEPATransferProtocol(**protocol_values, seeds=tuple(seeds))
    arms_raw = root["arms"]
    if type(arms_raw) is not list or len(arms_raw) != len(FROZEN_ARM_IDS) * len(protocol.seeds):
        raise ValueError("serialized arms have the wrong bounded length")
    arm_fields = {field.name for field in dataclasses.fields(TransferArmReceipt)}
    resource_fields = {field.name for field in dataclasses.fields(TransferResources)}
    timing_fields = {field.name for field in dataclasses.fields(TimingTelemetry)}
    arms: list[TransferArmReceipt] = []
    for raw in arms_raw:
        if type(raw) is not dict or set(raw) != arm_fields:
            raise ValueError("serialized arm differs from the schema")
        values = cast(dict[str, Any], raw)
        resource_raw, timing_raw = values["resources"], values["timing"]
        if type(resource_raw) is not dict or set(resource_raw) != resource_fields:
            raise ValueError("serialized resources differ from the schema")
        if type(timing_raw) is not dict or set(timing_raw) != timing_fields:
            raise ValueError("serialized timing differs from the schema")
        values = dict(values)
        values["resources"] = TransferResources(**resource_raw)
        values["timing"] = TimingTelemetry(**timing_raw)
        arms.append(TransferArmReceipt(**values))
    result = JEPATransferResult(
        schema=root["schema"],
        protocol=protocol,
        research_pins=_research_pins_from_payload(root["research_pins"]),
        arms=tuple(arms),
        identity=identity_from_payload(root["identity"]),
        development_only=root["development_only"],
        scientific_promotion_allowed=root["scientific_promotion_allowed"],
        visual_robotics_parity_claimed=root["visual_robotics_parity_claimed"],
    )
    require_current_identity(result.identity, _current_identity())
    return result


def _model(*, encoder_learning: bool) -> LatentWorldModel:
    return LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=_LATENT_DIM,
            hidden_sizes=(),
            sparsity=0.0,
            use_layer_norm=False,
            include_action_interactions=True,
            encoder_learning=encoder_learning,
            encoder_collapse_gate_threshold=1.0,
        )
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _model_action(model: LatentWorldModel, state: object, observation: jax.Array) -> jax.Array:
    model_state = cast(Any, state)
    rewards = jnp.stack(
        tuple(
            model.predict(model_state, observation, jnp.asarray(action, dtype=jnp.int32)).reward
            for action in range(2)
        )
    )
    return jnp.argmax(rewards).astype(jnp.int32)


def _base_actions(seed: int, steps: int) -> tuple[int, ...]:
    actions = jr.randint(jr.fold_in(jr.key(seed), 1577), (steps,), 0, 2, dtype=jnp.int32)
    return tuple(int(value) for value in np.asarray(actions))


def _pretrain(protocol: JEPATransferProtocol, seed: int) -> tuple[object, int, int, int]:
    """Train on one stored ASI-only trace and return state, bytes, elapsed ns."""
    started = time.perf_counter_ns()
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=protocol.phase_length))
    env_key, model_key = jr.split(jr.fold_in(jr.key(seed), 1))
    env_state = env.init(env_key)
    observation = env.observe(env_state)
    replay: list[tuple[jax.Array, int, jax.Array, jax.Array]] = []
    for step, action in enumerate(_base_actions(seed + 17, protocol.pretraining_steps)):
        next_observation, reward, env_state = env.step(
            env_state, jnp.asarray(action, dtype=jnp.int32), jr.fold_in(env_key, step)
        )
        replay.append((observation, action, reward, next_observation))
        observation = next_observation
    replay_bytes = sum(
        _tree_nbytes((obs, jnp.asarray(action, dtype=jnp.int32), reward, next_obs))
        for obs, action, reward, next_obs in replay
    )
    model = _model(encoder_learning=True)
    state = model.init(model_key)
    encoder_updates = 0
    for observation, action, reward, next_observation in replay:
        result = model.update(
            cast(Any, state), observation, jnp.asarray(action, dtype=jnp.int32), reward,
            jnp.asarray(0.99, dtype=jnp.float32), next_observation,
        )
        float(result.prediction_error)
        encoder_updates += int(result.encoder_update_applied)
        state = result.state
    elapsed = time.perf_counter_ns() - started
    return state, replay_bytes, elapsed, encoder_updates


def _deployment_state(
    arm_id: str, seed: int, pretrained: object | None
) -> tuple[LatentWorldModel, object]:
    model = _model(encoder_learning=False)
    fresh = model.init(jr.fold_in(jr.key(seed), 2))
    if pretrained is None:
        return model, fresh
    learned = cast(Any, pretrained)
    if arm_id == "full_warm_start_ceiling":
        return model, learned
    matrix, bias = learned.encoder_matrix, learned.encoder_bias
    if arm_id == "encoder_permuted":
        matrix = matrix[:, ::-1]
        bias = bias[::-1]
    return model, fresh.replace(  # type: ignore[attr-defined]
        encoder_matrix=matrix, encoder_bias=bias
    )


def _run_model_arm(protocol: JEPATransferProtocol, seed: int, arm_id: str) -> TransferArmReceipt:
    pretrained = None
    replay_bytes = pretraining_ns = encoder_updates = 0
    if arm_id != "no_pretraining":
        pretrained, replay_bytes, pretraining_ns, encoder_updates = _pretrain(protocol, seed)
    model, state = _deployment_state(arm_id, seed, pretrained)
    initial_bytes = _tree_nbytes(state)
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=protocol.phase_length))
    env_key, _ = jr.split(jr.key(seed))
    env_state = env.init(env_key)
    observation = env.observe(env_state)
    base = _base_actions(seed, protocol.steps)
    actions: list[int] = []
    rewards: list[float] = []
    losses: list[float] = []
    decision_queries = decision_ns = update_ns = environment_ns = 0
    for step in range(protocol.steps):
        base_decision = (
            arm_id == "transfer_decision_off"
            or step < protocol.warmup_steps
            or step % protocol.exploration_period == 0
        )
        if base_decision:
            action = base[step]
        else:
            started = time.perf_counter_ns()
            action = int(_model_action(model, state, observation))
            decision_ns += time.perf_counter_ns() - started
            decision_queries += 2
        started = time.perf_counter_ns()
        next_observation, reward, env_state = env.step(
            env_state, jnp.asarray(action, dtype=jnp.int32), jr.fold_in(env_key, step)
        )
        float(reward)
        environment_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        result = model.update(
            cast(Any, state), observation, jnp.asarray(action, dtype=jnp.int32), reward,
            jnp.asarray(0.99, dtype=jnp.float32), next_observation,
        )
        loss = float(result.prediction_error)
        update_ns += time.perf_counter_ns() - started
        state = result.state
        observation = next_observation
        actions.append(action)
        rewards.append(float(reward))
        losses.append(loss)
    pretrained_examples = protocol.pretraining_steps if pretrained is not None else 0
    encoder_bytes = int(np.asarray(cast(Any, state).encoder_matrix).nbytes) + int(
        np.asarray(cast(Any, state).encoder_bias).nbytes
    )
    resources = TransferResources(
        environment_steps=protocol.steps,
        pretraining_examples=pretrained_examples,
        pretraining_updates=pretrained_examples,
        pretraining_encoder_updates=encoder_updates,
        imported_pretraining_bytes=0,
        pretraining_replay_bytes=replay_bytes,
        online_replay_bytes=0,
        encoder_queries=6 * pretrained_examples + 4 * protocol.steps + decision_queries,
        model_queries=2 * pretrained_examples + protocol.steps + decision_queries,
        control_queries=0,
        encoder_bytes=encoder_bytes,
        persistent_mechanism_bytes=max(initial_bytes, _tree_nbytes(state)),
        persistent_environment_bytes=_tree_nbytes(env_state),
    )
    return TransferArmReceipt(
        arm_id=arm_id,
        seed=seed,
        return_sum=float(sum(rewards)),
        late_return_sum=float(sum(rewards[-protocol.phase_length :])),
        mean_prediction_loss=float(np.mean(np.asarray(losses, dtype=np.float64))),
        action_sha256=_sha(actions),
        reward_sha256=_sha(rewards, floats=True),
        resources=resources,
        timing=TimingTelemetry(pretraining_ns, environment_ns, decision_ns, update_ns, 0),
    )


def _run_control(protocol: JEPATransferProtocol, seed: int, *, sarsa: bool) -> TransferArmReceipt:
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=protocol.phase_length))
    env_key, agent_key = jr.split(jr.key(seed))
    env_state = env.init(env_key)
    observation = env.observe(env_state)
    base = _base_actions(seed, protocol.steps)
    state: Any = None
    action: Any = None
    initial_bytes = 0
    if sarsa:
        agent = SARSAAgent(
            SARSAConfig(
                n_actions=2, gamma=0.99, epsilon_start=0.1, epsilon_end=0.1,
                epsilon_decay_steps=1,
            ), hidden_sizes=(), sparsity=0.0, use_layer_norm=False,
        )
        state = agent.init(2, agent_key)
        initial_bytes = _tree_nbytes(state)
        action, next_key = agent.select_action(state, observation)
        state = state.replace(last_action=action, last_observation=observation, rng_key=next_key)
    actions: list[int] = []
    rewards: list[float] = []
    control_ns = environment_ns = 0
    for step in range(protocol.steps):
        selected = int(action) if sarsa else base[step]
        started = time.perf_counter_ns()
        next_observation, reward, env_state = env.step(
            env_state, jnp.asarray(selected, dtype=jnp.int32), jr.fold_in(env_key, step)
        )
        float(reward)
        environment_ns += time.perf_counter_ns() - started
        if sarsa:
            started = time.perf_counter_ns()
            next_action, next_key = agent.select_action(state, next_observation)
            result = agent.update(
                state, reward=reward, observation=next_observation,
                terminated=jnp.asarray(False), next_action=next_action,
            )
            float(result.td_error)
            control_ns += time.perf_counter_ns() - started
            state = result.state.replace(
                last_action=next_action, last_observation=next_observation, rng_key=next_key
            )
            action = next_action
        observation = next_observation
        actions.append(selected)
        rewards.append(float(reward))
    del observation
    resources = TransferResources(
        environment_steps=protocol.steps,
        pretraining_examples=0,
        pretraining_updates=0,
        pretraining_encoder_updates=0,
        imported_pretraining_bytes=0,
        pretraining_replay_bytes=0,
        online_replay_bytes=0,
        encoder_queries=0,
        model_queries=0,
        control_queries=protocol.steps + 1 if sarsa else 0,
        encoder_bytes=0,
        persistent_mechanism_bytes=max(initial_bytes, _tree_nbytes(state)) if sarsa else 0,
        persistent_environment_bytes=_tree_nbytes(env_state),
    )
    return TransferArmReceipt(
        arm_id="sarsa_control" if sarsa else "mechanism_off",
        seed=seed,
        return_sum=float(sum(rewards)),
        late_return_sum=float(sum(rewards[-protocol.phase_length :])),
        mean_prediction_loss=None,
        action_sha256=_sha(actions),
        reward_sha256=_sha(rewards, floats=True),
        resources=resources,
        timing=TimingTelemetry(0, environment_ns, 0, 0, control_ns),
    )


def run_jepa_transfer_feasibility(
    protocol: JEPATransferProtocol | None = None,
) -> JEPATransferResult:
    """Run the native bounded feasibility lane without external assets."""
    protocol = JEPATransferProtocol() if protocol is None else protocol
    if type(protocol) is not JEPATransferProtocol:
        raise ValueError("protocol must be exact")
    arms: list[TransferArmReceipt] = []
    for seed in protocol.seeds:
        arms.extend(
            _run_model_arm(protocol, seed, arm_id)
            for arm_id in FROZEN_ARM_IDS[:5]
        )
        arms.append(_run_control(protocol, seed, sarsa=False))
        arms.append(_run_control(protocol, seed, sarsa=True))
    return JEPATransferResult(
        schema=JEPA_TRANSFER_SCHEMA,
        protocol=protocol,
        research_pins=PINNED_RESEARCH_ITEMS,
        arms=tuple(arms),
        identity=_current_identity(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-catalog", action="store_true")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--phase-length", type=int, default=32)
    parser.add_argument("--pretraining-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--exploration-period", type=int, default=4)
    args = parser.parse_args(argv)
    if args.external_catalog:
        from alberta_framework.benchmarks.jepa_external_qualification import (
            catalog_payload,
        )

        print(json.dumps(catalog_payload(), sort_keys=True, separators=(",", ":")))
        return 0
    delattr(args, "external_catalog")
    result = run_jepa_transfer_feasibility(JEPATransferProtocol(**vars(args)))
    payload = result.to_payload()
    validate_jepa_transfer_payload(payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FROZEN_ARM_IDS", "FROZEN_DEVELOPMENT_SEEDS", "JEPA_TRANSFER_SCHEMA",
    "PINNED_RESEARCH", "JEPATransferProtocol", "JEPATransferResult",
    "TimingTelemetry", "TransferArmReceipt", "TransferResources",
    "run_jepa_transfer_feasibility", "validate_jepa_transfer_payload",
]
