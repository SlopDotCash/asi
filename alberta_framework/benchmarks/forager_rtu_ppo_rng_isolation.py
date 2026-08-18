"""Exact-source adapter for an RNG-isolated upstream PPO/RTU-PPO runner.

The audited upstream ``rtu_ppo.py`` uses the same random key for policy action
sampling and for the environment transition.  That makes its reward trace
ineligible for paired comparisons with runners whose environment key stream is
independent of agent-side random-number consumption.

This module does not silently monkeypatch an import.  It accepts only the
audited upstream source bytes, applies a fixed set of single-occurrence byte
replacements, validates the resulting Python AST, and returns both the derived
source and a canonical provenance descriptor.  A matched-suite builder can
then bake both source trees into one content-addressed OCI image.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import hashlib
import json
import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, cast

UPSTREAM_RTU_PPO_SOURCE_SHA256: Final = (
    "e75a6762690832067a24a649559a55e0aa89abba005d600f090b1bf284b3fc24"
)
UPSTREAM_SOURCE_COMMIT: Final = "9710f60fa30da5badc451ad7ce3ff296d5070830"
UPSTREAM_SOURCE_TREE_GIT_SHA1: Final = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
UPSTREAM_RTU_PPO_BLOB_GIT_SHA1: Final = "63bdc359079ef14b0de1e5964ed49b02c62b3e59"
UPSTREAM_SOURCE_ARCHIVE_SHA256: Final = (
    "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
)
UPSTREAM_SOURCE_PATH: Final = "src/rtu_ppo.py"
# Arbitrary-but-frozen fold-in tag deriving the agent-side key stream from the
# run seed (``agent_root = fold_in(root, tag)``) so agent-side draws never
# consume the environment chain.  The specific value carries no meaning, but
# the pinned EXPECTED_* digests and probe words below depend on it.
ISOLATED_AGENT_RNG_NAMESPACE: Final = 0xA63E7C11
REQUIRED_PRNG_IMPL: Final = "threefry2x32"
PATCH_SCHEMA_VERSION: Final = (
    "alberta.forager_rtu_ppo_rng_isolation_patch.v1"
)
PATCH_MODE: Final = "exact_single_occurrence_source_replacements_v1"
PATCH_FORMAT: Final = "unified_diff_utf8_p1_v1"
ENVIRONMENT_RNG_SCHEDULE: Final = "dedicated_environment_split_chain_v1"
ENVIRONMENT_RNG_SCHEDULE_SHA256: Final = (
    "51d811e6fccd2b015b1703f22775f880089bbca3fc8938421ad3e18526882cb0"
)
AGENT_RNG_IDENTITY: Final = "isolated_agent_rng_v1"
EXPECTED_PATCHED_RTU_PPO_SHA256: Final = (
    "70bbdd0943d82570c1dc0d28494cf93f9c1b208ef67b3a547585fe5897cdf409"
)
EXPECTED_UNIFIED_DIFF_SHA256: Final = (
    "46ac3d6c1ae5740bee97fea23abf002ffb161ab4b1b35c041b24b717645e076f"
)
PUBLIC_PROBE_SEED: Final = 0
PUBLIC_PROBE_TRANSITIONS: Final = 4
PUBLIC_PROBE_AGENT_SPLIT_COUNTS: Final = (0, 1, 7, 32)
PUBLIC_PROBE_EXPECTED_AGENT_ROOT_WORDS: Final = (2795197240, 2837457689)
PUBLIC_PROBE_EXPECTED_ENVIRONMENT_TRACE_SHA256: Final = (
    "b69e024840289fc737f2e29912b4c39707b9d73a2682181983f0bd15cbb3706e"
)


class RTUPPORngIsolationError(ValueError):
    """Raised when source identity or derived-runner structure is not exact."""


_SHA256_RE: Final = re.compile(r"\A[0-9a-f]{64}\Z")


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise RTUPPORngIsolationError("name must be an exact string")
    if type(value) is not str:
        raise RTUPPORngIsolationError(f"{name} must be an exact string")
    return value


@dataclasses.dataclass(frozen=True)
class SourceReplacement:
    """One exact, single-occurrence source transformation."""

    replacement_id: str
    before: bytes
    after: bytes

    def __post_init__(self) -> None:
        if type(self.replacement_id) is not str or not self.replacement_id:
            raise RTUPPORngIsolationError("replacement_id must be a non-empty string")
        if type(self.before) is not bytes or not self.before:
            raise RTUPPORngIsolationError("before must be non-empty bytes")
        if type(self.after) is not bytes or not self.after:
            raise RTUPPORngIsolationError("after must be non-empty bytes")


@dataclasses.dataclass(frozen=True)
class IsolatedRTUPPOSource:
    """Derived source bytes and their immutable provenance descriptor."""

    source: bytes
    upstream_source_sha256: str
    source_sha256: str
    patch: bytes
    patch_sha256: str
    descriptor: Mapping[str, Any]
    descriptor_sha256: str

    def __post_init__(self) -> None:
        if type(self.source) is not bytes or not self.source:
            raise RTUPPORngIsolationError("source must be non-empty bytes")
        if type(self.patch) is not bytes or not self.patch:
            raise RTUPPORngIsolationError("patch must be non-empty bytes")
        if not isinstance(self.descriptor, Mapping):
            raise RTUPPORngIsolationError("descriptor must be a mapping")
        for name in (
            "upstream_source_sha256",
            "source_sha256",
            "patch_sha256",
            "descriptor_sha256",
        ):
            val = getattr(self, name)
            if type(val) is not str or _SHA256_RE.fullmatch(val) is None:
                raise RTUPPORngIsolationError(f"{name} must be a 64-character lowercase SHA-256")


_REPLACEMENTS: Final = (
    SourceReplacement(
        replacement_id="declare_isolated_agent_namespace",
        before=b"PERIOD = 182500\n",
        after=(
            b"PERIOD = 182500\n"
            b"ISOLATED_AGENT_RNG_NAMESPACE = 0xA63E7C11\n"
        ),
    ),
    SourceReplacement(
        replacement_id="carry_environment_rng_in_gymnax_state",
        before=b"    env_state: Any = struct.field(pytree_node=True)\n",
        after=(
            b"    env_state: Any = struct.field(pytree_node=True)\n"
            b"    environment_rng: Any = struct.field(pytree_node=True)\n"
        ),
    ),
    SourceReplacement(
        replacement_id="split_environment_rng_before_transition",
        before=(
            b"    # STEP ENV\n"
            b"    obs, env_state, reward, done, info = gymnax_state.env_step(\n"
            b"        _rng, gymnax_state.env_state, action.squeeze(), "
            b"gymnax_state.env_params\n"
            b"    )\n"
        ),
        after=(
            b"    # STEP ENV\n"
            b"    environment_rng, environment_step_rng = jax.random.split(\n"
            b"        gymnax_state.environment_rng\n"
            b"    )\n"
            b"    obs, env_state, reward, done, info = gymnax_state.env_step(\n"
            b"        environment_step_rng,\n"
            b"        gymnax_state.env_state,\n"
            b"        action.squeeze(),\n"
            b"        gymnax_state.env_params,\n"
            b"    )\n"
        ),
    ),
    SourceReplacement(
        replacement_id="persist_environment_rng_after_transition",
        before=(
            b"        env_params=gymnax_state.env_params,\n"
            b"        env_state=env_state,\n"
            b"    )\n"
            b"    runner_state = (\n"
        ),
        after=(
            b"        env_params=gymnax_state.env_params,\n"
            b"        env_state=env_state,\n"
            b"        environment_rng=environment_rng,\n"
            b"    )\n"
            b"    runner_state = (\n"
        ),
    ),
    SourceReplacement(
        replacement_id="root_environment_and_agent_streams",
        before=(
            b"    rng, reset_rng = jax.random.split(rng)\n"
            b"    obs, env_state = env.reset(reset_rng, env.default_params)\n"
        ),
        after=(
            b"    environment_rng = rng\n"
            b"    environment_rng, reset_rng = jax.random.split(environment_rng)\n"
            b"    rng = jax.random.fold_in(rng, ISOLATED_AGENT_RNG_NAMESPACE)\n"
            b"    obs, env_state = env.reset(reset_rng, env.default_params)\n"
        ),
    ),
    SourceReplacement(
        replacement_id="initialize_environment_rng_carry",
        before=(
            b"        env_params=env.default_params,\n"
            b"        env_state=env_state,\n"
            b"    )\n"
            b"    action_dim = 4\n"
        ),
        after=(
            b"        env_params=env.default_params,\n"
            b"        env_state=env_state,\n"
            b"        environment_rng=environment_rng,\n"
            b"    )\n"
            b"    action_dim = 4\n"
        ),
    ),
    SourceReplacement(
        replacement_id="preserve_environment_rng_across_update_boundary",
        before=(
            b"            env_params=gymnax_state.env_params,\n"
            b"            env_state=gymnax_state.env_state,\n"
            b"        )\n"
            b"\n"
            b"        env_step_state = (\n"
        ),
        after=(
            b"            env_params=gymnax_state.env_params,\n"
            b"            env_state=gymnax_state.env_state,\n"
            b"            environment_rng=gymnax_state.environment_rng,\n"
            b"        )\n"
            b"\n"
            b"        env_step_state = (\n"
        ),
    ),
    SourceReplacement(
        replacement_id="use_explicit_threefry_seed_root",
        before=b"        rng = jax.random.PRNGKey(seed)\n",
        after=b'        rng = jax.random.key(seed, impl="threefry2x32")\n',
    ),
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unified_diff(upstream: bytes, derived: bytes) -> bytes:
    """Return the one canonical, directly reviewable ``patch -p1`` artifact."""
    try:
        upstream_lines = upstream.decode("utf-8").splitlines(keepends=True)
        derived_lines = derived.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise RTUPPORngIsolationError("source derivation is not valid UTF-8") from exc
    patch = "".join(
        difflib.unified_diff(
            upstream_lines,
            derived_lines,
            fromfile=f"a/{UPSTREAM_SOURCE_PATH}",
            tofile=f"b/{UPSTREAM_SOURCE_PATH}",
            lineterm="\n",
        )
    ).encode("utf-8")
    if not patch:
        raise RTUPPORngIsolationError("source derivation produced an empty patch")
    return patch


def _plain_json(value: Any, *, path: str = "descriptor") -> Any:
    """Return a detached JSON value from mutable or frozen containers."""
    if type(value) is float:
        if not math.isfinite(value):
            raise RTUPPORngIsolationError(
                f"{path} contains a non-finite JSON number"
            )
        return value
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) in (dict, MappingProxyType):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RTUPPORngIsolationError(
                    f"{path} contains a non-string object key"
                )
            result[key] = _plain_json(item, path=f"{path}.{key}")
        return result
    if type(value) in (list, tuple):
        return [
            _plain_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RTUPPORngIsolationError(
        f"{path} contains a non-JSON value of type {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    """Recursively freeze a previously validated JSON value."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RTUPPORngIsolationError(
            "patch descriptor is not canonical JSON"
        ) from exc


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RTUPPORngIsolationError(
            f"derived source must define exactly one {name}()"
        )
    return matches[0]


def _call_paths(function: ast.FunctionDef) -> list[tuple[str, ast.Call]]:
    result: list[tuple[str, ast.Call]] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            path = _attribute_path(node.func)
            if path is not None:
                result.append((path, node))
    return result


def _assigned_calls(
    function: ast.FunctionDef,
    path: str,
) -> list[tuple[ast.Assign, ast.Call]]:
    result: list[tuple[ast.Assign, ast.Call]] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _attribute_path(node.value.func) == path
        ):
            result.append((node, node.value))
    return result


def _single_tuple_target_names(assignment: ast.Assign) -> tuple[str, ...] | None:
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], (ast.Tuple, ast.List))
    ):
        return None
    names: list[str] = []
    for element in assignment.targets[0].elts:
        if not isinstance(element, ast.Name):
            return None
        names.append(element.id)
    return tuple(names)


def _single_name_target(assignment: ast.Assign) -> str | None:
    if len(assignment.targets) != 1 or not isinstance(
        assignment.targets[0],
        ast.Name,
    ):
        return None
    return assignment.targets[0].id


def _keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    values = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(values) != 1:
        return None
    return values[0]


def _validate_derived_ast(source: bytes) -> None:
    try:
        decoded = source.decode("utf-8")
        module = ast.parse(decoded, filename=UPSTREAM_SOURCE_PATH)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RTUPPORngIsolationError(
            "derived source is not valid UTF-8 Python"
        ) from exc

    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "GymnaxEnvState"
    ]
    if len(classes) != 1:
        raise RTUPPORngIsolationError(
            "derived source must define exactly one GymnaxEnvState"
        )
    annotated_fields = {
        node.target.id
        for node in classes[0].body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    if "environment_rng" not in annotated_fields:
        raise RTUPPORngIsolationError(
            "GymnaxEnvState does not carry the environment RNG"
        )

    env_step = _function(module, "env_step")
    env_calls = _call_paths(env_step)
    split_assignments = [
        (assignment, call)
        for assignment, call in _assigned_calls(env_step, "jax.random.split")
        if len(call.args) == 1
        and _attribute_path(call.args[0])
        == "gymnax_state.environment_rng"
    ]
    if (
        len(split_assignments) != 1
        or _single_tuple_target_names(split_assignments[0][0])
        != ("environment_rng", "environment_step_rng")
    ):
        raise RTUPPORngIsolationError(
            "env_step must split and retain exactly one carried environment RNG"
        )
    transition_calls = [
        call
        for path, call in env_calls
        if path == "gymnax_state.env_step"
    ]
    if (
        len(transition_calls) != 1
        or not transition_calls[0].args
        or not isinstance(transition_calls[0].args[0], ast.Name)
        or transition_calls[0].args[0].id != "environment_step_rng"
    ):
        raise RTUPPORngIsolationError(
            "environment transition does not consume environment_step_rng"
        )
    action_calls = [
        call for path, call in env_calls if path == "agent_step"
    ]
    if (
        len(action_calls) != 1
        or len(action_calls[0].args) < 3
        or not isinstance(action_calls[0].args[2], ast.Name)
        or action_calls[0].args[2].id != "_rng"
    ):
        raise RTUPPORngIsolationError(
            "policy action sampling no longer consumes the upstream agent key"
        )

    experiment = _function(module, "experiment")
    experiment_calls = _call_paths(experiment)
    reset_splits = [
        (assignment, call)
        for assignment, call in _assigned_calls(experiment, "jax.random.split")
        if len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "environment_rng"
    ]
    agent_folds = [
        (assignment, call)
        for assignment, call in _assigned_calls(experiment, "jax.random.fold_in")
        if len(call.args) == 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "rng"
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "ISOLATED_AGENT_RNG_NAMESPACE"
    ]
    roots = [
        node
        for node in experiment.body
        if isinstance(node, ast.Assign)
        and _single_name_target(node) == "environment_rng"
        and isinstance(node.value, ast.Name)
        and node.value.id == "rng"
    ]
    if (
        len(roots) != 1
        or len(reset_splits) != 1
        or _single_tuple_target_names(reset_splits[0][0])
        != ("environment_rng", "reset_rng")
        or len(agent_folds) != 1
        or _single_name_target(agent_folds[0][0]) != "rng"
    ):
        raise RTUPPORngIsolationError(
            "experiment must root one environment chain and one namespaced "
            "agent chain"
        )

    main = _function(module, "main")
    root_assignments = _assigned_calls(main, "jax.random.key")
    if len(root_assignments) != 1:
        raise RTUPPORngIsolationError(
            "main must create exactly one explicit environment seed root"
        )
    root_assignment, root_call = root_assignments[0]
    root_impl = _keyword_value(root_call, "impl")
    if (
        _single_name_target(root_assignment) != "rng"
        or len(root_call.args) != 1
        or not isinstance(root_call.args[0], ast.Name)
        or root_call.args[0].id != "seed"
        or not isinstance(root_impl, ast.Constant)
        or root_impl.value != REQUIRED_PRNG_IMPL
    ):
        raise RTUPPORngIsolationError(
            "environment root must be jax.random.key(seed, impl=threefry2x32)"
        )
    legacy_roots = [
        call
        for path, call in _call_paths(main)
        if path == "jax.random.PRNGKey"
    ]
    if legacy_roots:
        raise RTUPPORngIsolationError(
            "derived source must not retain a legacy PRNGKey seed root"
        )

    gymnax_creates: list[ast.Call] = []
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and _attribute_path(node.func) == "GymnaxEnvState.create"
        ):
            gymnax_creates.append(node)
    if len(gymnax_creates) != 3:
        raise RTUPPORngIsolationError(
            "derived source has an unexpected GymnaxEnvState.create count"
        )
    for call in gymnax_creates:
        names = [keyword.arg for keyword in call.keywords]
        if names.count("environment_rng") != 1:
            raise RTUPPORngIsolationError(
                "every GymnaxEnvState.create must bind environment_rng once"
            )
    env_step_creates = [
        call
        for path, call in env_calls
        if path == "GymnaxEnvState.create"
    ]
    if len(env_step_creates) != 1:
        raise RTUPPORngIsolationError(
            "env_step must rebuild exactly one GymnaxEnvState"
        )
    env_step_carry = _keyword_value(env_step_creates[0], "environment_rng")
    if not (
        isinstance(env_step_carry, ast.Name)
        and env_step_carry.id == "environment_rng"
    ):
        raise RTUPPORngIsolationError(
            "env_step must persist the post-split environment RNG"
        )
    experiment_creates = [
        call
        for path, call in experiment_calls
        if path == "GymnaxEnvState.create"
    ]
    experiment_carries = [
        _attribute_path(value)
        if isinstance(value, ast.Attribute)
        else value.id
        if isinstance(value, ast.Name)
        else None
        for call in experiment_creates
        if (value := _keyword_value(call, "environment_rng")) is not None
    ]
    if sorted(experiment_carries, key=lambda value: "" if value is None else value) != [
        "environment_rng",
        "gymnax_state.environment_rng",
    ]:
        raise RTUPPORngIsolationError(
            "experiment must initialize and preserve the environment RNG carry"
        )


def derive_isolated_rtu_ppo_source(source: bytes) -> IsolatedRTUPPOSource:
    """Return the one audited RNG-isolated derivation of upstream source."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    observed_source_sha256 = _sha256(source)
    if observed_source_sha256 != UPSTREAM_RTU_PPO_SOURCE_SHA256:
        raise RTUPPORngIsolationError(
            "upstream rtu_ppo.py SHA-256 differs from the audited source"
        )

    derived = source
    replacement_records: list[dict[str, Any]] = []
    for replacement in _REPLACEMENTS:
        count = derived.count(replacement.before)
        if count != 1:
            host_id = _require_exact_str("replacement_id", replacement.replacement_id)
            raise RTUPPORngIsolationError(
                f"replacement '{host_id}' matched {count} source locations instead of one"
            )
        derived = derived.replace(
            replacement.before,
            replacement.after,
            1,
        )
        replacement_records.append(
            {
                "replacement_id": replacement.replacement_id,
                "before_sha256": _sha256(replacement.before),
                "after_sha256": _sha256(replacement.after),
            }
        )

    _validate_derived_ast(derived)
    derived_sha256 = _sha256(derived)
    if derived_sha256 != EXPECTED_PATCHED_RTU_PPO_SHA256:
        raise RTUPPORngIsolationError(
            "derived rtu_ppo.py SHA-256 differs from the frozen adapter output"
        )
    patch = _unified_diff(source, derived)
    patch_sha256 = _sha256(patch)
    if patch_sha256 != EXPECTED_UNIFIED_DIFF_SHA256:
        raise RTUPPORngIsolationError(
            "unified RNG-isolation patch differs from the frozen artifact"
        )
    descriptor: dict[str, Any] = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "patch_mode": PATCH_MODE,
        "patch_format": PATCH_FORMAT,
        "patch_sha256": patch_sha256,
        "upstream": {
            "repository": "https://github.com/steventango/continual-foragax-agents",
            "commit": UPSTREAM_SOURCE_COMMIT,
            "tree_git_sha1": UPSTREAM_SOURCE_TREE_GIT_SHA1,
            "archive_sha256": UPSTREAM_SOURCE_ARCHIVE_SHA256,
            "path": UPSTREAM_SOURCE_PATH,
            "blob_git_sha1": UPSTREAM_RTU_PPO_BLOB_GIT_SHA1,
            "sha256": observed_source_sha256,
        },
        "derived_source_sha256": derived_sha256,
        "environment_rng": {
            "schedule": ENVIRONMENT_RNG_SCHEDULE,
            "schedule_sha256": ENVIRONMENT_RNG_SCHEDULE_SHA256,
            "prng_impl": REQUIRED_PRNG_IMPL,
            "root": "jax.random.key(seed, impl=threefry2x32)",
            "reset": (
                "environment_rng, reset_rng = "
                "jax.random.split(environment_rng)"
            ),
            "transition": (
                "environment_rng, environment_step_rng = "
                "jax.random.split(environment_rng)"
            ),
            "action_key_shared_with_environment": False,
        },
        "agent_rng": {
            "identity": AGENT_RNG_IDENTITY,
            "root": (
                "jax.random.fold_in("
                "jax.random.key(seed, impl=threefry2x32), "
                "ISOLATED_AGENT_RNG_NAMESPACE)"
            ),
            "namespace": ISOLATED_AGENT_RNG_NAMESPACE,
            "environment_key_shared": False,
            "consumption_can_advance_environment_rng": False,
        },
        "promotion_authorized": False,
        "replacement_records": replacement_records,
    }
    canonical = _canonical_json(descriptor)
    descriptor["payload_sha256"] = _sha256(canonical)
    descriptor_sha256 = _sha256(_canonical_json(descriptor))
    return IsolatedRTUPPOSource(
        source=derived,
        upstream_source_sha256=observed_source_sha256,
        source_sha256=derived_sha256,
        patch=patch,
        patch_sha256=patch_sha256,
        descriptor=_freeze_json(descriptor),
        descriptor_sha256=descriptor_sha256,
    )


def validate_isolated_rtu_ppo_source(
    upstream_source: bytes,
    derived_source: bytes,
    patch: bytes,
    descriptor: Mapping[str, Any],
) -> None:
    """Re-derive and compare source, patch, and descriptor byte-for-byte."""
    if not isinstance(upstream_source, bytes):
        raise TypeError("upstream_source must be bytes")
    if not isinstance(derived_source, bytes):
        raise TypeError("derived_source must be bytes")
    if not isinstance(patch, bytes):
        raise TypeError("patch must be bytes")
    if not isinstance(descriptor, Mapping):
        raise TypeError("descriptor must be a mapping")
    expected = derive_isolated_rtu_ppo_source(upstream_source)
    if not isinstance(descriptor, dict):
        raise RTUPPORngIsolationError(
            "descriptor must be a detached canonical object"
        )
    if derived_source != expected.source:
        raise RTUPPORngIsolationError(
            "derived source bytes differ from the exact re-derivation"
        )
    if patch != expected.patch:
        raise RTUPPORngIsolationError(
            "patch bytes differ from the exact re-derivation"
        )
    try:
        actual_descriptor = _canonical_json(descriptor)
        expected_descriptor = _canonical_json(dict(expected.descriptor))
    except RTUPPORngIsolationError:
        raise
    if actual_descriptor != expected_descriptor:
        raise RTUPPORngIsolationError(
            "patch descriptor differs from the exact re-derivation"
        )


def _runtime_key_words(random_module: Any, key: Any) -> list[int]:
    words = random_module.key_data(key).tolist()
    if not isinstance(words, list) or len(words) != 2:
        raise RTUPPORngIsolationError("runtime key data is not one threefry key")
    result = [int(word) for word in words]
    if any(word < 0 or word > 2**32 - 1 for word in result):
        raise RTUPPORngIsolationError("runtime key data is outside uint32")
    return result


def _runtime_environment_trace(random_module: Any, root: Any) -> dict[str, Any]:
    def split_frame(key: Any) -> tuple[Any, dict[str, list[int]]]:
        next_key, operation_key = random_module.split(key)
        return next_key, {
            "input_key": _runtime_key_words(random_module, key),
            "next_key": _runtime_key_words(random_module, next_key),
            "operation_key": _runtime_key_words(random_module, operation_key),
        }

    environment_rng, reset = split_frame(root)
    transitions: list[dict[str, Any]] = []
    for index in range(PUBLIC_PROBE_TRANSITIONS):
        environment_rng, keys = split_frame(environment_rng)
        transitions.append({"index": index, "keys": keys})
    return {
        "root_key": _runtime_key_words(random_module, root),
        "reset": reset,
        "transitions": transitions,
    }


def run_public_rng_isolation_probe(upstream_source: bytes) -> Mapping[str, Any]:
    """Exercise the frozen public key schedule without an environment or training.

    The probe is deliberately fixed to seed zero and four transitions.  It is a
    structural/runtime check of this exact source derivation, not a benchmark,
    qualification, scientific result, or authorization to promote a claim.
    """
    derivation = derive_isolated_rtu_ppo_source(upstream_source)
    try:
        import jax
        from jax import random as jr
    except ImportError as exc:  # pragma: no cover - JAX is a project dependency
        raise RTUPPORngIsolationError("JAX is required for the runtime probe") from exc

    if jax.config.jax_threefry_partitionable is not True:
        raise RTUPPORngIsolationError(
            "runtime probe requires jax_threefry_partitionable=True"
        )
    root = jr.key(PUBLIC_PROBE_SEED, impl=REQUIRED_PRNG_IMPL)
    if str(jr.key_impl(root)) != REQUIRED_PRNG_IMPL:
        raise RTUPPORngIsolationError("runtime root does not use threefry2x32")
    agent_root = jr.fold_in(root, ISOLATED_AGENT_RNG_NAMESPACE)
    if tuple(_runtime_key_words(jr, agent_root)) != PUBLIC_PROBE_EXPECTED_AGENT_ROOT_WORDS:
        raise RTUPPORngIsolationError("public agent root differs from the frozen namespace")

    environment_trace = _runtime_environment_trace(jr, root)
    environment_trace_sha256 = _sha256(_canonical_json(environment_trace))
    if environment_trace_sha256 != PUBLIC_PROBE_EXPECTED_ENVIRONMENT_TRACE_SHA256:
        raise RTUPPORngIsolationError(
            "public environment key trace differs from the frozen split chain"
        )

    consumption_checks: list[dict[str, Any]] = []
    for split_count in PUBLIC_PROBE_AGENT_SPLIT_COUNTS:
        agent_rng = agent_root
        for _ in range(split_count):
            agent_rng, _agent_operation_rng = jr.split(agent_rng)
        candidate_trace = _runtime_environment_trace(jr, root)
        candidate_trace_sha256 = _sha256(_canonical_json(candidate_trace))
        if candidate_trace_sha256 != environment_trace_sha256:
            raise RTUPPORngIsolationError(
                "agent RNG consumption changed the environment key trace"
            )
        consumption_checks.append(
            {
                "agent_split_count": split_count,
                "agent_final_key": _runtime_key_words(jr, agent_rng),
                "environment_trace_sha256": candidate_trace_sha256,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "alberta.forager_rtu_ppo_rng_isolation_probe.v1",
        "evidence_boundary": "public_seed_key_schedule_only_no_environment_or_training",
        "promotion_authorized": False,
        "seed": PUBLIC_PROBE_SEED,
        "transition_count": PUBLIC_PROBE_TRANSITIONS,
        "prng_impl": REQUIRED_PRNG_IMPL,
        "jax_threefry_partitionable": True,
        "agent_namespace": ISOLATED_AGENT_RNG_NAMESPACE,
        "upstream_source_sha256": derivation.upstream_source_sha256,
        "derived_source_sha256": derivation.source_sha256,
        "patch_sha256": derivation.patch_sha256,
        "descriptor_sha256": derivation.descriptor_sha256,
        "environment_trace_sha256": environment_trace_sha256,
        "environment_trace": environment_trace,
        "agent_consumption_checks": consumption_checks,
    }
    payload["payload_sha256"] = _sha256(_canonical_json(payload))
    return cast(Mapping[str, Any], _freeze_json(payload))
