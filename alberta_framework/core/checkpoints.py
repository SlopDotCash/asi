"""Checkpoint utilities for saving and loading learner state.

Provides ``save_checkpoint`` and ``load_checkpoint`` for persisting any
learner state (``LearnerState``, ``MLPLearnerState``, ``MultiHeadMLPState``,
``TDLearnerState``) to disk using ``orbax-checkpoint``.

The caller provides a *template* state (from ``learner.init()``) to
``load_checkpoint`` so the tree structure is known at load time.

For loading just metadata without a template (e.g. to read learner config
before constructing the template), use ``load_checkpoint_metadata``.

Examples
--------
```python
import jax.random as jr
from alberta_framework import MultiHeadMLPLearner, save_checkpoint, load_checkpoint

learner = MultiHeadMLPLearner(n_heads=5, hidden_sizes=(64, 64))
state = learner.init(feature_dim=20, key=jr.key(42))

# Save (creates a checkpoint directory at the given path)
save_checkpoint(state, "agent.ckpt", metadata={"epoch": 1})

# Load (template provides tree structure)
template = learner.init(feature_dim=20, key=jr.key(0))
loaded_state, meta = load_checkpoint(template, "agent.ckpt")
assert meta["epoch"] == 1

# Load metadata only (no template needed)
meta = load_checkpoint_metadata("agent.ckpt")
assert meta["epoch"] == 1
```
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

# Stamped into checkpoint metadata on save, but deliberately NOT validated on
# load: compatibility is enforced structurally instead, by restoring into the
# caller-provided template PyTree (orbax raises ValueError on a tree
# mismatch). The stamp is provenance for external tooling and any future
# migration path.
_FORMAT_VERSION = 2

# Internal metadata key — stripped from user-facing metadata
_VERSION_KEY = "_format_version"
_EMPTY_ARRAYS_KEY = "_empty_array_leaves"
# Same ceiling as security._JSON_MAX_DEPTH. Origin json.dumps RecursionErrors
# on a 10_000-deep metadata nest and cannot reject it as ValueError.
_JSON_MAX_DEPTH = 32
_JSON_MAX_NODES = 4096


def _is_empty_array(value: object) -> bool:
    return isinstance(value, (jax.Array, np.ndarray)) and value.size == 0


def _orbax_compatible_tree(tree: Any) -> Any:
    """Represent empty leaves with typed sentinels Orbax can serialize.

    Empty arrays contain no payload bytes. Their exact shapes and dtypes come
    from the caller's restore template, so a one-element sentinel carries only
    the leaf type through Orbax without weakening structural restoration.
    """

    def compatible(leaf: Any) -> Any:
        if not _is_empty_array(leaf):
            return leaf
        if isinstance(leaf, jax.Array):
            return jnp.zeros((1,), dtype=leaf.dtype)
        return np.zeros((1,), dtype=leaf.dtype)

    return jax.tree.map(compatible, tree)


def _empty_array_manifest(tree: Any) -> list[dict[str, Any] | None] | None:
    """Describe exact empty leaves so sentinels cannot alias real data."""

    manifest: list[dict[str, Any] | None] = []
    has_empty = False
    for leaf in jax.tree.leaves(tree):
        if not _is_empty_array(leaf):
            manifest.append(None)
            continue
        has_empty = True
        manifest.append({"shape": list(leaf.shape), "dtype": str(leaf.dtype)})
    return manifest if has_empty else None


def _restore_empty_arrays(template: Any, restored: Any, manifest: object) -> Any:
    """Validate and reinstate exact empty leaves from checkpoint metadata."""

    template_leaves = jax.tree.leaves(template)
    if manifest is None:
        if any(_is_empty_array(leaf) for leaf in template_leaves):
            raise ValueError("checkpoint does not identify its empty array leaves")
        return restored
    if type(manifest) is not list or len(manifest) != len(template_leaves):
        raise ValueError("checkpoint empty array manifest does not match the state tree")

    for expected, raw_spec in zip(template_leaves, manifest, strict=True):
        if raw_spec is None:
            if _is_empty_array(expected):
                raise ValueError("checkpoint state leaf is nonempty but template leaf is empty")
            continue
        if type(raw_spec) is not dict or set(raw_spec) != {"shape", "dtype"}:
            raise ValueError("checkpoint empty array manifest is invalid")
        raw_shape = raw_spec["shape"]
        raw_dtype = raw_spec["dtype"]
        if (
            not _is_empty_array(expected)
            or type(raw_shape) is not list
            or any(type(dimension) is not int or dimension < 0 for dimension in raw_shape)
            or type(raw_dtype) is not str
            or list(expected.shape) != raw_shape
            or str(expected.dtype) != raw_dtype
        ):
            raise ValueError("checkpoint empty array leaf does not match the restore template")

    return jax.tree.map(
        lambda expected, actual: expected if _is_empty_array(expected) else actual,
        template,
        restored,
    )


def _copy_mapping(payload: object, *, name: str) -> dict[str, Any]:
    if not issubclass(type(payload), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        values = dict(cast(Mapping[str, Any], payload))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error
    if any(type(key) is not str for key in values):
        raise ValueError(f"{name} keys must be exact strings")
    return values


def _require_path(value: object, *, name: str = "checkpoint path") -> Path:
    if type(value) is str:
        return Path(value)
    if isinstance(value, Path) and type(value).__module__.startswith("pathlib"):
        return Path(value)
    raise ValueError(f"{name} must be an exact str or Path")


def _require_json_nest(
    value: object, *, name: str, depth: int, budget: list[int]
) -> None:
    """Reject unbounded metadata nests before ``json.dumps`` RecursionError."""
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError(f"{name} exceeds the JSON value resource limit")
    if depth > _JSON_MAX_DEPTH:
        raise ValueError(f"{name} exceeds the JSON nesting limit")
    value_type = type(value)
    if value is None or value_type is bool or value_type is int or value_type is str:
        return
    if value_type is float:
        return
    if value_type is list:
        for item in cast(list[object], value):
            _require_json_nest(item, name=name, depth=depth + 1, budget=budget)
        return
    if value_type is dict:
        for item in cast(dict[object, object], value).values():
            _require_json_nest(item, name=name, depth=depth + 1, budget=budget)
        return


def _require_json_safe_metadata(metadata: dict[str, Any]) -> None:
    """Reject metadata that cannot round-trip as finite JSON."""
    try:
        _require_json_nest(
            metadata, name="checkpoint metadata", depth=0, budget=[_JSON_MAX_NODES]
        )
    except RecursionError as exc:
        raise ValueError("checkpoint metadata exceeds the JSON nesting limit") from exc
    try:
        json.dumps(metadata, allow_nan=False)
    except RecursionError as exc:
        raise ValueError("checkpoint metadata exceeds the JSON nesting limit") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be JSON-safe and finite") from exc


def save_checkpoint(
    state: Any,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save learner state to disk.

    Creates a checkpoint directory at ``path`` containing the serialized
    state PyTree and optional user metadata as JSON.

    Args:
        state: Any learner state (LearnerState, MLPLearnerState,
            MultiHeadMLPState, TDLearnerState)
        path: Path for the checkpoint directory.
        metadata: Optional user metadata dict to store alongside
            the checkpoint (e.g. epoch, learner config, etc.)
    """
    path = _require_path(path)

    if metadata is None:
        meta_to_save: dict[str, Any] = {}
    else:
        meta_to_save = _copy_mapping(metadata, name="checkpoint metadata")
    _require_json_safe_metadata(meta_to_save)
    meta_to_save.pop(_EMPTY_ARRAYS_KEY, None)
    meta_to_save[_VERSION_KEY] = _FORMAT_VERSION
    empty_array_manifest = _empty_array_manifest(state)
    if empty_array_manifest is not None:
        meta_to_save[_EMPTY_ARRAYS_KEY] = empty_array_manifest
    path.parent.mkdir(parents=True, exist_ok=True)

    with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as ckptr:
        ckptr.save(
            str(path),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(_orbax_compatible_tree(state)),
                metadata=ocp.args.JsonSave(meta_to_save),
            ),
        )


def load_checkpoint(
    state_template: Any,
    path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Load checkpoint into a state matching the template's tree structure.

    The template state (from ``learner.init()``) provides the PyTree
    structure for deserialization.

    Args:
        state_template: A state of the same type and structure as the
            saved state. Typically created via ``learner.init()`` with
            the same architecture.
        path: Path to the checkpoint directory.

    Returns:
        Tuple of ``(loaded_state, user_metadata)`` where ``user_metadata``
        is the dict passed to ``save_checkpoint``, or an empty dict if
        none was provided.

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist
        ValueError: If state structure doesn't match template
    """
    path = _require_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        compatible_template = _orbax_compatible_tree(state_template)
        with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as ckptr:
            loaded = ckptr.restore(
                str(path),
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(compatible_template),
                    metadata=ocp.args.JsonRestore(),
                ),
            )
    except ValueError as e:
        raise ValueError(
            f"State structure mismatch. "
            f"Ensure the learner architecture matches the saved checkpoint. "
            f"Original error: {e}"
        ) from e

    raw_metadata = loaded.metadata
    user_metadata = _copy_mapping(raw_metadata, name="checkpoint metadata")
    user_metadata.pop(_VERSION_KEY, None)
    empty_array_manifest = user_metadata.pop(_EMPTY_ARRAYS_KEY, None)
    _require_json_safe_metadata(user_metadata)
    return (
        _restore_empty_arrays(state_template, loaded.state, empty_array_manifest),
        user_metadata,
    )


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Load only the user metadata from a checkpoint, without a state template.

    This is useful when metadata contains configuration needed to construct
    the state template (e.g. learner_config in rlsecd).

    Args:
        path: Path to the checkpoint directory.

    Returns:
        The user metadata dict, or an empty dict if none was stored.

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist
    """
    path = _require_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as ckptr:
        loaded = ckptr.restore(
            str(path),
            args=ocp.args.Composite(
                metadata=ocp.args.JsonRestore(),
            ),
        )

    raw_metadata = loaded.metadata
    user_metadata = _copy_mapping(raw_metadata, name="checkpoint metadata")
    user_metadata.pop(_VERSION_KEY, None)
    user_metadata.pop(_EMPTY_ARRAYS_KEY, None)
    _require_json_safe_metadata(user_metadata)
    return user_metadata


def checkpoint_exists(path: str | Path) -> bool:
    """Check whether a checkpoint exists at the given path.

    Args:
        path: Path to check for a checkpoint directory.

    Returns:
        True if a checkpoint directory exists at the path.
    """
    path = _require_path(path)
    # Orbax checkpoints are directories containing a state/ subdirectory
    return path.is_dir() and (path / "state").is_dir()
