"""Contract tests for :mod:`alberta_framework.benchmarks.forager_matched_qualification`.

Qualification is the reward-blind first phase of the matched-current
pipeline: it stages content-addressed source snapshots, runs seed-zero
structural probes inside the already-qualified networkless CPU image, and
emits content-only capability receipts (trust anchor
``content_only_unendorsed_v1`` — no endorsement or performance claim).  The
tests check that staging is deterministic and bounded (member/directory
caps, symlink and hard-link rejection), probe commands are seed-zero,
networkless, and pinned to the exact image, receipts stay reward-blind and
unendorsed, and publication is atomic with bottom-up fsync.  Failures before
publication leave no visible tree; failures after the atomic rename have a
distinct published-but-uncertain outcome that forbids destination reuse.

Probes and container invocations are stubbed via injected runners; nothing
here executes a real container or a benchmark horizon.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import selectors
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_open_protocol as builder
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks.forager_matched_protocol import SourceBinding

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fresh_replay_fixture(
    tmp_path: Path,
) -> tuple[Path, Any, str, dict[str, Any]]:
    qualification_root = tmp_path / "qualification"
    module = (
        qualification_root
        / "sources"
        / "alberta"
        / "source"
        / "alberta_framework"
        / "benchmarks"
        / "forager_matched_qualification.py"
    )
    module.parent.mkdir(parents=True)
    module.write_bytes(b"# staged qualification fixture\n")
    qualification._normalize_qualification_tree_permissions(  # noqa: SLF001
        qualification_root
    )
    module_sha256 = hashlib.sha256(module.read_bytes()).hexdigest()
    manifest_sha256 = _sha("fresh-manifest")
    protocol_sha256 = _sha("fresh-protocol")
    plan_sha256 = _sha("fresh-plan")
    runtime = qualification._ProbeRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "docker",
        executable_sha256=_sha("fresh-runtime"),
        version={},
        image_inspection={},
    )
    payload = {
        "schema_version": qualification._FRESH_SNAPSHOT_REPLAY_SCHEMA,  # noqa: SLF001
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "plan_sha256": plan_sha256,
        "plan_qualification_manifest_sha256": manifest_sha256,
        "qualification_module_path": (
            "alberta_framework/benchmarks/forager_matched_qualification.py"
        ),
        "qualification_module_sha256": module_sha256,
    }
    return qualification_root, runtime, module_sha256, payload


def _dummy_source(key: qualification.SourceKey, root: Path, binding: SourceBinding) -> Any:
    inventory_path = root / f"{key}-inventory.json"
    archive = root / f"{key}.tar"
    return qualification._StagedSource(  # noqa: SLF001
        key=key,
        root=root,
        archive=archive,
        inventory_path=inventory_path,
        inventory={"schema_version": "test", "files": []},
        binding=binding,
        descriptor_path=None,
        patch_path=None,
    )


def _source_bindings() -> dict[qualification.SourceKey, SourceBinding]:
    return {
        "alberta": SourceBinding(
            provenance_kind="reviewed_snapshot",
            repository=builder.MATCHED_CURRENT_ALBERTA_REPOSITORY,
            base_commit=builder.MATCHED_CURRENT_ALBERTA_BASE_COMMIT,
            tree_git_sha1=None,
            archive_sha256=_sha("alberta-archive"),
            inventory_sha256=_sha("alberta-inventory"),
            snapshot_descriptor_sha256=_sha("alberta-snapshot"),
        ),
        "upstream": SourceBinding(
            provenance_kind="git_tree",
            repository=builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY,
            base_commit=builder.MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
            tree_git_sha1=builder.MATCHED_CURRENT_UPSTREAM_TREE_GIT_SHA1,
            archive_sha256=builder.MATCHED_CURRENT_UPSTREAM_ARCHIVE_SHA256,
            inventory_sha256=builder.MATCHED_CURRENT_UPSTREAM_ARCHIVE_INVENTORY_SHA256,
            snapshot_descriptor_sha256=None,
        ),
        "upstream_rng_isolated": SourceBinding(
            provenance_kind="reviewed_snapshot",
            repository=builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY,
            base_commit=builder.MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
            tree_git_sha1=None,
            archive_sha256=_sha("isolated-archive"),
            inventory_sha256=_sha("isolated-inventory"),
            snapshot_descriptor_sha256=_sha("isolated-snapshot"),
        ),
    }


def _configuration_sources(tmp_path: Path) -> dict[qualification.SourceKey, Any]:
    bindings = _source_bindings()
    upstream = tmp_path / "upstream-source"
    search = upstream / (
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
        "Baselines/Search-Oracle.json"
    )
    search.parent.mkdir(parents=True)
    fixture = (
        _PROJECT_ROOT / "tests/fixtures/forager_matched/Search-Oracle.json"
    ).read_bytes()
    search.write_bytes(fixture.rstrip(b"\n"))
    dqn = upstream / (
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json"
    )
    dqn.parent.mkdir(parents=True, exist_ok=True)
    dqn.write_bytes((_PROJECT_ROOT / "tests/fixtures/forager_matched/DQN.json").read_bytes())
    return {
        key: _dummy_source(key, upstream if key != "alberta" else tmp_path, bindings[key])
        for key in ("alberta", "upstream", "upstream_rng_isolated")
    }


def _artifact_tree_inputs(
    root: Path,
) -> tuple[Any, dict[qualification.SourceKey, Any], dict[str, Any], dict[str, Any]]:
    for name in ("manifest.json", "manifest.json.sha256"):
        (root / name).write_bytes(b"bound")
    bindings = _source_bindings()
    sources: dict[qualification.SourceKey, Any] = {}
    for key in ("alberta", "upstream", "upstream_rng_isolated"):
        source_parent = root / "sources" / key
        source_root = source_parent / "source"
        source_root.mkdir(parents=True)
        archive = source_parent / "source.tar"
        inventory = source_parent / "inventory.json"
        archive.write_bytes(b"archive")
        inventory.write_bytes(b"inventory")
        sources[key] = qualification._StagedSource(  # noqa: SLF001
            key,
            source_root,
            archive,
            inventory,
            {},
            bindings[key],
            None,
            None,
        )
    configuration_root = root / "configurations" / "candidate"
    configuration_root.mkdir(parents=True)
    original = configuration_root / "original.json"
    derived = configuration_root / "derived.json"
    original.write_bytes(b"{}")
    derived.write_bytes(b"{}")
    configurations = {
        "candidate": qualification._MaterializedConfiguration(  # noqa: SLF001
            "candidate",
            original,
            derived,
            None,
        )
    }
    probe = root / "probes/candidate.json"
    receipt = root / "receipts/candidate.json"
    probe.parent.mkdir()
    receipt.parent.mkdir()
    probe.write_bytes(b"{}")
    receipt.write_bytes(b"{}")
    records = {
        "candidate": {
            "probe": {"path": "probes/candidate.json"},
            "capability_receipt": {"path": "receipts/candidate.json"},
        }
    }
    cpu_root = root / "executor-qualification/cpu"
    rng_root = root / "executor-qualification/rng-parity"
    cpu_root.mkdir(parents=True)
    rng_root.mkdir(parents=True)
    executor_qualifications = qualification._StagedExecutorQualifications(  # noqa: SLF001
        cpu_root,
        rng_root,
        {},
        {},
    )
    return executor_qualifications, sources, configurations, records


def _stub_qualification_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qualification,
        "_bind_project_root",
        lambda path: path.resolve(),
    )
    monkeypatch.setattr(
        qualification,
        "_stage_executor_qualification_roots",
        lambda _root: object(),
    )
    monkeypatch.setattr(qualification, "_stage_sources", lambda *_args: {})
    monkeypatch.setattr(qualification, "_materialize_configurations", lambda *_args: {})
    monkeypatch.setattr(qualification, "_probe_invocations", lambda *_args: ())
    runtime = qualification._ProbeRuntimeIdentity(  # noqa: SLF001
        executable=Path("/stub/oci-runtime"),
        executable_sha256=_sha("stub-runtime"),
        version={},
        image_inspection={},
    )
    monkeypatch.setattr(
        qualification,
        "_bind_probe_runtime",
        lambda *_args: runtime,
    )

    def assemble(root: Path, *_args: Any) -> None:
        (root / "staged.marker").write_bytes(b"staged")

    monkeypatch.setattr(qualification, "_assemble_and_write", assemble)
    monkeypatch.setattr(
        qualification,
        "_verify_staged_bundle_in_fresh_process",
        lambda *_args: qualification._FreshReplayClosure(  # noqa: SLF001
            _sha("manifest"),
            _sha("protocol"),
            _sha("plan"),
            _sha("qualification-module"),
        ),
        raising=False,
    )


def _resources(candidate_id: str) -> dict[str, int]:
    if candidate_id in builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS:
        return {
            "parameter_count": 0,
            "optimizer_update_count": 0,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": 100,
        }
    if candidate_id in builder.MATCHED_CURRENT_HORDE_CANDIDATE_IDS:
        return {
            "parameter_count": 1_000,
            "optimizer_update_count": builder.MATCHED_CURRENT_HORIZON,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": (
                64 if candidate_id == "alberta_horde_recurrent64" else 0
            ),
        }
    if candidate_id == "alberta_rtu_h08_taylor":
        return {
            "parameter_count": 1_000,
            "optimizer_update_count": builder.MATCHED_CURRENT_HORIZON,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": 32,
        }
    if candidate_id == "search_oracle":
        return {
            "parameter_count": 0,
            "optimizer_update_count": 0,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": 0,
        }
    replay = {
        "external_dqn_ln": 10_000,
        "external_dqn_crelu": 10_000,
        "external_dqn_plain": 10_000,
        "external_dqn_redo": 1_000,
        "external_drqn_paper": 1_000,
        "isolated_ppo": 0,
        "isolated_rtu": 0,
        "exact_ppo": 0,
    }[candidate_id]
    optimizer_updates = {
        "external_dqn_ln": 124_920,
        "external_dqn_crelu": 124_920,
        "external_dqn_plain": 124_920,
        "external_dqn_redo": 124_915,
        "external_drqn_paper": 124_915,
    }.get(candidate_id, 100)
    return {
        "parameter_count": 1_000,
        "optimizer_update_count": optimizer_updates,
        "replay_capacity_transitions": replay,
        "recurrent_state_elements": (
            64 if candidate_id in {"external_drqn_paper", "isolated_rtu"} else 0
        ),
    }


def _resource_supplement(candidate_id: str) -> dict[str, Any]:
    return {
        "fixed_substrate_parameter_count": (
            1_000 if candidate_id == "alberta_horde_recurrent64" else 0
        ),
        "target_snapshot_parameter_count": (
            1_000
            if candidate_id.startswith("external_dqn")
            or candidate_id == "external_drqn_paper"
            else 0
        ),
        "non_gradient_operations": {
            "causal_nonparametric_transition_updates": (
                builder.MATCHED_CURRENT_HORIZON
                if candidate_id in builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS
                else 0
            ),
            "target_snapshot_refreshes": 0,
            "redo_recycles": 0,
        },
    }


def _probe_payload(invocation: qualification.ProbeInvocation) -> dict[str, Any]:
    parser_identity = {
        "alberta_causal_map": "MatchedAlbertaWorkerConfiguration:CausalMapForagerConfig",
        "alberta_horde_actor_critic": (
            "MatchedAlbertaWorkerConfiguration:AlbertaForagerConfig"
        ),
        "alberta_rtu_rtrl": "MatchedAlbertaWorkerConfiguration:RTURTRLForagerConfig",
        "upstream_ppo_isolated_rng": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_rtu_ppo_isolated_rng": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_ppo": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_dqn_ln": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_crelu": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_plain": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_redo_post_ln": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_drqn": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_search_oracle": "PyExpUtils.ExperimentModel+problem.registry",
    }[invocation.implementation_kind]
    required_literals = {
        "alberta_single_seed_v1": [
            "--configuration",
            "--horizon",
            "--output-root",
            "--seed",
        ],
        "official_foragax_continuing_main_v4": [
            "--exp",
            "--idxs",
            "--max_steps",
            "--save_path",
            "-e",
            "-i",
        ],
        "official_foragax_ppo_frozen_updates_v1": [
            "--exp",
            "--idxs",
            "--save_path",
            "-e",
            "-i",
        ],
    }[invocation.invocation_style]
    if invocation.source_key == "upstream_rng_isolated":
        agent_derivation = "fold_in_isolated_agent_namespace_v1"
        agent_words = [2795197240, 2837457689]
    elif invocation.invocation_style == "official_foragax_ppo_frozen_updates_v1":
        agent_derivation = "shared_root_post_reset_split_v1"
        agent_words = [1797259609, 2579123966]
    else:
        agent_derivation = "effective_seed_constructor_input_v1"
        agent_words = [0, 0]
    return {
        "schema_version": qualification.MATCHED_CURRENT_PROBE_SCHEMA_VERSION,
        "status": "structurally_qualified_content_only",
        "candidate_id": invocation.candidate_id,
        "qualification_seed": 0,
        "source_key": invocation.source_key,
        "qualification_probe": {
            "path": "alberta_framework/benchmarks/forager_matched_qualification.py",
            "sha256": invocation.probe_sha256,
        },
        "configuration": {
            "path": "/run/alberta/configuration.json",
            "sha256": invocation.configuration_sha256,
            "parser_identity": parser_identity,
            "round_trip_accepted": True,
        },
        "entrypoint": {
            "family": invocation.entrypoint_family,
            "path": invocation.entrypoint_path,
            "sha256": invocation.entrypoint_sha256,
            "python_ast_parsed": True,
            "ast_node_count": 1,
            "required_cli_literals": required_literals,
            "required_cli_literals_present": True,
        },
        "implementation_kind": invocation.implementation_kind,
        "resolved_agent": invocation.expected_agent,
        "result_root": invocation.result_root,
        "seed_resolution": {
            "candidate_id": invocation.candidate_id,
            "qualification_seed_class": "public_nonbenchmark_seed",
            "requested_seed": 0,
            "stored_seed": 0,
            "offset": 0,
            "effective_seed": 0,
            "transport": invocation.seed_transport,
            "prng_impl": "threefry2x32",
            "effective_seed_key_words": [0, 0],
            "agent_rng_provenance_derivation": agent_derivation,
            "agent_rng_provenance_key_words": agent_words,
            "environment_transition_count": 0,
            "reward_array_read_count": 0,
        },
        "resources": _resources(invocation.candidate_id),
        "resource_supplement": _resource_supplement(invocation.candidate_id),
        "runtime": {
            "image_sha256": builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256,
            "python": "3.12.3",
            "jax_backend": "cpu",
            "device_platforms": ["cpu"],
        },
        "reward_blind_boundary": {
            "environment_resets": 1,
            "environment_transitions": 0,
            "reward_arrays_read": 0,
            "result_archives_opened": 0,
            "benchmark_seeds_used": [],
        },
        "authority": {
            "identity": qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "external_signature_created": False,
            "trust_profile_created": False,
            "promotion_authorized": False,
            "performance_claim": False,
        },
    }


def test_deterministic_archive_and_filtered_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "agent.py").write_text("value = 1\n", encoding="utf-8")
    (source / "nested").mkdir()
    executable = source / "nested/run.py"
    executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    executable.chmod(0o755)
    (source / "__pycache__").mkdir()
    (source / "__pycache__/agent.pyc").write_bytes(b"not-source")
    staged = tmp_path / "staged"
    qualification._copy_tree(source, staged, alberta_filter=True)  # noqa: SLF001
    assert not (staged / "__pycache__").exists()
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    qualification._archive_tree(staged, first)  # noqa: SLF001
    qualification._archive_tree(staged, second)  # noqa: SLF001
    assert first.read_bytes() == second.read_bytes()
    assert executable.stat().st_mtime_ns != 0


def test_source_tree_rejects_excessive_empty_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        (source / f"empty-{index}").mkdir()
    monkeypatch.setattr(qualification, "_MAX_SOURCE_DIRECTORIES", 2)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_ENTRIES", 10)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="directory bound",
    ):
        qualification._copy_tree(  # noqa: SLF001
            source,
            tmp_path / "staged",
            alberta_filter=False,
        )


def test_tar_extraction_rejects_excessive_empty_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "directories.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for index in range(3):
            member = tarfile.TarInfo(f"empty-{index}/")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_DIRECTORIES", 2)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_ENTRIES", 10)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="directory bound",
    ):
        qualification._extract_git_archive(  # noqa: SLF001
            archive_path,
            tmp_path / "extracted",
        )


def test_tar_extraction_rejects_excessive_total_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "members.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for index in range(3):
            member = tarfile.TarInfo(f"empty-{index}/")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_DIRECTORIES", 10)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_ENTRIES", 2)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="total-member bound",
    ):
        qualification._extract_git_archive(  # noqa: SLF001
            archive_path,
            tmp_path / "extracted",
        )


def test_materializes_every_frozen_worker_envelope_and_external_transform(
    tmp_path: Path,
) -> None:
    sources = _configuration_sources(tmp_path)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    configurations = qualification._materialize_configurations(  # noqa: SLF001
        _PROJECT_ROOT,
        sources,
        artifact_root,
    )
    assert tuple(configurations) == builder.MATCHED_CURRENT_CANDIDATE_IDS
    assert len(builder.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS) == 14
    assert len(configurations) == 23
    for candidate_id, expected in (
        builder.matched_current_alberta_configuration_fingerprints().items()
    ):
        assert hashlib.sha256(configurations[candidate_id].derived.read_bytes()).hexdigest() == (
            expected
        )
    assert configurations["isolated_ppo"].derived.read_bytes() == configurations[
        "exact_ppo"
    ].derived.read_bytes()
    assert configurations["external_dqn_crelu"].binding.derived_sha256 == (
        "4a969466bd5e7d35b937cbc7fc10354f87dc1e39b42b5ee770cf67a73bb47450"
    )
    assert configurations["external_dqn_plain"].binding.derived_sha256 == (
        "cbcaa3949b3c4e898bc615dc157272bd63603cdf25b579aff3a0dddc2d61c7bc"
    )


def test_probe_commands_are_seed_zero_networkless_and_exact_image(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    probe = tmp_path / "probe.py"
    probe.write_bytes(b"# staged probe\n")
    invocation = qualification.ProbeInvocation(
        candidate_id="external_dqn_plain",
        source_key="upstream",
        source_root=source,
        probe_path=probe,
        probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
        configuration=config,
        configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
        entrypoint_path="src/continuing_main.py",
        entrypoint_sha256=_sha("entrypoint"),
        entrypoint_family="continuing_main",
        implementation_kind="upstream_dqn_plain",
        invocation_style="official_foragax_continuing_main_v4",
        result_root="results/results/run/alberta/DQN",
        seed_transport="top_level_seed",
        expected_agent="DQN",
        horizon=builder.MATCHED_CURRENT_HORIZON,
    )
    command = qualification._probe_command("docker", invocation)  # noqa: SLF001
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--seed=0" in command
    assert f"sha256:{builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256}" in command
    assert any(f"source={probe}" in item for item in command)
    assert f"--probe-sha256={invocation.probe_sha256}" in command
    for variable in (
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        assert f"--env={variable}=" in command
    assert not any(str(seed) in command for seed in builder.MATCHED_CURRENT_TUNING_SEEDS)
    assert not any(str(seed) in command for seed in builder.MATCHED_CURRENT_EVALUATION_SEEDS)


def test_git_value_uses_a_bound_executable_sanitized_environment_and_active_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = qualification._GitRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "git",
        executable_sha256=_sha("git-executable"),
    )
    events: list[str] = []
    observed: dict[str, Any] = {}

    def rebind(value: Any) -> None:
        assert value is identity
        events.append("rebind")

    def run(command: Any, **kwargs: Any) -> qualification.QualificationProcessResult:
        events.append("run")
        observed["command"] = tuple(command)
        observed.update(kwargs)
        return qualification.QualificationProcessResult(
            0,
            f"{qualification._QUALIFIED_UPSTREAM_COMMIT}\n".encode("ascii"),  # noqa: SLF001
            b"",
        )

    monkeypatch.setattr(qualification, "_rebind_git_runtime", rebind)
    monkeypatch.setattr(qualification, "_run_bounded_process", run)

    assert qualification._git_value(  # noqa: SLF001
        identity,
        tmp_path,
        "rev-parse",
        "HEAD",
    ) == qualification._QUALIFIED_UPSTREAM_COMMIT  # noqa: SLF001
    assert events == ["rebind", "run", "rebind"]
    assert observed["command"][0] == identity.executable.as_posix()
    assert "--no-replace-objects" in observed["command"]
    assert "tar.umask=0002" in observed["command"]
    assert observed["maximum_stdout_bytes"] == qualification._MAX_GIT_METADATA_BYTES  # noqa: SLF001
    assert observed["maximum_stderr_bytes"] == qualification._MAX_GIT_METADATA_BYTES  # noqa: SLF001
    environment = observed["environment"]
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert not any("proxy" in name.lower() for name in environment)


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        (FileNotFoundError("injected Git launch failure"), "could not run"),
        (
            subprocess.TimeoutExpired(("git", "rev-parse"), 1),
            "could not run",
        ),
        (
            qualification._BoundedProcessOutputError(  # noqa: SLF001
                "injected git metadata overflow"
            ),
            "output exceeds its bound",
        ),
    ),
    ids=("oserror", "timeout", "overflow"),
)
def test_git_value_wraps_runner_failures_in_the_public_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: BaseException,
    message: str,
) -> None:
    identity = qualification._GitRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "git",
        executable_sha256=_sha("git-executable"),
    )
    rebinds = 0

    def rebind(value: Any) -> None:
        nonlocal rebinds
        assert value is identity
        rebinds += 1

    def run(*_args: Any, **_kwargs: Any) -> Any:
        raise fault

    monkeypatch.setattr(qualification, "_rebind_git_runtime", rebind)
    monkeypatch.setattr(qualification, "_run_bounded_process", run)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match=message,
    ) as caught:
        qualification._git_value(  # noqa: SLF001
            identity,
            tmp_path,
            "rev-parse",
            "HEAD",
        )
    assert caught.value.__cause__ is fault
    assert rebinds == 2


def test_default_git_binding_ignores_the_ambient_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "git"
    executable.write_bytes(b"bound-system-git")
    executable.chmod(0o755)
    observed: dict[str, str] = {}

    def which(requested: str, *, path: str | None = None) -> str:
        observed["requested"] = requested
        observed["path"] = cast(str, path)
        return executable.as_posix()

    monkeypatch.setattr(qualification.shutil, "which", which)
    identity = qualification._bind_git_runtime()  # noqa: SLF001

    assert observed == {"requested": "git", "path": os.defpath}
    assert identity.executable == executable
    assert identity.executable_sha256 == hashlib.sha256(
        b"bound-system-git"
    ).hexdigest()


def test_exact_git_archive_streams_to_a_size_bounded_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = b"pinned-archive-bytes"
    identity = qualification._GitRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "git",
        executable_sha256=_sha("git-executable"),
    )
    values = iter(
        (
            qualification._QUALIFIED_UPSTREAM_COMMIT,  # noqa: SLF001
            qualification._QUALIFIED_UPSTREAM_TREE,  # noqa: SLF001
        )
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(qualification, "_bind_git_runtime", lambda: identity)
    monkeypatch.setattr(qualification, "_rebind_git_runtime", lambda value: None)
    monkeypatch.setattr(qualification, "_git_value", lambda *_args: next(values))
    monkeypatch.setattr(
        qualification,
        "_QUALIFIED_UPSTREAM_ARCHIVE_SIZE_BYTES",
        len(archive),
        raising=False,
    )
    monkeypatch.setattr(
        qualification,
        "_QUALIFIED_UPSTREAM_ARCHIVE_SHA256",
        hashlib.sha256(archive).hexdigest(),
    )

    def run(command: Any, **kwargs: Any) -> qualification.QualificationProcessResult:
        observed["command"] = tuple(command)
        observed.update(kwargs)
        assert kwargs["stdout_sink"].write(archive) == len(archive)
        return qualification.QualificationProcessResult(0, b"", b"")

    monkeypatch.setattr(qualification, "_run_bounded_process", run)
    output = tmp_path / "source.tar"
    qualification._build_exact_git_archive(tmp_path, output)  # noqa: SLF001

    assert output.read_bytes() == archive
    assert observed["command"][0] == identity.executable.as_posix()
    assert observed["maximum_stdout_bytes"] == len(archive)
    assert observed["maximum_stderr_bytes"] == qualification._MAX_GIT_METADATA_BYTES  # noqa: SLF001
    assert observed["environment"] == qualification._git_environment()  # noqa: SLF001


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        (FileNotFoundError("injected Git launch failure"), "could not run"),
        (
            subprocess.TimeoutExpired(("git", "archive"), 1),
            "could not run",
        ),
        (
            qualification._BoundedProcessOutputError(  # noqa: SLF001
                "injected git archive overflow"
            ),
            "output exceeds its bound",
        ),
    ),
    ids=("oserror", "timeout", "overflow"),
)
def test_exact_git_archive_wraps_runner_failures_in_the_public_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: BaseException,
    message: str,
) -> None:
    identity = qualification._GitRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "git",
        executable_sha256=_sha("git-executable"),
    )
    values = iter(
        (
            qualification._QUALIFIED_UPSTREAM_COMMIT,  # noqa: SLF001
            qualification._QUALIFIED_UPSTREAM_TREE,  # noqa: SLF001
        )
    )
    monkeypatch.setattr(qualification, "_bind_git_runtime", lambda: identity)
    monkeypatch.setattr(qualification, "_rebind_git_runtime", lambda _value: None)
    monkeypatch.setattr(qualification, "_git_value", lambda *_args: next(values))

    def run(*_args: Any, **_kwargs: Any) -> Any:
        raise fault

    monkeypatch.setattr(qualification, "_run_bounded_process", run)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match=message,
    ) as caught:
        qualification._build_exact_git_archive(  # noqa: SLF001
            tmp_path,
            tmp_path / "source.tar",
        )
    assert caught.value.__cause__ is fault


def test_probe_runtime_binding_uses_one_absolute_executable_and_exact_image(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "docker"
    runtime.write_bytes(b"#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    commands: list[tuple[str, ...]] = []
    version = {"Client": {"Version": "test"}, "Server": {"Version": "test"}}
    inspection = {
        "Id": f"sha256:{builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256}",
        "Config": {
            "Labels": {
                "io.elizaos.alberta.foragax.launcher-contract": (
                    "oci-read-only-stdout-tar-v4"
                )
            }
        },
    }

    def runner(command: Any) -> qualification.QualificationProcessResult:
        materialized = tuple(command)
        commands.append(materialized)
        if materialized[1:3] == ("version", "--format={{json .}}"):
            payload = version
        elif materialized[1:4] == (
            "image",
            "inspect",
            "--format={{json .}}",
        ):
            payload = inspection
        else:
            raise AssertionError(materialized)
        return qualification.QualificationProcessResult(
            0,
            qualification._canonical_json_bytes(payload),  # noqa: SLF001
            b"",
        )

    identity = qualification._bind_probe_runtime(runtime, runner)  # noqa: SLF001
    qualification._rebind_probe_runtime(identity, runner)  # noqa: SLF001

    assert identity.executable == runtime.resolve()
    assert len(commands) == 4
    assert all(command[0] == runtime.resolve().as_posix() for command in commands)
    image_commands = [command for command in commands if command[1] == "image"]
    assert len(image_commands) == 2
    assert all(
        command[-1] == f"sha256:{builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256}"
        for command in image_commands
    )


def test_probe_runtime_binding_rejects_wrong_image_executable_and_daemon_drift(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "docker"
    runtime.write_bytes(b"#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    version = {"Client": {"Version": "test"}, "Server": {"Version": "test"}}

    def result(value: Any) -> qualification.QualificationProcessResult:
        return qualification.QualificationProcessResult(
            0,
            qualification._canonical_json_bytes(value),  # noqa: SLF001
            b"",
        )

    wrong_inspection = {
        "Id": f"sha256:{_sha('wrong-image')}",
        "Config": {
            "Labels": {
                "io.elizaos.alberta.foragax.launcher-contract": (
                    "oci-read-only-stdout-tar-v4"
                )
            }
        },
    }

    def wrong_image_runner(command: Any) -> qualification.QualificationProcessResult:
        return result(version if tuple(command)[1] == "version" else wrong_inspection)

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="different image ID",
    ):
        qualification._bind_probe_runtime(runtime, wrong_image_runner)  # noqa: SLF001

    exact_inspection = {
        "Id": f"sha256:{builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256}",
        "Config": {
            "Labels": {
                "io.elizaos.alberta.foragax.launcher-contract": (
                    "oci-read-only-stdout-tar-v4"
                )
            }
        },
    }
    current_version = version

    def stable_runner(command: Any) -> qualification.QualificationProcessResult:
        return result(current_version if tuple(command)[1] == "version" else exact_inspection)

    identity = qualification._bind_probe_runtime(runtime, stable_runner)  # noqa: SLF001
    runtime.write_bytes(b"#!/bin/sh\nexit 1\n")
    runtime.chmod(0o755)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="executable changed",
    ):
        qualification._rebind_probe_runtime(identity, stable_runner)  # noqa: SLF001

    runtime.write_bytes(b"#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    identity = qualification._bind_probe_runtime(runtime, stable_runner)  # noqa: SLF001
    current_version = {"Client": {"Version": "drift"}, "Server": {"Version": "test"}}
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="daemon version or image identity changed",
    ):
        qualification._rebind_probe_runtime(identity, stable_runner)  # noqa: SLF001


def test_bound_probe_rechecks_runtime_and_transitive_source_on_both_sides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    probe_path = tmp_path / "probe.py"
    probe_path.write_bytes(b"# probe\n")
    configuration = tmp_path / "configuration.json"
    configuration.write_bytes(b"{}")
    invocation = qualification.ProbeInvocation(
        candidate_id="external_dqn_plain",
        source_key="upstream",
        source_root=source_root,
        probe_path=probe_path,
        probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        configuration=configuration,
        configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
        entrypoint_path="src/continuing_main.py",
        entrypoint_sha256=_sha("entrypoint"),
        entrypoint_family="continuing_main",
        implementation_kind="upstream_dqn_plain",
        invocation_style="official_foragax_continuing_main_v4",
        result_root="results/results/run/alberta/DQN",
        seed_transport="top_level_seed",
        expected_agent="DQN",
        horizon=builder.MATCHED_CURRENT_HORIZON,
    )
    source = _dummy_source("upstream", source_root, _source_bindings()["upstream"])
    runtime = qualification._ProbeRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "docker",
        executable_sha256=_sha("runtime"),
        version={},
        image_inspection={},
    )
    events: list[str] = []
    monkeypatch.setattr(
        qualification,
        "_rebind_probe_runtime",
        lambda *_args: events.append("runtime"),
    )
    monkeypatch.setattr(
        qualification,
        "_reverify_staged_source",
        lambda *_args: events.append("source"),
    )

    def run(*_args: Any) -> tuple[dict[str, bool], str]:
        events.append("run")
        return {"ok": True}, _sha("stderr")

    monkeypatch.setattr(qualification, "_run_probe", run)
    observed = qualification._run_bound_probe(  # noqa: SLF001
        runtime,
        invocation,
        source,
        lambda _command: qualification.QualificationProcessResult(1, b"", b""),
    )
    assert observed == ({"ok": True}, _sha("stderr"))
    assert events == ["runtime", "source", "run", "source", "runtime"]


def test_fresh_replay_command_uses_the_bound_networkless_readonly_image(
    tmp_path: Path,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    qualification_root.chmod(0o755)
    runtime = tmp_path / "docker"

    command = qualification._fresh_snapshot_replay_command(  # noqa: SLF001
        runtime,
        qualification_root,
        qualification_module_sha256=_sha("qualification-module"),
    )

    assert command[0] == runtime.as_posix()
    assert command[1:4] == ["run", "--rm", "--pull=never"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--user=65532:65532" in command
    assert "--env=LD_PRELOAD=" in command
    for variable in (
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        assert f"--env={variable}=" in command
    assert not any("PYTHONHASHSEED" in item for item in command)
    assert (
        "--mount=type=bind,"
        f"source={qualification_root},"
        "destination=/qualification/bundle,readonly"
    ) in command
    assert f"sha256:{qualification._QUALIFIED_IMAGE_SHA256}" in command  # noqa: SLF001
    python_index = command.index(qualification._QUALIFIED_PYTHON)  # noqa: SLF001
    assert command[python_index + 1 : python_index + 4] == ["-I", "-B", "-c"]
    assert command[-1] == _sha("qualification-module")


def test_project_root_binding_rejects_a_foreign_source_tree(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign"
    (foreign / "alberta_framework").mkdir(parents=True)

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="loaded Alberta project root",
    ):
        qualification._bind_project_root(foreign)  # noqa: SLF001

    assert qualification._bind_project_root(_PROJECT_ROOT) == _PROJECT_ROOT  # noqa: SLF001


@pytest.mark.parametrize("unsafe_character", [",", "\n", "\r"])
def test_fresh_replay_command_rejects_unsafe_mount_paths(
    tmp_path: Path,
    unsafe_character: str,
) -> None:
    root = tmp_path / f"unsafe{unsafe_character}root"
    root.mkdir()
    root.chmod(0o755)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="unsafe character",
    ):
        qualification._fresh_snapshot_replay_command(  # noqa: SLF001
            tmp_path / "docker",
            root,
            qualification_module_sha256=_sha("module"),
        )


def test_fresh_replay_command_rejects_owner_only_root(tmp_path: Path) -> None:
    root = tmp_path / "qualification"
    root.mkdir()
    root.chmod(0o700)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="not OCI-readable",
    ):
        qualification._fresh_snapshot_replay_command(  # noqa: SLF001
            tmp_path / "docker",
            root,
            qualification_module_sha256=_sha("module"),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "wrong.schema.v1"),
        ("manifest_sha256", _sha("wrong-manifest")),
        ("protocol_sha256", _sha("wrong-protocol")),
        ("plan_sha256", _sha("wrong-plan")),
        ("plan_qualification_manifest_sha256", _sha("wrong-plan-manifest")),
        ("qualification_module_path", "alberta_framework/wrong.py"),
        ("qualification_module_sha256", _sha("wrong-module")),
    ],
)
def test_fresh_replay_rejects_every_child_closure_mismatch(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    root, runtime, module_sha256, payload = _fresh_replay_fixture(tmp_path)
    changed = {**payload, field: replacement}

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="differs from the parent closure",
    ):
        qualification._run_fresh_snapshot_replay(  # noqa: SLF001
            root,
            runtime,
            lambda _command: qualification.QualificationProcessResult(
                0,
                qualification._canonical_json_bytes(changed),  # noqa: SLF001
                b"",
            ),
            expected_manifest_sha256=payload["manifest_sha256"],
            expected_protocol_sha256=payload["protocol_sha256"],
            expected_plan_sha256=payload["plan_sha256"],
            expected_qualification_module_sha256=module_sha256,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong_type", "wrong result type"),
        ("nonzero", "staged replay failed"),
        ("stderr", "unexpected stderr"),
        ("malformed", "staged replay"),
        ("noncanonical", "not canonical JSON"),
        ("extra_field", "fields drifted"),
        ("oversized", "output exceeds"),
    ],
)
def test_fresh_replay_rejects_invalid_runner_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    root, runtime, module_sha256, payload = _fresh_replay_fixture(tmp_path)
    valid = qualification._canonical_json_bytes(payload)  # noqa: SLF001

    def runner(_command: Any) -> Any:
        if case == "wrong_type":
            return object()
        if case == "nonzero":
            return qualification.QualificationProcessResult(7, b"", b"child failed")
        if case == "stderr":
            return qualification.QualificationProcessResult(0, valid, b"warning")
        if case == "malformed":
            return qualification.QualificationProcessResult(0, b"{", b"")
        if case == "noncanonical":
            return qualification.QualificationProcessResult(0, valid + b"\n", b"")
        if case == "extra_field":
            return qualification.QualificationProcessResult(
                0,
                qualification._canonical_json_bytes(  # noqa: SLF001
                    {**payload, "unexpected": True}
                ),
                b"",
            )
        monkeypatch.setattr(qualification, "_MAX_JSON_BYTES", 1)
        return qualification.QualificationProcessResult(0, valid, b"")

    with pytest.raises(qualification.ForagerMatchedQualificationError, match=message):
        qualification._run_fresh_snapshot_replay(  # noqa: SLF001
            root,
            runtime,
            runner,
            expected_manifest_sha256=payload["manifest_sha256"],
            expected_protocol_sha256=payload["protocol_sha256"],
            expected_plan_sha256=payload["plan_sha256"],
            expected_qualification_module_sha256=module_sha256,
        )


def test_fresh_replay_wraps_injected_runner_overflow(
    tmp_path: Path,
) -> None:
    root, runtime, module_sha256, payload = _fresh_replay_fixture(tmp_path)
    overflow = qualification._BoundedProcessOutputError(  # noqa: SLF001
        "injected fresh replay output overflow"
    )

    def runner(_command: Any) -> Any:
        raise overflow

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="fresh-process staged replay runner output exceeds its bound",
    ) as caught:
        qualification._run_fresh_snapshot_replay(  # noqa: SLF001
            root,
            runtime,
            runner,
            expected_manifest_sha256=payload["manifest_sha256"],
            expected_protocol_sha256=payload["protocol_sha256"],
            expected_plan_sha256=payload["plan_sha256"],
            expected_qualification_module_sha256=module_sha256,
        )
    assert caught.value.__cause__ is overflow


def test_fresh_bundle_replay_rebinds_runtime_and_reloads_both_sides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "qualification"
    staged_module = (
        root
        / "sources"
        / "alberta"
        / "source"
        / "alberta_framework"
        / "benchmarks"
        / "forager_matched_qualification.py"
    )
    staged_module.parent.mkdir(parents=True)
    staged_module.write_bytes(
        Path(qualification.__file__).resolve(strict=True).read_bytes()
    )
    qualification._normalize_qualification_tree_permissions(root)  # noqa: SLF001
    manifest = {
        "schema_version": qualification.MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION
    }
    manifest_bytes = qualification._canonical_json_bytes(manifest)  # noqa: SLF001
    bundle = qualification.MatchedCurrentQualificationBundle(
        output_root=root,
        cpu_qualification_root=root / "cpu",
        rng_parity_qualification_root=root / "rng",
        runtime_qualification=object(),
        candidate_qualifications={},
        candidate_assets={},
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    protocol = SimpleNamespace(protocol_sha256=_sha("protocol"))
    plan = SimpleNamespace(protocol=protocol, plan_sha256=_sha("plan"))
    runtime = qualification._ProbeRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "docker",
        executable_sha256=_sha("runtime"),
        version={},
        image_inspection={},
    )
    events: list[str] = []

    def build(_bundle: Any) -> tuple[Any, Any]:
        events.append("build")
        return protocol, plan

    monkeypatch.setattr(qualification, "build_open_protocol_and_execution_plan", build)
    monkeypatch.setattr(
        qualification,
        "_rebind_probe_runtime",
        lambda *_args: events.append("runtime"),
    )
    monkeypatch.setattr(
        qualification,
        "_run_fresh_snapshot_replay",
        lambda *_args, **_kwargs: events.append("child"),
    )

    def reload(_root: Path) -> Any:
        events.append("reload")
        return bundle

    monkeypatch.setattr(qualification, "load_matched_current_qualification_bundle", reload)

    closure = qualification._verify_staged_bundle_in_fresh_process(  # noqa: SLF001
        bundle,
        runtime,
        lambda _command: qualification.QualificationProcessResult(1, b"", b""),
    )
    assert closure.manifest_sha256 == bundle.manifest_sha256
    assert events == ["build", "runtime", "child", "runtime", "reload", "build"]

    events.clear()

    def fail_child(*_args: Any, **_kwargs: Any) -> None:
        events.append("child")
        raise qualification.ForagerMatchedQualificationError("injected child failure")

    monkeypatch.setattr(qualification, "_run_fresh_snapshot_replay", fail_child)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="injected child failure",
    ):
        qualification._verify_staged_bundle_in_fresh_process(  # noqa: SLF001
            bundle,
            runtime,
            lambda _command: qualification.QualificationProcessResult(1, b"", b""),
        )
    assert events == ["build", "runtime", "child", "runtime"]


@pytest.mark.parametrize(
    ("failure_phase", "cleanup_error", "expected_type", "expected_message"),
    (
        (
            "kill",
            OSError("injected kill failure"),
            qualification.ForagerMatchedQualificationError,
            "bounded child could not be terminated cleanly",
        ),
        (
            "kill",
            ProcessLookupError("injected exited-child race"),
            qualification._BoundedProcessOutputError,  # noqa: SLF001
            "active byte limit",
        ),
        (
            "wait",
            subprocess.TimeoutExpired(("unused",), 10),
            qualification.ForagerMatchedQualificationError,
            "bounded child could not be reaped after termination",
        ),
        (
            "wait",
            OSError("injected wait failure"),
            qualification.ForagerMatchedQualificationError,
            "bounded child could not be inspected after termination",
        ),
        (
            "selector_close",
            OSError("injected selector close failure"),
            qualification.ForagerMatchedQualificationError,
            "bounded child resources could not be closed cleanly",
        ),
        (
            "stdout_close",
            OSError("injected stdout close failure"),
            qualification.ForagerMatchedQualificationError,
            "bounded child resources could not be closed cleanly",
        ),
        (
            "stderr_close",
            OSError("injected stderr close failure"),
            qualification.ForagerMatchedQualificationError,
            "bounded child resources could not be closed cleanly",
        ),
    ),
    ids=(
        "kill-error",
        "kill-process-gone",
        "reap-timeout",
        "reap-error",
        "selector-close-error",
        "stdout-close-error",
        "stderr-close-error",
    ),
)
def test_bounded_process_normalizes_cleanup_failures_and_attempts_every_close(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    cleanup_error: BaseException,
    expected_type: type[BaseException],
    expected_message: str,
) -> None:
    class CloseBuffer(io.BytesIO):
        def __init__(self, failure_name: str) -> None:
            super().__init__()
            self.failure_name = failure_name
            self.close_called = False

        def close(self) -> None:
            self.close_called = True
            super().close()
            if failure_phase == self.failure_name:
                raise cleanup_error

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = CloseBuffer("stdout_close")
            self.stderr = CloseBuffer("stderr_close")
            self.kill_called = False
            self.waited = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.kill_called = True
            if failure_phase == "kill":
                raise cleanup_error

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waited = True
            if failure_phase == "wait":
                raise cleanup_error
            return -9

    class SelectorKey:
        data = "stdout"
        fd = 101

    class OverflowSelector:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self) -> OverflowSelector:
            return self

        def __exit__(self, *_args: Any) -> None:
            self.close()

        def register(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def get_map(self) -> dict[str, bool]:
            return {"active": True}

        def select(self, _timeout: float) -> list[tuple[SelectorKey, int]]:
            return [(SelectorKey(), selectors.EVENT_READ)]

        def close(self) -> None:
            self.closed = True
            if failure_phase == "selector_close":
                raise cleanup_error

    process = FakeProcess()
    selector = OverflowSelector()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(os, "read", lambda _descriptor, _size: b"xx")

    with pytest.raises(expected_type, match=expected_message) as caught:
        qualification._run_bounded_process(  # noqa: SLF001
            ("unused",),
            timeout=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
        )

    assert process.kill_called is True
    assert process.waited is True
    assert selector.closed is True
    assert process.stdout.close_called is True
    assert process.stderr.close_called is True
    if isinstance(cleanup_error, ProcessLookupError):
        assert caught.value.__cause__ is None
    else:
        assert caught.value.__cause__ is cleanup_error


def test_default_runner_actively_rejects_oversized_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "b" * 64
    calls: list[tuple[str, ...]] = []

    def exceed(command: Any, **_kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        cid_argument = next(item for item in materialized if item.startswith("--cidfile="))
        Path(cid_argument.split("=", 1)[1]).write_text(container_id, encoding="ascii")
        raise qualification._BoundedProcessOutputError("injected overflow")  # noqa: SLF001

    def fake_cleanup(command: Any, **_kwargs: Any) -> Any:
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(qualification, "_run_bounded_process", exceed)
    monkeypatch.setattr(subprocess, "run", fake_cleanup)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="output exceeds.*cleanup=force_removed",
    ):
        qualification._default_runner(("fake-runtime", "run", "image"))  # noqa: SLF001
    assert calls[1] == ("fake-runtime", "rm", "--force", container_id)


def test_default_runner_force_removes_probe_container_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "a" * 64
    calls: list[tuple[str, ...]] = []

    def timeout(command: Any, **kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        cid_argument = next(item for item in materialized if item.startswith("--cidfile="))
        Path(cid_argument.split("=", 1)[1]).write_text(container_id, encoding="ascii")
        raise subprocess.TimeoutExpired(materialized, kwargs["timeout"])

    def fake_cleanup(command: Any, **_kwargs: Any) -> Any:
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(qualification, "_run_bounded_process", timeout)
    monkeypatch.setattr(subprocess, "run", fake_cleanup)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="cleanup=force_removed",
    ):
        qualification._default_runner(("fake-runtime", "run", "image"))  # noqa: SLF001
    assert calls[1] == ("fake-runtime", "rm", "--force", container_id)


def test_default_runner_force_removes_probe_by_name_before_cidfile_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    observed_name = ""

    def timeout(command: Any, **kwargs: Any) -> Any:
        nonlocal observed_name
        materialized = tuple(command)
        calls.append(materialized)
        name_argument = next(item for item in materialized if item.startswith("--name="))
        observed_name = name_argument.split("=", 1)[1]
        assert re.fullmatch(r"alberta-matched-qualification-[0-9a-f]{32}", observed_name)
        raise subprocess.TimeoutExpired(materialized, kwargs["timeout"])

    def fake_cleanup(command: Any, **_kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        assert materialized == ("fake-runtime", "rm", "--force", observed_name)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(qualification, "_run_bounded_process", timeout)
    monkeypatch.setattr(subprocess, "run", fake_cleanup)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="cleanup=force_removed_by_name",
    ):
        qualification._default_runner(("fake-runtime", "run", "image"))  # noqa: SLF001
    assert len(calls) == 2


def test_probe_cleanup_uses_exact_name_for_a_partial_cidfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cidfile = tmp_path / "container.cid"
    cidfile.write_bytes(b"partial")
    container_name = "alberta-matched-qualification-" + "a" * 32
    observed: list[tuple[str, ...]] = []

    def fake_cleanup(command: Any, **_kwargs: Any) -> Any:
        materialized = tuple(command)
        observed.append(materialized)
        assert materialized == ("fake-runtime", "rm", "--force", container_name)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(subprocess, "run", fake_cleanup)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="cidfile contract failed after cleanup=force_removed_by_name",
    ):
        qualification._cleanup_interrupted_probe_container(  # noqa: SLF001
            ("fake-runtime", "run"),
            cidfile,
            container_name,
        )
    assert len(observed) == 1


def test_probe_cleanup_wraps_cidfile_read_oserror_after_exact_name_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cidfile = tmp_path / "container.cid"
    cidfile.write_bytes(b"a" * 64)
    container_name = "alberta-matched-qualification-" + "b" * 32
    read_error = FileNotFoundError("injected cidfile disappearance")
    observed: list[tuple[str, ...]] = []

    def fail_read(*_args: Any, **_kwargs: Any) -> Any:
        raise read_error

    def cleanup(command: Any, **_kwargs: Any) -> Any:
        materialized = tuple(command)
        observed.append(materialized)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(qualification, "_read_stable", fail_read)
    monkeypatch.setattr(subprocess, "run", cleanup)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="cidfile contract failed after cleanup=force_removed_by_name",
    ) as caught:
        qualification._cleanup_interrupted_probe_container(  # noqa: SLF001
            ("fake-runtime", "run"),
            cidfile,
            container_name,
        )
    assert caught.value.__cause__ is read_error
    assert observed == [("fake-runtime", "rm", "--force", container_name)]


def test_default_runner_cleans_a_completed_nonzero_container_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    observed_name = ""

    def completed(command: Any, **_kwargs: Any) -> Any:
        nonlocal observed_name
        materialized = tuple(command)
        calls.append(materialized)
        observed_name = next(
            item.split("=", 1)[1]
            for item in materialized
            if item.startswith("--name=")
        )
        return qualification.QualificationProcessResult(125, b"", b"failed")

    def cleanup(command: Any, **_kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        assert materialized == ("fake-runtime", "rm", "--force", observed_name)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(qualification, "_run_bounded_process", completed)
    monkeypatch.setattr(subprocess, "run", cleanup)
    result = qualification._default_runner(  # noqa: SLF001
        ("fake-runtime", "run", "image")
    )
    assert result.returncode == 125
    assert len(calls) == 2


def test_probe_cleanup_accepts_bounded_proof_that_name_is_already_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "alberta-matched-qualification-" + "c" * 32
    commands: list[tuple[str, ...]] = []

    def missing(command: Any, **_kwargs: Any) -> Any:
        materialized = tuple(command)
        commands.append(materialized)
        return subprocess.CompletedProcess(materialized, 1)

    def inspect(command: Any, **_kwargs: Any) -> Any:
        materialized = tuple(command)
        commands.append(materialized)
        return qualification.QualificationProcessResult(0, b"", b"")

    monkeypatch.setattr(subprocess, "run", missing)
    monkeypatch.setattr(qualification, "_run_bounded_process", inspect)
    state = qualification._cleanup_interrupted_probe_container(  # noqa: SLF001
        ("fake-runtime", "run"),
        tmp_path / "missing.cid",
        container_name,
    )
    assert state == "already_absent_by_name"
    assert commands == [
        ("fake-runtime", "rm", "--force", container_name),
        (
            "fake-runtime",
            "container",
            "ls",
            "--all",
            "--quiet",
            "--no-trunc",
            f"--filter=name=^/{container_name}$",
        ),
    ]


def test_probe_cleanup_wraps_bounded_absence_query_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "alberta-matched-qualification-" + "d" * 32
    overflow = qualification._BoundedProcessOutputError(  # noqa: SLF001
        "injected cleanup inspection overflow"
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1),
    )

    def inspect(*_args: Any, **_kwargs: Any) -> Any:
        raise overflow

    monkeypatch.setattr(qualification, "_run_bounded_process", inspect)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="could not prove the exact name absent",
    ) as caught:
        qualification._cleanup_interrupted_probe_container(  # noqa: SLF001
            ("fake-runtime", "run"),
            tmp_path / "missing.cid",
            container_name,
        )
    assert caught.value.__cause__ is overflow


@pytest.mark.parametrize(
    "caller_option",
    ("--name=caller-owned", "--cidfile=/tmp/caller-owned.cid"),
)
def test_default_runner_rejects_caller_owned_cleanup_identifiers(
    caller_option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def run(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("rejected commands must not run")

    monkeypatch.setattr(qualification, "_run_bounded_process", run)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="already contains a name or cidfile",
    ):
        qualification._default_runner(  # noqa: SLF001
            ("fake-runtime", "run", caller_option, "image")
        )
    assert called is False


def test_probe_cleanup_fails_if_the_exact_name_still_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "alberta-matched-qualification-" + "e" * 32

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1),
    )
    monkeypatch.setattr(
        qualification,
        "_run_bounded_process",
        lambda *_args, **_kwargs: qualification.QualificationProcessResult(
            0,
            b"f" * 64 + b"\n",
            b"",
        ),
    )
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="did not remove or prove absent",
    ):
        qualification._cleanup_interrupted_probe_container(  # noqa: SLF001
            ("fake-runtime", "run"),
            tmp_path / "missing.cid",
            container_name,
        )


@pytest.mark.parametrize(
    "fault",
    (
        FileNotFoundError("injected inspector launch failure"),
        subprocess.TimeoutExpired(("docker", "version"), 1),
        qualification._BoundedProcessOutputError(  # noqa: SLF001
            "injected inspector output overflow"
        ),
    ),
    ids=("oserror", "timeout", "overflow"),
)
def test_executor_runner_adapter_wraps_injected_runner_failures(
    fault: BaseException,
) -> None:
    def runner(_command: Any) -> Any:
        raise fault

    adapted = qualification._executor_runner_adapter(runner)  # noqa: SLF001
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="qualification runtime inspector runner",
    ) as caught:
        adapted(("docker", "version"))
    assert caught.value.__cause__ is fault


def test_probe_result_accepts_only_reward_blind_unendorsed_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    probe = tmp_path / "probe.py"
    probe.write_bytes(b"# staged probe\n")
    invocation = qualification.ProbeInvocation(
        candidate_id="external_dqn_plain",
        source_key="upstream",
        source_root=source,
        probe_path=probe,
        probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
        configuration=config,
        configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
        entrypoint_path="src/continuing_main.py",
        entrypoint_sha256=_sha("entrypoint"),
        entrypoint_family="continuing_main",
        implementation_kind="upstream_dqn_plain",
        invocation_style="official_foragax_continuing_main_v4",
        result_root="results/results/run/alberta/DQN",
        seed_transport="top_level_seed",
        expected_agent="DQN",
        horizon=builder.MATCHED_CURRENT_HORIZON,
    )
    payload = _probe_payload(invocation)

    def runner(_command: Any) -> qualification.QualificationProcessResult:
        return qualification.QualificationProcessResult(
            0,
            qualification._canonical_json_bytes(payload),  # noqa: SLF001
            b"",
        )

    parsed, stderr_sha = qualification._run_probe(  # noqa: SLF001
        "docker",
        invocation,
        runner,
    )
    assert parsed == payload
    assert stderr_sha == hashlib.sha256(b"").hexdigest()
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="unexpected stderr",
    ):
        qualification._run_probe(  # noqa: SLF001
            "docker",
            invocation,
            lambda _command: qualification.QualificationProcessResult(
                0,
                qualification._canonical_json_bytes(payload),  # noqa: SLF001
                b"warning\n",
            ),
        )
    payload["reward_blind_boundary"]["reward_arrays_read"] = 1
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="reward-blind",
    ):
        qualification._run_probe("docker", invocation, runner)  # noqa: SLF001

    overflow = qualification._BoundedProcessOutputError(  # noqa: SLF001
        "injected probe output overflow"
    )

    def overflow_runner(_command: Any) -> Any:
        raise overflow

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="OCI probe runner output exceeds its bound",
    ) as caught:
        qualification._run_probe(  # noqa: SLF001
            "docker",
            invocation,
            overflow_runner,
        )
    assert caught.value.__cause__ is overflow


@pytest.mark.parametrize(
    ("section", "field", "alias"),
    [
        (None, "qualification_seed", 0.0),
        ("seed_resolution", "requested_seed", False),
        ("reward_blind_boundary", "environment_resets", True),
        ("configuration", "round_trip_accepted", 1),
        ("authority", "content_only", 1),
        ("authority", "promotion_authorized", 0),
    ],
)
def test_persisted_probe_rejects_wrong_type_numeric_aliases(
    tmp_path: Path,
    section: str | None,
    field: str,
    alias: object,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    probe = tmp_path / "probe.py"
    probe.write_bytes(b"# staged probe\n")
    invocation = qualification.ProbeInvocation(
        candidate_id="external_dqn_plain",
        source_key="upstream",
        source_root=source,
        probe_path=probe,
        probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
        configuration=config,
        configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
        entrypoint_path="src/continuing_main.py",
        entrypoint_sha256=_sha("entrypoint"),
        entrypoint_family="continuing_main",
        implementation_kind="upstream_dqn_plain",
        invocation_style="official_foragax_continuing_main_v4",
        result_root="results/results/run/alberta/DQN",
        seed_transport="top_level_seed",
        expected_agent="DQN",
        horizon=builder.MATCHED_CURRENT_HORIZON,
    )
    payload = _probe_payload(invocation)
    target = payload if section is None else payload[section]
    target[field] = alias

    with pytest.raises(qualification.ForagerMatchedQualificationError):
        qualification._verify_probe_payload(payload, invocation)  # noqa: SLF001


@pytest.mark.parametrize(
    ("buffer_min_size", "expected"),
    [(32, 124_920), (50, 124_915)],
)
def test_replay_optimizer_count_excludes_warmup_gates(
    buffer_min_size: int,
    expected: int,
) -> None:
    assert qualification._exact_continuing_main_v4_optimizer_update_count(  # noqa: SLF001
        horizon=builder.MATCHED_CURRENT_HORIZON,
        buffer_min_size=buffer_min_size,
        update_frequency=4,
        initial_agent_step=1,
        freeze_steps=float("inf"),
    ) == expected


def test_candidate_qualification_values_close_under_open_protocol(tmp_path: Path) -> None:
    sources = _configuration_sources(tmp_path)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    configurations = qualification._materialize_configurations(  # noqa: SLF001
        _PROJECT_ROOT,
        sources,
        artifact_root,
    )
    source_roots = {
        key: _dummy_source(key, source.root, _source_bindings()[key])
        for key, source in sources.items()
    }
    probes = {
        candidate_id: {
            "seed_resolution": {
                "candidate_id": candidate_id,
                "requested_seed": 0,
                "effective_seed": 0,
            },
            "resources": _resources(candidate_id),
        }
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    receipt_digests = {
        candidate_id: _sha(f"receipt:{candidate_id}")
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    values = qualification._candidate_qualifications(  # noqa: SLF001
        source_roots,
        configurations,
        probes,
        receipt_digests,
    )
    protocol = builder.build_forager_matched_open_protocol(
        runtime=qualification._runtime_qualification(),  # noqa: SLF001
        candidate_qualifications=values,
    )
    assert tuple(protocol.candidate_index) == builder.MATCHED_CURRENT_CANDIDATE_IDS
    assert len(protocol.candidates) == 23
    assert all(
        candidate.runtime_binding.qualification_trust_anchor_identity
        == qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY
        for candidate in protocol.candidates
    )


def test_bundle_preserves_exact_manifest_bytes_and_threads_one_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "schema_version": qualification.MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION,
        "payload": {"first": 1, "second": 2, "label": "Alberta café"},
    }
    manifest_bytes = qualification._canonical_json_bytes(manifest)  # noqa: SLF001
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    caller_candidate_qualifications: dict[str, Any] = {}
    caller_candidate_assets: dict[str, Any] = {"fixture": object()}
    bundle = qualification.MatchedCurrentQualificationBundle(
        output_root=tmp_path,
        cpu_qualification_root=tmp_path / "cpu",
        rng_parity_qualification_root=tmp_path / "rng-parity",
        runtime_qualification=object(),
        candidate_qualifications=caller_candidate_qualifications,
        candidate_assets=caller_candidate_assets,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
    )
    built_protocol = object()
    built_plan = object()
    captured: dict[str, Any] = {}

    def build_protocol(**kwargs: Any) -> object:
        captured["protocol"] = kwargs
        return built_protocol

    def build_plan(protocol: object, assets: dict[str, Any], **kwargs: Any) -> object:
        captured["plan_protocol"] = protocol
        captured["plan_assets"] = assets
        captured["plan"] = kwargs
        return built_plan

    monkeypatch.setattr(builder, "build_forager_matched_open_protocol", build_protocol)
    monkeypatch.setattr(executor, "build_execution_plan", build_plan)

    protocol, plan = qualification.build_open_protocol_and_execution_plan(bundle)

    assert bundle.manifest_bytes is manifest_bytes
    assert hashlib.sha256(bundle.manifest_bytes).hexdigest() == bundle.manifest_sha256
    assert captured["plan"]["qualification_manifest_sha256"] == manifest_sha256
    assert captured["plan_protocol"] is built_protocol
    assert captured["plan_assets"] == dict(bundle.candidate_assets)
    assert protocol is built_protocol
    assert plan is built_plan

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="exact canonical content",
    ):
        replace(bundle, manifest_bytes=b"{}")
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="digest differs",
    ):
        replace(bundle, manifest_sha256=_sha("different-manifest"))
    manifest["payload"]["first"] = 999
    caller_candidate_qualifications["late"] = object()
    caller_candidate_assets["late"] = object()
    assert bundle.manifest["payload"]["first"] == 1
    assert "late" not in bundle.candidate_qualifications
    assert "late" not in bundle.candidate_assets
    with pytest.raises(TypeError):
        cast(Any, bundle.manifest["payload"])["first"] = 999
    with pytest.raises(TypeError):
        cast(Any, bundle.candidate_qualifications)["replacement"] = object()
    with pytest.raises(TypeError):
        cast(Any, bundle.candidate_assets)["replacement"] = object()

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="bounded canonical JSON",
    ):
        replace(
            bundle,
            manifest={
                "schema_version": qualification.MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION,
                "nonfinite": float("nan"),
            },
        )
    cyclic: dict[str, Any] = {
        "schema_version": qualification.MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION,
    }
    cyclic["cycle"] = cyclic
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="bounded canonical JSON",
    ):
        replace(bundle, manifest=cyclic)


def test_capability_receipts_remain_content_only_and_patch_both_isolated_arms(
    tmp_path: Path,
) -> None:
    sources = _configuration_sources(tmp_path)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    configurations = qualification._materialize_configurations(  # noqa: SLF001
        _PROJECT_ROOT,
        sources,
        artifact_root,
    )
    for source in sources.values():
        (source.root / "src").mkdir(exist_ok=True)
        (source.root / "src/continuing_main.py").write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('-e', '--exp')\n"
            "parser.add_argument('-i', '--idxs')\n"
            "parser.add_argument('--max_steps')\n"
            "parser.add_argument('--save_path')\n",
            encoding="utf-8",
        )
        (source.root / "src/rtu_ppo.py").write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('-e', '--exp')\n"
            "parser.add_argument('-i', '--idxs')\n"
            "parser.add_argument('--save_path')\n",
            encoding="utf-8",
        )
    alberta_entrypoint = (
        sources["alberta"].root
        / "alberta_framework/benchmarks/_forager_matched_alberta_worker.py"
    )
    alberta_entrypoint.parent.mkdir(parents=True, exist_ok=True)
    alberta_entrypoint.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--configuration')\n"
        "parser.add_argument('--seed')\n"
        "parser.add_argument('--horizon')\n"
        "parser.add_argument('--output-root')\n",
        encoding="utf-8",
    )
    staged_probe = (
        sources["alberta"].root
        / "alberta_framework/benchmarks/forager_matched_qualification.py"
    )
    staged_probe.write_bytes(
        (_PROJECT_ROOT / "alberta_framework/benchmarks/forager_matched_qualification.py")
        .read_bytes()
    )
    invocations = qualification._probe_invocations(sources, configurations)  # noqa: SLF001
    probes = {
        invocation.candidate_id: _probe_payload(invocation)
        for invocation in invocations
    }
    placeholder = {
        candidate_id: _sha(f"placeholder:{candidate_id}")
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    values = qualification._candidate_qualifications(  # noqa: SLF001
        sources,
        configurations,
        probes,
        placeholder,
    )
    protocol = builder.build_forager_matched_open_protocol(
        runtime=qualification._runtime_qualification(),  # noqa: SLF001
        candidate_qualifications=values,
    )
    invocation_index = {item.candidate_id: item for item in invocations}
    receipts = {
        candidate_id: qualification._capability_receipt(  # noqa: SLF001
            protocol.candidate_index[candidate_id],
            invocation_index[candidate_id],
        )
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    assert all(
        receipt["qualification_trust_anchor_identity"]
        == qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY
        for receipt in receipts.values()
    )
    assert receipts["isolated_ppo"]["rng_isolation_patch_sha256"] == (
        qualification._QUALIFIED_RNG_PATCH_SHA256  # noqa: SLF001
    )
    assert receipts["isolated_rtu"]["rng_isolation_patch_sha256"] == (
        qualification._QUALIFIED_RNG_PATCH_SHA256  # noqa: SLF001
    )
    assert receipts["exact_ppo"]["rng_isolation_patch_sha256"] is None
    assert all("signature" not in receipt for receipt in receipts.values())


def test_public_api_rejects_existing_output_before_staging(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    output = tmp_path / "already-present"
    output.mkdir()
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="already exists",
    ):
        qualification.qualify_matched_current_candidates(
            _PROJECT_ROOT,
            upstream,
            output,
            runner=lambda _command: qualification.QualificationProcessResult(1, b"", b""),
        )


@pytest.mark.parametrize("input_name", ["alberta", "upstream", "cpu", "rng"])
def test_public_api_rejects_output_nested_in_every_staged_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
) -> None:
    project = tmp_path / "project"
    alberta_source = project / "alberta_framework"
    alberta_source.mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu-qualification"
    cpu_root.mkdir()
    rng_root = tmp_path / "rng-qualification"
    rng_root.mkdir()
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    inputs = {
        "alberta": alberta_source,
        "upstream": upstream,
        "cpu": cpu_root,
        "rng": rng_root,
    }
    output = inputs[input_name] / "uncreated-parent/qualification"
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="output root overlaps",
    ):
        qualification.qualify_matched_current_candidates(
            project,
            upstream,
            output,
            runner=lambda _command: qualification.QualificationProcessResult(1, b"", b""),
        )
    assert not output.parent.exists()


def test_public_api_allows_intended_outputs_forager_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    forager_root = project / "outputs/forager"
    cpu_root = forager_root / "cpu-qualification"
    rng_root = forager_root / "rng-qualification"
    cpu_root.mkdir(parents=True)
    rng_root.mkdir()
    output = forager_root / "matched-current-qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    monkeypatch.setattr(
        qualification,
        "_bind_project_root",
        lambda path: path.resolve(),
    )

    class StoppedBeforeStagingError(RuntimeError):
        pass

    def stop_before_staging(*_args: Any, **_kwargs: Any) -> str:
        raise StoppedBeforeStagingError

    monkeypatch.setattr(tempfile, "mkdtemp", stop_before_staging)
    with pytest.raises(StoppedBeforeStagingError):
        qualification.qualify_matched_current_candidates(
            project,
            upstream,
            output,
            runner=lambda _command: qualification.QualificationProcessResult(1, b"", b""),
        )
    assert not output.exists()


def test_publication_is_atomic_and_never_replaces_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "partial"
    source.mkdir()
    (source / "manifest.json").write_bytes(b"{}")
    destination = tmp_path / "final"
    qualification._publish_directory_no_replace(  # noqa: SLF001
        source,
        destination,
        tmp_path,
    )
    assert not source.exists()
    assert (destination / "manifest.json").read_bytes() == b"{}"

    second = tmp_path / "second-partial"
    second.mkdir()
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="created concurrently",
    ):
        qualification._publish_directory_no_replace(  # noqa: SLF001
            second,
            destination,
            tmp_path,
        )
    assert second.is_dir()
    assert (destination / "manifest.json").read_bytes() == b"{}"


def test_publication_parent_fsync_failure_is_published_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "partial"
    source.mkdir()
    (source / "manifest.json").write_bytes(b"{}")
    destination = tmp_path / "final"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    real_fsync = os.fsync

    def fail_parent_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == parent_identity:
            raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    with pytest.raises(
        qualification.QualificationPublishedButUncertainError,
        match="do not reuse this output root",
    ):
        qualification._publish_directory_no_replace(  # noqa: SLF001
            source,
            destination,
            tmp_path,
            expected_parent_identity=parent_identity,
            expected_source_identity=source_identity,
        )

    assert not source.exists()
    assert (destination / "manifest.json").read_bytes() == b"{}"


def test_publication_rejects_parent_or_staging_inode_substitution(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "partial"
    source.mkdir()
    parent_identity = (parent.stat().st_dev, parent.stat().st_ino)
    source_identity = (source.stat().st_dev, source.stat().st_ino)

    displaced = tmp_path / "displaced-parent"
    parent.rename(displaced)
    parent.mkdir()
    replacement = parent / "partial"
    replacement.mkdir()
    destination = parent / "final"
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="parent changed before publication",
    ):
        qualification._publish_directory_no_replace(  # noqa: SLF001
            replacement,
            destination,
            parent,
            expected_parent_identity=parent_identity,
            expected_source_identity=source_identity,
        )
    assert not destination.exists()

    stable_parent = tmp_path / "stable-parent"
    stable_parent.mkdir()
    original = stable_parent / "partial"
    original.mkdir()
    original_identity = (original.stat().st_dev, original.stat().st_ino)
    original.rename(stable_parent / "displaced-partial")
    substituted = stable_parent / "partial"
    substituted.mkdir()
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="staging directory changed before publication",
    ):
        qualification._publish_directory_no_replace(  # noqa: SLF001
            substituted,
            stable_parent / "final",
            stable_parent,
            expected_parent_identity=(
                stable_parent.stat().st_dev,
                stable_parent.stat().st_ino,
            ),
            expected_source_identity=original_identity,
        )
    assert not (stable_parent / "final").exists()


def test_publication_detects_destination_inode_substitution_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "partial"
    source.mkdir()
    (source / "manifest.json").write_bytes(b"{}")
    destination = tmp_path / "final"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    real_fsync = os.fsync

    def substitute_after_rename(descriptor: int) -> None:
        real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == parent_identity:
            destination.rename(tmp_path / "displaced-final")
            destination.mkdir()

    monkeypatch.setattr(os, "fsync", substitute_after_rename)
    with pytest.raises(
        qualification.QualificationPublishedButUncertainError,
        match="do not reuse this output root",
    ):
        qualification._publish_directory_no_replace(  # noqa: SLF001
            source,
            destination,
            tmp_path,
            expected_parent_identity=parent_identity,
            expected_source_identity=source_identity,
        )

    assert destination.is_dir()
    assert (tmp_path / "displaced-final/manifest.json").read_bytes() == b"{}"


def test_publication_detects_destination_replacement_inside_validator(
    tmp_path: Path,
) -> None:
    source = tmp_path / "partial"
    source.mkdir()
    (source / "manifest.json").write_bytes(b"{}")
    destination = tmp_path / "final"
    displaced = tmp_path / "displaced-final"

    def replace_during_validation(path: Path) -> None:
        assert path == destination
        path.rename(displaced)
        path.mkdir()

    with pytest.raises(
        qualification.QualificationPublishedButUncertainError,
        match="do not reuse this output root",
    ):
        qualification._publish_directory_no_replace(  # noqa: SLF001
            source,
            destination,
            tmp_path,
            expected_parent_identity=(tmp_path.stat().st_dev, tmp_path.stat().st_ino),
            expected_source_identity=(source.stat().st_dev, source.stat().st_ino),
            post_publish_validator=replace_during_validation,
        )

    assert destination.is_dir()
    assert (displaced / "manifest.json").read_bytes() == b"{}"


def test_verified_tree_fsyncs_files_and_directories_bottom_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "staging"
    nested = root / "a-nested"
    nested.mkdir(parents=True)
    nested_file = nested / "payload.bin"
    nested_file.write_bytes(b"nested")
    root_file = root / "z-root.bin"
    root_file.write_bytes(b"root")
    events: list[Path] = []

    def record_fsync(descriptor: int) -> None:
        events.append(Path(os.readlink(f"/proc/self/fd/{descriptor}")))

    monkeypatch.setattr(os, "fsync", record_fsync)
    qualification._durably_sync_verified_tree(root)  # noqa: SLF001

    assert events.count(nested_file) == 1
    assert events.count(root_file) == 1
    assert events.count(nested) == 1
    assert events.count(root) == 1
    assert events.index(nested_file) < events.index(nested) < events.index(root)
    assert events.index(root_file) < events.index(root)


@pytest.mark.parametrize(
    ("case", "limit_name", "message"),
    [
        ("files", "_MAX_QUALIFICATION_FILES", "file bound"),
        ("directories", "_MAX_QUALIFICATION_DIRECTORIES", "directory bound"),
        ("entries", "_MAX_QUALIFICATION_ENTRIES", "entry bound"),
        ("depth", "_MAX_QUALIFICATION_DEPTH", "depth bound"),
        ("bytes", "_MAX_QUALIFICATION_BYTES", "byte bound"),
    ],
)
def test_verified_tree_enforces_global_resource_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    limit_name: str,
    message: str,
) -> None:
    root = tmp_path / case
    root.mkdir()
    if case in {"files", "entries"}:
        (root / "first").write_bytes(b"")
        (root / "second").write_bytes(b"")
    elif case == "directories":
        (root / "first").mkdir()
        (root / "second").mkdir()
    elif case == "depth":
        (root / "first" / "second").mkdir(parents=True)
    else:
        (root / "payload").write_bytes(b"12")
    monkeypatch.setattr(qualification, limit_name, 1)

    with pytest.raises(qualification.ForagerMatchedQualificationError, match=message):
        qualification._durably_sync_verified_tree(root)  # noqa: SLF001


def test_verified_tree_rejects_link_and_detects_replacement_during_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    target = linked_root / "target"
    target.write_bytes(b"target")
    (linked_root / "alias").symlink_to(target.name)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="link or special",
    ):
        qualification._durably_sync_verified_tree(linked_root)  # noqa: SLF001

    race_root = tmp_path / "race"
    race_root.mkdir()
    payload = race_root / "payload"
    payload.write_bytes(b"original")
    real_fsync = os.fsync

    def replace_during_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        if Path(os.readlink(f"/proc/self/fd/{descriptor}")) == payload:
            payload.unlink()
            payload.write_bytes(b"replacement")

    monkeypatch.setattr(os, "fsync", replace_during_fsync)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="changed during traversal",
    ):
        qualification._durably_sync_verified_tree(race_root)  # noqa: SLF001


def test_qualification_permissions_are_fixed_nonroot_oci_readable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "qualification"
    nested = root / "nested"
    nested.mkdir(parents=True)
    plain = nested / "plain.json"
    executable = nested / "worker.py"
    plain.write_bytes(b"{}")
    executable.write_bytes(b"#!/usr/bin/env python\n")
    root.chmod(0o700)
    nested.chmod(0o700)
    plain.chmod(0o600)
    executable.chmod(0o700)

    qualification._normalize_qualification_tree_permissions(root)  # noqa: SLF001

    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert stat.S_IMODE(nested.stat().st_mode) == 0o755
    assert stat.S_IMODE(plain.stat().st_mode) == 0o644
    assert stat.S_IMODE(executable.stat().st_mode) == 0o755


def test_permission_normalization_never_follows_a_swapped_intermediate_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "qualification"
    inside = root / "inside"
    inside.mkdir(parents=True)
    (inside / "payload").write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_payload = outside / "payload"
    outside_payload.write_bytes(b"outside")
    outside_payload.chmod(0o600)
    original_walk = qualification._bounded_tree_walk  # noqa: SLF001
    calls = 0

    def swap_after_first_walk(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        result = original_walk(*args, **kwargs)
        calls += 1
        if calls == 1:
            inside.rename(root / "detached")
            inside.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(qualification, "_bounded_tree_walk", swap_after_first_walk)
    with pytest.raises(qualification.ForagerMatchedQualificationError):
        qualification._normalize_qualification_tree_permissions(root)  # noqa: SLF001

    assert stat.S_IMODE(outside_payload.stat().st_mode) == 0o600


def test_qualification_validates_then_fsyncs_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    output = tmp_path / "qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    _stub_qualification_stages(monkeypatch)
    events: list[tuple[str, Path]] = []

    def load(path: Path) -> Path:
        events.append(("validate", path))
        return path

    def sync(path: Path) -> None:
        events.append(("fsync", path))

    def replay(path: Path, *_args: Any) -> Any:
        events.append(("fresh-replay", path))
        return qualification._FreshReplayClosure(  # noqa: SLF001
            _sha("manifest"),
            _sha("protocol"),
            _sha("plan"),
            _sha("qualification-module"),
        )

    def publish(
        source: Path,
        destination: Path,
        _parent: Path,
        *,
        post_publish_validator: Any,
        **_identity: Any,
    ) -> Any:
        events.append(("publish", destination))
        source.rename(destination)
        return post_publish_validator(destination)

    monkeypatch.setattr(qualification, "load_matched_current_qualification_bundle", load)
    monkeypatch.setattr(
        qualification,
        "_verify_staged_bundle_in_fresh_process",
        replay,
    )
    monkeypatch.setattr(qualification, "_durably_sync_verified_tree", sync)
    monkeypatch.setattr(qualification, "_publish_directory_no_replace", publish)

    result: Any = qualification.qualify_matched_current_candidates(project, upstream, output)
    assert result == output
    assert [event for event, _path in events] == [
        "validate",
        "fresh-replay",
        "fsync",
        "publish",
        "validate",
        "fresh-replay",
    ]
    assert events[0][1] == events[1][1] == events[2][1]
    assert events[3][1] == output
    assert events[4][1] == events[5][1] == output


def test_qualification_fsync_failure_cleans_staging_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    output = tmp_path / "qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    _stub_qualification_stages(monkeypatch)
    temporary_roots: list[Path] = []

    def load(path: Path) -> Path:
        temporary_roots.append(path)
        return path

    def fail_sync(_path: Path) -> None:
        raise qualification.ForagerMatchedQualificationError("injected fsync failure")

    def publish(*_args: Any) -> None:
        raise AssertionError("publication must not run after an fsync failure")

    monkeypatch.setattr(qualification, "load_matched_current_qualification_bundle", load)
    monkeypatch.setattr(qualification, "_durably_sync_verified_tree", fail_sync)
    monkeypatch.setattr(qualification, "_publish_directory_no_replace", publish)

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="injected fsync failure",
    ):
        qualification.qualify_matched_current_candidates(project, upstream, output)
    assert len(temporary_roots) == 1
    assert not temporary_roots[0].exists()
    assert not output.exists()


def test_qualification_fresh_replay_failure_cleans_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    output = tmp_path / "qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    _stub_qualification_stages(monkeypatch)
    staged_roots: list[Path] = []

    def load(path: Path) -> Path:
        staged_roots.append(path)
        return path

    def fail_replay(*_args: Any) -> Any:
        raise qualification.ForagerMatchedQualificationError(
            "injected fresh replay failure"
        )

    def publish(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("publication must not run after fresh replay failure")

    monkeypatch.setattr(qualification, "load_matched_current_qualification_bundle", load)
    monkeypatch.setattr(
        qualification,
        "_verify_staged_bundle_in_fresh_process",
        fail_replay,
    )
    monkeypatch.setattr(qualification, "_publish_directory_no_replace", publish)

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="injected fresh replay failure",
    ):
        qualification.qualify_matched_current_candidates(project, upstream, output)
    assert len(staged_roots) == 1
    assert not staged_roots[0].exists()
    assert not output.exists()


def test_qualification_final_loader_failure_is_published_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    output = tmp_path / "qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    _stub_qualification_stages(monkeypatch)
    calls: list[Path] = []

    def load(path: Path) -> Path:
        calls.append(path)
        if len(calls) == 2:
            raise qualification.ForagerMatchedQualificationError(
                "injected final replay failure"
            )
        return path

    monkeypatch.setattr(qualification, "load_matched_current_qualification_bundle", load)
    with pytest.raises(
        qualification.QualificationPublishedButUncertainError,
        match="do not reuse this output root",
    ):
        qualification.qualify_matched_current_candidates(project, upstream, output)

    assert len(calls) == 2
    assert calls[0].name.startswith(f".{output.name}.partial-")
    assert calls[1] == output
    assert output.is_dir()
    assert (output / "staged.marker").read_bytes() == b"staged"
    assert not list(tmp_path.glob(f".{output.name}.partial-*"))


def test_qualification_post_publish_replay_failure_is_published_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    output = tmp_path / "qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    _stub_qualification_stages(monkeypatch)
    replay_calls = 0

    def replay(*_args: Any) -> Any:
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls == 2:
            raise qualification.ForagerMatchedQualificationError(
                "injected published replay failure"
            )
        return qualification._FreshReplayClosure(  # noqa: SLF001
            _sha("manifest"),
            _sha("protocol"),
            _sha("plan"),
            _sha("qualification-module"),
        )

    monkeypatch.setattr(
        qualification,
        "load_matched_current_qualification_bundle",
        lambda path: path,
    )
    monkeypatch.setattr(
        qualification,
        "_verify_staged_bundle_in_fresh_process",
        replay,
    )

    with pytest.raises(
        qualification.QualificationPublishedButUncertainError,
        match="do not reuse this output root",
    ):
        qualification.qualify_matched_current_candidates(project, upstream, output)
    assert replay_calls == 2
    assert output.is_dir()
    assert (output / "staged.marker").read_bytes() == b"staged"
    assert not list(tmp_path.glob(f".{output.name}.partial-*"))


def test_qualification_post_publish_returned_closure_mismatch_is_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    output = tmp_path / "qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    _stub_qualification_stages(monkeypatch)
    closures = iter(
        (
            qualification._FreshReplayClosure(  # noqa: SLF001
                _sha("manifest"),
                _sha("protocol"),
                _sha("plan"),
                _sha("qualification-module"),
            ),
            qualification._FreshReplayClosure(  # noqa: SLF001
                _sha("manifest"),
                _sha("different-protocol"),
                _sha("plan"),
                _sha("qualification-module"),
            ),
        )
    )
    replay_calls = 0

    def replay(*_args: Any) -> Any:
        nonlocal replay_calls
        replay_calls += 1
        return next(closures)

    monkeypatch.setattr(
        qualification,
        "load_matched_current_qualification_bundle",
        lambda path: path,
    )
    monkeypatch.setattr(
        qualification,
        "_verify_staged_bundle_in_fresh_process",
        replay,
    )

    with pytest.raises(
        qualification.QualificationPublishedButUncertainError,
        match="do not reuse this output root",
    ):
        qualification.qualify_matched_current_candidates(project, upstream, output)
    assert replay_calls == 2
    assert output.is_dir()
    assert (output / "staged.marker").read_bytes() == b"staged"


def test_artifact_tree_rejects_every_unreferenced_file(tmp_path: Path) -> None:
    executor_qualifications, sources, configurations, records = _artifact_tree_inputs(tmp_path)
    qualification._verify_qualification_artifact_tree(  # noqa: SLF001
        tmp_path,
        executor_qualifications,
        sources,
        configurations,
        records,
    )
    (tmp_path / "unexpected.npz").write_bytes(b"")
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="unreferenced",
    ):
        qualification._verify_qualification_artifact_tree(  # noqa: SLF001
            tmp_path,
            executor_qualifications,
            sources,
            configurations,
            records,
        )


def test_artifact_tree_rejects_before_unbounded_non_source_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_qualifications, sources, configurations, records = _artifact_tree_inputs(tmp_path)
    monkeypatch.setattr(qualification, "_MAX_QUALIFICATION_ARTIFACT_ENTRIES", 2)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="entry bound",
    ):
        qualification._verify_qualification_artifact_tree(  # noqa: SLF001
            tmp_path,
            executor_qualifications,
            sources,
            configurations,
            records,
        )


def test_cli_converts_protocol_validator_failure_to_clean_exit_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: Any) -> int:
        raise builder.ForagerMatchedOpenProtocolBuildError("frozen protocol drift")

    monkeypatch.setattr(qualification, "_cli", fail)
    assert qualification.main(("verify",)) == 2
    assert "frozen protocol drift" in capsys.readouterr().err


def test_cli_reports_published_uncertain_as_exit_three(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: Any) -> int:
        raise qualification.QualificationPublishedButUncertainError(
            "qualification was published; do not reuse this output root"
        )

    monkeypatch.setattr(qualification, "_cli", fail)
    assert qualification.main(("qualify",)) == 3
    error = capsys.readouterr().err
    assert "PUBLISHED-UNCERTAIN" in error
    assert "do not reuse this output root" in error


def test_copy_tree_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    os.symlink(target.name, source / "linked.py")
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="link or special",
    ):
        qualification._copy_tree(source, tmp_path / "staged", alberta_filter=True)  # noqa: SLF001


def test_alberta_snapshot_copy_rejects_concurrent_live_tree_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "alberta_framework"
    source.mkdir()
    module = source / "module.py"
    module.write_bytes(b"VALUE = 1\n")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"ignored")

    stable = tmp_path / "stable"
    qualification._copy_alberta_tree_stably(source, stable)  # noqa: SLF001
    assert (stable / "module.py").read_bytes() == b"VALUE = 1\n"
    assert not (stable / "__pycache__").exists()

    real_copy = qualification._copy_tree  # noqa: SLF001

    def copy_then_mutate(
        source_root: Path,
        destination: Path,
        *,
        alberta_filter: bool,
    ) -> None:
        real_copy(source_root, destination, alberta_filter=alberta_filter)
        module.write_bytes(b"VALUE = 2\n")

    monkeypatch.setattr(qualification, "_copy_tree", copy_then_mutate)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="live Alberta source changed during snapshot staging",
    ):
        qualification._copy_alberta_tree_stably(source, tmp_path / "raced")  # noqa: SLF001


def test_staged_source_reverification_covers_transitive_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    entrypoint = source / "entrypoint.py"
    entrypoint.write_bytes(b"import helper\n")
    helper = source / "helper.py"
    helper.write_bytes(b"VALUE = 1\n")
    inventory = executor.source_inventory(source)
    normalized = executor.source_inventory_sha256(source)
    binding = replace(_source_bindings()["alberta"], inventory_sha256=normalized)
    staged = qualification._StagedSource(  # noqa: SLF001
        key="alberta",
        root=source,
        archive=tmp_path / "source.tar",
        inventory_path=tmp_path / "inventory.json",
        inventory=inventory,
        binding=binding,
        descriptor_path=None,
        patch_path=None,
    )

    qualification._reverify_staged_source(staged)  # noqa: SLF001
    helper.write_bytes(b"VALUE = 2\n")
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="staged source changed",
    ):
        qualification._reverify_staged_source(staged)  # noqa: SLF001


def test_fresh_snapshot_replay_accepts_only_the_exact_child_closure(
    tmp_path: Path,
) -> None:
    qualification_root = tmp_path / "qualification"
    source_root = qualification_root / "sources" / "alberta" / "source"
    package = source_root / "alberta_framework" / "benchmarks"
    package.mkdir(parents=True)
    (source_root / "alberta_framework" / "__init__.py").write_bytes(b"")
    (package / "__init__.py").write_bytes(b"")
    manifest_sha256 = _sha("manifest")
    protocol_sha256 = _sha("protocol")
    plan_sha256 = _sha("plan")
    module = package / "forager_matched_qualification.py"
    module.write_text(
        "\n".join(
            (
                "class Bundle:",
                f"    manifest_sha256 = {manifest_sha256!r}",
                "",
                "class Protocol:",
                f"    protocol_sha256 = {protocol_sha256!r}",
                "",
                "class Plan:",
                f"    plan_sha256 = {plan_sha256!r}",
                f"    qualification_manifest_sha256 = {manifest_sha256!r}",
                "",
                "def load_matched_current_qualification_bundle(_root):",
                "    return Bundle()",
                "",
                "def build_open_protocol_and_execution_plan(_bundle):",
                "    return Protocol(), Plan()",
                "",
            )
        ),
        encoding="utf-8",
    )
    qualification._normalize_qualification_tree_permissions(  # noqa: SLF001
        qualification_root
    )
    module_sha256 = hashlib.sha256(module.read_bytes()).hexdigest()
    runtime = qualification._ProbeRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "docker",
        executable_sha256=_sha("runtime"),
        version={},
        image_inspection={},
    )
    payload = {
        "schema_version": qualification._FRESH_SNAPSHOT_REPLAY_SCHEMA,  # noqa: SLF001
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "plan_sha256": plan_sha256,
        "plan_qualification_manifest_sha256": manifest_sha256,
        "qualification_module_path": (
            "alberta_framework/benchmarks/forager_matched_qualification.py"
        ),
        "qualification_module_sha256": module_sha256,
    }
    commands: list[tuple[str, ...]] = []

    def runner(command: Any) -> qualification.QualificationProcessResult:
        commands.append(tuple(command))
        return qualification.QualificationProcessResult(
            0,
            qualification._canonical_json_bytes(payload),  # noqa: SLF001
            b"",
        )

    qualification._run_fresh_snapshot_replay(  # noqa: SLF001
        qualification_root,
        runtime,
        runner,
        expected_manifest_sha256=manifest_sha256,
        expected_protocol_sha256=protocol_sha256,
        expected_plan_sha256=plan_sha256,
        expected_qualification_module_sha256=module_sha256,
    )
    assert len(commands) == 1
    assert "--network=none" in commands[0]
    assert f"sha256:{qualification._QUALIFIED_IMAGE_SHA256}" in commands[0]  # noqa: SLF001
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="fresh-process staged replay differs",
    ):
        qualification._run_fresh_snapshot_replay(  # noqa: SLF001
            qualification_root,
            runtime,
            runner,
            expected_manifest_sha256=manifest_sha256,
            expected_protocol_sha256=protocol_sha256,
            expected_plan_sha256=_sha("wrong-plan"),
            expected_qualification_module_sha256=module_sha256,
        )
