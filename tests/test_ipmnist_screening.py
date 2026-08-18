"""Unit tests for the IPMNIST mechanism-combination screening lane.

These pin the screening harness to the full-horizon lane (control parity and
prefix property), pin combination steps to their reference equations, and
test shard/merge/validation plumbing. Benchmark executions never run here.
"""

import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.benchmarks.ipmnist_screening as ipmnist_screening
from alberta_framework.benchmarks.ipmnist_screening import (
    _CBP_LAYERS,
    CONFIRMATION_THRESHOLD,
    LEGACY_SHARD_SCHEMA,
    NONPROMOTING_POLICY,
    PROXY_N_TASKS,
    SCREENING_REGISTRY,
    SHARD_SCHEMA,
    AdamCBPState,
    CBPState,
    EMANormState,
    ScreeningSpec,
    UPGDAdaptiveNormState,
    _atomic_write_json,
    _ema_frozen_probe_input,
    _hidden_rms_normalize,
    _make_adamw_cbp_ema_norm_learner,
    _make_adamw_cbp_learner,
    _make_adamw_cbp_noreset_learner,
    _make_colnorm_gate_learner,
    _make_fade_head_ema_norm_learner,
    _make_guarded_cbp_adam_learner,
    _make_l2init_ema_norm_learner,
    _make_lion_gate_learner,
    _make_muon_gate_learner,
    _make_naive_bayes_learner,
    _make_norm_adam_fastv_learner,
    _make_norm_apollo_gate_learner,
    _make_norm_rmsprop_gate_learner,
    _make_rff_rls_learner,
    _make_sgd_ema_norm_learner,
    _make_sgd_momentum_gate_learner,
    _make_sigma0_gated_l2init_learner,
    _make_snr_ema_norm_learner,
    _make_upgd_alpha_utility_learner,
    _make_upgd_autostep_learner,
    _make_upgd_ema_norm_ext_learner,
    _make_upgd_ema_norm_learner,
    _make_upgd_idbd_learner,
    _make_upgd_shiftnorm_learner,
    _make_upgd_w_fade_head_learner,
    _make_upgd_w_wclip_learner,
    _make_upgd_warmnorm_learner,
    _make_wclip_ema_norm_learner,
    _newton_schulz_orthogonalize,
    _rff_frozen_probe_input,
    _upgd_utility_and_gate,
    adam_elem_step,
    adam_elem_update,
    cbp_maybe_replace_layer,
    ema_normalize,
    guarded_adam_update,
    load_shard,
    main,
    merge_shards,
    naive_bayes_logits,
    run_screening_config,
    screening_spec,
    shard_payload,
    shift_adaptive_normalize,
    snr_maybe_reset_layer,
    upgd_alpha_utility_update,
    upgd_autostep_update,
    upgd_idbd_swift_update,
    upgd_idbd_update,
    upgd_l2init_update,
    upgd_w_fade_head_update,
    upgd_w_localgate_update,
    upgd_w_wclip_update,
    validate_proxy,
    warm_restart_normalize,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    LeanUPGDState,
    cross_entropy_loss,
    init_mlp_params,
    lean_upgd_w_update,
    run_ipmnist,
)
from alberta_framework.core.baseline_optimizers import Adam
from alberta_framework.core.normalizers import EMANormalizer

SMALL = IPMNISTConfig(
    n_tasks=3, task_length=30, input_dim=12, hidden1=8, hidden2=6, n_classes=5
)


def _test_source_provenance(
    *, source_sha256: str = "3" * 64,
) -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.source_provenance.v1",
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "git_object_format": "sha1",
        "relevant_source_scope": "tracked:alberta_framework/**,pyproject.toml,uv.lock",
        "relevant_source_file_count": 3,
        "relevant_source_sha256": source_sha256,
        "uv_lock_sha256": "4" * 64,
        "worktree_clean": True,
    }


def _test_runtime_environment(
    *, machine: str = "test-machine",
) -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.runtime.v1",
        "python": {"implementation": "CPython", "version": "3.12.12"},
        "platform": {"system": "TestOS", "release": "1", "machine": machine},
        "packages": {
            "chex": "0.1.91",
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
            "numpy": "2.5.1",
            "scikit-learn": "1.7.1",
        },
        "jax": {
            "backend": "cpu",
            "devices": [
                {"id": 0, "platform": "cpu", "device_kind": "test-cpu", "process_index": 0}
            ],
            "config": {
                "jax_enable_x64": False,
                "jax_default_matmul_precision": None,
                "jax_disable_jit": False,
                "jax_numpy_dtype_promotion": "standard",
                "jax_numpy_rank_promotion": "allow",
                "jax_random_seed_offset": 0,
                "jax_threefry_partitionable": True,
                "jax_default_prng_impl": "threefry2x32",
            },
        },
        "process_environment": {
            "CUDA_VISIBLE_DEVICES": None,
            "JAX_DEFAULT_MATMUL_PRECISION": None,
            "JAX_ENABLE_X64": None,
            "JAX_PLATFORM_NAME": None,
            "JAX_PLATFORMS": None,
            "OMP_NUM_THREADS": "1",
            "XLA_FLAGS": None,
        },
    }


def _test_dataset_provenance() -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
        "source": {
            "provider": "openml",
            "name": "mnist_784",
            "version": 1,
            "row_start": 0,
            "row_stop_exclusive": 60000,
        },
        "materialization": "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1",
        "x": {"dtype": "<f4", "shape": [60000, 784], "sha256": "5" * 64},
        "y": {"dtype": "<i4", "shape": [60000], "sha256": "6" * 64},
    }


def _bound_shard_payload(
    result: ipmnist_screening.ScreeningRunResult,
) -> dict[str, object]:
    artifact_config = IPMNISTConfig(
        n_tasks=result.config.n_tasks,
        task_length=result.config.task_length,
        input_dim=784,
        hidden1=result.config.hidden1,
        hidden2=result.config.hidden2,
        n_classes=10,
    )
    return shard_payload(
        replace(result, config=artifact_config),
        source_provenance=_test_source_provenance(),
        dataset_provenance=_test_dataset_provenance(),
        environment=_test_runtime_environment(),
    )


@pytest.fixture(scope="module")
def small_data():
    key = jr.key(1234)
    kx, ky = jr.split(key)
    x = jr.uniform(kx, (64, SMALL.input_dim), jnp.float32, -1.0, 1.0)
    y = jr.randint(ky, (64,), 0, SMALL.n_classes)
    return np.asarray(x), np.asarray(y)


class TestRegistry:
    def test_expected_configs_present(self):
        expected = {
            "disc_r1",
            "disc_r2",
            "disc_r3",
            "disc_r1_pscale",
            "disc_r1_pscale_norms",
            "upgd_w_control",
            "adamw_control",
            "upgd_idbd",
            "upgd_idbd_meta1e2",
            "upgd_autostep",
            "upgd_l2init",
            "upgd_ema_norm",
            "upgd_cbp",
            "adamw_cbp",
            "upgd_w_sigma005",
            "upgd_w_sigma02",
            "upgd_w_udecay0999",
            "upgd_w_udecay099999",
            "upgd_w_wd0005",
            "upgd_w_wd002",
            "upgd_w_wclip_k1",
            "upgd_w_wclip_k2",
            "upgd_w_wclip_k1_wd0",
            "upgd_w_wclip_k2_wd0",
            "upgd_w_localgate",
            "upgd_w_fade_head",
            "upgd_w_idbd_swift",
            "guarded_cbp_adam",
            "adamw_cbp_noreset",
            "adamw_cbp_ema_norm",
            "upgd_w_sigma0",
            "upgd_alpha_utility",
            "adamw_cbp_r3e5",
            "adamw_cbp_r3e4",
            "adamw_cbp_m50",
            "adamw_cbp_m200",
            "upgd_ema_norm_wd0005",
            "upgd_ema_norm_lr003",
            "upgd_ema_norm_lr0003",
            "upgd_ema_norm_sigma0",
            "sgd_ema_norm",
            "sigma0_ndecay099",
            "sigma0_ndecay09999",
            "sigma0_ndecay09",
            "sigma0_ndecay095",
            "sigma0_ndecay098",
            "ema_norm_ndecay099",
            "sigma0_eps1e6",
            "sigma0_eps1e4",
            "sigma0_hidden_norm",
            "sigma0_gate_beta05",
            "sigma0_gate_beta2",
            "sigma0_localgate",
            "sigma0_shiftnorm",
            "sigma0_shiftnorm_k05",
            "sigma0_shiftnorm_d099",
            "sigma0_shiftnorm_d099_k05",
            "sigma0_shiftnorm_d099_k2",
            "sigma0_shiftnorm_d098",
            "sigma0_shiftnorm_d099_f08",
            "sigma0_shiftnorm_d099_f095",
            "sigma0_shiftnorm_d099_r200",
            "sigma0_warmnorm",
            "sigma0_gateplus",
            "colnorm_gate",
            "muon_gate",
            "lion_gate",
            "rff_rls",
            "lin_rls",
            "sgd_ema_norm_d099",
            "wclip_ema_norm",
            "fade_head_ema_norm",
            "snr_ema_norm",
            "l2init_ema_norm",
            "norm_adam_fastv",
            "norm_adam_fastv_b2099",
            "norm_adam_gate",
            "norm_rmsprop_gate",
            "norm_apollo_gate",
            "sgd_momentum_gate",
            "sgd_momentum_gate_m099",
            "naive_bayes",
            "nb_ensemble_champion",
            "nb_ensemble_nbreset",
            "nb_ensemble_rls3",
            "rls_head_l0999",
            "rls_head_l0995",
            "rls_head_l1",
            "rls_head_l0999_preset005",
            "rls_head_resid",
            "rls_head_l1_preset005",
            "rls_head_l1_preset003",
            "rls_head_l0999_pcap",
            "rls_head_resid_l1_preset005",
            "rls_head_resid_l1_noreset",
            "rls_head_resid_l1_preset005_l2init",
            "rls_head_resid_l1_preset005_nogate",
            "rls_head_l0999_preset005_r01",
            "rls_head_l0999_preset005_r003",
            "rls_head_l0999_preset005_r001",
            "rls_head_resid_preset005_r01",
            "rls_head_resid_preset005_r001",
        }
        # Concurrent waves may register additional arms; this set must at
        # least be present (exact-set equality would race sibling lanes).
        assert expected <= set(SCREENING_REGISTRY)

    def test_unknown_config_rejected(self):
        with pytest.raises(ValueError, match="unknown screening config"):
            screening_spec("nope")

    def test_control_uses_published_hyperparameters(self):
        assert (
            screening_spec("upgd_w_control").hyperparameters
            == UPGD_W_PROTOCOL_HYPERPARAMETERS
        )

    def test_proxy_default_horizon(self):
        assert PROXY_N_TASKS == 60


class TestControlParity:
    """The screening runner must reproduce the full lane for control arms."""

    @pytest.mark.parametrize("name,learner", [
        ("upgd_w_control", "upgd_w"),
        ("adamw_control", "adamw"),
    ])
    def test_control_matches_run_ipmnist(self, small_data, name, learner):
        x, y = small_data
        ours = run_screening_config(x, y, screening_spec(name), seed=7, config=SMALL)
        reference = run_ipmnist(x, y, learner, seeds=[7], config=SMALL)
        np.testing.assert_allclose(
            ours.per_task_accuracy, reference.per_task_accuracy[0], atol=1e-7
        )
        np.testing.assert_allclose(
            ours.per_task_loss, reference.per_task_loss[0], rtol=1e-5
        )

    def test_prefix_property(self, small_data):
        """A shorter-horizon run is an exact prefix of a longer one (same seed)."""
        x, y = small_data
        spec = screening_spec("upgd_w_control")
        short = run_screening_config(
            x, y, spec, seed=3, config=IPMNISTConfig(
                n_tasks=2, task_length=30, input_dim=12, hidden1=8, hidden2=6, n_classes=5
            )
        )
        long = run_screening_config(x, y, spec, seed=3, config=SMALL)
        np.testing.assert_allclose(
            short.per_task_accuracy, long.per_task_accuracy[:2], atol=1e-7
        )


@pytest.mark.unit
class TestScreeningInputDomain:
    """Issue #527: run_screening_config must refuse out-of-domain data before the factory."""

    @staticmethod
    def _boom_factory(hyperparameters):
        del hyperparameters
        raise AssertionError("out-of-domain data reached the learner factory")

    @pytest.mark.parametrize("noise_mode", ["step", "pool"])
    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda x, y: (x, y + 100), "must be smaller than"),
            (lambda x, y: (x, y - 1), "non-negative"),
            (lambda x, y: (x, y.astype(np.float32) + 0.9), "integer class labels"),
            (lambda x, y: (x.at[0, 0].set(np.inf), y), "finite"),
            (lambda x, y: (x.at[3, 1].set(np.nan), y), "finite"),
            (
                lambda x, y: (np.full(np.shape(x), np.timedelta64("NaT", "s")), y),
                "real numeric",
            ),
            (
                lambda x, y: (
                    x[: SMALL.task_length - 1],
                    y[: SMALL.task_length - 1],
                ),
                "task_length",
            ),
        ],
    )
    def test_rejects_before_learner_factory(
        self, small_data, noise_mode: str, mutate, message: str
    ) -> None:
        x, y = small_data
        x, y = mutate(jnp.asarray(x), jnp.asarray(y))
        spec = replace(screening_spec("upgd_w_control"), factory=self._boom_factory)
        with pytest.raises(ValueError, match=message):
            run_screening_config(
                x, y, spec, seed=0, config=SMALL, noise_mode=noise_mode
            )

    @pytest.mark.parametrize("progress_every", [0, -1, True, 1.5])
    def test_rejects_invalid_progress_interval_before_learner_factory(
        self, small_data, progress_every: object
    ) -> None:
        """Host-boundary progress identities must not reach the learner factory."""
        x, y = small_data
        spec = replace(screening_spec("upgd_w_control"), factory=self._boom_factory)
        with pytest.raises(ValueError, match="progress_every"):
            run_screening_config(
                x,
                y,
                spec,
                seed=0,
                config=SMALL,
                progress_every=progress_every,  # type: ignore[arg-type]
            )


class TestIDBDCombo:
    def test_meta_zero_reduces_to_lean_upgd(self):
        """With meta=0 and initial alpha = published lr, IDBD == lean UPGD-W."""
        key = jr.key(0)
        params = init_mlp_params(key, SMALL)
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 0.0, "initial_step_size": hp["step_size"]})
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state = init_fn(params)
        lean_state = LeanUPGDState(utility=state.utility, step=state.step)
        kg, kn = jr.split(jr.key(9))
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                 for i, (n, v) in enumerate(sorted(params.items()))}
        noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                 for i, (n, v) in enumerate(sorted(params.items()))}
        for _ in range(3):
            p_idbd, state = upgd_idbd_update(params, state, grads, noise, hp)
            p_lean, lean_state = lean_upgd_w_update(params, lean_state, grads, noise, hp)
            for n in params:
                np.testing.assert_allclose(p_idbd[n], p_lean[n], atol=1e-7)
            params = p_idbd

    def test_log_alpha_stays_within_bounds(self):
        key = jr.key(0)
        params = init_mlp_params(key, SMALL)
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 10.0, "initial_step_size": 0.01})
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state = init_fn(params)
        grads = {n: jnp.ones_like(v) for n, v in params.items()}
        noise = {n: jnp.zeros_like(v) for n, v in params.items()}
        for _ in range(5):
            params, state = upgd_idbd_update(params, state, grads, noise, hp)
        for n in params:
            assert bool(jnp.all(state.log_alpha[n] >= -10.0))
            assert bool(jnp.all(state.log_alpha[n] <= 0.0))
            assert bool(jnp.all(jnp.isfinite(params[n])))

    def test_nonfinite_gated_gradient_does_not_poison_log_alpha(self):
        params = init_mlp_params(jr.key(0), SMALL)
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 0.01, "initial_step_size": 0.01})
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state = init_fn(params)
        grads = {name: jnp.zeros_like(value) for name, value in params.items()}
        grads["w1"] = jnp.full_like(params["w1"], jnp.inf)
        noise = {name: jnp.zeros_like(value) for name, value in params.items()}

        _, guarded = upgd_idbd_update(params, state, grads, noise, hp)

        np.testing.assert_array_equal(guarded.log_alpha["w1"], state.log_alpha["w1"])
        assert bool(jnp.all(jnp.isfinite(guarded.log_alpha["w1"])))


class TestAutostepCombo:
    def test_nonfinite_gated_gradient_does_not_poison_meta_state(self):
        params = init_mlp_params(jr.key(0), SMALL)
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 0.01, "initial_step_size": 0.01, "tau": 1e4})
        init_fn, _ = _make_upgd_autostep_learner(hp)
        state = init_fn(params)
        grads = {name: jnp.zeros_like(value) for name, value in params.items()}
        grads["w1"] = jnp.full_like(params["w1"], jnp.inf)
        noise = {name: jnp.zeros_like(value) for name, value in params.items()}

        _, guarded = upgd_autostep_update(params, state, grads, noise, hp)

        np.testing.assert_array_equal(guarded.normalizer["w1"], state.normalizer["w1"])
        np.testing.assert_array_equal(guarded.alpha["w1"], state.alpha["w1"])
        np.testing.assert_array_equal(guarded.trace["w1"], state.trace["w1"])
        assert bool(jnp.all(jnp.isfinite(guarded.normalizer["w1"])))
        assert bool(jnp.all(jnp.isfinite(guarded.alpha["w1"])))

    def test_finite_square_overflow_keeps_state_finite_and_triggers_bound(self):
        params = init_mlp_params(jr.key(0), SMALL)
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 0.01, "initial_step_size": 0.01, "tau": 1e4})
        init_fn, _ = _make_upgd_autostep_learner(hp)
        state = init_fn(params)
        grads = {name: jnp.zeros_like(value) for name, value in params.items()}
        grads["w1"] = jnp.full_like(params["w1"], 1e30)
        noise = {name: jnp.zeros_like(value) for name, value in params.items()}

        _, guarded = upgd_autostep_update(params, state, grads, noise, hp)

        assert bool(jnp.all(jnp.isfinite(guarded.normalizer["w1"])))
        assert bool(jnp.all(jnp.isfinite(guarded.trace["w1"])))
        np.testing.assert_array_equal(guarded.normalizer["w1"], state.normalizer["w1"])
        np.testing.assert_array_equal(
            guarded.alpha["w1"], jnp.full_like(state.alpha["w1"], 1e-8)
        )


class TestFadeHead:
    """FADE meta-learned per-parameter weight decay on the output layer."""

    def _fade_hp(self, **overrides):
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"fade_alpha": 0.005, "fade_gamma0": -6.9, "fade_theta_lambda": 0.1})
        hp.update(overrides)
        return hp

    def _random_inputs(self, params, hp, seed):
        kg, kn = jr.split(jr.key(seed))
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                 for i, (n, v) in enumerate(sorted(params.items()))}
        noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                 for i, (n, v) in enumerate(sorted(params.items()))}
        return grads, noise

    def test_lambda_zero_theta_zero_reduces_to_control_on_head(self):
        """theta_lambda=0, gamma_0=-inf (lambda=0): head == control with zero
        head decay, hidden layers == published control, bit-exact."""
        key = jr.key(0)
        params = init_mlp_params(key, SMALL)
        hp = self._fade_hp(fade_gamma0=-math.inf, fade_theta_lambda=0.0)
        init_fn, _ = _make_upgd_w_fade_head_learner(hp)
        state = init_fn(params)
        grads, noise = self._random_inputs(params, hp, seed=21)
        head_x = jnp.abs(jr.normal(jr.key(33), (SMALL.hidden2,), jnp.float32))
        lean_state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        p_fade, _ = upgd_w_fade_head_update(params, state, grads, noise, head_x, hp)
        p_wd, _ = lean_upgd_w_update(params, lean_state, grads, noise, hp)
        hp_wd0 = dict(hp)
        hp_wd0["weight_decay"] = 0.0
        p_wd0, _ = lean_upgd_w_update(params, lean_state, grads, noise, hp_wd0)
        for n in ("w3", "b3"):
            np.testing.assert_array_equal(np.asarray(p_fade[n]), np.asarray(p_wd0[n]))
        for n in ("w1", "b1", "w2", "b2"):
            np.testing.assert_array_equal(np.asarray(p_fade[n]), np.asarray(p_wd[n]))

        # Multi-step: with weight_decay=0 everywhere and lambda=0 the whole
        # trajectory reduces bit-exactly to the lean UPGD-W trajectory.
        params = init_mlp_params(key, SMALL)
        state = init_fn(params)
        lean_state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        for step in range(3):
            grads, noise = self._random_inputs(params, hp_wd0, seed=100 + step)
            p_fade, state = upgd_w_fade_head_update(
                params, state, grads, noise, head_x, hp_wd0
            )
            p_lean, lean_state = lean_upgd_w_update(
                params, lean_state, grads, noise, hp_wd0
            )
            for n in params:
                np.testing.assert_array_equal(np.asarray(p_fade[n]), np.asarray(p_lean[n]))
            params = p_fade

    def test_frozen_gamma_at_control_decay_matches_published_control(self):
        """theta_lambda=0, lambda_0 = step_size*weight_decay: one step matches
        the published control on every layer (head decay factors coincide)."""
        key = jr.key(1)
        params = init_mlp_params(key, SMALL)
        base = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        gamma0 = math.log(base["step_size"] * base["weight_decay"])
        hp = self._fade_hp(fade_gamma0=gamma0, fade_theta_lambda=0.0)
        init_fn, _ = _make_upgd_w_fade_head_learner(hp)
        state = init_fn(params)
        lean_state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        grads, noise = self._random_inputs(params, hp, seed=55)
        head_x = jnp.abs(jr.normal(jr.key(56), (SMALL.hidden2,), jnp.float32))
        p_fade, _ = upgd_w_fade_head_update(params, state, grads, noise, head_x, hp)
        p_lean, _ = lean_upgd_w_update(params, lean_state, grads, noise, hp)
        for n in params:
            np.testing.assert_allclose(
                np.asarray(p_fade[n]), np.asarray(p_lean[n]), atol=1e-7
            )

    def test_lambda_stays_finite_positive_over_random_steps(self):
        """200 random steps at published hyperparameters: gamma stays finite
        and capped, lambda = exp(gamma) stays in (0, 1], trace stays finite."""
        key = jr.key(2)
        params = init_mlp_params(key, SMALL)
        hp = self._fade_hp()
        init_fn, _ = _make_upgd_w_fade_head_learner(hp)
        state = init_fn(params)
        for step in range(200):
            grads, noise = self._random_inputs(params, hp, seed=1000 + step)
            head_x = jnp.abs(jr.normal(jr.fold_in(jr.key(3), step), (SMALL.hidden2,)))
            params, state = upgd_w_fade_head_update(
                params, state, grads, noise, head_x, hp
            )
        for n in ("w3", "b3"):
            gamma = np.asarray(state.gamma[n])
            lam = np.exp(gamma)
            assert np.all(np.isfinite(gamma)), n
            assert np.all(gamma <= 0.0), n
            assert np.all(lam > 0.0), n
            assert np.all(lam <= 1.0), n
            assert np.all(np.isfinite(np.asarray(state.fade_trace[n]))), n
            assert np.all(np.isfinite(np.asarray(params[n]))), n

    def test_gamma_increases_when_decay_helps(self):
        """Stale positive head weight the new task's gradient keeps pushing
        toward zero (decay helps): gamma on w3 must rise above gamma_0."""
        key = jr.key(4)
        params = init_mlp_params(key, SMALL)
        params = dict(params)
        params["w3"] = jnp.full_like(params["w3"], 2.0)  # stale, far from init
        hp = self._fade_hp()
        init_fn, _ = _make_upgd_w_fade_head_learner(hp)
        state = init_fn(params)
        kg = jr.key(8)
        head_x = jnp.ones((SMALL.hidden2,), jnp.float32)
        zeros = {n: jnp.zeros_like(v) for n, v in params.items()}
        for step in range(100):
            # Persistent error: descent wants the stale positive w3 to shrink
            # (positive gradient); small random grads elsewhere keep the
            # utility gate's global max well-defined.
            grads = {n: jr.normal(jr.fold_in(kg, i * 1000 + step), v.shape) * 0.1
                     for i, (n, v) in enumerate(sorted(params.items()))}
            grads["w3"] = jnp.full_like(params["w3"], 0.5)
            grads["b3"] = jnp.zeros_like(params["b3"])
            params, state = upgd_w_fade_head_update(
                params, state, grads, zeros, head_x, hp
            )
        assert bool(jnp.all(state.gamma["w3"] > hp["fade_gamma0"]))
        assert bool(jnp.all(jnp.isfinite(state.gamma["w3"])))


class TestIDBDSwift:
    """UPGD+IDBD with SwiftTD's supervised-mode stabilizers."""

    def _hp(self, **overrides):
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 1e-3, "initial_step_size": 0.01,
                   "swift_eta": 0.1, "swift_eps": 0.99})
        hp.update(overrides)
        return hp

    def test_eta_inf_eps_one_reduces_to_upgd_idbd(self):
        """swift_eta=inf, swift_eps=1: bit-exact upgd_idbd trajectory."""
        key = jr.key(0)
        params = init_mlp_params(key, SMALL)
        hp = self._hp(swift_eta=math.inf, swift_eps=1.0)
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state_swift = init_fn(params)
        state_plain = init_fn(params)
        kg, kn = jr.split(jr.key(9))
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                 for i, (n, v) in enumerate(sorted(params.items()))}
        noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                 for i, (n, v) in enumerate(sorted(params.items()))}
        for _ in range(3):
            p_swift, state_swift = upgd_idbd_swift_update(
                params, state_swift, grads, noise, hp
            )
            p_plain, state_plain = upgd_idbd_update(params, state_plain, grads, noise, hp)
            for n in params:
                np.testing.assert_array_equal(np.asarray(p_swift[n]), np.asarray(p_plain[n]))
                np.testing.assert_array_equal(
                    np.asarray(state_swift.log_alpha[n]), np.asarray(state_plain.log_alpha[n])
                )
                np.testing.assert_array_equal(
                    np.asarray(state_swift.trace[n]), np.asarray(state_plain.trace[n])
                )
            params = p_swift

    def test_nonfinite_gated_gradient_does_not_poison_log_alpha(self):
        params = init_mlp_params(jr.key(0), SMALL)
        hp = self._hp()
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state = init_fn(params)
        grads = {name: jnp.zeros_like(value) for name, value in params.items()}
        grads["w1"] = jnp.full_like(params["w1"], jnp.inf)
        noise = {name: jnp.zeros_like(value) for name, value in params.items()}

        _, guarded = upgd_idbd_swift_update(params, state, grads, noise, hp)

        np.testing.assert_array_equal(guarded.log_alpha["w1"], state.log_alpha["w1"])
        assert bool(jnp.all(jnp.isfinite(guarded.log_alpha["w1"])))

    def test_overshoot_bound_triggers_and_caps_effective_step(self):
        """Large alpha: sum_i alpha_i z_i^2 >> eta, so the applied step is the
        unbounded step scaled by eta/tau, the effective correction ratio is
        capped at eta, step-sizes decay persistently, and traces reset."""
        key = jr.key(2)
        params = init_mlp_params(key, SMALL)
        hp = self._hp(
            weight_decay=0.0, meta_step_size=0.0,
            initial_step_size=1.0, swift_eta=1.0,
        )
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state_swift = init_fn(params)
        state_plain = init_fn(params)
        kg = jr.key(7)
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape)
                 for i, (n, v) in enumerate(sorted(params.items()))}
        zeros = {n: jnp.zeros_like(v) for n, v in params.items()}
        p_swift, s_swift = upgd_idbd_swift_update(params, state_swift, grads, zeros, hp)
        p_plain, s_plain = upgd_idbd_update(params, state_plain, grads, zeros, hp)
        alpha = hp["initial_step_size"]  # meta=0 keeps alpha frozen at init
        delta_plain = {n: np.asarray(params[n] - p_plain[n]) for n in params}  # alpha*z
        delta_swift = {n: np.asarray(params[n] - p_swift[n]) for n in params}
        tau = sum(float(np.sum(d * d)) for d in delta_plain.values()) / alpha
        eta = hp["swift_eta"]
        assert tau > eta  # the bound triggers on this constructed case
        scale = eta / tau
        assert scale < 1.0
        for n in params:
            np.testing.assert_allclose(
                delta_swift[n], scale * delta_plain[n], rtol=1e-4, atol=1e-7
            )
        # Effective correction ratio sum_i alpha_eff_i z_i^2 is capped at eta.
        tau_eff = sum(
            float(np.sum(delta_swift[n] * delta_plain[n])) for n in params
        ) / alpha
        assert tau_eff == pytest.approx(eta, rel=1e-3)
        # Persistent decay: beta_i += z_i^2 * ln(eps) (plain arm's log stays
        # frozen at ln(initial_step_size) because meta=0).
        for n in params:
            z_sq = (delta_plain[n] / alpha) ** 2
            np.testing.assert_allclose(
                np.asarray(s_swift.log_alpha[n]),
                np.asarray(s_plain.log_alpha[n]) + math.log(hp["swift_eps"]) * z_sq,
                rtol=1e-5, atol=1e-6,
            )
        # The trigger also resets the meta traces (SwiftTD's decay block).
        for n in params:
            np.testing.assert_array_equal(np.asarray(s_swift.trace[n]), 0.0)
        assert any(float(np.max(np.abs(np.asarray(s_plain.trace[n])))) > 0.0
                   for n in params)


class TestL2Init:
    def test_zero_grads_pull_toward_init_only(self):
        key = jr.key(4)
        params = init_mlp_params(key, SMALL)
        w0 = {n: v + 1.0 for n, v in params.items()}  # pretend init is elsewhere
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        from alberta_framework.benchmarks.ipmnist_screening import UPGDL2InitState

        # Nonzero utility keeps the global-max gate well-defined under zero
        # grads (the lean UPGD equations divide by the global utility max).
        state = UPGDL2InitState(
            utility={n: jnp.ones_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            init_params=w0,
        )
        grads = {n: jnp.zeros_like(v) for n, v in params.items()}
        noise = {n: jnp.zeros_like(v) for n, v in params.items()}
        new_params, _ = upgd_l2init_update(params, state, grads, noise, hp)
        lam = hp["step_size"] * hp["weight_decay"]
        for n in params:
            expected = params[n] - lam * (params[n] - w0[n])
            np.testing.assert_allclose(new_params[n], expected, atol=1e-7)


class TestEMANorm:
    def test_decay_one_preserves_initial_moment_pseudo_sample(self):
        """The scan restatement includes the initialized moments as one sample."""
        state = EMANormState(
            mean=jnp.zeros(1),
            var=jnp.ones(1),
            count=jnp.array(0.0),
        )

        _, state = ema_normalize(
            state,
            jnp.asarray([5.0], dtype=jnp.float32),
            1.0,
            1e-8,
        )
        np.testing.assert_array_equal(np.asarray(state.mean), np.asarray([2.5]))
        np.testing.assert_array_equal(np.asarray(state.var), np.asarray([6.75]))

        _, state = ema_normalize(
            state,
            jnp.asarray([7.0], dtype=jnp.float32),
            1.0,
            1e-8,
        )
        np.testing.assert_array_equal(np.asarray(state.mean), np.asarray([4.0]))
        np.testing.assert_array_equal(np.asarray(state.var), np.asarray([9.0]))

    def test_parity_with_core_ema_normalizer(self):
        normalizer = EMANormalizer(epsilon=1e-8, decay=0.999)
        core_state = normalizer.init(6)
        mine = EMANormState(
            mean=jnp.zeros(6), var=jnp.ones(6), count=jnp.array(0.0)
        )
        key = jr.key(11)
        for i in range(20):
            obs = jr.normal(jr.fold_in(key, i), (6,)) * 3.0 + 1.0
            ref, core_state = normalizer.normalize(core_state, obs)
            got, mine = ema_normalize(mine, obs, 0.999, 1e-8)
            np.testing.assert_allclose(got, ref, atol=1e-6)


class TestPerElementAdam:
    def test_uniform_count_parity_with_baseline_adam(self):
        hp = {"step_size": 1e-4, "beta1": 0.0, "beta2": 0.99, "eps": 1e-8,
              "weight_decay": 0.0}
        optimizer = Adam(**hp)
        shape = (5, 4)
        ref_state = optimizer.init_for_shape(shape)
        key = jr.key(2)
        param = jr.normal(jr.fold_in(key, 99), shape)
        ref_param = param
        m = jnp.zeros(shape)
        v = jnp.zeros(shape)
        count = jnp.zeros(shape)
        for i in range(6):
            grad = jr.normal(jr.fold_in(key, i), shape)
            step, ref_state = optimizer.update_from_gradient(
                ref_state, grad, error=None, param=ref_param
            )
            ref_param = ref_param - step
            param, m, v, count = adam_elem_update(param, m, v, count, grad, hp)
            np.testing.assert_allclose(param, ref_param, atol=1e-7)


class TestCBPReplacement:
    def test_replacement_recycles_lowest_utility_mature_unit(self):
        key = jr.key(5)
        params = init_mlp_params(key, SMALL)
        h1 = SMALL.hidden1
        utility = jnp.arange(1, h1 + 1, dtype=jnp.float32)  # unit 0 lowest
        age = jnp.full((h1,), 200, dtype=jnp.int32)
        opt = {n: jnp.ones((2,) + v.shape, dtype=jnp.float32) for n, v in params.items()}
        new_params, new_opt, new_util, new_age, new_acc = cbp_maybe_replace_layer(
            params, opt, utility, age, jnp.array(0.5), _CBP_LAYERS[0], jr.key(6),
            replacement_rate=1.0 / h1, maturity_threshold=100,
        )
        # accumulator 0.5 + 1.0 -> fired and decremented
        assert float(new_acc) == pytest.approx(0.5)
        assert float(new_util[0]) == 0.0
        assert int(new_age[0]) == 0
        # incoming column replaced, in protocol init range
        bound = 1.0 / math.sqrt(SMALL.input_dim)
        col = np.asarray(new_params["w1"][:, 0])
        assert not np.allclose(col, np.asarray(params["w1"][:, 0]))
        assert np.all(np.abs(col) <= bound)
        assert float(new_params["b1"][0]) == 0.0
        # outgoing row zeroed
        np.testing.assert_allclose(np.asarray(new_params["w2"][0, :]), 0.0)
        # optimizer slices reset
        np.testing.assert_allclose(np.asarray(new_opt["w1"][:, :, 0]), 0.0)
        np.testing.assert_allclose(np.asarray(new_opt["b1"][:, 0]), 0.0)
        np.testing.assert_allclose(np.asarray(new_opt["w2"][:, 0, :]), 0.0)
        # untouched units keep their values
        np.testing.assert_allclose(
            np.asarray(new_params["w1"][:, 1]), np.asarray(params["w1"][:, 1])
        )

    def test_no_replacement_when_accumulator_below_one(self):
        key = jr.key(5)
        params = init_mlp_params(key, SMALL)
        h1 = SMALL.hidden1
        utility = jnp.arange(1, h1 + 1, dtype=jnp.float32)
        age = jnp.full((h1,), 200, dtype=jnp.int32)
        new_params, _, new_util, new_age, new_acc = cbp_maybe_replace_layer(
            params, None, utility, age, jnp.array(0.0), _CBP_LAYERS[0], jr.key(6),
            replacement_rate=0.01 / h1, maturity_threshold=100,
        )
        assert float(new_acc) == pytest.approx(0.01)
        for n in params:
            np.testing.assert_allclose(np.asarray(new_params[n]), np.asarray(params[n]))

    def test_no_replacement_when_no_mature_units(self):
        key = jr.key(5)
        params = init_mlp_params(key, SMALL)
        h1 = SMALL.hidden1
        utility = jnp.arange(1, h1 + 1, dtype=jnp.float32)
        age = jnp.zeros((h1,), dtype=jnp.int32)
        new_params, _, _, _, new_acc = cbp_maybe_replace_layer(
            params, None, utility, age, jnp.array(2.0), _CBP_LAYERS[0], jr.key(6),
            replacement_rate=1.0 / h1, maturity_threshold=100,
        )
        # budget accumulates but nothing fires
        assert float(new_acc) == pytest.approx(3.0)
        for n in params:
            np.testing.assert_allclose(np.asarray(new_params[n]), np.asarray(params[n]))


class TestWeightClipping:
    """UPGD-W + weight clipping (Elsayed et al., RLC 2024)."""

    def test_max_finite_kappa_reduces_exactly_to_control(self, small_data):
        """With max-finite kappa the clip is a no-op: bit-exact control trajectory."""
        x, y = small_data
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp["clip_kappa"] = float(np.finfo(np.float32).max)
        spec = ScreeningSpec(
            name="upgd_w_control",  # reuse control identity for shard plumbing
            base_learner="upgd_w",
            mechanism="weight_clipping",
            hyperparameters=hp,
            factory=_make_upgd_w_wclip_learner,
        )
        ours = run_screening_config(x, y, spec, seed=7, config=SMALL)
        control = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=7, config=SMALL
        )
        np.testing.assert_array_equal(ours.per_task_accuracy, control.per_task_accuracy)
        np.testing.assert_allclose(ours.per_task_loss, control.per_task_loss, rtol=1e-6)

    def test_clip_bounds_enforced_per_layer(self):
        """After one step every parameter obeys |w| <= kappa / sqrt(fan_in)."""
        key = jr.key(4)
        params = init_mlp_params(key, SMALL)
        big = {n: v + 100.0 for n, v in params.items()}  # way outside every bound
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp["clip_kappa"] = 1.0
        state = LeanUPGDState(
            utility={n: jnp.ones_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        grads = {n: jnp.zeros_like(v) for n, v in params.items()}
        noise = {n: jnp.zeros_like(v) for n, v in params.items()}
        new_params, _ = upgd_w_wclip_update(big, state, grads, noise, hp)
        fan_in = {"1": SMALL.input_dim, "2": SMALL.hidden1, "3": SMALL.hidden2}
        for n in new_params:
            bound = hp["clip_kappa"] / math.sqrt(fan_in[n[1:]])
            values = np.asarray(new_params[n])
            assert np.all(values <= bound + 1e-7), n
            assert np.all(values >= -bound - 1e-7), n
            # saturated: the +100 shift puts everything at the upper bound
            np.testing.assert_allclose(values, bound, atol=1e-6)

    def test_registry_wd0_variants_disable_weight_decay(self):
        assert screening_spec("upgd_w_wclip_k1_wd0").hyperparameters["weight_decay"] == 0.0
        assert screening_spec("upgd_w_wclip_k2_wd0").hyperparameters["weight_decay"] == 0.0
        assert screening_spec("upgd_w_wclip_k1").hyperparameters["weight_decay"] == 0.01
        assert screening_spec("upgd_w_wclip_k1").hyperparameters["clip_kappa"] == 1.0
        assert screening_spec("upgd_w_wclip_k2").hyperparameters["clip_kappa"] == 2.0


class TestLocalGateNorm:
    """Per-tensor utility-gate normalization vs the published global max."""

    def _random_inputs(self, params, hp, seed=9):
        kg, kn = jr.split(jr.key(seed))
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                 for i, (n, v) in enumerate(sorted(params.items()))}
        noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                 for i, (n, v) in enumerate(sorted(params.items()))}
        return grads, noise

    def test_single_tensor_equals_global(self):
        """With one parameter tensor, per-tensor max == global max: identical."""
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        key = jr.key(3)
        params = {"w1": jr.normal(key, (12, 8), jnp.float32) * 0.1}
        state = LeanUPGDState(
            utility={"w1": jnp.zeros_like(params["w1"])},
            step=jnp.array(0, dtype=jnp.int32),
        )
        lean_state = LeanUPGDState(utility=dict(state.utility), step=state.step)
        grads, noise = self._random_inputs(params, hp)
        for _ in range(3):
            p_local, state = upgd_w_localgate_update(params, state, grads, noise, hp)
            p_global, lean_state = lean_upgd_w_update(params, lean_state, grads, noise, hp)
            np.testing.assert_array_equal(
                np.asarray(p_local["w1"]), np.asarray(p_global["w1"])
            )
            params = p_local

    def test_multi_tensor_differs_from_global(self):
        """With several tensors whose utility maxima differ, the gates differ."""
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        key = jr.key(5)
        params = init_mlp_params(key, SMALL)
        state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        lean_state = LeanUPGDState(utility=dict(state.utility), step=state.step)
        grads, noise = self._random_inputs(params, hp, seed=17)
        p_local, _ = upgd_w_localgate_update(params, state, grads, noise, hp)
        p_global, _ = lean_upgd_w_update(params, lean_state, grads, noise, hp)
        assert any(
            not np.allclose(np.asarray(p_local[n]), np.asarray(p_global[n]))
            for n in params
        )


class TestSeedBoundary:
    @pytest.mark.parametrize("seed", [True, np.int64(0), 0.0, -1, 2**32])
    def test_runner_rejects_noncanonical_seed_before_setup(
        self, monkeypatch: pytest.MonkeyPatch, seed: object
    ) -> None:
        def unexpected_setup(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("invalid seed reached screening setup")

        monkeypatch.setattr(
            ipmnist_screening, "_validated_screening_noise_mode", unexpected_setup
        )
        with pytest.raises(ValueError, match="seed.*uint32"):
            run_screening_config(
                np.empty((1, 1), dtype=np.float32),
                np.empty((1,), dtype=np.int32),
                screening_spec("upgd_w_control"),
                seed=seed,  # type: ignore[arg-type]
                config=SMALL,
            )

    def test_cli_rejects_aliased_seed_before_loading_mnist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected_load(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("invalid seed reached dataset loading")

        monkeypatch.setattr(ipmnist_screening, "load_mnist_train", unexpected_load)
        output = tmp_path / "must-not-exist.json"
        with pytest.raises(ValueError, match="seed.*uint32"):
            main(
                [
                    "run",
                    "--config-name",
                    "upgd_w_control",
                    "--seed",
                    str(2**32),
                    "--out",
                    str(output),
                ]
            )
        assert not output.exists()


class TestSmokeRuns:
    """Every combination runs at tiny scale, learns above chance, stays finite."""

    @pytest.mark.parametrize("name", [
        "upgd_idbd", "upgd_autostep", "upgd_l2init", "upgd_ema_norm",
        "upgd_cbp", "adamw_cbp", "upgd_w_wclip_k1", "upgd_w_wclip_k2_wd0",
        "upgd_w_localgate", "upgd_w_fade_head", "upgd_w_idbd_swift",
        "guarded_cbp_adam", "adamw_cbp_noreset", "adamw_cbp_ema_norm",
        "upgd_w_sigma0", "upgd_alpha_utility", "adamw_cbp_r3e4",
        "sgd_ema_norm", "sigma0_ndecay099", "sigma0_ndecay09999",
        "sigma0_eps1e6", "sigma0_eps1e4", "sigma0_hidden_norm",
        "sigma0_gate_beta05", "sigma0_gate_beta2", "sigma0_localgate",
        "sigma0_shiftnorm", "sigma0_shiftnorm_d099_r200", "sigma0_warmnorm",
        "sigma0_gateplus",
    ])
    def test_combo_runs_and_is_finite(self, small_data, name):
        x, y = small_data
        result = run_screening_config(x, y, screening_spec(name), seed=1, config=SMALL)
        assert result.per_task_accuracy.shape == (SMALL.n_tasks,)
        assert np.all(np.isfinite(result.per_task_accuracy))
        assert np.all(result.per_task_accuracy >= 0.0)
        assert np.all(result.per_task_accuracy <= 1.0)
        assert np.all(np.isfinite(result.per_task_loss))
        assert np.all(result.per_task_plasticity >= 0.0)
        assert np.all(result.per_task_plasticity <= 1.0)


class TestShardsAndMerge:
    def _make_shard(self, tmp_path, small_data, config_name, seed):
        x, y = small_data
        result = run_screening_config(
            x,
            y,
            screening_spec(config_name),
            seed=seed,
            config=SMALL,
        )
        path = tmp_path / f"{config_name}_seed{seed}.json"
        path.write_text(json.dumps(_bound_shard_payload(result)), encoding="utf-8")
        return path

    def test_shard_roundtrip_and_merge(self, tmp_path, small_data):
        paths = [
            self._make_shard(tmp_path, small_data, "upgd_w_control", 0),
            self._make_shard(tmp_path, small_data, "upgd_w_control", 1),
            self._make_shard(tmp_path, small_data, "upgd_l2init", 0),
            self._make_shard(tmp_path, small_data, "upgd_l2init", 1),
        ]
        for p in paths:
            assert load_shard(p)["schema"] == SHARD_SCHEMA
        summary = merge_shards(paths, control_name="upgd_w_control", slope_window=2)
        names = {e["config_name"] for e in summary["results"]}
        assert names == {"upgd_w_control", "upgd_l2init"}
        l2 = next(e for e in summary["results"] if e["config_name"] == "upgd_l2init")
        assert l2["paired_vs_control"]["seeds"] == [0, 1]
        assert len(l2["paired_vs_control"]["per_seed_diff"]) == 2
        assert isinstance(l2["paired_vs_control"]["confirmation_candidate"], bool)
        control = next(e for e in summary["results"] if e["config_name"] == "upgd_w_control")
        assert "paired_vs_control" not in control

    def test_load_shard_rejects_seed_outside_jax_key_domain(
        self, tmp_path, small_data
    ) -> None:
        path = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seed"] = 2**32
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="seed.*uint32"):
            load_shard(path)

    @pytest.mark.parametrize("location", ["top-level", "nested"])
    def test_load_shard_rejects_duplicate_top_level_and_nested_keys(
        self, tmp_path: Path, location: str
    ) -> None:
        spec = screening_spec("upgd_w_control")
        payload = _bound_shard_payload(
            ipmnist_screening.ScreeningRunResult(
                config_name=spec.name,
                base_learner=spec.base_learner,
                hyperparameters=dict(spec.hyperparameters),
                seed=0,
                config=SMALL,
                per_task_accuracy=np.zeros(SMALL.n_tasks),
                per_task_loss=np.zeros(SMALL.n_tasks),
                per_task_plasticity=np.zeros(SMALL.n_tasks),
                wall_clock_seconds=0.0,
            )
        )
        encoded = json.dumps(payload, separators=(",", ":"))
        encoded_config = json.dumps(payload["config"], separators=(",", ":"))
        duplicate_config = '{"n_tasks":999,' + encoded_config[1:]
        variants = {
            "top-level": '{"schema":"forged",' + encoded[1:],
            "nested": encoded.replace(
                f'"config":{encoded_config}',
                f'"config":{duplicate_config}',
                1,
            ),
        }

        path = tmp_path / f"duplicate-{location}.json"
        path.write_text(variants[location], encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate JSON object key"):
            load_shard(path)

    def test_merge_rejects_zero_seed_overlap_with_control(self, tmp_path, small_data):
        """An arm sharing no seeds with a present control must refuse to merge:
        the entry would rank in the summary with no paired_vs_control block and
        nothing in the artifact marking it unpaired (issue #49)."""
        paths = [
            self._make_shard(tmp_path, small_data, "upgd_w_control", 0),
            self._make_shard(tmp_path, small_data, "upgd_w_control", 1),
            self._make_shard(tmp_path, small_data, "upgd_l2init", 2),
        ]
        with pytest.raises(
            ValueError,
            match=r"'upgd_l2init' shares no seeds with control 'upgd_w_control'",
        ):
            merge_shards(paths, control_name="upgd_w_control", slope_window=2)

    def test_merge_rejects_partial_seed_overlap(self, tmp_path):
        """A ranked summary must compare every arm on exactly paired seeds."""
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.40),
            self._write_inband_shard(tmp_path, "upgd_w_control", 1, 0.60),
            self._write_inband_shard(tmp_path, "upgd_l2init", 1, 0.55),
            self._write_inband_shard(tmp_path, "upgd_l2init", 2, 0.90),
        ]

        with pytest.raises(
            ValueError,
            match=r"seed sets differ across configs.*ranks configs on paired seeds only",
        ):
            merge_shards(paths, control_name="upgd_w_control", slope_window=2)

    def _write_inband_shard(self, tmp_path, config_name, seed, accuracy):
        """Write a structurally valid shard with controlled accuracy."""
        payload = {
            "schema": LEGACY_SHARD_SCHEMA,
            "config_name": config_name,
            "base_learner": "upgd_w",
            "hyperparameters": {},
            "seed": seed,
            "noise_mode": "step",
            "config": SMALL.to_config(),
            "per_task_accuracy": [float(accuracy)] * SMALL.n_tasks,
            "per_task_loss": [0.1] * SMALL.n_tasks,
            "per_task_plasticity": [0.5] * SMALL.n_tasks,
            "wall_clock_seconds": 1.0,
            "environment": {
                "jax": "test-jax",
                "numpy": "test-numpy",
                "python": "test-python",
                "platform": "test-platform",
            },
        }
        path = tmp_path / f"{config_name}_seed{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @pytest.mark.parametrize(
        ("fieldname", "mutate", "message"),
        [
            ("per_task_accuracy", lambda curve: [5.0] * len(curve), r"must be in \[0, 1\]"),
            ("per_task_accuracy", lambda curve: [-0.5] * len(curve), r"must be in \[0, 1\]"),
            ("per_task_plasticity", lambda curve: [1.5] * len(curve), r"must be in \[0, 1\]"),
            ("per_task_loss", lambda curve: [-1.0] * len(curve), "must be non-negative"),
            (
                "per_task_accuracy",
                lambda curve: [str(v) for v in curve],
                "must be a list of finite JSON numbers",
            ),
            (
                "per_task_accuracy",
                lambda curve: [True] * len(curve),
                "must be a list of finite JSON numbers",
            ),
            (
                "per_task_loss",
                lambda curve: [str(v) for v in curve],
                "must be a list of finite JSON numbers",
            ),
        ],
    )
    def test_load_shard_applies_curve_typing_and_domain_to_legacy_v1(
        self, tmp_path, fieldname, mutate, message
    ):
        """v1 is the campaign's live shard format; its curves get the same domain checks as v2."""
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[fieldname] = mutate(list(payload[fieldname]))
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_shard(path)

    def test_load_shard_canonicalizes_legacy_v1_integer_curve_entries(self, tmp_path):
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["per_task_accuracy"] = [1] + payload["per_task_accuracy"][1:]
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_shard(path)
        assert loaded["per_task_accuracy"][0] == 1.0
        assert type(loaded["per_task_accuracy"][0]) is float

    def test_out_of_domain_legacy_shard_cannot_top_a_merge(self, tmp_path):
        control = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        winner = self._write_inband_shard(tmp_path, "upgd_l2init", 0, 0.6)
        payload = json.loads(winner.read_text(encoding="utf-8"))
        payload["per_task_accuracy"] = [5.0] * SMALL.n_tasks
        winner.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            merge_shards([control, winner], control_name="upgd_w_control", slope_window=2)

    def test_load_shard_rejects_invalid_wall_clock_types_and_values(self, tmp_path):
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))

        for wall_clock in (
            None,
            True,
            False,
            "1.0",
            [],
            {},
            math.inf,
            -math.inf,
            math.nan,
            -1,
            10**309,
        ):
            payload["wall_clock_seconds"] = wall_clock
            path.write_text(json.dumps(payload), encoding="utf-8")
            message = (
                "non-standard JSON numeric constant"
                if type(wall_clock) is float and not math.isfinite(wall_clock)
                else "wall_clock_seconds"
            )
            with pytest.raises(ValueError, match=message):
                load_shard(path)

    @pytest.mark.parametrize("wall_clock", [0, 0.0, 1, 1.25, 1e308])
    def test_load_shard_preserves_valid_wall_clock(self, tmp_path, wall_clock):
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["wall_clock_seconds"] = wall_clock
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_shard(path)["wall_clock_seconds"]
        assert type(loaded) is float
        assert loaded == float(wall_clock)

    def test_load_shard_rejects_unknown_noise_mode(self, tmp_path):
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["noise_mode"] = "teleport"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            load_shard(path)

        assert str(exc_info.value) == (
            f"{path}: noise_mode must be 'step' or 'pool', got 'teleport'"
        )

    @pytest.mark.parametrize("noise_mode", [None, True, 0, [], {}])
    def test_load_shard_rejects_non_string_noise_mode(self, tmp_path, noise_mode):
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["noise_mode"] = noise_mode
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="noise_mode must be"):
            load_shard(path)

    def test_load_shard_rejects_pool_mode_for_incompatible_arm(self, tmp_path):
        path = self._write_inband_shard(tmp_path, "upgd_idbd", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["noise_mode"] = "pool"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            load_shard(path)

        assert str(exc_info.value) == (
            f"{path}: noise_mode='pool' is unsupported for 'upgd_idbd': "
            "the arm declares no noise-consuming update"
        )

    @pytest.mark.parametrize(
        ("config_name", "noise_mode"),
        [("upgd_idbd", "step"), ("upgd_w_control", "pool")],
    )
    def test_load_shard_accepts_runner_supported_noise_modes(
        self, tmp_path, config_name, noise_mode
    ):
        path = self._write_inband_shard(tmp_path, config_name, 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["noise_mode"] = noise_mode
        if noise_mode == "pool":
            payload["noise_pool_steps"] = 64
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_shard(path)
        assert loaded["noise_mode"] == noise_mode
        assert loaded["noise_pool_steps"] == (64 if noise_mode == "pool" else None)

    @pytest.mark.parametrize(
        "noise_pool_steps",
        [None, True, False, 8.0, "8", [], {}, 1, 0, -1],
    )
    def test_load_shard_rejects_invalid_pool_size(
        self, tmp_path: Path, noise_pool_steps: object
    ) -> None:
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["noise_mode"] = "pool"
        payload["noise_pool_steps"] = noise_pool_steps
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="noise_pool_steps"):
            load_shard(path)

    def test_load_shard_preserves_unrecorded_legacy_pool_size_as_unknown(
        self, tmp_path: Path
    ) -> None:
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["noise_mode"] = "pool"
        payload.pop("noise_pool_steps", None)
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_shard(path)
        assert loaded["noise_mode"] == "pool"
        assert loaded["noise_pool_steps"] is None

        with pytest.raises(ValueError, match="do not record noise_pool_steps"):
            merge_shards([path], control_name="upgd_w_control", slope_window=2)

    def test_load_shard_rejects_pool_size_for_step_mode(self, tmp_path: Path) -> None:
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["noise_pool_steps"] = 64
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="null or absent.*noise_mode='step'"):
            load_shard(path)

    @pytest.mark.parametrize("noise_mode", [None, "step"])
    def test_load_shard_normalizes_legacy_and_step_pool_size_to_none(
        self, tmp_path: Path, noise_mode: str | None
    ) -> None:
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if noise_mode is None:
            payload.pop("noise_mode")
        payload.pop("noise_pool_steps", None)
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_shard(path)
        assert loaded["noise_mode"] == "step"
        assert loaded["noise_pool_steps"] is None

    def test_merge_rejects_mixed_pool_sizes(self, tmp_path: Path) -> None:
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", seed, 0.5)
            for seed in (0, 1)
        ]
        for path, noise_pool_steps in zip(paths, (8, 512), strict=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["noise_mode"] = "pool"
            payload["noise_pool_steps"] = noise_pool_steps
            path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="multiple noise_pool_steps"):
            merge_shards(paths, control_name="upgd_w_control", slope_window=2)

    def test_merge_rejects_known_and_unrecorded_pool_sizes(self, tmp_path: Path) -> None:
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", seed, 0.5)
            for seed in (0, 1)
        ]
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["noise_mode"] = "pool"
            path.write_text(json.dumps(payload), encoding="utf-8")
        known = json.loads(paths[0].read_text(encoding="utf-8"))
        known["noise_pool_steps"] = 8
        paths[0].write_text(json.dumps(known), encoding="utf-8")

        with pytest.raises(ValueError, match="do not record noise_pool_steps"):
            merge_shards(paths, control_name="upgd_w_control", slope_window=2)

    def test_merge_requires_one_pool_size_across_all_arms(self, tmp_path: Path) -> None:
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5),
            self._write_inband_shard(tmp_path, "upgd_w_localgate", 0, 0.6),
        ]
        for path, noise_pool_steps in zip(paths, (8, 512), strict=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["noise_mode"] = "pool"
            payload["noise_pool_steps"] = noise_pool_steps
            path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="multiple noise_pool_steps"):
            merge_shards(paths, control_name="upgd_w_control", slope_window=2)

    def test_merge_records_one_pool_size_across_all_arms(self, tmp_path: Path) -> None:
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5),
            self._write_inband_shard(tmp_path, "upgd_w_localgate", 0, 0.6),
        ]
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["noise_mode"] = "pool"
            payload["noise_pool_steps"] = 8
            path.write_text(json.dumps(payload), encoding="utf-8")

        summary = merge_shards(paths, control_name="upgd_w_control", slope_window=2)

        assert summary["noise_mode"] == "pool"
        assert summary["noise_pool_steps"] == 8

    @pytest.mark.parametrize(
        ("config_name", "noise_mode", "error"),
        [
            (
                "upgd_w_control",
                "teleport",
                "noise_mode must be 'step' or 'pool'",
            ),
            (
                "upgd_idbd",
                "pool",
                "noise_mode='pool' is unsupported for 'upgd_idbd'",
            ),
        ],
    )
    def test_merge_cli_rejects_impossible_noise_mode_without_publishing(
        self, tmp_path, config_name, noise_mode, error
    ):
        paths = [
            self._write_inband_shard(tmp_path, config_name, seed, 0.5)
            for seed in (0, 1)
        ]
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["noise_mode"] = noise_mode
            path.write_text(json.dumps(payload), encoding="utf-8")
        output = tmp_path / "summary.json"

        with pytest.raises(ValueError, match=error):
            main(
                [
                    "merge",
                    "--shards",
                    *(str(path) for path in paths),
                    "--control-name",
                    config_name,
                    "--slope-window",
                    "2",
                    "--output",
                    str(output),
                ]
            )

        assert not output.exists()

    @pytest.mark.parametrize("noise_mode", [None, "step", "pool"])
    def test_merge_preserves_legacy_and_supported_noise_modes(
        self, tmp_path, noise_mode
    ):
        path = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if noise_mode is None:
            del payload["noise_mode"]
            expected = "step"
        else:
            payload["noise_mode"] = noise_mode
            expected = noise_mode
        if noise_mode == "pool":
            payload["noise_pool_steps"] = 64
        path.write_text(json.dumps(payload), encoding="utf-8")

        summary = merge_shards(
            [path], control_name="upgd_w_control", slope_window=2
        )

        assert summary["noise_mode"] == expected
        assert summary["noise_pool_steps"] == (64 if noise_mode == "pool" else None)

    def test_merge_preserves_valid_wall_clock_summary(self, tmp_path):
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", seed, 0.5)
            for seed in (0, 1)
        ]
        for path, wall_clock in zip(paths, (0, 1.25), strict=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["wall_clock_seconds"] = wall_clock
            path.write_text(json.dumps(payload), encoding="utf-8")

        summary = merge_shards(paths, control_name="upgd_w_control", slope_window=2)

        assert summary["results"][0]["wall_clock_seconds_total"] == 1.25

    def test_merge_rejects_wall_clock_total_overflow(self, tmp_path):
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", seed, 0.5)
            for seed in (0, 1)
        ]
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["wall_clock_seconds"] = 1e308
            path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="wall_clock_seconds_total must be finite"):
            merge_shards(paths, control_name="upgd_w_control", slope_window=2)

    def test_confirmation_candidate_requires_two_paired_seeds(self, tmp_path):
        """One paired seed cannot authorize a confirmation wave."""
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.50),
            self._write_inband_shard(tmp_path, "upgd_l2init", 0, 0.56),
        ]
        summary = merge_shards(paths, control_name="upgd_w_control", slope_window=2)
        result = next(
            entry for entry in summary["results"] if entry["config_name"] == "upgd_l2init"
        )
        paired = result["paired_vs_control"]

        assert paired["seeds"] == [0]
        assert paired["mean_diff"] > CONFIRMATION_THRESHOLD
        assert paired["stderr_diff"] == 0.0
        assert paired["all_seeds_improve"] is True
        assert paired["beats_control"] is True
        assert paired["confirmation_candidate"] is False

    def test_confirmation_candidate_allows_two_paired_seeds(self, tmp_path):
        """Two shared improving seeds preserve the existing decision boundary."""
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.50),
            self._write_inband_shard(tmp_path, "upgd_w_control", 1, 0.50),
            self._write_inband_shard(tmp_path, "upgd_l2init", 0, 0.56),
            self._write_inband_shard(tmp_path, "upgd_l2init", 1, 0.56),
        ]
        summary = merge_shards(paths, control_name="upgd_w_control", slope_window=2)
        result = next(
            entry for entry in summary["results"] if entry["config_name"] == "upgd_l2init"
        )
        paired = result["paired_vs_control"]

        assert paired["seeds"] == [0, 1]
        assert paired["mean_diff"] > CONFIRMATION_THRESHOLD
        assert paired["confirmation_candidate"] is True

    def test_confirmation_candidate_rejects_mixed_sign_paired_differences(self, tmp_path):
        """A strong mean cannot hide an aligned seed that regresses."""
        paths = [
            self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.50),
            self._write_inband_shard(tmp_path, "upgd_w_control", 1, 0.50),
            self._write_inband_shard(tmp_path, "upgd_l2init", 0, 0.60),
            self._write_inband_shard(tmp_path, "upgd_l2init", 1, 0.499),
        ]
        summary = merge_shards(paths, control_name="upgd_w_control", slope_window=2)
        result = next(
            entry for entry in summary["results"] if entry["config_name"] == "upgd_l2init"
        )
        paired = result["paired_vs_control"]

        assert paired["seeds"] == [0, 1]
        assert paired["mean_diff"] > CONFIRMATION_THRESHOLD
        assert paired["all_seeds_improve"] is False
        assert paired["confirmation_candidate"] is False

    def test_atomic_writer_refuses_duplicate_without_mutating_first_result(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "one-result.json"
        first = {"config_name": "first", "seed": 0}
        second = {"config_name": "second", "seed": 0}

        _atomic_write_json(path, first)
        published = path.read_bytes()
        with pytest.raises(FileExistsError, match="refusing to overwrite immutable output"):
            _atomic_write_json(path, second)

        assert path.read_bytes() == published
        assert published == (json.dumps(first, indent=1, sort_keys=True) + "\n").encode("utf-8")
        assert json.loads(published) == first
        assert list(tmp_path.iterdir()) == [path]

    def test_simultaneous_atomic_publishers_produce_one_complete_result(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "contended-result.json"
        contenders = 8
        barrier = threading.Barrier(contenders)

        def publish(index: int) -> bool:
            payload = {"config_name": f"candidate-{index}", "seed": index}
            barrier.wait()
            try:
                _atomic_write_json(path, payload)
            except FileExistsError:
                return False
            return True

        with ThreadPoolExecutor(max_workers=contenders) as executor:
            successes = list(executor.map(publish, range(contenders)))

        assert successes.count(True) == 1
        published = json.loads(path.read_bytes())
        assert published == {
            "config_name": f"candidate-{published['seed']}",
            "seed": published["seed"],
        }
        assert 0 <= published["seed"] < contenders
        assert list(tmp_path.iterdir()) == [path]

    @pytest.mark.parametrize(
        ("argv", "guarded_function"),
        [
            (
                ["run", "--config-name", "upgd_w_control", "--seed", "0", "--out"],
                "load_mnist_train",
            ),
            (["merge", "--shards", "unused-shard", "--output"], "merge_shards"),
            (
                [
                    "validate-proxy",
                    "--shards",
                    "unused-shard",
                    "--output",
                ],
                "validate_proxy",
            ),
        ],
    )
    def test_cli_refuses_occupied_output_before_expensive_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        guarded_function: str,
    ) -> None:
        import alberta_framework.benchmarks.ipmnist_screening as screening

        output = tmp_path / "occupied.json"
        original = b"already complete\n"
        output.write_bytes(original)

        def must_not_run(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(f"{guarded_function} ran before output preflight")

        monkeypatch.setattr(screening, guarded_function, must_not_run)
        with pytest.raises(FileExistsError, match="refusing to overwrite immutable output"):
            main([*argv, str(output)])

        assert output.read_bytes() == original
        assert list(tmp_path.iterdir()) == [output]

    @pytest.mark.parametrize(("accepted", "expected_status"), [(True, 0), (False, 2)])
    def test_proxy_cli_publishes_receipt_and_returns_fail_closed_status(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        accepted: bool,
        expected_status: int,
    ) -> None:
        import alberta_framework.benchmarks.ipmnist_screening as screening

        report = {
            "schema": "synthetic.proxy.receipt",
            "proxy_validated": accepted,
            "all_prefixes_match": accepted,
            "proxy_preserves_upgd_over_adamw": accepted,
        }
        monkeypatch.setattr(screening, "validate_proxy", lambda *_args, **_kwargs: report)
        monkeypatch.setattr(
            screening, "_screening_derivation_bindings", lambda _paths: None
        )
        output = tmp_path / f"proxy-{accepted}.json"

        status = screening.main(
            [
                "validate-proxy",
                "--shards",
                "unused-shard",
                "--output",
                str(output),
            ]
        )

        assert status == expected_status
        assert json.loads(output.read_text(encoding="utf-8")) == report

    def test_merge_rejects_absent_control(self, tmp_path, small_data):
        paths = [
            self._make_shard(tmp_path, small_data, "upgd_l2init", 0),
            self._make_shard(tmp_path, small_data, "upgd_l2init", 1),
        ]
        with pytest.raises(
            ValueError,
            match=r"control 'upgd_w_control' is not among the merged shards",
        ):
            merge_shards(paths, control_name="upgd_w_control", slope_window=2)

    def test_merge_rejects_typoed_control_name(self, tmp_path, small_data):
        paths = [
            self._make_shard(tmp_path, small_data, "upgd_w_control", 0),
            self._make_shard(tmp_path, small_data, "upgd_l2init", 0),
        ]
        with pytest.raises(
            ValueError,
            match=r"control 'upgd_w_contrl' is not among the merged shards",
        ):
            merge_shards(paths, control_name="upgd_w_contrl", slope_window=2)

    def test_merge_rejects_duplicate_seed(self, tmp_path, small_data):
        p1 = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        p2 = tmp_path / "dup.json"
        p2.write_text(p1.read_text(encoding="utf-8"), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate shard"):
            merge_shards([p1, p2])

    def test_merge_rejects_hyperparameter_drift_across_seeds(self, tmp_path):
        """Two shards under one config_name with different hyperparameters must
        not merge: nothing else catches a registry value that changed between
        two `run` invocations, and a silent merge would average per-task
        accuracy across genuinely different mechanisms while reporting only
        the first seed's hyperparameters as if representative of the arm."""
        p0 = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        p1 = self._write_inband_shard(tmp_path, "upgd_l2init", 0, 0.5)
        p2 = self._write_inband_shard(tmp_path, "upgd_l2init", 1, 0.5)

        payload2 = json.loads(p2.read_text(encoding="utf-8"))
        payload2["hyperparameters"] = {
            **payload2["hyperparameters"],
            "meta_step_size": 999.0,
        }
        p2.write_text(json.dumps(payload2), encoding="utf-8")

        with pytest.raises(
            ValueError,
            match=r"'upgd_l2init' has inconsistent hyperparameters across seeds",
        ):
            merge_shards([p0, p1, p2], control_name="upgd_w_control", slope_window=2)

    def test_merge_rejects_base_learner_drift_across_seeds(self, tmp_path):
        p0 = self._write_inband_shard(tmp_path, "upgd_w_control", 0, 0.5)
        p1 = self._write_inband_shard(tmp_path, "upgd_l2init", 0, 0.5)
        p2 = self._write_inband_shard(tmp_path, "upgd_l2init", 1, 0.5)

        payload2 = json.loads(p2.read_text(encoding="utf-8"))
        payload2["base_learner"] = "adamw"
        p2.write_text(json.dumps(payload2), encoding="utf-8")

        with pytest.raises(
            ValueError,
            match=r"'upgd_l2init' has inconsistent base_learner across seeds",
        ):
            merge_shards([p0, p1, p2], control_name="upgd_w_control", slope_window=2)

    def test_merge_rejects_environment_drift_and_records_environment(
        self, tmp_path, small_data
    ):
        p0 = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        p1 = self._make_shard(tmp_path, small_data, "upgd_w_control", 1)
        reference_environment = load_shard(p0)["environment"]

        summary = merge_shards([p0, p1], control_name="upgd_w_control", slope_window=2)
        assert summary["environment"] == reference_environment

        payload1 = json.loads(p1.read_text(encoding="utf-8"))
        payload1["environment"] = _test_runtime_environment(machine="different-machine")
        p1.write_text(json.dumps(payload1), encoding="utf-8")

        with pytest.raises(ValueError, match="shards span multiple runtime environments"):
            merge_shards([p0, p1], control_name="upgd_w_control", slope_window=2)

    @pytest.mark.parametrize(
        "environment",
        [
            None,
            {},
            {"jax": "test", "numpy": "test", "python": "test"},
            {"jax": "", "numpy": "test", "python": "test", "platform": "test"},
        ],
    )
    def test_load_rejects_incomplete_environment(
        self, tmp_path, small_data, environment
    ):
        path = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["environment"] = environment
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="runtime environment"):
            load_shard(path)

    @pytest.mark.parametrize(
        ("fieldname", "bad_value"),
        [("base_learner", []), ("hyperparameters", "not-an-object")],
    )
    def test_load_rejects_invalid_arm_contract_field(
        self, tmp_path, small_data, fieldname, bad_value
    ):
        path = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[fieldname] = bad_value
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=fieldname):
            load_shard(path)

    def test_validate_proxy_prefix_and_ordering(self, tmp_path, small_data):
        x, y = small_data
        partials = tmp_path / "partials"
        partials.mkdir()
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        shard_paths = []
        for name, learner in (("upgd_w_control", "upgd_w"), ("adamw_control", "adamw")):
            full = run_ipmnist(x, y, learner, seeds=[0], config=SMALL)
            reference_payload = {
                "schema": "upgd_ipmnist.partial.v1",
                "learner": learner,
                "hyperparameters": full.hyperparameters,
                "seeds": [0],
                "config": {
                    **full.config.to_config(),
                    "input_dim": 784,
                    "n_classes": 10,
                },
                "per_task_accuracy": full.per_task_accuracy.tolist(),
                "per_task_loss": full.per_task_loss.tolist(),
                "per_task_plasticity": full.per_task_plasticity.tolist(),
                "wall_clock_seconds": full.wall_clock_seconds,
            }
            (partials / f"{learner}_seed0.json").write_text(
                json.dumps(reference_payload), encoding="utf-8"
            )
            proxy = run_screening_config(
                x, y, screening_spec(name), seed=0, config=IPMNISTConfig(
                    n_tasks=2, task_length=30, input_dim=12,
                    hidden1=8, hidden2=6, n_classes=5,
                )
            )
            path = shard_dir / f"{name}_seed0.json"
            path.write_text(json.dumps(_bound_shard_payload(proxy)), encoding="utf-8")
            shard_paths.append(path)
        report = validate_proxy(shard_paths, partials, atol=1e-6)
        assert report["all_prefixes_match"] is True
        assert report["environment"] == load_shard(shard_paths[0])["environment"]
        assert report["evidence_policy"] == NONPROMOTING_POLICY
        assert report["source_provenance"] == _test_source_provenance()
        assert report["dataset_provenance"] == _test_dataset_provenance()
        assert report["schema"].endswith(".v2")
        assert len(report["shard_manifest"]) == 2
        assert len(report["reference_partial_manifest"]) == 2
        for field in ("shard_manifest", "reference_partial_manifest"):
            for entry in report[field]:
                raw = Path(entry["path"]).read_bytes()
                assert entry["size_bytes"] == len(raw)
                assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        for check in report["checks"]:
            assert check["max_abs_per_task_diff"] <= 1e-6
        # ordering flags are booleans (tiny-scale runs may order either way)
        assert isinstance(report["proxy_preserves_upgd_over_adamw"], bool)
        assert isinstance(report["full_prefix_preserves_upgd_over_adamw"], bool)

        changed = json.loads(shard_paths[-1].read_text(encoding="utf-8"))
        changed["environment"] = _test_runtime_environment(machine="different-machine")
        changed_path = tmp_path / "changed-environment.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="shards span multiple runtime environments"):
            validate_proxy([shard_paths[0], changed_path], partials, atol=1e-6)

        reference_payloads = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in partials.iterdir()
        }
        tamper_cases = [
            (
                "learner",
                lambda payload: payload.update(learner="adamw"),
                # Hyperparameter-domain validation may reject the mismatched
                # learner before the identity comparison does; either refusal
                # is a correct fail-closed outcome.
                "does not match expected learner|invalid hyperparameters",
            ),
            (
                "seed",
                lambda payload: payload.update(seeds=[1]),
                "reference seeds must equal",
            ),
            (
                "config",
                lambda payload: payload["config"].update(hidden1=999),
                "reference config is incompatible",
            ),
            (
                "hyperparameters",
                lambda payload: payload["hyperparameters"].update(step_size=999.0),
                "reference hyperparameters do not match",
            ),
        ]
        for case_name, mutate, message in tamper_cases:
            tampered_dir = tmp_path / f"tampered-{case_name}"
            tampered_dir.mkdir()
            for filename, payload in reference_payloads.items():
                candidate = json.loads(json.dumps(payload))
                if filename == "upgd_w_seed0.json":
                    mutate(candidate)
                (tampered_dir / filename).write_text(
                    json.dumps(candidate), encoding="utf-8"
                )
            with pytest.raises(ValueError, match=message):
                validate_proxy(shard_paths, tampered_dir, atol=1e-6)

    def test_validate_proxy_rejects_duplicate_control_seed(self, tmp_path, small_data):
        path = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        with pytest.raises(ValueError, match="duplicate proxy-validation shard"):
            validate_proxy([path, path], tmp_path)

    def test_validate_proxy_requires_paired_control_seed_sets(
        self, tmp_path, small_data
    ):
        paths = [
            self._make_shard(tmp_path, small_data, "upgd_w_control", 0),
            self._make_shard(tmp_path, small_data, "upgd_w_control", 1),
            self._make_shard(tmp_path, small_data, "adamw_control", 0),
        ]
        with pytest.raises(ValueError, match="control seed sets differ"):
            validate_proxy(paths, tmp_path)

    def test_validate_proxy_rejects_mixed_protocol_configs(self, tmp_path, small_data):
        upgd = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        adam = self._make_shard(tmp_path, small_data, "adamw_control", 0)
        changed = json.loads(adam.read_text(encoding="utf-8"))
        changed["config"] = {**changed["config"], "hidden1": changed["config"]["hidden1"] + 1}
        changed_path = tmp_path / "changed-config.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")

        with pytest.raises(ValueError, match="multiple protocol configs or horizons"):
            validate_proxy([upgd, changed_path], tmp_path)

    def test_validate_proxy_rejects_misreported_base_learner(
        self, tmp_path, small_data
    ):
        upgd = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        adam = self._make_shard(tmp_path, small_data, "adamw_control", 0)
        changed = json.loads(adam.read_text(encoding="utf-8"))
        changed["base_learner"] = "upgd_w"
        changed_path = tmp_path / "changed-base.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")

        with pytest.raises(ValueError, match="base_learner must match registered arm 'adamw'"):
            validate_proxy([upgd, changed_path], tmp_path)

    def test_validate_proxy_rejects_misreported_control_hyperparameters(
        self, tmp_path, small_data
    ):
        upgd = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        adam = self._make_shard(tmp_path, small_data, "adamw_control", 0)
        changed = json.loads(adam.read_text(encoding="utf-8"))
        changed["hyperparameters"] = {
            **changed["hyperparameters"],
            "step_size": 999.0,
        }
        changed_path = tmp_path / "changed-hyperparameters.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")

        with pytest.raises(ValueError, match="hyperparameters must exactly match"):
            validate_proxy([upgd, changed_path], tmp_path)


class TestPoolConfirmation:
    """Screening-only pool-noise mode for full-protocol confirmation runs."""

    @pytest.mark.parametrize(
        "noise_pool_steps", [None, True, False, np.int64(8), 8.0, "8", 1, 0, -1]
    )
    def test_pool_rejects_noncanonical_pool_size_before_data_setup(
        self, noise_pool_steps: object
    ) -> None:
        with pytest.raises(ValueError, match="noise_pool_steps.*built-in integer >= 2"):
            run_screening_config(
                np.empty((1, 1), dtype=np.float32),
                np.empty((1,), dtype=np.int32),
                screening_spec("upgd_w_control"),
                seed=0,
                config=SMALL,
                noise_mode="pool",
                noise_pool_steps=noise_pool_steps,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("noise_pool_steps", [None, True, np.int64(8), 8.0, 1])
    def test_shard_payload_rejects_noncanonical_pool_size(
        self, noise_pool_steps: object
    ) -> None:
        spec = screening_spec("upgd_w_control")
        with pytest.raises(ValueError, match="noise_pool_steps"):
            ipmnist_screening.ScreeningRunResult(
                config_name=spec.name,
                base_learner=spec.base_learner,
                hyperparameters=dict(spec.hyperparameters),
                seed=0,
                config=SMALL,
                per_task_accuracy=np.zeros(SMALL.n_tasks),
                per_task_loss=np.zeros(SMALL.n_tasks),
                per_task_plasticity=np.zeros(SMALL.n_tasks),
                wall_clock_seconds=0.0,
                noise_mode="pool",
                noise_pool_steps=noise_pool_steps,  # type: ignore[arg-type]
            )

    def test_pool_control_matches_run_ipmnist_pool(self, small_data):
        """Control arm under pool mode reproduces run_ipmnist's pool chain."""
        x, y = small_data
        ours = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=5, config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        reference = run_ipmnist(
            x, y, "upgd_w", seeds=[5], config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        assert ours.noise_mode == "pool"
        assert ours.noise_pool_steps == 8
        assert _bound_shard_payload(ours)["noise_pool_steps"] == 8
        np.testing.assert_allclose(
            ours.per_task_accuracy, reference.per_task_accuracy[0], atol=1e-7
        )
        np.testing.assert_allclose(
            ours.per_task_loss, reference.per_task_loss[0], rtol=1e-5
        )

    def test_pool_differs_from_exact_but_stays_close_at_tiny_scale(self, small_data):
        x, y = small_data
        exact = run_screening_config(
            x, y, screening_spec("upgd_w_wclip_k2"), seed=5, config=SMALL
        )
        pool = run_screening_config(
            x, y, screening_spec("upgd_w_wclip_k2"), seed=5, config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        assert exact.noise_mode == "step"
        # different noise stream => different (continuous) loss trajectory
        assert not np.array_equal(pool.per_task_loss, exact.per_task_loss)

    def test_pool_rejected_for_arms_without_noise_update(self, small_data):
        x, y = small_data
        with pytest.raises(ValueError, match="pool"):
            run_screening_config(
                x, y, screening_spec("upgd_idbd"), seed=0, config=SMALL,
                noise_mode="pool",
            )

    def test_pool_shards_record_mode_and_never_merge_with_exact(
        self, tmp_path, small_data
    ):
        x, y = small_data
        exact = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=0, config=SMALL
        )
        pool = run_screening_config(
            x, y, screening_spec("upgd_w_localgate"), seed=0, config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        exact_payload = _bound_shard_payload(exact)
        pool_payload = _bound_shard_payload(pool)
        assert exact_payload["noise_mode"] == "step"
        assert exact_payload["noise_pool_steps"] is None
        assert pool_payload["noise_mode"] == "pool"
        assert pool_payload["noise_pool_steps"] == 8
        p_exact = tmp_path / "exact.json"
        p_pool = tmp_path / "pool.json"
        p_exact.write_text(json.dumps(exact_payload), encoding="utf-8")
        p_pool.write_text(json.dumps(pool_payload), encoding="utf-8")
        with pytest.raises(ValueError, match="noise mode"):
            merge_shards([p_exact, p_pool])

    def test_validate_proxy_rejects_pool_shards(self, tmp_path, small_data):
        x, y = small_data
        pool = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=0, config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        path = tmp_path / "pool.json"
        path.write_text(json.dumps(_bound_shard_payload(pool)), encoding="utf-8")
        with pytest.raises(ValueError, match="noise_mode"):
            validate_proxy([path], tmp_path)


class TestGuardedCBPAdam:
    """AdamW+CBP with UPGD-style utility protection scaling Adam's delta."""

    def test_registry_config(self):
        spec = screening_spec("guarded_cbp_adam")
        assert spec.base_learner == "adamw"
        assert spec.hyperparameters["guard_scale"] == 1.0
        assert spec.hyperparameters["utility_decay"] == 0.9999
        assert spec.hyperparameters["cbp_replacement_rate"] == 1e-4
        assert spec.hyperparameters["cbp_maturity_threshold"] == 100.0
        # protection only — the arm must have no perturbation channel
        assert "noise_std" not in spec.hyperparameters
        assert spec.noise_update is None

    def test_guard_zero_reduces_to_adamw_cbp_bitwise(self, small_data):
        """guard_scale=0: the whole trajectory equals adamw_cbp bit-for-bit."""
        x, y = small_data
        hp = dict(screening_spec("guarded_cbp_adam").hyperparameters)
        hp["guard_scale"] = 0.0
        spec = ScreeningSpec(
            name="adamw_cbp",  # reuse registry identity for shard plumbing
            base_learner="adamw",
            mechanism="utility_guarded_recycling",
            hyperparameters=hp,
            factory=_make_guarded_cbp_adam_learner,
        )
        ours = run_screening_config(x, y, spec, seed=11, config=SMALL)
        ref = run_screening_config(
            x, y, screening_spec("adamw_cbp"), seed=11, config=SMALL
        )
        np.testing.assert_array_equal(ours.per_task_accuracy, ref.per_task_accuracy)
        np.testing.assert_array_equal(ours.per_task_loss, ref.per_task_loss)
        np.testing.assert_array_equal(
            ours.per_task_plasticity, ref.per_task_plasticity
        )

    def test_high_gate_weight_takes_smaller_step_moments_untouched(self):
        """The gate scales only the applied delta; Adam moments see raw grads."""
        hp = {"step_size": 1e-4, "beta1": 0.0, "beta2": 0.99, "eps": 1e-8,
              "weight_decay": 0.0, "guard_scale": 1.0}
        params = {"w": jnp.array([1.0, 1.0], jnp.float32)}
        zeros = {"w": jnp.zeros(2, jnp.float32)}
        grads = {"w": jnp.array([0.5, 0.5], jnp.float32)}
        gate = {"w": jnp.array([0.9, 0.1], jnp.float32)}
        new_params, new_m, new_v, _ = guarded_adam_update(
            params, dict(zeros), dict(zeros), dict(zeros), grads, gate, hp
        )
        delta = np.asarray(params["w"] - new_params["w"])
        assert abs(delta[0]) < abs(delta[1])  # protected weight moves less
        # equal grads => identical moments regardless of gate
        assert float(new_m["w"][0]) == float(new_m["w"][1])
        assert float(new_v["w"][0]) == float(new_v["w"][1])
        # unprotected delta scales by (1 - gate)
        np.testing.assert_allclose(
            delta[0] / delta[1], (1.0 - 0.9) / (1.0 - 0.1), rtol=1e-2
        )


class TestAdamCBPNoReset:
    """adamw_cbp without per-unit optimizer-state reset at replacement."""

    @pytest.mark.parametrize("name", ["adamw_cbp", "adamw_cbp_noreset"])
    def test_replacement_rate_zero_reduces_to_adamw_control(self, small_data, name):
        x, y = small_data
        hp = dict(screening_spec(name).hyperparameters)
        hp["cbp_replacement_rate"] = 0.0
        spec = ScreeningSpec(
            name="adamw_control",
            base_learner="adamw",
            mechanism="dormant_unit_recycling",
            hyperparameters=hp,
            factory=screening_spec(name).factory,
        )
        ours = run_screening_config(x, y, spec, seed=3, config=SMALL)
        ref = run_screening_config(
            x, y, screening_spec("adamw_control"), seed=3, config=SMALL
        )
        np.testing.assert_allclose(
            ours.per_task_accuracy, ref.per_task_accuracy, atol=1e-7
        )
        np.testing.assert_allclose(ours.per_task_loss, ref.per_task_loss, rtol=1e-5)

    def test_stale_moments_kept_on_replacement(self, small_data):
        """When a layer-1 replacement fires, the reset arm zeroes the recycled
        unit's m/v/count slices while the noreset arm keeps them."""
        x, y = small_data
        params = init_mlp_params(jr.key(0), SMALL)
        hp = dict(screening_spec("adamw_cbp").hyperparameters)
        h1, h2 = SMALL.hidden1, SMALL.hidden2
        fired = AdamCBPState(
            m={n: jnp.full_like(v, 0.25) for n, v in params.items()},
            v={n: jnp.full_like(v, 0.5) for n, v in params.items()},
            count={n: jnp.full_like(v, 7.0) for n, v in params.items()},
            cbp=CBPState(
                util1=jnp.arange(1, h1 + 1, dtype=jnp.float32),  # unit 0 lowest
                util2=jnp.arange(1, h2 + 1, dtype=jnp.float32),
                age1=jnp.full((h1,), 200, dtype=jnp.int32),
                age2=jnp.zeros((h2,), dtype=jnp.int32),  # immature: never fires
                accumulator=jnp.array([1.0, 0.0], jnp.float32),
            ),
        )
        x0 = jnp.asarray(x[0], jnp.float32)
        y0 = jnp.asarray(y[0], jnp.int32)
        _, step_reset = _make_adamw_cbp_learner(hp)
        _, step_nores = _make_adamw_cbp_noreset_learner(hp)
        _, s_reset, _ = step_reset(params, fired, x0, y0, jr.key(42))
        _, s_nores, _ = step_nores(params, fired, x0, y0, jr.key(42))
        # the fire consumed the layer-1 budget in both arms
        assert float(s_reset.cbp.accumulator[0]) < 1.0
        assert float(s_nores.cbp.accumulator[0]) < 1.0
        # reset arm: recycled unit-0 slices zeroed (count restarts bias correction)
        np.testing.assert_allclose(np.asarray(s_reset.count["w1"][:, 0]), 0.0)
        np.testing.assert_allclose(np.asarray(s_reset.m["w1"][:, 0]), 0.0)
        np.testing.assert_allclose(np.asarray(s_reset.v["w1"][:, 0]), 0.0)
        # noreset arm: stale moments/counts carried through the replacement
        np.testing.assert_allclose(np.asarray(s_nores.count["w1"][:, 0]), 8.0)
        assert np.all(np.asarray(s_nores.v["w1"][:, 0]) > 0.0)
        # untouched units identical across the two arms
        np.testing.assert_array_equal(
            np.asarray(s_reset.m["w1"][:, 1]), np.asarray(s_nores.m["w1"][:, 1])
        )

    def test_differs_from_adamw_cbp_when_recycling(self, small_data):
        """With recycling active the reset/noreset trajectories separate."""
        x, y = small_data
        overrides = {"cbp_replacement_rate": 0.2, "cbp_maturity_threshold": 5.0}
        specs = {}
        for name, factory in (
            ("adamw_cbp", _make_adamw_cbp_learner),
            ("adamw_cbp_noreset", _make_adamw_cbp_noreset_learner),
        ):
            hp = dict(screening_spec(name).hyperparameters)
            hp.update(overrides)
            specs[name] = ScreeningSpec(
                name=name, base_learner="adamw", mechanism="dormant_unit_recycling",
                hyperparameters=hp, factory=factory,
            )
        reset = run_screening_config(x, y, specs["adamw_cbp"], seed=2, config=SMALL)
        nores = run_screening_config(
            x, y, specs["adamw_cbp_noreset"], seed=2, config=SMALL
        )
        assert not np.array_equal(reset.per_task_loss, nores.per_task_loss)


class TestAdamCBPEMANorm:
    """Composition arm: the adamw_cbp leader behind upgd_ema_norm's normalizer."""

    def test_registry_config(self):
        spec = screening_spec("adamw_cbp_ema_norm")
        base = screening_spec("adamw_cbp")
        norm = screening_spec("upgd_ema_norm")
        assert spec.base_learner == "adamw"
        assert spec.noise_update is None
        # exact adamw_cbp optimizer/CBP hyperparameters, unchanged
        for key, value in base.hyperparameters.items():
            assert spec.hyperparameters[key] == value, key
        # exact upgd_ema_norm normalizer hyperparameters, unchanged
        assert spec.hyperparameters["norm_decay"] == norm.hyperparameters["norm_decay"]
        assert (
            spec.hyperparameters["norm_epsilon"] == norm.hyperparameters["norm_epsilon"]
        )
        assert spec.hyperparameters["norm_enabled"] == 1.0

    def test_norm_disabled_reduces_to_adamw_cbp_bitwise(self, small_data):
        """norm_enabled=0: the whole trajectory equals adamw_cbp bit-for-bit."""
        x, y = small_data
        hp = dict(screening_spec("adamw_cbp_ema_norm").hyperparameters)
        hp["norm_enabled"] = 0.0
        spec = ScreeningSpec(
            name="adamw_cbp",  # reuse registry identity for shard plumbing
            base_learner="adamw",
            mechanism="input_normalization_recycling",
            hyperparameters=hp,
            factory=_make_adamw_cbp_ema_norm_learner,
        )
        ours = run_screening_config(x, y, spec, seed=13, config=SMALL)
        ref = run_screening_config(
            x, y, screening_spec("adamw_cbp"), seed=13, config=SMALL
        )
        np.testing.assert_array_equal(ours.per_task_accuracy, ref.per_task_accuracy)
        np.testing.assert_array_equal(ours.per_task_loss, ref.per_task_loss)
        np.testing.assert_array_equal(ours.per_task_plasticity, ref.per_task_plasticity)

    def test_normalizer_outputs_match_upgd_ema_norm(self):
        """One shared observation stream through both arms' registry
        (decay, eps): bitwise-equal normalized outputs and normalizer states
        at every step."""
        ours_hp = screening_spec("adamw_cbp_ema_norm").hyperparameters
        ref_hp = screening_spec("upgd_ema_norm").hyperparameters
        s_ours = EMANormState(mean=jnp.zeros(6), var=jnp.ones(6), count=jnp.array(0.0))
        s_ref = EMANormState(mean=jnp.zeros(6), var=jnp.ones(6), count=jnp.array(0.0))
        key = jr.key(29)
        for i in range(25):
            obs = jr.normal(jr.fold_in(key, i), (6,)) * 4.0 - 2.0
            got, s_ours = ema_normalize(
                s_ours, obs, ours_hp["norm_decay"], ours_hp["norm_epsilon"]
            )
            want, s_ref = ema_normalize(
                s_ref, obs, ref_hp["norm_decay"], ref_hp["norm_epsilon"]
            )
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        for field in ("mean", "var", "count"):
            np.testing.assert_array_equal(
                np.asarray(getattr(s_ours, field)), np.asarray(getattr(s_ref, field))
            )

    def test_full_step_threads_identical_normalizer_state(self, small_data):
        """Both arms' full steps on the same (x, y) stream keep the threaded
        normalizer states bit-identical even as their params diverge."""
        x, y = small_data
        params = init_mlp_params(jr.key(0), SMALL)
        spec_ours = screening_spec("adamw_cbp_ema_norm")
        spec_ref = screening_spec("upgd_ema_norm")
        init_ours, step_ours = spec_ours.factory(spec_ours.hyperparameters)
        init_ref, step_ref = spec_ref.factory(spec_ref.hyperparameters)
        s_ours = init_ours(params)
        s_ref = init_ref(params)
        p_ours = p_ref = params
        for i in range(5):
            xi = jnp.asarray(x[i], jnp.float32)
            yi = jnp.asarray(y[i], jnp.int32)
            p_ours, s_ours, _ = step_ours(p_ours, s_ours, xi, yi, jr.key(100 + i))
            p_ref, s_ref, _ = step_ref(p_ref, s_ref, xi, yi, jr.key(100 + i))
            for field in ("mean", "var", "count"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(s_ours.norm, field)),
                    np.asarray(getattr(s_ref.norm, field)),
                )
        assert float(s_ours.norm.count) == 5.0

    def test_normalization_changes_the_trajectory(self, small_data):
        """Sanity: with normalization enabled the trajectory separates from
        adamw_cbp's."""
        x, y = small_data
        ours = run_screening_config(
            x, y, screening_spec("adamw_cbp_ema_norm"), seed=13, config=SMALL
        )
        ref = run_screening_config(
            x, y, screening_spec("adamw_cbp"), seed=13, config=SMALL
        )
        assert not np.array_equal(ours.per_task_loss, ref.per_task_loss)


class TestUPGDSigma0:
    """Perturbation dissection: lean UPGD-W with sigma=0."""

    def test_registry_sigma_zero(self):
        spec = screening_spec("upgd_w_sigma0")
        assert spec.hyperparameters["noise_std"] == 0.0
        assert spec.base_learner == "upgd_w"
        # everything else stays at the published UPGD-W configuration
        for k in ("step_size", "utility_decay", "weight_decay"):
            assert spec.hyperparameters[k] == UPGD_W_PROTOCOL_HYPERPARAMETERS[k]

    def test_matches_control_factory_at_sigma_zero_bitwise(self, small_data):
        """Skipping the noise draw == drawing and scaling by zero, bit-for-bit."""
        x, y = small_data
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp["noise_std"] = 0.0
        ref_spec = ScreeningSpec(
            name="upgd_w_control",
            base_learner="upgd_w",
            mechanism="control",
            hyperparameters=hp,
            factory=screening_spec("upgd_w_control").factory,
        )
        ours = run_screening_config(
            x, y, screening_spec("upgd_w_sigma0"), seed=5, config=SMALL
        )
        ref = run_screening_config(x, y, ref_spec, seed=5, config=SMALL)
        np.testing.assert_array_equal(ours.per_task_accuracy, ref.per_task_accuracy)
        np.testing.assert_array_equal(ours.per_task_loss, ref.per_task_loss)

    def test_sigma0_differs_from_published_control(self, small_data):
        """Sanity: removing the perturbation changes the trajectory."""
        x, y = small_data
        ours = run_screening_config(
            x, y, screening_spec("upgd_w_sigma0"), seed=5, config=SMALL
        )
        control = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=5, config=SMALL
        )
        assert not np.array_equal(ours.per_task_loss, control.per_task_loss)


class TestSGDEMANorm:
    """Gate ablation: plain SGD + decoupled decay behind the exact
    upgd_ema_norm EMA input normalizer — no utility, no gate, no noise."""

    def test_registry_config(self):
        spec = screening_spec("sgd_ema_norm")
        norm = screening_spec("upgd_ema_norm")
        assert spec.mechanism == "input_normalization"
        assert spec.noise_update is None
        assert spec.factory is _make_sgd_ema_norm_learner
        # exact upgd_ema_norm non-noise optimizer + normalizer values
        assert spec.hyperparameters == {
            "step_size": norm.hyperparameters["step_size"],
            "weight_decay": norm.hyperparameters["weight_decay"],
            "norm_decay": norm.hyperparameters["norm_decay"],
            "norm_epsilon": norm.hyperparameters["norm_epsilon"],
        }
        assert spec.hyperparameters["step_size"] == 0.01
        assert spec.hyperparameters["weight_decay"] == 0.01
        # no utility, no gate, no noise — not even as inert hyperparameters
        assert "utility_decay" not in spec.hyperparameters
        assert "noise_std" not in spec.hyperparameters

    def test_reduction_pin_hand_computed_sgd_decay(self):
        """The full step equals a hand-computed normalize -> grad ->
        ``w * (1 - lr*wd) - lr*grad`` trajectory, bit-for-bit."""
        hp = screening_spec("sgd_ema_norm").hyperparameters
        init_fn, step_fn = _make_sgd_ema_norm_learner(hp)
        params = init_mlp_params(jr.key(3), SMALL)
        state = init_fn(params)
        ref_params = params
        norm_state = EMANormState(
            mean=jnp.zeros(SMALL.input_dim),
            var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        key = jr.key(21)
        for i in range(4):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,)) * 2.0 + 0.5
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(1000 + i))
            x_norm, norm_state = ema_normalize(
                norm_state, x, hp["norm_decay"], hp["norm_epsilon"]
            )
            _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                ref_params, x_norm, y
            )
            ref_params = {
                n: ref_params[n] * decay - hp["step_size"] * grads[n]
                for n in ref_params
            }
            for n in ref_params:
                np.testing.assert_array_equal(
                    np.asarray(params[n]), np.asarray(ref_params[n])
                )

    def test_key_is_unused(self):
        """No noise: stepping with different RNG keys is bit-identical."""
        hp = screening_spec("sgd_ema_norm").hyperparameters
        init_fn, step_fn = _make_sgd_ema_norm_learner(hp)
        params = init_mlp_params(jr.key(7), SMALL)
        state = init_fn(params)
        x = jr.normal(jr.key(40), (SMALL.input_dim,))
        y = jnp.array(2, jnp.int32)
        p_a, s_a, _ = step_fn(params, state, x, y, jr.key(0))
        p_b, s_b, _ = step_fn(params, state, x, y, jr.key(123456))
        for n in params:
            np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(p_b[n]))
        for field in ("mean", "var", "count"):
            np.testing.assert_array_equal(
                np.asarray(getattr(s_a.norm, field)),
                np.asarray(getattr(s_b.norm, field)),
            )

    def test_full_step_threads_identical_normalizer_state(self, small_data):
        """Both arms' full steps on the same (x, y) stream keep the threaded
        normalizer states bit-identical even as their params diverge."""
        x, y = small_data
        params = init_mlp_params(jr.key(0), SMALL)
        spec_ours = screening_spec("sgd_ema_norm")
        spec_ref = screening_spec("upgd_ema_norm")
        init_ours, step_ours = spec_ours.factory(spec_ours.hyperparameters)
        init_ref, step_ref = spec_ref.factory(spec_ref.hyperparameters)
        s_ours = init_ours(params)
        s_ref = init_ref(params)
        p_ours = p_ref = params
        for i in range(5):
            xi = jnp.asarray(x[i], jnp.float32)
            yi = jnp.asarray(y[i], jnp.int32)
            p_ours, s_ours, _ = step_ours(p_ours, s_ours, xi, yi, jr.key(100 + i))
            p_ref, s_ref, _ = step_ref(p_ref, s_ref, xi, yi, jr.key(100 + i))
            for field in ("mean", "var", "count"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(s_ours.norm, field)),
                    np.asarray(getattr(s_ref.norm, field)),
                )
        assert float(s_ours.norm.count) == 5.0

    def test_gate_removal_changes_trajectory(self, small_data):
        """Sanity: dropping the utility gate separates the trajectory from
        upgd_ema_norm_sigma0 (same normalizer, same lr/wd, no noise)."""
        x, y = small_data
        ours = run_screening_config(
            x, y, screening_spec("sgd_ema_norm"), seed=11, config=SMALL
        )
        ref = run_screening_config(
            x, y, screening_spec("upgd_ema_norm_sigma0"), seed=11, config=SMALL
        )
        assert not np.array_equal(ours.per_task_loss, ref.per_task_loss)


class TestAlphaUtility:
    """UPGD-W protection gate driven by passive IDBD step-size drift."""

    def _hp(self, **overrides):
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 1e-2, "initial_step_size": 0.01})
        hp.update(overrides)
        return hp

    def test_registry_config(self):
        spec = screening_spec("upgd_alpha_utility")
        assert spec.hyperparameters["meta_step_size"] == 1e-2
        assert spec.hyperparameters["initial_step_size"] == 0.01
        assert spec.hyperparameters["step_size"] == 0.01

    def test_meta_zero_reduces_to_half_gated_step(self):
        """meta=0: log-alphas never leave init, so the gate is exactly 0.5 and
        the update is the closed-form half-gated UPGD-W step, bit-for-bit."""
        hp = self._hp(meta_step_size=0.0)
        params = init_mlp_params(jr.key(3), SMALL)
        init_fn, _ = _make_upgd_alpha_utility_learner(hp)
        state = init_fn(params)
        decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        for step in range(3):
            kg, kn = jr.split(jr.key(70 + step))
            grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                     for i, (n, v) in enumerate(sorted(params.items()))}
            noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                     for i, (n, v) in enumerate(sorted(params.items()))}
            new_params, state = upgd_alpha_utility_update(
                params, state, grads, noise, hp
            )
            for n in params:
                expected = params[n] * decay - hp["step_size"] * (
                    (grads[n] + noise[n]) * 0.5
                )
                np.testing.assert_array_equal(
                    np.asarray(new_params[n]), np.asarray(expected)
                )
            params = new_params

    def test_nonfinite_raw_gradient_does_not_poison_log_alpha(self):
        params = init_mlp_params(jr.key(0), SMALL)
        hp = self._hp()
        init_fn, _ = _make_upgd_alpha_utility_learner(hp)
        state = init_fn(params)
        grads = {name: jnp.zeros_like(value) for name, value in params.items()}
        grads["w1"] = jnp.full_like(params["w1"], jnp.inf)
        noise = {name: jnp.zeros_like(value) for name, value in params.items()}

        _, guarded = upgd_alpha_utility_update(params, state, grads, noise, hp)

        np.testing.assert_array_equal(guarded.log_alpha["w1"], state.log_alpha["w1"])
        assert bool(jnp.all(jnp.isfinite(guarded.log_alpha["w1"])))

    def test_consistent_gradient_earns_more_protection(self):
        """A weight with a persistent-sign gradient must end with higher
        log-alpha (more protection => smaller applied delta) than a weight
        whose gradient alternates sign every step."""
        hp = self._hp(weight_decay=0.0, noise_std=0.0)
        params = {"w": jnp.array([0.5, 0.5], jnp.float32)}
        init_fn, _ = _make_upgd_alpha_utility_learner(hp)
        state = init_fn(params)
        zeros = {"w": jnp.zeros(2, jnp.float32)}
        for step in range(60):
            sign = 1.0 if step % 2 == 0 else -1.0
            grads = {"w": jnp.array([0.2, 0.2 * sign], jnp.float32)}
            params, state = upgd_alpha_utility_update(
                params, state, grads, zeros, hp
            )
        la = np.asarray(state.log_alpha["w"])
        la0 = math.log(hp["initial_step_size"])
        assert la[0] > la0 > la[1]
        # the protected weight takes the smaller applied step
        grads = {"w": jnp.array([0.2, 0.2], jnp.float32)}
        new_params, _ = upgd_alpha_utility_update(params, state, grads, zeros, hp)
        delta = np.abs(np.asarray(new_params["w"] - params["w"]))
        assert delta[0] < delta[1]

    def test_bounds_and_finiteness(self):
        hp = self._hp(meta_step_size=10.0)
        params = init_mlp_params(jr.key(6), SMALL)
        init_fn, _ = _make_upgd_alpha_utility_learner(hp)
        state = init_fn(params)
        zeros = {n: jnp.zeros_like(v) for n, v in params.items()}
        for step in range(50):
            kg = jr.fold_in(jr.key(8), step)
            grads = {n: jr.normal(jr.fold_in(kg, i), v.shape)
                     for i, (n, v) in enumerate(sorted(params.items()))}
            params, state = upgd_alpha_utility_update(params, state, grads, zeros, hp)
        for n in params:
            assert bool(jnp.all(state.log_alpha[n] >= -10.0)), n
            assert bool(jnp.all(state.log_alpha[n] <= 0.0)), n
            assert bool(jnp.all(jnp.isfinite(params[n]))), n
            assert bool(jnp.all(jnp.isfinite(state.trace[n]))), n


class TestAdamCBPTunedStar:
    """Axis-aligned mini-star around the untuned adamw_cbp leader."""

    def test_star_hyperparameters(self):
        base = screening_spec("adamw_cbp").hyperparameters
        star = {
            "adamw_cbp_r3e5": {"cbp_replacement_rate": 3e-5},
            "adamw_cbp_r3e4": {"cbp_replacement_rate": 3e-4},
            "adamw_cbp_m50": {"cbp_maturity_threshold": 50.0},
            "adamw_cbp_m200": {"cbp_maturity_threshold": 200.0},
        }
        for name, overrides in star.items():
            hp = screening_spec(name).hyperparameters
            assert screening_spec(name).base_learner == "adamw"
            for key, value in overrides.items():
                assert hp[key] == value, (name, key)
            for key, value in base.items():
                if key not in overrides:
                    assert hp[key] == value, (name, key)


class TestSigma0Frontier:
    """Wave-7 frontier extensions on the ``upgd_ema_norm_sigma0`` champion:
    normalizer statistics (decay/epsilon), hidden-layer RMS conditioning, and
    gate temperature/normalization under input conditioning."""

    AXES = {
        "sigma0_ndecay099": {"norm_decay": 0.99},
        "sigma0_ndecay09999": {"norm_decay": 0.9999},
        "sigma0_eps1e6": {"norm_epsilon": 1e-6},
        "sigma0_eps1e4": {"norm_epsilon": 1e-4},
        "sigma0_hidden_norm": {"hidden_rms": 1.0, "hidden_rms_epsilon": 1e-8},
        "sigma0_gate_beta05": {"gate_beta": 0.5},
        "sigma0_gate_beta2": {"gate_beta": 2.0},
        "sigma0_localgate": {"local_gate": 1.0},
    }
    EXT_DEFAULTS = {"gate_beta": 1.0, "local_gate": 0.0, "hidden_rms": 0.0}

    def test_registry_single_axis_arms(self):
        """Every arm varies exactly one axis over the sigma0 champion's
        hyperparameters plus inert extension defaults."""
        base = screening_spec("upgd_ema_norm_sigma0").hyperparameters
        for name, overrides in self.AXES.items():
            spec = screening_spec(name)
            assert spec.base_learner == "upgd_w", name
            assert spec.noise_update is None, name
            assert spec.factory is _make_upgd_ema_norm_ext_learner, name
            assert spec.hyperparameters == {
                **base,
                **self.EXT_DEFAULTS,
                **overrides,
            }, name
            assert spec.hyperparameters["noise_std"] == 0.0, name

    def test_ext_defaults_reduce_to_upgd_ema_norm_sigma0_bitwise(self, small_data):
        """gate_beta=1, local_gate=0, hidden_rms=0: the extension factory's
        whole trajectory equals upgd_ema_norm_sigma0's bit-for-bit."""
        x, y = small_data
        base = screening_spec("upgd_ema_norm_sigma0")
        spec = ScreeningSpec(
            name="upgd_ema_norm_sigma0",  # reuse registry identity for plumbing
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters={**base.hyperparameters, **self.EXT_DEFAULTS},
            factory=_make_upgd_ema_norm_ext_learner,
        )
        ours = run_screening_config(x, y, spec, seed=17, config=SMALL)
        ref = run_screening_config(x, y, base, seed=17, config=SMALL)
        np.testing.assert_array_equal(ours.per_task_accuracy, ref.per_task_accuracy)
        np.testing.assert_array_equal(ours.per_task_loss, ref.per_task_loss)
        np.testing.assert_array_equal(ours.per_task_plasticity, ref.per_task_plasticity)

    def test_ext_defaults_lower_to_identical_hlo(self):
        """At inert defaults the extension factory compiles to the same graph
        as upgd_ema_norm_sigma0's factory.  Identical lowered HLO leaves XLA
        no fusion or reassociation freedom, so derived float32 metrics cannot
        drift between the two factories on any backend — value equality alone
        does not guarantee this (issue #46)."""
        base = screening_spec("upgd_ema_norm_sigma0")
        params = init_mlp_params(jr.key(11), SMALL)
        x = jr.normal(jr.key(12), (SMALL.input_dim,))
        y = jnp.array(1, jnp.int32)
        key = jr.key(13)

        def lowered_text(factory, hp):
            init_fn, step_fn = factory(hp)
            jitted = jax.jit(lambda p, s, xx, yy, k: step_fn(p, s, xx, yy, k))
            return jitted.lower(params, init_fn(params), x, y, key).as_text()

        ref = lowered_text(_make_upgd_ema_norm_learner, base.hyperparameters)
        ext = lowered_text(
            _make_upgd_ema_norm_ext_learner,
            {**base.hyperparameters, **self.EXT_DEFAULTS},
        )
        assert ref == ext

    def test_key_is_unused_on_every_arm(self):
        """sigma0 arms consume no randomness: different RNG keys, same step."""
        params = init_mlp_params(jr.key(4), SMALL)
        x = jr.normal(jr.key(41), (SMALL.input_dim,))
        y = jnp.array(1, jnp.int32)
        for name in self.AXES:
            spec = screening_spec(name)
            init_fn, step_fn = spec.factory(spec.hyperparameters)
            state = init_fn(params)
            p_a, s_a, _ = step_fn(params, state, x, y, jr.key(0))
            p_b, s_b, _ = step_fn(params, state, x, y, jr.key(987654))
            for n in params:
                np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(p_b[n]), name)
            for field in ("mean", "var", "count"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(s_a.norm, field)),
                    np.asarray(getattr(s_b.norm, field)),
                    name,
                )

    def test_gate_temperature_hand_computed(self):
        """sigma0_gate_beta2 equals normalize -> grads -> utility EMA ->
        sigmoid(2 * scaled utility) -> gated decayed step, bit-for-bit."""
        hp = screening_spec("sigma0_gate_beta2").hyperparameters
        init_fn, step_fn = _make_upgd_ema_norm_ext_learner(hp)
        params = init_mlp_params(jr.key(9), SMALL)
        state = init_fn(params)
        ref_params = params
        ref_utility = {n: jnp.zeros_like(v) for n, v in params.items()}
        norm_state = EMANormState(
            mean=jnp.zeros(SMALL.input_dim),
            var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        beta = hp["utility_decay"]
        decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        for i in range(3):
            x = jr.normal(jr.fold_in(jr.key(50), i), (SMALL.input_dim,)) * 1.5
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(1000 + i))
            x_norm, norm_state = ema_normalize(
                norm_state, x, hp["norm_decay"], hp["norm_epsilon"]
            )
            _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                ref_params, x_norm, y
            )
            ref_utility = {
                n: beta * ref_utility[n] + (1.0 - beta) * (-grads[n] * ref_params[n])
                for n in ref_params
            }
            global_max = jnp.max(
                jnp.stack([jnp.max(ref_utility[n]) for n in sorted(ref_params)])
            )
            bias_correction = 1.0 - jnp.power(
                jnp.asarray(beta, jnp.float32), jnp.asarray(i + 1, jnp.float32)
            )
            new_ref = {}
            for n in ref_params:
                gate = jax.nn.sigmoid(
                    hp["gate_beta"] * ((ref_utility[n] / bias_correction) / global_max)
                )
                new_ref[n] = ref_params[n] * decay - hp["step_size"] * (
                    grads[n] * (1.0 - gate)
                )
            ref_params = new_ref
            for n in ref_params:
                np.testing.assert_array_equal(
                    np.asarray(params[n]), np.asarray(ref_params[n])
                )

    def test_localgate_matches_localgate_update_behind_normalizer(self):
        """sigma0_localgate equals upgd_w_localgate_update (zero noise) applied
        to the EMA-normalized input, bit-for-bit, with threaded state."""
        hp = screening_spec("sigma0_localgate").hyperparameters
        init_fn, step_fn = _make_upgd_ema_norm_ext_learner(hp)
        params = init_mlp_params(jr.key(11), SMALL)
        state = init_fn(params)
        ref_params = params
        ref_state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, jnp.int32),
        )
        norm_state = EMANormState(
            mean=jnp.zeros(SMALL.input_dim),
            var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        zeros = {n: jnp.zeros_like(v) for n, v in params.items()}
        for i in range(3):
            x = jr.normal(jr.fold_in(jr.key(60), i), (SMALL.input_dim,)) * 2.0
            y = jnp.array((2 * i) % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(2000 + i))
            x_norm, norm_state = ema_normalize(
                norm_state, x, hp["norm_decay"], hp["norm_epsilon"]
            )
            _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                ref_params, x_norm, y
            )
            ref_params, ref_state = upgd_w_localgate_update(
                ref_params, ref_state, grads, zeros, hp
            )
            for n in ref_params:
                np.testing.assert_array_equal(
                    np.asarray(params[n]), np.asarray(ref_params[n])
                )
            for n in ref_params:
                np.testing.assert_array_equal(
                    np.asarray(state.utility[n]), np.asarray(ref_state.utility[n])
                )

    def test_hidden_rms_normalize_properties(self):
        """Unit RMS on nonzero vectors; an all-zero ReLU vector stays finite."""
        v = jnp.array([3.0, 4.0, 0.0, 1.0], jnp.float32)
        out = _hidden_rms_normalize(v, 1e-8)
        rms = float(jnp.sqrt(jnp.mean(out * out)))
        assert math.isclose(rms, 1.0, rel_tol=1e-5)
        zeros = jnp.zeros(4, jnp.float32)
        out_zero = _hidden_rms_normalize(zeros, 1e-8)
        assert bool(jnp.all(jnp.isfinite(out_zero)))
        np.testing.assert_array_equal(np.asarray(out_zero), np.zeros(4, np.float32))

    def test_hidden_rms_step_hand_computed(self):
        """sigma0_hidden_norm equals a manual RMS-normalized forward/backward
        plus the global-gate sigma0 update, bit-for-bit."""
        hp = screening_spec("sigma0_hidden_norm").hyperparameters
        init_fn, step_fn = _make_upgd_ema_norm_ext_learner(hp)
        params = init_mlp_params(jr.key(13), SMALL)
        state = init_fn(params)
        rms_eps = hp["hidden_rms_epsilon"]

        def manual_loss(p, x, y):
            z1 = x @ p["w1"] + p["b1"]
            a1 = jax.nn.relu(z1)
            h1 = a1 / jnp.sqrt(jnp.mean(a1 * a1) + rms_eps)
            z2 = h1 @ p["w2"] + p["b2"]
            a2 = jax.nn.relu(z2)
            h2 = a2 / jnp.sqrt(jnp.mean(a2 * a2) + rms_eps)
            logits = h2 @ p["w3"] + p["b3"]
            return -jax.nn.log_softmax(logits)[y], logits

        ref_params = params
        ref_utility = {n: jnp.zeros_like(v) for n, v in params.items()}
        norm_state = EMANormState(
            mean=jnp.zeros(SMALL.input_dim),
            var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        beta = hp["utility_decay"]
        decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        for i in range(3):
            x = jr.normal(jr.fold_in(jr.key(70), i), (SMALL.input_dim,)) + 0.25
            y = jnp.array((i + 1) % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(3000 + i))
            x_norm, norm_state = ema_normalize(
                norm_state, x, hp["norm_decay"], hp["norm_epsilon"]
            )
            _, grads = jax.value_and_grad(manual_loss, has_aux=True)(
                ref_params, x_norm, y
            )
            ref_utility = {
                n: beta * ref_utility[n] + (1.0 - beta) * (-grads[n] * ref_params[n])
                for n in ref_params
            }
            global_max = jnp.max(
                jnp.stack([jnp.max(ref_utility[n]) for n in sorted(ref_params)])
            )
            bias_correction = 1.0 - jnp.power(
                jnp.asarray(beta, jnp.float32), jnp.asarray(i + 1, jnp.float32)
            )
            new_ref = {}
            for n in ref_params:
                gate = jax.nn.sigmoid((ref_utility[n] / bias_correction) / global_max)
                new_ref[n] = ref_params[n] * decay - hp["step_size"] * (
                    grads[n] * (1.0 - gate)
                )
            ref_params = new_ref
            for n in ref_params:
                np.testing.assert_array_equal(
                    np.asarray(params[n]), np.asarray(ref_params[n])
                )

    def test_hidden_norm_frozen_probe_rejected(self):
        """The plain-MLP sentinel probe cannot describe the RMS-normalized
        forward pass; the arm must refuse instead of probing the wrong model."""
        spec = screening_spec("sigma0_hidden_norm")
        with pytest.raises(NotImplementedError, match="hidden"):
            spec.frozen_probe_input(
                None, jnp.zeros((2, SMALL.input_dim)), spec.hyperparameters
            )

    def test_each_axis_changes_the_trajectory(self, small_data):
        """Sanity against silently-ignored hyperparameters: every arm's
        trajectory separates from the sigma0 champion's."""
        x, y = small_data
        ref = run_screening_config(
            x, y, screening_spec("upgd_ema_norm_sigma0"), seed=9, config=SMALL
        )
        for name in self.AXES:
            if name.startswith("sigma0_ndecay"):
                continue  # inert during the effective-decay warmup; below
            ours = run_screening_config(
                x, y, screening_spec(name), seed=9, config=SMALL
            )
            assert not np.array_equal(ours.per_task_loss, ref.per_task_loss), name

    def test_norm_decay_changes_the_trajectory_past_warmup(self):
        """``ema_normalize`` clamps the decay to ``1 - 1/(count+1)`` during
        warmup, so 0.99/0.999/0.9999 coincide for the first ~100/1000 steps;
        past 1,000 steps each ndecay arm must separate from the champion."""
        key = jr.key(4321)
        kx, ky = jr.split(key)
        x = np.asarray(jr.uniform(kx, (700, SMALL.input_dim), jnp.float32, -1.0, 1.0))
        y = np.asarray(jr.randint(ky, (700,), 0, SMALL.n_classes))
        config = IPMNISTConfig(
            n_tasks=3, task_length=600, input_dim=12, hidden1=8, hidden2=6, n_classes=5
        )
        ref = run_screening_config(
            x, y, screening_spec("upgd_ema_norm_sigma0"), seed=9, config=config
        )
        for name in ("sigma0_ndecay099", "sigma0_ndecay09999"):
            ours = run_screening_config(
                x, y, screening_spec(name), seed=9, config=config
            )
            assert not np.array_equal(ours.per_task_loss, ref.per_task_loss), name


class TestUpdateRuleFamily:
    """Wave-8 update-rule family swaps under the sigma0_ndecay099 champion's
    conditioning: colnorm_gate, muon_gate, lion_gate."""

    NAMES = ("colnorm_gate", "muon_gate", "lion_gate")
    CHAMPION = {"norm_decay": 0.99, "norm_epsilon": 1e-8, "noise_std": 0.0}
    FACTORIES = {
        "colnorm_gate": _make_colnorm_gate_learner,
        "muon_gate": _make_muon_gate_learner,
        "lion_gate": _make_lion_gate_learner,
    }

    def _learnable_stream(self, n_examples=400, seed=88):
        """Deterministically learnable labels (argmax of a fixed linear map)
        so a short smoke run can rise above chance."""
        kx, kw = jr.split(jr.key(seed))
        x = jr.uniform(kx, (n_examples, SMALL.input_dim), jnp.float32, -1.0, 1.0)
        w_true = jr.normal(kw, (SMALL.input_dim, SMALL.n_classes), jnp.float32)
        y = jnp.argmax(x @ w_true, axis=1).astype(jnp.int32)
        return np.asarray(x), np.asarray(y)

    def test_registry_arms(self):
        """Each arm carries the champion's conditioning (norm decay 0.99,
        sigma=0) plus only its own update-rule constants."""
        expected = {
            # step sizes re-calibrated after the lr-transfer failure: the
            # champion's raw-gradient lr 0.01 is ~10-100x too large for
            # normalized/orthogonalized/sign updates (all three scored chance
            # at 0.01; 2-task sweeps picked the values below).
            "colnorm_gate": {
                "step_size": 0.001, "col_decay": 0.99, "col_epsilon": 1e-8
            },
            "muon_gate": {
                "step_size": 0.003, "muon_momentum": 0.95, "muon_ns_steps": 5.0
            },
            "lion_gate": {
                "step_size": 0.0001,
                "weight_decay": 0.05,
                "lion_beta1": 0.9,
                "lion_beta2": 0.99,
            },
        }
        for name, extras in expected.items():
            spec = screening_spec(name)
            assert spec.base_learner == "upgd_w", name
            assert spec.mechanism == "update_rule_family", name
            assert spec.noise_update is None, name
            assert spec.factory is self.FACTORIES[name], name
            assert spec.frozen_probe_input is _ema_frozen_probe_input, name
            assert spec.hyperparameters == {
                **UPGD_W_PROTOCOL_HYPERPARAMETERS,
                **self.CHAMPION,
                **extras,
            }, name
            # the champion's utility gate is unchanged
            assert spec.hyperparameters["utility_decay"] == 0.9999, name

    def test_colnorm_vcol_has_fan_in_dimension(self):
        """v_col is per input dimension (axis 0 of the (fan_in, fan_out)
        weights); biases keep their full per-element shape."""
        spec = screening_spec("colnorm_gate")
        params = init_mlp_params(jr.key(5), SMALL)
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        state = init_fn(params)
        assert state.vcol["w1"].shape == (SMALL.input_dim,)
        assert state.vcol["w2"].shape == (SMALL.hidden1,)
        assert state.vcol["w3"].shape == (SMALL.hidden2,)
        assert state.vcol["b1"].shape == params["b1"].shape
        assert state.vcol["b2"].shape == params["b2"].shape
        assert state.vcol["b3"].shape == params["b3"].shape
        x = jr.normal(jr.key(51), (SMALL.input_dim,))
        _, new_state, _ = step_fn(params, state, x, jnp.array(2, jnp.int32), jr.key(0))
        for name in state.vcol:
            assert new_state.vcol[name].shape == state.vcol[name].shape, name
            assert bool(jnp.all(jnp.isfinite(new_state.vcol[name]))), name

    def test_key_is_unused_on_every_arm(self):
        """No perturbation: different RNG keys give the identical step."""
        params = init_mlp_params(jr.key(14), SMALL)
        x = jr.normal(jr.key(42), (SMALL.input_dim,))
        y = jnp.array(1, jnp.int32)
        for name in self.NAMES:
            spec = screening_spec(name)
            init_fn, step_fn = spec.factory(spec.hyperparameters)
            state = init_fn(params)
            p_a, _, _ = step_fn(params, state, x, y, jr.key(0))
            p_b, _, _ = step_fn(params, state, x, y, jr.key(987654))
            for n in params:
                np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(p_b[n]), name)

    def test_newton_schulz_orthogonalizes(self):
        """Scaled identity input lands in the Muon singular-value band around
        1 with near-zero off-diagonals, and the Frobenius normalization makes
        the output scale-invariant."""
        eye = 0.5 * jnp.eye(4, dtype=jnp.float32)
        out = _newton_schulz_orthogonalize(eye, 5)
        diag = np.diag(np.asarray(out))
        assert np.all(diag > 0.6) and np.all(diag < 1.3)
        off = np.asarray(out) - np.diag(diag)
        np.testing.assert_allclose(off, np.zeros_like(off), atol=1e-5)
        m = jr.normal(jr.key(3), (4, 6), jnp.float32)
        np.testing.assert_allclose(
            np.asarray(_newton_schulz_orthogonalize(m, 5)),
            np.asarray(_newton_schulz_orthogonalize(3.0 * m, 5)),
            rtol=1e-4,
            atol=1e-5,
        )
        # a tall matrix is handled through the transposed branch
        tall = _newton_schulz_orthogonalize(jr.normal(jr.key(7), (6, 3)), 5)
        assert tall.shape == (6, 3)
        assert bool(jnp.all(jnp.isfinite(tall)))

    def test_smoke_runs_above_chance(self):
        """2 tasks x 200 steps on a learnable stream: finite metrics, online
        accuracy above the 1/n_classes chance floor for every arm."""
        x, y = self._learnable_stream()
        config = IPMNISTConfig(
            n_tasks=2, task_length=200, input_dim=12, hidden1=8, hidden2=6, n_classes=5
        )
        for name in self.NAMES:
            result = run_screening_config(
                x, y, screening_spec(name), seed=2, config=config
            )
            acc = np.asarray(result.per_task_accuracy)
            assert np.all(np.isfinite(acc)), name
            assert np.all(np.isfinite(np.asarray(result.per_task_loss))), name
            assert float(acc.mean()) > 1.0 / config.n_classes, name


class TestAdamElemStep:
    def test_update_composes_step(self):
        """adam_elem_update == param - adam_elem_step delta, same moments."""
        hp = {"step_size": 1e-4, "beta1": 0.0, "beta2": 0.99, "eps": 1e-8,
              "weight_decay": 0.01}
        key = jr.key(12)
        param = jr.normal(jr.fold_in(key, 0), (4, 3))
        grad = jr.normal(jr.fold_in(key, 1), (4, 3))
        m = v = jnp.zeros((4, 3))
        count = jnp.zeros((4, 3))
        delta, m1, v1, c1 = adam_elem_step(param, m, v, count, grad, hp)
        p2, m2, v2, c2 = adam_elem_update(param, m, v, count, grad, hp)
        np.testing.assert_array_equal(np.asarray(param - delta), np.asarray(p2))
        np.testing.assert_array_equal(np.asarray(m1), np.asarray(m2))
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v2))
        np.testing.assert_array_equal(np.asarray(c1), np.asarray(c2))


class TestSpecShape:
    def test_specs_are_json_serializable(self):
        for spec in SCREENING_REGISTRY.values():
            assert isinstance(spec, ScreeningSpec)
            json.dumps(spec.hyperparameters)
            assert spec.base_learner in ("upgd_w", "adamw")

    def test_noise_update_present_exactly_on_lean_family_arms(self):
        with_pool = {
            name for name, spec in SCREENING_REGISTRY.items()
            if spec.noise_update is not None
        }
        assert with_pool == {
            "upgd_w_control",
            "upgd_w_wclip_k1",
            "upgd_w_wclip_k2",
            "upgd_w_wclip_k1_wd0",
            "upgd_w_wclip_k2_wd0",
            "upgd_w_localgate",
            "upgd_w_sigma005",
            "upgd_w_sigma02",
            "upgd_w_udecay0999",
            "upgd_w_udecay099999",
            "upgd_w_wd0005",
            "upgd_w_wd002",
        }


class TestShiftNorm:
    """Next-rung wave: per-feature shift-triggered re-conditioning normalizer.

    ``shift_adaptive_normalize`` keeps the champion's slow EMA statistics but
    tracks a fast per-feature mean; when the fast mean diverges from the slow
    mean beyond ``shift_k * std + shift_delta`` the feature's anneal count is
    reset, so its effective decay drops (fast re-conditioning) and then
    anneals back toward the slow ``norm_decay``.
    """

    HP = {
        "fast_decay": 0.9,
        "shift_k": 1.0,
        "shift_delta": 0.02,
        "shift_refractory": 0.0,
    }

    def test_registry_configs(self):
        base = screening_spec("upgd_ema_norm_sigma0").hyperparameters
        expected = {
            "sigma0_shiftnorm": {"norm_decay": 0.999, **self.HP},
            "sigma0_shiftnorm_k05": {"norm_decay": 0.999, **self.HP, "shift_k": 0.5},
            "sigma0_shiftnorm_d099": {"norm_decay": 0.99, **self.HP},
            # d099 detector mini-star: detector sensitivity (shift_k), detector
            # speed (fast_decay), trigger rate-limiting (shift_refractory), and
            # the d098 base (the frontier-2 decay plateau ties 0.98/0.99).
            "sigma0_shiftnorm_d099_k05": {
                "norm_decay": 0.99, **self.HP, "shift_k": 0.5
            },
            "sigma0_shiftnorm_d099_k2": {
                "norm_decay": 0.99, **self.HP, "shift_k": 2.0
            },
            "sigma0_shiftnorm_d098": {"norm_decay": 0.98, **self.HP},
            "sigma0_shiftnorm_d099_f08": {
                "norm_decay": 0.99, **self.HP, "fast_decay": 0.8
            },
            "sigma0_shiftnorm_d099_f095": {
                "norm_decay": 0.99, **self.HP, "fast_decay": 0.95
            },
            "sigma0_shiftnorm_d099_r200": {
                "norm_decay": 0.99, **self.HP, "shift_refractory": 200.0
            },
        }
        ext_defaults = {"gate_beta": 1.0, "local_gate": 0.0, "hidden_rms": 0.0}
        for name, overrides in expected.items():
            spec = screening_spec(name)
            assert spec.base_learner == "upgd_w", name
            assert spec.noise_update is None, name
            assert spec.factory is _make_upgd_shiftnorm_learner, name
            assert spec.frozen_probe_input is _ema_frozen_probe_input, name
            assert spec.hyperparameters == {**base, **ext_defaults, **overrides}, name
            assert spec.hyperparameters["noise_std"] == 0.0, name

    def test_no_trigger_reduces_to_ema_normalize_bitwise(self):
        """With an untriggerable threshold the chain equals ema_normalize."""
        d = 5
        key = jr.key(11)
        xs = jr.uniform(key, (40, d), jnp.float32, -1.0, 1.0)
        plain = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.array(0.0))
        shift = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.zeros(d))
        fast = jnp.zeros(d)
        for t in range(40):
            n_plain, plain = ema_normalize(plain, xs[t], 0.999, 1e-8)
            n_shift, shift, fast, mask = shift_adaptive_normalize(
                shift, fast, xs[t], decay=0.999, fast_decay=0.9, epsilon=1e-8,
                shift_k=1e9, shift_delta=1e9,
            )
            assert not bool(jnp.any(mask))
            np.testing.assert_array_equal(np.asarray(n_plain), np.asarray(n_shift))
            np.testing.assert_array_equal(np.asarray(plain.mean), np.asarray(shift.mean))
            np.testing.assert_array_equal(np.asarray(plain.var), np.asarray(shift.var))
            np.testing.assert_array_equal(
                np.full(d, float(plain.count)), np.asarray(shift.count)
            )

    @staticmethod
    def _run_constant_then_shift(
        n_pre: int, v_pre: np.ndarray, v_post: np.ndarray, n_post: int,
        **kwargs: float,
    ):
        d = v_pre.shape[0]
        state = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.zeros(d))
        fast = jnp.zeros(d)
        masks = []
        for _ in range(n_pre):
            _, state, fast, _ = shift_adaptive_normalize(
                state, fast, jnp.asarray(v_pre, jnp.float32), **kwargs
            )
        for _ in range(n_post):
            _, state, fast, mask = shift_adaptive_normalize(
                state, fast, jnp.asarray(v_post, jnp.float32), **kwargs
            )
            masks.append(np.asarray(mask))
        return state, fast, masks

    def test_shift_resets_only_the_shifted_feature(self):
        v_pre = np.array([0.0, 0.5, -0.3, 1.0], np.float32)
        v_post = v_pre.copy()
        v_post[0] = 2.0
        state, _, masks = self._run_constant_then_shift(
            200, v_pre, v_post, 1,
            decay=0.999, fast_decay=0.9, epsilon=1e-8, shift_k=1.0, shift_delta=0.02,
        )
        assert masks[0][0], "shifted feature must trigger detection"
        assert not masks[0][1:].any(), "unshifted features must not trigger"
        count = np.asarray(state.count)
        assert count[0] == 1.0, "triggered feature's anneal count resets"
        np.testing.assert_array_equal(count[1:], np.full(3, 201.0))

    def test_post_shift_reconditioning_is_faster_than_slow_ema(self):
        v_pre = np.array([0.0, 0.5, -0.3, 1.0], np.float32)
        v_post = v_pre.copy()
        v_post[0] = 2.0
        state, _, _ = self._run_constant_then_shift(
            200, v_pre, v_post, 30,
            decay=0.999, fast_decay=0.9, epsilon=1e-8, shift_k=1.0, shift_delta=0.02,
        )
        plain = EMANormState(
            mean=jnp.zeros(4), var=jnp.ones(4), count=jnp.array(0.0)
        )
        for _ in range(200):
            _, plain = ema_normalize(plain, jnp.asarray(v_pre, jnp.float32), 0.999, 1e-8)
        for _ in range(30):
            _, plain = ema_normalize(plain, jnp.asarray(v_post, jnp.float32), 0.999, 1e-8)
        err_shift = abs(float(state.mean[0]) - 2.0)
        err_plain = abs(float(plain.mean[0]) - 2.0)
        assert err_shift < 0.1, f"reset feature must recondition fast, err={err_shift}"
        assert err_plain > 1.0, f"slow EMA must still lag, err={err_plain}"

    def test_refractory_zero_is_bitwise_prior_equations(self):
        """``shift_refractory=0`` (and omitting it) is bitwise the unguarded
        detector: counts are nonnegative, so the eligibility conjunct is
        identically true and no float changes."""
        d = 4
        xs = np.asarray(
            jr.uniform(jr.key(23), (120, d), jnp.float32, -0.05, 0.05)
        ).copy()
        xs[80:, 0] += 2.0  # force a mid-stream shift so triggers occur
        kw = dict(decay=0.99, fast_decay=0.9, epsilon=1e-8,
                  shift_k=1.0, shift_delta=0.02)

        def reference(state, fast_mean, observation):
            """Pre-refractory equations, verbatim."""
            effective_fast = jnp.minimum(
                kw["fast_decay"], 1.0 - 1.0 / (state.count + 2.0)
            )
            new_fast = (
                effective_fast * fast_mean + (1.0 - effective_fast) * observation
            )
            threshold = kw["shift_k"] * jnp.sqrt(state.var) + kw["shift_delta"]
            shifted = jnp.abs(new_fast - state.mean) > threshold
            new_count = jnp.where(shifted, 0.0, state.count) + 1.0
            effective_decay = jnp.minimum(
                kw["decay"], 1.0 - 1.0 / (new_count + 1.0)
            )
            delta = observation - state.mean
            new_mean = state.mean + (1.0 - effective_decay) * delta
            delta2 = observation - new_mean
            new_var = jnp.maximum(
                kw["epsilon"],
                effective_decay * state.var
                + (1.0 - effective_decay) * delta * delta2,
            )
            normalized = (observation - new_mean) / (jnp.sqrt(new_var) + kw["epsilon"])
            return normalized, EMANormState(  # type: ignore[call-arg]
                mean=new_mean, var=new_var, count=new_count
            ), new_fast, shifted

        ref = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.zeros(d))
        r0 = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.zeros(d))
        omit = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.zeros(d))
        f_ref = f_r0 = f_omit = jnp.zeros(d)
        any_trigger = False
        for t in range(120):
            x = jnp.asarray(xs[t])
            n_ref, ref, f_ref, m_ref = reference(ref, f_ref, x)
            n_r0, r0, f_r0, m_r0 = shift_adaptive_normalize(
                r0, f_r0, x, **kw, shift_refractory=0.0
            )
            n_omit, omit, f_omit, m_omit = shift_adaptive_normalize(
                omit, f_omit, x, **kw
            )
            any_trigger = any_trigger or bool(jnp.any(m_ref))
            for a, b in ((n_r0, n_ref), (n_omit, n_ref)):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            for got in (r0, omit):
                np.testing.assert_array_equal(np.asarray(got.mean), np.asarray(ref.mean))
                np.testing.assert_array_equal(np.asarray(got.var), np.asarray(ref.var))
                np.testing.assert_array_equal(
                    np.asarray(got.count), np.asarray(ref.count)
                )
            np.testing.assert_array_equal(np.asarray(m_r0), np.asarray(m_ref))
            np.testing.assert_array_equal(np.asarray(m_omit), np.asarray(m_ref))
        assert any_trigger, "stream must exercise the trigger path"

    def test_refractory_gates_immature_features_only(self):
        """A diverged feature triggers only once its anneal count has matured
        past ``shift_refractory``; mature features are unaffected."""
        state = EMANormState(
            mean=jnp.zeros(2),
            var=jnp.full((2,), 1e-6),
            count=jnp.array([5.0, 500.0]),
        )
        fast = jnp.full((2,), 3.0)  # both features far from the slow mean
        _, new_state, _, mask = shift_adaptive_normalize(
            state, fast, jnp.full((2,), 3.0),
            decay=0.99, fast_decay=0.9, epsilon=1e-8,
            shift_k=1.0, shift_delta=0.02, shift_refractory=100.0,
        )
        assert not bool(mask[0]), "immature feature (count 5 < 100) must be gated"
        assert bool(mask[1]), "mature feature (count 500 >= 100) must trigger"
        count = np.asarray(new_state.count)
        assert count[0] == 6.0, "gated feature keeps annealing"
        assert count[1] == 1.0, "triggered feature resets"

    def test_refractory_blocks_then_allows_boundary_trigger(self):
        """End-to-end: with refractory above the pre-shift count the boundary
        trigger is suppressed; below it, the trigger fires as before."""
        v_pre = np.array([0.0, 0.5, -0.3, 1.0], np.float32)
        v_post = v_pre.copy()
        v_post[0] = 2.0
        kw = dict(decay=0.99, fast_decay=0.9, epsilon=1e-8,
                  shift_k=1.0, shift_delta=0.02)
        _, _, masks_blocked = self._run_constant_then_shift(
            200, v_pre, v_post, 1, **kw, shift_refractory=300.0
        )
        assert not masks_blocked[0].any(), "count 200 < refractory 300: no trigger"
        state, _, masks_open = self._run_constant_then_shift(
            200, v_pre, v_post, 30, **kw, shift_refractory=100.0
        )
        assert masks_open[0][0], "count 200 >= refractory 100: boundary triggers"
        assert not np.stack(masks_open[1:])[:, 0].any(), (
            "after the reset the feature is immature again: no rapid re-triggering"
        )
        assert np.asarray(state.count)[0] == 30.0, (
            "post-reset count anneals monotonically under the refractory"
        )

    def test_learner_key_unused_and_state_shape(self):
        spec = screening_spec("sigma0_shiftnorm")
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        params = init_mlp_params(jr.key(4), SMALL)
        state = init_fn(params)
        assert isinstance(state, UPGDAdaptiveNormState)
        assert isinstance(state.norm, EMANormState)
        assert state.norm.count.shape == (SMALL.input_dim,)
        assert state.fast_mean.shape == (SMALL.input_dim,)
        x = jr.normal(jr.key(41), (SMALL.input_dim,))
        y = jnp.array(1, jnp.int32)
        p_a, s_a, _ = step_fn(params, state, x, y, jr.key(0))
        p_b, s_b, _ = step_fn(params, state, x, y, jr.key(987654))
        for n in params:
            np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(p_b[n]))
        np.testing.assert_array_equal(
            np.asarray(s_a.norm.mean), np.asarray(s_b.norm.mean)
        )


class TestWarmNorm:
    """Next-rung wave: globally shift-reset annealed normalizer decay.

    ``warm_restart_normalize`` detects a global distribution shift from
    fast/slow EMA divergence (never task boundaries) and resets the scalar
    anneal clock, so the effective decay warms up again from 1/2 toward
    ``norm_decay`` exactly as at stream start.
    """

    KW = {
        "decay": 0.999, "fast_decay": 0.9, "epsilon": 1e-8,
        "warm_threshold": 1.0, "warm_pad": 0.01, "warm_refractory": 50.0,
    }

    def test_registry_config(self):
        base = screening_spec("upgd_ema_norm_sigma0").hyperparameters
        spec = screening_spec("sigma0_warmnorm")
        assert spec.base_learner == "upgd_w"
        assert spec.noise_update is None
        assert spec.factory is _make_upgd_warmnorm_learner
        assert spec.frozen_probe_input is _ema_frozen_probe_input
        assert spec.hyperparameters == {
            **base,
            "gate_beta": 1.0, "local_gate": 0.0, "hidden_rms": 0.0,
            "fast_decay": 0.9, "warm_threshold": 1.0, "warm_pad": 0.01,
            "warm_refractory": 50.0,
        }

    def test_threshold_inf_reduces_to_ema_normalize_bitwise(self):
        d = 5
        xs = jr.uniform(jr.key(13), (40, d), jnp.float32, -1.0, 1.0)
        plain = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.array(0.0))
        warm = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.array(0.0))
        fast = jnp.zeros(d)
        kw = dict(self.KW, warm_threshold=float("inf"))
        for t in range(40):
            n_plain, plain = ema_normalize(plain, xs[t], 0.999, 1e-8)
            n_warm, warm, fast, trig = warm_restart_normalize(warm, fast, xs[t], **kw)
            assert not bool(trig)
            np.testing.assert_array_equal(np.asarray(n_plain), np.asarray(n_warm))
            np.testing.assert_array_equal(np.asarray(plain.mean), np.asarray(warm.mean))
            np.testing.assert_array_equal(np.asarray(plain.var), np.asarray(warm.var))
            assert float(plain.count) == float(warm.count)

    def test_global_shift_resets_clock_with_refractory(self):
        d = 6
        v_a = jnp.asarray(np.linspace(-1.0, 1.0, d), jnp.float32)
        v_b = v_a + 3.0
        state = EMANormState(mean=jnp.zeros(d), var=jnp.ones(d), count=jnp.array(0.0))
        fast = jnp.zeros(d)
        for _ in range(300):
            _, state, fast, trig = warm_restart_normalize(state, fast, v_a, **self.KW)
            assert not bool(trig), "steady within-task stream must never trigger"
        assert float(state.count) == 300.0
        counts = []
        for _ in range(50):
            _, state, fast, trig = warm_restart_normalize(state, fast, v_b, **self.KW)
            counts.append(float(state.count))
        assert counts[0] == 1.0, "detected global shift must reset the anneal clock"
        np.testing.assert_array_equal(
            np.asarray(counts), np.arange(1.0, 51.0)
        ), "refractory must block re-triggering while the clock re-anneals"
        err_warm = float(jnp.max(jnp.abs(state.mean - v_b)))
        assert err_warm < 0.1, f"warm restart must recondition fast, err={err_warm}"

    def test_learner_smoke_state_shape(self):
        spec = screening_spec("sigma0_warmnorm")
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        params = init_mlp_params(jr.key(4), SMALL)
        state = init_fn(params)
        assert isinstance(state, UPGDAdaptiveNormState)
        assert state.norm.count.shape == ()
        assert state.fast_mean.shape == (SMALL.input_dim,)
        x = jr.normal(jr.key(42), (SMALL.input_dim,))
        y = jnp.array(3, jnp.int32)
        p_a, s_a, _ = step_fn(params, state, x, y, jr.key(0))
        p_b, s_b, _ = step_fn(params, state, x, y, jr.key(5))
        for n in params:
            np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(p_b[n]))
        assert float(s_a.norm.count) == 1.0


class TestGatePlus:
    """Next-rung wave: composed gate refinement on the fast-decay champion —
    per-tensor gate normalization AND temperature 2 together over
    ``sigma0_ndecay099``'s conditioning."""

    def test_registry_config(self):
        base = screening_spec("upgd_ema_norm_sigma0").hyperparameters
        spec = screening_spec("sigma0_gateplus")
        assert spec.base_learner == "upgd_w"
        assert spec.noise_update is None
        assert spec.factory is _make_upgd_ema_norm_ext_learner
        assert spec.frozen_probe_input is _ema_frozen_probe_input
        assert spec.hyperparameters == {
            **base,
            "gate_beta": 2.0, "local_gate": 1.0, "hidden_rms": 0.0,
            "norm_decay": 0.99,
        }

    def test_hand_computed_composition(self):
        hp = screening_spec("sigma0_gateplus").hyperparameters
        init_fn, step_fn = _make_upgd_ema_norm_ext_learner(hp)
        params = init_mlp_params(jr.key(7), SMALL)
        x = jr.uniform(jr.key(71), (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
        y = jnp.array(2, jnp.int32)
        state = init_fn(params)
        new_params, new_state, _ = step_fn(params, state, x, y, jr.key(0))

        norm0 = EMANormState(
            mean=jnp.zeros(SMALL.input_dim), var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        x_n, _ = ema_normalize(norm0, x, hp["norm_decay"], hp["norm_epsilon"])
        (_, _), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_n, y
        )
        beta_u = hp["utility_decay"]
        bias = 1.0 - jnp.power(
            jnp.asarray(beta_u, dtype=jnp.float32), jnp.asarray(1, jnp.int32).astype(jnp.float32)
        )
        param_decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        for name in params:
            u = beta_u * jnp.zeros_like(params[name]) + (1.0 - beta_u) * (
                -grads[name] * params[name]
            )
            local_max = jnp.max(u)
            divisor = jnp.where(local_max == 0.0, 1.0, local_max)
            scaled = (u / bias) / divisor
            gate = jax.nn.sigmoid(2.0 * scaled)
            zeros = jnp.zeros_like(params[name])
            expected = params[name] * param_decay - hp["step_size"] * (
                (grads[name] + zeros) * (1.0 - gate)
            )
            np.testing.assert_array_equal(
                np.asarray(new_params[name]), np.asarray(expected), err_msg=name
            )


class TestRFFRLS:
    """Pre-registered existential control: frozen random Fourier features +
    streaming RLS behind the champion's EMA input normalizer — no backprop,
    no MLP."""

    EXPECTED_HP = {
        "rff_m": 1024.0,
        "rff_gamma": 0.001,
        "rff_clip": 3.0,
        "rls_lambda": 0.999,
        "rls_ridge_init": 1.0,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
        "noise_std": 0.0,
    }

    def _learnable_stream(self, n_examples=400, seed=88):
        """Deterministically learnable labels (argmax of a fixed linear map)
        so a short smoke run can rise above chance."""
        kx, kw = jr.split(jr.key(seed))
        x = jr.uniform(kx, (n_examples, SMALL.input_dim), jnp.float32, -1.0, 1.0)
        w_true = jr.normal(kw, (SMALL.input_dim, SMALL.n_classes), jnp.float32)
        y = jnp.argmax(x @ w_true, axis=1).astype(jnp.int32)
        return np.asarray(x), np.asarray(y)

    def test_registry_arm(self):
        spec = screening_spec("rff_rls")
        assert spec.base_learner == "upgd_w"  # reporting bucket only
        assert spec.mechanism == "random_features"
        assert spec.noise_update is None
        assert spec.factory is _make_rff_rls_learner
        assert spec.frozen_probe_input is _rff_frozen_probe_input
        assert spec.hyperparameters == self.EXPECTED_HP

    def test_lin_rls_registry_and_smoke(self):
        """Linear floor: d+1 features (scaled z-scores + bias), no projection;
        smoke run above chance on a linearly separable stream."""
        from alberta_framework.benchmarks.ipmnist_screening import _make_lin_rls_learner

        spec = screening_spec("lin_rls")
        assert spec.mechanism == "random_features"
        assert spec.factory is _make_lin_rls_learner
        assert spec.frozen_probe_input is _rff_frozen_probe_input
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        params = init_mlp_params(jr.key(7), SMALL)
        state = init_fn(params)
        assert state.p.shape == (SMALL.input_dim + 1, SMALL.input_dim + 1)
        assert state.wout.shape == (SMALL.input_dim + 1, SMALL.n_classes)
        x, y = self._learnable_stream()
        correct = 0
        for i in range(len(x)):
            _, state, (acc, loss, plas) = step_fn(
                params, state, jnp.asarray(x[i]), jnp.asarray(y[i]), jr.key(i)
            )
            assert np.isfinite(float(loss))
            correct += float(acc)
        assert correct / len(x) > 0.2

    def test_state_shapes_init_and_symmetry(self):
        """Omega/phase/P/Wout shapes; P starts at (1/ridge)*I, Wout at zero;
        one step keeps P exactly symmetric."""
        spec = screening_spec("rff_rls")
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        params = init_mlp_params(jr.key(5), SMALL)
        state = init_fn(params)
        m = int(self.EXPECTED_HP["rff_m"])
        assert state.omega.shape == (m, SMALL.input_dim)
        assert state.phase.shape == (m,)
        assert state.p.shape == (m, m)
        assert state.wout.shape == (m, SMALL.n_classes)
        assert state.omega.dtype == jnp.float32
        assert state.p.dtype == jnp.float32
        np.testing.assert_array_equal(
            np.asarray(state.p), np.eye(m, dtype=np.float32)
        )
        assert not np.any(np.asarray(state.wout))
        x = jr.normal(jr.key(3), (SMALL.input_dim,))
        y = jnp.array(2, jnp.int32)
        _, new_state, _ = step_fn(params, state, x, y, jr.key(0))
        np.testing.assert_array_equal(
            np.asarray(new_state.p), np.asarray(new_state.p).T
        )
        assert bool(jnp.all(jnp.isfinite(new_state.p)))
        assert bool(jnp.all(jnp.isfinite(new_state.wout)))

    def test_no_backprop_params_and_projection_frozen(self):
        """The MLP params pass through untouched and the random projection is
        never updated; the harness RNG key is unused (identical step for
        different keys)."""
        spec = screening_spec("rff_rls")
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        params = init_mlp_params(jr.key(11), SMALL)
        state = init_fn(params)
        x = jr.normal(jr.key(42), (SMALL.input_dim,))
        y = jnp.array(1, jnp.int32)
        new_params, new_state, _ = step_fn(params, state, x, y, jr.key(0))
        for name in params:
            np.testing.assert_array_equal(
                np.asarray(new_params[name]), np.asarray(params[name]), name
            )
        np.testing.assert_array_equal(
            np.asarray(new_state.omega), np.asarray(state.omega)
        )
        np.testing.assert_array_equal(
            np.asarray(new_state.phase), np.asarray(state.phase)
        )
        _, state_b, metrics_a = step_fn(params, state, x, y, jr.key(987654))
        np.testing.assert_array_equal(
            np.asarray(new_state.wout), np.asarray(state_b.wout)
        )
        assert all(bool(jnp.isfinite(value)) for value in metrics_a)

    def test_deterministic_given_seed(self):
        """Same seed twice gives the bitwise-identical trajectory; different
        seeds draw different frozen projections."""
        x, y = self._learnable_stream(n_examples=64)
        spec = screening_spec("rff_rls")
        a = run_screening_config(x, y, spec, seed=2, config=SMALL)
        b = run_screening_config(x, y, spec, seed=2, config=SMALL)
        np.testing.assert_array_equal(a.per_task_accuracy, b.per_task_accuracy)
        np.testing.assert_array_equal(a.per_task_loss, b.per_task_loss)
        init_fn, _ = spec.factory(spec.hyperparameters)
        omega_0 = init_fn(init_mlp_params(jr.key(0), SMALL)).omega
        omega_1 = init_fn(init_mlp_params(jr.key(1), SMALL)).omega
        assert not np.array_equal(np.asarray(omega_0), np.asarray(omega_1))

    def test_smoke_runs_above_chance(self):
        """2 tasks x 200 steps on a learnable stream: finite metrics, online
        accuracy above the 1/n_classes = 0.2 chance floor."""
        x, y = self._learnable_stream()
        config = IPMNISTConfig(
            n_tasks=2, task_length=200, input_dim=12, hidden1=8, hidden2=6, n_classes=5
        )
        result = run_screening_config(
            x, y, screening_spec("rff_rls"), seed=2, config=config
        )
        acc = np.asarray(result.per_task_accuracy)
        assert np.all(np.isfinite(acc))
        assert np.all(np.isfinite(np.asarray(result.per_task_loss)))
        assert np.all(np.isfinite(np.asarray(result.per_task_plasticity)))
        assert np.all(np.asarray(result.per_task_plasticity) >= 0.0)
        assert np.all(np.asarray(result.per_task_plasticity) <= 1.0)
        assert float(acc.mean()) > 0.2

    def test_frozen_probe_fails_closed(self):
        """No trained protocol MLP exists — sentinel probes must refuse, like
        the _hidden_rms_frozen_probe_input precedent, so merge/reporting can
        never emit a meaningless probe number."""
        spec = screening_spec("rff_rls")
        init_fn, _ = spec.factory(spec.hyperparameters)
        state = init_fn(init_mlp_params(jr.key(0), SMALL))
        with pytest.raises(NotImplementedError, match="rff_rls"):
            spec.frozen_probe_input(
                state, jnp.zeros((3, SMALL.input_dim)), spec.hyperparameters
            )


class TestOptimizerFloorHybrids:
    """Wave-B optimizer-floor hybrids: Adam-class step adaptation under the
    champion's full stability package (fast conditioning + utility gate +
    decoupled decay).

    Diagnosis driving the wave (from the pinned ``confirm_full/`` artifacts):
    ``adamw_cbp_ema_norm`` reaches 0.8425 on task 1 — the best first-task
    score of any arm measured — then decays monotonically to 0.743
    (tasks 150-200) because the composition ran the protocol AdamW
    hyperparameters: no utility gate, weight_decay 0, slow 0.999 normalizer.
    Adam-class convergence under conditioning is real; continual stability
    was simply never attached to it.  These arms attach it.
    """

    NEW_ARMS = (
        "norm_adam_fastv",
        "norm_adam_fastv_b2099",
        "norm_adam_gate",
        "norm_rmsprop_gate",
        "norm_apollo_gate",
        "sgd_momentum_gate",
        "sgd_momentum_gate_m099",
    )

    _STABILITY = {
        "weight_decay": 0.01,
        "utility_decay": 0.9999,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }
    _SHIFT = {
        "fast_decay": 0.9,
        "shift_k": 1.0,
        "shift_delta": 0.02,
        "shift_refractory": 0.0,
    }

    def test_registry_configs(self):
        expected_hp = {
            "norm_adam_fastv": {
                **self._STABILITY, **self._SHIFT,
                "step_size": 0.001, "beta1": 0.0, "beta2": 0.9,
                "eps": 1e-8, "vreset_enabled": 1.0,
            },
            "norm_adam_fastv_b2099": {
                **self._STABILITY, **self._SHIFT,
                "step_size": 0.0003, "beta1": 0.0, "beta2": 0.99,
                "eps": 1e-8, "vreset_enabled": 1.0,
            },
            "norm_adam_gate": {
                **self._STABILITY, **self._SHIFT,
                "step_size": 0.0003, "beta1": 0.0, "beta2": 0.99,
                "eps": 1e-8, "vreset_enabled": 0.0,
            },
            "norm_rmsprop_gate": {
                **self._STABILITY,
                "step_size": 0.001, "rms_rho": 0.9, "rms_epsilon": 1e-8,
            },
            "norm_apollo_gate": {
                **self._STABILITY,
                "step_size": 0.0003, "apollo_decay": 0.99, "apollo_epsilon": 1e-8,
            },
            "sgd_momentum_gate": {
                **self._STABILITY, "step_size": 0.01, "momentum": 0.9,
            },
            "sgd_momentum_gate_m099": {
                **self._STABILITY, "step_size": 0.01, "momentum": 0.99,
            },
        }
        factories = {
            "norm_adam_fastv": _make_norm_adam_fastv_learner,
            "norm_adam_fastv_b2099": _make_norm_adam_fastv_learner,
            "norm_adam_gate": _make_norm_adam_fastv_learner,
            "norm_rmsprop_gate": _make_norm_rmsprop_gate_learner,
            "norm_apollo_gate": _make_norm_apollo_gate_learner,
            "sgd_momentum_gate": _make_sgd_momentum_gate_learner,
            "sgd_momentum_gate_m099": _make_sgd_momentum_gate_learner,
        }
        for name in self.NEW_ARMS:
            spec = screening_spec(name)
            assert spec.mechanism == "optimizer_floor_hybrid", name
            assert spec.noise_update is None, name
            assert spec.factory is factories[name], name
            assert spec.frozen_probe_input is _ema_frozen_probe_input, name
            assert spec.hyperparameters == expected_hp[name], name
            # no perturbation channel at all — not even as an inert hp
            assert "noise_std" not in spec.hyperparameters, name

    def _eager_norm_and_gate(self, norm_state, fast_mean, params, utility, clock,
                             x, y, hp):
        """Shared reference head: shift-normalize, grads, utility gate."""
        x_norm, new_norm, new_fast, shifted = shift_adaptive_normalize(
            norm_state, fast_mean, x,
            decay=hp["norm_decay"], fast_decay=hp["fast_decay"],
            epsilon=hp["norm_epsilon"], shift_k=hp["shift_k"],
            shift_delta=hp["shift_delta"],
            shift_refractory=hp["shift_refractory"],
        )
        _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        new_utility, gate = _upgd_utility_and_gate(
            params, grads, utility, clock, hp["utility_decay"]
        )
        return x_norm, new_norm, new_fast, shifted, grads, new_utility, gate

    def test_norm_adam_fastv_reduction_hand_composed(self):
        """With an untriggerable detector the step equals the hand-composed
        shift-normalize -> gated per-element AdamW (wd outside the moments)
        trajectory, bit-for-bit."""
        hp = dict(screening_spec("norm_adam_fastv").hyperparameters)
        hp["shift_k"] = 1e9  # untriggerable: pure normalize -> gated AdamW
        init_fn, step_fn = _make_norm_adam_fastv_learner(hp)
        params = init_mlp_params(jr.key(5), SMALL)
        state = init_fn(params)
        adam_hp = {
            "beta1": hp["beta1"], "beta2": hp["beta2"],
            "step_size": hp["step_size"], "eps": hp["eps"], "weight_decay": 0.0,
        }
        ref = {
            "params": params,
            "utility": {n: jnp.zeros_like(v) for n, v in params.items()},
            "m": {n: jnp.zeros_like(v) for n, v in params.items()},
            "v": {n: jnp.zeros_like(v) for n, v in params.items()},
            "count": {n: jnp.zeros_like(v) for n, v in params.items()},
            "norm": EMANormState(
                mean=jnp.zeros(SMALL.input_dim),
                var=jnp.ones(SMALL.input_dim),
                count=jnp.zeros(SMALL.input_dim),
            ),
            "fast": jnp.zeros(SMALL.input_dim),
        }
        param_decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        key = jr.key(31)
        for i in range(4):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,)) * 1.5 + 0.25
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(900 + i))
            clock = jnp.array(i + 1, jnp.int32)
            (_, ref["norm"], ref["fast"], _, grads, ref["utility"], gate,
             ) = self._eager_norm_and_gate(
                ref["norm"], ref["fast"], ref["params"], ref["utility"],
                clock, x, y, hp,
            )
            new_ref_params = {}
            for n in ref["params"]:
                step_arr, ref["m"][n], ref["v"][n], ref["count"][n] = adam_elem_step(
                    ref["params"][n], ref["m"][n], ref["v"][n], ref["count"][n],
                    grads[n], adam_hp,
                )
                new_ref_params[n] = ref["params"][n] * param_decay - (
                    step_arr * (1.0 - gate[n])
                )
            ref["params"] = new_ref_params
            for n in ref["params"]:
                np.testing.assert_array_equal(
                    np.asarray(params[n]), np.asarray(ref["params"][n]), err_msg=n
                )
                np.testing.assert_array_equal(
                    np.asarray(state.v[n]), np.asarray(ref["v"][n]), err_msg=n
                )
                np.testing.assert_array_equal(
                    np.asarray(state.count[n]), np.asarray(ref["count"][n]), err_msg=n
                )

    def test_norm_adam_fastv_resets_w1_rows_on_shift(self):
        """A detected input shift resets the shifted feature's w1-row Adam
        moments (m, v, count) and nothing else; with vreset_enabled = 0 the
        moments carry through the identical detector trigger."""
        hp_reset = dict(screening_spec("norm_adam_fastv").hyperparameters)
        hp_carry = dict(hp_reset)
        hp_carry["vreset_enabled"] = 0.0
        init_r, step_r = _make_norm_adam_fastv_learner(hp_reset)
        init_c, step_c = _make_norm_adam_fastv_learner(hp_carry)
        params = init_mlp_params(jr.key(9), SMALL)
        s_r = init_r(params)
        s_c = init_c(params)
        p_r = p_c = params
        warmup = 30
        for i in range(warmup):
            x = jnp.zeros(SMALL.input_dim, jnp.float32)
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            p_r, s_r, _ = step_r(p_r, s_r, x, y, jr.key(i))
            p_c, s_c, _ = step_c(p_c, s_c, x, y, jr.key(i))
        # feature 3 jumps far beyond shift_k * sqrt(var) + shift_delta
        x = jnp.zeros(SMALL.input_dim, jnp.float32).at[3].set(50.0)
        y = jnp.array(0, jnp.int32)
        p_r, s_r, _ = step_r(p_r, s_r, x, y, jr.key(777))
        p_c, s_c, _ = step_c(p_c, s_c, x, y, jr.key(777))
        count_r = np.asarray(s_r.count["w1"])
        count_c = np.asarray(s_c.count["w1"])
        # the shifted feature's row restarted its per-element Adam clock
        np.testing.assert_array_equal(count_r[3], np.ones(count_r.shape[1]))
        assert np.all(count_c[3] == warmup + 1)
        # every other row carried
        mask = np.ones(count_r.shape[0], dtype=bool)
        mask[3] = False
        np.testing.assert_array_equal(count_r[mask], count_c[mask])
        for field in ("m", "v"):
            arr_r = np.asarray(getattr(s_r, field)["w1"])
            arr_c = np.asarray(getattr(s_c, field)["w1"])
            np.testing.assert_array_equal(arr_r[mask], arr_c[mask])
        # non-input tensors never reset
        for n in ("b1", "w2", "b2", "w3", "b3"):
            np.testing.assert_array_equal(
                np.asarray(s_r.count[n]), np.asarray(s_c.count[n]), err_msg=n
            )

    def test_norm_rmsprop_gate_hand_computed(self):
        """The full step equals hand-computed normalize -> RMSprop (rho, no
        momentum, no bias correction) -> gate -> decoupled decay."""
        hp = screening_spec("norm_rmsprop_gate").hyperparameters
        init_fn, step_fn = _make_norm_rmsprop_gate_learner(hp)
        params = init_mlp_params(jr.key(13), SMALL)
        state = init_fn(params)
        ref_params = params
        ref_utility = {n: jnp.zeros_like(v) for n, v in params.items()}
        ref_v = {n: jnp.zeros_like(v) for n, v in params.items()}
        norm_state = EMANormState(
            mean=jnp.zeros(SMALL.input_dim),
            var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        rho = hp["rms_rho"]
        param_decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        key = jr.key(37)
        for i in range(4):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,)) * 2.0 - 0.5
            y = jnp.array((i + 1) % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(300 + i))
            x_norm, norm_state = ema_normalize(
                norm_state, x, hp["norm_decay"], hp["norm_epsilon"]
            )
            _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                ref_params, x_norm, y
            )
            ref_utility, gate = _upgd_utility_and_gate(
                ref_params, grads, ref_utility, jnp.array(i + 1, jnp.int32),
                hp["utility_decay"],
            )
            new_ref = {}
            for n in ref_params:
                ref_v[n] = rho * ref_v[n] + (1.0 - rho) * grads[n] * grads[n]
                direction = grads[n] / (jnp.sqrt(ref_v[n]) + hp["rms_epsilon"])
                new_ref[n] = ref_params[n] * param_decay - hp["step_size"] * (
                    direction * (1.0 - gate[n])
                )
            ref_params = new_ref
            for n in ref_params:
                np.testing.assert_array_equal(
                    np.asarray(params[n]), np.asarray(ref_params[n]), err_msg=n
                )

    def test_norm_apollo_gate_channel_axis_and_hand_computed(self):
        """Channel = fan-out (per-neuron): vchan is (fan_out,) for 2-D weights
        and per-element for biases; the applied step divides each column by
        its channel RMS. Hand-computed pin over 3 steps."""
        hp = screening_spec("norm_apollo_gate").hyperparameters
        init_fn, step_fn = _make_norm_apollo_gate_learner(hp)
        params = init_mlp_params(jr.key(17), SMALL)
        state = init_fn(params)
        assert state.vchan["w1"].shape == (SMALL.hidden1,)
        assert state.vchan["w2"].shape == (SMALL.hidden2,)
        assert state.vchan["w3"].shape == (SMALL.n_classes,)
        assert state.vchan["b1"].shape == (SMALL.hidden1,)
        ref_params = params
        ref_utility = {n: jnp.zeros_like(v) for n, v in params.items()}
        ref_v = {
            n: (jnp.zeros(v.shape[1], jnp.float32) if v.ndim == 2
                else jnp.zeros_like(v))
            for n, v in params.items()
        }
        norm_state = EMANormState(
            mean=jnp.zeros(SMALL.input_dim),
            var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        rho = hp["apollo_decay"]
        param_decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        key = jr.key(41)
        for i in range(3):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,))
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(500 + i))
            x_norm, norm_state = ema_normalize(
                norm_state, x, hp["norm_decay"], hp["norm_epsilon"]
            )
            _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                ref_params, x_norm, y
            )
            ref_utility, gate = _upgd_utility_and_gate(
                ref_params, grads, ref_utility, jnp.array(i + 1, jnp.int32),
                hp["utility_decay"],
            )
            new_ref = {}
            for n in ref_params:
                g = grads[n]
                if ref_params[n].ndim == 2:
                    stat = jnp.mean(g * g, axis=0)
                    ref_v[n] = rho * ref_v[n] + (1.0 - rho) * stat
                    denom = jnp.sqrt(ref_v[n])[None, :] + hp["apollo_epsilon"]
                else:
                    ref_v[n] = rho * ref_v[n] + (1.0 - rho) * (g * g)
                    denom = jnp.sqrt(ref_v[n]) + hp["apollo_epsilon"]
                new_ref[n] = ref_params[n] * param_decay - hp["step_size"] * (
                    (g / denom) * (1.0 - gate[n])
                )
            ref_params = new_ref
            for n in ref_params:
                np.testing.assert_array_equal(
                    np.asarray(params[n]), np.asarray(ref_params[n]), err_msg=n
                )

    def test_norm_apollo_bias_step_equals_rmsprop_bias_step(self):
        """The bias path is the exact 1-element-channel specialization: one
        step from identical inits with matched decays produces bitwise equal
        bias updates in both arms."""
        base = dict(screening_spec("norm_rmsprop_gate").hyperparameters)
        apollo_hp = dict(screening_spec("norm_apollo_gate").hyperparameters)
        apollo_hp["apollo_decay"] = base["rms_rho"]
        apollo_hp["apollo_epsilon"] = base["rms_epsilon"]
        apollo_hp["step_size"] = base["step_size"]
        init_r, step_r = _make_norm_rmsprop_gate_learner(base)
        init_a, step_a = _make_norm_apollo_gate_learner(apollo_hp)
        params = init_mlp_params(jr.key(23), SMALL)
        x = jr.normal(jr.key(70), (SMALL.input_dim,))
        y = jnp.array(1, jnp.int32)
        p_r, _, _ = step_r(params, init_r(params), x, y, jr.key(0))
        p_a, _, _ = step_a(params, init_a(params), x, y, jr.key(0))
        for n in ("b1", "b2", "b3"):
            np.testing.assert_array_equal(
                np.asarray(p_r[n]), np.asarray(p_a[n]), err_msg=n
            )

    def test_sgd_momentum_gate_mu0_reduces_to_sigma0_ndecay099_bitwise(self):
        """momentum = 0 collapses the EMA-corrected momentum to the raw
        gradient, so the trajectory equals the sigma0_ndecay099 champion
        bit-for-bit (same normalizer, same gate, same decay)."""
        hp = dict(screening_spec("sgd_momentum_gate").hyperparameters)
        hp["momentum"] = 0.0
        init_fn, step_fn = _make_sgd_momentum_gate_learner(hp)
        champion = screening_spec("sigma0_ndecay099")
        init_ref, step_ref = champion.factory(champion.hyperparameters)
        params = init_mlp_params(jr.key(29), SMALL)
        s_ours = init_fn(params)
        s_ref = init_ref(params)
        p_ours = p_ref = params
        key = jr.key(43)
        for i in range(5):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,)) + 0.1
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            p_ours, s_ours, _ = step_fn(p_ours, s_ours, x, y, jr.key(600 + i))
            p_ref, s_ref, _ = step_ref(p_ref, s_ref, x, y, jr.key(600 + i))
            for n in p_ours:
                np.testing.assert_array_equal(
                    np.asarray(p_ours[n]), np.asarray(p_ref[n]), err_msg=n
                )

    def test_key_is_unused_on_every_arm(self):
        """No perturbation anywhere in the wave: different RNG keys produce
        bit-identical steps."""
        params = init_mlp_params(jr.key(3), SMALL)
        x = jr.normal(jr.key(90), (SMALL.input_dim,))
        y = jnp.array(2, jnp.int32)
        for name in self.NEW_ARMS:
            spec = screening_spec(name)
            init_fn, step_fn = spec.factory(spec.hyperparameters)
            state = init_fn(params)
            p_a, _, _ = step_fn(params, state, x, y, jr.key(0))
            p_b, _, _ = step_fn(params, state, x, y, jr.key(987654))
            for n in params:
                np.testing.assert_array_equal(
                    np.asarray(p_a[n]), np.asarray(p_b[n]), err_msg=f"{name}:{n}"
                )

    def test_smoke_runs_finite_and_mechanisms_engage(self, small_data):
        """Tiny-protocol smoke: finite metrics on every arm, and each
        mechanism family separates from the champion trajectory."""
        x, y = small_data
        champion = run_screening_config(
            x, y, screening_spec("sigma0_ndecay099"), seed=19, config=SMALL
        )
        for name in ("norm_adam_fastv", "norm_rmsprop_gate", "norm_apollo_gate",
                     "sgd_momentum_gate"):
            result = run_screening_config(
                x, y, screening_spec(name), seed=19, config=SMALL
            )
            assert np.all(np.isfinite(result.per_task_accuracy)), name
            assert np.all(np.isfinite(result.per_task_loss)), name
            assert not np.array_equal(
                result.per_task_loss, champion.per_task_loss
            ), name


class TestGatedL2Init:
    """``sigma0_ndecay099_gated_l2init``: an additive, utility-gated pull
    toward the initial weights on top of a historical comparison baseline (ported CCBP /
    Calibrated-Partial-Resets-style graded reset; see
    :func:`alberta_framework.benchmarks.ipmnist_screening._make_sigma0_gated_l2init_learner`)."""

    def test_registry_binds_ema_frozen_probe_input(self):
        spec = screening_spec("sigma0_ndecay099_gated_l2init")
        assert spec.frozen_probe_input is _ema_frozen_probe_input

    def test_pull_scale_zero_reduces_to_sigma0_ndecay099_bitwise(self):
        """``l2init_pull_scale=0`` collapses the additive pull term to
        exactly zero, so the trajectory equals the ``sigma0_ndecay099``
        champion bit-for-bit (same normalizer, same gate, same decay)."""
        hp = dict(screening_spec("sigma0_ndecay099_gated_l2init").hyperparameters)
        hp["l2init_pull_scale"] = 0.0
        init_fn, step_fn = _make_sigma0_gated_l2init_learner(hp)
        champion = screening_spec("sigma0_ndecay099")
        init_ref, step_ref = champion.factory(champion.hyperparameters)
        params = init_mlp_params(jr.key(29), SMALL)
        s_ours = init_fn(params)
        s_ref = init_ref(params)
        p_ours = p_ref = params
        key = jr.key(43)
        for i in range(5):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,)) + 0.1
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            p_ours, s_ours, _ = step_fn(p_ours, s_ours, x, y, jr.key(600 + i))
            p_ref, s_ref, _ = step_ref(p_ref, s_ref, x, y, jr.key(600 + i))
            for n in p_ours:
                np.testing.assert_array_equal(
                    np.asarray(p_ours[n]), np.asarray(p_ref[n]), err_msg=n
                )

    def test_registered_pull_scale_hand_computed(self):
        """The registered arm equals a hand-computed normalize -> grad ->
        gate -> ``w*(1-lr*wd) - lr*pull_scale*(1-gate)*(w-w0) -
        lr*(1-gate)*grad`` trajectory, bit for bit."""
        hp = screening_spec("sigma0_ndecay099_gated_l2init").hyperparameters
        assert hp["l2init_pull_scale"] == 0.01
        init_fn, step_fn = _make_sigma0_gated_l2init_learner(hp)
        params = init_mlp_params(jr.key(14), SMALL)
        w0 = {n: v for n, v in params.items()}
        state = init_fn(params)
        ref_params = params
        ref_utility = {n: jnp.zeros_like(v) for n, v in params.items()}
        norm_state = EMANormState(
            mean=jnp.zeros(SMALL.input_dim),
            var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        key = jr.key(15)
        for i in range(4):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,)) * 2.0 + 0.5
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(500 + i))
            x_norm, norm_state = ema_normalize(
                norm_state, x, hp["norm_decay"], hp["norm_epsilon"]
            )
            _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                ref_params, x_norm, y
            )
            beta = hp["utility_decay"]
            ref_utility = {
                n: beta * ref_utility[n] + (1.0 - beta) * (-grads[n] * ref_params[n])
                for n in ref_params
            }
            count = i + 1
            bias_correction = 1.0 - beta**count
            global_max = jnp.max(
                jnp.stack([jnp.max(ref_utility[n]) for n in sorted(ref_params)])
            )
            new_ref: dict[str, jnp.ndarray] = {}
            for n in ref_params:
                gate = jax.nn.sigmoid((ref_utility[n] / bias_correction) / global_max)
                pull = hp["l2init_pull_scale"] * (1.0 - gate) * (ref_params[n] - w0[n])
                new_ref[n] = (
                    ref_params[n] * (1.0 - hp["step_size"] * hp["weight_decay"])
                    - hp["step_size"] * pull
                    - hp["step_size"] * (grads[n] * (1.0 - gate))
                )
            ref_params = new_ref
            for n in ref_params:
                # Independently reordered floating-point ops (this hand
                # derivation vs. the factory's fused expression) can diverge
                # by a few ULP even when both implement the identical
                # equation; the bit-exact contract lives in
                # test_pull_scale_zero_reduces_to_sigma0_ndecay099_bitwise,
                # which calls the champion's own reduction path directly.
                np.testing.assert_allclose(
                    np.asarray(params[n]),
                    np.asarray(ref_params[n]),
                    atol=1e-6,
                    rtol=1e-5,
                    err_msg=n,
                )

    def test_nonzero_pull_diverges_from_champion_and_stays_finite(self, small_data):
        """The registered arm's trajectory differs from the champion's and
        remains finite over a tiny-protocol smoke run."""
        x, y = small_data
        champion = run_screening_config(
            x, y, screening_spec("sigma0_ndecay099"), seed=19, config=SMALL
        )
        result = run_screening_config(
            x, y, screening_spec("sigma0_ndecay099_gated_l2init"), seed=19, config=SMALL
        )
        assert np.all(np.isfinite(result.per_task_accuracy))
        assert np.all(np.isfinite(result.per_task_loss))
        assert not np.array_equal(result.per_task_loss, champion.per_task_loss)

    def test_key_is_unused_for_no_extra_randomness(self):
        """The arm draws no perturbation noise (noise_std=0 by inheritance
        from ``_sigma0_ext_hp``): different RNG keys must still reach an
        identical result via the shared per-step noise draw's sigma=0
        short-circuit."""
        params = init_mlp_params(jr.key(3), SMALL)
        x = jr.normal(jr.key(90), (SMALL.input_dim,))
        y = jnp.array(2, jnp.int32)
        spec = screening_spec("sigma0_ndecay099_gated_l2init")
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        state = init_fn(params)
        p_a, _, _ = step_fn(params, state, x, y, jr.key(0))
        p_b, _, _ = step_fn(params, state, x, y, jr.key(987654))
        for n in params:
            np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(p_b[n]), err_msg=n)


class TestComparisonArms:
    """Reviewer comparison rows: published plasticity mechanisms behind the
    champion's EMA input conditioning (decay 0.99) on a plain-SGD base — no
    utility gate, no perturbation.  Each mechanism reduces bit-exactly to the
    shared normalized-SGD base when its constant is inert."""

    BASE = {"step_size": 0.01, "norm_decay": 0.99, "norm_epsilon": 1e-8}
    ARMS = (
        "sgd_ema_norm_d099",
        "wclip_ema_norm",
        "fade_head_ema_norm",
        "snr_ema_norm",
        "l2init_ema_norm",
    )

    def test_registry_configs(self):
        expected = {
            "sgd_ema_norm_d099": (
                _make_sgd_ema_norm_learner,
                {**self.BASE, "weight_decay": 0.01},
            ),
            "wclip_ema_norm": (
                _make_wclip_ema_norm_learner,
                {**self.BASE, "weight_decay": 0.0, "clip_kappa": 2.0},
            ),
            "fade_head_ema_norm": (
                _make_fade_head_ema_norm_learner,
                {
                    **self.BASE,
                    "weight_decay": 0.0,
                    "fade_alpha": 0.005,
                    "fade_gamma0": -6.9,
                    "fade_theta_lambda": 0.1,
                },
            ),
            "snr_ema_norm": (
                _make_snr_ema_norm_learner,
                {
                    **self.BASE,
                    "weight_decay": 0.0,
                    "snr_eta": 0.005,
                    "snr_rate_decay": 0.999,
                    "snr_rate_floor": 1e-4,
                },
            ),
            "l2init_ema_norm": (
                _make_l2init_ema_norm_learner,
                {**self.BASE, "weight_decay": 0.01},
            ),
        }
        for name, (factory, hp) in expected.items():
            spec = screening_spec(name)
            assert spec.base_learner == "upgd_w", name
            assert spec.noise_update is None, name
            assert spec.factory is factory, name
            assert spec.frozen_probe_input is _ema_frozen_probe_input, name
            assert spec.hyperparameters == hp, name
            # no utility, no gate, no noise — not even as inert hyperparameters
            assert "utility_decay" not in spec.hyperparameters, name
            assert "noise_std" not in spec.hyperparameters, name

    def _cloned(self, name_for_plumbing, factory, hp):
        return ScreeningSpec(
            name=name_for_plumbing,
            base_learner="upgd_w",
            mechanism="comparison_reduction_pin",
            hyperparameters=hp,
            factory=factory,
        )

    def test_wclip_max_finite_kappa_reduces_to_sgd_base_bitwise(self, small_data):
        """Max-finite clip_kappa is a no-op: bit-exact sgd_ema_norm_d099."""
        x, y = small_data
        ref_spec = screening_spec("sgd_ema_norm_d099")
        ours = run_screening_config(
            x, y,
            self._cloned(
                "sgd_ema_norm_d099", _make_wclip_ema_norm_learner,
                {
                    **ref_spec.hyperparameters,
                    "clip_kappa": float(np.finfo(np.float32).max),
                },
            ),
            seed=5, config=SMALL,
        )
        ref = run_screening_config(x, y, ref_spec, seed=5, config=SMALL)
        np.testing.assert_array_equal(ours.per_task_accuracy, ref.per_task_accuracy)
        np.testing.assert_array_equal(ours.per_task_loss, ref.per_task_loss)

    def test_wclip_bounds_enforced_per_layer(self):
        """After any step every weight and bias obeys |w| <= kappa/sqrt(fan_in)
        (Elsayed et al. RLC 2024 Algorithm 1 clips biases too)."""
        spec = screening_spec("wclip_ema_norm")
        hp = {**spec.hyperparameters, "clip_kappa": 1.0}
        init_fn, step_fn = _make_wclip_ema_norm_learner(hp)
        params = init_mlp_params(jr.key(4), SMALL)
        big = {n: v + 100.0 for n, v in params.items()}  # far outside every bound
        state = init_fn(big)
        x = jr.normal(jr.key(41), (SMALL.input_dim,))
        y = jnp.array(1, jnp.int32)
        new_params, _, _ = step_fn(big, state, x, y, jr.key(0))
        fan_in = {"1": SMALL.input_dim, "2": SMALL.hidden1, "3": SMALL.hidden2}
        for n in new_params:
            bound = hp["clip_kappa"] / math.sqrt(fan_in[n[1:]])
            values = np.asarray(new_params[n])
            assert np.all(values <= bound + 1e-7), n
            assert np.all(values >= -bound - 1e-7), n

    def test_fade_lambda_zero_reduces_to_sgd_base_bitwise(self, small_data):
        """theta=0 and a min-finite gamma0 (lambda=0) pin the head to the plain SGD
        step, so the whole arm equals the wd=0 normalized-SGD base."""
        x, y = small_data
        base_hp = {**self.BASE, "weight_decay": 0.0}
        ours = run_screening_config(
            x, y,
            self._cloned(
                "sgd_ema_norm_d099", _make_fade_head_ema_norm_learner,
                {
                    **base_hp,
                    "fade_alpha": 0.005,
                    "fade_gamma0": -float(np.finfo(np.float32).max),
                    "fade_theta_lambda": 0.0,
                },
            ),
            seed=6, config=SMALL,
        )
        ref = run_screening_config(
            x, y,
            self._cloned("sgd_ema_norm_d099", _make_sgd_ema_norm_learner, base_hp),
            seed=6, config=SMALL,
        )
        np.testing.assert_array_equal(ours.per_task_accuracy, ref.per_task_accuracy)
        np.testing.assert_array_equal(ours.per_task_loss, ref.per_task_loss)

    def test_fade_lambda_stays_bounded_over_random_steps(self):
        """gamma is capped at 0 so lambda = exp(gamma) stays in [0, 1] and the
        head decay factor 1 - lambda stays in [0, 1]; traces stay finite."""
        spec = screening_spec("fade_head_ema_norm")
        init_fn, step_fn = _make_fade_head_ema_norm_learner(spec.hyperparameters)
        params = init_mlp_params(jr.key(8), SMALL)
        state = init_fn(params)
        key = jr.key(77)
        for i in range(25):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,)) * 2.0
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(300 + i))
            for n in ("w3", "b3"):
                gamma = np.asarray(state.gamma[n])
                assert np.all(gamma <= 0.0), n
                lam = np.exp(gamma)
                assert np.all(lam >= 0.0) and np.all(lam <= 1.0), n
                assert np.all(np.isfinite(np.asarray(state.fade_trace[n]))), n
            assert np.all(np.isfinite(np.asarray(params["w3"])))

    def test_snr_eta_zero_never_resets_matches_sgd_base_bitwise(self, small_data):
        """snr_eta=0 disables the rejection test exactly: the parameter
        trajectory is the plain normalized-SGD base, bit for bit."""
        x, y = small_data
        spec = screening_spec("snr_ema_norm")
        ours = run_screening_config(
            x, y,
            self._cloned(
                "sgd_ema_norm_d099", _make_snr_ema_norm_learner,
                {**spec.hyperparameters, "snr_eta": 0.0},
            ),
            seed=9, config=SMALL,
        )
        ref = run_screening_config(
            x, y,
            self._cloned(
                "sgd_ema_norm_d099", _make_sgd_ema_norm_learner,
                {**self.BASE, "weight_decay": 0.0},
            ),
            seed=9, config=SMALL,
        )
        np.testing.assert_array_equal(ours.per_task_accuracy, ref.per_task_accuracy)
        np.testing.assert_array_equal(ours.per_task_loss, ref.per_task_loss)

    def test_snr_resets_silenced_high_rate_unit_only(self):
        """A unit with high historical firing rate and a long silence fails
        the geometric-tail test and resets (incoming redrawn in the init
        range, incoming bias redrawn, outgoing zeroed, statistics cleared);
        a unit whose observed rate was always ~0 keeps a long-tailed null
        and does NOT reset; firing units are untouched."""
        hp = dict(screening_spec("snr_ema_norm").hyperparameters)
        params = init_mlp_params(jr.key(5), SMALL)
        h1 = SMALL.hidden1
        silence = jnp.zeros(h1, jnp.int32).at[0].set(50).at[1].set(50)
        # unit 0: healthy history (rate EMA ~0.5); unit 1: dead from the start
        rate = jnp.zeros(h1, jnp.float32).at[0].set(0.5)
        rate = rate.at[2:].set(0.5)
        age = jnp.full((h1,), 10_000, jnp.int32)
        new_params, new_silence, new_rate, new_age, mask = snr_maybe_reset_layer(
            params, silence, rate, age, _CBP_LAYERS[0], jr.key(6), hp
        )
        mask = np.asarray(mask)
        assert mask[0], "silenced high-rate unit must reset"
        assert not mask[1], "always-quiet unit must not reset (rate floor)"
        assert not mask[2:].any(), "firing units must not reset"
        bound = 1.0 / math.sqrt(SMALL.input_dim)
        col = np.asarray(new_params["w1"][:, 0])
        assert not np.allclose(col, np.asarray(params["w1"][:, 0]))
        assert np.all(np.abs(col) <= bound)
        assert abs(float(new_params["b1"][0])) <= bound
        assert float(new_params["b1"][0]) != float(params["b1"][0])
        np.testing.assert_allclose(np.asarray(new_params["w2"][0, :]), 0.0)
        assert int(new_silence[0]) == 0
        assert float(new_rate[0]) == 0.0
        assert int(new_age[0]) == 0
        # untouched units keep everything
        np.testing.assert_array_equal(
            np.asarray(new_params["w1"][:, 1]), np.asarray(params["w1"][:, 1])
        )
        assert int(new_silence[1]) == 50 and int(new_age[1]) == 10_000

    def test_l2init_wd_zero_reduces_to_sgd_base_bitwise(self, small_data):
        """weight_decay=0 removes the pull toward init: bit-exact base."""
        x, y = small_data
        base_hp = {**self.BASE, "weight_decay": 0.0}
        ours = run_screening_config(
            x, y,
            self._cloned(
                "sgd_ema_norm_d099", _make_l2init_ema_norm_learner, base_hp
            ),
            seed=13, config=SMALL,
        )
        ref = run_screening_config(
            x, y,
            self._cloned("sgd_ema_norm_d099", _make_sgd_ema_norm_learner, base_hp),
            seed=13, config=SMALL,
        )
        np.testing.assert_array_equal(ours.per_task_accuracy, ref.per_task_accuracy)
        np.testing.assert_array_equal(ours.per_task_loss, ref.per_task_loss)

    def test_l2init_reduction_pin_hand_computed(self):
        """The full step equals a hand-computed normalize -> grad ->
        ``w - lr*wd*(w - w0) - lr*grad`` trajectory, bit for bit."""
        hp = screening_spec("l2init_ema_norm").hyperparameters
        init_fn, step_fn = _make_l2init_ema_norm_learner(hp)
        params = init_mlp_params(jr.key(14), SMALL)
        w0 = {n: v for n, v in params.items()}
        state = init_fn(params)
        ref_params = params
        norm_state = EMANormState(
            mean=jnp.zeros(SMALL.input_dim),
            var=jnp.ones(SMALL.input_dim),
            count=jnp.array(0.0),
        )
        key = jr.key(15)
        for i in range(4):
            x = jr.normal(jr.fold_in(key, i), (SMALL.input_dim,)) * 2.0 + 0.5
            y = jnp.array(i % SMALL.n_classes, jnp.int32)
            params, state, _ = step_fn(params, state, x, y, jr.key(500 + i))
            x_norm, norm_state = ema_normalize(
                norm_state, x, hp["norm_decay"], hp["norm_epsilon"]
            )
            _, grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                ref_params, x_norm, y
            )
            ref_params = {
                n: ref_params[n]
                - hp["step_size"] * hp["weight_decay"] * (ref_params[n] - w0[n])
                - hp["step_size"] * grads[n]
                for n in ref_params
            }
            for n in ref_params:
                np.testing.assert_array_equal(
                    np.asarray(params[n]), np.asarray(ref_params[n])
                )

    def test_key_is_unused_except_snr(self):
        """wclip/fade/l2init consume no randomness; snr consumes the key only
        for redraw material, which cannot reach params when nothing resets."""
        params = init_mlp_params(jr.key(3), SMALL)
        x = jr.normal(jr.key(91), (SMALL.input_dim,))
        y = jnp.array(2, jnp.int32)
        for name in ("sgd_ema_norm_d099", "wclip_ema_norm", "fade_head_ema_norm",
                     "l2init_ema_norm"):
            spec = screening_spec(name)
            init_fn, step_fn = spec.factory(spec.hyperparameters)
            state = init_fn(params)
            p_a, _, _ = step_fn(params, state, x, y, jr.key(0))
            p_b, _, _ = step_fn(params, state, x, y, jr.key(424242))
            for n in params:
                np.testing.assert_array_equal(
                    np.asarray(p_a[n]), np.asarray(p_b[n]), err_msg=f"{name}:{n}"
                )

    def test_smoke_runs_finite_and_mechanisms_engage(self, small_data):
        """Tiny-protocol smoke: finite metrics on every comparison arm, and
        each mechanism separates from the shared normalized-SGD base."""
        x, y = small_data
        base = run_screening_config(
            x, y, screening_spec("sgd_ema_norm_d099"), seed=23, config=SMALL
        )
        assert np.all(np.isfinite(base.per_task_accuracy))
        for name in ("wclip_ema_norm", "fade_head_ema_norm", "snr_ema_norm",
                     "l2init_ema_norm"):
            result = run_screening_config(
                x, y, screening_spec(name), seed=23, config=SMALL
            )
            assert np.all(np.isfinite(result.per_task_accuracy)), name
            assert np.all(np.isfinite(result.per_task_loss)), name
            assert not np.array_equal(
                result.per_task_loss, base.per_task_loss
            ), name


class TestNaiveBayes:
    """V3 validation: streaming class-conditional diagonal Gaussians.

    No gradients, no MLP — prediction is the argmax class posterior under
    annealed fast-EMA per-class feature statistics (equation parity with
    ``ema_normalize``, conditioned on the observed label).
    """

    def test_registry_config(self):
        spec = screening_spec("naive_bayes")
        assert spec.mechanism == "streaming_generative_classifier"
        assert spec.factory is _make_naive_bayes_learner
        assert spec.hyperparameters["noise_std"] == 0.0
        assert 0.0 < spec.hyperparameters["nb_decay"] < 1.0
        assert spec.hyperparameters["nb_var_epsilon"] > 0.0
        assert spec.noise_update is None
        with pytest.raises(NotImplementedError):
            spec.frozen_probe_input(None, jnp.zeros(4), spec.hyperparameters)

    def test_first_update_hand_computed(self):
        """One step from init: annealed EMA gives running-average semantics."""
        spec = screening_spec("naive_bayes")
        hp = dict(spec.hyperparameters)
        eps = hp["nb_var_epsilon"]
        params = init_mlp_params(jr.key(0), SMALL)
        init_fn, step_fn = spec.factory(hp)
        state = init_fn(params)
        n_classes = SMALL.n_classes
        np.testing.assert_allclose(np.asarray(state.prior), np.full(n_classes, 1.0 / n_classes))
        x = jnp.linspace(-1.0, 1.0, SMALL.input_dim, dtype=jnp.float32)
        y = jnp.array(1, jnp.int32)
        _, new_state, metrics = step_fn(params, state, x, y, jr.key(9))
        xf = np.asarray(x, dtype=np.float64)
        # Class 1: count 0 -> 1; effective decay min(decay, 1 - 1/2) = 0.5.
        np.testing.assert_allclose(np.asarray(new_state.cmean[1]), 0.5 * xf, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(new_state.cvar[1]),
            np.maximum(0.5 + 0.5 * xf * 0.5 * xf, eps),
            rtol=1e-6,
        )
        assert float(new_state.ccount[1]) == 1.0
        # Untouched class rows stay bitwise at init.
        for c in range(n_classes):
            if c == 1:
                continue
            np.testing.assert_array_equal(np.asarray(new_state.cmean[c]), 0.0)
            np.testing.assert_array_equal(np.asarray(new_state.cvar[c]), 1.0)
            assert float(new_state.ccount[c]) == 0.0
        # Prior: annealed one-hot EMA, effective decay 0.5 at step 1.
        expected_prior = 0.5 * np.full(n_classes, 1.0 / n_classes)
        expected_prior[1] += 0.5
        np.testing.assert_allclose(np.asarray(new_state.prior), expected_prior, rtol=1e-6)
        accuracy, loss, plasticity = metrics
        assert float(loss) > 0.0 and np.isfinite(float(loss))
        assert float(accuracy) in (0.0, 1.0)
        assert 0.0 <= float(plasticity) <= 1.0

    def test_logits_match_manual_posterior(self):
        spec = screening_spec("naive_bayes")
        params = init_mlp_params(jr.key(4), SMALL)
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        state = init_fn(params)
        # Drive a few updates so statistics differ across classes.
        key = jr.key(77)
        for t in range(20):
            key, kx = jr.split(key)
            x = jr.uniform(kx, (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
            y = jnp.array(t % SMALL.n_classes, jnp.int32)
            _, state, _ = step_fn(params, state, x, y, jr.key(t))
        x = jr.uniform(jr.key(123), (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
        logits = np.asarray(naive_bayes_logits(state, x), dtype=np.float64)
        mu = np.asarray(state.cmean, dtype=np.float64)
        var = np.asarray(state.cvar, dtype=np.float64)
        prior = np.asarray(state.prior, dtype=np.float64)
        xf = np.asarray(x, dtype=np.float64)
        manual = np.log(prior) - 0.5 * np.sum(
            np.log(2.0 * math.pi * var) + (xf[None, :] - mu) ** 2 / var, axis=1
        )
        np.testing.assert_allclose(logits, manual, rtol=1e-5)

    def test_params_untouched_and_key_unused(self):
        spec = screening_spec("naive_bayes")
        params = init_mlp_params(jr.key(5), SMALL)
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        state = init_fn(params)
        x = jr.uniform(jr.key(6), (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
        y = jnp.array(3, jnp.int32)
        p_a, s_a, m_a = step_fn(params, state, x, y, jr.key(0))
        p_b, s_b, m_b = step_fn(params, state, x, y, jr.key(424242))
        for n in params:
            np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(params[n]))
            np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(p_b[n]))
        np.testing.assert_array_equal(np.asarray(s_a.cmean), np.asarray(s_b.cmean))
        assert float(m_a[1]) == float(m_b[1])

    def test_permutation_covariance(self):
        """Permuting the input features permutes the learned means and leaves
        the posterior (hence the accuracy stream) unchanged."""
        spec = screening_spec("naive_bayes")
        params = init_mlp_params(jr.key(8), SMALL)
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        perm = np.asarray(jr.permutation(jr.key(21), SMALL.input_dim))
        key = jr.key(31)
        state_raw = init_fn(params)
        state_perm = init_fn(params)
        for t in range(30):
            key, kx = jr.split(key)
            x = jr.uniform(kx, (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
            y = jnp.array(t % SMALL.n_classes, jnp.int32)
            _, state_raw, m_raw = step_fn(params, state_raw, x, y, jr.key(t))
            _, state_perm, m_perm = step_fn(params, state_perm, x[perm], y, jr.key(t))
            np.testing.assert_allclose(
                float(m_raw[1]), float(m_perm[1]), rtol=1e-4
            )
        np.testing.assert_array_equal(
            np.asarray(state_raw.cmean)[:, perm], np.asarray(state_perm.cmean)
        )
        np.testing.assert_array_equal(
            np.asarray(state_raw.cvar)[:, perm], np.asarray(state_perm.cvar)
        )

    def test_smoke_run_finite(self, small_data):
        x, y = small_data
        result = run_screening_config(
            x, y, screening_spec("naive_bayes"), seed=11, config=SMALL
        )
        assert np.all(np.isfinite(result.per_task_accuracy))
        assert np.all(np.isfinite(result.per_task_loss))
        assert np.all(result.per_task_accuracy >= 0.0)
        assert np.all(result.per_task_accuracy <= 1.0)


class TestDiscoveredRuleFactory:
    """Discovered-rule translation factory (rule_discovery -> screening arm).

    The factory materializes a rule-DSL genome (mechanism flags + constants)
    as a protocol-scale screening learner. Champion-form flags must reduce
    bit-exactly to the registered ``sigma0_shiftnorm_d099`` champion step.
    """

    def _hp(self, **overrides):
        from alberta_framework.benchmarks.ipmnist_screening import _discovered_rule_hp

        return _discovered_rule_hp(**overrides)

    def _setup(self, seed=0):
        from alberta_framework.benchmarks.upgd_ipmnist import init_mlp_params

        config = SMALL
        params = init_mlp_params(jr.key(seed), config)
        return config, params

    def test_champion_form_reduces_bitexact_to_shiftnorm_champion(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
            _make_upgd_shiftnorm_learner,
            _sigma0_ext_hp,
        )

        champ_hp = _sigma0_ext_hp(
            norm_decay=0.99,
            fast_decay=0.9,
            shift_k=1.0,
            shift_delta=0.02,
            shift_refractory=0.0,
        )
        champ_init, champ_step = _make_upgd_shiftnorm_learner(champ_hp)
        disc_init, disc_step = _make_discovered_rule_learner(
            self._hp(flag_norm=1.0, flag_shift_reset=1.0, flag_gate=1.0)
        )
        config, params = self._setup(seed=5)
        champ_params, disc_params = params, params
        champ_state, disc_state = champ_init(params), disc_init(params)
        key = jr.key(3)
        for step in range(12):
            key, kx = jr.split(key)
            x = jr.uniform(kx, (config.input_dim,), jnp.float32, -1.0, 1.0) * (
                1.0 + step % 3
            )
            y = jnp.array(step % config.n_classes, jnp.int32)
            champ_params, champ_state, m_champ = champ_step(
                champ_params, champ_state, x, y, jr.key(step)
            )
            disc_params, disc_state, m_disc = disc_step(
                disc_params, disc_state, x, y, jr.key(step)
            )
            for name in sorted(params):
                np.testing.assert_array_equal(
                    np.asarray(disc_params[name]), np.asarray(champ_params[name])
                )
            for a, b in zip(m_disc, m_champ):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_surprise_budget_scales_applied_delta(self):
        import dataclasses as _dc

        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
        )

        base_hp = self._hp(surprise_gain=1.0, weight_decay=0.0001)
        on_init, on_step = _make_discovered_rule_learner(
            {**base_hp, "flag_surprise_budget": 1.0}
        )
        off_init, off_step = _make_discovered_rule_learner(base_hp)
        config, params = self._setup(seed=6)
        surprised_on = _dc.replace(
            on_init(params),
            err_fast=jnp.asarray(4.0, jnp.float32),
            err_slow=jnp.asarray(1.0, jnp.float32),
        )
        surprised_off = _dc.replace(
            off_init(params),
            err_fast=jnp.asarray(4.0, jnp.float32),
            err_slow=jnp.asarray(1.0, jnp.float32),
        )
        x = jnp.linspace(0.1, 1.0, config.input_dim, dtype=jnp.float32)
        y = jnp.array(1, jnp.int32)
        stepped_on, _, _ = on_step(params, surprised_on, x, y, jr.key(0))
        stepped_off, _, _ = off_step(params, surprised_off, x, y, jr.key(0))
        delta_on = float(jnp.abs(stepped_on["w3"] - params["w3"]).sum())
        delta_off = float(jnp.abs(stepped_off["w3"] - params["w3"]).sum())
        assert delta_on == pytest.approx(4.0 * delta_off, rel=1e-3)

    def test_w1_shift_reset_restores_init_rows(self):
        import dataclasses as _dc

        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
        )

        init_fn, step_fn = _make_discovered_rule_learner(
            self._hp(flag_norm=1.0, flag_w1_shift_reset=1.0, shift_k=0.5)
        )
        config, params = self._setup(seed=7)
        drifted = {
            name: value + 0.25 if name == "w1" else value
            for name, value in params.items()
        }
        state = init_fn(drifted)
        mature = _dc.replace(
            state,
            init_params=dict(params),
            norm=_dc.replace(
                state.norm,
                mean=jnp.zeros(config.input_dim, jnp.float32),
                var=jnp.full((config.input_dim,), 1e-4, jnp.float32),
                count=jnp.full((config.input_dim,), 1000.0, jnp.float32),
            ),
            fast_mean=jnp.zeros(config.input_dim, jnp.float32),
        )
        x = jnp.full((config.input_dim,), 10.0, jnp.float32)
        y = jnp.array(0, jnp.int32)
        new_params, _, _ = step_fn(drifted, mature, x, y, jr.key(0))
        np.testing.assert_allclose(
            np.asarray(new_params["w1"]), np.asarray(params["w1"]), rtol=0, atol=1e-7
        )

    def test_decay_to_init_holds_w1_at_init_under_zero_input(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
        )

        init_fn, step_fn = _make_discovered_rule_learner(
            self._hp(flag_decay_to_init=1.0, weight_decay=0.03)
        )
        config, params = self._setup(seed=8)
        state = init_fn(params)
        x = jnp.zeros((config.input_dim,), jnp.float32)
        y = jnp.array(0, jnp.int32)
        new_params, _, _ = step_fn(params, state, x, y, jr.key(0))
        np.testing.assert_allclose(
            np.asarray(new_params["w1"]), np.asarray(params["w1"]), rtol=0, atol=1e-7
        )

    @pytest.mark.parametrize(
        ("name", "flags", "lr"),
        [
            (
                "disc_r1",
                {
                    "flag_norm": 1.0,
                    "flag_shift_reset": 1.0,
                    "flag_surprise_budget": 1.0,
                    "flag_hidden_rms": 1.0,
                    "flag_gate": 0.0,
                    "flag_decay_to_init": 0.0,
                    "flag_w1_shift_reset": 0.0,
                },
                0.0370901404621786,
            ),
            (
                "disc_r2",
                {
                    "flag_norm": 1.0,
                    "flag_shift_reset": 1.0,
                    "flag_decay_to_init": 1.0,
                    "flag_surprise_budget": 1.0,
                    "flag_hidden_rms": 1.0,
                    "flag_gate": 0.0,
                },
                0.04385333652867646,
            ),
            (
                "disc_r3",
                {
                    "flag_norm": 1.0,
                    "flag_shift_reset": 0.0,
                    "flag_decay_to_init": 1.0,
                    "flag_surprise_budget": 1.0,
                    "flag_w1_shift_reset": 1.0,
                    "flag_hidden_rms": 1.0,
                    "flag_gate": 0.0,
                },
                0.04512338013332415,
            ),
        ],
    )
    def test_discovered_arms_registered(self, name, flags, lr):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _ema_frozen_probe_input,
            _hidden_rms_frozen_probe_input,
            _make_discovered_rule_learner,
        )

        spec = screening_spec(name)
        assert spec.mechanism == "discovered_rule"
        assert spec.factory is _make_discovered_rule_learner
        expected_probe = (
            _hidden_rms_frozen_probe_input
            if flags["flag_hidden_rms"] != 0.0
            else _ema_frozen_probe_input
        )
        assert spec.frozen_probe_input is expected_probe
        assert spec.hyperparameters["step_size"] == pytest.approx(lr, rel=1e-12)
        for key, value in flags.items():
            assert spec.hyperparameters[key] == value
        # Smoke: one finite step at protocol-small scale.
        from alberta_framework.benchmarks.upgd_ipmnist import init_mlp_params

        params = init_mlp_params(jr.key(0), SMALL)
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        state = init_fn(params)
        x = jnp.linspace(-1.0, 1.0, SMALL.input_dim, dtype=jnp.float32)
        y = jnp.array(1, jnp.int32)
        new_params, _, metrics = step_fn(params, state, x, y, jr.key(1))
        for name_ in sorted(new_params):
            assert bool(jnp.all(jnp.isfinite(new_params[name_])))
        assert all(bool(jnp.isfinite(m)) for m in metrics)

    @pytest.mark.parametrize(
        ("name", "rms"),
        [("disc_r1_pscale", 1.0), ("disc_r1_pscale_norms", 0.0)],
    )
    def test_disc_r1_protocol_scale_diagnostics_registered(self, name, rms):
        """Structure-vs-constants dissection of the discovered rule disc_r1:
        same flags (surprise budget, no gate) at champion-scale constants,
        with hidden RMS isolated as its own axis."""
        spec = screening_spec(name)
        assert spec.mechanism == "discovered_rule_diagnostic"
        hp = spec.hyperparameters
        assert hp["flag_surprise_budget"] == 1.0
        assert hp["flag_gate"] == 0.0
        assert hp["flag_hidden_rms"] == rms
        # Champion-scale constants replace the micro-tuned ones.
        assert hp["step_size"] == pytest.approx(0.01)
        assert hp["weight_decay"] == pytest.approx(0.01)
        assert hp["norm_decay"] == pytest.approx(0.99)
        assert hp["fast_decay"] == pytest.approx(0.9)
        assert hp["shift_k"] == pytest.approx(1.0)
        # The discovered surprise constants carry over unchanged.
        assert hp["surprise_gain"] == pytest.approx(0.8360796272754669)


class TestDiscoveredRuleFactoryWave2:
    """Wave-2 mechanism classes of the discovered-rule translation factory.

    The expanded rule-DSL genome (RLS ensemble head, naive-Bayes ensemble
    member, surprise-driven lr annealing, per-layer lr ratio, Kalman-style
    normalizer) must translate to protocol scale with build-time (Python
    level) composition: every new flag left at 0.0 leaves the traced champion
    step untouched (the bit-exact reduction pin above still passes), and each
    mechanism expresses its rule-DSL semantics when enabled.
    """

    def _hp(self, **overrides):
        from alberta_framework.benchmarks.ipmnist_screening import _discovered_rule_hp

        return _discovered_rule_hp(**overrides)

    def _setup(self, seed=0):
        from alberta_framework.benchmarks.upgd_ipmnist import init_mlp_params

        config = SMALL
        params = init_mlp_params(jr.key(seed), config)
        return config, params

    def test_wave2_defaults_present_and_off(self):
        hp = self._hp()
        for key in (
            "flag_rls_head",
            "flag_rls_reset_p",
            "flag_nb_member",
            "flag_lr_anneal",
            "flag_layer_lr",
            "flag_kalman_norm",
        ):
            assert hp[key] == 0.0
        for key in (
            "rls_lambda",
            "nb_decay",
            "vote_decay",
            "anneal_lo",
            "anneal_hi",
            "layer_lr_ratio",
            "kalman_q",
        ):
            assert key in hp

    def test_lr_anneal_scales_step_with_error_ratio(self):
        import dataclasses as _dc

        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
        )

        base_hp = self._hp(weight_decay=0.0001, anneal_lo=0.25, anneal_hi=2.0)
        on_init, on_step = _make_discovered_rule_learner(
            {**base_hp, "flag_lr_anneal": 1.0}
        )
        off_init, off_step = _make_discovered_rule_learner(base_hp)
        config, params = self._setup(seed=21)
        x = jnp.linspace(0.2, 1.0, config.input_dim, dtype=jnp.float32)
        y = jnp.array(1, jnp.int32)

        def _with_errors(state, fast, slow):
            return _dc.replace(
                state,
                err_fast=jnp.asarray(fast, jnp.float32),
                err_slow=jnp.asarray(slow, jnp.float32),
            )

        hot, _, _ = on_step(params, _with_errors(on_init(params), 4.0, 1.0), x, y, jr.key(0))
        calm, _, _ = on_step(params, _with_errors(on_init(params), 1.0, 1.0), x, y, jr.key(0))
        base, _, _ = off_step(params, _with_errors(off_init(params), 1.0, 1.0), x, y, jr.key(0))
        delta_hot = float(jnp.abs(hot["w3"] - params["w3"]).sum())
        delta_calm = float(jnp.abs(calm["w3"] - params["w3"]).sum())
        delta_base = float(jnp.abs(base["w3"] - params["w3"]).sum())
        assert delta_hot == pytest.approx(2.0 * delta_base, rel=1e-3)
        assert delta_calm == pytest.approx(0.25 * delta_base, rel=1e-3)

    def test_layer_lr_ratio_scales_head_vs_input(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
        )

        base_hp = self._hp(weight_decay=0.0001, layer_lr_ratio=2.0)
        on_init, on_step = _make_discovered_rule_learner(
            {**base_hp, "flag_layer_lr": 1.0}
        )
        off_init, off_step = _make_discovered_rule_learner(base_hp)
        config, params = self._setup(seed=22)
        x = jnp.linspace(0.2, 1.0, config.input_dim, dtype=jnp.float32)
        y = jnp.array(0, jnp.int32)
        stepped, _, _ = on_step(params, on_init(params), x, y, jr.key(0))
        stepped_base, _, _ = off_step(params, off_init(params), x, y, jr.key(0))
        head = float(jnp.abs(stepped["w3"] - params["w3"]).sum())
        head_base = float(jnp.abs(stepped_base["w3"] - params["w3"]).sum())
        w1 = float(jnp.abs(stepped["w1"] - params["w1"]).sum())
        w1_base = float(jnp.abs(stepped_base["w1"] - params["w1"]).sum())
        assert head == pytest.approx(2.0 * head_base, rel=1e-3)
        assert w1 == pytest.approx(0.5 * w1_base, rel=1e-3)

    def test_nb_member_updates_only_observed_class(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
        )

        init_fn, step_fn = _make_discovered_rule_learner(
            self._hp(flag_nb_member=1.0)
        )
        config, params = self._setup(seed=23)
        state = init_fn(params)
        x = jnp.linspace(0.5, 2.0, config.input_dim, dtype=jnp.float32)
        y = jnp.array(2, jnp.int32)
        _, new_state, metrics = step_fn(params, state, x, y, jr.key(0))
        moved = np.abs(np.asarray(new_state.nb_mean - state.nb_mean)).sum(axis=1)
        assert moved[2] > 0.0
        for klass in (0, 1, 3, 4):
            assert moved[klass] == pytest.approx(0.0, abs=1e-8)
        assert float(new_state.nb_count[2]) == pytest.approx(1.0)
        assert all(bool(jnp.isfinite(m)) for m in metrics)

    def test_rls_head_updates_and_shift_reset_p(self):
        import dataclasses as _dc

        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
        )

        init_fn, step_fn = _make_discovered_rule_learner(
            self._hp(
                flag_norm=1.0, flag_rls_head=1.0, flag_rls_reset_p=1.0, shift_k=0.5
            )
        )
        config, params = self._setup(seed=24)
        state = init_fn(params)
        x = jnp.linspace(0.1, 1.0, config.input_dim, dtype=jnp.float32)
        y = jnp.array(3, jnp.int32)
        _, stepped_state, _ = step_fn(params, state, x, y, jr.key(0))
        assert float(jnp.abs(stepped_state.rls_w).sum()) > 0.0
        assert bool(jnp.all(jnp.isfinite(stepped_state.rls_p)))
        mature = _dc.replace(
            stepped_state,
            norm=_dc.replace(
                stepped_state.norm,
                mean=jnp.zeros(config.input_dim, jnp.float32),
                var=jnp.full((config.input_dim,), 1e-4, jnp.float32),
                count=jnp.full((config.input_dim,), 1000.0, jnp.float32),
            ),
            fast_mean=jnp.zeros(config.input_dim, jnp.float32),
        )
        x_shift = jnp.full((config.input_dim,), 10.0, jnp.float32)
        _, reset_state, _ = step_fn(params, mature, x_shift, y, jr.key(0))
        off_diag = reset_state.rls_p - jnp.diag(jnp.diag(reset_state.rls_p))
        np.testing.assert_allclose(np.asarray(off_diag), 0.0, atol=1e-6)

    def test_kalman_norm_gain_tracks_uncertainty(self):
        import dataclasses as _dc

        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_discovered_rule_learner,
        )

        init_fn, step_fn = _make_discovered_rule_learner(
            self._hp(flag_norm=1.0, flag_kalman_norm=1.0, kalman_q=0.001)
        )
        config, params = self._setup(seed=25)
        state = init_fn(params)
        x = jnp.full((config.input_dim,), 5.0, jnp.float32)
        y = jnp.array(0, jnp.int32)
        confident = _dc.replace(
            state,
            norm=_dc.replace(
                state.norm, count=jnp.full((config.input_dim,), 1000.0, jnp.float32)
            ),
            kalman_p=jnp.full((config.input_dim,), 1e-4, jnp.float32),
        )
        uncertain = _dc.replace(
            state,
            norm=_dc.replace(
                state.norm, count=jnp.full((config.input_dim,), 1000.0, jnp.float32)
            ),
            kalman_p=jnp.full((config.input_dim,), 10.0, jnp.float32),
        )
        _, state_conf, _ = step_fn(params, confident, x, y, jr.key(0))
        _, state_unc, _ = step_fn(params, uncertain, x, y, jr.key(0))
        assert float(state_unc.norm.mean[0]) > float(state_conf.norm.mean[0])
        assert float(state_unc.kalman_p[0]) < 10.0


class TestRLSHead:
    """Convergence-shortfall attack: champion body + RLS readout.

    ``_make_rls_head_learner`` keeps the ``sigma0_shiftnorm_d099`` champion
    body (shift-adaptive EMA-norm d099 + utility-gated sigma-0 SGD) and
    replaces the deployed readout with streaming recursive least squares on
    the 150-dim penultimate ReLU features (bias-augmented, one-vs-all
    one-hot regression, argmax prediction).  Design decisions probed:
    (a) one-hot LS regression + argmax (standard streaming practice;
    softmax/logistic targets admit no exact RLS recursion), (b) forgetting
    factor ``rls_lambda`` plus optional detector-driven P resets reusing the
    champion's own shift detector, (c) body error signal — parallel champion
    SGD head (``head_resid=0``, safer; body trajectory bit-exact champion)
    vs the RLS head's own residual (``head_resid=1``, cleanest).
    """

    def _hp(self, **overrides):
        from alberta_framework.benchmarks.ipmnist_screening import _rls_head_hp

        return _rls_head_hp(**overrides)

    def _factory(self, **overrides):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_rls_head_learner,
        )

        return _make_rls_head_learner(self._hp(**overrides))

    def _champion_factory(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_upgd_shiftnorm_learner,
            _sigma0_ext_hp,
        )

        return _make_upgd_shiftnorm_learner(
            _sigma0_ext_hp(
                norm_decay=0.99,
                fast_decay=0.9,
                shift_k=1.0,
                shift_delta=0.02,
                shift_refractory=0.0,
            )
        )

    def _stream(self, n_steps=12, seed=3):
        key = jr.key(seed)
        xs, ys = [], []
        for step in range(n_steps):
            key, kx = jr.split(key)
            xs.append(
                jr.uniform(kx, (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
                * (1.0 + step % 3)
            )
            ys.append(jnp.array(step % SMALL.n_classes, jnp.int32))
        return xs, ys

    def _learnable_stream(self, n_examples=400, seed=88):
        kx, kw = jr.split(jr.key(seed))
        x = jr.uniform(kx, (n_examples, SMALL.input_dim), jnp.float32, -1.0, 1.0)
        w_true = jr.normal(kw, (SMALL.input_dim, SMALL.n_classes), jnp.float32)
        y = jnp.argmax(x @ w_true, axis=1).astype(jnp.int32)
        return np.asarray(x), np.asarray(y)

    def test_parallel_body_is_bitexact_champion(self):
        """head_resid=0: all six MLP tensors and the normalizer state follow
        the sigma0_shiftnorm_d099 champion bit-for-bit; only the deployed
        readout (metrics) differs."""
        champ_init, champ_step = self._champion_factory()
        rls_init, rls_step = self._factory()
        params = init_mlp_params(jr.key(5), SMALL)
        champ_params, rls_params = params, params
        champ_state, rls_state = champ_init(params), rls_init(params)
        xs, ys = self._stream()
        for step, (x, y) in enumerate(zip(xs, ys)):
            champ_params, champ_state, _ = champ_step(
                champ_params, champ_state, x, y, jr.key(step)
            )
            rls_params, rls_state, _ = rls_step(
                rls_params, rls_state, x, y, jr.key(step)
            )
            for name in sorted(params):
                np.testing.assert_array_equal(
                    np.asarray(rls_params[name]), np.asarray(champ_params[name]), name
                )
            np.testing.assert_array_equal(
                np.asarray(rls_state.norm.mean), np.asarray(champ_state.norm.mean)
            )
            np.testing.assert_array_equal(
                np.asarray(rls_state.norm.count), np.asarray(champ_state.norm.count)
            )

    def test_gate_scale_one_is_bitexact_legacy_incumbent(self):
        """The explicit gate-on endpoint must preserve the pre-ablation
        incumbent trajectory, including every carried state and metric bit."""
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_rls_head_learner,
        )

        incumbent = {
            "rls_lambda": 1.0,
            "rls_reset_frac": 0.05,
            "head_resid": 1.0,
        }
        legacy_hp = self._hp(**incumbent)
        assert legacy_hp["gate_scale"] == 1.0
        legacy_hp.pop("gate_scale")
        legacy_init, legacy_step = _make_rls_head_learner(legacy_hp)
        gated_init, gated_step = self._factory(**incumbent, gate_scale=1.0)
        params = init_mlp_params(jr.key(31), SMALL)
        legacy_params = gated_params = params
        legacy_state, gated_state = legacy_init(params), gated_init(params)
        xs, ys = self._stream(n_steps=10, seed=37)
        for step, (x, y) in enumerate(zip(xs, ys)):
            legacy_params, legacy_state, legacy_metrics = legacy_step(
                legacy_params, legacy_state, x, y, jr.key(step)
            )
            gated_params, gated_state, gated_metrics = gated_step(
                gated_params, gated_state, x, y, jr.key(step)
            )
            for name in sorted(params):
                np.testing.assert_array_equal(
                    np.asarray(gated_params[name]), np.asarray(legacy_params[name]), name
                )
                np.testing.assert_array_equal(
                    np.asarray(gated_state.utility[name]),
                    np.asarray(legacy_state.utility[name]),
                    name,
                )
            for field in ("step", "fast_mean", "p", "wout"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(gated_state, field)),
                    np.asarray(getattr(legacy_state, field)),
                    field,
                )
            for field in ("mean", "var", "count"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(gated_state.norm, field)),
                    np.asarray(getattr(legacy_state.norm, field)),
                    f"norm.{field}",
                )
            for actual, expected in zip(gated_metrics, legacy_metrics):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    @pytest.mark.parametrize("gate_scale", [-1.0, 0.25, 0.5, 2.0, math.nan])
    def test_gate_scale_rejects_non_endpoint_values(self, gate_scale):
        with pytest.raises(ValueError, match="gate_scale"):
            self._factory(head_resid=1.0, gate_scale=gate_scale)

    def test_gate_off_requires_residual_body(self):
        with pytest.raises(ValueError, match="residual"):
            self._factory(gate_scale=0.0)

    def test_gate_off_two_step_plain_sgd_and_identical_head(self):
        """The ablation does only decayed SGD on the residual body.

        Step one has a zero residual-body gradient because the readout starts
        at zero, so the gated incumbent and ablation enter step two with the
        same body.  Their RLS transitions must therefore remain bit-identical
        for both steps, while the second ablation body update follows the
        explicit chain-rule gradient and carries no utility clock or EMA.
        """
        params = {
            "w1": jnp.full((SMALL.input_dim, SMALL.hidden1), 0.01, jnp.float32),
            "b1": jnp.full((SMALL.hidden1,), 0.5, jnp.float32),
            "w2": jnp.full((SMALL.hidden1, SMALL.hidden2), 0.02, jnp.float32),
            "b2": jnp.full((SMALL.hidden2,), 0.5, jnp.float32),
            "w3": jnp.full((SMALL.hidden2, SMALL.n_classes), 0.03, jnp.float32),
            "b3": jnp.full((SMALL.n_classes,), 0.04, jnp.float32),
        }
        overrides = {
            "rls_lambda": 1.0,
            "rls_reset_frac": 0.05,
            "head_resid": 1.0,
        }
        plain_init, plain_step = self._factory(**overrides, gate_scale=0.0)
        gated_init, gated_step = self._factory(**overrides, gate_scale=1.0)
        plain_state, gated_state = plain_init(params), gated_init(params)
        plain_params = gated_params = params
        expected_params = {name: jnp.asarray(value) for name, value in params.items()}
        expected_norm = plain_state.norm
        expected_fast = plain_state.fast_mean
        expected_p = plain_state.p
        expected_wout = plain_state.wout
        hp = self._hp(**overrides, gate_scale=0.0)
        decay = jnp.asarray(
            1.0 - hp["step_size"] * hp["weight_decay"], jnp.float32
        )
        scale = jnp.asarray(
            1.0 / math.sqrt(SMALL.hidden2 + 1), dtype=jnp.float32
        )
        xs = (
            jnp.linspace(0.2, 1.3, SMALL.input_dim, dtype=jnp.float32),
            jnp.linspace(1.1, 0.1, SMALL.input_dim, dtype=jnp.float32),
        )
        ys = (jnp.array(0, jnp.int32), jnp.array(1, jnp.int32))

        for step, (x, y) in enumerate(zip(xs, ys)):
            x_norm, expected_norm, expected_fast, shifted = shift_adaptive_normalize(
                expected_norm,
                expected_fast,
                x,
                decay=hp["norm_decay"],
                fast_decay=hp["fast_decay"],
                epsilon=hp["norm_epsilon"],
                shift_k=hp["shift_k"],
                shift_delta=hp["shift_delta"],
                shift_refractory=hp["shift_refractory"],
            )
            assert not bool(jnp.any(shifted))
            z1 = x_norm @ expected_params["w1"] + expected_params["b1"]
            a1 = jax.nn.relu(z1)
            z2 = a1 @ expected_params["w2"] + expected_params["b2"]
            a2 = jax.nn.relu(z2)
            phi = jnp.concatenate([a2 * scale, jnp.ones((1,), jnp.float32)])
            logits = expected_wout.T @ phi
            target = jax.nn.one_hot(y, SMALL.n_classes, dtype=jnp.float32)
            dlogits = logits - target
            dphi = expected_wout @ dlogits
            dz2 = (dphi[:-1] * scale) * (z2 > 0.0)
            grads = {
                "w2": jnp.outer(a1, dz2),
                "b2": dz2,
            }
            dz1 = (expected_params["w2"] @ dz2) * (z1 > 0.0)
            grads["w1"] = jnp.outer(x_norm, dz1)
            grads["b1"] = dz1
            for name in ("w1", "b1", "w2", "b2"):
                expected_params[name] = (
                    expected_params[name] * decay - hp["step_size"] * grads[name]
                )

            err = target - logits
            pp = expected_p @ phi
            gain = pp / (hp["rls_lambda"] + phi @ pp)
            expected_wout = expected_wout + jnp.outer(gain, err)
            expected_p = (expected_p - jnp.outer(gain, pp)) / hp["rls_lambda"]
            expected_p = 0.5 * (expected_p + expected_p.T)

            plain_params, plain_state, plain_metrics = plain_step(
                plain_params, plain_state, x, y, jr.key(step)
            )
            gated_params, gated_state, gated_metrics = gated_step(
                gated_params, gated_state, x, y, jr.key(step)
            )
            for name in ("w1", "b1", "w2", "b2"):
                np.testing.assert_allclose(
                    np.asarray(plain_params[name]),
                    np.asarray(expected_params[name]),
                    rtol=0.0,
                    atol=2e-7,
                    err_msg=name,
                )
            for name in ("w3", "b3"):
                np.testing.assert_array_equal(
                    np.asarray(plain_params[name]), np.asarray(params[name]), name
                )
            np.testing.assert_array_equal(
                np.asarray(plain_state.p), np.asarray(expected_p)
            )
            np.testing.assert_array_equal(
                np.asarray(plain_state.wout), np.asarray(expected_wout)
            )
            np.testing.assert_array_equal(
                np.asarray(plain_state.p), np.asarray(gated_state.p)
            )
            np.testing.assert_array_equal(
                np.asarray(plain_state.wout), np.asarray(gated_state.wout)
            )
            np.testing.assert_array_equal(
                np.asarray(plain_state.fast_mean), np.asarray(expected_fast)
            )
            np.testing.assert_array_equal(
                np.asarray(plain_state.fast_mean), np.asarray(gated_state.fast_mean)
            )
            for field in ("mean", "var", "count"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(plain_state.norm, field)),
                    np.asarray(getattr(expected_norm, field)),
                    f"norm.{field}",
                )
                np.testing.assert_array_equal(
                    np.asarray(getattr(plain_state.norm, field)),
                    np.asarray(getattr(gated_state.norm, field)),
                    f"gated norm.{field}",
                )
            for actual, expected in zip(plain_metrics, gated_metrics):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            assert int(plain_state.step) == 0
            for name in sorted(params):
                assert not np.any(np.asarray(plain_state.utility[name])), name

        assert not np.array_equal(
            np.asarray(plain_params["w1"]), np.asarray(gated_params["w1"])
        )

    def test_infinite_ridge_is_frozen_degenerate_head(self):
        """Reduction pin: rls_ridge_init=inf gives P=0 exactly, so the head
        never updates (wout stays 0), every prediction is the constant argmax
        of the zero vector (class 0) — a measurable degenerate — while the
        parallel body still trains exactly like the champion."""
        champ_init, champ_step = self._champion_factory()
        rls_init, rls_step = self._factory(rls_ridge_init=math.inf)
        params = init_mlp_params(jr.key(6), SMALL)
        champ_params, rls_params = params, params
        champ_state, rls_state = champ_init(params), rls_init(params)
        assert not np.any(np.asarray(rls_state.p))
        xs, ys = self._stream(n_steps=8, seed=11)
        for step, (x, y) in enumerate(zip(xs, ys)):
            champ_params, champ_state, _ = champ_step(
                champ_params, champ_state, x, y, jr.key(step)
            )
            rls_params, rls_state, metrics = rls_step(
                rls_params, rls_state, x, y, jr.key(step)
            )
            assert not np.any(np.asarray(rls_state.p))
            assert not np.any(np.asarray(rls_state.wout))
            expected_acc = 1.0 if int(y) == 0 else 0.0
            assert float(metrics[0]) == expected_acc
            assert all(bool(jnp.isfinite(m)) for m in metrics)
            for name in sorted(params):
                np.testing.assert_array_equal(
                    np.asarray(rls_params[name]), np.asarray(champ_params[name]), name
                )

    def test_lambda1_frozen_body_matches_closed_form_ridge(self):
        """RLS exactness pin: with lambda=1 and a frozen body (step_size=0)
        the streaming head equals closed-form ridge regression on the exact
        phi sequence it consumed."""
        from alberta_framework.benchmarks.ipmnist_screening import (
            EMANormState,
            shift_adaptive_normalize,
        )

        hp = self._hp(rls_lambda=1.0, step_size=0.0)
        init_fn, step_fn = self._factory(rls_lambda=1.0, step_size=0.0)
        params = init_mlp_params(jr.key(7), SMALL)
        state = init_fn(params)
        m = SMALL.hidden2 + 1
        scale = 1.0 / math.sqrt(m)
        norm = EMANormState(
            mean=jnp.zeros(SMALL.input_dim, jnp.float32),
            var=jnp.ones(SMALL.input_dim, jnp.float32),
            count=jnp.zeros(SMALL.input_dim, jnp.float32),
        )
        fast_mean = jnp.zeros(SMALL.input_dim, jnp.float32)
        phis, targets = [], []
        xs, ys = self._stream(n_steps=40, seed=13)
        for step, (x, y) in enumerate(zip(xs, ys)):
            x_norm, norm, fast_mean, _ = shift_adaptive_normalize(
                norm, fast_mean, x,
                decay=hp["norm_decay"],
                fast_decay=hp["fast_decay"],
                epsilon=hp["norm_epsilon"],
                shift_k=hp["shift_k"],
                shift_delta=hp["shift_delta"],
                shift_refractory=hp["shift_refractory"],
            )
            a1 = jax.nn.relu(x_norm @ params["w1"] + params["b1"])
            a2 = jax.nn.relu(a1 @ params["w2"] + params["b2"])
            phis.append(
                np.concatenate([np.asarray(a2) * scale, np.ones(1, np.float32)])
            )
            targets.append(np.eye(SMALL.n_classes, dtype=np.float32)[int(y)])
            params, state, _ = step_fn(params, state, x, y, jr.key(step))
        phi_mat = np.stack(phis).astype(np.float64)
        target_mat = np.stack(targets).astype(np.float64)
        ridge = float(self._hp()["rls_ridge_init"])
        closed_form = np.linalg.solve(
            ridge * np.eye(m) + phi_mat.T @ phi_mat, phi_mat.T @ target_mat
        )
        np.testing.assert_allclose(
            np.asarray(state.wout, dtype=np.float64), closed_form, atol=2e-3
        )

    def test_p_reset_untriggerable_is_bitexact_off(self):
        """rls_reset_frac > 1 can never fire (the shifted fraction is <= 1),
        and must be bitwise the plain no-reset path."""
        init_off, step_off = self._factory()
        init_on, step_on = self._factory(rls_reset_frac=2.0)
        params = init_mlp_params(jr.key(9), SMALL)
        state_off, state_on = init_off(params), init_on(params)
        params_off = params_on = params
        xs, ys = self._stream(n_steps=10, seed=17)
        for step, (x, y) in enumerate(zip(xs, ys)):
            params_off, state_off, m_off = step_off(
                params_off, state_off, x, y, jr.key(step)
            )
            params_on, state_on, m_on = step_on(
                params_on, state_on, x, y, jr.key(step)
            )
            np.testing.assert_array_equal(
                np.asarray(state_on.p), np.asarray(state_off.p)
            )
            np.testing.assert_array_equal(
                np.asarray(state_on.wout), np.asarray(state_off.wout)
            )
            for a, b in zip(m_on, m_off):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_p_reset_fires_on_detected_shift_and_keeps_wout(self):
        """A mature normalizer hit with a far-shifted input trips the
        detector on every feature; with the reset enabled P returns exactly
        to eye/ridge while the readout weights are kept."""
        import dataclasses as _dc

        init_fn, step_fn = self._factory(rls_reset_frac=0.5)
        params = init_mlp_params(jr.key(10), SMALL)
        state = init_fn(params)
        # Run a few normal steps so wout and P move off their init.
        xs, ys = self._stream(n_steps=5, seed=19)
        for step, (x, y) in enumerate(zip(xs, ys)):
            params, state, _ = step_fn(params, state, x, y, jr.key(step))
        assert np.any(np.asarray(state.wout))
        m = SMALL.hidden2 + 1
        ridge = float(self._hp()["rls_ridge_init"])
        assert not np.allclose(
            np.asarray(state.p), np.eye(m, dtype=np.float32) / ridge
        )
        mature = _dc.replace(
            state,
            norm=_dc.replace(
                state.norm,
                mean=jnp.zeros(SMALL.input_dim, jnp.float32),
                var=jnp.full((SMALL.input_dim,), 1e-4, jnp.float32),
                count=jnp.full((SMALL.input_dim,), 1000.0, jnp.float32),
            ),
            fast_mean=jnp.zeros(SMALL.input_dim, jnp.float32),
        )
        wout_before = np.asarray(mature.wout)
        x_shift = jnp.full((SMALL.input_dim,), 10.0, jnp.float32)
        _, shifted_state, _ = step_fn(
            params, mature, x_shift, jnp.array(0, jnp.int32), jr.key(99)
        )
        np.testing.assert_array_equal(
            np.asarray(shifted_state.p), np.eye(m, dtype=np.float32) / ridge
        )
        # wout was updated by the step (kept + one RLS update), not zeroed.
        assert np.any(np.asarray(shifted_state.wout))
        assert not np.array_equal(np.asarray(shifted_state.wout), wout_before)

    def test_resid_trains_body_and_never_touches_sgd_head(self):
        """head_resid=1: the body error signal is the RLS residual — body
        tensors move off the pure-decay path once wout is nonzero, while
        w3/b3 (no SGD head exists) pass through bitwise untouched."""
        init_fn, step_fn = self._factory(head_resid=1.0)
        params = init_mlp_params(jr.key(12), SMALL)
        state = init_fn(params)
        hp = self._hp()
        param_decay = 1.0 - hp["step_size"] * hp["weight_decay"]
        decay_only = {k: jnp.asarray(v) for k, v in params.items()}
        run_params = params
        xs, ys = self._stream(n_steps=6, seed=23)
        for step, (x, y) in enumerate(zip(xs, ys)):
            run_params, state, metrics = step_fn(run_params, state, x, y, jr.key(step))
            decay_only = {
                k: (v * param_decay if k in ("w1", "b1", "w2", "b2") else v)
                for k, v in decay_only.items()
            }
            assert all(bool(jnp.isfinite(m)) for m in metrics)
        np.testing.assert_array_equal(
            np.asarray(run_params["w3"]), np.asarray(params["w3"])
        )
        np.testing.assert_array_equal(
            np.asarray(run_params["b3"]), np.asarray(params["b3"])
        )
        assert np.any(np.asarray(state.wout))
        assert not np.allclose(
            np.asarray(run_params["w1"]), np.asarray(decay_only["w1"]), atol=1e-7
        )

    def test_resid_frozen_head_is_pure_decay_no_nan(self):
        """Guard pin: head_resid=1 with rls_ridge_init=inf keeps wout at 0,
        so the residual gradient is exactly zero — the zero-utility gate must
        be guarded (0.5, not NaN) and the body follows pure decoupled decay
        bit-for-bit."""
        init_fn, step_fn = self._factory(head_resid=1.0, rls_ridge_init=math.inf)
        params = init_mlp_params(jr.key(14), SMALL)
        state = init_fn(params)
        hp = self._hp()
        param_decay = jnp.asarray(
            1.0 - hp["step_size"] * hp["weight_decay"], jnp.float32
        )
        expected = {k: jnp.asarray(v) for k, v in params.items()}
        run_params = params
        xs, ys = self._stream(n_steps=4, seed=29)
        for step, (x, y) in enumerate(zip(xs, ys)):
            run_params, state, metrics = step_fn(run_params, state, x, y, jr.key(step))
            expected = {
                k: (v * param_decay if k in ("w1", "b1", "w2", "b2") else v)
                for k, v in expected.items()
            }
            for name in sorted(params):
                assert bool(jnp.all(jnp.isfinite(run_params[name]))), name
                np.testing.assert_array_equal(
                    np.asarray(run_params[name]), np.asarray(expected[name]), name
                )
            assert all(bool(jnp.isfinite(m)) for m in metrics)

    def test_key_is_unused_in_both_modes(self):
        for overrides in ({}, {"head_resid": 1.0}):
            init_fn, step_fn = self._factory(**overrides)
            params = init_mlp_params(jr.key(15), SMALL)
            state = init_fn(params)
            x = jnp.linspace(-1.0, 1.0, SMALL.input_dim, dtype=jnp.float32)
            y = jnp.array(2, jnp.int32)
            params_a, state_a, m_a = step_fn(params, state, x, y, jr.key(0))
            params_b, state_b, m_b = step_fn(params, state, x, y, jr.key(987654))
            for name in sorted(params):
                np.testing.assert_array_equal(
                    np.asarray(params_a[name]), np.asarray(params_b[name])
                )
            np.testing.assert_array_equal(
                np.asarray(state_a.wout), np.asarray(state_b.wout)
            )
            for a, b in zip(m_a, m_b):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_state_shapes_and_p_symmetry(self):
        init_fn, step_fn = self._factory()
        params = init_mlp_params(jr.key(16), SMALL)
        state = init_fn(params)
        m = SMALL.hidden2 + 1
        assert state.p.shape == (m, m)
        assert state.wout.shape == (m, SMALL.n_classes)
        assert state.p.dtype == jnp.float32
        ridge = float(self._hp()["rls_ridge_init"])
        np.testing.assert_array_equal(
            np.asarray(state.p), np.eye(m, dtype=np.float32) / ridge
        )
        assert not np.any(np.asarray(state.wout))
        x = jr.normal(jr.key(30), (SMALL.input_dim,))
        _, new_state, _ = step_fn(params, state, x, jnp.array(1, jnp.int32), jr.key(0))
        np.testing.assert_array_equal(
            np.asarray(new_state.p), np.asarray(new_state.p).T
        )
        assert bool(jnp.all(jnp.isfinite(new_state.p)))

    def test_smoke_runs_above_chance_all_body_modes(self):
        x, y = self._learnable_stream()
        config = IPMNISTConfig(
            n_tasks=2, task_length=200, input_dim=12, hidden1=8, hidden2=6, n_classes=5
        )
        from alberta_framework.benchmarks.ipmnist_screening import _rls_head_hp

        for overrides in (
            {},
            {"head_resid": 1.0},
            {"head_resid": 1.0, "gate_scale": 0.0},
        ):
            spec = ScreeningSpec(
                name="rls_head_smoke",
                base_learner="upgd_w",
                mechanism="rls_readout",
                hyperparameters=_rls_head_hp(**overrides),
                factory=__import__(
                    "alberta_framework.benchmarks.ipmnist_screening",
                    fromlist=["_make_rls_head_learner"],
                )._make_rls_head_learner,
            )
            result = run_screening_config(x, y, spec, seed=2, config=config)
            acc = np.asarray(result.per_task_accuracy)
            assert np.all(np.isfinite(acc)), overrides
            assert np.all(np.isfinite(np.asarray(result.per_task_loss))), overrides
            plas = np.asarray(result.per_task_plasticity)
            assert np.all((plas >= 0.0) & (plas <= 1.0)), overrides
            assert float(acc.mean()) > 1.0 / config.n_classes, overrides

    def test_frozen_probe_fails_closed(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _rls_head_frozen_probe_input,
        )

        init_fn, _ = self._factory()
        state = init_fn(init_mlp_params(jr.key(0), SMALL))
        with pytest.raises(NotImplementedError, match="rls_head"):
            _rls_head_frozen_probe_input(
                state, jnp.zeros((3, SMALL.input_dim)), self._hp()
            )

    def test_p_trace_cap_zero_is_bitexact_off(self):
        """rls_p_trace_cap=0 disables the cap at build time and must be
        bitwise the uncapped path."""
        init_off, step_off = self._factory()
        init_on, step_on = self._factory(rls_p_trace_cap=0.0)
        params = init_mlp_params(jr.key(21), SMALL)
        state_off, state_on = init_off(params), init_on(params)
        params_off = params_on = params
        xs, ys = self._stream(n_steps=8, seed=31)
        for step, (x, y) in enumerate(zip(xs, ys)):
            params_off, state_off, m_off = step_off(
                params_off, state_off, x, y, jr.key(step)
            )
            params_on, state_on, m_on = step_on(
                params_on, state_on, x, y, jr.key(step)
            )
            np.testing.assert_array_equal(
                np.asarray(state_on.p), np.asarray(state_off.p)
            )
            for a, b in zip(m_on, m_off):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_p_trace_cap_rescales_oversized_p(self):
        """Covariance wind-up guard: when trace(P) exceeds the cap after the
        RLS update, P is rescaled to trace == cap exactly (direction
        preserved); an under-cap P passes through untouched."""
        import dataclasses as _dc

        cap = 3.0
        init_fn, step_fn = self._factory(rls_p_trace_cap=cap)
        params = init_mlp_params(jr.key(22), SMALL)
        state = init_fn(params)
        m = SMALL.hidden2 + 1
        blown = _dc.replace(
            state, p=jnp.eye(m, dtype=jnp.float32) * 1e6
        )
        x = jr.normal(jr.key(41), (SMALL.input_dim,))
        y = jnp.array(1, jnp.int32)
        _, capped_state, metrics = step_fn(params, blown, x, y, jr.key(0))
        trace = float(jnp.trace(capped_state.p))
        assert trace == pytest.approx(cap, rel=1e-4)
        assert all(bool(jnp.isfinite(v)) for v in metrics)
        # Under the cap (cap far above the trajectory's traces): bitwise
        # identical to the uncapped step (scale is exactly 1.0).
        init_off, step_off = self._factory()
        _, big_cap_step = self._factory(rls_p_trace_cap=1e6)
        _, plain_state, _ = step_off(params, state, x, y, jr.key(0))
        _, small_state, _ = big_cap_step(params, state, x, y, jr.key(0))
        np.testing.assert_array_equal(
            np.asarray(small_state.p), np.asarray(plain_state.p)
        )

    def test_registry_arms(self):
        """Screen wave frozen by the 2-task seed-0 diagnostic (champion
        reference 0.825 in the same loop): lambda star {0.995, 0.999, 1.0}
        (2-task means .8302/.8361/.8217 — the head helps at every lambda's
        task-2 and lambda 0.999 wins overall), the detector-driven P reset at
        the calibrated 0.05 fraction (within-task shifted fraction <= 0.018,
        boundary step 0.061 — 2.8x margin; 2-task read -0.0066 on task 2 at
        lambda 0.999, screened anyway across 59 boundaries), and the
        residual-driven body at lambda 0.999 (best task-2 of the diagnostic,
        0.8774; its lambda-0.995 variant COLLAPSED to 0.105 on task 2 —
        fast-forgetting heads are unstable as the body's error signal — and
        is deliberately not registered)."""
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_rls_head_learner,
            _rls_head_frozen_probe_input,
            _rls_head_hp,
        )

        expected = {
            "rls_head_l0999": {"rls_lambda": 0.999},
            "rls_head_l0995": {"rls_lambda": 0.995},
            "rls_head_l1": {"rls_lambda": 1.0},
            "rls_head_l0999_preset005": {
                "rls_lambda": 0.999, "rls_reset_frac": 0.05
            },
            "rls_head_resid": {"rls_lambda": 0.999, "head_resid": 1.0},
            # Wave 2 — wind-up stabilized: exponential forgetting (lambda<1)
            # overflows P along unexcited (dead-ReLU) feature directions at
            # (1/lambda)^t (float32 overflow ~ e^88.7 ~ task 18 at 0.999 —
            # the wave-1 collapse). lambda=1 cannot wind up (P is
            # nonincreasing PSD); staleness is handled by the detector-driven
            # P reset instead, and the trace cap salvages the forgetting
            # mechanism as a bounded probe.
            "rls_head_l1_preset005": {
                "rls_lambda": 1.0, "rls_reset_frac": 0.05
            },
            "rls_head_l1_preset003": {
                "rls_lambda": 1.0, "rls_reset_frac": 0.03
            },
            "rls_head_l0999_pcap": {
                "rls_lambda": 0.999, "rls_p_trace_cap": 1e4
            },
            "rls_head_resid_l1_preset005": {
                "rls_lambda": 1.0, "rls_reset_frac": 0.05, "head_resid": 1.0
            },
            "rls_head_resid_l1_noreset": {
                "rls_lambda": 1.0, "rls_reset_frac": 2.0, "head_resid": 1.0
            },
            "rls_head_resid_l1_preset005_nogate": {
                "rls_lambda": 1.0,
                "rls_reset_frac": 0.05,
                "head_resid": 1.0,
                "gate_scale": 0.0,
            },
            # Wave 3 — ridge star (2-task seed-0 diagnostic: smaller initial
            # ridge = larger early/post-reset gains; .8328/.8465/.853/.8578/
            # .8596 for ridge 1.0/0.3/0.1/0.03/0.01, monotone), plus the
            # residual body rerun at small ridge (0.8648 at 2 tasks, the
            # family's best diagnostic number: a fast-converging head makes
            # the residual signal reliable early).
            "rls_head_l0999_preset005_r01": {
                "rls_lambda": 0.999, "rls_reset_frac": 0.05,
                "rls_ridge_init": 0.1,
            },
            "rls_head_l0999_preset005_r003": {
                "rls_lambda": 0.999, "rls_reset_frac": 0.05,
                "rls_ridge_init": 0.03,
            },
            "rls_head_l0999_preset005_r001": {
                "rls_lambda": 0.999, "rls_reset_frac": 0.05,
                "rls_ridge_init": 0.01,
            },
            "rls_head_resid_preset005_r01": {
                "rls_lambda": 0.999, "rls_reset_frac": 0.05,
                "rls_ridge_init": 0.1, "head_resid": 1.0,
            },
            "rls_head_resid_preset005_r001": {
                "rls_lambda": 0.999, "rls_reset_frac": 0.05,
                "rls_ridge_init": 0.01, "head_resid": 1.0,
            },
        }
        for name, overrides in expected.items():
            spec = screening_spec(name)
            assert spec.base_learner == "upgd_w", name
            assert spec.mechanism == "rls_readout", name
            assert spec.factory is _make_rls_head_learner, name
            assert spec.frozen_probe_input is _rls_head_frozen_probe_input, name
            assert spec.noise_update is None, name
            assert spec.hyperparameters == _rls_head_hp(**overrides), name
            # Champion-body constants are intact on every arm.
            hp = spec.hyperparameters
            assert hp["step_size"] == pytest.approx(0.01), name
            assert hp["weight_decay"] == pytest.approx(0.01), name
            assert hp["utility_decay"] == pytest.approx(0.9999), name
            assert hp["norm_decay"] == pytest.approx(0.99), name
            assert hp["fast_decay"] == pytest.approx(0.9), name
            assert hp["shift_k"] == pytest.approx(1.0), name
            assert hp["shift_delta"] == pytest.approx(0.02), name
            assert hp["noise_std"] == 0.0, name
        incumbent_hp = screening_spec(
            "rls_head_resid_l1_preset005"
        ).hyperparameters
        nogate_hp = screening_spec(
            "rls_head_resid_l1_preset005_nogate"
        ).hyperparameters
        assert nogate_hp == {**incumbent_hp, "gate_scale": 0.0}


class TestRLSHeadNoReset:
    """Issue #184's code-only detector-reset ablation endpoint."""

    @staticmethod
    def _assert_tree_equal(actual, expected):
        actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
        expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
        assert actual_tree == expected_tree
        assert len(actual_leaves) == len(expected_leaves)
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf), np.asarray(expected_leaf)
            )

    def test_registry_differs_from_incumbent_only_at_reset_fraction(self):
        incumbent = screening_spec("rls_head_resid_l1_preset005")
        candidate = screening_spec("rls_head_resid_l1_noreset")
        assert candidate.base_learner == incumbent.base_learner == "upgd_w"
        assert candidate.mechanism == incumbent.mechanism == "rls_readout"
        assert candidate.factory is incumbent.factory
        assert candidate.frozen_probe_input is incumbent.frozen_probe_input
        assert candidate.noise_update is incumbent.noise_update is None
        assert candidate.hyperparameters == {
            **incumbent.hyperparameters,
            "rls_reset_frac": 2.0,
        }
        assert {
            name
            for name in incumbent.hyperparameters
            if incumbent.hyperparameters[name] != candidate.hyperparameters[name]
        } == {"rls_reset_frac"}

    @pytest.mark.parametrize("compiled", [False, True])
    def test_pretrigger_trajectory_is_bitexact_incumbent(self, compiled):
        incumbent = screening_spec("rls_head_resid_l1_preset005")
        candidate = screening_spec("rls_head_resid_l1_noreset")
        incumbent_init, incumbent_step = incumbent.factory(incumbent.hyperparameters)
        candidate_init, candidate_step = candidate.factory(candidate.hyperparameters)
        if compiled:
            incumbent_step = jax.jit(incumbent_step)
            candidate_step = jax.jit(candidate_step)

        params = init_mlp_params(jr.key(184), SMALL)
        incumbent_params = candidate_params = params
        incumbent_state = incumbent_init(params)
        candidate_state = candidate_init(params)
        xs = [
            jnp.linspace(0.05 * i, 0.05 * i + 0.3, SMALL.input_dim, dtype=jnp.float32)
            for i in range(1, 7)
        ]
        ys = [jnp.array(i % SMALL.n_classes, jnp.int32) for i in range(6)]
        for step, (x, y) in enumerate(zip(xs, ys)):
            incumbent_out = incumbent_step(
                incumbent_params, incumbent_state, x, y, jr.key(step)
            )
            candidate_out = candidate_step(
                candidate_params, candidate_state, x, y, jr.key(step)
            )
            self._assert_tree_equal(candidate_out, incumbent_out)
            incumbent_params, incumbent_state, _ = incumbent_out
            candidate_params, candidate_state, _ = candidate_out

    @pytest.mark.parametrize("compiled", [False, True])
    def test_forced_trigger_resets_only_incumbent_covariance(self, compiled):
        incumbent = screening_spec("rls_head_resid_l1_preset005")
        candidate = screening_spec("rls_head_resid_l1_noreset")
        incumbent_init, incumbent_step = incumbent.factory(incumbent.hyperparameters)
        candidate_init, candidate_step = candidate.factory(candidate.hyperparameters)
        if compiled:
            incumbent_step = jax.jit(incumbent_step)
            candidate_step = jax.jit(candidate_step)

        params = init_mlp_params(jr.key(185), SMALL)
        state = incumbent_init(params)
        candidate_state = candidate_init(params)
        self._assert_tree_equal(candidate_state, state)
        mature = replace(
            state,
            norm=replace(
                state.norm,
                mean=jnp.zeros(SMALL.input_dim, jnp.float32),
                var=jnp.full((SMALL.input_dim,), 1e-4, jnp.float32),
                count=jnp.full((SMALL.input_dim,), 1000.0, jnp.float32),
            ),
            fast_mean=jnp.zeros(SMALL.input_dim, jnp.float32),
        )
        x_shift = jnp.full((SMALL.input_dim,), 10.0, jnp.float32)
        y = jnp.array(2, jnp.int32)
        incumbent_out = incumbent_step(params, mature, x_shift, y, jr.key(0))
        candidate_out = candidate_step(params, mature, x_shift, y, jr.key(0))
        incumbent_params, incumbent_state, incumbent_metrics = incumbent_out
        candidate_params, candidate_state, candidate_metrics = candidate_out

        self._assert_tree_equal(candidate_params, incumbent_params)
        self._assert_tree_equal(candidate_metrics, incumbent_metrics)
        np.testing.assert_array_equal(
            np.asarray(candidate_state.wout), np.asarray(incumbent_state.wout)
        )
        ridge = incumbent.hyperparameters["rls_ridge_init"]
        expected_reset = np.eye(mature.p.shape[0], dtype=np.float32) / ridge
        np.testing.assert_array_equal(np.asarray(incumbent_state.p), expected_reset)
        assert not np.array_equal(np.asarray(candidate_state.p), expected_reset)
        assert not np.array_equal(np.asarray(candidate_state.p), np.asarray(mature.p))
        assert np.any(np.asarray(candidate_state.wout))

        x_norm, _, _, shifted = shift_adaptive_normalize(
            mature.norm,
            mature.fast_mean,
            x_shift,
            decay=incumbent.hyperparameters["norm_decay"],
            fast_decay=incumbent.hyperparameters["fast_decay"],
            epsilon=incumbent.hyperparameters["norm_epsilon"],
            shift_k=incumbent.hyperparameters["shift_k"],
            shift_delta=incumbent.hyperparameters["shift_delta"],
            shift_refractory=incumbent.hyperparameters["shift_refractory"],
        )
        assert bool(jnp.all(shifted))
        a1 = jax.nn.relu(x_norm @ params["w1"] + params["b1"])
        a2 = jax.nn.relu(a1 @ params["w2"] + params["b2"])
        phi = jnp.concatenate(
            [
                a2 / math.sqrt(SMALL.hidden2 + 1),
                jnp.ones((1,), jnp.float32),
            ]
        )
        pp = mature.p @ phi
        gain = pp / (incumbent.hyperparameters["rls_lambda"] + phi @ pp)
        expected_p = mature.p - jnp.outer(gain, pp)
        expected_p = 0.5 * (expected_p + expected_p.T)
        np.testing.assert_allclose(
            np.asarray(candidate_state.p), np.asarray(expected_p), rtol=1e-6, atol=1e-7
        )

        for field in ("utility", "step", "norm", "fast_mean"):
            self._assert_tree_equal(
                getattr(candidate_state, field), getattr(incumbent_state, field)
            )


class TestRLSHeadL2Init:
    """Issue #14's body-only L2-to-initialization code prerequisite."""

    _BODY = ("w1", "b1", "w2", "b2")

    @staticmethod
    def _assert_tree_equal(actual, expected):
        actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
        expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
        assert actual_tree == expected_tree
        assert len(actual_leaves) == len(expected_leaves)
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf), np.asarray(expected_leaf)
            )

    @staticmethod
    def _manual_params() -> dict[str, jax.Array]:
        return {
            "w1": jnp.full((SMALL.input_dim, SMALL.hidden1), 0.05, jnp.float32),
            "b1": jnp.full((SMALL.hidden1,), 0.10, jnp.float32),
            "w2": jnp.full((SMALL.hidden1, SMALL.hidden2), 0.04, jnp.float32),
            "b2": jnp.full((SMALL.hidden2,), 0.10, jnp.float32),
            "w3": jnp.full((SMALL.hidden2, SMALL.n_classes), 0.03, jnp.float32),
            "b3": jnp.full((SMALL.n_classes,), 0.02, jnp.float32),
        }

    def test_registry_is_exactly_incumbent_plus_frozen_endpoint(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            RLSHeadL2InitState,
            RLSHeadState,
            _make_rls_head_l2init_learner,
            _rls_head_frozen_probe_input,
        )

        incumbent = screening_spec("rls_head_resid_l1_preset005")
        candidate = screening_spec("rls_head_resid_l1_preset005_l2init")
        assert candidate.base_learner == incumbent.base_learner == "upgd_w"
        assert candidate.mechanism == incumbent.mechanism == "rls_readout"
        assert candidate.factory is _make_rls_head_l2init_learner
        assert candidate.frozen_probe_input is _rls_head_frozen_probe_input
        assert candidate.noise_update is None
        assert candidate.hyperparameters == {
            **incumbent.hyperparameters,
            "decay_to_init": 1.0,
        }

        params = self._manual_params()
        incumbent_state = incumbent.factory(incumbent.hyperparameters)[0](params)
        candidate_state = candidate.factory(candidate.hyperparameters)[0](params)
        assert type(incumbent_state) is RLSHeadState
        assert not hasattr(incumbent_state, "init_params")
        assert type(candidate_state) is RLSHeadL2InitState
        assert set(candidate_state.init_params) == set(self._BODY)
        for name in self._BODY:
            np.testing.assert_array_equal(
                np.asarray(candidate_state.init_params[name]), np.asarray(params[name])
            )
        for field in ("utility", "step", "norm", "fast_mean", "p", "wout"):
            self._assert_tree_equal(
                jax.device_get(getattr(candidate_state, field)),
                jax.device_get(getattr(incumbent_state, field)),
            )

    def test_factory_rejects_every_nonfrozen_config(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_rls_head_l2init_learner,
            _rls_head_l2init_hp,
        )

        expected = _rls_head_l2init_hp()
        invalid = []
        without_endpoint = dict(expected)
        without_endpoint.pop("decay_to_init")
        invalid.append(without_endpoint)
        invalid.extend(
            [
                {**expected, "decay_to_init": 0.0},
                {**expected, "decay_to_init": True},
                {**expected, "step_size": 0.02},
                {**expected, "rls_lambda": 1},
                {**expected, "noise_std": -0.0},
                {**expected, "unexpected": 0.0},
            ]
        )
        for hp in invalid:
            with pytest.raises(ValueError, match="frozen L2-Init configuration"):
                _make_rls_head_l2init_learner(hp)

        _make_rls_head_l2init_learner(expected)

    def test_body_and_rls_updates_match_equations_and_freeze_sgd_head(self):
        import dataclasses

        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_rls_head_l2init_learner,
            _rls_head_l2init_hp,
        )

        hp = _rls_head_l2init_hp()
        init_fn, step_fn = _make_rls_head_l2init_learner(hp)
        init_params = self._manual_params()
        state = init_fn(init_params)
        params = dict(init_params)
        for index, name in enumerate(self._BODY, start=1):
            params[name] = init_params[name] + jnp.asarray(
                0.01 * index, dtype=jnp.float32
            )

        m = SMALL.hidden2 + 1
        wout = jnp.linspace(
            -0.2, 0.3, m * SMALL.n_classes, dtype=jnp.float32
        ).reshape((m, SMALL.n_classes))
        state = dataclasses.replace(
            state,
            p=jnp.eye(m, dtype=jnp.float32) * 0.7,
            wout=wout,
        )
        x = jnp.linspace(0.1, 0.4, SMALL.input_dim, dtype=jnp.float32)
        y = jnp.asarray(2, dtype=jnp.int32)

        x_norm, expected_norm, expected_fast, shifted = shift_adaptive_normalize(
            state.norm,
            state.fast_mean,
            x,
            decay=hp["norm_decay"],
            fast_decay=hp["fast_decay"],
            epsilon=hp["norm_epsilon"],
            shift_k=hp["shift_k"],
            shift_delta=hp["shift_delta"],
            shift_refractory=hp["shift_refractory"],
        )
        assert not bool(jnp.any(shifted))

        body = {name: params[name] for name in self._BODY}

        def residual_loss(
            body_params: dict[str, jax.Array],
        ) -> jax.Array:
            merged = dict(params)
            merged.update(body_params)
            a1 = jax.nn.relu(x_norm @ merged["w1"] + merged["b1"])
            a2 = jax.nn.relu(a1 @ merged["w2"] + merged["b2"])
            phi = jnp.concatenate(
                [
                    a2 * (1.0 / math.sqrt(m)),
                    jnp.ones((1,), dtype=jnp.float32),
                ]
            )
            logits = state.wout.T @ phi
            target = jax.nn.one_hot(y, SMALL.n_classes, dtype=jnp.float32)
            error = target - logits
            return 0.5 * jnp.sum(error * error)

        grads = jax.grad(residual_loss)(body)
        count = state.step + jnp.asarray(1, dtype=jnp.int32)
        utility = dict(state.utility)
        for name in self._BODY:
            utility[name] = hp["utility_decay"] * state.utility[name] + (
                1.0 - hp["utility_decay"]
            ) * (-grads[name] * params[name])
        bias_correction = 1.0 - jnp.power(
            jnp.asarray(hp["utility_decay"], dtype=jnp.float32),
            count.astype(jnp.float32),
        )
        global_max = jnp.max(
            jnp.stack([jnp.max(utility[name]) for name in sorted(self._BODY)])
        )
        global_max = jnp.where(global_max == 0.0, 1.0, global_max)
        expected_params = dict(params)
        for name in self._BODY:
            gated_gradient = grads[name] * (
                1.0
                - jax.nn.sigmoid((utility[name] / bias_correction) / global_max)
            )
            expected_params[name] = (
                params[name]
                - hp["step_size"]
                * hp["weight_decay"]
                * (params[name] - state.init_params[name])
                - hp["step_size"] * gated_gradient
            )

        a1 = jax.nn.relu(x_norm @ params["w1"] + params["b1"])
        a2 = jax.nn.relu(a1 @ params["w2"] + params["b2"])
        phi = jnp.concatenate(
            [
                a2 * (1.0 / math.sqrt(m)),
                jnp.ones((1,), dtype=jnp.float32),
            ]
        )
        target = jax.nn.one_hot(y, SMALL.n_classes, dtype=jnp.float32)
        error = target - state.wout.T @ phi
        pp = state.p @ phi
        gain = pp / (hp["rls_lambda"] + phi @ pp)
        expected_wout = state.wout + jnp.outer(gain, error)
        expected_p = (state.p - jnp.outer(gain, pp)) / hp["rls_lambda"]
        expected_p = 0.5 * (expected_p + expected_p.T)

        new_params, new_state, _ = step_fn(params, state, x, y, jr.key(99))

        for name in self._BODY:
            np.testing.assert_array_equal(
                np.asarray(new_params[name]), np.asarray(expected_params[name]), name
            )
            np.testing.assert_array_equal(
                np.asarray(new_state.utility[name]), np.asarray(utility[name]), name
            )
            np.testing.assert_array_equal(
                np.asarray(new_state.init_params[name]),
                np.asarray(state.init_params[name]),
                name,
            )
        for name in ("w3", "b3"):
            np.testing.assert_array_equal(
                np.asarray(new_params[name]), np.asarray(params[name]), name
            )
            np.testing.assert_array_equal(
                np.asarray(new_state.utility[name]),
                np.asarray(state.utility[name]),
                name,
            )
        assert not np.array_equal(np.asarray(expected_p), np.asarray(state.p))
        assert not np.array_equal(np.asarray(expected_wout), np.asarray(state.wout))
        np.testing.assert_array_equal(
            np.asarray(new_state.p), np.asarray(expected_p)
        )
        np.testing.assert_array_equal(
            np.asarray(new_state.wout), np.asarray(expected_wout)
        )
        self._assert_tree_equal(
            jax.device_get(new_state.norm), jax.device_get(expected_norm)
        )
        np.testing.assert_array_equal(
            np.asarray(new_state.fast_mean), np.asarray(expected_fast)
        )
        assert int(new_state.step) == 1

    def test_jit_and_pytree_state_roundtrip_preserve_initial_snapshot(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            RLSHeadL2InitState,
            _make_rls_head_l2init_learner,
            _rls_head_l2init_hp,
        )

        init_fn, step_fn = _make_rls_head_l2init_learner(_rls_head_l2init_hp())
        params = self._manual_params()
        state = init_fn(params)
        leaves, tree = jax.tree_util.tree_flatten(state)
        restored = jax.tree_util.tree_unflatten(tree, leaves)
        assert type(restored) is RLSHeadL2InitState
        self._assert_tree_equal(jax.device_get(restored), jax.device_get(state))

        compiled = jax.jit(step_fn)
        x = jnp.linspace(-0.2, 0.4, SMALL.input_dim, dtype=jnp.float32)
        new_params, new_state, metrics = compiled(
            params, restored, x, jnp.asarray(1, jnp.int32), jr.key(7)
        )
        assert type(new_state) is RLSHeadL2InitState
        for name in self._BODY:
            np.testing.assert_array_equal(
                np.asarray(new_state.init_params[name]), np.asarray(params[name]), name
            )
        for name in ("w3", "b3"):
            np.testing.assert_array_equal(
                np.asarray(new_params[name]), np.asarray(params[name]), name
            )
        assert all(bool(jnp.isfinite(metric)) for metric in metrics)

    @pytest.mark.integration
    def test_registered_arm_runs_through_synthetic_screening_harness(self, small_data):
        x, y = small_data
        spec = screening_spec("rls_head_resid_l1_preset005_l2init")
        result = run_screening_config(x, y, spec, seed=2, config=SMALL)
        assert result.config_name == spec.name
        assert result.hyperparameters == spec.hyperparameters
        assert np.all(np.isfinite(result.per_task_accuracy))
        assert np.all(np.isfinite(result.per_task_loss))
        assert np.all(np.isfinite(result.per_task_plasticity))


class TestNBEnsemble:
    """Transient attack: adaptive ensemble of the shiftnorm champion and the
    streaming naive-Bayes tracker (``nb_ensemble_champion`` family).

    Prediction is an accuracy-weighted probability mixture whose weights are
    learned ONLINE from the stream itself: per-member annealed EMAs of each
    member's own pre-update correctness (fast decay, no oracle, no task
    boundaries), squashed through a softmax with temperature ``ens_beta``.
    Right after a permutation the champion's recent-accuracy EMA collapses
    while naive Bayes stays flat, so the vote swings to NB; mid-task the
    champion re-converges and takes the vote back.  Probes: (b) detector-
    driven NB statistics reset (``nb_ensemble_nbreset``), (c) a third
    closed-form fast-converging member — linear RLS over normalized pixels
    (``nb_ensemble_rls3``).
    """

    _ARMS = ("nb_ensemble_champion", "nb_ensemble_nbreset", "nb_ensemble_rls3")

    def _hp(self, **overrides):
        from alberta_framework.benchmarks.ipmnist_screening import _nb_ensemble_hp

        return _nb_ensemble_hp(**overrides)

    def _factory(self, **overrides):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_nb_ensemble_learner,
        )

        return _make_nb_ensemble_learner(self._hp(**overrides))

    def _champion_factory(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_upgd_shiftnorm_learner,
            _sigma0_ext_hp,
        )

        return _make_upgd_shiftnorm_learner(
            _sigma0_ext_hp(
                norm_decay=0.99,
                fast_decay=0.9,
                shift_k=1.0,
                shift_delta=0.02,
                shift_refractory=0.0,
            )
        )

    def _stream(self, n_steps=12, seed=3):
        key = jr.key(seed)
        xs, ys = [], []
        for step in range(n_steps):
            key, kx = jr.split(key)
            xs.append(
                jr.uniform(kx, (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
                * (1.0 + step % 3)
            )
            ys.append(jnp.array(step % SMALL.n_classes, jnp.int32))
        return xs, ys

    def test_registry_configs(self):
        from alberta_framework.benchmarks.ipmnist_screening import (
            _make_nb_ensemble_learner,
        )

        for name in self._ARMS:
            spec = screening_spec(name)
            assert spec.mechanism == "transient_ensemble", name
            assert spec.factory is _make_nb_ensemble_learner, name
            assert spec.noise_update is None, name
            hp = spec.hyperparameters
            # Champion-member constants are verbatim sigma0_shiftnorm_d099.
            champ = screening_spec("sigma0_shiftnorm_d099").hyperparameters
            for k in (
                "step_size", "weight_decay", "utility_decay", "norm_decay",
                "norm_epsilon", "fast_decay", "shift_k", "shift_delta",
                "shift_refractory",
            ):
                assert hp[k] == champ[k], (name, k)
            # NB-member constants are verbatim the naive_bayes arm.
            nb = screening_spec("naive_bayes").hyperparameters
            assert hp["nb_decay"] == nb["nb_decay"], name
            assert hp["nb_var_epsilon"] == nb["nb_var_epsilon"], name
            assert 0.0 < hp["ens_decay"] < 1.0, name
            assert hp["ens_beta"] > 0.0, name
            assert hp["ens_lock_network"] == 0.0, name
            with pytest.raises(NotImplementedError, match="nb_ensemble"):
                spec.frozen_probe_input(None, jnp.zeros(4), hp)
        assert screening_spec("nb_ensemble_champion").hyperparameters[
            "ens_nb_reset"
        ] == 0.0
        assert screening_spec("nb_ensemble_nbreset").hyperparameters[
            "ens_nb_reset"
        ] == 1.0
        assert screening_spec("nb_ensemble_rls3").hyperparameters[
            "ens_use_rls"
        ] == 1.0
        # The RLS member's constants are verbatim the lin_rls arm.
        lin = screening_spec("lin_rls").hyperparameters
        hp3 = screening_spec("nb_ensemble_rls3").hyperparameters
        for k in ("rff_clip", "rls_lambda", "rls_ridge_init"):
            assert hp3[k] == lin[k], k

    def test_lock_network_reduces_to_shiftnorm_champion_bitwise(self):
        """ens_lock_network=1: params AND metrics follow the registered
        sigma0_shiftnorm_d099 champion bit-for-bit (the reduction pin)."""
        champ_init, champ_step = self._champion_factory()
        ens_init, ens_step = self._factory(ens_lock_network=1.0)
        params = init_mlp_params(jr.key(5), SMALL)
        champ_params, ens_params = params, params
        champ_state, ens_state = champ_init(params), ens_init(params)
        xs, ys = self._stream()
        for step, (x, y) in enumerate(zip(xs, ys)):
            champ_params, champ_state, m_champ = champ_step(
                champ_params, champ_state, x, y, jr.key(step)
            )
            ens_params, ens_state, m_ens = ens_step(
                ens_params, ens_state, x, y, jr.key(step)
            )
            for name in sorted(params):
                np.testing.assert_array_equal(
                    np.asarray(ens_params[name]), np.asarray(champ_params[name]), name
                )
            for a, b in zip(m_ens, m_champ):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        # The lock changes only the deployed prediction: the member EMAs
        # still learned from the stream.
        assert float(jnp.sum(ens_state.member_acc)) >= 0.0

    def test_mixture_and_weight_update_hand_computed(self):
        """One step from a crafted state: the ensemble prediction is the
        log-domain accuracy-weighted probability mixture of the members'
        pre-update posteriors, and the member EMAs update with the annealed
        recurrence min(ens_decay, 1 - 1/(t+1)) on each member's own
        correctness."""
        import dataclasses as _dc

        from alberta_framework.benchmarks.ipmnist_screening import (
            naive_bayes_logits as _nb_logits,
        )
        from alberta_framework.benchmarks.ipmnist_screening import (
            shift_adaptive_normalize as _shift_norm,
        )
        from alberta_framework.benchmarks.upgd_ipmnist import mlp_logits as _mlp_logits

        hp = self._hp(ens_beta=7.0, ens_decay=0.9)
        ens_init, ens_step = self._factory(ens_beta=7.0, ens_decay=0.9)
        params = init_mlp_params(jr.key(11), SMALL)
        state = ens_init(params)
        # Drive a few steps so NB statistics and the normalizer are nontrivial.
        xs, ys = self._stream(n_steps=6, seed=21)
        p = params
        for step, (x, y) in enumerate(zip(xs, ys)):
            p, state, _ = ens_step(p, state, x, y, jr.key(step))
        # Craft asymmetric member EMAs so the weights are far from uniform.
        state = _dc.replace(
            state, member_acc=jnp.array([0.9, 0.2], dtype=jnp.float32)
        )
        x = jr.uniform(jr.key(99), (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
        y = jnp.array(2, jnp.int32)
        # Hand-computed member posteriors (pre-update).
        x_norm, _, _, _ = _shift_norm(
            state.net.norm, state.net.fast_mean, x,
            decay=hp["norm_decay"], fast_decay=hp["fast_decay"],
            epsilon=hp["norm_epsilon"], shift_k=hp["shift_k"],
            shift_delta=hp["shift_delta"],
            shift_refractory=hp["shift_refractory"],
        )
        net_logits = _mlp_logits(p, x_norm)
        nb_logits = _nb_logits(state.nb, x)
        log_w = jax.nn.log_softmax(7.0 * state.member_acc)
        stacked = jnp.stack(
            [jax.nn.log_softmax(net_logits), jax.nn.log_softmax(nb_logits)]
        )
        mixture = jax.nn.logsumexp(stacked + log_w[:, None], axis=0)
        expected_pred = int(jnp.argmax(mixture))
        expected_loss = float(-jax.nn.log_softmax(mixture)[y])
        new_p, new_state, (acc, loss, plasticity) = ens_step(
            p, state, x, y, jr.key(123)
        )
        assert float(acc) == float(expected_pred == int(y))
        np.testing.assert_allclose(float(loss), expected_loss, rtol=1e-5)
        assert 0.0 <= float(plasticity) <= 1.0
        # Member EMA update: annealed recurrence on each member's own
        # pre-update correctness.
        t = float(state.ens_step) + 1.0
        eff = min(0.9, 1.0 - 1.0 / (t + 1.0))
        net_correct = float(jnp.argmax(net_logits) == y)
        nb_correct = float(jnp.argmax(nb_logits) == y)
        expected_acc = np.array(
            [
                eff * 0.9 + (1.0 - eff) * net_correct,
                eff * 0.2 + (1.0 - eff) * nb_correct,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            np.asarray(new_state.member_acc), expected_acc, rtol=1e-6
        )
        assert float(new_state.ens_step) == t

    def test_weights_swing_to_nb_when_network_is_wrong(self):
        """Feed a stream the NB member predicts correctly while the champion
        member is wrong: the NB weight must rise above the network's."""
        ens_init, ens_step = self._factory(ens_decay=0.99)
        params = init_mlp_params(jr.key(1), SMALL)
        state = ens_init(params)
        # A fixed input whose label is the class the init network is most
        # biased AGAINST (argmin of the output bias): under a constant input
        # the normalized features collapse toward zero, so the network's
        # prediction is bias-driven and wrong for many steps, while NB locks
        # on after one observation.
        x = jnp.linspace(-1.0, 1.0, SMALL.input_dim, dtype=jnp.float32)
        y = jnp.argmin(params["b3"]).astype(jnp.int32)
        p = params
        for step in range(40):
            p, state, _ = ens_step(p, state, x, y, jr.key(step))
        acc = np.asarray(state.member_acc)
        assert acc[1] > 0.8  # NB is essentially always right on this stream
        assert acc[1] > acc[0] + 0.2  # ... and the vote must reflect it

    def test_nb_reset_triggers_on_global_shift_and_respects_refractory(self):
        """nb_ensemble_nbreset: a global input-statistics shift resets the NB
        class-count anneal clocks exactly once per refractory window; means
        and variances are never zeroed by the reset."""
        ens_init, ens_step = self._factory(
            ens_nb_reset=1.0, ens_reset_frac=0.25, ens_reset_refractory=8.0
        )
        params = init_mlp_params(jr.key(2), SMALL)
        state = ens_init(params)
        p = params
        key = jr.key(7)
        # Mature the detector on a stationary stream.
        for step in range(30):
            key, kx = jr.split(key)
            x = jr.uniform(kx, (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
            p, state, _ = ens_step(
                p, state, x, jnp.array(step % SMALL.n_classes, jnp.int32),
                jr.key(step),
            )
        counts_before = np.asarray(state.nb.ccount)
        assert counts_before.sum() > 0.0
        means_before = np.asarray(state.nb.cmean)
        # A gross global shift: every feature jumps by +10.
        x_shift = jnp.full((SMALL.input_dim,), 10.0, dtype=jnp.float32)
        p, state, _ = ens_step(p, state, x_shift, jnp.array(0, jnp.int32), jr.key(90))
        counts_after = np.asarray(state.nb.ccount)
        # All class clocks reset to zero except the observed class's +1 is
        # also wiped (reset applies after the member update).
        np.testing.assert_array_equal(counts_after, 0.0)
        assert float(state.reset_age) == 0.0
        # Means were NOT zeroed by the reset (only clocks).
        assert np.any(np.asarray(state.nb.cmean) != 0.0) or np.any(means_before == 0.0)
        # Within the refractory window an equally-shifted step cannot re-trigger:
        # clocks accumulate again immediately.
        x_shift2 = jnp.full((SMALL.input_dim,), -10.0, dtype=jnp.float32)
        p, state, _ = ens_step(p, state, x_shift2, jnp.array(1, jnp.int32), jr.key(91))
        assert float(np.asarray(state.nb.ccount).sum()) > 0.0
        assert float(state.reset_age) == 1.0
        # The plain arm never resets: same stream, clocks keep accumulating.
        plain_init, plain_step = self._factory()
        pstate = plain_init(params)
        pp = params
        key = jr.key(7)
        for step in range(30):
            key, kx = jr.split(key)
            x = jr.uniform(kx, (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
            pp, pstate, _ = plain_step(
                pp, pstate, x, jnp.array(step % SMALL.n_classes, jnp.int32),
                jr.key(step),
            )
        pp, pstate, _ = plain_step(
            pp, pstate, x_shift, jnp.array(0, jnp.int32), jr.key(90)
        )
        assert float(np.asarray(pstate.nb.ccount).sum()) == 31.0

    def test_rls3_carries_lin_rls_member(self):
        """nb_ensemble_rls3: the third member is the lin_rls pipeline (bias-
        augmented normalized pixels, Sherman-Morrison RLS) and its EMA slot
        exists; the 2-member arms carry no RLS state."""
        ens_init, _ = self._factory(ens_use_rls=1.0)
        params = init_mlp_params(jr.key(3), SMALL)
        state = ens_init(params)
        assert state.rls is not None
        assert state.rls.p.shape == (SMALL.input_dim + 1, SMALL.input_dim + 1)
        assert state.rls.wout.shape == (SMALL.input_dim + 1, SMALL.n_classes)
        assert state.member_acc.shape == (3,)
        plain_init, _ = self._factory()
        plain_state = plain_init(params)
        assert plain_state.rls is None
        assert plain_state.member_acc.shape == (2,)

    def test_key_is_unused_on_every_arm(self):
        """All members are closed-form or sigma-0: the RNG key is inert."""
        params = init_mlp_params(jr.key(4), SMALL)
        for overrides in ({}, {"ens_nb_reset": 1.0}, {"ens_use_rls": 1.0}):
            ens_init, ens_step = self._factory(**overrides)
            state = ens_init(params)
            x = jr.uniform(jr.key(6), (SMALL.input_dim,), jnp.float32, -1.0, 1.0)
            y = jnp.array(3, jnp.int32)
            p_a, s_a, m_a = ens_step(params, state, x, y, jr.key(0))
            p_b, s_b, m_b = ens_step(params, state, x, y, jr.key(424242))
            for n in params:
                np.testing.assert_array_equal(np.asarray(p_a[n]), np.asarray(p_b[n]))
            np.testing.assert_array_equal(
                np.asarray(s_a.member_acc), np.asarray(s_b.member_acc)
            )
            for a, b in zip(m_a, m_b):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_smoke_runs_finite(self, small_data):
        x, y = small_data
        for name in self._ARMS:
            result = run_screening_config(
                x, y, screening_spec(name), seed=2, config=SMALL
            )
            acc = np.asarray(result.per_task_accuracy)
            assert np.all(np.isfinite(acc)), name
            assert np.all((acc >= 0.0) & (acc <= 1.0)), name
            assert np.all(np.isfinite(np.asarray(result.per_task_loss))), name
            plas = np.asarray(result.per_task_plasticity)
            assert np.all((plas >= 0.0) & (plas <= 1.0)), name
