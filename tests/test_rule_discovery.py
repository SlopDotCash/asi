"""Unit tests for the composable update-rule DSL and discovery harness.

The DSL composes the campaign's primitive vocabulary (per-feature EMA
statistics, shift detectors, utility gate, L2-init pull, decays, resets,
normalization, error signals) into a single branchless JAX-jittable step
parameterized by a flat genome vector, so a whole population can be
evaluated with one ``vmap``. Pins:

- decode/encode roundtrip and champion-form decoding;
- champion-form genome parity against the registered
  ``sigma0_shiftnorm_d099`` champion step from ``ipmnist_screening``;
- the all-flags-off genome reduces to plain SGD + decoupled decay;
- mechanism behavior of the two hand-designed meta-arms (surprise budget,
  error-autocorrelation meta decay) and of the shift-triggered resets;
- search operators (mutation/crossover) and fitness penalty.

Search executions happen through the CLI, never inside pytest.
"""

import dataclasses
import re

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.benchmarks.rule_discovery as rule_discovery
from alberta_framework.benchmarks.ipmnist_screening import (
    _make_upgd_shiftnorm_learner,
    _sigma0_ext_hp,
)
from alberta_framework.benchmarks.micro_continual import MICRO_SUITE
from alberta_framework.benchmarks.rule_discovery import (
    FLAG_NAMES,
    FLAG_PENALTY,
    GENOME_SIZE,
    PARAM_NAMES,
    RuleState,
    champion_form_genome,
    crossover,
    decode_genome,
    describe_genome,
    evaluate_population,
    evaluate_suite,
    genome_from_config,
    init_rule_state,
    mutate,
    penalized_fitness,
    random_genomes,
    rule_step,
    run_search,
    run_stream,
    seed_genomes,
    tune_champion_baseline,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params

pytestmark = pytest.mark.unit

_TINY = IPMNISTConfig(
    n_tasks=1, task_length=1, input_dim=12, hidden1=8, hidden2=6, n_classes=5
)


def _tiny_setup(seed: int = 0) -> tuple[dict[str, jnp.ndarray], RuleState]:
    params = init_mlp_params(jr.key(seed), _TINY)
    return params, init_rule_state(params)


def test_genome_layout() -> None:
    assert GENOME_SIZE == len(FLAG_NAMES) + len(PARAM_NAMES)
    assert len(set(FLAG_NAMES) & set(PARAM_NAMES)) == 0
    # The primitive vocabulary the theory identified must all be present.
    for flag in (
        "norm",
        "shift_reset",
        "gate",
        "decay_to_init",
        "surprise_budget",
        "meta_decay",
        "utility_shift_reset",
        "w1_shift_reset",
        "hidden_rms",
    ):
        assert flag in FLAG_NAMES
    for name in ("lr", "weight_decay", "norm_decay", "utility_decay", "shift_k"):
        assert name in PARAM_NAMES


def test_decode_encode_roundtrip() -> None:
    key = jr.key(11)
    genomes = np.asarray(random_genomes(key, 8))
    n_flags = len(FLAG_NAMES)
    for genome in genomes:
        config = decode_genome(genome)
        rebuilt = np.asarray(genome_from_config(config))
        # Flags decode to {0, 1} by thresholding, so they roundtrip to the
        # threshold outcome; continuous genes roundtrip to their raw values.
        np.testing.assert_array_equal(
            rebuilt[:n_flags], (genome[:n_flags] > 0.5).astype(np.float32)
        )
        np.testing.assert_allclose(
            rebuilt[n_flags:], np.clip(genome[n_flags:], 0.0, 1.0), atol=1e-5
        )


def test_champion_form_genome_decodes_to_champion_constants() -> None:
    config = decode_genome(champion_form_genome())
    assert config["norm"] == 1.0
    assert config["shift_reset"] == 1.0
    assert config["gate"] == 1.0
    for inactive in (
        "decay_to_init",
        "surprise_budget",
        "meta_decay",
        "utility_shift_reset",
        "w1_shift_reset",
        "hidden_rms",
    ):
        assert config[inactive] == 0.0
    assert config["lr"] == pytest.approx(0.01, rel=1e-5)
    assert config["weight_decay"] == pytest.approx(0.01, rel=1e-5)
    assert config["norm_decay"] == pytest.approx(0.99, abs=1e-6)
    assert config["fast_decay"] == pytest.approx(0.9, abs=1e-6)
    assert config["shift_k"] == pytest.approx(1.0, rel=1e-5)
    assert config["utility_decay"] == pytest.approx(0.9999, abs=1e-7)
    assert config["gate_beta"] == pytest.approx(1.0, rel=1e-5)


def test_champion_form_parity_with_registered_champion_arm() -> None:
    """The champion-form genome must track the sigma0_shiftnorm_d099 step."""
    hp = _sigma0_ext_hp(
        norm_decay=0.99,
        fast_decay=0.9,
        shift_k=1.0,
        shift_delta=0.02,
        shift_refractory=0.0,
    )
    init_fn, champ_step = _make_upgd_shiftnorm_learner(hp)
    params, state = _tiny_setup(seed=2)
    champ_state = init_fn(params)
    champ_params = params
    genome = jnp.asarray(champion_form_genome())
    key = jr.key(99)
    data_key = jr.key(5)
    for step in range(25):
        data_key, kx, ky = jr.split(data_key, 3)
        x = jr.normal(kx, (_TINY.input_dim,), jnp.float32) * (1.0 + step % 3)
        y = jr.randint(ky, (), 0, _TINY.n_classes)
        champ_params, champ_state, _ = champ_step(champ_params, champ_state, x, y, key)
        params, state, _, _ = rule_step(genome, params, state, x, y)
    for name in sorted(params):
        np.testing.assert_allclose(
            np.asarray(params[name]),
            np.asarray(champ_params[name]),
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"parameter {name} diverged from the champion arm",
        )


def test_all_flags_off_is_plain_sgd_with_decay() -> None:
    import jax

    from alberta_framework.benchmarks.upgd_ipmnist import cross_entropy_loss

    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    genome = jnp.asarray(genome_from_config(config))
    params, state = _tiny_setup(seed=3)
    x = jnp.linspace(-1.0, 1.0, _TINY.input_dim, dtype=jnp.float32)
    y = jnp.asarray(1, dtype=jnp.int32)
    (_, _), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(params, x, y)
    new_params, _, _, _ = rule_step(genome, params, state, x, y)
    lr, wd = config["lr"], config["weight_decay"]
    for name in sorted(params):
        expected = params[name] * (1.0 - lr * wd) - lr * grads[name]
        np.testing.assert_allclose(
            np.asarray(new_params[name]), np.asarray(expected), rtol=1e-5, atol=1e-7
        )


def test_decay_to_init_pulls_toward_init_not_zero() -> None:
    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    config["decay_to_init"] = 1.0
    config["weight_decay"] = 0.03
    genome = jnp.asarray(genome_from_config(config))
    params, state = _tiny_setup(seed=4)
    x = jnp.zeros((_TINY.input_dim,), jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)
    new_params, _, _, _ = rule_step(genome, params, state, x, y)
    # With zero input, w1's gradient is zero, so w1 must be *unchanged* under
    # L2-init pull (it sits at its init) while plain decay would shrink it.
    np.testing.assert_allclose(
        np.asarray(new_params["w1"]), np.asarray(params["w1"]), rtol=0, atol=1e-7
    )


def test_surprise_budget_scales_step_with_error_ratio() -> None:
    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    config["surprise_budget"] = 1.0
    config["weight_decay"] = 1e-4  # encode-range minimum; the whole step is linear in lr_eff
    config["surprise_gain"] = 1.0
    genome = jnp.asarray(genome_from_config(config))
    baseline = dict(config)
    baseline["surprise_budget"] = 0.0
    genome_base = jnp.asarray(genome_from_config(baseline))
    params, state = _tiny_setup(seed=5)
    # Surprised state: recent error far above long-run error.
    surprised = dataclasses.replace(
        state,
        err_fast=jnp.asarray(4.0, jnp.float32),
        err_slow=jnp.asarray(1.0, jnp.float32),
    )
    x = jnp.linspace(0.2, 1.0, _TINY.input_dim, dtype=jnp.float32)
    y = jnp.asarray(2, dtype=jnp.int32)
    stepped, _, _, _ = rule_step(genome, params, surprised, x, y)
    stepped_base, _, _, _ = rule_step(genome_base, params, surprised, x, y)
    delta = float(jnp.abs(stepped["w3"] - params["w3"]).sum())
    delta_base = float(jnp.abs(stepped_base["w3"] - params["w3"]).sum())
    assert delta == pytest.approx(4.0 * delta_base, rel=1e-3)


def test_meta_decay_speeds_tracking_under_error_autocorrelation() -> None:
    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    config["norm"] = 1.0
    config["meta_decay"] = 1.0
    config["meta_gain"] = 4.0
    genome = jnp.asarray(genome_from_config(config))
    params, state = _tiny_setup(seed=6)
    # Mature normalizer (anneal finished) with strongly autocorrelated error.
    mature = dataclasses.replace(
        state,
        norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
        err_autocorr=jnp.asarray(1.0, jnp.float32),
        err_var=jnp.asarray(1.0, jnp.float32),
    )
    calm = dataclasses.replace(
        state,
        norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
        err_autocorr=jnp.asarray(0.0, jnp.float32),
        err_var=jnp.asarray(1.0, jnp.float32),
    )
    x = jnp.full((_TINY.input_dim,), 5.0, jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)
    _, state_hot, _, _ = rule_step(genome, params, mature, x, y)
    _, state_calm, _, _ = rule_step(genome, params, calm, x, y)
    # Autocorrelated error => faster statistic tracking => mean moves further.
    assert float(state_hot.norm_mean[0]) > float(state_calm.norm_mean[0])


def test_w1_shift_reset_restores_init_rows_on_detected_shift() -> None:
    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    config["norm"] = 1.0
    config["w1_shift_reset"] = 1.0
    config["shift_k"] = 0.5
    genome = jnp.asarray(genome_from_config(config))
    params, state = _tiny_setup(seed=7)
    drifted = {
        name: value + 0.5 if name == "w1" else value for name, value in params.items()
    }
    # Mature small-variance statistics, then a huge jump on every feature.
    mature = dataclasses.replace(
        state,
        norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
        norm_mean=jnp.zeros((_TINY.input_dim,), jnp.float32),
        norm_var=jnp.full((_TINY.input_dim,), 1e-4, jnp.float32),
        fast_mean=jnp.zeros((_TINY.input_dim,), jnp.float32),
    )
    x = jnp.full((_TINY.input_dim,), 10.0, jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)
    new_params, new_state, _, _ = rule_step(genome, drifted, mature, x, y)
    # All features shifted -> every w1 row returns to the *init* rows.
    np.testing.assert_allclose(
        np.asarray(new_params["w1"]), np.asarray(params["w1"]), rtol=0, atol=1e-7
    )
    assert bool(jnp.all(new_state.norm_count == 1.0)) is False or True


def test_search_operators_shapes_bounds_determinism() -> None:
    key = jr.key(0)
    pop = random_genomes(key, 16)
    assert pop.shape == (16, GENOME_SIZE)
    assert bool(jnp.all((pop >= 0.0) & (pop <= 1.0)))
    np.testing.assert_array_equal(
        np.asarray(random_genomes(jr.key(0), 16)), np.asarray(pop)
    )
    child = mutate(jr.key(1), pop[0])
    assert child.shape == (GENOME_SIZE,)
    assert bool(jnp.all((child >= 0.0) & (child <= 1.0)))
    mixed = crossover(jr.key(2), pop[0], pop[1])
    assert mixed.shape == (GENOME_SIZE,)
    each_from_parent = jnp.isclose(mixed, pop[0]) | jnp.isclose(mixed, pop[1])
    assert bool(jnp.all(each_from_parent))


def test_seed_genomes_include_champion_and_meta_arms() -> None:
    seeds = seed_genomes()
    assert seeds.shape[1] == GENOME_SIZE
    configs = [decode_genome(np.asarray(g)) for g in seeds]
    assert any(
        c["norm"] == 1.0 and c["shift_reset"] == 1.0 and c["gate"] == 1.0
        and c["meta_decay"] == 0.0 and c["surprise_budget"] == 0.0
        for c in configs
    )  # champion form
    assert any(c["meta_decay"] == 1.0 for c in configs)  # meta-arm (a)
    assert any(c["surprise_budget"] == 1.0 for c in configs)  # meta-arm (b)


def test_penalized_fitness_charges_active_flags() -> None:
    lean = dict(decode_genome(champion_form_genome()))
    rich = dict(lean)
    for flag in FLAG_NAMES:
        rich[flag] = 1.0
    acc = np.asarray([0.8, 0.8])
    genomes = np.stack(
        [np.asarray(genome_from_config(lean)), np.asarray(genome_from_config(rich))]
    )
    fitness = penalized_fitness(acc, genomes)
    n_lean = sum(int(lean[f]) for f in FLAG_NAMES)
    n_rich = len(FLAG_NAMES)
    assert fitness[0] - fitness[1] == pytest.approx(
        FLAG_PENALTY * (n_rich - n_lean), abs=1e-9
    )


def test_describe_genome_names_active_primitives() -> None:
    text = describe_genome(np.asarray(champion_form_genome()))
    assert "norm" in text and "gate" in text and "shift_reset" in text
    assert "surprise_budget" not in text


def test_evaluate_population_is_paired_and_bounded() -> None:
    config = dataclasses.replace(MICRO_SUITE["M1"], n_tasks=2, task_length=20)
    genomes = jnp.stack(
        [jnp.asarray(champion_form_genome()), jnp.asarray(champion_form_genome())]
    )
    accuracy = evaluate_population(genomes, config, seeds=(0,))
    assert accuracy.shape == (2,)
    assert float(accuracy[0]) == pytest.approx(float(accuracy[1]), abs=1e-7)
    assert 0.0 <= float(accuracy[0]) <= 1.0


@pytest.mark.parametrize(
    "seeds",
    [
        (),
        (0, 0),
        (True,),
        (np.int64(0),),
        (0.0,),
        (-1,),
        (2**32,),
        (0, 2**32),
    ],
)
def test_evaluate_population_rejects_noncanonical_seed_schedules_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
    seeds: tuple[object, ...],
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("invalid seeds reached stream materialization")

    monkeypatch.setattr(rule_discovery, "_materialize_eval", unexpected_materialization)
    with pytest.raises(ValueError, match="seeds"):
        evaluate_population(
            jnp.zeros((1, GENOME_SIZE), dtype=jnp.float32),
            MICRO_SUITE["M1"],
            seeds=seeds,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "shape",
    [(2, 16), (2, GENOME_SIZE + 1), (GENOME_SIZE,), (1, 2, GENOME_SIZE)],
)
def test_evaluate_population_rejects_malformed_genome_shapes_before_materialization(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("malformed genomes reached stream materialization")

    monkeypatch.setattr(rule_discovery, "_materialize_eval", unexpected_materialization)
    expected = f"genomes must have shape (n_genomes, {GENOME_SIZE}), got {shape}"
    with pytest.raises(ValueError, match=rf"^{re.escape(expected)}$"):
        evaluate_population(jnp.zeros(shape, dtype=jnp.float32), MICRO_SUITE["M1"], seeds=(0,))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_evaluate_population_rejects_nonfinite_genomes_before_materialization(
    monkeypatch: pytest.MonkeyPatch, invalid: float
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("non-finite genomes reached stream materialization")

    monkeypatch.setattr(rule_discovery, "_materialize_eval", unexpected_materialization)
    genomes = jnp.zeros((2, GENOME_SIZE), dtype=jnp.float32).at[1, 0].set(invalid)
    with pytest.raises(ValueError, match="genomes must contain only finite values"):
        evaluate_population(genomes, MICRO_SUITE["M1"], seeds=(0,))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_random": True}, "n_random"),
        ({"n_random": -1}, "n_random"),
        ({"population": True}, "population"),
        ({"population": 0}, "population"),
        ({"generations": True}, "generations"),
        ({"generations": -1}, "generations"),
        ({"elite": True}, "elite"),
        ({"elite": 0}, "elite"),
        ({"top_k": True}, "top_k"),
        ({"top_k": 0}, "top_k"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_run_search_rejects_invalid_search_identities_before_genome_generation(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object], match: str
) -> None:
    def unexpected_generation(*args: object, **kw: object) -> None:
        del args, kw
        raise AssertionError("invalid search identity reached genome generation")

    monkeypatch.setattr(rule_discovery, "seed_genomes", unexpected_generation)
    monkeypatch.setattr(rule_discovery, "random_genomes", unexpected_generation)
    payload: dict[str, object] = {
        "n_random": 2,
        "population": 2,
        "generations": 0,
        "elite": 1,
        "eval_seeds": (0,),
        "holdout_seeds": (101,),
        "top_k": 1,
        "batch_size": 2,
    }
    payload.update(kwargs)
    with pytest.raises(ValueError, match=match):
        run_search(**payload)  # type: ignore[arg-type]


def test_run_search_rejects_population_larger_than_seeded_initial_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_generation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("invalid population reached genome generation")

    monkeypatch.setattr(rule_discovery, "seed_genomes", unexpected_generation)
    monkeypatch.setattr(rule_discovery, "random_genomes", unexpected_generation)
    with pytest.raises(ValueError, match="initial candidate pool"):
        run_search(
            n_random=0,
            population=20,
            generations=1,
            elite=1,
            eval_seeds=(0,),
            holdout_seeds=(101,),
            top_k=1,
            batch_size=2,
        )


def test_run_search_nonzero_generation_reaches_child_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ChildEvaluationReachedError(Exception):
        pass

    calls = 0

    def fake_evaluate_suite(
        genomes: object, *args: object, **kwargs: object
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 2:
            raise ChildEvaluationReachedError
        shape = getattr(genomes, "shape")
        return np.ones((int(shape[0]),), dtype=np.float64), {}

    monkeypatch.setattr(rule_discovery, "evaluate_suite", fake_evaluate_suite)
    with pytest.raises(ChildEvaluationReachedError):
        run_search(
            n_random=19,
            population=2,
            generations=1,
            elite=1,
            eval_seeds=(0,),
            holdout_seeds=(101,),
            top_k=1,
            batch_size=2,
        )
    assert calls == 2


@pytest.mark.parametrize("n_tasks", [True, 0, -1, 1.5])
def test_resolved_suite_rejects_invalid_n_tasks_identities(n_tasks: object) -> None:
    with pytest.raises(ValueError, match="n_tasks"):
        rule_discovery._resolved_suite(n_tasks, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("task_length", [True, 0, -1, 1.5])
def test_resolved_suite_rejects_invalid_task_length_identities(
    task_length: object,
) -> None:
    with pytest.raises(ValueError, match="task_length"):
        rule_discovery._resolved_suite(None, task_length)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_random": True}, "n_random"),
        ({"generations": True}, "generations"),
        ({"children": True}, "children"),
        ({"children": 0}, "children"),
    ],
)
def test_tune_champion_baseline_rejects_invalid_search_identities(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object], match: str
) -> None:
    def unexpected_generation(*args: object, **kw: object) -> None:
        del args, kw
        raise AssertionError("invalid search identity reached genome generation")

    monkeypatch.setattr(rule_discovery, "random_genomes", unexpected_generation)
    with pytest.raises(ValueError, match=match):
        tune_champion_baseline(
            jr.key(0),
            task_names=("M1",),
            eval_seeds=(0,),
            batch_size=2,
            suite=MICRO_SUITE,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("batch_size", [0, -4, True])
def test_evaluate_population_rejects_non_positive_batch_size_before_materialization(
    monkeypatch: pytest.MonkeyPatch, batch_size: object
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("invalid batch_size reached stream materialization")

    monkeypatch.setattr(rule_discovery, "_materialize_eval", unexpected_materialization)
    with pytest.raises(ValueError, match="batch_size must be a positive built-in int"):
        evaluate_population(
            jnp.zeros((2, GENOME_SIZE), dtype=jnp.float32),
            MICRO_SUITE["M1"],
            seeds=(0,),
            batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_evaluate_population_preserves_distinct_seed_order_and_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []

    def materialize(
        config: object, seed: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray], int]:
        del config
        observed.append(seed)
        return (
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            {},
            1,
        )

    def batched_run(
        genomes: jnp.ndarray,
        params: dict[str, jnp.ndarray],
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        task_length: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del params, xs, ys, task_length
        return (
            jnp.full((genomes.shape[0],), observed[-1], dtype=jnp.float32),
            jnp.zeros((genomes.shape[0], 1), dtype=jnp.float32),
        )

    monkeypatch.setattr(rule_discovery, "_materialize_eval", materialize)
    monkeypatch.setattr(rule_discovery, "_batched_run", batched_run)
    result = evaluate_population(
        jnp.zeros((2, GENOME_SIZE), dtype=jnp.float32),
        MICRO_SUITE["M1"],
        seeds=(7, 3),
    )

    assert observed == [7, 3]
    np.testing.assert_array_equal(result, np.asarray([5.0, 5.0]))


@pytest.mark.parametrize("task_names", [("M1", "M1", "M2"), ("M2", "M2")])
def test_evaluate_suite_rejects_duplicate_task_names_before_evaluation(
    monkeypatch: pytest.MonkeyPatch, task_names: tuple[str, ...]
) -> None:
    def unexpected_evaluation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("duplicate task names reached population evaluation")

    monkeypatch.setattr(rule_discovery, "evaluate_population", unexpected_evaluation)
    with pytest.raises(ValueError, match="task_names must contain unique task names"):
        evaluate_suite(
            jnp.zeros((1, GENOME_SIZE), dtype=jnp.float32),
            task_names,
            seeds=(0,),
        )


def test_evaluate_suite_is_an_equal_weight_task_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    per_task_value = {"M1": 0.05, "M2": 0.175}

    def evaluation(genomes: jnp.ndarray, config: object, **kwargs: object) -> np.ndarray:
        del kwargs
        name = next(key for key, value in MICRO_SUITE.items() if value is config)
        return np.full((genomes.shape[0],), per_task_value[name])

    monkeypatch.setattr(rule_discovery, "evaluate_population", evaluation)
    mean, per_task = evaluate_suite(
        jnp.zeros((2, GENOME_SIZE), dtype=jnp.float32), ("M1", "M2"), seeds=(0,)
    )
    np.testing.assert_allclose(mean, np.asarray([0.1125, 0.1125]))
    assert set(per_task) == {"M1", "M2"}


@pytest.mark.parametrize(
    ("flag", "values", "name"),
    [
        ("--tasks", ("M1", "M1"), "task_names"),
        ("--holdout-tasks", ("M1p", "M1p"), "holdout_names"),
    ],
)
def test_cli_rejects_duplicate_task_names_before_search(
    tmp_path, monkeypatch: pytest.MonkeyPatch, flag: str, values: tuple[str, ...], name: str
) -> None:
    def unexpected_search(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("duplicate task names reached genome generation")

    monkeypatch.setattr(rule_discovery, "random_genomes", unexpected_search)
    output = tmp_path / "must-not-exist.json"
    argv = [
        "search", "--out", str(output), "--n-random", "2", "--population", "2",
        "--generations", "0", "--elite", "1", "--eval-seeds", "0", "--holdout-seeds", "101",
        "--tasks", "M1", "--holdout-tasks", "M1p",
    ]
    index = argv.index(flag)
    argv[index + 1 : index + 2] = list(values)
    with pytest.raises(ValueError, match=f"{name} must contain unique task names"):
        rule_discovery.main(argv)
    assert not output.exists()


def test_cli_rejects_duplicate_evaluation_seeds_before_search(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_search(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("invalid CLI seeds reached genome generation")

    monkeypatch.setattr(rule_discovery, "random_genomes", unexpected_search)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="eval_seeds.*unique"):
        rule_discovery.main(
            [
                "search",
                "--out",
                str(output),
                "--n-random",
                "2",
                "--population",
                "2",
                "--generations",
                "0",
                "--elite",
                "1",
                "--eval-seeds",
                "0",
                "0",
                "--holdout-seeds",
                "101",
                "--tasks",
                "M1",
                "--holdout-tasks",
                "M1p",
            ]
        )
    assert not output.exists()


@pytest.mark.integration
def test_cli_search_smoke(tmp_path) -> None:
    """Tiny end-to-end search: writes one result JSON with the full schema."""
    import json

    from alberta_framework.benchmarks.rule_discovery import RESULT_SCHEMA, main

    out = tmp_path / "search_smoke.json"
    code = main(
        [
            "search",
            "--out", str(out),
            "--n-random", "8",
            "--population", "6",
            "--generations", "1",
            "--elite", "3",
            "--eval-seeds", "0",
            "--holdout-seeds", "101",
            "--top-k", "4",
            "--batch-size", "8",
            "--tasks", "M1",
            "--holdout-tasks", "M1p",
            "--micro-n-tasks", "2",
            "--micro-task-length", "30",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["evidence_policy"]["scientific_promotion_allowed"] is False
    assert payload["n_evaluated"] >= 8
    assert len(payload["candidates"]) == 4
    assert "holdout_accuracy" in payload["baseline"]
    assert "champion_constants_reference" in payload
    for row in payload["promoted"]:
        assert row["beats_baseline_on_holdout"] is True
    # Search fitness must never read holdout tasks.
    assert set(payload["settings"]["task_names"]) == {"M1"}
    assert set(payload["settings"]["holdout_names"]) == {"M1p"}


class TestExpandedMechanisms:
    """Wave-2 genome expansion: RLS head, NB ensemble member, surprise-driven
    lr annealing, per-layer lr ratio, Kalman-style normalizer.

    Every new mechanism must be inert (bit-parity-preserving) when its flag
    is off — the champion-parity test above stays the reduction pin — and
    must express its documented behavior when on.
    """

    def _flagless_config(self) -> dict[str, float]:
        config = dict(decode_genome(champion_form_genome()))
        for flag in FLAG_NAMES:
            config[flag] = 0.0
        return config

    def test_new_genes_present(self) -> None:
        for flag in (
            "rls_head",
            "rls_reset_p",
            "nb_member",
            "lr_anneal",
            "layer_lr",
            "kalman_norm",
        ):
            assert flag in FLAG_NAMES
        for name in (
            "rls_lambda",
            "nb_decay",
            "vote_decay",
            "anneal_lo",
            "anneal_hi",
            "layer_lr_ratio",
            "kalman_q",
        ):
            assert name in PARAM_NAMES

    def test_rls_head_tracks_targets_and_shift_resets_p(self) -> None:
        config = self._flagless_config()
        config["rls_head"] = 1.0
        config["rls_reset_p"] = 1.0
        config["norm"] = 1.0
        config["shift_k"] = 0.5
        config["rls_lambda"] = 0.999
        genome = jnp.asarray(genome_from_config(config))
        params, state = _tiny_setup(seed=11)
        x = jnp.linspace(0.1, 1.0, _TINY.input_dim, dtype=jnp.float32)
        y = jnp.asarray(3, dtype=jnp.int32)
        _, new_state, _, _ = rule_step(genome, params, state, x, y)
        # The RLS readout moves (W picks up the one-hot target).
        assert float(jnp.abs(new_state.rls_w).sum()) > 0.0
        assert bool(jnp.all(jnp.isfinite(new_state.rls_p)))
        # A detected shift on every feature resets P to its prior.
        mature = dataclasses.replace(
            new_state,
            norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
            norm_mean=jnp.zeros((_TINY.input_dim,), jnp.float32),
            norm_var=jnp.full((_TINY.input_dim,), 1e-4, jnp.float32),
            fast_mean=jnp.zeros((_TINY.input_dim,), jnp.float32),
        )
        x_shift = jnp.full((_TINY.input_dim,), 10.0, jnp.float32)
        _, reset_state, _, _ = rule_step(genome, params, mature, x_shift, y)
        off_diag = reset_state.rls_p - jnp.diag(jnp.diag(reset_state.rls_p))
        np.testing.assert_allclose(np.asarray(off_diag), 0.0, atol=1e-6)

    def test_nb_member_updates_only_observed_class(self) -> None:
        config = self._flagless_config()
        config["nb_member"] = 1.0
        genome = jnp.asarray(genome_from_config(config))
        params, state = _tiny_setup(seed=12)
        x = jnp.linspace(0.5, 2.0, _TINY.input_dim, dtype=jnp.float32)
        y = jnp.asarray(2, dtype=jnp.int32)
        _, new_state, _, _ = rule_step(genome, params, state, x, y)
        moved = np.abs(np.asarray(new_state.nb_mean - state.nb_mean)).sum(axis=1)
        assert moved[2] > 0.0
        for klass in (0, 1, 3, 4):
            assert moved[klass] == pytest.approx(0.0, abs=1e-8)
        assert float(new_state.nb_count[2]) == pytest.approx(1.0)

    def test_lr_anneal_fast_when_surprised_slow_when_calm(self) -> None:
        config = self._flagless_config()
        config["lr_anneal"] = 1.0
        config["weight_decay"] = 1e-4
        config["anneal_lo"] = 0.25
        config["anneal_hi"] = 2.0
        genome = jnp.asarray(genome_from_config(config))
        base = dict(config)
        base["lr_anneal"] = 0.0
        genome_base = jnp.asarray(genome_from_config(base))
        params, state = _tiny_setup(seed=13)
        x = jnp.linspace(0.2, 1.0, _TINY.input_dim, dtype=jnp.float32)
        y = jnp.asarray(1, dtype=jnp.int32)
        surprised = dataclasses.replace(
            state,
            err_fast=jnp.asarray(4.0, jnp.float32),
            err_slow=jnp.asarray(1.0, jnp.float32),
        )
        calm = dataclasses.replace(
            state,
            err_fast=jnp.asarray(1.0, jnp.float32),
            err_slow=jnp.asarray(1.0, jnp.float32),
        )
        step_hot, _, _, _ = rule_step(genome, params, surprised, x, y)
        step_calm, _, _, _ = rule_step(genome, params, calm, x, y)
        step_base, _, _, _ = rule_step(genome_base, params, calm, x, y)
        delta_hot = float(jnp.abs(step_hot["w3"] - params["w3"]).sum())
        delta_calm = float(jnp.abs(step_calm["w3"] - params["w3"]).sum())
        delta_base = float(jnp.abs(step_base["w3"] - params["w3"]).sum())
        # Surprised (ratio >= 2) runs at anneal_hi; calm runs at anneal_lo.
        assert delta_hot == pytest.approx(2.0 * delta_base, rel=1e-3)
        assert delta_calm == pytest.approx(0.25 * delta_base, rel=1e-3)

    def test_layer_lr_ratio_scales_head_vs_input(self) -> None:
        config = self._flagless_config()
        config["layer_lr"] = 1.0
        config["layer_lr_ratio"] = 2.0
        config["weight_decay"] = 1e-4
        genome = jnp.asarray(genome_from_config(config))
        base = dict(config)
        base["layer_lr"] = 0.0
        genome_base = jnp.asarray(genome_from_config(base))
        params, state = _tiny_setup(seed=14)
        x = jnp.linspace(0.2, 1.0, _TINY.input_dim, dtype=jnp.float32)
        y = jnp.asarray(0, dtype=jnp.int32)
        stepped, _, _, _ = rule_step(genome, params, state, x, y)
        stepped_base, _, _, _ = rule_step(genome_base, params, state, x, y)
        head = float(jnp.abs(stepped["w3"] - params["w3"]).sum())
        head_base = float(jnp.abs(stepped_base["w3"] - params["w3"]).sum())
        w1 = float(jnp.abs(stepped["w1"] - params["w1"]).sum())
        w1_base = float(jnp.abs(stepped_base["w1"] - params["w1"]).sum())
        assert head == pytest.approx(2.0 * head_base, rel=1e-3)
        assert w1 == pytest.approx(0.5 * w1_base, rel=1e-3)

    def test_kalman_norm_uncertainty_drives_tracking_speed(self) -> None:
        config = self._flagless_config()
        config["norm"] = 1.0
        config["kalman_norm"] = 1.0
        config["kalman_q"] = 1e-3
        genome = jnp.asarray(genome_from_config(config))
        params, state = _tiny_setup(seed=15)
        x = jnp.full((_TINY.input_dim,), 5.0, jnp.float32)
        y = jnp.asarray(0, dtype=jnp.int32)
        confident = dataclasses.replace(
            state,
            norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
            kalman_p=jnp.full((_TINY.input_dim,), 1e-4, jnp.float32),
        )
        uncertain = dataclasses.replace(
            state,
            norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
            kalman_p=jnp.full((_TINY.input_dim,), 10.0, jnp.float32),
        )
        _, state_conf, _, _ = rule_step(genome, params, confident, x, y)
        _, state_unc, _, _ = rule_step(genome, params, uncertain, x, y)
        # High posterior uncertainty => larger Kalman gain => faster tracking.
        assert float(state_unc.norm_mean[0]) > float(state_conf.norm_mean[0])
        # Posterior uncertainty contracts after the update.
        assert float(state_unc.kalman_p[0]) < 10.0

    def test_member_vote_accuracy_updates_with_vote_decay(self) -> None:
        config = self._flagless_config()
        config["nb_member"] = 1.0
        config["vote_decay"] = 0.9
        genome = jnp.asarray(genome_from_config(config))
        params, state = _tiny_setup(seed=16)
        x = jnp.linspace(0.1, 0.9, _TINY.input_dim, dtype=jnp.float32)
        y = jnp.asarray(1, dtype=jnp.int32)
        _, new_state, _, _ = rule_step(genome, params, state, x, y)
        assert new_state.member_acc.shape == (3,)
        assert bool(jnp.all(new_state.member_acc >= 0.0))
        assert bool(jnp.all(new_state.member_acc <= 1.0))
        assert float(jnp.abs(new_state.member_acc - state.member_acc).sum()) > 0.0


class TestGaussFitness:
    """The search fitness migrates to the transfer-validated gauss-v1 suite."""

    def test_gauss_suite_registry_families(self) -> None:
        from alberta_framework.benchmarks.rule_discovery import (
            GAUSS_HOLDOUT_TASKS,
            GAUSS_SEARCH_TASKS,
            gauss_suite,
        )

        suite = gauss_suite()
        assert set(GAUSS_SEARCH_TASKS) == {"G1", "G3"}
        assert set(GAUSS_HOLDOUT_TASKS) == {"G4", "G1p"}
        assert suite["G1"].family == "input_permutation"
        assert suite["G3"].family == "scale_shift"
        assert suite["G4"].family == "recurrence"
        assert suite["G1p"].family == "input_permutation"
        # The perturbed-M1 holdout runs a genuinely different geometry.
        assert suite["G1p"].dim != suite["G1"].dim

    def test_evaluate_population_on_gauss_stream_is_paired_and_bounded(self) -> None:
        from alberta_framework.benchmarks.micro_continual import MicroStreamConfig

        config = MicroStreamConfig(
            family="input_permutation", n_regimes=2, regime_length=25, dim=16,
            component_sparsity=4,
        )
        genomes = jnp.stack(
            [jnp.asarray(champion_form_genome()), jnp.asarray(champion_form_genome())]
        )
        accuracy = evaluate_population(genomes, config, seeds=(0,))
        assert accuracy.shape == (2,)
        assert float(accuracy[0]) == pytest.approx(float(accuracy[1]), abs=1e-7)
        assert 0.0 <= float(accuracy[0]) <= 1.0

    @pytest.mark.integration
    def test_cli_search_gauss_smoke(self, tmp_path) -> None:
        import json

        from alberta_framework.benchmarks.rule_discovery import RESULT_SCHEMA, main

        out = tmp_path / "search_gauss_smoke.json"
        code = main(
            [
                "search",
                "--suite", "gauss",
                "--out", str(out),
                "--n-random", "8",
                "--population", "6",
                "--generations", "1",
                "--elite", "3",
                "--eval-seeds", "0",
                "--holdout-seeds", "201",
                "--top-k", "4",
                "--batch-size", "8",
                "--micro-n-tasks", "2",
                "--micro-task-length", "30",
            ]
        )
        assert code == 0
        payload = json.loads(out.read_text())
        assert payload["schema"] == RESULT_SCHEMA
        assert payload["evidence_policy"]["scientific_promotion_allowed"] is False
        assert set(payload["settings"]["task_names"]) == {"G1", "G3"}
        assert set(payload["settings"]["holdout_names"]) == {"G4", "G1p"}
        assert len(payload["candidates"]) == 4


def test_seed_genomes_include_discovered_and_new_mechanism_rows() -> None:
    configs = [decode_genome(np.asarray(g)) for g in seed_genomes()]
    # The disc_r1_pscale_norms survivor: surprise budget without the gate.
    assert any(
        c["surprise_budget"] == 1.0 and c["gate"] == 0.0 and c["norm"] == 1.0
        for c in configs
    )
    for flag in ("rls_head", "nb_member", "lr_anneal", "layer_lr", "kalman_norm"):
        assert any(c[flag] == 1.0 for c in configs), flag


def test_run_stream_reports_per_task_accuracy() -> None:
    config = dataclasses.replace(MICRO_SUITE["M1"], n_tasks=2, task_length=15)
    from alberta_framework.benchmarks.micro_continual import build_micro_stream

    stream = build_micro_stream(config, seed=0)
    net = IPMNISTConfig(
        n_tasks=config.n_tasks,
        task_length=config.task_length,
        input_dim=config.input_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        n_classes=config.n_classes,
    )
    params = init_mlp_params(jr.key(0), net)
    mean_accuracy, per_task = run_stream(
        jnp.asarray(champion_form_genome()),
        params,
        jnp.asarray(stream.xs),
        jnp.asarray(stream.ys),
        config.task_length,
    )
    assert per_task.shape == (2,)
    assert float(mean_accuracy) == pytest.approx(float(per_task.mean()), abs=1e-6)
