"""Validation of the publication-statistics machinery in ``utils/statistics.py``.

Every certification claim in the framework leans on these functions (confidence
intervals, paired significance tests, multiple-comparison corrections, effect
sizes), so this file validates them empirically against known distributions and
hand-computed fixtures rather than trusting the implementation.

Calibration (measured on this machine, scripts in the session scratchpad):

- t-based 95% CI empirical coverage over 2000 replications of n=10 draws from
  N(3, 2): 0.9445 / 0.9570 / 0.9480 / 0.9500 across four data seeds
  (binomial std at 2000 reps is ~0.0049). Assertion band [0.93, 0.97] is
  ~3x the observed spread on each side.
- Percentile-bootstrap 95% CI coverage (n=25 samples, 250 replications,
  400 bootstrap resamples): 0.920 / 0.896 / 0.924 across three data seeds —
  the percentile method under-covers at small n, which is expected.
  Assertion band [0.85, 0.99] leaves >2 binomial sigma below the worst
  observed value.
- Paired t-test under the null (400 replications, 20 pairs, alpha=0.05):
  rejection rate 0.0575; Wilcoxon: 0.0500. Assertion <= 0.10 (= 2*alpha,
  ~4.5 binomial sigma above the mean under exact calibration).
- Power under a strong paired shift (+1.0 with sd-0.5 noise, 20 pairs):
  1.000 for both tests over 400 replications. Assertion >= 0.95.
- Holm-vs-Bonferroni superset property held on 2000/2000 random p-value
  draws, with Holm strictly larger on 295 of them. Assertion requires the
  superset always and strictness on >= 50 draws.
"""

import sys
import warnings
from typing import Any

import numpy as np
import pytest

from alberta_framework.utils.experiments import AggregatedResults
from alberta_framework.utils.statistics import (
    SignificanceResult,
    bonferroni_correction,
    bootstrap_ci,
    cohens_d,
    common_final_window,
    compute_statistics,
    compute_timeseries_statistics,
    holm_correction,
    mann_whitney_comparison,
    pairwise_comparisons,
    ttest_comparison,
    wilcoxon_comparison,
)

# ---------------------------------------------------------------------------
# 1. Confidence intervals: hand-computed fixtures + empirical coverage
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    def test_hand_computed_fixture(self) -> None:
        """All summary fields match hand-derived values for [1, 2, 3, 4, 5]."""
        s = compute_statistics([1.0, 2.0, 3.0, 4.0, 5.0], confidence_level=0.95)
        assert s.mean == pytest.approx(3.0)
        assert s.std == pytest.approx(np.sqrt(2.5))  # ddof=1
        assert s.sem == pytest.approx(np.sqrt(2.5) / np.sqrt(5))
        assert s.median == pytest.approx(3.0)
        assert s.iqr == pytest.approx(2.0)  # 75th=4, 25th=2
        assert s.n_seeds == 5
        # t_{0.975, df=4} = 2.7764; margin = 2.7764 * 0.70711 = 1.9633
        assert s.ci_lower == pytest.approx(3.0 - 2.7764 * np.sqrt(0.5), abs=1e-3)
        assert s.ci_upper == pytest.approx(3.0 + 2.7764 * np.sqrt(0.5), abs=1e-3)
        # CI must bracket the mean symmetrically
        assert s.ci_lower < s.mean < s.ci_upper

    def test_single_value_degenerate(self) -> None:
        s = compute_statistics([4.2])
        assert s.mean == pytest.approx(4.2)
        assert s.std == 0.0
        assert s.sem == 0.0
        assert s.ci_lower == pytest.approx(4.2)
        assert s.ci_upper == pytest.approx(4.2)
        assert s.n_seeds == 1

    def test_large_constant_values_remain_finite_without_warnings(self) -> None:
        values = np.asarray([1.0e308, 1.0e308], dtype=np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary = compute_statistics(values)

        assert summary.mean == 1.0e308
        assert summary.std == 0.0
        assert summary.sem == 0.0
        assert summary.ci_lower == 1.0e308
        assert summary.ci_upper == 1.0e308
        assert summary.median == 1.0e308
        assert summary.iqr == 0.0

    def test_unrepresentable_large_interval_fails_closed_without_warnings(self) -> None:
        values = np.asarray([-1.0e308, 1.0e308], dtype=np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError,
                match=r"^values cannot be represented as finite float64 statistics$",
            ):
                compute_statistics(values)

    def test_empty_values_rejected_without_warnings(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"^values must be non-empty$"):
                compute_statistics([])

    def test_empty_array_rejected_without_warnings(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"^values must be non-empty$"):
                compute_statistics(np.array([], dtype=np.float64))

    @pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_values_rejected_without_warnings(self, poison: float) -> None:
        """A NaN or inf seed must not become a NaN mean/CI that looks published."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"^values must be finite$"):
                compute_statistics([1.0, poison, 2.0])

    def test_empirical_ci_coverage(self) -> None:
        """95% t-CI covers the true mean ~95% of the time.

        2000 replications of n=10 Gaussian draws. Measured coverage across
        seeds: 0.9445-0.9570 (see module docstring); band [0.93, 0.97] is a
        multi-sigma margin around nominal 0.95.
        """
        rng = np.random.default_rng(0)
        true_mean, true_sd = 3.0, 2.0
        n_reps, n = 2000, 10
        data = rng.normal(true_mean, true_sd, size=(n_reps, n))
        covered = 0
        for i in range(n_reps):
            s = compute_statistics(data[i], confidence_level=0.95)
            covered += int(s.ci_lower <= true_mean <= s.ci_upper)
        coverage = covered / n_reps
        assert 0.93 <= coverage <= 0.97, f"coverage {coverage} outside [0.93, 0.97]"

    def test_wider_confidence_level_gives_wider_interval(self) -> None:
        values = np.random.default_rng(1).normal(0.0, 1.0, size=20)
        s95 = compute_statistics(values, confidence_level=0.95)
        s99 = compute_statistics(values, confidence_level=0.99)
        assert (s99.ci_upper - s99.ci_lower) > (s95.ci_upper - s95.ci_lower)

    @pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1, float("nan")])
    def test_invalid_confidence_level_rejected_without_warnings(
        self,
        confidence_level: float,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="confidence_level.*strictly between 0 and 1"):
                compute_statistics([4.2], confidence_level=confidence_level)

    def test_missing_scipy_raises_instead_of_silently_wrong_ci(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without scipy, compute_statistics must fail loudly (matching
        ttest_comparison/mann_whitney_comparison/wilcoxon_comparison's own
        established convention in this file), not silently substitute the
        99% z-quantile for every confidence_level != 0.95.
        """
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        with pytest.raises(ImportError, match="scipy is required for compute_statistics"):
            compute_statistics([1.0, 2.0, 3.0, 4.0, 5.0], confidence_level=0.90)


class TestSampleVectorContract:
    """Every per-seed sample surface takes exactly one value per seed."""

    _MATRIX = np.tile(np.arange(1.0, 6.0), (3, 1))  # (n_seeds=3, n_steps=5), rows identical

    def test_compute_statistics_rejects_a_seed_by_step_matrix(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError,
                match=r"^values must be a one-dimensional sample vector \(one value per seed\), "
                r"got shape \(3, 5\); reduce per seed first or use "
                r"compute_timeseries_statistics$",
            ):
                compute_statistics(self._MATRIX)

    def test_bootstrap_ci_rejects_a_seed_by_step_matrix(self) -> None:
        with pytest.raises(ValueError, match="values must be a one-dimensional sample vector"):
            bootstrap_ci(self._MATRIX, n_bootstrap=10)

    def test_cohens_d_rejects_seed_by_step_matrices(self) -> None:
        with pytest.raises(
            ValueError, match="values_a must be a one-dimensional sample vector"
        ):
            cohens_d(self._MATRIX, np.arange(1.0, 6.0))
        with pytest.raises(
            ValueError, match="values_b must be a one-dimensional sample vector"
        ):
            cohens_d(np.arange(1.0, 6.0), self._MATRIX)

    @pytest.mark.parametrize(
        "comparison",
        [
            lambda a, b: ttest_comparison(a, b),
            lambda a, b: ttest_comparison(a, b, paired=True),
            mann_whitney_comparison,
            wilcoxon_comparison,
        ],
        ids=["ttest", "paired-ttest", "mann_whitney", "wilcoxon"],
    )
    def test_comparisons_reject_seed_by_step_matrices(self, comparison: Any) -> None:
        vector = np.arange(1.0, 6.0) + 0.5
        with pytest.raises(
            ValueError, match="values_a must be a one-dimensional sample vector"
        ):
            comparison(self._MATRIX, vector)
        with pytest.raises(
            ValueError, match="values_b must be a one-dimensional sample vector"
        ):
            comparison(vector, self._MATRIX)

    def test_scalar_and_zero_dimensional_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="values must be a one-dimensional sample vector"):
            compute_statistics(np.asarray(4.2))
        with pytest.raises(ValueError, match="values must be a one-dimensional sample vector"):
            bootstrap_ci(np.asarray(4.2), n_bootstrap=10)


class TestComparisonsRejectNonFiniteSamples:
    """A poisoned seed must raise, not become p=nan / significant=False."""

    @pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize(
        "comparison",
        [
            lambda a, b: ttest_comparison(a, b, paired=False),
            lambda a, b: ttest_comparison(a, b, paired=True),
            mann_whitney_comparison,
            wilcoxon_comparison,
            cohens_d,
        ],
        ids=["ttest", "paired-ttest", "mann_whitney", "wilcoxon", "cohens_d"],
    )
    def test_nonfinite_sample_rejected_without_warnings(
        self, comparison: Any, poison: float
    ) -> None:
        clean = np.asarray([1.0, 2.0, 3.0, 4.5])
        poisoned = np.asarray([2.0, poison, 3.0, 5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"^values_b must be finite$"):
                comparison(clean, poisoned)
            with pytest.raises(ValueError, match=r"^values_a must be finite$"):
                comparison(poisoned, clean)

    def test_pairwise_comparisons_rejects_a_poisoned_seed(self) -> None:
        a = _make_seeded_aggregated("a", [0, 1, 2], [0.0, 1.0, 2.0])
        b = _make_seeded_aggregated("b", [0, 1, 2], [1.0, float("nan"), 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="must be finite"):
                pairwise_comparisons({"a": a, "b": b}, test="ttest", window=1)


class _FloatClassSpoof:
    """Plain object whose reported ``__class__`` fools ``isinstance``."""

    def __repr__(self) -> str:
        """Keep parametrized node IDs stable across xdist worker processes."""
        return "_FloatClassSpoof()"

    @property
    def __class__(self) -> type[float]:
        return float

    def __float__(self) -> float:
        return 0.05


class TestProbabilityContracts:
    """Decision thresholds and published p-values must be real probabilities."""

    _A = np.asarray([1.0, 2.0, 4.0, 8.0])
    _B = np.asarray([1.5, 2.5, 3.5, 6.0])

    @pytest.mark.parametrize(
        "alpha",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            1.0,
            -0.1,
            1.1,
            True,
            "0.05",
            None,
            _FloatClassSpoof(),
        ],
    )
    @pytest.mark.parametrize(
        "comparison",
        [
            lambda a, b, alpha: ttest_comparison(a, b, paired=False, alpha=alpha),
            lambda a, b, alpha: ttest_comparison(a, b, paired=True, alpha=alpha),
            lambda a, b, alpha: mann_whitney_comparison(a, b, alpha=alpha),
            lambda a, b, alpha: wilcoxon_comparison(a, b, alpha=alpha),
        ],
        ids=["independent-ttest", "paired-ttest", "mann-whitney", "wilcoxon"],
    )
    def test_comparisons_reject_invalid_alpha(self, comparison: Any, alpha: Any) -> None:
        with pytest.raises(
            ValueError,
            match=r"^alpha must be a finite real strictly between 0 and 1$",
        ):
            comparison(self._A, self._B, alpha)

    @pytest.mark.parametrize("correction", [bonferroni_correction, holm_correction])
    @pytest.mark.parametrize(
        "alpha",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            1.0,
            -0.1,
            1.1,
            True,
            _FloatClassSpoof(),
        ],
    )
    def test_corrections_reject_invalid_alpha_even_for_an_empty_family(
        self, correction: Any, alpha: Any
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"^alpha must be a finite real strictly between 0 and 1$",
        ):
            correction([], alpha=alpha)

    @pytest.mark.parametrize("correction", [bonferroni_correction, holm_correction])
    @pytest.mark.parametrize(
        "invalid_p",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.1,
            1.1,
            True,
            "0.1",
            None,
            _FloatClassSpoof(),
        ],
    )
    def test_corrections_reject_invalid_p_values(
        self, correction: Any, invalid_p: Any
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"^p_values\[1\] must be a finite real in \[0, 1\]$",
        ):
            correction([0.01, invalid_p, 0.2], alpha=0.05)

    @pytest.mark.parametrize("correction", [bonferroni_correction, holm_correction])
    def test_corrections_accept_exact_probability_boundaries(self, correction: Any) -> None:
        result = correction([0.0, 1.0], alpha=0.05)
        significant = result[0] if isinstance(result, tuple) else result
        assert significant == [True, False]

    def test_comparison_rejects_a_nonfinite_scipy_p_value(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(
                ValueError,
                match=(
                    r"^p_value returned by independent t-test must be a finite real "
                    r"in \[0, 1\]$"
                ),
            ):
                ttest_comparison([1.0, 1.0], [1.0, 1.0], paired=False)


class TestTimeseriesStatistics:
    def test_matches_per_column_compute_statistics(self) -> None:
        """Vectorised timeseries CI agrees with per-step scalar CI."""
        rng = np.random.default_rng(2)
        arr = rng.normal(1.0, 0.5, size=(8, 6))  # (n_seeds, n_steps)
        mean, lo, hi = compute_timeseries_statistics(arr, confidence_level=0.95)
        assert mean.shape == lo.shape == hi.shape == (6,)
        for step in range(6):
            s = compute_statistics(arr[:, step], confidence_level=0.95)
            assert mean[step] == pytest.approx(s.mean)
            assert lo[step] == pytest.approx(s.ci_lower)
            assert hi[step] == pytest.approx(s.ci_upper)

    def test_large_constant_column_remains_finite_without_warnings(self) -> None:
        arr = np.asarray([[1.0e308, 1.0], [1.0e308, 2.0]], dtype=np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mean, lo, hi = compute_timeseries_statistics(arr)

        assert np.isfinite(mean).all()
        assert np.isfinite(lo).all()
        assert np.isfinite(hi).all()
        assert mean[0] == 1.0e308
        assert lo[0] == 1.0e308
        assert hi[0] == 1.0e308

    def test_zero_seed_matrix_rejected_without_warnings(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError, match=r"^metric_array must contain at least one seed row$"
            ):
                compute_timeseries_statistics(np.empty((0, 3)))

    def test_nonfinite_seed_rejected_without_warnings(self) -> None:
        arr = np.array([[1.0, 2.0], [np.nan, 3.0]], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"^metric_array must be finite$"):
                compute_timeseries_statistics(arr)

    def test_single_seed_returns_finite_point_interval(self) -> None:
        """One seed yields the point trajectory, finite, without any warning."""
        arr = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mean, lo, hi = compute_timeseries_statistics(arr, confidence_level=0.95)
        assert mean.shape == lo.shape == hi.shape == (3,)
        np.testing.assert_array_equal(mean, np.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(lo, mean)
        np.testing.assert_array_equal(hi, mean)
        assert np.isfinite(mean).all()
        assert np.isfinite(lo).all()
        assert np.isfinite(hi).all()

    def test_single_seed_matches_per_column_compute_statistics(self) -> None:
        """The n_seeds == 1 contract agrees with the scalar degenerate case."""
        rng = np.random.default_rng(8)
        arr = rng.normal(0.0, 1.0, size=(1, 5))
        mean, lo, hi = compute_timeseries_statistics(arr, confidence_level=0.95)
        for step in range(5):
            s = compute_statistics(arr[:, step], confidence_level=0.95)
            assert mean[step] == pytest.approx(s.mean)
            assert lo[step] == pytest.approx(s.ci_lower)
            assert hi[step] == pytest.approx(s.ci_upper)

    @pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1, float("nan")])
    def test_invalid_confidence_level_rejected_without_warnings(
        self,
        confidence_level: float,
    ) -> None:
        arr = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="confidence_level.*strictly between 0 and 1"):
                compute_timeseries_statistics(arr, confidence_level=confidence_level)

    def test_missing_scipy_raises_instead_of_silently_wrong_ci(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same contract as compute_statistics: without scipy this must fail
        loudly rather than silently substitute the 99% z-quantile for every
        confidence_level != 0.95.
        """
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        arr = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with pytest.raises(
            ImportError, match="scipy is required for compute_timeseries_statistics"
        ):
            compute_timeseries_statistics(arr, confidence_level=0.90)


class TestBootstrapCI:
    def test_deterministic_and_brackets_estimate(self) -> None:
        values = np.random.default_rng(3).normal(5.0, 1.0, size=30)
        r1 = bootstrap_ci(values, statistic="mean", n_bootstrap=500, seed=42)
        r2 = bootstrap_ci(values, statistic="mean", n_bootstrap=500, seed=42)
        assert r1 == r2  # same seed, same result
        point, lo, hi = r1
        assert point == pytest.approx(float(np.mean(values)))
        assert lo < point < hi

    def test_median_statistic(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        point, lo, hi = bootstrap_ci(values, statistic="median", n_bootstrap=300, seed=0)
        assert point == pytest.approx(3.0)
        assert lo <= point <= hi

    def test_empirical_coverage(self) -> None:
        """Percentile-bootstrap 95% CI coverage stays near nominal.

        250 replications, n=25 samples, 400 resamples. Measured coverage
        0.896-0.924 across seeds (the percentile method under-covers at
        small n); assert the wide calibrated band [0.85, 0.99].
        """
        rng = np.random.default_rng(0)
        true_mean = 3.0
        n_reps, n, n_boot = 250, 25, 400
        covered = 0
        for i in range(n_reps):
            sample = rng.normal(true_mean, 2.0, size=n)
            _, lo, hi = bootstrap_ci(sample, statistic="mean", n_bootstrap=n_boot, seed=i)
            covered += int(lo <= true_mean <= hi)
        coverage = covered / n_reps
        assert 0.85 <= coverage <= 0.99, f"coverage {coverage} outside [0.85, 0.99]"

    @pytest.mark.parametrize(
        "empty", [[], np.array([], dtype=np.float64)], ids=["list", "ndarray"]
    )
    def test_empty_input_rejected(self, empty: list[float] | np.ndarray) -> None:
        """Empty input raises a descriptive ValueError, with no RuntimeWarning.

        Before the guard, ``np.mean([])`` warned and the helper returned
        ``(nan, nan, nan)`` — a NaN interval indistinguishable from a real CI.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="empty"):
                bootstrap_ci(empty)

    @pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_values_rejected_without_warnings(self, poison: float) -> None:
        """A poisoned seed must not become a non-finite bootstrap estimate or CI."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"^values must be finite$"):
                bootstrap_ci([1.0, poison, 2.0], n_bootstrap=20, seed=0)

    @pytest.mark.parametrize("statistic", ["typo", "Mean", ""])
    def test_unknown_statistic_rejected_without_warnings(self, statistic: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="statistic.*mean.*median"):
                bootstrap_ci([1.0, 2.0, 100.0], statistic=statistic, n_bootstrap=10)

    @pytest.mark.parametrize("n_bootstrap", [0, -1])
    def test_nonpositive_bootstrap_count_rejected_without_warnings(
        self,
        n_bootstrap: int,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="n_bootstrap.*positive"):
                bootstrap_ci([1.0, 2.0, 3.0], n_bootstrap=n_bootstrap)

    @pytest.mark.parametrize("n_bootstrap", [True, False, 1.0, float("nan"), float("inf")])
    def test_bool_and_noninteger_bootstrap_count_rejected_without_warnings(
        self,
        n_bootstrap: object,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="n_bootstrap.*positive integer"):
                bootstrap_ci([1.0, 2.0, 3.0], n_bootstrap=n_bootstrap, seed=0)  # type: ignore[arg-type]

    def test_hostile_bootstrap_count_is_rejected_without_hooks(self) -> None:
        class HostileInt(int):
            def __index__(self) -> int:  # pragma: no cover
                raise AssertionError("index hook executed")

            def __repr__(self) -> str:  # pragma: no cover
                raise AssertionError("repr hook executed")

        with pytest.raises(ValueError, match="n_bootstrap.*positive integer"):
            bootstrap_ci([1.0, 2.0, 3.0], n_bootstrap=HostileInt(10))

    @pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1, float("nan")])
    def test_invalid_confidence_level_rejected_without_warnings(
        self,
        confidence_level: float,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="confidence_level.*strictly between 0 and 1"):
                bootstrap_ci(
                    [4.2],
                    confidence_level=confidence_level,
                    n_bootstrap=10,
                )


# ---------------------------------------------------------------------------
# 2. Paired significance tests: null calibration + power
# ---------------------------------------------------------------------------


def _paired_replication_rates(
    test_fn,
    n_reps: int = 400,
    n_pairs: int = 20,
) -> tuple[float, float]:
    """Return (null rejection rate, power under a +1.0 paired shift)."""
    null_rej = alt_rej = 0
    for i in range(n_reps):
        r = np.random.default_rng(10_000 + i)
        base = r.normal(0.0, 1.0, size=n_pairs)
        a_null = base + r.normal(0.0, 0.5, size=n_pairs)
        b = base + r.normal(0.0, 0.5, size=n_pairs)
        null_rej += int(test_fn(a_null, b).significant)
        a_alt = b + 1.0 + r.normal(0.0, 0.5, size=n_pairs)
        alt_rej += int(test_fn(a_alt, b).significant)
    return null_rej / n_reps, alt_rej / n_reps


class TestPairedTests:
    def test_paired_ttest_null_and_power(self) -> None:
        """Null rejection ~alpha (measured 0.0575), power ~1.0 (measured 1.000)."""
        null_rate, power = _paired_replication_rates(
            lambda a, b: ttest_comparison(a, b, paired=True, alpha=0.05)
        )
        assert null_rate <= 0.10, f"null rejection rate {null_rate} > 2*alpha"
        assert power >= 0.95, f"power {power} < 0.95 under a strong true shift"

    def test_wilcoxon_null_and_power(self) -> None:
        """Null rejection ~alpha (measured 0.0500), power ~1.0 (measured 1.000)."""
        null_rate, power = _paired_replication_rates(
            lambda a, b: wilcoxon_comparison(a, b, alpha=0.05)
        )
        assert null_rate <= 0.10, f"null rejection rate {null_rate} > 2*alpha"
        assert power >= 0.95, f"power {power} < 0.95 under a strong true shift"

    def test_ttest_result_fields(self) -> None:
        res = ttest_comparison(
            [1.0, 2.0, 3.2], [1.3, 2.4, 3.5], paired=True, method_a="x", method_b="y"
        )
        assert isinstance(res, SignificanceResult)
        assert res.test_name == "paired t-test"
        assert res.method_a == "x" and res.method_b == "y"
        assert 0.0 <= res.p_value <= 1.0
        # every a_i < b_i: effect size sign must reflect a < b
        assert res.effect_size < 0.0

    def test_paired_ttest_rejects_identical_samples(self) -> None:
        values = [0.91, 0.88, 0.95]
        with pytest.raises(
            ValueError,
            match=r"^Paired comparison 'pin' vs 'base' has identical samples; "
            r"the paired t statistic is undefined$",
        ):
            ttest_comparison(values, list(values), paired=True, method_a="pin", method_b="base")

    def test_constant_nonzero_shift_stays_out_of_scope(self) -> None:
        res = ttest_comparison([1.0, 2.0, 3.0], [0.5, 1.5, 2.5], paired=True)
        assert np.isposinf(res.statistic)
        assert res.p_value == 0.0

    def test_unpaired_identical_samples_stay_out_of_scope(self) -> None:
        values = [0.91, 0.88, 0.95]
        res = ttest_comparison(values, list(values), paired=False)
        assert res.test_name == "independent t-test"
        assert res.p_value == pytest.approx(1.0)
        assert not res.significant

    def test_unpaired_ttest_separated_groups(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.normal(10.0, 0.5, size=15)
        b = rng.normal(0.0, 0.5, size=15)
        res = ttest_comparison(a, b, paired=False, alpha=0.01)
        assert res.test_name == "independent t-test"
        assert res.significant
        assert res.effect_size > 5.0  # enormous separation

    def test_mann_whitney_separated_groups(self) -> None:
        rng = np.random.default_rng(6)
        a = rng.normal(10.0, 0.5, size=15)
        b = rng.normal(0.0, 0.5, size=15)
        res = mann_whitney_comparison(a, b, alpha=0.01)
        assert res.test_name == "Mann-Whitney U"
        assert res.significant


class TestIdenticalWilcoxonRejection:
    """All-zero paired differences fail closed before version-dependent SciPy behavior."""

    @pytest.mark.parametrize("values", [[0.91], [0.91, 0.88, 0.95]])
    def test_wilcoxon_rejects_identical_samples_without_warning(
        self, values: list[float]
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError,
                match=r"^Paired comparison 'pin' vs 'base' has identical samples; "
                r"the Wilcoxon signed-rank statistic is undefined$",
            ):
                wilcoxon_comparison(
                    values,
                    list(values),
                    method_a="pin",
                    method_b="base",
                )

    def test_constant_nonzero_shift_stays_defined(self) -> None:
        result = wilcoxon_comparison([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])

        assert result.test_name == "Wilcoxon signed-rank"
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value < 1.0
        assert result.effect_size == cohens_d([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])


class TestOneSampleRejection:
    """Undefined one-sample contracts reject without narrowing valid t-tests (#35).

    A 1-vs-1 comparison has zero pooled degrees of freedom, so neither the
    paired/independent t statistic nor Cohen's d is defined. The helpers must
    reject before SciPy instead of emitting RuntimeWarning and crashing with
    ZeroDivisionError. An equal-variance independent t-test with one singleton
    group and one multi-value group has positive pooled degrees of freedom and
    stays defined. Mann-Whitney keeps its defined length-1 behavior.
    """

    def test_cohens_d_length_one_both_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="positive pooled degrees of freedom"):
            cohens_d([1.0], [2.0])

    def test_cohens_d_singleton_group_with_positive_pooled_df_stays_defined(self) -> None:
        expected = 3.0 / np.sqrt(2.0)
        assert cohens_d([1.0], [2.0, 3.0]) == pytest.approx(-expected)
        assert cohens_d([2.0, 3.0], [1.0]) == pytest.approx(expected)

    @pytest.mark.parametrize("values_a, values_b", [([], [1.0, 2.0]), ([1.0, 2.0], [])])
    def test_cohens_d_empty_group_raises_without_warning(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="non-empty groups"):
                cohens_d(values_a, values_b)

    def test_paired_ttest_length_one_raises_without_runtime_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="at least 2"):
                ttest_comparison([1.0], [2.0], paired=True)

    def test_paired_ttest_mismatched_lengths_raise_before_scipy(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            ttest_comparison([1.0, 2.0], [1.0, 2.0, 3.0], paired=True)

    def test_wilcoxon_mismatched_lengths_raise_before_scipy(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            wilcoxon_comparison([1.0, 2.0], [1.0, 2.0, 3.0])

    @pytest.mark.parametrize("values_a, values_b", [([], []), ([1.0], [2.0])])
    def test_wilcoxon_requires_two_pairs_without_warning(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="at least 2 pairs"):
                wilcoxon_comparison(values_a, values_b)

    def test_unpaired_ttest_length_one_raises_without_runtime_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="positive pooled degrees of freedom"):
                ttest_comparison([1.0], [2.0], paired=False)

    def test_unpaired_ttest_singleton_with_positive_pooled_df_stays_defined(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            forward = ttest_comparison([1.0], [2.0, 3.0], paired=False)
            reverse = ttest_comparison([2.0, 3.0], [1.0], paired=False)
        assert forward.statistic == pytest.approx(-np.sqrt(3.0))
        assert forward.p_value == pytest.approx(1.0 / 3.0)
        assert forward.effect_size == pytest.approx(-3.0 / np.sqrt(2.0))
        assert reverse.statistic == pytest.approx(-forward.statistic)
        assert reverse.p_value == pytest.approx(forward.p_value)
        assert reverse.effect_size == pytest.approx(-forward.effect_size)

    @pytest.mark.parametrize("values_a, values_b", [([], [1.0, 2.0]), ([1.0, 2.0], [])])
    def test_unpaired_ttest_empty_group_raises_without_warning(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="non-empty groups"):
                ttest_comparison(values_a, values_b, paired=False)

    @pytest.mark.parametrize(
        ("values_a", "values_b"),
        [
            ([], [1.0, 2.0]),
            ([1.0, 2.0], []),
            ([], []),
            (np.array([], dtype=np.float64), np.array([1.0, 2.0], dtype=np.float64)),
            (np.array([1.0, 2.0], dtype=np.float64), np.array([], dtype=np.float64)),
        ],
    )
    def test_mann_whitney_empty_group_raises_without_warning(
        self, values_a: list[float] | np.ndarray, values_b: list[float] | np.ndarray
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="non-empty groups"):
                mann_whitney_comparison(values_a, values_b)

    def test_mann_whitney_length_one_contract_unchanged(self) -> None:
        res = mann_whitney_comparison([1.0], [2.0])
        assert res.p_value == pytest.approx(1.0)
        assert res.effect_size == pytest.approx(-1.0)

    def test_mann_whitney_all_ties_have_the_exact_null_result(self) -> None:
        """Zero asymptotic tie variance is an exact p=1 null, never p=nan."""
        res = mann_whitney_comparison([1.0, 1.0], [1.0, 1.0, 1.0])
        assert res.statistic == pytest.approx(3.0)
        assert res.p_value == 1.0
        assert res.effect_size == 0.0
        assert not res.significant


# ---------------------------------------------------------------------------
# 3. Multiple-comparison corrections
# ---------------------------------------------------------------------------


class TestCorrections:
    def test_bonferroni_empty_p_values(self) -> None:
        significant, corrected_alpha = bonferroni_correction([], alpha=0.05)
        assert significant == []
        assert corrected_alpha == 0.05

    def test_bonferroni_hand_computed(self) -> None:
        significant, corrected_alpha = bonferroni_correction([0.01, 0.02, 0.04], alpha=0.05)
        assert corrected_alpha == pytest.approx(0.05 / 3)
        assert significant == [True, False, False]

    def test_holm_hand_computed_step_down(self) -> None:
        # sorted p: [0.01, 0.02, 0.03, 0.04]; thresholds [1/80, 1/60, 1/40, 1/20]
        # 0.01 < 0.0125 -> reject; 0.02 > 0.0167 -> stop: only p=0.01 rejected.
        assert holm_correction([0.03, 0.01, 0.04, 0.02], alpha=0.05) == [
            False,
            True,
            False,
            False,
        ]

    def test_holm_strictly_more_powerful_fixture(self) -> None:
        # Bonferroni (alpha/3 = 0.0167) rejects only the first two;
        # Holm thresholds [0.0167, 0.025, 0.05] reject all three.
        p = [0.01, 0.015, 0.04]
        bonf, _ = bonferroni_correction(p, alpha=0.05)
        holm = holm_correction(p, alpha=0.05)
        assert bonf == [True, True, False]
        assert holm == [True, True, True]

    def test_holm_rejects_superset_of_bonferroni_property(self) -> None:
        """Property: Holm rejections are always a superset of Bonferroni's.

        2000 random p-vectors (uniform, small-skewed beta, and mixed).
        Calibration: superset held on 2000/2000 draws and Holm was strictly
        larger on 295 draws; assert strictness on >= 50 (>10 sigma margin).
        """
        n_draws = 2000
        strictly_more = 0
        for i in range(n_draws):
            r = np.random.default_rng(50_000 + i)
            m = int(r.integers(2, 12))
            kind = i % 3
            if kind == 0:
                p = r.uniform(0, 1, size=m)
            elif kind == 1:
                p = r.beta(0.3, 4.0, size=m)
            else:
                p = np.concatenate(
                    [r.beta(0.2, 8.0, size=m // 2 + 1), r.uniform(0, 1, size=m // 2)]
                )[:m]
            p_list = [float(v) for v in p]
            bonf, _ = bonferroni_correction(p_list, alpha=0.05)
            holm = holm_correction(p_list, alpha=0.05)
            assert len(holm) == len(bonf) == m
            for b_sig, h_sig in zip(bonf, holm, strict=True):
                assert (not b_sig) or h_sig, (
                    f"Bonferroni rejected but Holm did not on p={p_list}"
                )
            strictly_more += int(sum(holm) > sum(bonf))
        assert strictly_more >= 50, f"Holm strictly larger on only {strictly_more}/2000 draws"

    def test_corrections_all_significant_and_none_significant(self) -> None:
        tiny = [1e-6, 1e-7, 1e-8]
        assert holm_correction(tiny, alpha=0.05) == [True, True, True]
        assert bonferroni_correction(tiny, alpha=0.05)[0] == [True, True, True]
        huge = [0.5, 0.9, 0.7]
        assert holm_correction(huge, alpha=0.05) == [False, False, False]
        assert bonferroni_correction(huge, alpha=0.05)[0] == [False, False, False]


# ---------------------------------------------------------------------------
# 4. Effect sizes: hand-computed fixtures
# ---------------------------------------------------------------------------


class TestEffectSizes:
    def test_cohens_d_hand_computed(self) -> None:
        # a=[2,4,6]: mean 4, var 4; b=[1,3,5]: mean 3, var 4.
        # pooled sd = sqrt((2*4 + 2*4)/4) = 2 -> d = (4-3)/2 = 0.5
        assert cohens_d([2.0, 4.0, 6.0], [1.0, 3.0, 5.0]) == pytest.approx(0.5)

    def test_cohens_d_antisymmetric(self) -> None:
        a, b = [2.0, 4.0, 6.0], [1.0, 3.0, 5.0]
        assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))

    def test_cohens_d_zero_variance_returns_zero(self) -> None:
        assert cohens_d([3.0, 3.0, 3.0], [3.0, 3.0, 3.0]) == 0.0

    @pytest.mark.parametrize(
        "values_a, values_b, expected_sign",
        [
            ([2.0, 2.0], [1.0, 1.0], 1.0),
            ([1.0, 1.0], [2.0, 2.0], -1.0),
            ([2.0], [1.0, 1.0], 1.0),
            ([1.0, 1.0], [2.0], -1.0),
        ],
    )
    def test_cohens_d_unequal_zero_variance_groups_returns_signed_infinity(
        self,
        values_a: list[float],
        values_b: list[float],
        expected_sign: float,
    ) -> None:
        effect = cohens_d(values_a, values_b)
        assert np.isinf(effect)
        assert np.sign(effect) == expected_sign

    def test_cohens_d_positive_means_a_greater(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(2.0, 1.0, size=50)
        b = rng.normal(0.0, 1.0, size=50)
        assert cohens_d(a, b) > 1.0

    def test_mann_whitney_rank_biserial_direction(self) -> None:
        """Rank-biserial sign must match the module convention (positive => a > b).

        With a completely dominating b, every one of the n_a*n_b pairs favors
        a, so the rank-biserial correlation is exactly +1; fully reversed
        groups give exactly -1 (Kerby 2014: r = 2*U1/(n_a*n_b) - 1).
        """
        a_dom = [10.0, 11.0, 12.0, 13.0]
        b_low = [1.0, 2.0, 3.0, 4.0]
        res_a_wins = mann_whitney_comparison(a_dom, b_low)
        assert res_a_wins.effect_size == pytest.approx(1.0)
        res_b_wins = mann_whitney_comparison(b_low, a_dom)
        assert res_b_wins.effect_size == pytest.approx(-1.0)

    def test_mann_whitney_rank_biserial_partial_overlap(self) -> None:
        # a=[3,5], b=[1,4]: favorable pairs (3>1, 5>1, 5>4) = 3 of 4
        # => U1 = 3, r = 2*3/4 - 1 = 0.5
        res = mann_whitney_comparison([3.0, 5.0], [1.0, 4.0])
        assert res.statistic == pytest.approx(3.0)
        assert res.effect_size == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. pairwise_comparisons end-to-end on synthetic AggregatedResults
# ---------------------------------------------------------------------------


def _make_aggregated(name: str, level: float, seed: int, n_seeds: int = 12) -> AggregatedResults:
    """AggregatedResults whose metric hovers at `level` with small seed noise."""
    rng = np.random.default_rng(seed)
    n_steps = 30
    per_seed_offset = rng.normal(0.0, 0.02 * max(level, 0.1), size=(n_seeds, 1))
    arr = level + per_seed_offset + rng.normal(0.0, 0.01, size=(n_seeds, n_steps))
    return AggregatedResults(
        config_name=name,
        seeds=list(range(n_seeds)),
        metric_arrays={"squared_error": arr},
        summary={},
    )


def _make_seeded_aggregated(
    name: str,
    seeds: list[int],
    values: list[float],
) -> AggregatedResults:
    """AggregatedResults with one metric value per explicitly identified seed."""
    return AggregatedResults(
        config_name=name,
        seeds=seeds,
        metric_arrays={"squared_error": np.asarray(values, dtype=np.float64)[:, None]},
        summary={},
    )


class TestPairwiseComparisons:
    def _results(self) -> dict[str, AggregatedResults]:
        return {
            "good": _make_aggregated("good", 0.1, seed=1),
            "mid": _make_aggregated("mid", 0.5, seed=2),
            "bad": _make_aggregated("bad", 2.0, seed=3),
        }

    def test_all_pairs_present_and_significant(self) -> None:
        comps = pairwise_comparisons(
            self._results(), metric="squared_error", test="ttest", correction="holm", window=10
        )
        assert set(comps) == {("good", "mid"), ("good", "bad"), ("mid", "bad")}
        for (name_a, name_b), res in comps.items():
            assert res.significant, f"{name_a} vs {name_b} should separate cleanly"
            assert res.method_a == name_a and res.method_b == name_b
            assert "(holm)" in res.test_name
            # lower squared error listed first in every pair => negative d
            assert res.effect_size < 0.0

    def test_correction_matches_manual_holm(self) -> None:
        comps = pairwise_comparisons(
            self._results(), test="ttest", correction="holm", window=10
        )
        p_values = [r.p_value for r in comps.values()]
        expected = holm_correction(p_values, alpha=0.05)
        assert [r.significant for r in comps.values()] == expected

    def test_bonferroni_correction_path(self) -> None:
        comps = pairwise_comparisons(
            self._results(), test="wilcoxon", correction="bonferroni", window=10
        )
        p_values = [r.p_value for r in comps.values()]
        expected, _ = bonferroni_correction(p_values, alpha=0.05)
        assert [r.significant for r in comps.values()] == expected
        assert all("(bonferroni)" in r.test_name for r in comps.values())

    def test_fewer_than_two_methods_returns_empty(self) -> None:
        assert pairwise_comparisons({"only": _make_aggregated("only", 0.1, seed=4)}) == {}

    def test_unknown_test_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown test"):
            pairwise_comparisons(self._results(), test="anova")

    def test_unknown_correction_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown correction"):
            pairwise_comparisons(self._results(), correction="fdr")

    @pytest.mark.parametrize("window", [0, -1])
    def test_nonpositive_window_rejected_without_warnings(self, window: int) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="window.*positive"):
                pairwise_comparisons(self._results(), window=window)

    @pytest.mark.parametrize("window", [True, False, 1.0, float("nan"), float("inf")])
    def test_bool_and_noninteger_window_rejected_without_warnings(
        self,
        window: object,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="window.*positive integer"):
                pairwise_comparisons(self._results(), window=window)  # type: ignore[arg-type]

    def test_zero_step_metric_rejected_without_warnings(self) -> None:
        empty_steps = AggregatedResults(
            config_name="empty_steps",
            seeds=[0, 1],
            metric_arrays={"squared_error": np.empty((2, 0), dtype=np.float64)},
            summary={},
        )
        valid = _make_seeded_aggregated("valid", [0, 1], [1.0, 2.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="at least one metric step"):
                pairwise_comparisons({"empty_steps": empty_steps, "valid": valid})

    def test_non_aggregated_results_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            pairwise_comparisons({"a": object(), "b": object()})  # type: ignore[dict-item]

    @pytest.mark.parametrize("test", ["ttest", "wilcoxon"])
    @pytest.mark.parametrize("correction", ["bonferroni", "holm"])
    def test_paired_tests_align_reordered_equal_seed_sets(
        self,
        test: str,
        correction: str,
    ) -> None:
        a = _make_seeded_aggregated("a", [0, 1, 2, 3], [0.0, 10.0, 20.0, 30.0])
        aligned_b = _make_seeded_aggregated("b", [0, 1, 2, 3], [2.0, 11.0, 24.0, 28.0])
        aligned_c = _make_seeded_aggregated("c", [0, 1, 2, 3], [1.0, 14.0, 16.0, 35.0])
        reordered_b = _make_seeded_aggregated("b", [2, 0, 3, 1], [24.0, 2.0, 28.0, 11.0])
        reordered_c = _make_seeded_aggregated("c", [1, 3, 0, 2], [14.0, 35.0, 1.0, 16.0])

        aligned = pairwise_comparisons(
            {"a": a, "b": aligned_b, "c": aligned_c},
            test=test,
            correction=correction,
            window=1,
        )
        reordered = pairwise_comparisons(
            {"a": a, "b": reordered_b, "c": reordered_c},
            test=test,
            correction=correction,
            window=1,
        )

        assert reordered == aligned

    def test_paired_ttest_aligns_adversarial_seed_order(self) -> None:
        a = _make_seeded_aggregated("a", [0, 1, 2], [0.0, 10.0, 20.0])
        reversed_b = _make_seeded_aggregated("b", [2, 1, 0], [21.0, 11.0, 1.0])

        result = pairwise_comparisons({"a": a, "b": reversed_b}, window=1)[("a", "b")]

        assert np.isneginf(result.statistic)
        assert result.p_value == 0.0

    @pytest.mark.parametrize("test", ["ttest", "wilcoxon"])
    def test_paired_tests_reject_different_seed_sets(self, test: str) -> None:
        a = _make_seeded_aggregated("a", [0, 1, 2], [0.0, 1.0, 2.0])
        b = _make_seeded_aggregated("b", [1, 2, 3], [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="equal seed sets"):
            pairwise_comparisons({"a": a, "b": b}, test=test, window=1)

    def test_reduction_pin_pair_rejected_for_paired_ttest(self) -> None:
        rows = [0.10, 0.11, 0.09]
        results = {
            "base_arm": _make_seeded_aggregated("base_arm", [0, 1, 2], rows),
            "pinned_inert": _make_seeded_aggregated("pinned_inert", [0, 1, 2], list(rows)),
        }
        with pytest.raises(
            ValueError,
            match=r"^Paired comparison 'base_arm' vs 'pinned_inert' has identical "
            r"samples; the paired t statistic is undefined$",
        ):
            pairwise_comparisons(results, metric="squared_error", test="ttest")

    def test_reduction_pin_pair_rejected_for_wilcoxon(self) -> None:
        rows = [0.10, 0.11, 0.09]
        results = {
            "base_arm": _make_seeded_aggregated("base_arm", [0, 1, 2], rows),
            "pinned_inert": _make_seeded_aggregated("pinned_inert", [0, 1, 2], list(rows)),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError,
                match=r"^Paired comparison 'base_arm' vs 'pinned_inert' has identical "
                r"samples; the Wilcoxon signed-rank statistic is undefined$",
            ):
                pairwise_comparisons(results, metric="squared_error", test="wilcoxon")

    def test_duplicate_seeds_rejected(self) -> None:
        malformed = _make_seeded_aggregated("malformed", [0, 0, 1], [0.0, 1.0, 2.0])
        valid = _make_seeded_aggregated("valid", [10, 11, 12], [3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="duplicate seeds"):
            pairwise_comparisons({"malformed": malformed, "valid": valid}, test="mann_whitney")

    def test_seed_count_must_match_metric_rows(self) -> None:
        malformed = _make_seeded_aggregated("malformed", [0, 1], [0.0, 1.0, 2.0])

        with pytest.raises(ValueError, match="seed count.*metric rows"):
            pairwise_comparisons({"malformed": malformed})

    def test_mann_whitney_accepts_distinct_seed_sets(self) -> None:
        a = _make_seeded_aggregated("a", [0, 1, 2], [0.0, 1.0, 2.0])
        b = _make_seeded_aggregated("b", [10, 11, 12], [3.0, 4.0, 5.0])

        result = pairwise_comparisons({"a": a, "b": b}, test="mann_whitney", window=1)

        assert result[("a", "b")].statistic == pytest.approx(0.0)

    @staticmethod
    def _settled_trace(name: str, n_steps: int, transient: float) -> AggregatedResults:
        """Three seeds that start at ``transient`` and settle at exactly 1.0 after step 9."""
        arr = np.ones((3, n_steps), dtype=np.float64)
        arr[:, : min(10, n_steps)] = transient
        return AggregatedResults(
            config_name=name,
            seeds=[0, 1, 2],
            metric_arrays={"squared_error": arr},
            summary={},
        )

    @pytest.mark.parametrize("test", ["ttest", "wilcoxon", "mann_whitney"])
    def test_unequal_final_windows_rejected(self, test: str) -> None:
        """A window longer than the shortest trace must not silently shrink per method."""
        short = self._settled_trace("short", n_steps=20, transient=5.0)
        long = self._settled_trace("long", n_steps=400, transient=5.0)

        with pytest.raises(
            ValueError,
            match=r"^window=100 exceeds the shortest 'squared_error' trace and the traces "
            r"differ in length \(long: 400 steps, short: 20 steps\); every method must "
            r"average the same number of final steps$",
        ):
            pairwise_comparisons({"short": short, "long": long}, test=test, window=100)

    def test_unequal_trace_lengths_accepted_when_window_fits_every_trace(self) -> None:
        short = self._settled_trace("short", n_steps=20, transient=5.0)
        long = self._settled_trace("long", n_steps=400, transient=7.0)

        result = pairwise_comparisons(
            {"short": short, "long": long}, test="mann_whitney", window=10
        )

        assert result[("short", "long")].effect_size == pytest.approx(0.0)
        assert not result[("short", "long")].significant

    @pytest.mark.parametrize("window", [0, -1, True, 1.5])
    def test_common_final_window_requires_positive_integer(self, window: object) -> None:
        with pytest.raises(ValueError, match="window must be a positive integer"):
            common_final_window({"learner": 10}, window, "squared_error")


@pytest.mark.parametrize("timeseries", [False, True])
def test_unrepresentable_maximum_spread_raises_without_runtime_warning(timeseries: bool) -> None:
    maximum = np.finfo(np.float64).max
    values = np.asarray([-maximum, maximum])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="cannot be represented as finite float64 statistics"):
            if timeseries:
                compute_timeseries_statistics(values[:, None])
            else:
                compute_statistics(values)
